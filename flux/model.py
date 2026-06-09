from functools import partial
from types import MethodType
from typing import Callable

import torch
from torch import Tensor
from einops import rearrange, repeat
import comfy.ldm.common_dit

from comfy.ldm.flux.layers import (
    DoubleStreamBlock,
    SingleStreamBlock,
    timestep_embedding,
    apply_mod,
)
from comfy.ldm.flux.model import Flux

from .layers import NAGDoubleStreamBlock, NAGSingleStreamBlock
from ..utils import cat_context, check_nag_activation, poly1d, get_closure_vars, is_from_wavespeed, NAGSwitch


def make_txt_ids(model, batch_size, context_len, device):
    # NAG forward is bound onto ComfyUI's original Flux instance, so this helper
    # must not depend on methods defined only on NAGFlux.
    txt_ids = torch.zeros(
        (batch_size, context_len, len(model.params.axes_dim)),
        device=device,
        dtype=torch.float32,
    )
    for i in getattr(model.params, "txt_ids_dims", []):
        txt_ids[:, :, i] = torch.linspace(
            0,
            context_len - 1,
            steps=context_len,
            device=device,
            dtype=torch.float32,
        )
    return txt_ids


def uses_core_single_stream(model):
    axes_dim = getattr(model.params, "axes_dim", [])
    if len(axes_dim) > 3:
        return True

    for block in getattr(model, "single_blocks", []):
        if getattr(block, "yak_mlp", False):
            return True
        if getattr(block, "mlp_hidden_dim_first", None) != getattr(block, "mlp_hidden_dim", None):
            return True

    return False


def _pad_pe_txt_front(pe, txt_len, pad_len):
    # pe layout from EmbedND: (batch, 1, seq, head_dim/2, 2, 2).
    # The first `txt_len` entries along seq are txt positions, the rest are
    # img positions. Left-pad the txt portion by repeating its first entry so
    # the seq dim matches the padded txt length used in NAG single-stream.
    if pad_len <= 0:
        return pe
    txt_pe = pe[:, :, :txt_len]
    img_pe = pe[:, :, txt_len:]
    front = txt_pe[:, :, :1].expand(*txt_pe.shape[:2], pad_len, *txt_pe.shape[3:])
    return torch.cat([front, txt_pe, img_pe], dim=2)


def core_single_stream_forward(
        self,
        x,
        vec,
        pe,
        pe_negative=None,
        attn_mask=None,
        modulation_dims=None,
        transformer_options=None,
        original_forward=None,
        txt_length=None,
        context_pad_len=0,
        nag_pad_len=0,
        **kwargs,
):
    if pe is not None and pe_negative is not None:
        # pe / pe_negative were built from the *unpadded* txt_ids of each
        # batch, so their seq lengths differ whenever the positive and
        # negative contexts have different token counts (e.g. Flux.2 with
        # short uncond / long cond). The core single-stream block runs over
        # the padded txt+img sequence, so left-pad each pe along the txt
        # axis to the shared padded length before stacking over batch.
        # Fixes the "Sizes of tensors must match" error reported for the
        # Flux.2 klein 9B NAG workflow.
        if txt_length is not None:
            pe = _pad_pe_txt_front(pe, txt_length - context_pad_len, context_pad_len)
            pe_negative = _pad_pe_txt_front(pe_negative, txt_length - nag_pad_len, nag_pad_len)
        pe = torch.cat((pe, pe_negative), dim=0)
    return original_forward(
        x,
        vec=vec,
        pe=pe,
        attn_mask=attn_mask,
        modulation_dims=modulation_dims,
        transformer_options={} if transformer_options is None else transformer_options,
        **kwargs,
    )


def flux_vector_fallback(model, batch_size, reference):
    if getattr(model, "vector_in", None) is None or model.params.vec_in_dim is None:
        return None
    return torch.zeros(
        (batch_size, model.params.vec_in_dim),
        device=reference.device,
        dtype=reference.dtype,
    )


def normalize_nag_negative_y(y, nag_negative_y, nag_bsz):
    if y is None:
        return None
    if nag_negative_y is None:
        # Flux2 GGUF text encoders can omit pooled_output for CLIPTextEncode.
        # NAG still needs a vector branch, so use a neutral vector with the
        # same shape and dtype as the positive conditioning.
        return y.new_zeros((nag_bsz, *y.shape[1:]))

    nag_negative_y = nag_negative_y.to(y)
    if nag_negative_y.shape[0] == 0:
        return y.new_zeros((nag_bsz, *y.shape[1:]))
    if nag_negative_y.shape[0] != nag_bsz:
        nag_negative_y = nag_negative_y[:1].expand(nag_bsz, *nag_negative_y.shape[1:])
    return nag_negative_y


class NAGFlux(Flux):
    def forward_orig(
        self,
        img: Tensor,
        img_ids: Tensor,
        txt: Tensor,
        txt_ids: Tensor,
        txt_ids_negative: Tensor,
        timesteps: Tensor,
        y: Tensor,
        guidance: Tensor = None,
        control = None,
        transformer_options={},
        attn_mask: Tensor = None,
    ) -> Tensor:
        if self.vector_in is not None:
            if y is None:
                y = torch.zeros((img.shape[0], self.params.vec_in_dim), device=img.device, dtype=img.dtype)

        patches_replace = transformer_options.get("patches_replace", {})
        if img.ndim != 3 or txt.ndim != 3:
            raise ValueError("Input img and txt tensors must have 3 dimensions.")

        # running on sequences img
        img = self.img_in(img)
        vec = self.time_in(timestep_embedding(timesteps, 256).to(img.dtype))
        if self.params.guidance_embed:
            if guidance is not None:
                vec = vec + self.guidance_in(timestep_embedding(guidance, 256).to(img.dtype))

        origin_bsz = len(txt) - len(img)
        vec = torch.cat((vec, vec[-origin_bsz:]), dim=0)

        if self.vector_in is not None:
            vec = vec + self.vector_in(y[:,:self.params.vec_in_dim])
        vec_orig = vec
        if self.params.global_modulation:
            double_vec = (
                self.double_stream_modulation_img(vec_orig),
                self.double_stream_modulation_txt(vec_orig),
            )
        else:
            double_vec = vec
        txt = self.txt_in(txt)

        if img_ids is not None:
            ids = torch.cat((txt_ids, img_ids), dim=1)
            ids_negative = torch.cat((txt_ids_negative, img_ids[-origin_bsz:]), dim=1)
            pe = self.pe_embedder(ids)
            pe_negative = self.pe_embedder(ids_negative)
        else:
            pe = None
            pe_negative = None

        blocks_replace = patches_replace.get("dit", {})
        for i, block in enumerate(self.double_blocks):
            if ("double_block", i) in blocks_replace:
                def block_wrap(args):
                    out = {}
                    out["img"], out["txt"] = block(img=args["img"],
                                                   txt=args["txt"],
                                                   vec=args["vec"],
                                                   pe=args["pe"],
                                                   pe_negative=args["pe_negative"],
                                                   attn_mask=args.get("attn_mask"))
                    return out

                out = blocks_replace[("double_block", i)]({"img": img,
                                                           "txt": txt,
                                                           "vec": double_vec,
                                                           "pe": pe,
                                                           "pe_negative": pe_negative,
                                                           "attn_mask": attn_mask,
                                                           "transformer_options": transformer_options},
                                                          {"original_block": block_wrap})
                txt = out["txt"]
                img = out["img"]
            else:
                img, txt = block(img=img,
                                 txt=txt,
                                 vec=double_vec,
                                 pe=pe,
                                 pe_negative=pe_negative,
                                 attn_mask=attn_mask)

            if control is not None: # Controlnet
                control_i = control.get("input")
                if i < len(control_i):
                    add = control_i[i]
                    if add is not None:
                        img += add

        if img.dtype == torch.float16:
            img = torch.nan_to_num(img, nan=0.0, posinf=65504, neginf=-65504)

        img = torch.cat((img, img[-origin_bsz:]), dim=0)
        img = torch.cat((txt, img), 1)
        if self.params.global_modulation:
            single_vec, _ = self.single_stream_modulation(vec_orig)
        else:
            single_vec = vec

        for i, block in enumerate(self.single_blocks):
            if ("single_block", i) in blocks_replace:
                def block_wrap(args):
                    out = {}
                    out["img"] = block(args["img"],
                                       vec=args["vec"],
                                       pe=args["pe"],
                                       pe_negative=args["pe_negative"],
                                       attn_mask=args.get("attn_mask"))
                    return out

                out = blocks_replace[("single_block", i)]({"img": img,
                                                           "vec": single_vec,
                                                           "pe": pe,
                                                           "pe_negative": pe_negative,
                                                           "attn_mask": attn_mask,
                                                           "transformer_options": transformer_options},
                                                          {"original_block": block_wrap})
                img = out["img"]
            else:
                img = block(img, vec=single_vec, pe=pe, pe_negative=pe_negative, attn_mask=attn_mask)

            if control is not None: # Controlnet
                control_o = control.get("output")
                if i < len(control_o):
                    add = control_o[i]
                    if add is not None:
                        img[:, txt.shape[1] :, ...] += add

        img = img[:-origin_bsz]
        img = img[:, txt.shape[1] :, ...]

        final_vec = vec_orig if self.params.global_modulation else vec
        img = self.final_layer(img, final_vec[:-origin_bsz])  # (N, T, patch_size ** 2 * out_channels)
        return img

    def forward_orig_with_teacache(
        self,
        img: Tensor,
        img_ids: Tensor,
        txt: Tensor,
        txt_ids: Tensor,
        txt_ids_negative: Tensor,
        timesteps: Tensor,
        y: Tensor,
        guidance: Tensor = None,
        control = None,
        transformer_options={},
        attn_mask: Tensor = None,
    ) -> Tensor:
        enable_teacache = transformer_options.get("enable_teacache", True)
        rel_l1_thresh = transformer_options.get("rel_l1_thresh")
        coefficients = transformer_options.get("coefficients")
        cache_device = transformer_options.get("cache_device")

        if self.vector_in is not None:
            if y is None:
                y = torch.zeros((img.shape[0], self.params.vec_in_dim), device=img.device, dtype=img.dtype)

        patches_replace = transformer_options.get("patches_replace", {})
        if img.ndim != 3 or txt.ndim != 3:
            raise ValueError("Input img and txt tensors must have 3 dimensions.")

        # running on sequences img
        img = self.img_in(img)
        vec = self.time_in(timestep_embedding(timesteps, 256).to(img.dtype))
        if self.params.guidance_embed:
            if guidance is not None:
                vec = vec + self.guidance_in(timestep_embedding(guidance, 256).to(img.dtype))

        origin_bsz = len(txt) - len(img)
        vec = torch.cat((vec, vec[-origin_bsz:]), dim=0)

        if self.vector_in is not None:
            vec = vec + self.vector_in(y[:,:self.params.vec_in_dim])
        txt = self.txt_in(txt)

        if img_ids is not None:
            ids = torch.cat((txt_ids, img_ids), dim=1)
            ids_negative = torch.cat((txt_ids_negative, img_ids[-origin_bsz:]), dim=1)
            pe = self.pe_embedder(ids)
            pe_negative = self.pe_embedder(ids_negative)
        else:
            pe = None
            pe_negative = None

        blocks_replace = patches_replace.get("dit", {})

        if enable_teacache:
            img_mod1, _ = self.double_blocks[0].img_mod(vec)
            modulated_inp = self.double_blocks[0].img_norm1(img)
            modulated_inp = apply_mod(modulated_inp, (1 + img_mod1.scale), img_mod1.shift).to(cache_device)

            if not hasattr(self, 'accumulated_rel_l1_distance'):
                should_calc = True
                self.accumulated_rel_l1_distance = 0
            else:
                try:
                    self.accumulated_rel_l1_distance += \
                        poly1d(coefficients, ((modulated_inp - self.previous_modulated_input).abs().mean() / self.previous_modulated_input.abs().mean()))

                    if self.accumulated_rel_l1_distance < rel_l1_thresh:
                        should_calc = False
                    else:
                        should_calc = True
                        self.accumulated_rel_l1_distance = 0
                except:
                    should_calc = True
                    self.accumulated_rel_l1_distance = 0

            self.previous_modulated_input = modulated_inp
            if should_calc:
                ori_img = img.to(cache_device)

        else:
            should_calc = False

        if should_calc:
            for i, block in enumerate(self.double_blocks):
                if ("double_block", i) in blocks_replace:
                    def block_wrap(args):
                        out = {}
                        out["img"], out["txt"] = block(img=args["img"],
                                                       txt=args["txt"],
                                                       vec=args["vec"],
                                                       pe=args["pe"],
                                                       pe_negative=args["pe_negative"],
                                                       attn_mask=args.get("attn_mask"))
                        return out

                    out = blocks_replace[("double_block", i)]({"img": img,
                                                               "txt": txt,
                                                               "vec": vec,
                                                               "pe": pe,
                                                               "pe_negative": pe_negative,
                                                               "attn_mask": attn_mask,
                                                               "transformer_options": transformer_options},
                                                              {"original_block": block_wrap})
                    txt = out["txt"]
                    img = out["img"]
                else:
                    img, txt = block(img=img,
                                     txt=txt,
                                     vec=vec,
                                     pe=pe,
                                     pe_negative=pe_negative,
                                     attn_mask=attn_mask)

                if control is not None: # Controlnet
                    control_i = control.get("input")
                    if i < len(control_i):
                        add = control_i[i]
                        if add is not None:
                            img += add

            if img.dtype == torch.float16:
                img = torch.nan_to_num(img, nan=0.0, posinf=65504, neginf=-65504)

            img = torch.cat((img, img[-origin_bsz:]), dim=0)
            img = torch.cat((txt, img), 1)

            for i, block in enumerate(self.single_blocks):
                if ("single_block", i) in blocks_replace:
                    def block_wrap(args):
                        out = {}
                        out["img"] = block(args["img"],
                                           vec=args["vec"],
                                           pe=args["pe"],
                                           pe_negative=args["pe_negative"],
                                           attn_mask=args.get("attn_mask"))
                        return out

                    out = blocks_replace[("single_block", i)]({"img": img,
                                                               "vec": vec,
                                                               "pe": pe,
                                                               "pe_negative": pe_negative,
                                                               "attn_mask": attn_mask,
                                                               "transformer_options": transformer_options},
                                                              {"original_block": block_wrap})
                    img = out["img"]
                else:
                    img = block(img, vec=vec, pe=pe, pe_negative=pe_negative, attn_mask=attn_mask)

                if control is not None: # Controlnet
                    control_o = control.get("output")
                    if i < len(control_o):
                        add = control_o[i]
                        if add is not None:
                            img[:, txt.shape[1] :, ...] += add

            img = img[:-origin_bsz]
            img = img[:, txt.shape[1] :, ...]

        else:
            img += self.previous_residual.to(img.device)

        if enable_teacache and should_calc:
            self.previous_residual = img.to(cache_device) - ori_img

        img = self.final_layer(img, vec[:-origin_bsz])  # (N, T, patch_size ** 2 * out_channels)
        return img

    def forward_orig_with_wavespeed(
        self,
        img: Tensor,
        img_ids: Tensor,
        txt: Tensor,
        txt_ids: Tensor,
        txt_ids_negative: Tensor,
        timesteps: Tensor,
        y: Tensor,
        guidance: Tensor = None,
        control = None,
        transformer_options={},
        attn_mask: Tensor = None,
        use_cache: Callable = None,
        apply_prev_hidden_states_residual: Callable = None,
        set_buffer: Callable = None,
    ) -> Tensor:
        if self.vector_in is not None:
            if y is None:
                y = torch.zeros((img.shape[0], self.params.vec_in_dim), device=img.device, dtype=img.dtype)

        patches_replace = transformer_options.get("patches_replace", {})
        if img.ndim != 3 or txt.ndim != 3:
            raise ValueError("Input img and txt tensors must have 3 dimensions.")

        # running on sequences img
        img = self.img_in(img)
        vec = self.time_in(timestep_embedding(timesteps, 256).to(img.dtype))
        if self.params.guidance_embed:
            if guidance is not None:
                vec = vec + self.guidance_in(timestep_embedding(guidance, 256).to(img.dtype))

        origin_bsz = len(txt) - len(img)
        vec = torch.cat((vec, vec[-origin_bsz:]), dim=0)

        if self.vector_in is not None:
            vec = vec + self.vector_in(y[:,:self.params.vec_in_dim])
        txt = self.txt_in(txt)

        if img_ids is not None:
            ids = torch.cat((txt_ids, img_ids), dim=1)
            ids_negative = torch.cat((txt_ids_negative, img_ids[-origin_bsz:]), dim=1)
            pe = self.pe_embedder(ids)
            pe_negative = self.pe_embedder(ids_negative)
        else:
            pe = None
            pe_negative = None

        can_use_cache = False

        blocks_replace = patches_replace.get("dit", {})
        for i, block in enumerate(self.double_blocks):
            if i == 1:
                torch._dynamo.graph_break()
                if can_use_cache:
                    del first_img_residual
                    img = apply_prev_hidden_states_residual(img)
                    break
                else:
                    set_buffer("first_hidden_states_residual", first_img_residual)
                    del first_img_residual

                    original_img = img

            if ("double_block", i) in blocks_replace:
                def block_wrap(args):
                    out = {}
                    out["img"], out["txt"] = block(img=args["img"],
                                                   txt=args["txt"],
                                                   vec=args["vec"],
                                                   pe=args["pe"],
                                                   pe_negative=args["pe_negative"],
                                                   attn_mask=args.get("attn_mask"))
                    return out

                out = blocks_replace[("double_block", i)]({"img": img,
                                                           "txt": txt,
                                                           "vec": vec,
                                                           "pe": pe,
                                                           "pe_negative": pe_negative,
                                                           "attn_mask": attn_mask,
                                                           "transformer_options": transformer_options},
                                                          {"original_block": block_wrap})
                txt = out["txt"]
                img = out["img"]
            else:
                img, txt = block(img=img,
                                 txt=txt,
                                 vec=vec,
                                 pe=pe,
                                 pe_negative=pe_negative,
                                 attn_mask=attn_mask)

            if control is not None: # Controlnet
                control_i = control.get("input")
                if i < len(control_i):
                    add = control_i[i]
                    if add is not None:
                        img += add

            if i == 0:
                first_img_residual = img
                can_use_cache = use_cache(first_img_residual)

        if not can_use_cache:
            if img.dtype == torch.float16:
                img = torch.nan_to_num(img, nan=0.0, posinf=65504, neginf=-65504)

            img = torch.cat((img, img[-origin_bsz:]), dim=0)
            img = torch.cat((txt, img), 1)

            for i, block in enumerate(self.single_blocks):
                if ("single_block", i) in blocks_replace:
                    def block_wrap(args):
                        out = {}
                        out["img"] = block(args["img"],
                                           vec=args["vec"],
                                           pe=args["pe"],
                                           pe_negative=args["pe_negative"],
                                           attn_mask=args.get("attn_mask"))
                        return out

                    out = blocks_replace[("single_block", i)]({
                        "img": img,
                        "vec": vec,
                        "pe": pe,
                        "pe_negative": pe_negative,
                        "attn_mask": attn_mask,
                        "transformer_options": transformer_options,
                    }, {"original_block": block_wrap})
                    img = out["img"]
                else:
                    img = block(img, vec=vec, pe=pe, pe_negative=pe_negative, attn_mask=attn_mask)

                if control is not None: # Controlnet
                    control_o = control.get("output")
                    if i < len(control_o):
                        add = control_o[i]
                        if add is not None:
                            img[:, txt.shape[1] :, ...] += add

            img = img[:-origin_bsz]
            img = img[:, txt.shape[1] :, ...]

            img = img.contiguous()
            img_residual = img - original_img
            set_buffer("hidden_states_residual", img_residual)
            del original_img

        torch._dynamo.graph_break()

        img = self.final_layer(img, vec[:-origin_bsz])  # (N, T, patch_size ** 2 * out_channels)
        return img

    def forward(
            self,
            x,
            timestep,
            context,
            y=None,
            guidance=None,
            ref_latents=None,
            control=None,
            transformer_options={},

            nag_negative_context=None,
            nag_negative_y=None,
            nag_sigma_end=0.,

            **kwargs,
    ):
        bs, c, h_orig, w_orig = x.shape
        patch_size = self.patch_size

        h_len = ((h_orig + (patch_size // 2)) // patch_size)
        w_len = ((w_orig + (patch_size // 2)) // patch_size)
        img, img_ids = self.process_img(x, transformer_options=transformer_options)
        img_tokens = img.shape[1]
        if ref_latents is not None:
            h = 0
            w = 0
            for ref in ref_latents:
                h_offset = 0
                w_offset = 0
                if ref.shape[-2] + h > ref.shape[-1] + w:
                    w_offset = w
                else:
                    h_offset = h

                kontext, kontext_ids = self.process_img(
                    ref,
                    index=1,
                    h_offset=h_offset,
                    w_offset=w_offset,
                    transformer_options=transformer_options,
                )
                img = torch.cat([img, kontext], dim=1)
                img_ids = torch.cat([img_ids, kontext_ids], dim=1)
                h = max(h, ref.shape[-2] + h_offset)
                w = max(w, ref.shape[-1] + w_offset)

        apply_nag = check_nag_activation(transformer_options, nag_sigma_end)
        if apply_nag:
            if y is None and self.vector_in is not None:
                y = flux_vector_fallback(self, bs, img)
            origin_context_len = context.shape[1]
            nag_bsz, nag_negative_context_len = nag_negative_context.shape[:2]
            context = cat_context(context, nag_negative_context, trim_context=True)
            nag_negative_y = normalize_nag_negative_y(y, nag_negative_y, nag_bsz)
            if y is not None and nag_negative_y is not None:
                y = torch.cat((y, nag_negative_y), dim=0)
            context_pad_len = context.shape[1] - origin_context_len
            nag_pad_len = context.shape[1] - nag_negative_context_len
            # Flux2 / Flux.2 klein uses newer four-axis IDs and can expose a
            # single-stream layout the NAG wrapper does not reshape safely yet.
            # Keep those core blocks untouched and apply NAG in double-stream.
            use_nag_single_blocks = not uses_core_single_stream(self)

            forward_orig_ = self.forward_orig
            double_blocks_forward = list()
            single_blocks_forward = list()

            # The teacache / wavespeed forward_orig variants predate Flux2 and do
            # not implement global_modulation: they pass the raw `vec` to every
            # block and to final_layer. For a global-modulation (Flux2) model the
            # blocks expect precomputed modulation tuples instead, so routing
            # Flux2 through those variants crashes (e.g. double_blocks[0].img_mod
            # does not exist) or produces wrong modulation. Until the cache
            # variants learn global_modulation, fall back to the correct
            # (uncached) base forward_orig for Flux2 — NAG stays correct, only the
            # cache speed-up is skipped. Classic Flux keeps the cache paths.
            global_modulation = getattr(self.params, "global_modulation", False)

            if not global_modulation and transformer_options.get("enable_teacache", False):
                self.forward_orig = MethodType(NAGFlux.forward_orig_with_teacache, self)

            elif not global_modulation and is_from_wavespeed(forward_orig_):
                get_can_use_cache = forward_orig_.__globals__["get_can_use_cache"]
                set_buffer = forward_orig_.__globals__["set_buffer"]
                apply_prev_hidden_states_residual = forward_orig_.__globals__["apply_prev_hidden_states_residual"]
                closure_vars = get_closure_vars(forward_orig_)

                def use_cache(first_img_residual):
                    can_use_cache = get_can_use_cache(
                        first_img_residual,
                        threshold=closure_vars["residual_diff_threshold"],
                        validation_function=closure_vars["validate_can_use_cache_function"],
                    )
                    return can_use_cache

                self.forward_orig = MethodType(
                    partial(
                        NAGFlux.forward_orig_with_wavespeed,
                        use_cache=use_cache,
                        apply_prev_hidden_states_residual=apply_prev_hidden_states_residual,
                        set_buffer=set_buffer,
                    ),
                    self,
                )

            else:
                self.forward_orig = MethodType(NAGFlux.forward_orig, self)

            for block in self.double_blocks:
                double_blocks_forward.append(block.forward)
                block.forward = MethodType(
                    partial(
                        NAGDoubleStreamBlock.forward,
                        context_pad_len=context_pad_len,
                        nag_pad_len=nag_pad_len,
                    ),
                    block,
                )
            if use_nag_single_blocks:
                for block in self.single_blocks:
                    single_blocks_forward.append(block.forward)
                    block.forward = MethodType(
                        partial(
                            NAGSingleStreamBlock.forward,
                            txt_length=context.shape[1],
                            origin_bsz=nag_bsz,
                            context_pad_len=context_pad_len,
                            nag_pad_len=nag_pad_len,
                        ),
                        block,
                    )
            else:
                for block in self.single_blocks:
                    original_forward = block.forward
                    single_blocks_forward.append(original_forward)
                    block.forward = MethodType(
                        partial(
                            core_single_stream_forward,
                            original_forward=original_forward,
                            # Needed so pe / pe_negative can be padded to the
                            # shared txt length before cat over batch when the
                            # positive and negative contexts differ in length.
                            txt_length=context.shape[1],
                            context_pad_len=context_pad_len,
                            nag_pad_len=nag_pad_len,
                        ),
                        block,
                    )

            try:
                txt_ids = make_txt_ids(self, bs, origin_context_len, x.device)
                txt_ids_negative = make_txt_ids(self, nag_bsz, nag_negative_context_len, x.device)
                out = self.forward_orig(
                    img,
                    img_ids,
                    context,
                    txt_ids,
                    txt_ids_negative,
                    timestep,
                    y,
                    guidance=guidance,
                    control=control,
                    transformer_options=transformer_options,
                    attn_mask=kwargs.get("attention_mask", None),
                )
            finally:
                self.forward_orig = forward_orig_
                for block, block_forward in zip(self.double_blocks, double_blocks_forward):
                    block.forward = block_forward
                for block, block_forward in zip(self.single_blocks, single_blocks_forward):
                    block.forward = block_forward

        else:
            txt_ids = make_txt_ids(self, bs, context.shape[1], x.device)
            out = self.forward_orig(
                img,
                img_ids,
                context,
                txt_ids,
                timestep,
                y,
                guidance=guidance,
                control=control,
                transformer_options=transformer_options,
                attn_mask=kwargs.get("attention_mask", None),
            )

        out = out[:, :img_tokens]
        # Flux2 uses patch_size=1 and 128 latent channels; the original Flux
        # wrapper assumed patch_size=2, which reshaped Flux2 outputs to 32
        # channels and broke ComfyUI's denoised calculation.
        return rearrange(
            out,
            "b (h w) (c ph pw) -> b c (h ph) (w pw)",
            h=h_len,
            w=w_len,
            ph=self.patch_size,
            pw=self.patch_size,
        )[:, :, :h_orig, :w_orig]


class NAGFluxSwitch(NAGSwitch):
    def set_nag(self):
        self.model.forward = MethodType(
            partial(
                NAGFlux.forward,
                nag_negative_context=self.nag_negative_cond[0][0],
                # Some Flux2/GGUF CLIP encoders omit pooled_output; NAGFlux
                # handles None without disabling the attention-guidance patch.
                nag_negative_y=self.nag_negative_cond[0][1].get("pooled_output"),
                nag_sigma_end=self.nag_sigma_end,
            ),
            self.model,
        )
        for block in self.model.double_blocks:
            block.nag_scale = self.nag_scale
            block.nag_tau = self.nag_tau
            block.nag_alpha = self.nag_alpha
        for block in self.model.single_blocks:
            block.nag_scale = self.nag_scale
            block.nag_tau = self.nag_tau
            block.nag_alpha = self.nag_alpha
