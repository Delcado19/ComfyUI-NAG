# ComfyUI-NAG (Delcado fork)

> **Maintenance fork** of [ChenDarYen/ComfyUI-NAG](https://github.com/ChenDarYen/ComfyUI-NAG)
> by Dar-Yen Chen. All credit for the original implementation goes to the upstream
> author; this fork only adds compatibility patches and packaging.
>
> See [`NOTICE.md`](NOTICE.md) for the change list and [`LICENSE`](LICENSE) for
> the unmodified MIT license terms.

## Install

- **Comfy Registry:** [`comfyui-nag-delcado`](https://registry.comfy.org/nodes/comfyui-nag-delcado)
  (publisher [`@delcado`](https://registry.comfy.org/publishers/delcado))
- **From source:** `git clone https://github.com/Delcado19/ComfyUI-NAG.git` into
  your `ComfyUI/custom_nodes/` directory.

## What this fork fixes vs. upstream

- **`chroma/layers.py` None-stub crash on ComfyUI v0.21+**
  Upstream `chroma/layers.py:10` does
  `class NAGDoubleStreamBlock(DoubleStreamBlock):` after importing
  `DoubleStreamBlock` / `SingleStreamBlock` from `comfy.ldm.chroma.layers`.
  Recent ComfyUI versions deprecate those symbols and set them to `None`
  (the real classes moved to `comfy.ldm.flux.layers`), so the import fails
  with `TypeError: NoneType takes no arguments`. This fork detects both the
  `None` stub and a missing legacy import path, then falls back to the new
  location.
  Closes the same root cause as upstream issues
  [#79](https://github.com/ChenDarYen/ComfyUI-NAG/issues/79),
  [#60](https://github.com/ChenDarYen/ComfyUI-NAG/issues/60),
  [#54](https://github.com/ChenDarYen/ComfyUI-NAG/issues/54),
  [#53](https://github.com/ChenDarYen/ComfyUI-NAG/issues/53),
  [#55](https://github.com/ChenDarYen/ComfyUI-NAG/issues/55).
- **README Usage section direction** — closes upstream
  [#39](https://github.com/ChenDarYen/ComfyUI-NAG/issues/39).
- **Packaging / Registry** — `pyproject.toml` plus GitHub Actions for
  smoke-import test and Comfy-Registry publish.

---

Implementation of [Normalized Attention Guidance: Universal Negative Guidance for Diffusion Models](https://chendaryen.github.io/NAG.github.io/) for [ComfyUI](https://github.com/comfyanonymous/ComfyUI).

NAG restores effective negative prompting in few-step diffusion models, and complements CFG in multi-step sampling for improved quality and control.

Paper: https://arxiv.org/abs/2505.21179

Code: https://github.com/ChenDarYen/Normalized-Attention-Guidance

Wan2.1 Demo: https://huggingface.co/spaces/ChenDY/NAG_wan2-1-fast

LTX Video Demo: https://huggingface.co/spaces/ChenDY/NAG_ltx-video-distilled

Flux-Dev Demo: https://huggingface.co/spaces/ChenDY/NAG_FLUX.1-dev

![comfyui-nag](workflow.png?cache=20250628)

## News

2025-07-06: Add three new nodes:
- `KSamplerWithNAG (Advanced)` as a drop-in replacement for `KSampler (Advanced)`.
- `SamplerCustomWithNAG` for `SamplerCustom`.
- `NAGGuider` for `BasicGuider`.

2025-07-02: `HiDream` is now supported!

2025-07-02: Add support for `TeaCache` and `WaveSpeed` to accelerate NAG sampling!

2025-06-30: Fix a major bug affecting `Flux`, `Flux Kontext` and `Chroma`, resulting in degraded guidance. Please update your NAG node!

2025-06-29: Add compile model support. You can now use compile model nodes like `TorchCompileModel` to speed up NAG sampling!

2025-06-28: `Flux Kontext` is now supported. Check out the [workflow](https://github.com/ChenDarYen/ComfyUI-NAG/blob/main/workflows/NAG-Flux-Kontext-Dev-ComfyUI-Workflow.json)!

2025-06-26: `Hunyuan video` is now supported!

2025-06-25: `Wan` video generation is now supported (GGUF compatible)! Try it out with the new [workflow](https://github.com/ChenDarYen/ComfyUI-NAG/blob/main/workflows/NAG-Wan-Fast-ComfyUI-Workflow.json)!

## Nodes

- `KSamplerWithNAG`, `KSamplerWithNAG (Advanced)`, `SamplerCustomWithNAG`
- `BasicGuider`, `NAGCFGGuider`

## Usage

To use NAG, simply replace
- `KSampler` with `KSamplerWithNAG`.
- `KSampler (Advanced)` with `KSamplerWithNAG (Advanced)`.
- `SamplerCustom` with `SamplerCustomWithNAG`.
- `BasicGuider` with `NAGGuider`.
- `CFGGuider` with `NAGCFGGuider`.

We currently support `Flux`, `Flux Kontext`, `Wan`, `Vace Wan`, `Hunyuan Video`, `Choroma`, `SD3.5`, `SDXL` and `SD`.

Example workflows are available in the `./workflows` directory!

## Key Inputs

When working with a new model, it's recommended to first find a good combination of `nag_tau` and `nag_alpha`, which ensures that the negative guidance is effective without introducing artifacts.

Once you're satisfied, keep `nag_tau` and `nag_alpha` fixed and tune only `nag_scale` in most cases to control the strength of guidance.

Using `nag_sigma_end` to reduce computation without much quality drop.

For flow-based models like `Flux`, `nag_sigma_end = 0.75` achieves near-identical results with significantly improved speed. For diffusion-based `SDXL`, a good default is `nag_sigma_end = 4`.

- `nag_scale`: The scale for attention feature extrapolation. Higher values result in stronger negative guidance.
- `nag_tau`: The normalisation threshold. Higher values result in stronger negative guidance.
- `nag_alpha`: Blending factor between original and extrapolated attention. Higher values result in stronger negative guidance.
- `nag_sigma_end`: NAG will be active only until `nag_sigma_end`.

### Rule of Thumb

- For image-reference tasks (e.g., Image2Video), use lower `nag_tau` and `nag_alpha` to preserve the reference content more faithfully.
- For models that require more sampling steps and higher CFG, also prefer lower `nag_tau` and `nag_alpha`.
- For few-step models, you can use higher `nag_tau` and `nag_alpha` to have stronger negative guidance.
