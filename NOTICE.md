# NOTICE

This repository is a **fork** of [ChenDarYen/ComfyUI-NAG](https://github.com/ChenDarYen/ComfyUI-NAG),
the upstream implementation of *Normalized Attention Guidance* (NAG) for ComfyUI
by Dar-Yen Chen. The original work, including all node implementations under
`chroma/`, `flux/`, `hidream/`, `hunyuan_video/`, `sd/`, `sd3/`, `wan/`,
`samplers.py`, `node.py`, `sample.py`, and `utils.py`, is the sole intellectual
property of the upstream author and is licensed under the terms in `LICENSE`
(MIT). The upstream `LICENSE` file is **unmodified** in this fork.

## Why this fork exists

Upstream had no commits between 2025-11-03 and the time this fork was made, while
ComfyUI itself continued to move forward (v0.21+). Several user-reported
incompatibility issues (#58, #62, #63, #68, #79) remained without maintainer
response. This fork carries small compatibility patches so the node keeps
working on current ComfyUI builds, while leaving the algorithmic implementation
and parameter semantics untouched.

## Modifications relative to upstream HEAD (`ef8a641`)

| Date (ISO) | Commit | Change | Files | Author |
|------------|--------|--------|-------|--------|
| 2026-05-20 | `0920a10` | Handle `None` stubs at `comfy.ldm.chroma.layers.DoubleStreamBlock` / `SingleStreamBlock` in ComfyUI v0.21+ via fallback to `comfy.ldm.flux.layers`. Remove dead import of the same symbols in `chroma/model.py`. | `chroma/layers.py`, `chroma/model.py` | Delcado19 |
| 2026-05-20 | `036e11d` | Fork-identity files (this `NOTICE.md`, README banner). | `NOTICE.md`, `README.md` | Delcado19 |
| 2026-05-20 | `8fe3449` | Add `pyproject.toml` for ComfyUI Registry publishing under fork-specific identifier. | `pyproject.toml` | Delcado19 |
| 2026-05-20 | `54e2c2b` | Add GitHub Actions: smoke import test and ComfyUI Registry publish workflow. | `.github/workflows/smoke-import.yml`, `.github/workflows/publish.yml` | Delcado19 |
| 2026-05-20 | `3618385` | Correct `PublisherId` to `delcado` and rename slug to `comfyui-nag-delcado` to match the registry profile handle `@delcado`. | `pyproject.toml` | Delcado19 |
| 2026-05-20 | `efcc966` | Fix README Usage-section replacement direction (closes upstream [#39](https://github.com/ChenDarYen/ComfyUI-NAG/issues/39)). | `README.md` | Delcado19 |
| 2026-05-20 | `f048600` | CI hotfix: install ComfyUI's own `requirements.txt` instead of a hand-curated pip list (was missing `scipy`). | `.github/workflows/smoke-import.yml` | Delcado19 |
| 2026-05-20 | `f0e5e69` → `2fd1269` | CI hotfix: force CPU mode in the smoke-import job so ComfyUI's `model_management.py` does not call `torch.cuda.current_device()` on a CPU-only runner. First via `sys.argv = ['comfyui', '--cpu']` (insufficient), then via direct `comfy.cli_args.args.cpu = True` patch. | `.github/workflows/smoke-import.yml` | Delcado19 |
| 2026-05-20 | — | First publish to Comfy Registry as [`comfyui-nag-delcado`](https://registry.comfy.org/nodes/comfyui-nag-delcado) version `1.0.1`. | — | Delcado19 |
| 2026-05-22 | `4e6b640` | Treat missing `flipped_img_txt` on current ComfyUI Flux blocks as the default `txt, img` attention ordering, avoiding Flux / Flux Kontext `AttributeError` crashes. | `flux/layers.py`, `README.md`, `NOTICE.md` | Delcado19 |
| 2026-05-22 | — | Harden the Chroma block import fallback so future ComfyUI builds that remove the deprecated `comfy.ldm.chroma.layers` path still fall back to `comfy.ldm.flux.layers`. | `chroma/layers.py`, `README.md`, `NOTICE.md` | Delcado19 |
| 2026-05-22 | — | Make the Flux NAG wrapper derive text positional-ID dimensions from the loaded model, fixing Flux2 / Flux.2 klein models that use four-axis IDs instead of the older three-axis Flux assumption. | `flux/model.py`, `README.md`, `NOTICE.md` | Delcado19 |
| 2026-05-22 | — | Keep Flux2 / Flux.2 klein gated-MLP and four-axis-ID models on the ComfyUI core single-stream path with combined RoPE data while NAG remains active in the double-stream path, avoiding the shape regression in the NAG single-stream wrapper. | `flux/model.py`, `README.md`, `NOTICE.md` | Delcado19 |

## Upstream tracking

- Upstream remote is preserved as `upstream` in this repo's working copy:
  `git remote get-url upstream` -> `https://github.com/ChenDarYen/ComfyUI-NAG.git`
- To incorporate future upstream changes:
  `git fetch upstream && git merge upstream/main`

## License

The MIT license in `LICENSE` covers both the upstream code and the
modifications listed above. Modifications are contributed under the same MIT
terms.
