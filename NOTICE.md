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

| Date (ISO) | Change | Files | Author |
|------------|--------|-------|--------|
| 2026-05-20 | Handle `None` stubs at `comfy.ldm.chroma.layers.DoubleStreamBlock` / `SingleStreamBlock` in ComfyUI v0.21+ via fallback to `comfy.ldm.flux.layers`. Remove dead import of the same symbols in `chroma/model.py`. | `chroma/layers.py`, `chroma/model.py` | Delcado19 |
| 2026-05-20 | Add `pyproject.toml` for ComfyUI Registry publishing under fork-specific identifier. | `pyproject.toml` | Delcado19 |
| 2026-05-20 | Add GitHub Actions: smoke import test and ComfyUI Registry publish workflow. | `.github/workflows/*` | Delcado19 |
| 2026-05-20 | Fork-identity files (this `NOTICE.md`, README banner). | `NOTICE.md`, `README.md` | Delcado19 |

## Upstream tracking

- Upstream remote is preserved as `upstream` in this repo's working copy:
  `git remote get-url upstream` -> `https://github.com/ChenDarYen/ComfyUI-NAG.git`
- To incorporate future upstream changes:
  `git fetch upstream && git merge upstream/main`

## License

The MIT license in `LICENSE` covers both the upstream code and the
modifications listed above. Modifications are contributed under the same MIT
terms.
