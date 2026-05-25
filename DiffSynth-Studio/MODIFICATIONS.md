# Modifications to DiffSynth-Studio

This directory contains a modified version of [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) (v2.0.0), originally developed by the ModelScope Team and licensed under Apache 2.0.

## Changes Made

### Added: `examples/wanvideo/`

Four new scripts for computing Arrow-of-Time denoising loss on Wan-series video diffusion models:

- `compare_forward_backward_loss_v3_wan2.1_1.3b.py` — Wan2.1-T2V-1.3B evaluation
- `compare_forward_backward_loss_v3_wan2.1_14b.py` — Wan2.1-T2V-14B evaluation
- `compare_forward_backward_loss_v3_wan2.2_5b.py` — Wan2.2-TI2V-5B evaluation (T2V mode)
- `compare_forward_backward_loss_v3_wan2.2_a14b.py` — Wan2.2-T2V-A14B evaluation (dual DiT)

These scripts compare forward vs. backward video denoising loss using the DiffSynth pipeline, with support for:
- Bucket-based resolution selection
- Sliding window processing for long videos
- VRAM management (CPU/disk offloading)
- Multi-GPU parallel processing via index ranges
- Dual-DiT model switching (Wan2.2-A14B)

### Core Library

The `diffsynth/` core library is included as-is from the upstream release, with no modifications to the library source code. Only the `examples/wanvideo/` scripts are new additions.
