# Wan Model Forward/Backward Loss Comparison

Standalone scripts for computing Arrow-of-Time denoising loss on Wan-series video diffusion models. Each script loads a specific Wan model, iterates through a dataset of forward/backward video pairs, and computes MSE denoising loss at multiple timesteps to determine which direction the model "prefers."

These scripts use the DiffSynth-Studio pipeline directly rather than the generic `BaseVideoEvaluator` interface, allowing model-specific optimizations like bucket resolution selection, VRAM management with CPU/disk offloading, and dual-DiT model switching.

## Supported Models

| Script | Model | Parameters | Notes |
|--------|-------|------------|-------|
| `compare_forward_backward_loss_v3_wan2.1_1.3b.py` | Wan2.1-T2V-1.3B | 1.3B | Smallest, fastest |
| `compare_forward_backward_loss_v3_wan2.1_14b.py` | Wan2.1-T2V-14B | 14B | Large model |
| `compare_forward_backward_loss_v3_wan2.2_5b.py` | Wan2.2-TI2V-5B | 5B | Unified T2V/I2V, running in T2V mode |
| `compare_forward_backward_loss_v3_wan2.2_a14b.py` | Wan2.2-T2V-A14B | 14B | Dual DiT architecture (high/low noise models) |

## Usage

### Basic

```bash
python compare_forward_backward_loss_v3_wan2.1_14b.py \
    --dataset_path /path/to/YoCausal-dataset/subset/physics/dataset_metadata.json \
    --dataset_base_path /path/to/YoCausal-dataset/subset \
    --output_path results/wan2.1_14b_result.json
```

### With VRAM Management

For GPUs with limited memory:

```bash
# Standard CPU offloading (specify VRAM budget)
python compare_forward_backward_loss_v3_wan2.1_14b.py \
    --dataset_path /path/to/dataset_metadata.json \
    --vram_limit 40 \
    --output_path results/wan2.1_14b_result.json

# Aggressive low-VRAM mode (disk offloading, auto window size reduction)
python compare_forward_backward_loss_v3_wan2.1_14b.py \
    --dataset_path /path/to/dataset_metadata.json \
    --low_vram \
    --output_path results/wan2.1_14b_result.json
```

### Multi-GPU Parallel Processing

Split the dataset across GPUs using index ranges:

```bash
# GPU 0: process videos 0-99
CUDA_VISIBLE_DEVICES=0 python compare_forward_backward_loss_v3_wan2.1_14b.py \
    --dataset_path /path/to/dataset_metadata.json \
    --start_idx 0 --end_idx 100 \
    --output_path results/wan2.1_14b_gpu0.json

# GPU 1: process videos 100-199
CUDA_VISIBLE_DEVICES=1 python compare_forward_backward_loss_v3_wan2.1_14b.py \
    --dataset_path /path/to/dataset_metadata.json \
    --start_idx 100 --end_idx 200 \
    --output_path results/wan2.1_14b_gpu1.json
```

### Wan2.2-A14B (Dual DiT)

The A14B model uses two DiT models that switch at a configurable boundary:

```bash
python compare_forward_backward_loss_v3_wan2.2_a14b.py \
    --dataset_path /path/to/dataset_metadata.json \
    --switch_dit_boundary 0.5 \
    --output_path results/wan2.2_a14b_result.json
```

- Timestep >= boundary * 1000: uses the high-noise DiT (`dit`)
- Timestep < boundary * 1000: uses the low-noise DiT (`dit2`)

## Common Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset_path` | (required) | Path to `dataset_metadata.json` |
| `--dataset_base_path` | varies | Root containing source dataset folders or released `subset/` folders |
| `--output_path` | model-specific | Output JSON file path |
| `--target_fps` | 16 | Target FPS for video preprocessing |
| `--window_size` | 33-81 | Max frames per window (must satisfy 4n+1) |
| `--num_timesteps` | 10 | Number of uniformly sampled timesteps |
| `--num_noise_samples` | 10 | Number of noise samples per timestep |
| `--seed` | 42 | Base seed; pair at global index `i` uses seed `seed + i` |
| `--device` | cuda | Compute device |
| `--vram_limit` | None | VRAM budget in GB (enables CPU offloading) |
| `--low_vram` | False | Aggressive low-VRAM mode (disk offload + smaller windows) |
| `--use_gradient_checkpointing` | False | Enable gradient checkpointing to reduce memory |
| `--start_idx` | None | Start index for parallel processing (inclusive) |
| `--end_idx` | None | End index for parallel processing (exclusive) |
| `--max_samples` | None | Limit number of videos to process |

## Pipeline Details

### Video Preprocessing

1. **FPS resampling**: Videos are resampled to the model's target FPS (16 fps) using ffmpeg
2. **Bucket resolution**: The closest supported resolution bucket is selected based on the original aspect ratio, minimizing upscaling
3. **Center crop**: After scaling, frames are center-cropped to the exact bucket resolution
4. **Frame count adjustment**: Total frames are adjusted to satisfy the model's temporal requirement (4n+1)

### Sliding Window

For videos longer than the window size:
- The video is split into non-overlapping windows
- If the last window is shorter than the window size, frames from the previous segment are prepended as context
- Borrowed context frames are excluded from loss computation

### Loss Computation

At each timestep:
1. Encode video frames to latent space via VAE (using distribution mean for reproducibility)
2. For each pair at global index `i`, initialize separate forward/backward generators with the same seed (`seed + i`)
3. Add noise to latents at the given timestep (flow-matching schedule)
4. Run DiT forward pass to predict the denoised output
5. Compute MSE loss between prediction and training target
6. Average over all noise samples and timesteps

### Incremental Saving

Results are saved to disk after each video is processed, enabling safe interruption and resumption by checking the output file.

## Output Format

```json
{
  "model_info": {
    "model_name": "Wan2.1-T2V-14B",
    "parameters": 14000000000,
    "eval_settings": {
      "fps": 16,
      "timesteps_sampled": [90, 181, 272, ...],
      "num_noise_samples": 10,
      "window_size": 33
    }
  },
  "dataset_info": {
    "dataset_name": "animal-kingdom",
    "total_samples": 200,
    "path": ["data/animal-kingdom/dataset_metadata.json"]
  },
  "summary_metrics": {
    "forward_win_counts": 120,
    "backward_win_counts": 80,
    "ties": 0,
    "total_valid_comparisons": 200,
    "PPE": 0.40,
    "step_loss_differences": {"90": 0.002, "181": 0.001, ...}
  },
  "results": [
    {
      "id": 1,
      "prompt": "A bird is flying...",
      "forward_metrics": {"total_accumulated_loss": 0.45, "step_losses": {...}},
      "backward_metrics": {"total_accumulated_loss": 0.48, "step_losses": {...}},
      "comparison": {"loss_diff": 0.03, "winner": "forward"}
    }
  ]
}
```
