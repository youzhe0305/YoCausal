# YoCausal Evaluation Protocol

A general-purpose Arrow-of-Time evaluation framework that allows any video diffusion model to compute denoising loss through a unified interface and automatically produce RSI and CCI metrics.

## File Structure

```
evaluation/
├── base_evaluator.py                 # Abstract base class (model implementors must inherit)
├── evaluate.py                       # Main evaluation script (compute loss → RSI/CCI → output JSON)
├── metrics.py                        # RSI / CCI computation utilities
├── example_animatediff_evaluator.py  # Example: AnimateDiff implementation template
├── evaluation_example.py             # Legacy simple example (for reference)
└── evaluated_model_example.py        # Legacy interface example (for reference)
```

## Quick Start

### 1. Implement Your Model Evaluator

Inherit from `BaseVideoEvaluator` and implement 4 methods:

```python
from evaluation.base_evaluator import BaseVideoEvaluator

class MyModelEvaluator(BaseVideoEvaluator):
    def __init__(self, model_dir, device="cuda"):
        # Load model weights (VAE, UNet/DiT, Text Encoder, Scheduler, etc.)
        ...

    def get_model_info(self) -> dict:
        return {"model_name": "MyModel", "parameters": 1_000_000_000}

    def get_eval_settings(self) -> dict:
        return {"fps": 16, "resolution": [480, 720], "window_size": 49}

    def get_timesteps(self) -> list:
        # K=10 uniformly sampled timesteps
        return [90, 181, 272, 363, 454, 544, 635, 726, 817, 908]

    def compute_video_loss(self, video_path, prompt, timestep, noise_seed):
        # 1. Load and preprocess video (FPS, resolution, cropping)
        # 2. Encode prompt
        # 3. VAE encode → latents
        # 4. At specified timestep: add noise → model prediction → MSE loss
        # 5. Return loss (float)
        loss = ...
        return loss
```

### 2. Run Evaluation

**Command-line:**

```bash
python -m evaluation.evaluate \
    --evaluator_module my_evaluator \
    --evaluator_class MyModelEvaluator \
    --evaluator_args '{"model_dir": "/path/to/weights", "device": "cuda"}' \
    --dataset_paths \
        /path/to/YoCausal-dataset/subset/animal/dataset_metadata.json \
        /path/to/YoCausal-dataset/subset/general/dataset_metadata.json \
        /path/to/YoCausal-dataset/subset/human/dataset_metadata.json \
        /path/to/YoCausal-dataset/subset/physics/dataset_metadata.json \
    --dataset_base_path /path/to/YoCausal-dataset/subset \
    --output my_model_result.json
```

**Programmatic:**

```python
from evaluation.evaluate import run_evaluation
from my_evaluator import MyModelEvaluator

evaluator = MyModelEvaluator(model_dir="weights/", device="cuda")
results = run_evaluation(
    evaluator=evaluator,
    dataset_paths=[
        "/path/to/YoCausal-dataset/subset/animal/dataset_metadata.json",
        "/path/to/YoCausal-dataset/subset/general/dataset_metadata.json",
        "/path/to/YoCausal-dataset/subset/human/dataset_metadata.json",
        "/path/to/YoCausal-dataset/subset/physics/dataset_metadata.json",
    ],
    dataset_base_path="/path/to/YoCausal-dataset/subset",
    output_path="my_model_result.json",
)
```

### 3. Resume from Checkpoint

If evaluation is interrupted, you can resume from a checkpoint:

```bash
python -m evaluation.evaluate \
    --evaluator_module my_evaluator \
    --evaluator_class MyModelEvaluator \
    --resume my_model_result.json.checkpoint \
    --output my_model_result.json \
    ...
```

## Output JSON Format

```json
{
  "model_info": {
    "model_name": "MyModel",
    "parameters": 1000000000
  },
  "eval_settings": {
    "fps": 16,
    "resolution": [480, 720],
    "window_size": 49,
    "timesteps_sampled": [90, 181, ...]
  },
  "per_dataset_metrics": {
    "animal": {
      "rsi": 0.555,
      "forward_wins": 111,
      "backward_wins": 89,
      "ties": 0,
      "total": 200,
      "rsi_dc": 0.60,
      "rsi_dnc": 0.50,
      "cci": 0.10,
      "num_causal": 80,
      "num_non_causal": 120,
      "rsi_hd": 0.58,
      "rsi_hnd": 0.52,
      "num_human_disc": 110,
      "num_human_non_disc": 90
    },
    "kinetics": { ... },
    "mit": { ... },
    "physics": { ... }
  },
  "overall_metrics": {
    "rsi": 0.52,
    "rsi_dc": 0.58,
    "rsi_dnc": 0.47,
    "cci": 0.11,
    "rsi_hd": 0.55,
    "rsi_hnd": 0.49,
    "forward_wins": 640,
    "backward_wins": 592,
    "ties": 0,
    "total": 1232
  },
  "per_video_results": [
    {
      "id": 1,
      "dataset": "animal",
      "dataset_source": "Animal Kingdom",
      "vlm_causality": false,
      "human_discriminable": true,
      "prompt": "The anole lizard is puffing its throat.",
      "forward_metrics": {
        "total_accumulated_loss": 1.324,
        "step_losses": {"90": 0.437, "181": 0.293, ...}
      },
      "backward_metrics": {
        "total_accumulated_loss": 1.207,
        "step_losses": {"90": 0.400, "181": 0.264, ...}
      },
      "comparison": {
        "loss_diff": -0.117,
        "winner": "backward"
      }
    },
    ...
  ]
}
```

## Metric Definitions

| Metric | Definition | Description |
|--------|-----------|-------------|
| **RSI** | `(1/\|D\|) Σ RSI(Di)` | Average forward win rate across datasets. RSI=50% is equivalent to random guessing. |
| **RSI(Dc)** | RSI computed on causal subset | Computed only on the video subset with causal relationships |
| **RSI(Dnc)** | RSI computed on non-causal subset | Computed only on the video subset without causal relationships |
| **CCI** | `RSI(Dc) - RSI(Dnc)` | Separates temporal direction awareness from causal understanding. Higher CCI indicates better causal understanding. |
| **RSI(Hd)** | RSI computed on human-discriminable subset | Computed only on videos where humans can distinguish forward from backward |
| **RSI(Hnd)** | RSI computed on human-non-discriminable subset | Computed only on videos where humans cannot distinguish forward from backward |

## Dataset Metadata Format

Each `dataset_metadata.json` entry contains:

| Field | Description |
|-------|-------------|
| `id` | Video ID |
| `video_path_forward` | Forward video path |
| `video_path_backward` | Backward video path |
| `prompt` | Video text description |
| `dataset_source` | Dataset source name |
| `category` | Video category |
| `meta` | `{fps, resolution, total_frames}` |
| `vlm_causality` | Whether the VLM judges this video as having causal relationships (bool) |
| `human_discriminable` | Whether humans can distinguish forward from backward for this video (bool) |

## Important Implementation Details

1. **Same noise**: Forward and backward videos use the same `noise_seed` to ensure fair comparison.
2. **Same prompt**: Both forward and backward use the forward video's caption.
3. **K=10 timesteps**: The paper recommends uniformly sampling 10 timesteps (excluding boundaries).
4. **Windowing**: Long videos need to be sliced into multiple windows; context frames are excluded from loss.
5. **VAE uses mean**: VAE encoding uses the mean of the latent distribution (not sample) to ensure reproducibility.
