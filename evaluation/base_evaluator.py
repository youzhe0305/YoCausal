"""
YoCausal Base Evaluator — Abstract interface for video diffusion model evaluation.

Model implementors should subclass `BaseVideoEvaluator` and implement the required
methods. The evaluation script (`evaluate.py`) will call these methods to compute
denoising losses and produce RSI/CCI metrics.

Typical usage:
    1. Subclass BaseVideoEvaluator
    2. Implement __init__ (load your model weights)
    3. Implement get_model_info, get_timesteps, compute_video_loss
    4. Pass an instance to evaluate.py
"""

from abc import ABC, abstractmethod


class BaseVideoEvaluator(ABC):
    """Abstract base class for Arrow-of-Time denoising loss evaluation.

    Each model implementation handles its own:
    - Model loading and weight management
    - Video preprocessing (FPS resampling, resolution adaptation, cropping)
    - VAE encoding to latent space
    - Text prompt encoding
    - Noise scheduling (DDPM timesteps, flow-matching sigmas, etc.)
    - Denoising loss computation
    - Windowed processing for long videos
    """

    @abstractmethod
    def get_model_info(self) -> dict:
        """Return model metadata for the output JSON.

        Must include at minimum:
            {
                "model_name": str,       # e.g. "AnimateDiff-SD-1.5"
                "parameters": int,       # total parameter count
            }

        May optionally include any extra fields (architecture, version, etc.).
        """
        ...

    @abstractmethod
    def get_eval_settings(self) -> dict:
        """Return evaluation settings for the output JSON.

        Must include at minimum:
            {
                "fps": int,              # target FPS for video preprocessing
                "resolution": [H, W],    # target resolution [height, width]
                "window_size": int,      # max frames per forward pass
            }

        May optionally include extra fields (use_fp8, cpu_offload, etc.).
        """
        ...

    @abstractmethod
    def get_timesteps(self) -> list:
        """Return the list of timesteps (or sigmas) to evaluate at.

        For DDPM-style models: list of integer timesteps in [0, 1000),
            e.g. [90, 181, 272, ..., 908] (K=10 uniform samples).
        For flow-matching models: list of float sigmas in (0, 1),
            e.g. [0.09, 0.18, ..., 0.91].

        The paper recommends K=10 uniformly sampled timesteps excluding
        the boundaries (t=0 and t=T_max).
        """
        ...

    @abstractmethod
    def compute_video_loss(
        self,
        video_path: str,
        prompt: str,
        timestep,
        noise_seed: int,
    ) -> float:
        """Compute the denoising loss for a single video at a single timestep.

        The implementation should:
            1. Load and preprocess the video (FPS, resolution, crop)
            2. Encode the text prompt (caching is encouraged for efficiency)
            3. Encode video frames to latent space via VAE
            4. Set random seed to `noise_seed` (MUST use the same seed
               for every call so that fwd/bwd share identical noise)
            5. Sample noise ε ~ N(0, I)
            6. Add noise to latents at the given timestep (model-specific schedule)
            7. Run model forward pass to predict noise/velocity
            8. Compute MSE loss
            9. For long videos exceeding the window size, use windowed
               processing and accumulate losses across windows.

        The evaluation script handles iterating over multiple timesteps
        and averaging the losses.

        Args:
            video_path: Absolute path to the video file.
            prompt: Text prompt describing the forward video.
            timestep: A single timestep (int for DDPM) or sigma (float for
                flow-matching) to evaluate at.
            noise_seed: Random seed for noise generation. The evaluation script
                guarantees the same seed is used for both forward and backward
                videos in a pair.

        Returns:
            loss (float): Denoising loss at this timestep. Lower = higher
                model likelihood.
        """
        ...
