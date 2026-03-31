"""
Example: AnimateDiff-SD1.5 evaluator for YoCausal.

Demonstrates how to implement BaseVideoEvaluator for a DDPM-style model.
This is a reference template — adapt it for your own model.

Usage:
    python evaluate.py \\
        --evaluator_module example_animatediff_evaluator \\
        --evaluator_class AnimateDiffEvaluator \\
        --evaluator_args '{"model_dir": "/path/to/weights", "device": "cuda"}' \\
        --dataset_paths /path/to/dataset_metadata.json \\
        --output animatediff_result.json
"""

import subprocess
import tempfile
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from base_evaluator import BaseVideoEvaluator


class AnimateDiffEvaluator(BaseVideoEvaluator):
    """Example evaluator for AnimateDiff (SD 1.5 backbone, DDPM noise schedule)."""

    # ---------- Model-specific constants ----------
    TARGET_FPS = 8
    TARGET_HEIGHT = 512
    TARGET_WIDTH = 512
    WINDOW_SIZE = 16          # max frames per forward pass
    NUM_TIMESTEPS = 10        # K=10 as per paper
    T_MAX = 1000              # DDPM max timestep

    def __init__(self, model_dir: str, device: str = "cuda"):
        """Load model weights.

        Replace this with your actual model loading code.
        """
        self.device = torch.device(device)
        self.model_dir = model_dir

        # --- Pseudocode: replace with real loading ---
        # from diffusers import AutoencoderKL, UNet3DConditionModel, DDIMScheduler
        # from transformers import CLIPTokenizer, CLIPTextModel
        #
        # self.tokenizer = CLIPTokenizer.from_pretrained(model_dir, subfolder="tokenizer")
        # self.text_encoder = CLIPTextModel.from_pretrained(model_dir, subfolder="text_encoder").to(device)
        # self.vae = AutoencoderKL.from_pretrained(model_dir, subfolder="vae").to(device)
        # self.unet = UNet3DConditionModel.from_pretrained(model_dir, subfolder="unet").to(device)
        # self.scheduler = DDIMScheduler.from_pretrained(model_dir, subfolder="scheduler")
        #
        # self.text_encoder.eval()
        # self.vae.eval()
        # self.unet.eval()
        raise NotImplementedError(
            "This is an example template. Replace __init__ with your actual model loading code."
        )

    def get_model_info(self) -> dict:
        return {
            "model_name": "AnimateDiff-SD-1.5",
            "parameters": 1_276_658_564,
            "architecture": "UNet3D + CLIP",
        }

    def get_eval_settings(self) -> dict:
        return {
            "fps": self.TARGET_FPS,
            "resolution": [self.TARGET_HEIGHT, self.TARGET_WIDTH],
            "window_size": self.WINDOW_SIZE,
        }

    def get_timesteps(self) -> list:
        """K=10 uniformly sampled timesteps in [1, T_max), excluding boundaries."""
        K = self.NUM_TIMESTEPS
        return [int((i + 1) * self.T_MAX / (K + 1)) for i in range(K)]
        # e.g. [90, 181, 272, 363, 454, 544, 635, 726, 817, 908]

    def compute_video_loss(
        self,
        video_path: str,
        prompt: str,
        timestep: int,
        noise_seed: int,
    ) -> float:
        """Compute denoising loss for a single video at a single timestep."""
        # 1. Load and preprocess video
        frames = self._load_video(video_path)

        # 2. Encode prompt
        prompt_embeds = self._encode_prompt(prompt)

        # 3. Process in windows
        windows = self._slice_into_windows(frames)
        total_loss = 0.0

        for window_frames, valid_start in windows:
            # 4. Encode to latent space
            latents = self._encode_video(window_frames)

            # 5. Compute loss at this timestep
            # IMPORTANT: same seed for fwd/bwd
            torch.manual_seed(noise_seed)
            noise = torch.randn_like(latents)

            loss = self._compute_loss_at_timestep(
                latents, noise, prompt_embeds, timestep, valid_start
            )
            total_loss += loss

        return total_loss

    # ---------- Internal helpers (model-specific) ----------

    def _load_video(self, video_path: str) -> List[Image.Image]:
        """Load video, resample FPS, resize and center crop."""
        with tempfile.TemporaryDirectory() as tmp:
            out_pattern = f"{tmp}/frame_%06d.png"
            cmd = [
                "ffmpeg", "-i", video_path,
                "-vf", (
                    f"fps={self.TARGET_FPS},"
                    f"scale=-2:{self.TARGET_HEIGHT},"
                    f"crop={self.TARGET_WIDTH}:{self.TARGET_HEIGHT}"
                ),
                "-q:v", "2", out_pattern,
                "-y", "-loglevel", "error",
            ]
            subprocess.run(cmd, check=True)

            import glob
            frame_files = sorted(glob.glob(f"{tmp}/frame_*.png"))
            frames = [Image.open(f).convert("RGB") for f in frame_files]
        return frames

    def _encode_prompt(self, prompt: str):
        """Encode text prompt. Cache if prompt hasn't changed."""
        # tokens = self.tokenizer(prompt, padding="max_length",
        #                         max_length=77, truncation=True,
        #                         return_tensors="pt").to(self.device)
        # with torch.no_grad():
        #     embeds = self.text_encoder(**tokens).last_hidden_state
        # return embeds
        raise NotImplementedError

    def _encode_video(self, frames: List[Image.Image]) -> torch.Tensor:
        """Encode video frames to VAE latent space."""
        # Convert frames to tensor [B, C, F, H, W], normalize to [-1, 1]
        # tensors = [to_tensor(f) * 2 - 1 for f in frames]
        # video = torch.stack(tensors, dim=1).unsqueeze(0).to(self.device)  # [1, C, F, H, W]
        # with torch.no_grad():
        #     latents = self.vae.encode(video).latent_dist.mean
        #     latents = latents * self.vae.config.scaling_factor
        # return latents
        raise NotImplementedError

    def _compute_loss_at_timestep(
        self,
        latents: torch.Tensor,
        noise: torch.Tensor,
        prompt_embeds: torch.Tensor,
        timestep: int,
        valid_latent_start: int = 0,
    ) -> float:
        """Add noise and compute MSE between predicted and actual noise."""
        # # DDPM forward process
        # t = torch.tensor([timestep], device=self.device)
        # sqrt_alpha = self.scheduler.alphas_cumprod[timestep] ** 0.5
        # sqrt_one_minus_alpha = (1 - self.scheduler.alphas_cumprod[timestep]) ** 0.5
        # noisy = sqrt_alpha * latents + sqrt_one_minus_alpha * noise
        #
        # with torch.no_grad():
        #     pred = self.unet(noisy, t, encoder_hidden_states=prompt_embeds).sample
        #
        # # Only compute loss on valid frames (exclude context)
        # if valid_latent_start > 0:
        #     pred = pred[:, :, valid_latent_start:]
        #     target_noise = noise[:, :, valid_latent_start:]
        # else:
        #     target_noise = noise
        #
        # loss = F.mse_loss(pred, target_noise).item()
        # return loss
        raise NotImplementedError

    def _slice_into_windows(
        self, frames: List[Image.Image]
    ) -> List[Tuple[List[Image.Image], int]]:
        """Split frames into windows. Returns [(window_frames, valid_start)]."""
        if len(frames) <= self.WINDOW_SIZE:
            return [(frames, 0)]

        windows = []
        context_frames = 4  # overlap for temporal continuity

        start = 0
        while start < len(frames):
            end = min(start + self.WINDOW_SIZE, len(frames))
            if start == 0:
                windows.append((frames[start:end], 0))
            else:
                # Prepend context frames from previous segment
                ctx_start = max(0, start - context_frames)
                window = frames[ctx_start:end]
                valid_start = start - ctx_start  # context frames to skip in loss
                windows.append((window, valid_start))
            start = end

        return windows
