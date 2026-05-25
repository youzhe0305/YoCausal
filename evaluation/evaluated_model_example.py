from abc import ABC, abstractmethod
import torch

class CausalDiffusionEvaluator(ABC):
    @abstractmethod
    def __init__(self):
        """Participants load their own Base Model, VAE, Tokenizer, and other weights here."""
        pass

    @abstractmethod
    def get_denoising_loss(self, video_frames: torch.Tensor, prompt: str, t: int) -> float:
        """
        Input:
            video_frames: Normalized tensor of shape (B, C, F, H, W), value range [-1, 1].
            prompt: Text description corresponding to the video.
            t: Timestep between 0 and 1000.
        Output:
            Scalar loss value at the given timestep.

        Participants must handle the following in this function:
        1. VAE Encoding (video_frames -> z_0)
        2. Add noise corresponding to t (z_0 -> z_t)
        3. Execute Model Forward Pass
        4. Return Loss
        """
        pass