from abc import ABC, abstractmethod
import torch

class CausalDiffusionEvaluator(ABC):
    @abstractmethod
    def __init__(self):
        """參賽者在此載入他們自己的 Base Model、VAE、Tokenizer 等權重。"""
        pass

    @abstractmethod
    def get_denoising_loss(self, video_frames: torch.Tensor, prompt: str, t: int) -> float:
        """
        輸入:
            video_frames: 形狀為 (B, C, F, H, W) 的標準化張量，數值範圍 [-1, 1]。
            prompt: 影片對應的文字描述。
            t: 介於 0 到 1000 的時間步 (Timestep)。
        輸出:
            該時間步下的純量 Loss 值 (Scalar)。
        
        參賽者必須在此函數內處理:
        1. VAE Encoding (video_frames -> z_0)
        2. 加入對應 t 的雜訊 (z_0 -> z_t)
        3. 執行 Model Forward Pass
        4. 回傳 Loss
        """
        pass