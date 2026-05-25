"""
Compare forward and backward video denoising MSE loss
Using the same noise and timestep to ensure fair comparison
For Wan2.2-TI2V-5B model (T2V mode)

v3-Wan2.2-5B Changes:
- Optimized for Wan2.2-TI2V-5B model (5B parameters, unified T2V/I2V)
- Uses Wan2.2_VAE (different from Wan2.1)
- Same features as v3: JSON dataset, FPS resampling, bucket resolution, sliding window
- 10 noise samples, single measurement per sample
- Supports start_idx/end_idx for parallel processing
- Running in pure T2V mode (no input_image)
"""
import torch
import os
import argparse
import json
import subprocess
from PIL import Image
from tqdm import tqdm
import numpy as np
import tempfile
import gc
from typing import List, Tuple, Optional

from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
from diffsynth.diffusion import FlowMatchScheduler


# =============================================================================
# Model Constants for Wan2.2-TI2V-5B
# =============================================================================
MODEL_TARGET_FPS = 16
MODEL_FRAME_LIMIT = 33  # 4n+1 format, typical max frames per window
MODEL_FRAME_LIMIT_LOW_VRAM = 17  # Reduced frames for low VRAM mode
TIME_DIVISION_FACTOR = 4
TIME_DIVISION_REMAINDER = 1
HEIGHT_DIVISION_FACTOR = 16
WIDTH_DIVISION_FACTOR = 16

# Bucket resolutions supported by Wan2.2 (height, width)
# Wan2.2-TI2V-5B supports larger resolutions
BUCKET_RESOLUTIONS = [
    (480, 832),   # ~16:9 landscape
    (832, 480),   # ~16:9 portrait
    (480, 720),   # 3:2 landscape
    (720, 480),   # 3:2 portrait
    (544, 960),   # ~16:9 larger
    (960, 544),   # ~16:9 larger portrait
    (704, 1248),  # ~16:9 HD (recommended for Wan2.2-TI2V-5B)
    (1248, 704),  # ~16:9 HD portrait
    (720, 1280),  # 16:9 HD
    (1280, 720),  # 16:9 HD portrait
]

# Lower resolution buckets for low VRAM mode
BUCKET_RESOLUTIONS_LOW_VRAM = [
    (320, 576),   # ~16:9 landscape, small
    (576, 320),   # ~16:9 portrait, small
    (384, 672),   # ~16:9 landscape
    (672, 384),   # ~16:9 portrait
    (480, 832),   # ~16:9 landscape
    (832, 480),   # ~16:9 portrait
]


DATASET_SUBSET_ALIASES = {
    "animal-kingdom": "animal",
    "mit": "general",
    "tiny-Kinetics-400": "human",
    "physics-IQ-benchmark": "physics",
}


def resolve_dataset_video_path(raw_path: str, dataset_base_path: str) -> str:
    """Resolve paths in either the source or released subset dataset layout."""
    if os.path.exists(raw_path):
        return raw_path

    relative = raw_path.lstrip("/")
    for prefix in ("data/processed_dataset/", "processed_dataset/", "data/"):
        if relative.startswith(prefix):
            relative = relative[len(prefix):]
            break

    candidates = [relative]
    first, separator, remainder = relative.partition("/")
    alias = DATASET_SUBSET_ALIASES.get(first)
    if alias:
        candidates.append(os.path.join(alias, remainder) if separator else alias)

    if dataset_base_path:
        for candidate in candidates:
            resolved = os.path.join(dataset_base_path, candidate)
            if os.path.exists(resolved):
                return resolved
        return os.path.join(dataset_base_path, candidates[0])
    return raw_path


class LossComputer:
    def __init__(self, pipe, use_gradient_checkpointing=False):
        self.pipe = pipe
        self.use_gradient_checkpointing = use_gradient_checkpointing

    def transfer_data_to_device(self, data, device, torch_float_dtype=None):
        if data is None:
            return data
        elif isinstance(data, torch.Tensor):
            data = data.to(device)
            if torch_float_dtype is not None and data.dtype in [torch.float, torch.float16, torch.bfloat16]:
                data = data.to(torch_float_dtype)
            return data
        elif isinstance(data, tuple):
            return tuple(self.transfer_data_to_device(x, device, torch_float_dtype) for x in data)
        elif isinstance(data, list):
            return list(self.transfer_data_to_device(x, device, torch_float_dtype) for x in data)
        elif isinstance(data, dict):
            return {k: self.transfer_data_to_device(v, device, torch_float_dtype) for k, v in data.items()}
        else:
            return data

    def prepare_inputs(self, video_frames, prompt=""):
        """Prepare pipeline inputs"""
        inputs_shared = {
            "input_video": video_frames,
            "height": video_frames[0].size[1],
            "width": video_frames[0].size[0],
            "num_frames": len(video_frames),
            "cfg_scale": 1,
            "tiled": False,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": False,
            "cfg_merge": False,
            "vace_scale": 1,
            "max_timestep_boundary": 1.0,
            "min_timestep_boundary": 0.0,
        }
        inputs_posi = {"prompt": prompt}
        inputs_nega = {}
        return inputs_shared, inputs_posi, inputs_nega

    def get_latents(self, video_frames, prompt=""):
        """Get the latent representation of video frames"""
        inputs = self.prepare_inputs(video_frames, prompt)
        inputs = self.transfer_data_to_device(inputs, self.pipe.device, self.pipe.torch_dtype)

        # Run preprocessing units to obtain latents
        for unit in self.pipe.units:
            inputs = self.pipe.unit_runner(unit, self.pipe, *inputs)

        return inputs

    def compute_loss_with_fixed_noise(self, inputs, noise, timestep, skip_latent_frames: int = 0):
        """Compute loss with fixed noise and timestep

        Args:
            inputs: Prepared pipeline inputs
            noise: Noise to add
            timestep: Timestep
            skip_latent_frames: Number of frames to skip from the beginning of latents (to exclude borrowed context frames)
        """
        inputs_shared, inputs_posi, inputs_nega = inputs

        # Add noise
        latents = self.pipe.scheduler.add_noise(inputs_shared["input_latents"], noise, timestep)
        training_target = self.pipe.scheduler.training_target(inputs_shared["input_latents"], noise, timestep)

        # Update inputs
        inputs_shared_copy = inputs_shared.copy()
        inputs_shared_copy["latents"] = latents

        # Ensure DiT model is loaded to GPU
        self.pipe.load_models_to_device(self.pipe.in_iteration_models)

        # Predict noise
        models = {name: getattr(self.pipe, name) for name in self.pipe.in_iteration_models}
        all_inputs = {**inputs_shared_copy, **inputs_posi}
        noise_pred = self.pipe.model_fn(**models, **all_inputs, timestep=timestep)

        # Compute MSE loss (excluding borrowed context frames)
        if skip_latent_frames > 0:
            # latent shape: (batch, channels, frames, height, width)
            noise_pred = noise_pred[:, :, skip_latent_frames:, :, :]
            training_target = training_target[:, :, skip_latent_frames:, :, :]

        loss = torch.nn.functional.mse_loss(noise_pred.float(), training_target.float())

        # Cleanup
        del latents, training_target, noise_pred
        torch.cuda.empty_cache()

        return loss.item()


def get_video_info(video_path: str) -> Tuple[float, int, int, int]:
    """Get video information using ffprobe"""
    cmd = [
        'ffprobe', '-v', 'quiet', '-print_format', 'json',
        '-show_streams', '-select_streams', 'v:0', video_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    info = json.loads(result.stdout)

    if 'streams' not in info or len(info['streams']) == 0:
        raise ValueError(f"Cannot read video info from {video_path}")

    stream = info['streams'][0]

    fps_str = stream.get('r_frame_rate', '30/1')
    if '/' in fps_str:
        num, den = map(int, fps_str.split('/'))
        fps = num / den if den != 0 else 30.0
    else:
        fps = float(fps_str)

    width = int(stream.get('width', 0))
    height = int(stream.get('height', 0))

    nb_frames = stream.get('nb_frames')
    if nb_frames:
        total_frames = int(nb_frames)
    else:
        duration = float(stream.get('duration', 0))
        total_frames = int(duration * fps)

    return fps, width, height, total_frames


def select_bucket_resolution(orig_width: int, orig_height: int, low_vram: bool = False) -> Tuple[int, int]:
    """Select the best bucket resolution based on original video dimensions."""
    orig_aspect = orig_width / orig_height

    bucket_list = BUCKET_RESOLUTIONS_LOW_VRAM if low_vram else BUCKET_RESOLUTIONS

    best_bucket = None
    best_score = float('inf')

    for bucket_h, bucket_w in bucket_list:
        bucket_aspect = bucket_w / bucket_h
        aspect_diff = abs(orig_aspect - bucket_aspect)

        scale_factor = max(bucket_w / orig_width, bucket_h / orig_height)
        upscale_penalty = max(0, scale_factor - 1) * 2

        score = aspect_diff + upscale_penalty

        if score < best_score:
            best_score = score
            best_bucket = (bucket_h, bucket_w)

    return best_bucket


def load_video_with_ffmpeg(
    video_path: str,
    target_fps: int,
    target_height: int,
    target_width: int,
    max_frames: Optional[int] = None
) -> List[Image.Image]:
    """Load video using ffmpeg with FPS resampling and resize/center crop."""
    with tempfile.TemporaryDirectory() as tmpdir:
        scale_filter = f"scale='if(gt(iw/ih,{target_width}/{target_height}),-2,{target_width})':'if(gt(iw/ih,{target_width}/{target_height}),{target_height},-2)'"
        crop_filter = f"crop={target_width}:{target_height}"

        filter_chain = f"fps={target_fps},{scale_filter},{crop_filter}"

        output_pattern = os.path.join(tmpdir, 'frame_%06d.png')

        cmd = [
            'ffmpeg', '-i', video_path,
            '-vf', filter_chain,
            '-vsync', 'vfr',
            output_pattern,
            '-y', '-loglevel', 'quiet'
        ]

        subprocess.run(cmd, check=True)

        frames = []
        frame_files = sorted([f for f in os.listdir(tmpdir) if f.endswith('.png')])

        if max_frames:
            frame_files = frame_files[:max_frames]

        for fname in frame_files:
            frame_path = os.path.join(tmpdir, fname)
            frame = Image.open(frame_path).convert('RGB')
            frames.append(frame)

        return frames


def adjust_frame_count(num_frames: int) -> int:
    """Adjust frame count to satisfy model's time_division requirement (4n+1)."""
    while num_frames > 1 and num_frames % TIME_DIVISION_FACTOR != TIME_DIVISION_REMAINDER:
        num_frames -= 1
    return num_frames


def slice_video_into_windows(
    frames: List[Image.Image],
    window_size: int
) -> List[Tuple[List[Image.Image], int]]:
    """
    Slice video into windows of specified size.
    If last segment is shorter, prepend frames from previous segment.

    Returns list of tuples: (frame_window, num_borrowed_frames)
    where num_borrowed_frames indicates how many frames at the beginning
    are borrowed from the previous segment (for context only, not for loss).
    """
    if len(frames) <= window_size:
        return [(frames, 0)]

    windows = []
    start = 0

    while start < len(frames):
        end = min(start + window_size, len(frames))
        window = frames[start:end]
        num_borrowed = 0

        if len(window) < window_size and start > 0:
            needed = window_size - len(window)
            prepend_start = max(0, start - needed)
            prepend_frames = frames[prepend_start:start]
            num_borrowed = len(prepend_frames)
            window = prepend_frames + window

        adjusted_size = adjust_frame_count(len(window))
        window = window[:adjusted_size]

        if len(window) >= 5:
            windows.append((window, num_borrowed))

        start = end

    return windows


def compute_video_loss(
    loss_computer: LossComputer,
    pipe: WanVideoPipeline,
    frames: List[Image.Image],
    prompt: str,
    timestep_indices: np.ndarray,
    num_noise_samples: int,
    window_size: int,
    noise_generator: torch.Generator,
) -> Tuple[float, dict]:
    """
    Compute total accumulated loss for a video using sliding windows.
    Borrowed context frames are excluded from loss calculation.
    """
    windows = slice_video_into_windows(frames, window_size)

    all_step_losses = {str(int(pipe.scheduler.timesteps[t_idx].item())): []
                       for t_idx in timestep_indices}
    total_loss = 0.0
    total_samples = 0

    for window_frames, num_borrowed in windows:
        inputs = loss_computer.get_latents(window_frames, prompt)

        # Calculate number of latent frames to skip (temporal compression factor is 4)
        # Borrowed frames are used as context but excluded from loss
        skip_latent_frames = num_borrowed // TIME_DIVISION_FACTOR

        for t_idx in timestep_indices:
            timestep = pipe.scheduler.timesteps[t_idx].unsqueeze(0).to(
                dtype=pipe.torch_dtype, device=pipe.device
            )
            timestep_key = str(int(pipe.scheduler.timesteps[t_idx].item()))

            for _ in range(num_noise_samples):
                latents = inputs[0]["input_latents"]
                noise = torch.randn(
                    latents.shape, generator=noise_generator,
                    device=latents.device, dtype=latents.dtype,
                )
                loss = loss_computer.compute_loss_with_fixed_noise(
                    inputs, noise, timestep, skip_latent_frames=skip_latent_frames
                )

                all_step_losses[timestep_key].append(loss)
                total_loss += loss
                total_samples += 1

        # Cleanup
        del inputs
        torch.cuda.empty_cache()
        gc.collect()

    step_losses = {k: float(np.mean(v)) for k, v in all_step_losses.items()}
    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0

    return avg_loss, step_losses


def main():
    parser = argparse.ArgumentParser(description="Compare forward and backward video denoising loss for Wan2.2-TI2V-5B (T2V mode)")

    # Dataset arguments
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="Path to dataset metadata JSON file")
    parser.add_argument("--dataset_base_path", type=str, default="",
                        help="Base path to prepend to video paths in dataset")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum number of samples to process")

    # Model arguments
    parser.add_argument("--model_id_with_origin_paths", type=str,
                        default="Wan-AI/Wan2.2-TI2V-5B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.2-TI2V-5B:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.2-TI2V-5B:Wan2.2_VAE.pth",
                        help="Model paths for Wan2.2-TI2V-5B")
    parser.add_argument("--target_fps", type=int, default=MODEL_TARGET_FPS, help="Target FPS for model")
    parser.add_argument("--window_size", type=int, default=MODEL_FRAME_LIMIT, help="Frame window size (4n+1)")
    parser.add_argument("--num_timesteps", type=int, default=10, help="Number of timesteps to sample")
    parser.add_argument("--num_noise_samples", type=int, default=10, help="Number of noise samples")
    parser.add_argument("--output_path", type=str, default="./result_output_wan2.2_5b.json", help="Output path")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--use_gradient_checkpointing", action="store_true", help="Use gradient checkpointing")
    parser.add_argument("--vram_limit", type=float, default=None,
                        help="VRAM limit in GB for CPU offloading")
    parser.add_argument("--low_vram", action="store_true",
                        help="Enable aggressive low VRAM mode for limited GPU memory")

    # Video index range arguments (for multi-GPU parallel processing)
    parser.add_argument("--start_idx", type=int, default=None,
                        help="Start index for video processing (inclusive)")
    parser.add_argument("--end_idx", type=int, default=None,
                        help="End index for video processing (exclusive)")

    args = parser.parse_args()

    # Adjust parameters for low VRAM mode
    if args.low_vram:
        if args.window_size == MODEL_FRAME_LIMIT:
            args.window_size = MODEL_FRAME_LIMIT_LOW_VRAM
            print(f"[Info] Low VRAM mode: reduced window_size to {args.window_size}")

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load model
    print("Loading Wan2.2-TI2V-5B model (T2V mode)...")

    vram_config = None
    if args.low_vram:
        vram_config = {
            "offload_dtype": "disk",
            "offload_device": "disk",
            "onload_dtype": torch.bfloat16,
            "onload_device": "cpu",
            "preparing_dtype": torch.bfloat16,
            "preparing_device": "cuda",
            "computation_dtype": torch.bfloat16,
            "computation_device": "cuda",
        }
        if args.vram_limit is None:
            args.vram_limit = torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 2
        print(f"[Info] Low VRAM mode enabled with disk offload, VRAM limit: {args.vram_limit:.1f}GB")
    elif args.vram_limit is not None:
        vram_config = {
            "offload_dtype": torch.bfloat16,
            "offload_device": "cpu",
            "onload_dtype": torch.bfloat16,
            "onload_device": "cuda",
            "preparing_dtype": torch.bfloat16,
            "preparing_device": "cuda",
            "computation_dtype": torch.bfloat16,
            "computation_device": "cuda",
        }
        print(f"[Info] VRAM management enabled with limit {args.vram_limit}GB")

    model_configs = []
    for config in args.model_id_with_origin_paths.split(","):
        parts = config.split(":")
        model_id = parts[0]
        origin_file_pattern = parts[1] if len(parts) > 1 else None
        if vram_config is not None:
            model_configs.append(ModelConfig(model_id=model_id, origin_file_pattern=origin_file_pattern, **vram_config))
        else:
            model_configs.append(ModelConfig(model_id=model_id, origin_file_pattern=origin_file_pattern))

    # tokenizer_config must be set, otherwise tokenizer will be None
    tokenizer_config = ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/umt5-xxl/")

    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=args.device,
        model_configs=model_configs,
        tokenizer_config=tokenizer_config,
        vram_limit=args.vram_limit,
    )

    pipe.scheduler.set_timesteps(1000, training=True)

    pipe.dit.eval()
    if pipe.vae is not None:
        pipe.vae.eval()
    if pipe.text_encoder is not None:
        pipe.text_encoder.eval()

    loss_computer = LossComputer(pipe, args.use_gradient_checkpointing)

    # Load dataset
    print(f"Loading dataset from {args.dataset_path}...")
    with open(args.dataset_path, "r") as f:
        dataset = json.load(f)

    if args.max_samples:
        dataset = dataset[:args.max_samples]

    # Support start_idx and end_idx
    total_len = len(dataset)
    start_idx = args.start_idx if args.start_idx is not None else 0
    end_idx = args.end_idx if args.end_idx is not None else total_len
    end_idx = min(end_idx, total_len)

    if start_idx > 0 or end_idx < total_len:
        dataset = dataset[start_idx:end_idx]
        print(f"[Info] Processing subset: indices {start_idx} to {end_idx} (total {end_idx - start_idx} videos)")

    print(f"Found {len(dataset)} videos to process")

    # Select timesteps excluding head and tail (divided into 10 equal parts)
    # Timestep values typically range 0-1000; we pick 10 uniformly spaced points in between
    all_timesteps = pipe.scheduler.timesteps.cpu().numpy()
    min_t, max_t = all_timesteps.min(), all_timesteps.max()

    # Compute equally spaced points excluding boundaries (divide range into num_timesteps+1 parts, take the middle num_timesteps split points)
    target_timesteps = np.linspace(min_t, max_t, args.num_timesteps + 2)[1:-1]

    # Find the actual timestep indices closest to these target values
    timestep_indices = []
    for target_t in target_timesteps:
        idx = np.argmin(np.abs(all_timesteps - target_t))
        timestep_indices.append(idx)
    timestep_indices = np.array(timestep_indices)

    sampled_timesteps = [int(pipe.scheduler.timesteps[t_idx].item()) for t_idx in timestep_indices]
    print(f"[Info] Sampled timesteps (excluding head/tail): {sampled_timesteps}")

    # Prepare result records
    results = []
    forward_wins = 0
    backward_wins = 0
    ties = 0

    step_loss_diffs = {str(t): [] for t in sampled_timesteps}

    with torch.no_grad():
        for idx in tqdm(range(len(dataset)), desc="Processing videos"):
            item = dataset[idx]

            try:
                video_id = item.get("id", idx)
                prompt = item.get("prompt", "")
                dataset_source = item.get("dataset_source", "unknown")

                fwd_path = item.get("video_path_forward", "")
                bwd_path = item.get("video_path_backward", "")

                fwd_path = resolve_dataset_video_path(fwd_path, args.dataset_base_path)
                bwd_path = resolve_dataset_video_path(bwd_path, args.dataset_base_path)

                if not os.path.exists(fwd_path):
                    print(f"[Warning] Forward video not found: {fwd_path}")
                    continue
                if not os.path.exists(bwd_path):
                    print(f"[Warning] Backward video not found: {bwd_path}")
                    continue

                orig_fps, orig_width, orig_height, total_frames = get_video_info(fwd_path)
                original_resolution = [orig_height, orig_width]

                target_height, target_width = select_bucket_resolution(orig_width, orig_height, args.low_vram)
                model_resolution = [target_height, target_width]

                print(f"\n[{idx+1}/{len(dataset)}] Video {video_id}")
                print(f"  Original: {orig_width}x{orig_height} @ {orig_fps:.2f}fps, {total_frames} frames")
                print(f"  Target: {target_width}x{target_height} @ {args.target_fps}fps")

                fwd_frames = load_video_with_ffmpeg(
                    fwd_path, args.target_fps, target_height, target_width
                )

                bwd_frames = load_video_with_ffmpeg(
                    bwd_path, args.target_fps, target_height, target_width
                )

                if len(fwd_frames) < 5 or len(bwd_frames) < 5:
                    print(f"  Skipping: too few frames (fwd={len(fwd_frames)}, bwd={len(bwd_frames)})")
                    continue

                num_frames = adjust_frame_count(min(len(fwd_frames), len(bwd_frames)))
                fwd_frames = fwd_frames[:num_frames]
                bwd_frames = bwd_frames[:num_frames]

                print(f"  Processing {num_frames} frames (window={args.window_size})")

                video_seed = args.seed + start_idx + idx
                fwd_generator = torch.Generator(device=pipe.device).manual_seed(video_seed)
                bwd_generator = torch.Generator(device=pipe.device).manual_seed(video_seed)

                fwd_avg_loss, fwd_step_losses = compute_video_loss(
                    loss_computer, pipe, fwd_frames, prompt,
                    timestep_indices, args.num_noise_samples, args.window_size, fwd_generator
                )

                bwd_avg_loss, bwd_step_losses = compute_video_loss(
                    loss_computer, pipe, bwd_frames, prompt,
                    timestep_indices, args.num_noise_samples, args.window_size, bwd_generator
                )

                loss_diff = bwd_avg_loss - fwd_avg_loss
                if fwd_avg_loss < bwd_avg_loss:
                    forward_wins += 1
                    winner = "forward"
                elif bwd_avg_loss < fwd_avg_loss:
                    backward_wins += 1
                    winner = "backward"
                else:
                    ties += 1
                    winner = "tie"

                for t_key in fwd_step_losses:
                    diff = bwd_step_losses[t_key] - fwd_step_losses[t_key]
                    step_loss_diffs[t_key].append(diff)

                result = {
                    "id": video_id,
                    "dataset_source": dataset_source,
                    "noise_seed": video_seed,
                    "prompt": prompt,
                    "video_path_forward": fwd_path,
                    "video_path_backward": bwd_path,
                    "original_resolution": original_resolution,
                    "model_resolution": model_resolution,
                    "forward_metrics": {
                        "total_accumulated_loss": fwd_avg_loss,
                        "step_losses": fwd_step_losses
                    },
                    "backward_metrics": {
                        "total_accumulated_loss": bwd_avg_loss,
                        "step_losses": bwd_step_losses
                    },
                    "comparison": {
                        "loss_diff": loss_diff,
                        "winner": winner
                    }
                }
                results.append(result)

                print(f"  Forward loss: {fwd_avg_loss:.6f}")
                print(f"  Backward loss: {bwd_avg_loss:.6f}")
                print(f"  Winner: {winner}")

                # Save results immediately (save after each video is processed)
                total_valid = len(results)
                ppe = backward_wins / total_valid if total_valid > 0 else 0.0
                avg_step_loss_diffs = {k: float(np.mean(v)) if v else 0.0 for k, v in step_loss_diffs.items()}
                model_params = sum(p.numel() for p in pipe.dit.parameters())

                output = {
                    "model_info": {
                        "model_name": "Wan2.2-TI2V-5B",
                        "parameters": model_params,
                        "mode": "T2V",
                        "vae": "Wan2.2_VAE",
                        "eval_settings": {
                            "fps": args.target_fps,
                            "total_eval_videos": total_valid,
                            "timesteps_sampled": sampled_timesteps,
                            "num_noise_samples": args.num_noise_samples,
                            "window_size": args.window_size
                        }
                    },
                    "dataset_info": {
                        "dataset_name": os.path.basename(os.path.dirname(args.dataset_path)),
                        "total_samples": total_valid,
                        "path": [args.dataset_path],
                        "index_range": {
                            "start_idx": start_idx,
                            "end_idx": end_idx
                        }
                    },
                    "summary_metrics": {
                        "forward_win_counts": forward_wins,
                        "backward_win_counts": backward_wins,
                        "ties": ties,
                        "total_valid_comparisons": total_valid,
                        "PPE": ppe,
                        "step_loss_differences": avg_step_loss_diffs
                    },
                    "results": results
                }

                os.makedirs(os.path.dirname(args.output_path) if os.path.dirname(args.output_path) else ".", exist_ok=True)
                with open(args.output_path, "w") as f:
                    json.dump(output, f, indent=2, ensure_ascii=False)
                print(f"  [Saved] Results saved to {args.output_path}")

                # Cleanup
                del fwd_frames, bwd_frames
                torch.cuda.empty_cache()
                gc.collect()

            except Exception as e:
                print(f"Error processing index {idx}: {e}")
                import traceback
                traceback.print_exc()
                torch.cuda.empty_cache()
                gc.collect()
                continue

    # Output final summary
    total_valid = len(results)
    ppe = backward_wins / total_valid if total_valid > 0 else 0.0

    print("\n" + "="*50)
    print("Summary (Wan2.2-TI2V-5B T2V mode):")
    print(f"  Total videos: {total_valid}")
    if total_valid > 0:
        print(f"  Forward wins: {forward_wins} ({forward_wins/total_valid*100:.2f}%)")
        print(f"  Backward wins: {backward_wins} ({backward_wins/total_valid*100:.2f}%)")
        print(f"  Ties: {ties}")
        print(f"  PPE: {ppe:.4f}")
    print("="*50)
    print(f"\nFinal results saved to {args.output_path}")


if __name__ == "__main__":
    main()
