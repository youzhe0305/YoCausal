"""
YoCausal Evaluation Script — Run Arrow-of-Time evaluation for any video diffusion model.

This script takes a BaseVideoEvaluator implementation, iterates through dataset
video pairs, computes forward/backward denoising losses, and produces a JSON
report with per-video results plus RSI and CCI metrics.

Usage (standalone):
    python evaluate.py \\
        --evaluator_module path.to.my_evaluator \\
        --evaluator_class MyModelEvaluator \\
        --dataset_paths data/processed_dataset/animal-kingdom/dataset_metadata.json \\
                        data/processed_dataset/mit/dataset_metadata.json \\
        --dataset_base_path /project/AoTbenchmark/data/processed_dataset \\
        --output result.json

Usage (programmatic):
    from evaluate import run_evaluation
    from my_evaluator import MyModelEvaluator

    evaluator = MyModelEvaluator(model_dir="weights/", device="cuda")
    results = run_evaluation(
        evaluator=evaluator,
        dataset_paths=["data/.../dataset_metadata.json"],
    )
"""

import argparse
import importlib
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

from base_evaluator import BaseVideoEvaluator
from metrics import (
    compute_overall_metrics,
    compute_per_dataset_metrics,
    normalize_dataset_name,
)


def resolve_video_path(
    raw_path: str,
    dataset_base_path: Optional[str],
    dataset_json_path: str,
) -> str:
    """Resolve a video path from dataset metadata to an absolute path.

    Tries multiple strategies:
        1. If raw_path is already absolute and exists, use it.
        2. If dataset_base_path is given, join it with the relative portion.
        3. Resolve relative to the dataset JSON file's parent directory.
    """
    if os.path.isabs(raw_path) and os.path.exists(raw_path):
        return raw_path

    if dataset_base_path:
        # Strip leading /data/processed_dataset/ or similar prefixes
        for prefix in ["/data/processed_dataset/", "/processed_dataset/", "/data/"]:
            if raw_path.startswith(prefix):
                relative = raw_path[len(prefix):]
                candidate = os.path.join(dataset_base_path, relative)
                if os.path.exists(candidate):
                    return candidate

        # Try joining directly
        candidate = os.path.join(dataset_base_path, raw_path.lstrip("/"))
        if os.path.exists(candidate):
            return candidate

    # Resolve relative to dataset JSON directory
    json_dir = os.path.dirname(os.path.abspath(dataset_json_path))
    # Go up to find data root (dataset_metadata.json is typically 2-3 levels deep)
    for levels_up in range(4):
        base = json_dir
        for _ in range(levels_up):
            base = os.path.dirname(base)
        candidate = os.path.join(base, raw_path.lstrip("/"))
        if os.path.exists(candidate):
            return candidate

    # Fallback: return as-is (will fail at load time with a clear error)
    return raw_path


def load_datasets(
    dataset_paths: List[str],
    dataset_base_path: Optional[str] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """Load and group dataset metadata by canonical dataset name.

    Args:
        dataset_paths: List of paths to dataset_metadata.json files.
        dataset_base_path: Optional base path for resolving video paths.

    Returns:
        {dataset_name: [{"id", "prompt", "video_path_forward",
                         "video_path_backward", "dataset_source", ...}]}
    """
    grouped = {}

    for ds_path in dataset_paths:
        with open(ds_path, "r", encoding="utf-8") as f:
            entries = json.load(f)

        for entry in entries:
            # Determine canonical dataset name
            ds_source = entry.get("dataset_source", "")
            ds_name_raw = entry.get("dataset_name", ds_source)
            if not ds_name_raw:
                # Infer from path
                ds_name_raw = Path(ds_path).parent.name
            ds_name = normalize_dataset_name(ds_name_raw)

            # Resolve video paths
            entry["_resolved_fwd"] = resolve_video_path(
                entry["video_path_forward"], dataset_base_path, ds_path
            )
            entry["_resolved_bwd"] = resolve_video_path(
                entry["video_path_backward"], dataset_base_path, ds_path
            )
            entry["_dataset_json"] = ds_path

            grouped.setdefault(ds_name, []).append(entry)

    return grouped


def evaluate_single_pair(
    evaluator: BaseVideoEvaluator,
    entry: Dict[str, Any],
    timesteps: list,
    noise_seed: int,
) -> Dict[str, Any]:
    """Evaluate a single forward/backward video pair.

    Iterates over all timesteps, calls evaluator.compute_video_loss once per
    timestep, and averages the losses.

    Returns a result dict with forward_metrics, backward_metrics, and comparison.
    """
    fwd_path = entry["_resolved_fwd"]
    bwd_path = entry["_resolved_bwd"]
    prompt = entry.get("prompt", "")

    fwd_step_losses = {}
    bwd_step_losses = {}

    for t in timesteps:
        # Compute forward loss at this timestep
        fwd_loss = evaluator.compute_video_loss(
            video_path=fwd_path,
            prompt=prompt,
            timestep=t,
            noise_seed=noise_seed,
        )

        # Compute backward loss (same noise seed for fair comparison)
        bwd_loss = evaluator.compute_video_loss(
            video_path=bwd_path,
            prompt=prompt,
            timestep=t,
            noise_seed=noise_seed,
        )

        fwd_step_losses[str(t)] = fwd_loss
        bwd_step_losses[str(t)] = bwd_loss

    # Average over timesteps
    fwd_total = sum(fwd_step_losses.values()) / len(timesteps)
    bwd_total = sum(bwd_step_losses.values()) / len(timesteps)

    # Determine winner
    loss_diff = bwd_total - fwd_total
    if fwd_total < bwd_total:
        winner = "forward"
    elif bwd_total < fwd_total:
        winner = "backward"
    else:
        winner = "tie"

    return {
        "id": entry.get("id"),
        "dataset_source": entry.get("dataset_source", ""),
        "prompt": prompt,
        "video_path_forward": entry.get("video_path_forward", ""),
        "video_path_backward": entry.get("video_path_backward", ""),
        "original_resolution": entry.get("meta", {}).get("resolution"),
        "vlm_causality": entry.get("vlm_causality"),
        "human_discriminable": entry.get("human_discriminable"),
        "forward_metrics": {
            "total_accumulated_loss": fwd_total,
            "step_losses": fwd_step_losses,
        },
        "backward_metrics": {
            "total_accumulated_loss": bwd_total,
            "step_losses": bwd_step_losses,
        },
        "comparison": {
            "loss_diff": loss_diff,
            "winner": winner,
        },
    }


def run_evaluation(
    evaluator: BaseVideoEvaluator,
    dataset_paths: List[str],
    dataset_base_path: Optional[str] = None,
    output_path: Optional[str] = None,
    resume_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Run the full YoCausal evaluation pipeline.

    Causality labels (vlm_causality) and human discriminability labels
    (human_discriminable) are read directly from dataset_metadata.json entries.

    Args:
        evaluator: An instance of BaseVideoEvaluator.
        dataset_paths: Paths to dataset_metadata.json files.
        dataset_base_path: Base path for resolving video file paths.
        output_path: Where to write the output JSON. If None, not written.
        resume_path: Path to a partial result JSON to resume from.

    Returns:
        Complete evaluation result dict.
    """
    # --- Load datasets ---
    print(f"Loading datasets from {len(dataset_paths)} path(s)...")
    grouped_data = load_datasets(dataset_paths, dataset_base_path)
    total_videos = sum(len(v) for v in grouped_data.values())
    print(f"  Found {total_videos} video pairs across {len(grouped_data)} dataset(s): "
          f"{list(grouped_data.keys())}")

    # --- Extract labels from metadata ---
    causality_labels = {}  # {ds_name: {video_id: bool}}
    human_disc_labels = {}  # {ds_name: {video_id: bool}}
    for ds_name, entries in grouped_data.items():
        ds_causality = {}
        ds_human = {}
        for entry in entries:
            vid = entry.get("id")
            vlm_c = entry.get("vlm_causality")
            if vlm_c is not None:
                ds_causality[vid] = vlm_c
            hd = entry.get("human_discriminable")
            if hd is not None:
                ds_human[vid] = hd
        if ds_causality:
            causality_labels[ds_name] = ds_causality
            causal = sum(1 for v in ds_causality.values() if v)
            print(f"  {ds_name}: {len(ds_causality)} causality labels "
                  f"(Dc={causal}, Dnc={len(ds_causality)-causal})")
        if ds_human:
            human_disc_labels[ds_name] = ds_human
            disc = sum(1 for v in ds_human.values() if v)
            print(f"  {ds_name}: {len(ds_human)} human discriminability labels "
                  f"(discriminable={disc}, non-discriminable={len(ds_human)-disc})")

    # --- Get model config ---
    model_info = evaluator.get_model_info()
    eval_settings = evaluator.get_eval_settings()
    timesteps = evaluator.get_timesteps()
    eval_settings["timesteps_sampled"] = timesteps

    print(f"\nModel: {model_info.get('model_name', 'Unknown')}")
    print(f"Timesteps: {timesteps}")

    # --- Load partial results for resume ---
    completed_keys = set()
    all_per_video = []
    if resume_path and os.path.exists(resume_path):
        print(f"Resuming from {resume_path}...")
        with open(resume_path, "r", encoding="utf-8") as f:
            partial = json.load(f)
        all_per_video = partial.get("per_video_results", [])
        for r in all_per_video:
            completed_keys.add((r.get("dataset_source", ""), r.get("id")))
        print(f"  Loaded {len(completed_keys)} completed results")

    # --- Evaluate ---
    processed = 0
    errors = 0
    start_time = time.time()

    for ds_name in sorted(grouped_data.keys()):
        entries = grouped_data[ds_name]
        print(f"\n--- Dataset: {ds_name} ({len(entries)} pairs) ---")

        for idx, entry in enumerate(entries):
            video_id = entry.get("id")
            ds_source = entry.get("dataset_source", "")

            # Skip already completed (resume mode)
            if (ds_source, video_id) in completed_keys:
                continue

            noise_seed = video_id * 10000 if video_id is not None else idx * 10000

            try:
                result = evaluate_single_pair(evaluator, entry, timesteps, noise_seed)
                result["_dataset"] = ds_name
                all_per_video.append(result)
                completed_keys.add((ds_source, video_id))

                winner = result["comparison"]["winner"]
                fwd_l = result["forward_metrics"]["total_accumulated_loss"]
                bwd_l = result["backward_metrics"]["total_accumulated_loss"]
                processed += 1
                elapsed = time.time() - start_time
                rate = processed / elapsed if elapsed > 0 else 0

                print(
                    f"  [{processed}/{total_videos - len(completed_keys) + processed}] "
                    f"id={video_id} winner={winner} "
                    f"fwd={fwd_l:.6f} bwd={bwd_l:.6f} "
                    f"({rate:.2f} vid/s)"
                )

            except Exception as e:
                errors += 1
                print(f"  [ERROR] id={video_id}: {e}")
                traceback.print_exc()
                continue

            # Periodic checkpoint save
            if output_path and processed % 10 == 0:
                _save_checkpoint(
                    output_path, model_info, eval_settings,
                    dataset_paths, all_per_video,
                    causality_labels, human_disc_labels,
                )

    # --- Compute metrics ---
    print(f"\n{'='*60}")
    print(f"Evaluation complete. {processed} processed, {errors} errors.")

    output = _build_output(
        model_info, eval_settings, dataset_paths,
        all_per_video, causality_labels, human_disc_labels,
    )

    # --- Save final output ---
    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        print(f"\nResults saved to {output_path}")

    # --- Print summary ---
    _print_summary(output)

    return output


def _group_results_by_dataset(
    per_video_results: List[Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    """Group per-video results by canonical dataset name."""
    grouped = {}
    for r in per_video_results:
        ds = r.get("_dataset") or normalize_dataset_name(r.get("dataset_source", ""))
        grouped.setdefault(ds, []).append(r)
    return grouped


def _build_output(
    model_info: dict,
    eval_settings: dict,
    dataset_paths: List[str],
    per_video_results: List[Dict[str, Any]],
    causality_labels: Optional[Dict[str, Dict[int, bool]]],
    human_disc_labels: Optional[Dict[str, Dict[int, bool]]] = None,
) -> Dict[str, Any]:
    """Build the complete output JSON structure."""
    dataset_results = _group_results_by_dataset(per_video_results)

    # Convert to comparison-only format for metric computation
    comparison_data = {}
    for ds_name, results in dataset_results.items():
        comparison_data[ds_name] = [
            {"id": r["id"], "winner": r["comparison"]["winner"]}
            for r in results
        ]

    per_dataset = compute_per_dataset_metrics(
        comparison_data, causality_labels, human_disc_labels
    )
    overall = compute_overall_metrics(
        comparison_data, causality_labels, human_disc_labels
    )

    # Clean per-video results (remove internal fields)
    clean_results = []
    for r in per_video_results:
        clean = {k: v for k, v in r.items() if not k.startswith("_")}
        clean["dataset"] = r.get("_dataset") or normalize_dataset_name(
            r.get("dataset_source", "")
        )
        clean_results.append(clean)

    return {
        "model_info": model_info,
        "eval_settings": eval_settings,
        "dataset_paths": dataset_paths,
        "per_dataset_metrics": per_dataset,
        "overall_metrics": overall,
        "per_video_results": clean_results,
    }


def _save_checkpoint(
    output_path: str,
    model_info: dict,
    eval_settings: dict,
    dataset_paths: List[str],
    per_video_results: List[Dict[str, Any]],
    causality_labels: Optional[Dict[str, Dict[int, bool]]],
    human_disc_labels: Optional[Dict[str, Dict[int, bool]]] = None,
) -> None:
    """Save intermediate checkpoint during evaluation."""
    checkpoint_path = output_path + ".checkpoint"
    output = _build_output(
        model_info, eval_settings, dataset_paths,
        per_video_results, causality_labels, human_disc_labels,
    )
    os.makedirs(os.path.dirname(os.path.abspath(checkpoint_path)), exist_ok=True)
    with open(checkpoint_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)


def _print_summary(output: Dict[str, Any]) -> None:
    """Print a human-readable summary of the evaluation results."""
    overall = output["overall_metrics"]
    per_ds = output["per_dataset_metrics"]

    print(f"\n{'='*70}")
    print(f"  YoCausal Evaluation Summary — {output['model_info'].get('model_name', '?')}")
    print(f"{'='*70}")

    # Per-dataset table
    header = f"  {'Dataset':<12} | {'RSI':>7} | {'Fwd':>4} | {'Bwd':>4} | {'Total':>5}"
    if any(m.get("cci") is not None for m in per_ds.values()):
        header += f" | {'RSI(Dc)':>8} | {'RSI(Dnc)':>9} | {'CCI':>7}"
    print(header)
    print(f"  {'-'*len(header)}")

    for ds_name in sorted(per_ds.keys()):
        m = per_ds[ds_name]
        line = (
            f"  {ds_name:<12} | {m['rsi']*100:>6.2f}% | "
            f"{m['forward_wins']:>4} | {m['backward_wins']:>4} | {m['total']:>5}"
        )
        if m.get("cci") is not None:
            line += (
                f" | {m['rsi_dc']*100:>7.2f}% | {m['rsi_dnc']*100:>8.2f}% | "
                f"{m['cci']*100:>6.2f}%"
            )
        print(line)

    # Overall
    print(f"  {'-'*len(header)}")
    line = (
        f"  {'OVERALL':<12} | {overall['rsi']*100:>6.2f}% | "
        f"{overall['forward_wins']:>4} | {overall['backward_wins']:>4} | {overall['total']:>5}"
    )
    if overall.get("cci") is not None:
        line += (
            f" | {overall['rsi_dc']*100:>7.2f}% | {overall['rsi_dnc']*100:>8.2f}% | "
            f"{overall['cci']*100:>6.2f}%"
        )
    print(line)
    print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(
        description="YoCausal Evaluation — Run AoT evaluation for any video diffusion model."
    )
    parser.add_argument(
        "--evaluator_module", type=str, required=True,
        help="Python module path containing the evaluator class (e.g. 'my_evaluator')"
    )
    parser.add_argument(
        "--evaluator_class", type=str, required=True,
        help="Class name of the evaluator (must subclass BaseVideoEvaluator)"
    )
    parser.add_argument(
        "--evaluator_args", type=str, default="{}",
        help="JSON string of kwargs to pass to the evaluator constructor"
    )
    parser.add_argument(
        "--dataset_paths", type=str, nargs="+", required=True,
        help="Paths to dataset_metadata.json files"
    )
    parser.add_argument(
        "--dataset_base_path", type=str, default=None,
        help="Base path for resolving relative video file paths in dataset metadata"
    )
    parser.add_argument(
        "--output", type=str, default="evaluation_result.json",
        help="Output JSON file path"
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to a checkpoint JSON to resume from"
    )

    args = parser.parse_args()

    # Import the evaluator module and class
    # Add current directory and evaluator module's directory to sys.path
    sys.path.insert(0, os.getcwd())
    module_path = args.evaluator_module
    if os.path.isfile(module_path) or os.path.isfile(module_path + ".py"):
        # It's a file path, convert to module name
        module_dir = os.path.dirname(os.path.abspath(module_path))
        sys.path.insert(0, module_dir)
        module_name = os.path.splitext(os.path.basename(module_path))[0]
    else:
        module_name = module_path

    mod = importlib.import_module(module_name)
    evaluator_cls = getattr(mod, args.evaluator_class)

    # Instantiate evaluator
    evaluator_kwargs = json.loads(args.evaluator_args)
    print(f"Instantiating {args.evaluator_class} with kwargs: {evaluator_kwargs}")
    evaluator = evaluator_cls(**evaluator_kwargs)

    assert isinstance(evaluator, BaseVideoEvaluator), (
        f"{args.evaluator_class} must be a subclass of BaseVideoEvaluator"
    )

    # Run evaluation
    run_evaluation(
        evaluator=evaluator,
        dataset_paths=args.dataset_paths,
        dataset_base_path=args.dataset_base_path,
        output_path=args.output,
        resume_path=args.resume,
    )


if __name__ == "__main__":
    main()
