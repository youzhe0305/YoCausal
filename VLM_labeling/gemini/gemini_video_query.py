import argparse
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
load_dotenv(SCRIPT_DIR / ".env")
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_ROOT = PROJECT_ROOT / "data" / "processed_dataset"
DEFAULT_PROMPT_FILE = SCRIPT_DIR / "prompt.txt"

PRICING_PER_1M = {
    "input": 2,
    "output": 12.00,
    "thinking": 12.00,
}


class KeyManager:
    """Manages multiple Gemini API keys with automatic rotation on quota exhaustion."""

    def __init__(self, keys: list[str]):
        self.keys = keys
        self.current_idx = 0
        self.client = genai.Client(api_key=self.keys[0])
        print(f"Loaded {len(self.keys)} API key(s), starting with key #1")

    @property
    def current_key_num(self) -> int:
        return self.current_idx + 1

    def rotate(self) -> bool:
        next_idx = self.current_idx + 1
        if next_idx >= len(self.keys):
            return False
        self.current_idx = next_idx
        self.client = genai.Client(api_key=self.keys[self.current_idx])
        return True

    @staticmethod
    def load_keys_from_env() -> list[str]:
        keys = []
        i = 1
        while True:
            key = os.environ.get(f"GEMINI_API_KEY{i}")
            if not key:
                break
            keys.append(key)
            i += 1
        single = os.environ.get("GEMINI_API_KEY")
        if not keys and single:
            keys.append(single)
        return keys


def compute_cost(usage) -> dict:
    input_tokens = getattr(usage, "prompt_token_count", 0) or 0
    output_tokens = getattr(usage, "candidates_token_count", 0) or 0
    thinking_tokens = getattr(usage, "thoughts_token_count", 0) or 0

    input_cost = input_tokens / 1_000_000 * PRICING_PER_1M["input"]
    output_cost = output_tokens / 1_000_000 * PRICING_PER_1M["output"]
    thinking_cost = thinking_tokens / 1_000_000 * PRICING_PER_1M["thinking"]
    total_cost = input_cost + output_cost + thinking_cost

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "thinking_tokens": thinking_tokens,
        "cost_usd": total_cost,
    }


def load_metadata(dataset: str) -> list[dict]:
    metadata_path = DATA_ROOT / dataset / "dataset_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")
    with open(metadata_path, "r") as f:
        return json.load(f)


def get_video_paths(
    dataset: str, direction: str, id_start: int | None, id_end: int | None
) -> list[tuple[int, Path]]:
    metadata = load_metadata(dataset)
    id_map = {entry["id"]: entry for entry in metadata}

    if id_start is None and id_end is None:
        all_ids = sorted(id_map.keys())
    else:
        start = id_start if id_start is not None else min(id_map.keys())
        end = id_end if id_end is not None else max(id_map.keys())
        all_ids = range(start, end + 1)

    key = "video_path_forward" if direction == "fwd" else "video_path_backward"
    results = []
    for vid_id in all_ids:
        if vid_id not in id_map:
            print(f"[WARN] ID {vid_id} not found in metadata, skipping.")
            continue
        rel_path = id_map[vid_id][key].lstrip("/")
        full_path = PROJECT_ROOT / rel_path
        if not full_path.exists():
            print(f"[WARN] Video not found: {full_path}, skipping.")
            continue
        results.append((vid_id, full_path))
    return results


def is_quota_error(e: Exception) -> bool:
    return "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e)


def query_gemini(
    key_mgr: KeyManager,
    model_name: str,
    video_path: Path,
    prompt: str,
) -> tuple[str, dict]:
    video_data = video_path.read_bytes()
    video_part = types.Part.from_bytes(data=video_data, mime_type="video/mp4")

    response = key_mgr.client.models.generate_content(
        model=model_name,
        contents=[video_part, prompt],
        config=types.GenerateContentConfig(
            temperature=0,
            response_mime_type="application/json",
            media_resolution=types.MediaResolution.MEDIA_RESOLUTION_HIGH,
            thinking_config=types.ThinkingConfig(thinking_level="HIGH"),
        ),
    )
    cost_info = compute_cost(response.usage_metadata)
    return response.text, cost_info


def main():
    parser = argparse.ArgumentParser(
        description="Query Gemini API with videos from processed_dataset"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset folder name under processed_dataset (e.g. mit, animal-kingdom, tiny-Kinetics-400, physics-IQ-benchmark)",
    )
    parser.add_argument(
        "--direction",
        type=str,
        default="fwd",
        choices=["fwd", "bwd"],
        help="Video direction: fwd or bwd (default: fwd)",
    )
    parser.add_argument(
        "--id-start",
        type=int,
        default=None,
        help="Start video ID (inclusive). If not set, starts from the first ID.",
    )
    parser.add_argument(
        "--id-end",
        type=int,
        default=None,
        help="End video ID (inclusive). If not set, ends at the last ID.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Prompt text (if not set, reads from prompt.txt in the same folder)",
    )
    parser.add_argument(
        "--prompt-file",
        type=str,
        default=None,
        help="Path to prompt file (default: level2-VLM/prompt.txt)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemini-3-pro-preview",
        help="Gemini model name (default: gemini-3.1-pro-preview)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Gemini API key (or set GEMINI_API_KEY env var)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file path (if not set, print to stdout)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Delay in seconds between API calls (default: 1.0)",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=30,
        help="Max retries per video on failure (default: 3)",
    )

    args = parser.parse_args()

    if args.api_key:
        api_keys = [args.api_key]
    else:
        api_keys = KeyManager.load_keys_from_env()
    if not api_keys:
        print("ERROR: No API keys found. Set GEMINI_API_KEY1, GEMINI_API_KEY2, ... in .env")
        sys.exit(1)

    key_mgr = KeyManager(api_keys)

    if args.prompt:
        prompt_text = args.prompt
    else:
        prompt_file = Path(args.prompt_file) if args.prompt_file else DEFAULT_PROMPT_FILE
        if not prompt_file.exists():
            print(f"ERROR: Prompt file not found: {prompt_file}")
            sys.exit(1)
        prompt_text = prompt_file.read_text(encoding="utf-8").strip()
        print(f"Loaded prompt from: {prompt_file}")

    id_start_str = str(args.id_start) if args.id_start is not None else "first"
    id_end_str = str(args.id_end) if args.id_end is not None else "last"
    print(f"Dataset: {args.dataset}/{args.direction}")
    print(f"ID range: {id_start_str} ~ {id_end_str}")
    print(f"Model: {args.model}")
    print(f"Prompt: {prompt_text[:80]}{'...' if len(prompt_text) > 80 else ''}")
    print("-" * 60)

    video_paths = get_video_paths(
        args.dataset, args.direction, args.id_start, args.id_end
    )
    if not video_paths:
        print("No videos found for the given parameters.")
        sys.exit(1)

    print(f"Found {len(video_paths)} video(s) to process.\n")

    metadata = load_metadata(args.dataset)
    id_map = {entry["id"]: entry for entry in metadata}

    output_path = Path(args.output) if args.output else None
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    existing_results = []
    completed_ids = set()
    if output_path and output_path.exists() and output_path.stat().st_size > 0:
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                existing_results = json.load(f)
            completed_ids = {r["id"] for r in existing_results}
            print(f"Resuming: found {len(completed_ids)} already completed ID(s) in {output_path}")
        except json.JSONDecodeError:
            print(f"[WARN] Output file {output_path} is corrupted, starting fresh.")
            existing_results = []

    video_paths = [
        (vid_id, vpath) for vid_id, vpath in video_paths if vid_id not in completed_ids
    ]
    if not video_paths:
        print("All videos in the given range are already processed.")
        sys.exit(0)

    print(f"Remaining {len(video_paths)} video(s) to process.\n")

    success, failed = 0, 0
    total_cost = 0.0
    pbar = tqdm(video_paths, desc="Processing", unit="video")
    for i, (vid_id, video_path) in enumerate(pbar):
        pbar.set_postfix(id=vid_id, ok=success, fail=failed, cost=f"${total_cost:.4f}")

        entry = None
        for attempt in range(1, args.max_retries + 1):
            try:
                response_text, cost_info = query_gemini(key_mgr, args.model, video_path, prompt_text)
                try:
                    response_json = json.loads(response_text)
                except json.JSONDecodeError:
                    response_json = response_text
                total_cost += cost_info["cost_usd"]
                tqdm.write(
                    f"  [ID {vid_id}] {video_path.name} => OK (key#{key_mgr.current_key_num}) | "
                    f"in:{cost_info['input_tokens']} out:{cost_info['output_tokens']} "
                    f"think:{cost_info['thinking_tokens']} | "
                    f"${cost_info['cost_usd']:.4f} (total: ${total_cost:.4f})"
                )
                entry = {
                    "id": vid_id,
                    "video": video_path.name,
                    "category": id_map.get(vid_id, {}).get("category", ""),
                    "response": response_json,
                    "status": "success",
                }
                success += 1
                break
            except Exception as e:
                if is_quota_error(e) and key_mgr.rotate():
                    tqdm.write(f"  [ID {vid_id}] Key#{key_mgr.current_key_num - 1} quota exhausted, switched to key#{key_mgr.current_key_num}")
                    continue
                if attempt < args.max_retries:
                    wait = min(2 ** attempt, 60)
                    tqdm.write(f"  [ID {vid_id}] Attempt {attempt}/{args.max_retries} failed: {e}, retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    tqdm.write(f"  [ID {vid_id}] {video_path.name} => FAILED after {args.max_retries} attempts: {e}")
                    failed += 1

        if entry is not None:
            existing_results.append(entry)
            if output_path:
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(existing_results, f, ensure_ascii=False, indent=2)

        if i < len(video_paths) - 1:
            time.sleep(args.delay)

    pbar.close()

    if not output_path:
        print("\n" + "=" * 60)
        print("ALL RESULTS:")
        print("=" * 60)
        print(json.dumps(existing_results, ensure_ascii=False, indent=2))

    print(f"\nDone. Success: {success}, Failed: {failed}, Total: {success + failed}")
    print(f"Total cost: ${total_cost:.4f}")


if __name__ == "__main__":
    main()
