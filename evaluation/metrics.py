"""
YoCausal Metrics — RSI and CCI computation.

RSI (Reverse Surprise Index): Measures the proportion of video pairs where
    the model assigns lower denoising loss to the forward video.
    RSI(D) = (1/|D|) * Σ_{Di∈D} (1/|Di|) * Σ_{x∈Di} 1[L(x^r) > L(x^f)]

CCI (Causality Cognition Index): Measures the gap between RSI on causal
    vs. non-causal video subsets.
    CCI(D) = RSI(Dc) - RSI(Dnc)
"""

from typing import Any, Dict, List, Optional, Tuple


def normalize_dataset_name(dataset_name: str) -> str:
    """Map raw dataset names to canonical 4-category names."""
    dl = dataset_name.lower()
    if "animal" in dl:
        return "animal"
    if "kinetic" in dl:
        return "kinetics"
    if "mit" in dl or "moments" in dl:
        return "mit"
    if "physic" in dl:
        return "physics"
    return dataset_name


def compute_winrate(results: List[Dict[str, Any]]) -> Tuple[float, int, int, int]:
    """Compute forward win rate from a list of per-video results.

    Args:
        results: List of dicts, each must have a "winner" field
                 with value "forward", "backward", or "tie".

    Returns:
        (win_rate, forward_wins, backward_wins, total)
    """
    forward_wins = 0
    backward_wins = 0
    for r in results:
        winner = r.get("winner", "")
        if winner == "forward":
            forward_wins += 1
        elif winner == "backward":
            backward_wins += 1
    total = len(results)
    win_rate = forward_wins / total if total > 0 else 0.0
    return win_rate, forward_wins, backward_wins, total


def compute_rsi(dataset_results: Dict[str, List[Dict[str, Any]]]) -> float:
    """Compute overall RSI as the mean of per-dataset win rates.

    RSI(D) = (1/|D|) * Σ RSI(Di)

    Args:
        dataset_results: {dataset_name: [per_video_results]}

    Returns:
        Overall RSI value in [0, 1].
    """
    winrates = []
    for ds_name, results in dataset_results.items():
        if results:
            wr, _, _, _ = compute_winrate(results)
            winrates.append(wr)
    return sum(winrates) / len(winrates) if winrates else 0.0


def load_vlm_causality_labels(vlm_dir: str) -> Dict[str, Dict[int, bool]]:
    """Load VLM-based causal/non-causal labels for each dataset.

    Args:
        vlm_dir: Path to directory containing VLM result JSONs
                 (e.g. animal-kingdom_fwd.json, mit_fwd.json, etc.)

    Returns:
        {dataset_name: {video_id: has_causality}}
    """
    vlm_path = Path(vlm_dir)
    labels = {}

    for filename, dataset_name in VLM_FILE_TO_DATASET.items():
        filepath = vlm_path / filename
        if not filepath.exists():
            continue

        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        ds_labels = {}
        for entry in data:
            vid = entry.get("id")
            response = entry.get("response", {})
            if isinstance(response, list):
                response = response[0] if response else {}
            has_causality = response.get("video_has_general_causality", False)
            if entry.get("status") == "success" and vid is not None:
                ds_labels[vid] = has_causality

        labels[dataset_name] = ds_labels

    return labels


def compute_cci(
    dataset_results: Dict[str, List[Dict[str, Any]]],
    causality_labels: Dict[str, Dict[int, bool]],
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Compute CCI = RSI(Dc) - RSI(Dnc).

    Args:
        dataset_results: {dataset_name: [per_video_results]}
            Each result must have "id" and "winner" fields.
        causality_labels: {dataset_name: {video_id: has_causality}}

    Returns:
        (cci, rsi_dc, rsi_dnc) — any may be None if insufficient data.
    """
    dc_winrates = []
    dnc_winrates = []

    for ds_name, results in dataset_results.items():
        if not results:
            continue

        ds_labels = causality_labels.get(ds_name, {})
        dc = [r for r in results if ds_labels.get(r.get("id"), False)]
        dnc = [r for r in results if not ds_labels.get(r.get("id"), False)]

        if dc:
            wr_dc, _, _, _ = compute_winrate(dc)
            dc_winrates.append(wr_dc)
        if dnc:
            wr_dnc, _, _, _ = compute_winrate(dnc)
            dnc_winrates.append(wr_dnc)

    rsi_dc = sum(dc_winrates) / len(dc_winrates) if dc_winrates else None
    rsi_dnc = sum(dnc_winrates) / len(dnc_winrates) if dnc_winrates else None
    cci = (rsi_dc - rsi_dnc) if (rsi_dc is not None and rsi_dnc is not None) else None

    return cci, rsi_dc, rsi_dnc


def _split_by_labels(
    results: List[Dict[str, Any]],
    labels: Dict[int, bool],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split results into positive/negative groups based on labels."""
    pos = [r for r in results if labels.get(r.get("id"), False)]
    neg = [r for r in results if not labels.get(r.get("id"), False)]
    return pos, neg


def compute_per_dataset_metrics(
    dataset_results: Dict[str, List[Dict[str, Any]]],
    causality_labels: Optional[Dict[str, Dict[int, bool]]] = None,
    human_disc_labels: Optional[Dict[str, Dict[int, bool]]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Compute RSI and CCI metrics for each dataset independently.

    Args:
        dataset_results: {dataset_name: [per_video_results]}
        causality_labels: {dataset_name: {video_id: has_causality}}
        human_disc_labels: {dataset_name: {video_id: human_discriminable}}

    Returns:
        {dataset_name: {rsi, forward_wins, backward_wins, total,
                        rsi_dc, rsi_dnc, cci, num_causal, num_non_causal,
                        rsi_hd, rsi_hnd, num_human_disc, num_human_non_disc}}
    """
    metrics = {}

    for ds_name, results in dataset_results.items():
        if not results:
            continue

        wr, fw, bw, total = compute_winrate(results)
        entry = {
            "rsi": wr,
            "forward_wins": fw,
            "backward_wins": bw,
            "ties": total - fw - bw,
            "total": total,
        }

        # --- Causality (CCI) ---
        if causality_labels and ds_name in causality_labels:
            ds_labels = causality_labels[ds_name]
            dc, dnc = _split_by_labels(results, ds_labels)
            entry["num_causal"] = len(dc)
            entry["num_non_causal"] = len(dnc)
            entry["rsi_dc"] = compute_winrate(dc)[0] if dc else None
            entry["rsi_dnc"] = compute_winrate(dnc)[0] if dnc else None
            if entry["rsi_dc"] is not None and entry["rsi_dnc"] is not None:
                entry["cci"] = entry["rsi_dc"] - entry["rsi_dnc"]
            else:
                entry["cci"] = None
        else:
            entry["num_causal"] = None
            entry["num_non_causal"] = None
            entry["rsi_dc"] = None
            entry["rsi_dnc"] = None
            entry["cci"] = None

        # --- Human discriminability ---
        if human_disc_labels and ds_name in human_disc_labels:
            ds_hlabels = human_disc_labels[ds_name]
            hd, hnd = _split_by_labels(results, ds_hlabels)
            entry["num_human_disc"] = len(hd)
            entry["num_human_non_disc"] = len(hnd)
            entry["rsi_hd"] = compute_winrate(hd)[0] if hd else None
            entry["rsi_hnd"] = compute_winrate(hnd)[0] if hnd else None
        else:
            entry["num_human_disc"] = None
            entry["num_human_non_disc"] = None
            entry["rsi_hd"] = None
            entry["rsi_hnd"] = None

        metrics[ds_name] = entry

    return metrics


def compute_overall_metrics(
    dataset_results: Dict[str, List[Dict[str, Any]]],
    causality_labels: Optional[Dict[str, Dict[int, bool]]] = None,
    human_disc_labels: Optional[Dict[str, Dict[int, bool]]] = None,
) -> Dict[str, Any]:
    """Compute overall RSI and CCI across all datasets.

    RSI = mean of per-dataset RSIs (macro average).
    CCI = RSI(Dc) - RSI(Dnc), also macro averaged.

    Args:
        dataset_results: {dataset_name: [per_video_results]}
        causality_labels: {dataset_name: {video_id: has_causality}}
        human_disc_labels: {dataset_name: {video_id: human_discriminable}}

    Returns:
        dict with rsi, cci, rsi_dc, rsi_dnc, rsi_hd, rsi_hnd, etc.
    """
    rsi = compute_rsi(dataset_results)

    total_fw = sum(
        sum(1 for r in rs if r.get("winner") == "forward")
        for rs in dataset_results.values()
    )
    total_bw = sum(
        sum(1 for r in rs if r.get("winner") == "backward")
        for rs in dataset_results.values()
    )
    total = sum(len(rs) for rs in dataset_results.values())

    result = {
        "rsi": rsi,
        "forward_wins": total_fw,
        "backward_wins": total_bw,
        "ties": total - total_fw - total_bw,
        "total": total,
    }

    if causality_labels:
        cci, rsi_dc, rsi_dnc = compute_cci(dataset_results, causality_labels)
        result["rsi_dc"] = rsi_dc
        result["rsi_dnc"] = rsi_dnc
        result["cci"] = cci
    else:
        result["rsi_dc"] = None
        result["rsi_dnc"] = None
        result["cci"] = None

    # --- Human discriminability ---
    if human_disc_labels:
        hd_winrates = []
        hnd_winrates = []
        for ds_name, results in dataset_results.items():
            if not results:
                continue
            ds_hlabels = human_disc_labels.get(ds_name, {})
            hd, hnd = _split_by_labels(results, ds_hlabels)
            if hd:
                hd_winrates.append(compute_winrate(hd)[0])
            if hnd:
                hnd_winrates.append(compute_winrate(hnd)[0])
        result["rsi_hd"] = sum(hd_winrates) / len(hd_winrates) if hd_winrates else None
        result["rsi_hnd"] = sum(hnd_winrates) / len(hnd_winrates) if hnd_winrates else None
    else:
        result["rsi_hd"] = None
        result["rsi_hnd"] = None

    return result
