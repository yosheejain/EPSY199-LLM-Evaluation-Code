"""
Lightweight analysis helper for the collected tutoring runs.

Reads the JSONL dataset produced by `main.py` and prints aggregate scores per
model: total runs, how many were judged, average total score, and per-principle
averages plus the most frequently missing principles.

Optionally saves bar charts for each principle and an overall chart. Requires
`matplotlib` when `--plots-dir` is provided.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def parse_judge_json(text: str) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def extract_scores(record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if "judge_reply_parsed" in record:
        parsed = record["judge_reply_parsed"]
    elif "judge_reply_raw" in record:
        parsed = parse_judge_json(record["judge_reply_raw"])
    else:
        parsed = None
    if not parsed:
        return None
    scores = parsed.get("scores")
    missing = parsed.get("missing", [])
    return {"scores": scores, "missing": missing} if scores else None


def aggregate(
    records: Iterable[Dict[str, Any]], rubric: List[Dict[str, Any]]
) -> Dict[str, Any]:
    codes = [item["code"] for item in rubric]
    code_to_name = {item["code"]: item["name"] for item in rubric}
    stats: Dict[str, Any] = {}

    for record in records:
        model = record.get("model_key", "unknown")
        stats.setdefault(
            model,
            {
                "runs": 0,
                "judged": 0,
                "sum_scores": {code: 0 for code in codes},
                "missing_counts": {code: 0 for code in codes},
            },
        )
        stats[model]["runs"] += 1

        parsed = extract_scores(record)
        if not parsed:
            continue

        stats[model]["judged"] += 1
        scores = parsed["scores"]
        missing = set(parsed.get("missing", []))

        for code in codes:
            value = int(scores.get(code, 0))
            stats[model]["sum_scores"][code] += value
            if value == 0 or code in missing:
                stats[model]["missing_counts"][code] += 1

    return {"codes": codes, "code_to_name": code_to_name, "per_model": stats}


def format_summary(agg: Dict[str, Any]) -> str:
    codes: List[str] = agg["codes"]
    per_model: Dict[str, Any] = agg["per_model"]

    headers = ["model", "runs", "judged", "avg_total"] + [f"avg_{c}" for c in codes]
    col_widths = [max(len(h), 8) for h in headers]

    lines = []
    header_line = " | ".join(h.ljust(w) for h, w in zip(headers, col_widths))
    lines.append(header_line)
    lines.append("-" * len(header_line))

    for model, data in sorted(per_model.items()):
        runs = data["runs"]
        judged = data["judged"]
        sums = data["sum_scores"]
        avg_total = (
            sum(sums.values()) / (len(codes) * judged) if judged else 0.0
        )
        row = [
            model,
            str(runs),
            str(judged),
            f"{avg_total:0.2f}",
        ]
        for code in codes:
            avg = sums[code] / judged if judged else 0.0
            row.append(f"{avg:0.2f}")
        lines.append(" | ".join(cell.ljust(w) for cell, w in zip(row, col_widths)))

    lines.append("\nMost-missed principles per model (count of zero scores):")
    for model, data in sorted(per_model.items()):
        missing_counts = data["missing_counts"]
        sorted_missing = sorted(
            missing_counts.items(), key=lambda kv: kv[1], reverse=True
        )
        top_three = ", ".join(f"{code} ({count})" for code, count in sorted_missing[:3])
        lines.append(f"- {model}: {top_three}")

    return "\n".join(lines)


def compute_averages(
    agg: Dict[str, Any]
) -> Tuple[List[str], List[str], Dict[str, List[float]]]:
    """
    Prepare average scores for plotting.

    Returns:
        models: ordered list of model keys
        codes: list of rubric codes
        data: dict where keys are rubric codes plus "overall" and values are
              per-model averages (aligned to `models`).
    """
    codes: List[str] = agg["codes"]
    per_model: Dict[str, Any] = agg["per_model"]
    models = sorted(per_model.keys())
    data: Dict[str, List[float]] = {code: [] for code in codes}
    data["overall"] = []

    for model in models:
        entry = per_model[model]
        judged = entry["judged"]
        sums = entry["sum_scores"]
        overall_avg = (
            sum(sums.values()) / (len(codes) * judged) if judged else 0.0
        )
        data["overall"].append(overall_avg)
        for code in codes:
            avg = sums[code] / judged if judged else 0.0
            data[code].append(avg)

    return models, codes, data


def compute_question_averages(
    records: Iterable[Dict[str, Any]], codes: List[str]
) -> Tuple[List[str], List[str], Dict[str, List[float]]]:
    """
    Average overall score per misconception (question) per model.

    Returns:
        misconceptions: ordered list of misconception IDs
        models: ordered list of model keys
        data: dict model -> list of averages aligned to misconceptions
    """
    from collections import defaultdict

    totals = defaultdict(lambda: defaultdict(lambda: {"sum": 0.0, "count": 0}))
    models_seen = set()
    misconceptions_seen = set()

    for record in records:
        model = record.get("model_key", "unknown")
        mis = record.get("misconception_id", "unknown")
        parsed = extract_scores(record)
        if not parsed:
            continue
        scores = parsed["scores"]
        avg = sum(scores.get(code, 0) for code in codes) / len(codes)
        totals[mis][model]["sum"] += avg
        totals[mis][model]["count"] += 1
        models_seen.add(model)
        misconceptions_seen.add(mis)

    misconceptions = sorted(misconceptions_seen)
    models = sorted(models_seen)
    data: Dict[str, List[float]] = {m: [] for m in models}

    for mis in misconceptions:
        for model in models:
            entry = totals[mis][model]
            count = entry["count"]
            data[model].append(entry["sum"] / count if count else 0.0)

    return misconceptions, models, data


def plot_bars(
    models: List[str],
    codes: List[str],
    data: Dict[str, List[float]],
    code_to_name: Dict[str, str],
    per_model_stats: Dict[str, Any],
    records: List[Dict[str, Any]],
    out_dir: Path,
) -> None:
    """
    Save grouped charts in three images:
    - overall_performance.png
    - question_wise.png
    - principles.png
    """
    try:
        import matplotlib.pyplot as plt  # type: ignore
        import numpy as np  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib (and numpy) are required for plotting. "
            "Install with `pip install matplotlib numpy`."
        ) from exc

    out_dir.mkdir(parents=True, exist_ok=True)

    plt.style.use("seaborn-v0_8-whitegrid")
    palette = plt.cm.Pastel2.colors
    accent = "#222831"
    bg = "#f7f7f9"
    title_kwargs = {"fontsize": 13, "fontweight": "bold", "color": accent}
    label_kwargs = {"color": "#444"}

    # Helpers
    def annotate_bars(ax, bars):
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.05,
                f"{height:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    # Overall Performance (overall average + judged counts)
    overall_vals = data["overall"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), facecolor=bg)
    for ax in axes:
        ax.set_facecolor(bg)
    # Overall averages
    bars = axes[0].bar(models, overall_vals, color=palette[0])
    axes[0].set_ylim(0, 2)
    axes[0].set_title("Overall average (0–2)", **title_kwargs)
    axes[0].set_ylabel("Average score", **label_kwargs)
    axes[0].grid(axis="y", linestyle="--", alpha=0.3, color="#ccc")
    annotate_bars(axes[0], bars)

    # Judged counts
    judged_counts = [per_model_stats[m]["judged"] for m in models]
    bars2 = axes[1].bar(models, judged_counts, color=palette[1])
    axes[1].set_title("Judged runs", **title_kwargs)
    axes[1].set_ylabel("Count of judged outputs", **label_kwargs)
    axes[1].grid(axis="y", linestyle="--", alpha=0.3, color="#ccc")
    for bar, val in zip(bars2, judged_counts):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            val + 2,
            f"{val}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    for ax in axes:
        ax.tick_params(axis="x", rotation=20)
    plt.tight_layout()
    plt.savefig(out_dir / "overall_performance.png", dpi=200)
    plt.close(fig)

    # Question-Wise Aggregates (overall avg per misconception per model)
    mis_ids, q_models, q_data = compute_question_averages(records, codes)
    x = np.arange(len(mis_ids))
    width = 0.12
    fig, ax = plt.subplots(figsize=(12, 5), facecolor=bg)
    ax.set_facecolor(bg)
    palette_q = plt.cm.Set3.colors
    for idx, model in enumerate(q_models):
        offsets = x + (idx - (len(q_models) - 1) / 2) * width
        vals = q_data[model]
        bars = ax.bar(
            offsets, vals, width, label=model, color=palette_q[idx % len(palette_q)]
        )
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.03,
                f"{height:.2f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )
    ax.set_xticks(x)
    ax.set_xticklabels(mis_ids)
    ax.set_ylim(0, 2)
    ax.set_ylabel("Average score (0–2)", **label_kwargs)
    ax.set_title("Question-wise overall averages", **title_kwargs)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.3, color="#ccc")
    plt.tight_layout()
    plt.savefig(out_dir / "question_wise.png", dpi=200)
    plt.close(fig)

    # Principles Insights (subplots grid)
    cols = 2
    rows = math.ceil(len(codes) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(12, 3.5 * rows), facecolor=bg)
    axes = axes.flatten()
    for idx, code in enumerate(codes):
        ax = axes[idx]
        ax.set_facecolor(bg)
        name = code_to_name.get(code, code)
        vals = data[code]
        bars = ax.bar(models, vals, color=palette[2 % len(palette)])
        ax.set_ylim(0, 2)
        ax.set_title(f"{code}: {name}", **title_kwargs)
        ax.grid(axis="y", linestyle="--", alpha=0.3, color="#ccc")
        annotate_bars(ax, bars)
        ax.tick_params(axis="x", rotation=20)
    # Hide unused subplots
    for j in range(idx + 1, len(axes)):
        fig.delaxes(axes[j])
    plt.tight_layout()
    plt.savefig(out_dir / "principles.png", dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize model performance from the collected JSONL dataset."
    )
    parser.add_argument(
        "--data",
        default="data/runs.jsonl",
        type=Path,
        help="Path to the JSONL dataset produced by main.py.",
    )
    parser.add_argument(
        "--rubric",
        default="prompts/learning-science-rubric.json",
        type=Path,
        help="Rubric path (used to preserve principle ordering).",
    )
    parser.add_argument(
        "--plots-dir",
        type=Path,
        default=None,
        help="If provided, save bar charts for each principle and overall into this directory.",
    )
    args = parser.parse_args()

    if not args.data.exists():
        raise FileNotFoundError(f"No dataset found at {args.data}")

    records = load_jsonl(args.data)
    rubric = load_json(args.rubric)

    agg = aggregate(records, rubric)
    print(format_summary(agg))

    if args.plots_dir:
        models, codes, data = compute_averages(agg)
        plot_bars(
            models,
            codes,
            data,
            agg["code_to_name"],
            agg["per_model"],
            records,
            args.plots_dir,
        )
        print(f"\nSaved plots to {args.plots_dir}")


if __name__ == "__main__":
    main()
