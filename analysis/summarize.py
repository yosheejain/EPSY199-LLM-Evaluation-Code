"""
Lightweight analysis helper for the collected tutoring runs.

Reads the JSONL dataset produced by `main.py` and prints aggregate scores per
model: total runs, how many were judged, average total score, and per-principle
averages plus the most frequently missing principles.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


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

    return {"codes": codes, "per_model": stats}


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
    args = parser.parse_args()

    if not args.data.exists():
        raise FileNotFoundError(f"No dataset found at {args.data}")

    records = load_jsonl(args.data)
    rubric = load_json(args.rubric)

    agg = aggregate(records, rubric)
    print(format_summary(agg))


if __name__ == "__main__":
    main()
