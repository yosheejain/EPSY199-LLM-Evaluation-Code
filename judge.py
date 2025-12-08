"""
Judge existing tutor replies using the learning-science rubric.

Reads a JSONL dataset (produced by main.py with or without judging), sends each
record through a judge LLM, and writes an updated JSONL file containing the
judge output. By default, only records missing a parsed or raw judge reply are
graded; use --regrade to overwrite all.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

from models.chatgpt import get_chatgpt_response
from models.claude import get_claude_response
from models.llama import get_llama_response
from models.mistral import get_mistral_response
from models.qwen import get_qwen_response

DEFAULT_JUDGE_SYSTEM_PROMPT = (
    "You are an expert learning-science grader. Score the tutor's reply using "
    "the provided rubric. Be strict and return JSON only."
)


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


def append_record(path: Path, record: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=True))
        handle.write("\n")


def format_rubric_text(rubric: Iterable[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for item in rubric:
        code = item["code"]
        name = item["name"]
        definition = item["definition"]
        scoring = item.get("scoring", {})
        lines.append(
            f"- {code} ({name}): {definition}\n"
            f"  2 = {scoring.get('2', '').strip()}\n"
            f"  1 = {scoring.get('1', '').strip()}\n"
            f"  0 = {scoring.get('0', '').strip()}"
        )
    return "\n".join(lines)


def build_judge_prompt(
    rubric: Iterable[Dict[str, Any]], student_prompt: str, tutor_reply: str
) -> str:
    rubric_text = format_rubric_text(rubric)
    expected_json = {
        "scores": {item["code"]: 0 for item in rubric},
        "missing": [],
        "evidence": {},
        "overall_comment": "",
    }
    return (
        "Grade the tutor's reply for evidence-based learning principles.\n"
        "Use the rubric (0–2) for each principle. If a principle is absent, "
        "assign 0 and list it under 'missing'. Include a short evidence note "
        "for any non-zero score. Respond with JSON only.\n\n"
        f"Rubric:\n{rubric_text}\n\n"
        "Student prompt:\n"
        f"{student_prompt}\n\n"
        "Tutor reply:\n"
        f"{tutor_reply}\n\n"
        "Return JSON in this shape:\n"
        f"{json.dumps(expected_json, indent=2)}"
    )


def judge_registry(args: argparse.Namespace) -> Dict[str, Callable[[str, str], str]]:
    # Reuse tutor callables for judging.
    return {
        "chatgpt": lambda prompt, system: get_chatgpt_response(
            prompt,
            system_prompt=system,
            model=args.chatgpt_model,
            temperature=args.temperature,
        ),
        "claude": lambda prompt, system: get_claude_response(
            prompt,
            system_prompt=system,
            model=args.claude_model,
            temperature=args.temperature,
        ),
        "llama": lambda prompt, system: get_llama_response(
            prompt,
            system_prompt=system,
            model=args.llama_model,
            temperature=args.temperature,
        ),
        "mistral": lambda prompt, system: get_mistral_response(
            prompt,
            system_prompt=system,
            model=args.mistral_model,
            temperature=args.temperature,
        ),
        "qwen": lambda prompt, system: get_qwen_response(
            prompt,
            system_prompt=system,
            model=args.qwen_model,
            temperature=args.temperature,
        ),
    }


def parse_judge_json(text: str) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def needs_judging(record: Dict[str, Any], regrade: bool) -> bool:
    if regrade:
        return True
    return "judge_reply_parsed" not in record and "judge_reply_raw" not in record


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply judge LLM to existing tutor replies."
    )
    parser.add_argument(
        "--input",
        default="data/runs.jsonl",
        type=Path,
        help="JSONL dataset with tutor replies.",
    )
    parser.add_argument(
        "--output",
        default="data/runs_judged.jsonl",
        type=Path,
        help="Where to write judged records (JSONL).",
    )
    parser.add_argument(
        "--rubric",
        default="prompts/learning-science-rubric.json",
        type=Path,
        help="Path to rubric JSON.",
    )
    parser.add_argument(
        "--judge-model",
        default="chatgpt",
        help="Judge model key: chatgpt, claude, llama, mistral, qwen.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Generation temperature for the judge.",
    )
    parser.add_argument(
        "--chatgpt-model",
        default="gpt-4o-mini",
        help="ChatGPT model identifier.",
    )
    parser.add_argument(
        "--claude-model",
        default="claude-sonnet-4-5",
        help="Claude model identifier.",
    )
    parser.add_argument(
        "--llama-model",
        default="llama3.1",
        help="Ollama model identifier for Llama.",
    )
    parser.add_argument(
        "--mistral-model",
        default="mistral:7b",
        help="Ollama model identifier for Mistral.",
    )
    parser.add_argument(
        "--qwen-model",
        default="qwen2:7b",
        help="Ollama model identifier for Qwen.",
    )
    parser.add_argument(
        "--regrade",
        action="store_true",
        help="Re-run judging even if records already contain judge output.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output file instead of appending.",
    )

    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"No input dataset found at {args.input}")
    rubric = load_json(args.rubric)
    records = load_jsonl(args.input)

    judge_fns = judge_registry(args)
    judge_fn = judge_fns.get(args.judge_model)
    if judge_fn is None:
        raise ValueError(f"Unknown judge model key: {args.judge_model}")

    if args.overwrite and args.output.exists():
        args.output.unlink()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    total = len(records)
    processed = 0
    for record in records:
        if needs_judging(record, args.regrade):
            judge_prompt = build_judge_prompt(
                rubric, record.get("student_prompt", ""), record.get("tutor_reply", "")
            )
            judge_raw = judge_fn(judge_prompt, DEFAULT_JUDGE_SYSTEM_PROMPT)
            judge_parsed = parse_judge_json(judge_raw)
            record["judge_model_key"] = args.judge_model
            record["judge_prompt"] = judge_prompt
            record["judge_reply_raw"] = judge_raw
            if judge_parsed:
                record["judge_reply_parsed"] = judge_parsed
            processed += 1
        append_record(args.output, record)

    print(
        f"Judged {processed} of {total} records -> {args.output}"
        f" (model={args.judge_model}, temp={args.temperature})"
    )


if __name__ == "__main__":
    main()
