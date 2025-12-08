"""
Run tutoring prompts across multiple LLMs, collect responses, and optionally
grade them against the learning-science rubric.

This script reads the debugging misconceptions in `prompts/misconceptions.json`,
queries a set of tutor models, and (optionally) sends the tutor replies to a
judge model that scores the responses using `prompts/learning-science-rubric.json`.
Each run is appended to a JSONL dataset for later analysis.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

from models.chatgpt import get_chatgpt_response
from models.claude import get_claude_response
from models.llama import get_llama_response
from models.mistral import get_mistral_response
from models.qwen import get_qwen_response

DEFAULT_TUTOR_SYSTEM_PROMPT = (
    "You are a patient CS1 tutor helping a novice debug code. "
    "Encourage reasoning and metacognition. Guide them with short steps instead "
    "of dumping a full solution. Avoid writing long code unless necessary."
)

DEFAULT_JUDGE_SYSTEM_PROMPT = (
    "You are an expert learning-science grader. Score the tutor's reply using "
    "the provided rubric. Be strict and return JSON only."
)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def timestamp() -> str:
    return (
        dt.datetime.now(dt.timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def build_student_prompt(example: Dict[str, Any]) -> str:
    return (
        "A CS1 student described the following situation.\n\n"
        f"Context:\n{example.get('context', '').strip()}\n\n"
        "Their code:\n```python\n"
        f"{example.get('code', '').strip()}\n"
        "```\n\n"
        f"Question: {example.get('question', '').strip()}\n\n"
        "Act as a tutor."
    )


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


def _chatgpt_callable(model: str, temperature: float) -> Callable[[str, str], str]:
    return lambda prompt, system_prompt: get_chatgpt_response(
        prompt,
        system_prompt=system_prompt,
        model=model,
        temperature=temperature,
    )


def _claude_callable(model: str, temperature: float) -> Callable[[str, str], str]:
    return lambda prompt, system_prompt: get_claude_response(
        prompt,
        system_prompt=system_prompt,
        model=model,
        temperature=temperature,
    )


def _ollama_callable(
    fn: Callable[..., str], model: str, temperature: float
) -> Callable[[str, str], str]:
    return lambda prompt, system_prompt: fn(
        prompt, system_prompt=system_prompt, model=model, temperature=temperature
    )


def tutor_registry(args: argparse.Namespace) -> Dict[str, Callable[[str, str], str]]:
    return {
        "chatgpt": _chatgpt_callable(args.chatgpt_model, args.temperature),
        "claude": _claude_callable(args.claude_model, args.temperature),
        "llama": _ollama_callable(get_llama_response, args.llama_model, args.temperature),
        "mistral": _ollama_callable(
            get_mistral_response, args.mistral_model, args.temperature
        ),
        "qwen": _ollama_callable(get_qwen_response, args.qwen_model, args.temperature),
    }


def judge_registry(args: argparse.Namespace) -> Dict[str, Callable[[str, str], str]]:
    # For now, all judges reuse the same functions. This allows trying an open-source judge.
    registry = tutor_registry(args)
    return {name: fn for name, fn in registry.items()}


def parse_judge_json(text: str) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def append_record(path: Path, record: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=True))
        handle.write("\n")


def run_once(
    tutor_key: str,
    tutor_fn: Callable[[str, str], str],
    example: Dict[str, Any],
    rubric: List[Dict[str, Any]],
    args: argparse.Namespace,
    judge_fn: Optional[Callable[[str, str], str]] = None,
) -> Dict[str, Any]:
    student_prompt = build_student_prompt(example)
    tutor_reply = tutor_fn(student_prompt, DEFAULT_TUTOR_SYSTEM_PROMPT)

    record: Dict[str, Any] = {
        "timestamp": timestamp(),
        "misconception_id": example["id"],
        "model_key": tutor_key,
        "model_config": {
            "chatgpt_model": args.chatgpt_model,
            "claude_model": args.claude_model,
            "llama_model": args.llama_model,
            "mistral_model": args.mistral_model,
            "qwen_model": args.qwen_model,
            "temperature": args.temperature,
        },
        "student_prompt": student_prompt,
        "tutor_reply": tutor_reply,
    }

    if judge_fn:
        judge_prompt = build_judge_prompt(rubric, student_prompt, tutor_reply)
        judge_raw = judge_fn(judge_prompt, DEFAULT_JUDGE_SYSTEM_PROMPT)
        judge_parsed = parse_judge_json(judge_raw)
        record["judge_model_key"] = args.judge_model
        record["judge_prompt"] = judge_prompt
        record["judge_reply_raw"] = judge_raw
        if judge_parsed:
            record["judge_reply_parsed"] = judge_parsed

    return record


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Query tutor models on CS1 misconceptions and collect judged outputs."
    )
    parser.add_argument(
        "--misconceptions",
        default="prompts/misconceptions.json",
        type=Path,
        help="Path to the student debugging prompts.",
    )
    parser.add_argument(
        "--rubric",
        default="prompts/learning-science-rubric.json",
        type=Path,
        help="Path to the rubric used by the judge LLM.",
    )
    parser.add_argument(
        "--tutor-models",
        default="chatgpt",
        help="Comma-separated tutor model keys to run "
        "(chatgpt, claude, llama, mistral, qwen).",
    )
    parser.add_argument(
        "--judge-model",
        default="chatgpt",
        help="Model key to act as judge (same options as tutor).",
    )
    parser.add_argument(
        "--skip-judge",
        action="store_true",
        help="Collect tutor replies without sending them to a judge.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of misconceptions to run (for quick smoke tests).",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Number of times to repeat each misconception per tutor model "
        "(e.g., 100 for a 500-run dataset with 5 prompts).",
    )
    parser.add_argument(
        "--output",
        default="data/runs.jsonl",
        type=Path,
        help="JSONL output path for collected records.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output file instead of appending.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Generation temperature for tutors and judge.",
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
        "--ignore-errors",
        action="store_true",
        help="Skip a tutor run when a model call fails instead of stopping.",
    )

    args = parser.parse_args()

    misconceptions = load_json(args.misconceptions)
    rubric = load_json(args.rubric)

    tutor_keys = [key.strip() for key in args.tutor_models.split(",") if key.strip()]
    tutor_fns = tutor_registry(args)
    judge_fn = None
    if not args.skip_judge:
        judge_fns = judge_registry(args)
        judge_fn = judge_fns.get(args.judge_model)
        if judge_fn is None:
            raise ValueError(f"Unknown judge model key: {args.judge_model}")

    for key in tutor_keys:
        if key not in tutor_fns:
            raise ValueError(f"Unknown tutor model key: {key}")

    if args.overwrite and args.output.exists():
        args.output.unlink()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    subset = misconceptions[: args.limit] if args.limit else misconceptions

    for example in subset:
        for repeat_idx in range(args.repeat):
            for tutor_key in tutor_keys:
                try:
                    record = run_once(
                        tutor_key=tutor_key,
                        tutor_fn=tutor_fns[tutor_key],
                        example=example,
                        rubric=rubric,
                        args=args,
                        judge_fn=judge_fn,
                    )
                except Exception as exc:  # noqa: BLE001 - we want to keep going if asked
                    if args.ignore_errors:
                        print(
                            f"[{timestamp()}] skipped {example['id']} (rep {repeat_idx}) "
                            f"with {tutor_key} due to error: {exc}"
                        )
                        continue
                    raise

                record["repeat_index"] = repeat_idx
                append_record(args.output, record)
                print(
                    f"[{record['timestamp']}] saved {example['id']} (rep {repeat_idx}) "
                    f"with {tutor_key} -> {args.output}"
                )


if __name__ == "__main__":
    main()
