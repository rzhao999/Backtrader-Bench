#!/usr/bin/env python3
"""Run Backtrader MCQs through Cursor's `agent -p` CLI with tools enabled."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


ANSWER_RE = re.compile(r"<<<\s*([A-D])\s*>>>", re.IGNORECASE)
DEFAULT_INPUT = "Backtrader_MCQ/backtrader_mcq_balanced_30_all_strategies.jsonl"
DEFAULT_OUTPUT_DIR = "cursor_agent_runs"
DEFAULT_AGENT_BIN = "agent"
DEFAULT_TIMEOUT = 120
DIFFICULTY_ORDER = {"easy": 0, "medium": 1, "hard": 2, "unknown": 3}


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run each MCQ through Cursor `agent -p` in an isolated folder."
    )
    parser.add_argument("--input", default=DEFAULT_INPUT, help="JSON or JSONL MCQ file.")
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for per-question logs and reports.",
    )
    parser.add_argument(
        "--agent-bin",
        default=DEFAULT_AGENT_BIN,
        help="Cursor Agent CLI binary name or path.",
    )
    parser.add_argument("--model", default=None, help="Cursor model id to pass as `--model`.")
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT,
        help="Per-question timeout in seconds.",
    )
    parser.add_argument(
        "--extra-arg",
        action="append",
        default=[],
        help="Extra argv token forwarded to Cursor Agent CLI. Repeat for each token.",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="Best-effort model listing for the installed Cursor Agent CLI.",
    )
    return parser.parse_args(argv)


def try_list_models(agent_bin: str) -> tuple[int, str]:
    candidates = [
        [agent_bin, "models"],
        [agent_bin, "models", "--json"],
        [agent_bin, "--list-models"],
        [agent_bin, "list-models"],
        [agent_bin, "model", "list"],
    ]
    last_rc = 1
    last_out = ""
    for cmd in candidates:
        try:
            cp = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
                timeout=20,
                stdin=subprocess.DEVNULL,
            )
        except Exception as exc:  # pragma: no cover - depends on local CLI.
            last_rc = 1
            last_out = f"{type(exc).__name__}: {exc}"
            continue
        combined = "\n".join(
            part for part in ((cp.stdout or "").strip(), (cp.stderr or "").strip()) if part
        ).strip()
        last_rc = cp.returncode
        last_out = combined
        if combined:
            return cp.returncode, combined
    return last_rc, last_out


def load_questions(input_path: Path) -> list[dict[str, Any]]:
    raw = input_path.read_text(encoding="utf-8")
    trimmed = raw.strip()
    if not trimmed:
        return []

    try:
        parsed = json.loads(trimmed)
        if isinstance(parsed, list):
            return [item for item in parsed if isinstance(item, dict)]
        if isinstance(parsed, dict):
            return [parsed]
    except json.JSONDecodeError:
        pass

    questions: list[dict[str, Any]] = []
    for line_no, line in enumerate(trimmed.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL on line {line_no}: {exc}") from exc
        if isinstance(item, dict):
            questions.append(item)
    return questions


def safe_path_segment(value: object) -> str:
    segment = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "unknown").strip())
    return segment.strip("_") or "unknown"


def build_prompt(question: str) -> str:
    return f"""You are answering a Backtrader multiple-choice benchmark question.

You may use tools, write code, inspect files in this workspace, and run Python as needed. The repository includes the Backtrader MCQ generation and checking code plus the dataset source files.

Before giving your final answer, you MUST save the code you used to solve the question in a file named `solution.py` in the current workspace directory. The file must be runnable Python and must contain the computation or checks that support your answer. If you use exploratory commands, consolidate the final solving logic into `solution.py`. Answers without `solution.py` do not count.

Return exactly one final answer in the form <<< X >>> where X is A, B, C, or D. Do not include any other final text.

{question}
"""


def parse_answer(text: str) -> str | None:
    match = ANSWER_RE.search(text or "")
    return match.group(1).upper() if match else None


def normalize_difficulty(question_obj: dict[str, Any]) -> str:
    difficulty = question_obj.get("difficulty")
    return difficulty.strip() if isinstance(difficulty, str) and difficulty.strip() else "unknown"


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    chunks = re.findall(r"[A-Za-z0-9_]+|[^\sA-Za-z0-9_]", text)
    return max(1, int(len(chunks) * 1.25 + 0.999))


def pick_number(obj: Any, keys: list[str]) -> int | float | None:
    if not isinstance(obj, dict):
        return None
    for key in keys:
        value = obj.get(key)
        if isinstance(value, (int, float)):
            return value
    return None


def normalize_usage(payload: Any, model: str | None) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else payload
    input_tokens = pick_number(
        usage,
        ["input_tokens", "inputTokens", "prompt_tokens", "promptTokens", "promptTokenCount"],
    )
    output_tokens = pick_number(
        usage,
        [
            "output_tokens",
            "outputTokens",
            "completion_tokens",
            "completionTokens",
            "candidatesTokenCount",
        ],
    )
    total_tokens = pick_number(usage, ["total_tokens", "totalTokens", "totalTokenCount"])
    if output_tokens is None and input_tokens is not None and total_tokens is not None:
        output_tokens = total_tokens - input_tokens
    if input_tokens is None and output_tokens is None and total_tokens is None:
        return None
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "cache_read_tokens": pick_number(
            usage,
            [
                "cache_read_tokens",
                "cacheReadTokens",
                "cache_read_input_tokens",
                "cacheReadInputTokens",
                "cached_input_tokens",
                "cachedInputTokens",
            ],
        ),
        "cache_creation_tokens": pick_number(
            usage,
            [
                "cache_creation_tokens",
                "cacheCreationTokens",
                "cache_write_input_tokens",
                "cacheWriteInputTokens",
            ],
        ),
        "model": payload.get("model") if isinstance(payload.get("model"), str) else model,
        "source": "exact",
    }


def extract_usage(payload: Any, model: str | None, seen: set[int] | None = None) -> dict[str, Any] | None:
    if seen is None:
        seen = set()
    if not isinstance(payload, dict):
        return None
    obj_id = id(payload)
    if obj_id in seen:
        return None
    seen.add(obj_id)

    direct = normalize_usage(payload, model)
    if direct:
        return direct
    for value in payload.values():
        if isinstance(value, dict):
            nested = extract_usage(value, model, seen)
            if nested:
                return nested
    return None


def extract_text_from_cursor_json(stdout: str) -> tuple[str, dict[str, Any] | None]:
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError:
        return stdout, None
    if not isinstance(payload, dict):
        return stdout, None

    text_candidates = [
        payload.get("result"),
        payload.get("text"),
        payload.get("response"),
        payload.get("message"),
        payload.get("output"),
    ]
    for candidate in text_candidates:
        if isinstance(candidate, str):
            return candidate, payload
    return stdout, payload


def run_cursor_agent(
    *,
    agent_bin: str,
    prompt: str,
    timeout: float,
    cwd: Path,
    extra: list[str],
    model: str | None,
) -> subprocess.CompletedProcess[str]:
    cmd = [
        agent_bin,
        "-p",
        "--output-format",
        "json",
        "--yolo",
        "--trust",
        "--workspace",
        str(cwd),
    ]
    if model:
        cmd.extend(["--model", model])
    cmd.extend(extra)
    cmd.append(prompt)
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
        cwd=cwd,
        stdin=subprocess.DEVNULL,
    )


def accuracy_by_difficulty(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        if record.get("expected_answer"):
            grouped.setdefault(record.get("difficulty") or "unknown", []).append(record)

    def sort_key(item: tuple[str, list[dict[str, Any]]]) -> tuple[int, str]:
        difficulty = item[0]
        return (DIFFICULTY_ORDER.get(difficulty.lower(), len(DIFFICULTY_ORDER)), difficulty)

    result: dict[str, dict[str, Any]] = {}
    for difficulty, items in sorted(grouped.items(), key=sort_key):
        answered = [record for record in items if record.get("parsed_answer") is not None]
        correct = [record for record in items if record.get("is_correct")]
        result[difficulty] = {
            "gradable": len(items),
            "answered": len(answered),
            "correct": len(correct),
            "accuracy_pct": round((len(correct) / len(items)) * 100, 2) if items else 0,
            "answered_pct": round((len(answered) / len(items)) * 100, 2) if items else 0,
        }
    return result


def aggregate_usage(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    numeric = [
        record["usage"]
        for record in records
        if isinstance(record.get("usage"), dict)
        and any(
            isinstance(record["usage"].get(key), (int, float))
            for key in ("input_tokens", "output_tokens", "total_tokens")
        )
    ]
    if not numeric:
        return None
    keys = ["input_tokens", "output_tokens", "total_tokens", "cache_read_tokens", "cache_creation_tokens"]
    totals = {key: 0 for key in keys}
    counts = {key: 0 for key in keys}
    models = set()
    for usage in numeric:
        if isinstance(usage.get("model"), str):
            models.add(usage["model"])
        for key in keys:
            value = usage.get(key)
            if isinstance(value, (int, float)):
                totals[key] += value
                counts[key] += 1
    return {
        "n_with_usage": len(numeric),
        "models": sorted(models),
        "totals": totals,
        "averages": {
            f"avg_{key}": round(totals[key] / counts[key], 2) if counts[key] else None
            for key in keys
        },
    }


def run_one(
    *,
    question_obj: dict[str, Any],
    index: int,
    total: int,
    output_root: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    question = str(question_obj.get("question") or "").strip()
    expected = str(question_obj.get("answer")).upper() if question_obj.get("answer") else None
    difficulty = normalize_difficulty(question_obj)
    q_dir = output_root / f"q_{index:03d}"
    workspace = q_dir / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)

    prompt = build_prompt(question)
    (q_dir / "prompt.txt").write_text(prompt + "\n", encoding="utf-8")
    (q_dir / "question_meta.json").write_text(json.dumps(question_obj, indent=2) + "\n", encoding="utf-8")

    started = dt.datetime.now()
    stdout = ""
    stderr = ""
    returncode: int | None = None
    status = "ok"
    usage: dict[str, Any] | None = None
    try:
        cp = run_cursor_agent(
            agent_bin=args.agent_bin,
            prompt=prompt,
            timeout=args.timeout,
            cwd=workspace,
            extra=args.extra_arg,
            model=args.model,
        )
        stdout = cp.stdout or ""
        stderr = cp.stderr or ""
        returncode = cp.returncode
        if cp.returncode != 0:
            status = "error"
    except subprocess.TimeoutExpired as exc:
        status = "timeout"
        stdout = exc.stdout or ""
        stderr = exc.stderr or f"Timed out after {args.timeout}s"
    except Exception as exc:  # pragma: no cover - depends on local CLI.
        status = "error"
        stderr = f"{type(exc).__name__}: {exc}"

    response_text, payload = extract_text_from_cursor_json(stdout)
    parsed = parse_answer(response_text)
    usage = extract_usage(payload, args.model) if payload else None
    if usage is None:
        usage = {
            "input_tokens": estimate_tokens(prompt),
            "output_tokens": estimate_tokens(response_text),
            "total_tokens": estimate_tokens(prompt) + estimate_tokens(response_text),
            "cache_read_tokens": None,
            "cache_creation_tokens": None,
            "model": args.model,
            "source": "estimated",
        }

    solution_path = workspace / "solution.py"
    solution_saved = solution_path.exists() and solution_path.stat().st_size > 0
    if status == "ok" and not solution_saved:
        status = "missing_solution"
        stderr = (stderr + "\n" if stderr else "") + "Tool-enabled solver did not save workspace/solution.py"
    correct = (parsed == expected and solution_saved) if expected else None
    elapsed_sec = round((dt.datetime.now() - started).total_seconds(), 1)
    record = {
        "index": index,
        "model": usage.get("model") or args.model,
        "status": status,
        "returncode": returncode,
        "difficulty": difficulty,
        "elapsed_sec": elapsed_sec,
        "parsed_answer": parsed,
        "expected_answer": expected,
        "is_correct": correct,
        "question_dir": str(q_dir),
        "workspace_dir": str(workspace),
        "solution_path": str(solution_path),
        "solution_saved": solution_saved,
        "usage": usage,
    }
    (q_dir / "response.txt").write_text(response_text, encoding="utf-8")
    (q_dir / "stdout.json").write_text(stdout, encoding="utf-8")
    (q_dir / "stderr.txt").write_text(stderr, encoding="utf-8")

    print(
        f"[CursorAgent] [{index}/{total}] model={record['model']} status={status} "
        f"difficulty={difficulty} parsed={parsed} expected={expected} "
        f"correct={record['is_correct']} solution_saved={record['solution_saved']} "
        f"elapsed={elapsed_sec}s"
    )
    return record


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    if args.list_models:
        rc, out = try_list_models(args.agent_bin)
        if out:
            print(out)
            return 0 if rc == 0 else 0
        print("Could not list models via the Cursor Agent CLI.", file=sys.stderr)
        return 1

    input_path = Path(args.input).resolve()
    questions = load_questions(input_path)
    if not questions:
        raise SystemExit(f"No questions found in {input_path}")

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{timestamp}_{safe_path_segment(args.model or 'default')}"
    output_root = Path(args.output_dir).resolve() / run_id
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"[CursorAgent] Input: {input_path}")
    print(f"[CursorAgent] Questions: {len(questions)}")
    print(f"[CursorAgent] Output: {output_root}")
    if args.model:
        print(f"[CursorAgent] Model: {args.model}")

    records: list[dict[str, Any]] = []
    summary_path = output_root / "summary.jsonl"
    for i, question_obj in enumerate(questions, start=1):
        record = run_one(
            question_obj=question_obj,
            index=i,
            total=len(questions),
            output_root=output_root,
            args=args,
        )
        records.append(record)
        with summary_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")

    gradable = [record for record in records if record.get("expected_answer")]
    correct = [record for record in gradable if record.get("is_correct")]
    usage_totals = aggregate_usage(records)
    report = {
        "agent": "CursorAgent",
        "model": args.model,
        "input_file": str(input_path),
        "total_questions": len(records),
        "gradable": len(gradable),
        "correct": len(correct),
        "accuracy_pct": round((len(correct) / len(gradable)) * 100, 2) if gradable else 0,
        "accuracy_by_difficulty": accuracy_by_difficulty(records),
        "per_question": records,
    }
    if usage_totals:
        report["usage_totals"] = usage_totals
    (output_root / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    for difficulty, stats in report["accuracy_by_difficulty"].items():
        print(
            f"[CursorAgent] Accuracy ({difficulty}): {stats['correct']}/{stats['gradable']} "
            f"({stats['accuracy_pct']:.2f}%). Answered: {stats['answered']}/{stats['gradable']} "
            f"({stats['answered_pct']:.2f}%)."
        )
    print(
        f"[CursorAgent] Accuracy: {len(correct)}/{len(gradable)} "
        f"({report['accuracy_pct']:.2f}%)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
