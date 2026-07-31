#!/usr/bin/env python3
"""Mine high-quality Backtrader MCQs with verifier-backed answers."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional

import db
from agent_backends import get_agent, list_agents
from mcq_mining_core import (
    DEFAULT_AGENT_TIMEOUT_S,
    DEFAULT_CODE_TIMEOUT_S,
    PROMPTS_DIR,
    ROOT,
    VENV_PY,
    MinedQuestion,
    VerificationRun,
    build_options_from_values,
    invoke_agent,
    parse_generator_spec,
    render_mcq_prompt,
    run_verifier,
)

PROMPT_PATH = PROMPTS_DIR / "backtrader_network_generator.txt"
NO_TOOLS_SCRIPT = ROOT / "run_cursor_sdk_mcq_notool.mjs"
WITH_TOOLS_SCRIPT = ROOT / "run_cursor_mcq.py"
SOURCE_NAME = "mined_backtrader"
LOGS_DIR = ROOT / "logs"
DEFAULT_GENERATOR_MODEL = "gpt-5.5-medium"
DEFAULT_SOLVER_MODEL = "gpt-5.4"


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate, verify, screen, and store Backtrader MCQs."
    )
    parser.add_argument("--attempts", type=int, default=10, help="Mining rounds to run.")
    parser.add_argument("--interactive", "-I", action="store_true", help="Prompt for run settings.")
    parser.add_argument("--generator-agent", default="cursor", help="Registered generator agent.")
    parser.add_argument("--generator-model", default=DEFAULT_GENERATOR_MODEL, help="Generator model id.")
    parser.add_argument(
        "--generator-extra",
        action="append",
        default=[],
        help="Extra argv token forwarded to the generator agent. Repeat for each token.",
    )
    parser.add_argument("--no-tools-model", default=DEFAULT_SOLVER_MODEL, help="Cursor SDK no-tools model id.")
    parser.add_argument("--with-tools-model", default=DEFAULT_SOLVER_MODEL, help="Cursor Agent with-tools model id.")
    parser.add_argument("--agent-bin", default="agent", help="Cursor Agent CLI binary for with-tools runs.")
    parser.add_argument("--node-bin", default="node", help="Node binary for no-tools SDK runs.")
    parser.add_argument("--python", default=str(VENV_PY), help="Python interpreter for verifier code.")
    parser.add_argument("--generator-timeout", type=int, default=DEFAULT_AGENT_TIMEOUT_S)
    parser.add_argument("--code-timeout", type=int, default=DEFAULT_CODE_TIMEOUT_S)
    parser.add_argument("--no-tools-timeout", type=int, default=120)
    parser.add_argument("--with-tools-timeout", type=int, default=180)
    parser.add_argument("--run-name", default=None, help="Optional db/ subfolder name.")
    parser.add_argument("--base-seed", type=int, default=None, help="Base verifier seed.")
    parser.add_argument("--seed-step", type=int, default=1009, help="Seed delta between attempts.")
    parser.add_argument(
        "--choice-perturbations",
        type=int,
        default=8,
        help="Number of perturbed seeds to try for distractor values.",
    )
    return parser.parse_args(argv)


def _ask_value(label: str, current: Any, *, cast: Any = str, choices: Optional[list[str]] = None) -> Any:
    choices_text = f" options: {' / '.join(choices)}" if choices else ""
    raw = input(f"  {label} [{current}]:{choices_text} ").strip()
    if raw == "":
        return current
    if choices and raw not in choices:
        print(f"  (invalid choice {raw!r}; keeping {current!r})")
        return current
    try:
        return cast(raw)
    except Exception:
        print(f"  (could not parse {raw!r}; keeping {current!r})")
        return current


def apply_interactive_config(args: argparse.Namespace) -> argparse.Namespace:
    if not args.interactive:
        return args
    if not sys.stdin.isatty():
        print("[mining] --interactive ignored because stdin is not a TTY.")
        return args

    print("\n[mining] interactive config")
    print("  Press Enter to keep the current value.")
    agents = list_agents() or [args.generator_agent]
    args.attempts = _ask_value("Attempts", args.attempts, cast=int)
    args.generator_agent = _ask_value("Generator agent", args.generator_agent, choices=agents)
    args.generator_model = _ask_value("Generator model", args.generator_model)
    args.no_tools_model = _ask_value("No-tools solver model", args.no_tools_model)
    args.with_tools_model = _ask_value("With-tools solver model", args.with_tools_model)
    args.generator_timeout = _ask_value("Generator timeout seconds", args.generator_timeout, cast=int)
    args.code_timeout = _ask_value("Verifier timeout seconds", args.code_timeout, cast=int)
    args.no_tools_timeout = _ask_value("No-tools solver timeout seconds", args.no_tools_timeout, cast=int)
    args.with_tools_timeout = _ask_value("With-tools solver timeout seconds", args.with_tools_timeout, cast=int)
    return args


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def render_history(history: list[dict[str, Any]], limit: int = 8) -> str:
    if not history:
        return "(none yet)"
    lines = []
    for item in history[-limit:]:
        lines.append(
            f"- attempt {item.get('attempt')}: decision={item.get('decision')} "
            f"no_tools={item.get('no_tools_correct')} with_tools={item.get('with_tools_correct')} "
            f"reason={item.get('reason', '')}"
        )
    return "\n".join(lines)


def build_generator_prompt(history: list[dict[str, Any]]) -> str:
    template = PROMPT_PATH.read_text(encoding="utf-8")
    return template.replace("{HISTORY}", render_history(history))


def latest_report(output_dir: Path) -> tuple[Optional[Path], Optional[dict[str, Any]]]:
    reports = sorted(output_dir.glob("*/report.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not reports:
        return None, None
    try:
        return reports[0], json.loads(reports[0].read_text(encoding="utf-8"))
    except Exception:
        return reports[0], None


def first_record(report: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
    records = report.get("per_question") if isinstance(report, dict) else None
    if isinstance(records, list) and records and isinstance(records[0], dict):
        return records[0]
    return None


def usage_from_report(report: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
    usage = report.get("usage_totals") if isinstance(report, dict) else None
    return usage if isinstance(usage, dict) else None


def new_usage_totals() -> dict[str, dict[str, Any]]:
    return {
        "generator": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "models": {}},
        "no_tools": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "models": {}},
        "with_tools": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "models": {}},
    }


def _add_model_count(bucket: dict[str, Any], model: Optional[str]) -> None:
    if not model:
        return
    models = bucket.setdefault("models", {})
    models[model] = int(models.get(model, 0)) + 1


def add_usage(bucket: dict[str, Any], usage: Optional[dict[str, Any]], *, fallback_model: Optional[str]) -> None:
    if not isinstance(usage, dict):
        _add_model_count(bucket, fallback_model)
        return
    totals = usage.get("totals") if isinstance(usage.get("totals"), dict) else usage
    for key in ("input_tokens", "output_tokens", "total_tokens"):
        value = totals.get(key) if isinstance(totals, dict) else None
        if isinstance(value, (int, float)):
            bucket[key] = int(bucket.get(key, 0) + value)
    models = usage.get("models")
    if isinstance(models, list):
        for model in models:
            _add_model_count(bucket, model if isinstance(model, str) else None)
    else:
        model = usage.get("model") if isinstance(usage.get("model"), str) else fallback_model
        _add_model_count(bucket, model)


def usage_brief(bucket: dict[str, Any]) -> str:
    models = ", ".join(f"{model}x{count}" for model, count in sorted(bucket.get("models", {}).items()))
    return (
        f"in={bucket.get('input_tokens', 0)} "
        f"out={bucket.get('output_tokens', 0)} "
        f"total={bucket.get('total_tokens', 0)} "
        f"models={models or 'unknown'}"
    )


def final_counts() -> dict[str, int]:
    return {
        "attempts": 0,
        "generated_parseable": 0,
        "verified_well_pose": 0,
        "easy_discarded": 0,
        "saved_total": 0,
        "tool_solver_correct": 0,
        "tool_solver_wrong": 0,
        "generator_failed": 0,
        "parse_failed": 0,
        "verifier_fail": 0,
        "no_tools_runner_failed": 0,
    }


def run_command(
    cmd: list[str],
    *,
    cwd: Path,
    timeout: int,
    stdout_path: Path,
    stderr_path: Path,
) -> subprocess.CompletedProcess[str] | None:
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            stdin=subprocess.DEVNULL,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""
        stdout_path.write_text(stdout, encoding="utf-8")
        stderr_path.write_text(
            f"{stderr}\n[mining] command timed out after {timeout}s\n".strip() + "\n",
            encoding="utf-8",
        )
        return None
    stdout_path.write_text(proc.stdout or "", encoding="utf-8")
    stderr_path.write_text(proc.stderr or "", encoding="utf-8")
    return proc


def run_no_tools(*, args: argparse.Namespace, question_path: Path, attempt_dir: Path) -> dict[str, Any]:
    output_dir = attempt_dir / "no_tools_runs"
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        args.node_bin,
        str(NO_TOOLS_SCRIPT),
        "--input",
        str(question_path),
        "--output-dir",
        str(output_dir),
        "--model",
        args.no_tools_model,
        "--timeout",
        str(args.no_tools_timeout),
    ]
    proc = run_command(
        cmd,
        cwd=ROOT,
        timeout=max(args.no_tools_timeout + 60, 90),
        stdout_path=attempt_dir / "no_tools_stdout.txt",
        stderr_path=attempt_dir / "no_tools_stderr.txt",
    )
    report_path, report = latest_report(output_dir)
    return {
        "cmd": cmd,
        "returncode": proc.returncode if proc else 124,
        "report_path": str(report_path) if report_path else None,
        "report": report,
        "record": first_record(report),
        "artifacts_dir": str(output_dir),
    }


def run_with_tools(*, args: argparse.Namespace, question_path: Path, attempt_dir: Path) -> dict[str, Any]:
    output_dir = attempt_dir / "with_tools_runs"
    output_dir.mkdir(parents=True, exist_ok=True)
    python_bin = str(Path(args.python) if Path(args.python).exists() else Path(sys.executable))
    cmd = [
        python_bin,
        str(WITH_TOOLS_SCRIPT),
        "--input",
        str(question_path),
        "--output-dir",
        str(output_dir),
        "--agent-bin",
        args.agent_bin,
        "--model",
        args.with_tools_model,
        "--timeout",
        str(args.with_tools_timeout),
    ]
    proc = run_command(
        cmd,
        cwd=ROOT,
        timeout=max(args.with_tools_timeout + 60, 120),
        stdout_path=attempt_dir / "with_tools_stdout.txt",
        stderr_path=attempt_dir / "with_tools_stderr.txt",
    )
    report_path, report = latest_report(output_dir)
    return {
        "cmd": cmd,
        "returncode": proc.returncode if proc else 124,
        "report_path": str(report_path) if report_path else None,
        "report": report,
        "record": first_record(report),
        "artifacts_dir": str(output_dir),
    }


def runner_answer(record: Optional[dict[str, Any]]) -> Optional[str]:
    value = record.get("parsed_answer") if isinstance(record, dict) else None
    return value if isinstance(value, str) else None


def runner_correct(record: Optional[dict[str, Any]]) -> Optional[bool]:
    value = record.get("is_correct") if isinstance(record, dict) else None
    return value if isinstance(value, bool) else None


def write_verifier_artifacts(run: VerificationRun, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "stdout.txt").write_text(run.stdout, encoding="utf-8")
    (out_dir / "stderr.txt").write_text(run.stderr, encoding="utf-8")
    write_json(
        out_dir / "result.json",
        {
            "seed": run.seed,
            "returncode": run.returncode,
            "timed_out": run.timed_out,
            "question": run.question,
            "answer_value": run.answer_value,
            "answer_letter": run.answer_letter,
        },
    )


def build_mined_question(
    *,
    args: argparse.Namespace,
    spec: Any,
    attempt: int,
    attempt_dir: Path,
    base_seed: int,
) -> tuple[Optional[MinedQuestion], VerificationRun, list[VerificationRun], list[str]]:
    verifier_root = attempt_dir / "verifier"
    base_run = run_verifier(
        spec.code,
        seed=base_seed,
        work_dir=verifier_root / "base",
        timeout=args.code_timeout,
        python_path=args.python,
        pythonpath=[ROOT, ROOT / "Backtrader_MCQ"],
    )
    write_verifier_artifacts(base_run, verifier_root / "base")
    if not base_run.ok:
        reasons = []
        if base_run.timed_out:
            reasons.append("base verifier timed out")
        if base_run.returncode != 0:
            reasons.append(f"base verifier exited with {base_run.returncode}")
        if not base_run.answer_value:
            reasons.append("base verifier did not print ANSWER_VALUE")
        return None, base_run, [], reasons

    distractor_runs: list[VerificationRun] = []
    distractor_values: list[str] = []
    for idx in range(args.choice_perturbations):
        seed = base_seed + ((idx + 1) * args.seed_step)
        run = run_verifier(
            spec.code,
            seed=seed,
            work_dir=verifier_root / f"perturb_{idx + 1:02d}",
            timeout=args.code_timeout,
            python_path=args.python,
            pythonpath=[ROOT, ROOT / "Backtrader_MCQ"],
            script_name="verifier.py",
        )
        write_verifier_artifacts(run, verifier_root / f"perturb_{idx + 1:02d}")
        distractor_runs.append(run)
        if run.ok and run.answer_value:
            distractor_values.append(run.answer_value)
        unique = {str(base_run.answer_value).strip().lower()}
        unique.update(str(value).strip().lower() for value in distractor_values)
        if len(unique) >= 4:
            break

    try:
        options, answer, choices = build_options_from_values(
            correct_value=str(base_run.answer_value),
            distractor_values=distractor_values,
            shuffle_seed=base_seed + 17,
        )
    except ValueError as exc:
        return None, base_run, distractor_runs, [str(exc)]

    question = (base_run.question or spec.question).strip()
    raw_prompt = render_mcq_prompt(question, options)
    mined = MinedQuestion(
        question=question,
        options=options,
        answer=answer,
        raw_prompt=raw_prompt,
        code=spec.code,
        difficulty="mined",
        source=SOURCE_NAME,
        metadata={
            "source": SOURCE_NAME,
            "base_seed": base_seed,
            "choice_seeds": [run.seed for run in distractor_runs],
            "correct_value": base_run.answer_value,
            "choice_values": choices,
            "rationale": spec.rationale,
        },
    )
    return mined, base_run, distractor_runs, []


def record_runner_attempt(
    *,
    run_db: db.MiningRunDB,
    qid: str,
    args: argparse.Namespace,
    ground_truth: Optional[str],
    attempt: int,
    runner: dict[str, Any],
    eval_mode: str,
    model: str,
) -> None:
    record = runner.get("record")
    run_db.record_attempt(
        question_id=qid,
        eval_mode=eval_mode,
        model=model,
        answer=runner_answer(record),
        ground_truth=ground_truth,
        correct=runner_correct(record),
        attempt=attempt,
        usage=record.get("usage") if isinstance(record, dict) else None,
        metadata={
            "status": record.get("status") if isinstance(record, dict) else "runner_error",
            "artifacts_dir": runner.get("artifacts_dir"),
            "report_path": runner.get("report_path"),
            "tool_event_count": record.get("tool_event_count") if isinstance(record, dict) else None,
            "solution_path": record.get("solution_path") if isinstance(record, dict) else None,
            "solution_saved": record.get("solution_saved") if isinstance(record, dict) else None,
        },
    )


def main(argv: list[str] | None = None) -> int:
    args = apply_interactive_config(parse_args(argv or sys.argv[1:]))
    if args.attempts < 1:
        raise SystemExit("--attempts must be at least 1")
    if not PROMPT_PATH.exists():
        raise SystemExit(f"Missing prompt file: {PROMPT_PATH}")
    if not Path(args.python).exists():
        raise SystemExit(f"Verifier Python not found: {args.python}")

    generator = get_agent(args.generator_agent)
    if generator is None:
        raise SystemExit(f"Unknown generator agent: {args.generator_agent}")

    run_id = db.make_run_id(attempts=args.attempts, run_name=args.run_name)
    artifacts_dir = LOGS_DIR / run_id
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    run_db = db.MiningRunDB(run_id=run_id, artifacts_dir=artifacts_dir)
    question_index = run_db.load_question_index()
    history: list[dict[str, Any]] = []
    counts = final_counts()
    usage_totals = new_usage_totals()
    started_total = time.perf_counter()
    base_seed = args.base_seed if args.base_seed is not None else int(time.time())

    print(f"[mining] db run id: {run_id}")
    print(f"[mining] mcq file: {run_db.questions_path}")
    print(f"[mining] attempts file: {run_db.attempts_path}")
    print(f"[mining] summary file: {run_db.summary_path}")
    print(f"[mining] artifacts: {artifacts_dir}")
    print(f"[mining] attempts: {args.attempts}")
    print(f"[mining] verifier python: {args.python}")
    print(
        "[mining] models: "
        f"generator={args.generator_agent}/{args.generator_model}, "
        f"no_tools={args.no_tools_model}, with_tools={args.with_tools_model}"
    )
    if "CURSOR_API_KEY" not in os.environ:
        print("[mining] warning: CURSOR_API_KEY is not set; no-tools runs will fail.")

    for attempt in range(1, args.attempts + 1):
        counts["attempts"] += 1
        attempt_started = time.perf_counter()
        attempt_dir = run_db.attempt_dir(attempt)
        generator_dir = attempt_dir / "generator"
        generator_dir.mkdir(parents=True, exist_ok=True)
        attempt_seed = base_seed + ((attempt - 1) * args.seed_step * 10)

        print(f"\n[mining] attempt {attempt}/{args.attempts} seed={attempt_seed}")
        prompt = build_generator_prompt(history)
        (generator_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
        invo = invoke_agent(
            generator,
            prompt,
            cwd=generator_dir,
            timeout=args.generator_timeout,
            model=args.generator_model,
            user_extra=args.generator_extra,
        )
        (generator_dir / "stdout.txt").write_text(invo.stdout, encoding="utf-8")
        (generator_dir / "stderr.txt").write_text(invo.stderr, encoding="utf-8")
        add_usage(usage_totals["generator"], invo.usage, fallback_model=args.generator_model)

        attempt_summary: dict[str, Any] = {
            "attempt": attempt,
            "seed": attempt_seed,
            "generator_agent": args.generator_agent,
            "generator_model": args.generator_model,
            "no_tools_model": args.no_tools_model,
            "with_tools_model": args.with_tools_model,
            "generator_returncode": invo.returncode,
            "generator_usage": invo.usage,
            "decision": None,
        }

        if invo.returncode != 0:
            counts["generator_failed"] += 1
            attempt_summary.update({
                "decision": "generator_failed",
                "reason": invo.stderr[:500],
                "elapsed_sec": round(time.perf_counter() - attempt_started, 2),
            })
            history.append(attempt_summary)
            run_db.record_summary(attempt_summary)
            print(f"[mining] generator failed elapsed={attempt_summary['elapsed_sec']}s")
            continue

        try:
            spec = parse_generator_spec(invo.stdout)
        except Exception as exc:
            counts["parse_failed"] += 1
            attempt_summary.update({
                "decision": "parse_failed",
                "reason": str(exc),
                "elapsed_sec": round(time.perf_counter() - attempt_started, 2),
            })
            history.append(attempt_summary)
            run_db.record_summary(attempt_summary)
            print(f"[mining] parse failed elapsed={attempt_summary['elapsed_sec']}s: {exc}")
            continue

        counts["generated_parseable"] += 1
        (attempt_dir / "question_spec.txt").write_text(spec.question, encoding="utf-8")
        (attempt_dir / "verifier.py").write_text(spec.code, encoding="utf-8")

        mined, base_run, perturb_runs, verifier_reasons = build_mined_question(
            args=args,
            spec=spec,
            attempt=attempt,
            attempt_dir=attempt_dir,
            base_seed=attempt_seed,
        )
        attempt_summary.update({
            "base_verifier_returncode": base_run.returncode,
            "base_verifier_timed_out": base_run.timed_out,
            "base_answer_value": base_run.answer_value,
            "perturbation_count": len(perturb_runs),
        })

        if mined is None:
            counts["verifier_fail"] += 1
            attempt_summary.update({
                "decision": "verifier_fail",
                "labels": ["verifier-fail"],
                "reason": "; ".join(verifier_reasons),
                "elapsed_sec": round(time.perf_counter() - attempt_started, 2),
            })
            run_db.record_question_candidate(
                {
                    **attempt_summary,
                    "question": spec.question,
                    "options": {},
                    "answer": None,
                    "raw_prompt": spec.question,
                    "code": spec.code,
                    "base_verifier_stdout": base_run.stdout,
                    "base_verifier_stderr": base_run.stderr,
                },
                labels=["verifier-fail"],
            )
            history.append(attempt_summary)
            run_db.record_summary(attempt_summary)
            print(f"[mining] verifier failed elapsed={attempt_summary['elapsed_sec']}s: {attempt_summary['reason']}")
            continue

        well_defined, reasons = db.evaluate_well_defined(
            question=mined.question,
            options=mined.options,
            code=mined.code,
            verifier_returncode=base_run.returncode,
            verifier_timed_out=base_run.timed_out,
            answer=mined.answer,
        )
        attempt_summary.update({
            "ground_truth": mined.answer,
            "well_defined": well_defined,
            "well_defined_reasons": reasons,
        })
        if well_defined:
            counts["verified_well_pose"] += 1
        else:
            counts["verifier_fail"] += 1
            attempt_summary.update({
                "decision": "verifier_fail",
                "labels": ["verifier-fail"],
                "reason": "; ".join(reasons),
                "elapsed_sec": round(time.perf_counter() - attempt_started, 2),
            })
            run_db.record_question_candidate(
                {
                    **attempt_summary,
                    "question": mined.question,
                    "options": mined.options,
                    "answer": mined.answer,
                    "raw_prompt": mined.raw_prompt,
                    "code": mined.code,
                    "metadata": mined.metadata,
                },
                labels=["verifier-fail"],
            )
            history.append(attempt_summary)
            run_db.record_summary(attempt_summary)
            print(f"[mining] not well-posed elapsed={attempt_summary['elapsed_sec']}s: {attempt_summary['reason']}")
            continue

        question_path = attempt_dir / "question.jsonl"
        question_json_path = attempt_dir / "question.json"
        question_obj = {
            "question": mined.raw_prompt,
            "answer": mined.answer,
            "difficulty": mined.difficulty,
            "source": mined.source,
            "metadata": mined.metadata,
        }
        write_jsonl(question_path, [question_obj])
        write_json(question_json_path, question_obj)
        (attempt_dir / "question.txt").write_text(mined.raw_prompt, encoding="utf-8")

        no_tools = run_no_tools(args=args, question_path=question_path, attempt_dir=attempt_dir)
        write_json(attempt_dir / "no_tools_result.json", no_tools)
        no_record = no_tools.get("record")
        no_tools_correct = runner_correct(no_record)
        if no_record is None:
            counts["no_tools_runner_failed"] += 1
        add_usage(usage_totals["no_tools"], usage_from_report(no_tools.get("report")), fallback_model=args.no_tools_model)
        attempt_summary.update({
            "no_tools_correct": no_tools_correct,
            "no_tools_status": no_record.get("status") if isinstance(no_record, dict) else "runner_error",
            "no_tools_usage_totals": usage_from_report(no_tools.get("report")),
        })

        if no_tools_correct is True:
            counts["easy_discarded"] += 1
            qid, inserted = run_db.record_question(
                question=mined.question,
                options=mined.options,
                answer=mined.answer,
                code=mined.code,
                verifier_stdout=base_run.stdout,
                verifier_stderr=base_run.stderr,
                verifier_returncode=base_run.returncode,
                verifier_timed_out=base_run.timed_out,
                well_defined=True,
                well_defined_reasons=[],
                attempt=attempt,
                decision="easy_discarded",
                labels=["verified-well-pose", "easy-discarded"],
                metadata={
                    **mined.metadata,
                    "raw_prompt": mined.raw_prompt,
                    "no_tools_correct": no_tools_correct,
                    "no_tools_status": no_record.get("status") if isinstance(no_record, dict) else "runner_error",
                    "no_tools_artifacts_dir": no_tools.get("artifacts_dir"),
                },
                existing_index=question_index,
            )
            attempt_summary.update({
                "decision": "easy_discarded",
                "labels": ["verified-well-pose", "easy-discarded"],
                "question_id": qid,
                "question_inserted": inserted,
                "elapsed_sec": round(time.perf_counter() - attempt_started, 2),
            })
            history.append(attempt_summary)
            run_db.record_summary(attempt_summary)
            print(f"[mining] discarded easy elapsed={attempt_summary['elapsed_sec']}s")
            continue

        with_tools = run_with_tools(args=args, question_path=question_path, attempt_dir=attempt_dir)
        write_json(attempt_dir / "with_tools_result.json", with_tools)
        with_record = with_tools.get("record")
        with_tools_correct = runner_correct(with_record)
        add_usage(usage_totals["with_tools"], usage_from_report(with_tools.get("report")), fallback_model=args.with_tools_model)

        decision = "tool_solver_correct" if with_tools_correct is True else "tool_solver_wrong"
        labels = [
            "verified-well-pose",
            "tool-solver-correct" if with_tools_correct is True else "tool-solver-wrong",
        ]
        qid, inserted = run_db.record_question(
            question=mined.question,
            options=mined.options,
            answer=mined.answer,
            code=mined.code,
            verifier_stdout=base_run.stdout,
            verifier_stderr=base_run.stderr,
            verifier_returncode=base_run.returncode,
            verifier_timed_out=base_run.timed_out,
            well_defined=True,
            well_defined_reasons=[],
            attempt=attempt,
            decision=decision,
            labels=labels,
            metadata={
                **mined.metadata,
                "raw_prompt": mined.raw_prompt,
                "human_verification_required": with_tools_correct is not True,
                "no_tools_correct": no_tools_correct,
                "no_tools_status": no_record.get("status") if isinstance(no_record, dict) else "runner_error",
                "no_tools_artifacts_dir": no_tools.get("artifacts_dir"),
                "with_tools_correct": with_tools_correct,
                "with_tools_status": with_record.get("status") if isinstance(with_record, dict) else "runner_error",
                "with_tools_artifacts_dir": with_tools.get("artifacts_dir"),
            },
            existing_index=question_index,
        )
        record_runner_attempt(
            run_db=run_db,
            qid=qid,
            args=args,
            ground_truth=mined.answer,
            attempt=attempt,
            runner=no_tools,
            eval_mode="no_tools_screen",
            model=args.no_tools_model,
        )
        record_runner_attempt(
            run_db=run_db,
            qid=qid,
            args=args,
            ground_truth=mined.answer,
            attempt=attempt,
            runner=with_tools,
            eval_mode="with_tools_verify",
            model=args.with_tools_model,
        )

        counts["saved_total"] += 1
        if decision == "tool_solver_correct":
            counts["tool_solver_correct"] += 1
        else:
            counts["tool_solver_wrong"] += 1
        attempt_summary.update({
            "decision": decision,
            "labels": labels,
            "question_id": qid,
            "question_inserted": inserted,
            "with_tools_correct": with_tools_correct,
            "with_tools_status": with_record.get("status") if isinstance(with_record, dict) else "runner_error",
            "with_tools_usage_totals": usage_from_report(with_tools.get("report")),
            "elapsed_sec": round(time.perf_counter() - attempt_started, 2),
        })
        history.append(attempt_summary)
        run_db.record_summary(attempt_summary)
        print(
            f"[mining] stored {qid[:12]} decision={decision} "
            f"elapsed={attempt_summary['elapsed_sec']}s "
            f"generator_tokens={usage_brief(usage_totals['generator'])} "
            f"no_tools_tokens={usage_brief(usage_totals['no_tools'])} "
            f"with_tools_tokens={usage_brief(usage_totals['with_tools'])}"
        )

    elapsed_total = round(time.perf_counter() - started_total, 2)
    final_summary = {
        "run_id": run_id,
        "artifacts_dir": str(artifacts_dir),
        "summary_path": str(run_db.summary_path),
        "elapsed_sec": elapsed_total,
        "base_seed": base_seed,
        "models": {
            "generator_agent": args.generator_agent,
            "generator_model": args.generator_model,
            "no_tools_model": args.no_tools_model,
            "with_tools_model": args.with_tools_model,
        },
        "counts": counts,
        "usage_totals": usage_totals,
    }
    run_db.write_final_summary(final_summary)
    print(f"\n[mining] complete. Summary: {run_db.summary_path}")
    print(f"[mining] final summary: {run_db.final_summary_path}")
    print(
        "[mining] questions: "
        f"saved={counts['saved_total']} "
        f"tool_solver_correct={counts['tool_solver_correct']} "
        f"tool_solver_wrong={counts['tool_solver_wrong']} "
        f"easy_discarded={counts['easy_discarded']} "
        f"verified_well_pose={counts['verified_well_pose']}/{counts['generated_parseable']} "
        f"verifier_fail={counts['verifier_fail']}"
    )
    print(f"[mining] generator tokens: {usage_brief(usage_totals['generator'])}")
    print(f"[mining] no-tools solver tokens: {usage_brief(usage_totals['no_tools'])}")
    print(f"[mining] with-tools solver tokens: {usage_brief(usage_totals['with_tools'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
