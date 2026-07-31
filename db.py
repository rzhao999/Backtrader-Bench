"""Run-scoped flat JSON/JSONL storage for Backtrader MCQ mining."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parent
DB_DIR = ROOT / "db"
MCQ_DIR = DB_DIR / "mcq"
ATTEMPTS_DIR = DB_DIR / "attempts"
SUMMARY_DIR = DB_DIR / "summary"
LETTERS = ("A", "B", "C", "D")


def _now_iso() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


def _normalize_ws(value: str) -> str:
    return re.sub(r"\s+", " ", str(value)).strip()


def question_hash(question: str, options: dict[str, str]) -> str:
    parts = [_normalize_ws(question)]
    for letter in LETTERS:
        parts.append(f"{letter}:{_normalize_ws(options.get(letter, ''))}")
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()


def safe_path_segment(value: object) -> str:
    segment = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "").strip())
    return segment.strip("_") or "run"


def make_run_id(*, attempts: int, run_name: Optional[str] = None) -> str:
    for path in (MCQ_DIR, ATTEMPTS_DIR, SUMMARY_DIR):
        path.mkdir(parents=True, exist_ok=True)
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    base = f"mining_{timestamp}_rounds_{attempts}"
    if run_name:
        base = f"{base}_{safe_path_segment(run_name)}"
    if not _run_files_exist(base):
        return base
    suffix = 2
    while True:
        candidate = f"{base}_{suffix}"
        if not _run_files_exist(candidate):
            return candidate
        suffix += 1


def _run_files_exist(run_id: str) -> bool:
    return any(
        any(path.glob(f"{run_id}_*"))
        for path in (MCQ_DIR, ATTEMPTS_DIR, SUMMARY_DIR)
    )


def evaluate_well_defined(
    *,
    question: Optional[str],
    options: dict[str, str],
    code: Optional[str],
    verifier_returncode: int,
    verifier_timed_out: bool,
    answer: Optional[str],
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if not question or not question.strip():
        reasons.append("missing question text")
    for letter in LETTERS:
        if not options.get(letter, "").strip():
            reasons.append(f"missing option {letter}")
    if not code or not code.strip():
        reasons.append("missing verification code")
    if verifier_timed_out:
        reasons.append("verification code timed out")
    elif verifier_returncode != 0:
        reasons.append(f"verification code exited with non-zero status {verifier_returncode}")
    if answer not in LETTERS:
        reasons.append("could not determine a single ground-truth answer letter")
    return len(reasons) == 0, reasons


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                rows.append(item)
    return rows


@dataclass
class MiningRunDB:
    run_id: str
    artifacts_dir: Path

    @property
    def questions_path(self) -> Path:
        return MCQ_DIR / f"{self.run_id}_questions.jsonl"

    @property
    def attempts_path(self) -> Path:
        return ATTEMPTS_DIR / f"{self.run_id}_attempts.jsonl"

    @property
    def summary_path(self) -> Path:
        return SUMMARY_DIR / f"{self.run_id}_summary.jsonl"

    @property
    def final_summary_path(self) -> Path:
        return SUMMARY_DIR / f"{self.run_id}_final_summary.json"

    def load_question_index(self) -> dict[str, dict[str, Any]]:
        return {row["id"]: row for row in read_jsonl(self.questions_path) if row.get("id")}

    def attempt_dir(self, attempt: int) -> Path:
        path = self.artifacts_dir / f"attempt_{attempt:03d}"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def record_question(
        self,
        *,
        question: str,
        options: dict[str, str],
        answer: str,
        code: str,
        verifier_stdout: str,
        verifier_stderr: str,
        verifier_returncode: int,
        verifier_timed_out: bool,
        well_defined: bool,
        well_defined_reasons: list[str],
        attempt: int,
        decision: str,
        labels: list[str],
        metadata: Optional[dict[str, Any]] = None,
        existing_index: Optional[dict[str, dict[str, Any]]] = None,
    ) -> tuple[str, bool]:
        index = existing_index if existing_index is not None else self.load_question_index()
        qid = question_hash(question, options)
        if qid in index:
            return qid, False

        record: dict[str, Any] = {
            "type": "question",
            "id": qid,
            "created_at": _now_iso(),
            "run_id": self.run_id,
            "artifacts_dir": str(self.artifacts_dir),
            "attempt": attempt,
            "question": question,
            "options": {letter: options.get(letter, "") for letter in LETTERS},
            "answer": answer,
            "ground_truth": answer,
            "code": code,
            "code_returncode": verifier_returncode,
            "code_timed_out": verifier_timed_out,
            "code_stdout": verifier_stdout,
            "code_stderr": verifier_stderr,
            "well_defined": well_defined,
            "ill_defined_reasons": well_defined_reasons,
            "decision": decision,
            "labels": labels,
        }
        if metadata:
            record.update(metadata)

        append_jsonl(self.questions_path, record)
        if existing_index is not None:
            existing_index[qid] = record
        return qid, True

    def record_attempt(
        self,
        *,
        question_id: str,
        eval_mode: str,
        model: str,
        answer: Optional[str],
        ground_truth: Optional[str],
        correct: Optional[bool],
        attempt: int,
        usage: Optional[dict[str, Any]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        record: dict[str, Any] = {
            "type": "attempt",
            "timestamp": _now_iso(),
            "run_id": self.run_id,
            "artifacts_dir": str(self.artifacts_dir),
            "question_id": question_id,
            "attempt": attempt,
            "eval_mode": eval_mode,
            "model": model,
            "answer": answer,
            "ground_truth": ground_truth,
            "correct": correct,
        }
        if usage:
            record["usage"] = usage
        if metadata:
            record.update(metadata)
        append_jsonl(self.attempts_path, record)

    def record_summary(self, row: dict[str, Any]) -> None:
        append_jsonl(self.summary_path, row)

    def record_question_candidate(self, row: dict[str, Any], *, labels: list[str]) -> None:
        record = {
            "type": "question_candidate",
            "created_at": _now_iso(),
            "run_id": self.run_id,
            "artifacts_dir": str(self.artifacts_dir),
            "labels": labels,
            **row,
        }
        append_jsonl(self.questions_path, record)

    def write_final_summary(self, payload: dict[str, Any]) -> None:
        write_json(self.final_summary_path, payload)
