"""Core helpers for Backtrader MCQ mining.

The miner asks an agent for a Backtrader verifier program, runs that program
under controlled seeds, and turns the verified values into a four-choice MCQ.
This module keeps that parsing and execution logic independent from any match
or scoring loop.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from agent_backends import AgentDef, extract_usage

ROOT = Path(__file__).resolve().parent
VENV_PY = ROOT / ".venv" / "bin" / "python"
PROMPTS_DIR = ROOT / "prompts"
DB_DIR = ROOT / "db"
DEFAULT_MODEL = "composer-2"
DEFAULT_CODE_TIMEOUT_S = 120
DEFAULT_AGENT_TIMEOUT_S = 600
LETTERS = ("A", "B", "C", "D")

_TAG_RE_CACHE: dict[str, re.Pattern[str]] = {}
_ANSWER_RE = re.compile(r"^\s*ANSWER:\s*([ABCD])\s*$", re.MULTILINE)
_OPTION_RE = re.compile(r"^([A-D])\.\s+(.+)$", re.MULTILINE)
_NUM_RE = re.compile(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?")


@dataclass
class AgentInvocation:
    stdout: str
    stderr: str
    returncode: int
    usage: Optional[dict[str, Any]]
    cmd_ok: bool = True


@dataclass
class QuestionSpec:
    question: str
    code: str
    rationale: str = ""


@dataclass
class VerificationRun:
    seed: int
    stdout: str
    stderr: str
    returncode: int
    timed_out: bool
    question: Optional[str]
    answer_value: Optional[str]
    answer_letter: Optional[str]

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out and bool(self.answer_value)


@dataclass
class MinedQuestion:
    question: str
    options: dict[str, str]
    answer: str
    raw_prompt: str
    code: str
    difficulty: str
    source: str
    metadata: dict[str, Any]


def extract_block(text: str, name: str) -> Optional[str]:
    if name not in _TAG_RE_CACHE:
        _TAG_RE_CACHE[name] = re.compile(
            rf"<<<{re.escape(name)}>>>\s*(.*?)\s*<<</{re.escape(name)}>>>",
            re.DOTALL,
        )
    match = _TAG_RE_CACHE[name].search(text or "")
    return match.group(1).strip() if match else None


def logical_text(stdout: str) -> str:
    """Peel off common agent CLI JSON wrappers around model text."""
    if not stdout:
        return ""
    stripped = stdout.strip()
    if not stripped:
        return stdout

    def from_obj(obj: Any) -> Optional[str]:
        if not isinstance(obj, dict):
            return None
        result = obj.get("result")
        if isinstance(result, str) and result.strip():
            return result
        for key in ("content", "message", "output", "text"):
            value = obj.get(key)
            if isinstance(value, str) and value.strip():
                return value
        return None

    try:
        obj = json.loads(stripped)
    except json.JSONDecodeError:
        obj = None
    text = from_obj(obj)
    if text is not None:
        return text

    best: Optional[str] = None
    for line in stripped.splitlines():
        line = line.strip()
        if not (line.startswith("{") and line.endswith("}")):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        text = from_obj(obj)
        if text and (best is None or len(text) > len(best)):
            best = text
    return best if best is not None else stdout


def parse_generator_spec(raw_text: str) -> QuestionSpec:
    text = logical_text(raw_text)
    question = extract_block(text, "QUESTION") or extract_block(text, "QUESTION_TEMPLATE")
    code = extract_block(text, "CODE")
    rationale = extract_block(text, "RATIONALE") or ""
    if not question:
        raise ValueError("missing <<<QUESTION>>> block")
    if not code:
        raise ValueError("missing <<<CODE>>> block")
    return QuestionSpec(question=question.strip(), code=code.strip() + "\n", rationale=rationale.strip())


def parse_answer_letter(stdout: str) -> Optional[str]:
    matches = _ANSWER_RE.findall(stdout or "")
    return matches[-1] if matches else None


def _parse_prefixed_value(stdout: str, prefixes: tuple[str, ...]) -> Optional[str]:
    for line in (stdout or "").splitlines():
        stripped = line.strip()
        for prefix in prefixes:
            marker = prefix + ":"
            if not stripped.upper().startswith(marker):
                continue
            value = stripped[len(marker):].strip()
            if prefix.endswith("_JSON"):
                try:
                    decoded = json.loads(value)
                except json.JSONDecodeError:
                    return value
                return str(decoded)
            return value
    return None


def parse_verifier_output(stdout: str) -> tuple[Optional[str], Optional[str], Optional[str]]:
    question = _parse_prefixed_value(stdout, ("QUESTION_JSON", "QUESTION"))
    answer_value = _parse_prefixed_value(
        stdout,
        ("ANSWER_VALUE_JSON", "ANSWER_VALUE", "ANSWER_TEXT_JSON", "ANSWER_TEXT", "VALUE"),
    )
    answer_letter = parse_answer_letter(stdout)
    if answer_value is None and answer_letter is not None:
        answer_value = answer_letter
    return question, answer_value, answer_letter


def run_verifier(
    code: str,
    *,
    seed: int,
    work_dir: Path,
    timeout: int = DEFAULT_CODE_TIMEOUT_S,
    python_path: str | Path | None = None,
    pythonpath: list[str | Path] | None = None,
    script_name: str = "verifier.py",
) -> VerificationRun:
    work_dir = work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    script_path = work_dir / script_name
    script_path.write_text(code, encoding="utf-8")

    py = Path(python_path) if python_path is not None else VENV_PY
    if not py.is_absolute():
        py = py.resolve()
    if not py.exists():
        return VerificationRun(
            seed=seed,
            stdout="",
            stderr=f"[mining] verifier python missing at {py}",
            returncode=127,
            timed_out=False,
            question=None,
            answer_value=None,
            answer_letter=None,
        )

    env = os.environ.copy()
    env["MCQ_SEED"] = str(seed)
    if pythonpath:
        extra = os.pathsep.join(str(Path(p)) for p in pythonpath)
        existing = env.get("PYTHONPATH")
        env["PYTHONPATH"] = extra if not existing else extra + os.pathsep + existing

    try:
        proc = subprocess.run(
            [str(py), str(script_path)],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(work_dir),
            env=env,
            check=False,
            stdin=subprocess.DEVNULL,
        )
        stdout = proc.stdout or ""
        stderr = proc.stderr or ""
        returncode = proc.returncode
        timed_out = False
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr_body = exc.stderr if isinstance(exc.stderr, str) else ""
        stderr = f"{stderr_body}\n[mining] verifier timed out after {timeout}s".strip() + "\n"
        returncode = 124
        timed_out = True

    question, answer_value, answer_letter = parse_verifier_output(stdout)
    return VerificationRun(
        seed=seed,
        stdout=stdout,
        stderr=stderr,
        returncode=returncode,
        timed_out=timed_out,
        question=question,
        answer_value=answer_value,
        answer_letter=answer_letter,
    )


def parse_question_options(raw_text: str) -> tuple[str, dict[str, str]]:
    matches = list(_OPTION_RE.finditer(raw_text or ""))
    by_letter: dict[str, re.Match[str]] = {}
    for match in matches:
        by_letter.setdefault(match.group(1), match)
    for letter in LETTERS:
        if letter not in by_letter:
            raise ValueError(f"option {letter} not found")
    first_option_start = by_letter["A"].start()
    question = raw_text[:first_option_start].rstrip()
    options = {letter: by_letter[letter].group(2).strip() for letter in LETTERS}
    return question, options


def render_mcq_prompt(question: str, options: dict[str, str]) -> str:
    option_lines = "\n".join(f"{letter}. {options[letter]}" for letter in LETTERS)
    return (
        f"{question.strip()}\n\n"
        f"{option_lines}\n\n"
        "Respond in the form <<< X >>> where X is A, B, C, or D."
    )


def _normalize_choice(value: str) -> str:
    return re.sub(r"\s+", " ", str(value)).strip()


def _numeric_value(value: str) -> Optional[float]:
    match = _NUM_RE.search(str(value).replace(",", ""))
    if not match:
        return None
    try:
        return float(match.group(0).replace(",", ""))
    except ValueError:
        return None


def _format_like(reference: str, value: float) -> str:
    ref = str(reference)
    prefix = "$" if "$" in ref else ""
    suffix = "%" if "%" in ref else ""
    if re.search(r"\.\d", ref):
        body = f"{value:.2f}"
    else:
        body = str(int(round(value)))
    return f"{prefix}{body}{suffix}"


def build_options_from_values(
    *,
    correct_value: str,
    distractor_values: list[str],
    shuffle_seed: int,
) -> tuple[dict[str, str], str, list[str]]:
    correct = _normalize_choice(correct_value)
    choices: list[str] = []
    seen: set[str] = set()

    def add(value: str) -> None:
        normalized = _normalize_choice(value)
        key = normalized.lower()
        if normalized and key not in seen:
            choices.append(normalized)
            seen.add(key)

    add(correct)
    for value in distractor_values:
        add(value)
        if len(choices) >= 4:
            break

    correct_num = _numeric_value(correct)
    if len(choices) < 4 and correct_num is not None:
        deltas = [0.1, -0.1, 0.2, -0.2, 0.35, -0.35, 1.0, -1.0]
        for delta in deltas:
            candidate = correct_num * (1.0 + delta)
            if abs(correct_num) < 1e-9:
                candidate = delta * 100
            add(_format_like(correct, candidate))
            if len(choices) >= 4:
                break

    if len(choices) < 4:
        raise ValueError("could not build four distinct answer choices")

    import random

    rng = random.Random(shuffle_seed)
    picked = choices[:4]
    rng.shuffle(picked)
    options = {letter: picked[i] for i, letter in enumerate(LETTERS)}
    answer = next(letter for letter, value in options.items() if value == correct)
    return options, answer, picked


def invoke_agent(
    agent: AgentDef,
    prompt: str,
    *,
    cwd: Path,
    timeout: int,
    model: Optional[str],
    user_extra: list[str],
) -> AgentInvocation:
    cwd.mkdir(parents=True, exist_ok=True)
    bin_path = shutil.which(agent.bin_default)
    if not bin_path:
        return AgentInvocation(
            stdout="",
            stderr=f"[mining] binary {agent.bin_default!r} not found on PATH",
            returncode=127,
            usage=None,
            cmd_ok=False,
        )

    extras = list(user_extra)
    if agent.name in ("cursor", "claude", "copilot", "codex") and model and "--model" not in extras:
        extras = ["--model", model, *extras]
    if agent.name == "cursor" and "--mode" not in extras:
        extras = ["--mode", "ask", *extras]

    try:
        proc = agent.run_fn(bin_path, prompt, timeout, cwd, extras)
        stdout, stderr, rc = proc.stdout or "", proc.stderr or "", proc.returncode
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr_body = exc.stderr if isinstance(exc.stderr, str) else ""
        stderr = f"{stderr_body}\n[mining] agent timed out after {timeout}s".strip()
        rc = 124
    except Exception as exc:  # pragma: no cover
        stdout = ""
        stderr = f"[mining] agent invocation raised {type(exc).__name__}: {exc}"
        rc = 1

    usage = extract_usage(stdout, stderr)
    if usage is not None and model and not usage.get("model"):
        usage = dict(usage)
        usage["model"] = model
    return AgentInvocation(stdout=stdout, stderr=stderr, returncode=rc, usage=usage)
