"""Local agent CLI registry for Backtrader MCQ mining.

This vendors the small subset of the shared runner framework that the miner
needs: answer heuristics, token-usage extraction, agent registration, and the
non-interactive CLI wrappers (claude, cursor, codex, copilot).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any, Callable


ANSWER_HINT_RE = [
    re.compile(r"\banswer\s*(?:is|:)?\s*([A-D])\b", re.IGNORECASE),
    re.compile(r"\boption\s*([A-D])\b", re.IGNORECASE),
    re.compile(r"\bchoice\s*([A-D])\b", re.IGNORECASE),
    re.compile(r"[\(\[]\s*([A-D])\s*[\)\]]"),
]


def _as_int(val: Any) -> int | None:
    return int(val) if isinstance(val, (int, float)) else None


def _as_float(val: Any) -> float | None:
    return float(val) if isinstance(val, (int, float)) else None


def _adapt_claude(payload: dict) -> dict | None:
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        return None
    if not any(
        k in usage
        for k in (
            "input_tokens",
            "output_tokens",
            "cache_read_input_tokens",
            "cache_creation_input_tokens",
        )
    ):
        return None
    return {
        "input_tokens": _as_int(usage.get("input_tokens")),
        "output_tokens": _as_int(usage.get("output_tokens")),
        "cache_read_tokens": _as_int(usage.get("cache_read_input_tokens")),
        "cache_creation_tokens": _as_int(usage.get("cache_creation_input_tokens")),
        "total_cost_usd": _as_float(payload.get("total_cost_usd")),
        "num_turns": _as_int(payload.get("num_turns")),
        "model": payload.get("model"),
    }


def _adapt_cursor(payload: dict) -> dict | None:
    if payload.get("type") != "result":
        return None
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        return None

    def _pick(snake: str, camel: str):
        return usage.get(snake, usage.get(camel))

    in_t = _pick("input_tokens", "inputTokens")
    out_t = _pick("output_tokens", "outputTokens")
    if in_t is None and out_t is None:
        return None
    return {
        "input_tokens": _as_int(in_t),
        "output_tokens": _as_int(out_t),
        "cache_read_tokens": _as_int(
            _pick("cache_read_input_tokens", "cacheReadInputTokens")
        ),
        "cache_creation_tokens": _as_int(
            _pick("cache_write_input_tokens", "cacheWriteInputTokens")
        ),
        "total_cost_usd": None,
        "num_turns": None,
        "model": payload.get("model"),
    }


def _adapt_codex(payload: dict) -> dict | None:
    if payload.get("type") == "event_msg":
        inner = payload.get("payload")
        if isinstance(inner, dict) and inner.get("type") == "token_count":
            info = inner.get("info") or {}
            total = info.get("total_token_usage") if isinstance(info, dict) else None
            if isinstance(total, dict) and (
                "input_tokens" in total or "output_tokens" in total
            ):
                return {
                    "input_tokens": _as_int(total.get("input_tokens")),
                    "output_tokens": _as_int(total.get("output_tokens")),
                    "cache_read_tokens": _as_int(total.get("cached_input_tokens")),
                    "cache_creation_tokens": None,
                    "total_cost_usd": None,
                    "num_turns": None,
                    "model": None,
                }

    if payload.get("type") in {"turn.completed", "item.completed"}:
        usage = payload.get("usage") or payload.get("token_count")
        if isinstance(usage, dict) and (
            "input_tokens" in usage or "output_tokens" in usage
        ):
            return {
                "input_tokens": _as_int(usage.get("input_tokens")),
                "output_tokens": _as_int(usage.get("output_tokens")),
                "cache_read_tokens": _as_int(usage.get("cached_input_tokens")),
                "cache_creation_tokens": None,
                "total_cost_usd": None,
                "num_turns": None,
                "model": payload.get("model"),
            }
    return None


def _adapt_openai(payload: dict) -> dict | None:
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        return None
    if not any(k in usage for k in ("prompt_tokens", "completion_tokens")):
        return None
    prompt_details = usage.get("prompt_tokens_details")
    return {
        "input_tokens": _as_int(usage.get("prompt_tokens")),
        "output_tokens": _as_int(usage.get("completion_tokens")),
        "cache_read_tokens": _as_int(
            prompt_details.get("cached_tokens")
            if isinstance(prompt_details, dict)
            else None
        ),
        "cache_creation_tokens": None,
        "total_cost_usd": None,
        "num_turns": None,
        "model": payload.get("model"),
    }


_USAGE_ADAPTERS: list[Callable[[dict], dict | None]] = [
    _adapt_claude,
    _adapt_cursor,
    _adapt_codex,
    _adapt_openai,
]


def _iter_json_candidates(text: str):
    if not text:
        return
    stripped = text.strip()
    if not stripped:
        return

    try:
        yield json.loads(stripped)
    except json.JSONDecodeError:
        pass

    for line in stripped.splitlines():
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue

    last_open = stripped.rfind("{")
    if last_open >= 0:
        try:
            yield json.loads(stripped[last_open:])
        except json.JSONDecodeError:
            pass


def extract_usage(stdout: str, stderr: str = "") -> dict | None:
    """Return normalized token/cost usage when an agent emits recognizable JSON."""
    best: dict | None = None
    for stream in (stdout, stderr):
        for payload in _iter_json_candidates(stream):
            if not isinstance(payload, dict):
                continue
            for adapter in _USAGE_ADAPTERS:
                usage = adapter(payload)
                if usage:
                    best = usage
                    break
    return best


def aggregate_usage(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    per_q = [r.get("usage") for r in records if isinstance(r.get("usage"), dict)]
    if not per_q:
        return None

    totals: dict[str, float] = {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_read_tokens": 0,
        "cache_creation_tokens": 0,
        "total_cost_usd": 0.0,
        "num_turns": 0,
    }
    counts: dict[str, int] = {k: 0 for k in totals}
    models_seen: set[str] = set()

    for usage in per_q:
        model = usage.get("model")
        if isinstance(model, str):
            models_seen.add(model)
        for key in totals:
            value = usage.get(key)
            if isinstance(value, (int, float)):
                totals[key] += value
                counts[key] += 1

    avgs = {
        f"avg_{key}": (totals[key] / counts[key] if counts[key] else None)
        for key in totals
    }
    return {
        "n_with_usage": len(per_q),
        "models": sorted(models_seen),
        "totals": {
            key: (round(value, 6) if isinstance(value, float) else value)
            for key, value in totals.items()
        },
        "averages": {
            key: (round(value, 4) if isinstance(value, float) else value)
            for key, value in avgs.items()
        },
    }


def _sandbox_quote(path: str) -> str:
    return path.replace("\\", "\\\\").replace('"', '\\"')


def maybe_wrap_macos_sandbox_exec(cmd: list[str], *, cwd: str) -> list[str]:
    """Wrap an invocation in macOS sandbox-exec when sandboxing is enabled."""
    enabled = os.environ.get("MINING_OS_SANDBOX") == "1"
    if not enabled:
        return cmd

    sandbox_dir = os.environ.get("MINING_OS_SANDBOX_DIR") or cwd
    protect_dir = os.environ.get("MINING_OS_SANDBOX_PROTECT_DIR")

    sb = shutil.which("sandbox-exec") or "/usr/bin/sandbox-exec"
    if not sb or not os.path.exists(sb):
        return cmd

    sandbox_dir_q = _sandbox_quote(os.path.realpath(sandbox_dir))
    protect_block = ""
    if protect_dir:
        protect_dir_q = _sandbox_quote(os.path.realpath(protect_dir))
        protect_block = f"""
(deny file-read-data
  (require-all
    (subpath "{protect_dir_q}")
    (require-not (subpath "{sandbox_dir_q}"))))
(deny file-write*
  (require-all
    (subpath "{protect_dir_q}")
    (require-not (subpath "{sandbox_dir_q}"))))
"""

    profile = f"""(version 1)
(allow default)
{protect_block}"""
    return [sb, "-p", profile, *cmd]


RunAgentFn = Callable[
    [str, str, int, Path, list[str]],
    subprocess.CompletedProcess[str],
]


@dataclass
class AgentDef:
    name: str
    bin_default: str
    prompt_filename: str
    run_fn: RunAgentFn
    extra_flags: list[str] = field(default_factory=list)

    def label(self) -> str:
        return self.name.capitalize()


_AGENT_REGISTRY: dict[str, AgentDef] = {}


def register_agent(agent: AgentDef) -> AgentDef:
    _AGENT_REGISTRY[agent.name] = agent
    return agent


def get_agent(name: str) -> AgentDef | None:
    return _AGENT_REGISTRY.get(name)


def list_agents() -> list[str]:
    return sorted(_AGENT_REGISTRY.keys())


def run_cursor(
    agent_bin: str,
    prompt: str,
    timeout: int,
    cwd: Path,
    extra: list[str],
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
        *extra,
        prompt,
    ]
    cmd = maybe_wrap_macos_sandbox_exec(cmd, cwd=str(cwd))
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
        cwd=cwd,
        stdin=subprocess.DEVNULL,
    )


def run_claude(
    claude_bin: str,
    prompt: str,
    timeout: int,
    cwd: Path,
    extra: list[str],
) -> subprocess.CompletedProcess[str]:
    cmd = [
        claude_bin,
        "-p",
        prompt,
        "--output-format",
        "json",
        "--dangerously-skip-permissions",
        "--allowedTools",
        "Bash",
        *extra,
    ]
    cmd = maybe_wrap_macos_sandbox_exec(cmd, cwd=str(cwd))
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
        cwd=cwd,
        stdin=subprocess.DEVNULL,
    )


def run_codex(
    codex_bin: str,
    prompt: str,
    timeout: int,
    cwd: Path,
    extra: list[str],
) -> subprocess.CompletedProcess[str]:
    cmd = [
        codex_bin,
        "exec",
        "--json",
        "--skip-git-repo-check",
        *extra,
        prompt,
    ]
    cmd = maybe_wrap_macos_sandbox_exec(cmd, cwd=str(cwd))
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
        cwd=cwd,
        stdin=subprocess.DEVNULL,
    )


_detected_copilot_variant: int | None = None
_copilot_detect_lock = Lock()


def _try_copilot_cmd(
    cmd: list[str], timeout: int, cwd: Path,
) -> subprocess.CompletedProcess[str]:
    cmd = maybe_wrap_macos_sandbox_exec(cmd, cwd=str(cwd))
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
        cwd=cwd,
        stdin=subprocess.DEVNULL,
    )


def _build_copilot_variants(
    copilot_bin: str, prompt: str, extra: list[str],
) -> list[list[str]]:
    return [
        [copilot_bin, "--prompt", prompt, "--allow-all-tools", "--output-format", "json", *extra],
        [copilot_bin, "-p", prompt, "--allow-all-tools", "--output-format", "json", *extra],
        [copilot_bin, "--prompt", prompt, "--output-format", "json", *extra],
        [copilot_bin, "-p", prompt, "--output-format", "json", *extra],
        [copilot_bin, "--prompt", prompt, "--allow-all-tools", *extra],
        [copilot_bin, "-p", prompt, "--allow-all-tools", *extra],
        [copilot_bin, "--prompt", prompt, *extra],
        [copilot_bin, "-p", prompt, *extra],
    ]


def run_copilot(
    copilot_bin: str,
    prompt: str,
    timeout: int,
    cwd: Path,
    extra: list[str],
) -> subprocess.CompletedProcess[str]:
    global _detected_copilot_variant

    with _copilot_detect_lock:
        cached = _detected_copilot_variant

    variants = _build_copilot_variants(copilot_bin, prompt, extra)
    if cached is not None and 0 <= cached < len(variants):
        return _try_copilot_cmd(variants[cached], timeout, cwd)

    first_result: subprocess.CompletedProcess[str] | None = None
    attempted: list[str] = []
    for idx, cmd in enumerate(variants):
        attempted.append(" ".join(cmd[:3]))
        result = _try_copilot_cmd(cmd, timeout, cwd)
        if idx == 0:
            first_result = result
        if result.returncode == 0:
            with _copilot_detect_lock:
                _detected_copilot_variant = idx
            return result

    if first_result is not None:
        first_result.stderr = (
            (first_result.stderr or "")
            + "\n\nTried command variants: "
            + ", ".join(attempted)
        )
        return first_result

    return subprocess.CompletedProcess([], 1, "", "No command variant succeeded.")


register_agent(AgentDef(
    name="claude",
    bin_default="claude",
    prompt_filename="claude_prompt.txt",
    run_fn=run_claude,
))
register_agent(AgentDef(
    name="codex",
    bin_default="codex",
    prompt_filename="codex_prompt.txt",
    run_fn=run_codex,
))
register_agent(AgentDef(
    name="copilot",
    bin_default="copilot",
    prompt_filename="copilot_prompt.txt",
    run_fn=run_copilot,
))
register_agent(AgentDef(
    name="cursor",
    bin_default="agent",
    prompt_filename="cursor_prompt.txt",
    run_fn=run_cursor,
))
