#!/usr/bin/env python3
"""End-to-end pipeline: generate MCQ questions then verify every answer.

Usage:
    # ---- Easy mode: edit the config dict below, then just run ----
    python backtrader_MCQ_pipeline.py

    # ---- Interactive mode: get prompted for every value ----
    python backtrader_MCQ_pipeline.py -i
    python backtrader_MCQ_pipeline.py --interactive

    # ---- Or pass CLI flags (override anything) ----
    python backtrader_MCQ_pipeline.py --symbol GOOG --num-questions 30

    # ---- Remove the output file after a successful verification ----
    python backtrader_MCQ_pipeline.py --clean

    # ---- Print available MCQ templates and exit ----
    python backtrader_MCQ_pipeline.py --list-templates
    python backtrader_MCQ_pipeline.py --list-templates --difficulty hard

    # ---- Generate/verify one question per selected template as JSON ----
    python backtrader_MCQ_pipeline.py --all-templates --output-format json -o base_pool.json

Agent / token tracking:
    This pipeline intentionally stays focused on *question generation* and
    *ground-truth verification*. It does NOT run any agent on the questions.
    To mine and evaluate new questions with Cursor agents, run
    ``python mine_backtrader_questions.py`` from the repository root.
"""

import json
import sys
import time
from pathlib import Path

from backtrader_MCQ_builder import main as builder_main
from backtrader_MCQ_checker import check_mcq_file
from backtrader_MCQ import print_available_mcq_templates
from strategies import DEFAULT_STRATEGY_NAME, REGISTERED_STRATEGY_NAMES, get_default_params

# =====================================================================
# CUSTOMIZE HERE — change any value and re-run the script
# =====================================================================
PIPELINE_CONFIG = {
    "symbol":          "AAPL",
    # 2020-01-01..2024-01-01 spans a clear bull run, the 2022 drawdown,
    # and the 2023 recovery -- gives both the question generator and any
    # downstream agent enough variety (multiple SMA crossovers, plenty of
    # trades, distinct max-drawdown windows) without depending on
    # whatever date range a caller might forget to override. Standalone
    # runs and the mining workflow share these defaults.
    "start_date":      "2020-01-01",
    "end_date":        "2024-01-01",
    "initial_cash":    50000,
    "commission_rate": 0.001,
    "stake":           1000,
    "strategy":        DEFAULT_STRATEGY_NAME,
    "strategy_params": json.dumps(get_default_params(DEFAULT_STRATEGY_NAME)),
    "num_questions":   10,
    "seed":            42,
    "output":          str(Path(__file__).resolve().parent / "output" / "test_10.json"),
    "benchmark":       "backtrader",
    "author":          "MCQ_test",
    # Difficulty pool(s) to draw from. One of:
    #   "easy" / "medium" / "hard"   (single level)
    #   "all"                        – balanced across easy/medium/hard
    #   "easy,medium"                – comma-separated combo (distributed evenly)
    # Default starts at "easy" so a no-flag run is small and quick to
    # eyeball; bump to "all" / "hard" once you trust your config.
    "difficulty":      "easy",
}
# =====================================================================


_STRATEGY_CHOICES = list(REGISTERED_STRATEGY_NAMES)
_DEFAULT_STRATEGY_PARAMS_JSON = json.dumps(
    get_default_params(DEFAULT_STRATEGY_NAME),
    sort_keys=True,
)
_DIFFICULTY_CHOICES = [
    "all", "easy", "medium", "hard",
    "easy,medium", "easy,hard", "medium,hard",
]


def _strategy_params_json(strategy_name: str) -> str:
    return json.dumps(get_default_params(strategy_name), sort_keys=True)


def _value_after_flag(argv, flags):
    for i, arg in enumerate(argv):
        if arg in flags and i + 1 < len(argv):
            return argv[i + 1]
        for flag in flags:
            prefix = f"{flag}="
            if arg.startswith(prefix):
                return arg[len(prefix):]
    return None


def _normalize_strategy_params(config: dict, argv=None) -> dict:
    """Keep default strategy params synced with the selected strategy."""
    normalized = dict(config)
    argv = list(argv or [])
    cli_strategy = _value_after_flag(argv, ("--strategy",))
    strategy_name = cli_strategy or normalized.get("strategy", DEFAULT_STRATEGY_NAME)
    has_cli_params = _value_after_flag(argv, ("--strategy-params",)) is not None

    current_params = normalized.get("strategy_params")
    if not has_cli_params and current_params in (None, "", _DEFAULT_STRATEGY_PARAMS_JSON):
        normalized["strategy_params"] = _strategy_params_json(strategy_name)
    return normalized


def _prompt(label, default, cast=str, choices=None):
    """Prompt for a single value; blank input returns the default."""
    hint = f" [{'/'.join(choices)}]" if choices else ""
    raw = input(f"  {label}{hint} ({default}): ").strip()
    if raw == "":
        return default
    if choices and raw not in choices:
        print(f"    ! '{raw}' not in {choices}; using default {default}")
        return default
    try:
        return cast(raw)
    except (ValueError, TypeError):
        print(f"    ! Could not parse '{raw}' as {cast.__name__}; using default {default}")
        return default


def _prompt_interactive(defaults: dict) -> dict:
    """Walk the user through every PIPELINE_CONFIG field interactively.

    Press Enter on any prompt to keep the default shown in parentheses.
    """
    print("\nInteractive pipeline config")
    print("Press Enter to keep the default shown in (parentheses).\n")

    out = dict(defaults)
    out["symbol"]          = _prompt("Ticker symbol",            defaults["symbol"])
    out["start_date"]      = _prompt("Start date (YYYY-MM-DD)",  defaults["start_date"])
    out["end_date"]        = _prompt("End date   (YYYY-MM-DD)",  defaults["end_date"])
    out["initial_cash"]    = _prompt("Initial cash",             defaults["initial_cash"],    cast=float)
    out["commission_rate"] = _prompt("Commission rate",          defaults["commission_rate"], cast=float)
    out["stake"]           = _prompt("Stake (shares per trade)", defaults["stake"],           cast=int)
    out["strategy"]        = _prompt("Strategy",                 defaults["strategy"],        choices=_STRATEGY_CHOICES)
    out["strategy_params"] = _prompt("Strategy params (JSON)",   _strategy_params_json(out["strategy"]))
    out["num_questions"]   = _prompt("Number of questions",      defaults["num_questions"],   cast=int)
    out["seed"]            = _prompt("Random seed",              defaults["seed"],            cast=int)
    out["difficulty"]      = _prompt("Difficulty",               defaults["difficulty"],      choices=_DIFFICULTY_CHOICES)
    out["output"]          = _prompt("Output JSON path",         defaults["output"])

    print("\nFinal config:")
    for k, v in out.items():
        print(f"  {k:<16} = {v}")
    confirm = input("\nProceed? [Y/n]: ").strip().lower()
    if confirm in ("n", "no"):
        print("Aborted.")
        sys.exit(0)
    print()
    return out


def _config_to_argv(config: dict) -> list:
    """Convert the config dict into CLI argv that the builder understands."""
    argv = []
    key_to_flag = {
        "symbol":          "--symbol",
        "start_date":      "--start-date",
        "end_date":        "--end-date",
        "initial_cash":    "--initial-cash",
        "commission_rate": "--commission-rate",
        "stake":           "--stake",
        "strategy":        "--strategy",
        "strategy_params": "--strategy-params",
        "num_questions":   "--num-questions",
        "seed":            "--seed",
        "output":          "-o",
        "benchmark":       "--benchmark",
        "author":          "--author",
        "difficulty":      "--difficulty",
    }
    for key, flag in key_to_flag.items():
        val = config.get(key)
        if val is not None:
            argv += [flag, str(val)]
    return argv


def run_pipeline(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    if "--list-templates" in argv:
        difficulty = _value_after_flag(argv, ("--difficulty",)) or "all"
        print_available_mcq_templates(difficulty)
        return 0

    clean = "--clean" in argv
    if clean:
        argv = [a for a in argv if a != "--clean"]

    interactive = any(a in ("-i", "--interactive") for a in argv)
    if interactive:
        argv = [a for a in argv if a not in ("-i", "--interactive")]
        effective_config = _prompt_interactive(_normalize_strategy_params(PIPELINE_CONFIG, argv))
    else:
        effective_config = _normalize_strategy_params(PIPELINE_CONFIG, argv)

    base_argv = _config_to_argv(effective_config)

    cli_keys_present = set()
    for a in argv:
        if a.startswith("--") or a == "-o":
            cli_keys_present.add(a)

    merged = []
    skip_next = False
    for i, a in enumerate(base_argv):
        if skip_next:
            skip_next = False
            continue
        if a in cli_keys_present:
            skip_next = True
            continue
        merged.append(a)
        if i + 1 < len(base_argv) and not base_argv[i + 1].startswith("-"):
            merged.append(base_argv[i + 1])
            skip_next = True

    final_argv = merged + list(argv)

    output_path = PIPELINE_CONFIG["output"]
    for i, a in enumerate(final_argv):
        if a in ("-o", "--output") and i + 1 < len(final_argv):
            output_path = final_argv[i + 1]
            break

    print("=" * 60)
    print("STEP 1: Generate questions")
    print("=" * 60)
    t0 = time.time()
    builder_main(final_argv)
    gen_time = time.time() - t0
    print(f"Generation completed in {gen_time:.1f}s\n")

    print("=" * 60)
    print("STEP 2: Verify ground truth")
    print("=" * 60)
    t0 = time.time()
    results = check_mcq_file(output_path)
    check_time = time.time() - t0
    print(f"Verification completed in {check_time:.1f}s\n")

    passed = sum(1 for r in results if r["match"])
    failed = len(results) - passed

    print("=" * 60)
    print(f"PIPELINE RESULT: {passed}/{len(results)} ground-truth questions passed")
    print("=" * 60)

    if failed > 0:
        print(f"\nWARNING: {failed} question(s) have incorrect answers!")
        for r in results:
            if not r["match"]:
                diff = r.get("difficulty", "?")
                print(f"  Q{r['index']:02d} [{diff}] ({r['type']}): "
                      f"stated={r['stated_answer']} computed={r['computed_answer']}")
        print(f"\nKeeping {output_path} for inspection.")
        return 1

    if clean:
        Path(output_path).unlink(missing_ok=True)
        print(f"Cleaned up {output_path}")
    else:
        print(f"Output saved at {output_path}")
        print("\nUse this JSON/JSONL file with your downstream agent runner")
        print("(for example, an AI_Agents run folder that calls run_*_mcq.py).")

    return 0


if __name__ == "__main__":
    sys.exit(run_pipeline())
