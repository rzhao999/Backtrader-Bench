#!/usr/bin/env python3
"""CLI tool to generate Backtrader MCQ benchmark datasets.

Accepts a customized config (symbol, dates, cash, etc.), an output path,
and a trading strategy name.  Produces a JSONL file where each line includes
the rendered question, answer letter, question type, difficulty, strategy
metadata, and the full backtest config:

    {
      "question": "...",
      "answer": "A",
      "benchmark": "backtrader",
      "author": "MCQ_test",
      "type": "max_drawdown_window",
      "difficulty": "hard",
      "strategy_name": "sma_crossover",
      "strategy_params": {"pfast": 10, "pslow": 30, "target_percent": 1.0},
      "config": {...}
    }

Usage examples
--------------
# Minimal – uses defaults from backtrader_info.CONFIG
python backtrader_MCQ_builder.py -o my_questions.json

# Fully customized
python backtrader_MCQ_builder.py \\
    --symbol AAPL \\
    --start-date 2020-01-01 \\
    --end-date 2024-01-01 \\
    --initial-cash 50000 \\
    --commission-rate 0.001 \\
    --stake 1000 \\
    --strategy sma_crossover \\
    --strategy-params '{"pfast": 10, "pslow": 30}' \\
    --num-questions 30 \\
    --seed 42 \\
    --benchmark backtrader \\
    --author MCQ_test \\
    -o output/AAPL_mcq_30.json

# Load config from a JSON file (CLI flags override file values)
python backtrader_MCQ_builder.py \\
    --config my_config.json \\
    --symbol GOOG \\
    -o output/GOOG_mcq.json

# Print every available template without generating questions
python backtrader_MCQ_builder.py --list-templates
python backtrader_MCQ_builder.py --list-templates --difficulty hard

# Generate one verified base-pool candidate per available template as JSON
python backtrader_MCQ_builder.py --all-templates --output-format json -o base_pool.json
"""

import argparse
import json
from pathlib import Path

from backtrader_MCQ import print_available_mcq_templates, save_mcq_dataset_json
from strategies import DEFAULT_STRATEGY_NAME, REGISTERED_STRATEGY_NAMES


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Generate Backtrader MCQ benchmark datasets in JSONL format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "-o", "--output",
        help="Path to the output JSONL file (e.g. output/AAPL_mcq_30.json).",
    )

    cfg = parser.add_argument_group("config overrides (individual flags)")
    cfg.add_argument("--symbol", help="Ticker symbol (e.g. AAPL, MSFT).")
    cfg.add_argument("--start-date", help="Backtest start date (YYYY-MM-DD).")
    cfg.add_argument("--end-date", help="Backtest end date (YYYY-MM-DD).")
    cfg.add_argument("--initial-cash", type=float, help="Starting cash for the broker.")
    cfg.add_argument("--commission-rate", type=float, help="Broker commission rate (e.g. 0.001).")
    cfg.add_argument("--stake", type=int, help="Fixed order size (shares per trade).")
    cfg.add_argument(
        "--strategy-params",
        help="Strategy parameters as a JSON string, e.g. '{\"pfast\": 10, \"pslow\": 30}'.",
    )

    cfg.add_argument(
        "--config",
        help="Path to a JSON config file. Keys match backtrader_info.CONFIG. "
             "CLI flags take precedence over values in this file.",
    )

    gen = parser.add_argument_group("generation settings")
    gen.add_argument(
        "--strategy",
        default=DEFAULT_STRATEGY_NAME,
        choices=REGISTERED_STRATEGY_NAMES,
        help=(
            f"Name of the trading strategy (default: {DEFAULT_STRATEGY_NAME}). "
            f"Choices: {', '.join(REGISTERED_STRATEGY_NAMES)}."
        ),
    )
    gen.add_argument(
        "--num-questions",
        type=int,
        default=10,
        help="Number of MCQ questions to generate (default: 10).",
    )
    gen.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility.",
    )
    gen.add_argument(
        "--difficulty",
        default="all",
        help=(
            "Difficulty pool(s) to draw questions from. "
            "Options: 'all' (default, balanced across easy/medium/hard), "
            "'easy', 'medium', 'hard', or a comma-separated combo "
            "like 'easy,medium'. Questions are distributed roughly "
            "equally across the requested levels."
        ),
    )
    gen.add_argument(
        "--list-templates",
        action="store_true",
        help="Print all available MCQ templates for the selected difficulty and exit.",
    )
    gen.add_argument(
        "--all-templates",
        action="store_true",
        help=(
            "Generate one question per available template for the selected "
            "difficulty pool(s), instead of randomly sampling num_questions."
        ),
    )
    gen.add_argument(
        "--output-format",
        choices=("jsonl", "json"),
        default="jsonl",
        help="Write records as JSONL (default) or as one JSON array.",
    )

    meta = parser.add_argument_group("output metadata")
    meta.add_argument("--benchmark", default="backtrader", help="Benchmark label (default: backtrader).")
    meta.add_argument("--author", default="MCQ_test", help="Author label (default: MCQ_test).")

    return parser.parse_args(argv)


def _build_config_overrides(args):
    """Build a config dict containing ONLY the user-supplied overrides.

    This is passed to build_mcq_dataset which merges it on top of
    backtrader_info.CONFIG, so we must NOT start from CONFIG here
    (that would apply defaults twice).
    """
    overrides = {}

    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            overrides.update(json.load(f))

    flag_map = {
        "symbol": args.symbol,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "initial_cash": args.initial_cash,
        "commission_rate": args.commission_rate,
        "stake": args.stake,
    }
    for key, value in flag_map.items():
        if value is not None:
            overrides[key] = value

    if args.strategy_params is not None:
        overrides["strategy_params"] = json.loads(args.strategy_params)

    return overrides


def main(argv=None):
    args = _parse_args(argv)

    if args.list_templates:
        print_available_mcq_templates(args.difficulty)
        return []

    if not args.output:
        raise SystemExit("error: -o/--output is required unless --list-templates is used")

    overrides = _build_config_overrides(args)

    symbol = overrides.get("symbol", "CONFIG default")
    start = overrides.get("start_date", "CONFIG default")
    end = overrides.get("end_date", "CONFIG default")
    print(f"Config: symbol={symbol}, dates={start}..{end}, "
          f"strategy={args.strategy}, num_questions={args.num_questions}, "
          f"difficulty={args.difficulty}, all_templates={args.all_templates}, "
          f"output_format={args.output_format}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records = save_mcq_dataset_json(
        output_path=str(output_path),
        config=overrides or None,
        strategy_name=args.strategy,
        num_questions=args.num_questions,
        seed=args.seed,
        benchmark=args.benchmark,
        author=args.author,
        difficulty=args.difficulty,
        all_templates=args.all_templates,
        output_format=args.output_format,
    )

    print(f"Saved {len(records)} questions to {output_path}")
    return records


if __name__ == "__main__":
    main()
