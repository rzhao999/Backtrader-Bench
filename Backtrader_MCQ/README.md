---
pretty_name: Backtrader MCQ Benchmark
language:
  - en
license: other
task_categories:
  - question-answering
size_categories:
  - n<1K
tags:
  - multiple-choice
  - finance
  - trading
  - backtesting
  - backtrader
---

# Backtrader MCQ Benchmark

This dataset contains multiple-choice questions for evaluating whether a model can reason about trading-strategy behavior using the Backtrader backtesting framework. Each question provides a complete backtest configuration and asks for a single answer choice in the format `<<< X >>>`, where `X` is one of `A`, `B`, `C`, or `D`.

The primary evaluation file used in the paper is:

```text
backtrader_mcq_balanced_30_all_strategies.jsonl
```

The larger supporting pool is:

```text
backtrader_mcq_base_pool_all_strategies.jsonl
```

## Dataset Files

| File | Examples | Role |
| --- | ---: | --- |
| `backtrader_mcq_balanced_30_all_strategies.jsonl` | 30 | Balanced evaluation subset used in the paper |
| `backtrader_mcq_base_pool_all_strategies.jsonl` | 160 | Larger base pool from which evaluation questions can be inspected or sampled |

The balanced evaluation subset contains:

- 10 easy, 10 medium, and 10 hard questions.
- 6 questions for each strategy.
- Questions over AAPL from `2020-01-01` to `2024-01-01`.

The base pool contains 160 questions across the same benchmark setting, with a broader distribution of question types and difficulty levels.

## Task

Given a question prompt, a model must select the correct multiple-choice answer and respond exactly as:

```text
<<< A >>>
```

or the corresponding option letter. The ground-truth answer is stored in the `answer` field.

Many questions cannot be answered from surface text alone. They require computing or verifying values from the specified Backtrader run, including portfolio value, broker cash, positions, orders, trades, drawdowns, moving averages, and strategy comparisons.

## Strategies

The dataset covers five Backtrader strategies:

- `sma_crossover`
- `rolling_window_mean`
- `exponential_weighted_moving_average`
- `rsi_strategy`
- `macd_crossover`

Each row includes the exact `strategy_name`, `strategy_params`, and full `config` needed to reproduce the corresponding backtest.

## Schema

Each JSONL row has the following fields:

| Field | Description |
| --- | --- |
| `question` | Full multiple-choice prompt, including the backtest configuration and answer choices |
| `answer` | Correct option letter, one of `A`, `B`, `C`, or `D` |
| `benchmark` | Benchmark identifier; currently `backtrader` |
| `author` | Internal generation label |
| `type` | Question category, such as `portfolio_value`, `max_drawdown`, `first_buy`, or `roi_percentage` |
| `difficulty` | Difficulty label: `easy`, `medium`, or `hard` |
| `strategy_name` | Backtrader strategy used for the question |
| `strategy_params` | Strategy parameters used for the question |
| `config` | Full backtest configuration, including symbol, date range, cash, commission, stake, and strategy settings |

Example record shape:

```json
{
  "question": "...",
  "answer": "C",
  "benchmark": "backtrader",
  "author": "MCQ_test",
  "type": "first_close",
  "difficulty": "easy",
  "strategy_name": "sma_crossover",
  "strategy_params": {
    "pfast": 10,
    "pslow": 30,
    "target_percent": 1.0
  },
  "config": {
    "symbol": "AAPL",
    "start_date": "2020-01-01",
    "end_date": "2024-01-01",
    "initial_cash": 50000.0,
    "commission_rate": 0.001,
    "stake": 1000,
    "strategy_name": "sma_crossover",
    "strategy_params": {
      "pfast": 10,
      "pslow": 30,
      "target_percent": 1.0
    }
  }
}
```

## Data Generation

Questions are generated from deterministic Backtrader runs over historical AAPL market data loaded with `yfinance` using `auto_adjust=False`. Each run uses the row-level configuration embedded in the JSONL record.

The standard configuration used by the released questions is:

- Symbol: `AAPL`
- Date range: `2020-01-01` to `2024-01-01`
- Initial cash: `50000.0`
- Commission rate: `0.001`
- Stake cap: `1000`

Backtrader is used to compute the daily records and trading events needed for each question. These include OHLCV-derived values, moving averages, broker cash, portfolio value, position size, order executions, commissions, closed-trade PnL, and derived metrics such as ROI, drawdown, profit factor, and Calmar-style ratio.

## Evaluation

For the paper setting, evaluate on:

```text
backtrader_mcq_balanced_30_all_strategies.jsonl
```

Recommended scoring:

1. Prompt the model with the `question` field.
2. Parse the predicted answer choice from the response.
3. Compare the predicted choice against the `answer` field.
4. Report exact-match multiple-choice accuracy.

The answer should be treated as the option letter, not the full answer text.

## Intended Use

This dataset is intended for evaluating tool-use, programmatic reasoning, and quantitative reasoning in the context of trading backtests. It is especially suited for testing whether models can correctly use Backtrader or equivalent computations to answer finance-oriented multiple-choice questions.

This dataset is not intended to provide financial advice, trading recommendations, or claims about future market behavior.

## Limitations

- The released questions use one ticker, AAPL, over one historical date range.
- Correct answers depend on the exact Backtrader implementation, strategy definitions, commission model, position sizing logic, and market data version.
- Historical market data can change across vendors or download dates due to adjustments, corrections, or API behavior.
- The dataset is small and should be interpreted as a focused benchmark rather than a broad measure of financial expertise.
- Multiple-choice formatting may allow some models to exploit answer-choice artifacts; tool-based verification is recommended.

## Review Notes

This dataset is prepared for anonymous conference review. The dataset files,
metadata, and source code should remain accessible to reviewers during the
review period and should avoid revealing author identity.

If the conference requires additional Responsible AI or dataset-card fields,
complete those fields in the submission system or dataset hosting page before
final submission.

## License

The license is marked as `other` during anonymous review. Final public release terms should be confirmed before camera-ready release.
