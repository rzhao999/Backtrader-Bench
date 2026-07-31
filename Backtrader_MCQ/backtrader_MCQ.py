import json
import random
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import pandas as pd

from backtrader_info import CONFIG, run_backtrader_info
from strategies import DEFAULT_STRATEGY_NAME, get_default_params

_LETTERS = ["A", "B", "C", "D"]


# ---------------------------------------------------------------------------
# Difficulty taxonomy
# ---------------------------------------------------------------------------
# Each question type is tagged as easy, medium, or hard.
#   easy   – direct single-row / single-scalar lookup from the backtest df
#   medium – conditional row selection or simple aggregation over the df
#   hard   – multi-step derived metric, or requires running a second backtest
DIFFICULTY_BY_TYPE: Dict[str, str] = {
    "first_close":             "easy",
    "volume":                  "easy",
    "broker_cash":             "easy",
    "portfolio_value":         "easy",
    "sma":                     "easy",
    "max_shares":              "easy",
    "position_on_date":        "easy",
    "crossovers":              "easy",
    "buy_sell_signals":        "easy",

    "first_buy":               "medium",
    "trade_outcomes":          "medium",
    "net_pnl":                 "medium",
    "commission":              "medium",
    "total_shares_traded":     "medium",
    "days_in_position":        "medium",
    "peak_portfolio_date":     "medium",
    "best_trade_day":          "medium",
    "strategy_action_on_date": "medium",
    "nth_trade_pnl":           "medium",
    "cash_after_nth_order":    "medium",

    "max_drawdown":            "hard",
    "roi_percentage":          "hard",
    "win_rate":                "hard",
    "profit_factor":           "hard",
    "annualized_return":       "hard",
    "avg_holding_period":      "hard",
    "golden_cross_count":      "hard",
    "max_drawdown_window":     "hard",
    "drawdown_recovery_days":  "hard",
    "calmar_ratio":            "hard",
    "exposure_adjusted_return": "hard",
    "sma_comparison":          "hard",
    "signal_diff":             "hard",
}

DIFFICULTY_LEVELS: Tuple[str, ...] = ("easy", "medium", "hard")

MCQ_TEMPLATE_DESCRIPTIONS: Dict[str, str] = {
    "first_close": "Look up the first close price in the backtest window.",
    "volume": "Look up market volume for a sampled trading date.",
    "broker_cash": "Look up broker cash for a sampled trading date.",
    "portfolio_value": "Look up portfolio value for a sampled trading date.",
    "sma": "Look up the configured fast or slow SMA value for a sampled date.",
    "max_shares": "Compute the maximum initial shares affordable after commission.",
    "position_on_date": "Run the strategy and report position size on a sampled date.",
    "crossovers": "Count strategy crossover/signal events over the full backtest.",
    "buy_sell_signals": "Count buy and sell signals over the full backtest.",
    "first_buy": "Identify the first executed buy with shares, close price, and date.",
    "trade_outcomes": "Count profitable and losing closed trades.",
    "net_pnl": "Compute final net profit/loss from initial cash to final value.",
    "commission": "Combine total commission paid with executed order count.",
    "total_shares_traded": "Aggregate all bought and sold shares across the backtest.",
    "days_in_position": "Count trading days in and out of a position.",
    "peak_portfolio_date": "Find the date and value of the portfolio's all-time high.",
    "best_trade_day": "Find the most profitable realized closed-trade day.",
    "strategy_action_on_date": "Run the strategy and classify buy, sell, or hold on a date.",
    "nth_trade_pnl": "Locate the nth closed trade and report realized net P/L.",
    "cash_after_nth_order": "Find broker cash after the nth order execution day.",
    "max_drawdown": "Compute maximum peak-to-trough portfolio drawdown in dollars.",
    "roi_percentage": "Compute overall return on investment as net P/L divided by initial cash.",
    "win_rate": "Compute profitable closed trades as a percentage of all closed trades.",
    "profit_factor": "Compute gross profit divided by absolute gross loss.",
    "annualized_return": "Compound final value over the observed trading-day count.",
    "avg_holding_period": "Divide total days in position by number of closed trades.",
    "golden_cross_count": "Count SMA fast-over-slow crosses within a sampled subperiod.",
    "max_drawdown_window": "Identify the peak date, trough date, drawdown dollars, and drawdown percent.",
    "drawdown_recovery_days": "Find whether and when the max-drawdown peak value was recovered.",
    "calmar_ratio": "Compute annualized return divided by maximum drawdown percent.",
    "exposure_adjusted_return": "Compute ROI divided by market exposure percentage.",
    "sma_comparison": "Run a second parameter set and compare net P/L.",
    "signal_diff": "Run a second parameter set and compare buy-signal counts.",
}


def list_available_mcq_templates(
    difficulty: Union[None, str, Iterable[str]] = None,
) -> List[Dict[str, str]]:
    """Return all available MCQ templates grouped by difficulty."""
    levels = _normalize_difficulty(difficulty)
    templates = []
    for level in levels:
        for qtype, qlevel in DIFFICULTY_BY_TYPE.items():
            if qlevel == level:
                templates.append({
                    "type": qtype,
                    "difficulty": qlevel,
                    "description": MCQ_TEMPLATE_DESCRIPTIONS.get(qtype, ""),
                })
    return templates


def format_available_mcq_templates(
    difficulty: Union[None, str, Iterable[str]] = None,
) -> str:
    """Format the available MCQ templates for CLI output."""
    templates = list_available_mcq_templates(difficulty)
    lines = ["Available MCQ templates:"]
    current_level = None
    for template in templates:
        level = template["difficulty"]
        if level != current_level:
            current_level = level
            lines.append(f"\n{level.upper()}")
        lines.append(f"- {template['type']}: {template['description']}")
    return "\n".join(lines)


def print_available_mcq_templates(
    difficulty: Union[None, str, Iterable[str]] = None,
) -> None:
    print(format_available_mcq_templates(difficulty))


def _types_for_levels(levels: Iterable[str]) -> Dict[str, List[str]]:
    """Return {level: [qtypes...]} for the given levels, preserving insertion order."""
    pools: Dict[str, List[str]] = {l: [] for l in levels}
    for qtype, level in DIFFICULTY_BY_TYPE.items():
        if level in pools:
            pools[level].append(qtype)
    return pools


def _normalize_difficulty(difficulty: Union[None, str, Iterable[str]]) -> List[str]:
    """Normalize a difficulty spec into a validated list of level names.

    Accepts:
        - None / "all" / "ALL"                → all three levels
        - "easy" / "medium" / "hard"          → single level
        - "easy,medium" (comma-separated)     → multiple levels
        - ["easy", "medium"] (iterable)       → multiple levels
    """
    if difficulty is None:
        return list(DIFFICULTY_LEVELS)

    if isinstance(difficulty, str):
        raw = difficulty.strip().lower()
        if raw in ("", "all"):
            return list(DIFFICULTY_LEVELS)
        levels = [p.strip() for p in raw.split(",") if p.strip()]
    else:
        levels = [str(l).strip().lower() for l in difficulty if str(l).strip()]

    if not levels:
        return list(DIFFICULTY_LEVELS)

    seen: List[str] = []
    for l in levels:
        if l == "all":
            return list(DIFFICULTY_LEVELS)
        if l not in DIFFICULTY_LEVELS:
            raise ValueError(
                f"Invalid difficulty level: '{l}'. "
                f"Expected one of {DIFFICULTY_LEVELS} or 'all'."
            )
        if l not in seen:
            seen.append(l)
    return seen


def _allocate_counts(total: int, levels: List[str]) -> Dict[str, int]:
    """Split `total` questions across `levels` as evenly as possible."""
    if total <= 0:
        raise ValueError("num_questions must be a positive integer.")
    n = len(levels)
    base, extra = divmod(total, n)
    counts = {l: base for l in levels}
    for i in range(extra):
        counts[levels[i]] += 1
    return counts


# ---------------------------------------------------------------------------
# Option / formatting helpers
# ---------------------------------------------------------------------------

def _format_num(value: float, as_int: bool = False) -> str:
    if as_int:
        return str(int(round(value)))
    return f"{float(value):.2f}"


def _numeric_options(
    correct: float,
    rng: random.Random,
    *,
    as_int: bool = False,
    non_negative: bool = False,
) -> List[str]:
    canonical = int(round(correct)) if as_int else round(float(correct), 2)
    options = {canonical}
    scales = [0.7, 0.8, 0.9, 1.1, 1.2, 1.3]

    for _ in range(50):
        if len(options) >= 4:
            break
        scale = rng.choice(scales)
        jitter = rng.uniform(-0.03, 0.03)
        candidate = float(correct) * (scale + jitter)
        if abs(float(correct)) < 1e-9:
            candidate = rng.uniform(1.0, 20.0)
        if non_negative:
            candidate = max(0.0, candidate)
        if as_int:
            options.add(int(round(candidate)))
        else:
            options.add(round(candidate, 2))

    # Fallback: for values near zero where scaling produces duplicates,
    # add fixed offsets to guarantee 4 distinct options.
    offset = 1
    while len(options) < 4:
        fallback = (canonical + offset) if not non_negative else abs(canonical + offset)
        options.add(fallback)
        offset += 1

    vals = list(options)
    rng.shuffle(vals)
    return [_format_num(v, as_int=as_int) for v in vals]


def _dollar_options(correct: float, rng: random.Random) -> Tuple[List[str], str]:
    """Return (4 dollar-formatted options, correct_str)."""
    raw = _numeric_options(correct, rng)
    return [f"${x}" for x in raw], f"${correct:.2f}"


def _choice_block(question: str, options: List[str], correct: str) -> Dict[str, Any]:
    if len(options) != 4:
        raise ValueError("Each MCQ must have exactly 4 options.")
    if correct not in options:
        raise ValueError(f"Correct answer '{correct}' not in options: {options}")

    letter_map = dict(zip(_LETTERS, options))
    answer = next(l for l, v in letter_map.items() if v == correct)
    return {"question": question, "option_map": letter_map, "answer": answer}


def _pair_count_options(
    correct_left: int,
    correct_right: int,
    left_label: str,
    right_label: str,
    rng: random.Random,
) -> List[str]:
    correct = f"{correct_left} {left_label} and {correct_right} {right_label}"
    options = [correct]
    for _ in range(50):
        if len(options) >= 4:
            break
        left = max(0, correct_left + rng.choice([-3, -2, -1, 1, 2, 3]))
        right = max(0, correct_right + rng.choice([-3, -2, -1, 1, 2, 3]))
        cand = f"{left} {left_label} and {right} {right_label}"
        if cand not in options:
            options.append(cand)
    rng.shuffle(options)
    return options


def _render_question_text(
    article: str, instruction: str, question: str, option_map: Dict[str, str]
) -> str:
    lines = [
        article,
        "",
        instruction,
        "",
        question,
        "",
        *(f"{letter}. {option_map[letter]}" for letter in _LETTERS),
        "",
        "Respond in the form <<< X >>> where X is A, B, C, or D.",
    ]
    return "\n".join(lines)


def _format_article_from_config(runtime_config: Dict[str, Any], strategy_name: str) -> str:
    lines = [
        (
            "You are answering questions about a trading strategy and stock market behavior. "
            "Use the Backtrader package to code, compute or verify all answers."
        ),
        "",
        "Configuration for this question group:",
    ]
    merged = dict(runtime_config)
    merged["strategy_name"] = strategy_name
    for key in sorted(merged):
        lines.append(f"- {key}: {merged[key]}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Individual question builders
# ---------------------------------------------------------------------------

def _q_first_buy(df: pd.DataFrame, rng: random.Random) -> Dict[str, Any]:
    if "buy_shares" not in df.columns:
        raise ValueError("No 'buy_shares' column; cannot build first-buy question.")

    buys = df[df["buy_shares"] > 0].copy()
    if buys.empty:
        raise ValueError("No buy executions found; cannot build first-buy question.")

    buys = buys.sort_values("date")
    first = buys.iloc[0]
    correct = f"{int(first['buy_shares'])} shares at close ${first['close']:.2f} on {first['date']}"

    distractors: List[str] = []
    if len(buys) > 1:
        sampled = buys.sample(n=min(3, len(buys)), random_state=rng.randint(0, 10**9))
        for _, row in sampled.iterrows():
            cand = f"{int(row['buy_shares'])} shares at close ${row['close']:.2f} on {row['date']}"
            if cand != correct and cand not in distractors:
                distractors.append(cand)

    while len(distractors) < 3:
        shares = max(1, int(first["buy_shares"]) + rng.choice([-50, -20, 20, 50]))
        price = max(0.01, float(first["close"]) * rng.choice([0.92, 0.96, 1.04, 1.08]))
        cand = f"{shares} shares at close ${price:.2f} on {first['date']}"
        if cand != correct and cand not in distractors:
            distractors.append(cand)

    options = [correct] + distractors[:3]
    rng.shuffle(options)
    return _choice_block(
        "For the first buy execution in the backtest, how many shares were bought, "
        "at what closing price, and on which date?",
        options,
        correct,
    )


def _q_buy_sell_signals(stats: Dict, rng: random.Random) -> Dict[str, Any]:
    opts = _pair_count_options(
        stats["buy_signals"], stats["sell_signals"], "buys", "sells", rng,
    )
    correct = f"{stats['buy_signals']} buys and {stats['sell_signals']} sells"
    return _choice_block(
        "How many buy and sell signals occurred in this time period?", opts, correct,
    )


def _q_crossovers(stats: Dict, rng: random.Random) -> Dict[str, Any]:
    opts = _numeric_options(stats["crossovers"], rng, as_int=True, non_negative=True)
    return _choice_block(
        "How many crossover events occurred during the backtest period?",
        opts, str(stats["crossovers"]),
    )


def _q_trade_outcomes(stats: Dict, rng: random.Random) -> Dict[str, Any]:
    w, l = stats["profitable_closed_trades"], stats["losing_closed_trades"]
    opts = _pair_count_options(w, l, "profitable", "losing", rng)
    return _choice_block(
        "How many profitable and losing closed trades were recorded?",
        opts, f"{w} profitable and {l} losing",
    )


def _q_net_pnl(stats: Dict, rng: random.Random) -> Dict[str, Any]:
    opts, correct = _dollar_options(stats["net_pnl"], rng)
    return _choice_block(
        "What is the net profit/loss at the end of the backtest?", opts, correct,
    )


def _q_commission(
    stats: Dict, strategy_name: str, commission_rate: float, rng: random.Random,
) -> Dict[str, Any]:
    total_comm = stats["total_commission_fee"]
    executed = stats["executed_orders"]
    q = (
        f"What was the total commission fee for {strategy_name} with a "
        f"{commission_rate * 100:.2f}% commission rate, over how many executed orders?"
    )
    correct = f"${total_comm:.2f} over {executed} orders"
    opts = [correct]
    seen = {correct}
    for _ in range(20):
        if len(opts) >= 4:
            break
        fee = max(0.0, total_comm * rng.choice([0.8, 0.9, 1.1, 1.2]))
        orders = max(0, executed + rng.choice([-2, -1, 1, 2]))
        cand = f"${fee:.2f} over {orders} orders"
        if cand not in seen:
            seen.add(cand)
            opts.append(cand)
    idx = 0
    while len(opts) < 4:
        idx += 1
        cand = f"${total_comm + idx * 5:.2f} over {executed + idx} orders"
        if cand not in seen:
            seen.add(cand)
            opts.append(cand)
    rng.shuffle(opts)
    return _choice_block(q, opts[:4], correct)


def _q_max_shares(
    stats: Dict, symbol: str, initial_cash: float, rng: random.Random,
) -> Dict[str, Any]:
    q = (
        f"What is the maximum number of shares you can buy initially for {symbol} "
        f"with initial capital ${initial_cash:.2f}?"
    )
    opts = _numeric_options(stats["max_initial_shares"], rng, as_int=True, non_negative=True)
    return _choice_block(q, opts, str(stats["max_initial_shares"]))


def _q_sma(period_label: str, date: str, value: float, rng: random.Random) -> Dict[str, Any]:
    opts = _numeric_options(value, rng)
    return _choice_block(f"What is SMA({period_label}) on {date}?", opts, f"{value:.2f}")


def _q_volume(date: str, value: float, rng: random.Random) -> Dict[str, Any]:
    opts = _numeric_options(value, rng, as_int=True, non_negative=True)
    return _choice_block(
        f"What is volume on {date}?", opts, str(int(round(value))),
    )


def _q_broker_cash(date: str, value: float, rng: random.Random) -> Dict[str, Any]:
    opts, correct = _dollar_options(value, rng)
    return _choice_block(f"What is broker cash on {date}?", opts, correct)


def _q_portfolio_value(date: str, value: float, rng: random.Random) -> Dict[str, Any]:
    opts, correct = _dollar_options(value, rng)
    return _choice_block(f"What is portfolio value on {date}?", opts, correct)


def _q_first_close(symbol: str, value: float, rng: random.Random) -> Dict[str, Any]:
    opts, correct = _dollar_options(value, rng)
    return _choice_block(
        f"What was the first close price observed for {symbol} in this backtest window?",
        opts, correct,
    )


def _pct_options(correct: float, rng: random.Random) -> Tuple[List[str], str]:
    """Return (4 percentage-formatted options, correct_str)."""
    raw = _numeric_options(correct, rng)
    return [f"{x}%" for x in raw], f"{correct:.2f}%"


def _date_dollar_options(
    correct_date: str, correct_value: float, df: pd.DataFrame,
    value_col: str, rng: random.Random,
) -> Tuple[List[str], str]:
    """Generate options combining a date and a dollar value."""
    correct_str = f"{correct_date} at ${correct_value:.2f}"
    options = [correct_str]
    candidates = df[df["date"] != correct_date]
    if len(candidates) >= 3:
        sampled = candidates.sample(n=min(6, len(candidates)), random_state=rng.randint(0, 10**9))
        for _, row in sampled.iterrows():
            cand = f"{row['date']} at ${float(row[value_col]):.2f}"
            if cand not in options:
                options.append(cand)
            if len(options) >= 4:
                break
    idx = 1
    while len(options) < 4:
        fake_val = correct_value * (1 + idx * 0.1)
        options.append(f"{correct_date} at ${fake_val:.2f}")
        idx += 1
    rng.shuffle(options)
    return options, correct_str


# ---------------------------------------------------------------------------
# Complex question builders (multi-step computation)
# ---------------------------------------------------------------------------

def _q_max_drawdown(df: pd.DataFrame, rng: random.Random) -> Dict[str, Any]:
    """Max peak-to-trough drawdown in dollar terms from portfolio_value."""
    pv = df["portfolio_value"].astype(float)
    running_peak = pv.cummax()
    drawdown = running_peak - pv
    max_dd = float(drawdown.max())
    opts, correct = _dollar_options(max_dd, rng)
    return _choice_block(
        "What was the maximum portfolio drawdown (peak-to-trough) during the backtest?",
        opts, correct,
    )


def _q_peak_portfolio_date(df: pd.DataFrame, rng: random.Random) -> Dict[str, Any]:
    """Date when portfolio_value was at its all-time high."""
    idx = df["portfolio_value"].astype(float).idxmax()
    peak_row = df.loc[idx]
    peak_date = peak_row["date"]
    peak_val = float(peak_row["portfolio_value"])
    opts, correct = _date_dollar_options(peak_date, peak_val, df, "portfolio_value", rng)
    return _choice_block(
        "On which date did the portfolio reach its highest value, and what was it?",
        opts, correct,
    )


def _q_roi_percentage(stats: Dict, initial_cash: float, rng: random.Random) -> Dict[str, Any]:
    """Return on investment = net_pnl / initial_cash * 100."""
    roi = float(stats["net_pnl"]) / initial_cash * 100.0
    roi = round(roi, 2)
    opts, correct = _pct_options(roi, rng)
    return _choice_block(
        "What was the overall return on investment (ROI) percentage for this backtest?",
        opts, correct,
    )


def _q_total_shares_traded(df: pd.DataFrame, rng: random.Random) -> Dict[str, Any]:
    """Sum of all shares bought and sold throughout the backtest."""
    total_bought = int(df["buy_shares"].sum()) if "buy_shares" in df.columns else 0
    total_sold = int(df["sell_shares"].sum()) if "sell_shares" in df.columns else 0
    total = total_bought + total_sold
    correct_str = f"{total_bought} bought and {total_sold} sold ({total} total)"
    options = [correct_str]
    seen = {correct_str}
    for _ in range(20):
        if len(options) >= 4:
            break
        b = max(0, total_bought + rng.choice([-500, -200, -100, 100, 200, 500]))
        s = max(0, total_sold + rng.choice([-500, -200, -100, 100, 200, 500]))
        cand = f"{b} bought and {s} sold ({b + s} total)"
        if cand not in seen:
            seen.add(cand)
            options.append(cand)
    idx = 1
    while len(options) < 4:
        b = total_bought + idx * 100
        s = total_sold + idx * 100
        cand = f"{b} bought and {s} sold ({b + s} total)"
        if cand not in seen:
            seen.add(cand)
            options.append(cand)
        idx += 1
    rng.shuffle(options)
    return _choice_block(
        "How many total shares were traded (bought and sold) throughout the backtest?",
        options[:4], correct_str,
    )


def _q_win_rate(stats: Dict, rng: random.Random) -> Dict[str, Any]:
    """Win rate = profitable / (profitable + losing) * 100."""
    w = stats["profitable_closed_trades"]
    l = stats["losing_closed_trades"]
    total = w + l
    if total == 0:
        raise ValueError("No closed trades; cannot compute win rate.")
    win_rate = round(w / total * 100.0, 2)
    opts, correct = _pct_options(win_rate, rng)
    return _choice_block(
        f"What was the win rate across all {total} closed trades?",
        opts, correct,
    )


def _q_best_trade_day(df: pd.DataFrame, rng: random.Random) -> Dict[str, Any]:
    """Best (most profitable) single-day realized trade PnL."""
    trade_rows = df[df["closed_trade_pnl"] != 0].copy() if "closed_trade_pnl" in df.columns else pd.DataFrame()
    if trade_rows.empty:
        raise ValueError("No closed trade PnL data in DataFrame.")
    idx = trade_rows["closed_trade_pnl"].idxmax()
    best_row = trade_rows.loc[idx]
    best_date = best_row["date"]
    best_pnl = float(best_row["closed_trade_pnl"])
    correct_str = f"{best_date} with P/L ${best_pnl:.2f}"
    options = [correct_str]
    seen = {correct_str}
    other_rows = trade_rows[trade_rows.index != idx]
    for _, row in other_rows.sample(n=min(5, len(other_rows)), random_state=rng.randint(0, 10**9)).iterrows():
        cand = f"{row['date']} with P/L ${float(row['closed_trade_pnl']):.2f}"
        if cand not in seen:
            seen.add(cand)
            options.append(cand)
        if len(options) >= 4:
            break
    i = 1
    while len(options) < 4:
        cand = f"{best_date} with P/L ${best_pnl * (0.5 + i * 0.15):.2f}"
        if cand not in seen:
            seen.add(cand)
            options.append(cand)
        i += 1
    rng.shuffle(options)
    return _choice_block(
        "On which date was the most profitable closed trade, and what was its realized P/L?",
        options[:4], correct_str,
    )


def _q_profit_factor(df: pd.DataFrame, rng: random.Random) -> Dict[str, Any]:
    """Profit factor = sum(winning PnL) / abs(sum(losing PnL))."""
    if "closed_trade_pnl" not in df.columns:
        raise ValueError("No closed_trade_pnl column.")
    trade_pnls = df["closed_trade_pnl"]
    gross_profit = float(trade_pnls[trade_pnls > 0].sum())
    gross_loss = float(trade_pnls[trade_pnls < 0].sum())
    if abs(gross_loss) < 1e-9:
        raise ValueError("No losing trades; profit factor is infinite.")
    pf = round(gross_profit / abs(gross_loss), 2)
    opts = _numeric_options(pf, rng)
    return _choice_block(
        "What is the profit factor (gross profit / gross loss) for this backtest?",
        opts, f"{pf:.2f}",
    )


def _q_days_in_position(df: pd.DataFrame, rng: random.Random) -> Dict[str, Any]:
    """Count of days holding a position vs total trading days."""
    in_pos = int((df["position_size"] > 0).sum())
    total = len(df)
    out_pos = total - in_pos
    correct_str = f"{in_pos} days in position, {out_pos} days out ({total} total trading days)"
    options = [correct_str]
    seen = {correct_str}
    for _ in range(20):
        if len(options) >= 4:
            break
        ip = max(0, in_pos + rng.choice([-30, -15, -5, 5, 15, 30]))
        op = total - ip
        cand = f"{ip} days in position, {op} days out ({total} total trading days)"
        if cand != correct_str and cand not in seen:
            seen.add(cand)
            options.append(cand)
    i = 1
    while len(options) < 4:
        ip = in_pos + i * 10
        op = total - ip
        cand = f"{ip} days in position, {op} days out ({total} total trading days)"
        if cand not in seen:
            seen.add(cand)
            options.append(cand)
        i += 1
    rng.shuffle(options)
    return _choice_block(
        "How many trading days was the strategy in a position vs out of a position?",
        options[:4], correct_str,
    )


def _q_annualized_return(
    stats: Dict, initial_cash: float, df: pd.DataFrame, rng: random.Random,
) -> Dict[str, Any]:
    """Annualized return = ((final / initial)^(365 / trading_days) - 1) * 100."""
    final = float(stats["net_pnl"]) + initial_cash
    n_days = len(df)
    if n_days == 0 or initial_cash <= 0:
        raise ValueError("Cannot compute annualized return with no data or zero cash.")
    ann_return = round(((final / initial_cash) ** (365.0 / n_days) - 1) * 100.0, 2)
    opts, correct = _pct_options(ann_return, rng)
    return _choice_block(
        f"What is the annualized return over the {n_days} trading-day backtest period?",
        opts, correct,
    )


def _max_drawdown_window_values(df: pd.DataFrame) -> Dict[str, Any]:
    pv = df["portfolio_value"].astype(float)
    running_peak = pv.cummax()
    drawdown = running_peak - pv
    trough_idx = drawdown.idxmax()
    peak_idx = pv.loc[:trough_idx].idxmax()
    peak_value = float(pv.loc[peak_idx])
    trough_value = float(pv.loc[trough_idx])
    drawdown_dollars = peak_value - trough_value
    drawdown_pct = (drawdown_dollars / peak_value * 100.0) if peak_value > 0 else 0.0
    return {
        "peak_idx": peak_idx,
        "trough_idx": trough_idx,
        "peak_date": df.loc[peak_idx, "date"],
        "trough_date": df.loc[trough_idx, "date"],
        "peak_value": peak_value,
        "trough_value": trough_value,
        "drawdown_dollars": drawdown_dollars,
        "drawdown_pct": drawdown_pct,
    }


def _q_max_drawdown_window(df: pd.DataFrame, rng: random.Random) -> Dict[str, Any]:
    """Max drawdown window: peak date, trough date, dollars, and percentage."""
    window = _max_drawdown_window_values(df)
    correct = (
        f"{window['peak_date']} to {window['trough_date']}: "
        f"${window['drawdown_dollars']:.2f} ({window['drawdown_pct']:.2f}%)"
    )
    options = [correct]
    seen = {correct}

    candidate_rows = df[df.index != window["trough_idx"]]
    if len(candidate_rows) > 0:
        sampled = candidate_rows.sample(n=min(6, len(candidate_rows)), random_state=rng.randint(0, 10**9))
        for _, row in sampled.iterrows():
            fake_trough_idx = row.name
            fake_peak_idx = df["portfolio_value"].astype(float).loc[:fake_trough_idx].idxmax()
            fake_peak = float(df.loc[fake_peak_idx, "portfolio_value"])
            fake_trough = float(row["portfolio_value"])
            fake_dd = max(0.0, fake_peak - fake_trough)
            fake_pct = (fake_dd / fake_peak * 100.0) if fake_peak > 0 else 0.0
            cand = f"{df.loc[fake_peak_idx, 'date']} to {row['date']}: ${fake_dd:.2f} ({fake_pct:.2f}%)"
            if cand not in seen:
                seen.add(cand)
                options.append(cand)
            if len(options) >= 4:
                break

    i = 1
    while len(options) < 4:
        cand = (
            f"{window['peak_date']} to {window['trough_date']}: "
            f"${window['drawdown_dollars'] * (1 + i * 0.08):.2f} "
            f"({window['drawdown_pct'] * (1 + i * 0.08):.2f}%)"
        )
        if cand not in seen:
            seen.add(cand)
            options.append(cand)
        i += 1

    rng.shuffle(options)
    return _choice_block(
        "Reconstruct the maximum drawdown window from portfolio_value. "
        "Which peak-to-trough interval produced the largest drawdown, and what "
        "were the dollar and percentage drawdowns?",
        options[:4],
        correct,
    )


def _q_drawdown_recovery_days(df: pd.DataFrame, rng: random.Random) -> Dict[str, Any]:
    """Recovery time from the max-drawdown trough back to the prior peak value."""
    window = _max_drawdown_window_values(df)
    after_trough = df.loc[window["trough_idx"] + 1:].copy()
    recovered = after_trough[
        after_trough["portfolio_value"].astype(float) >= window["peak_value"]
    ]

    if recovered.empty:
        correct = f"Not recovered by end of backtest after {window['trough_date']}"
        distractor_dates = after_trough["date"].head(3).tolist()
        options = [correct] + [
            f"{i + 1} trading days after {window['trough_date']} (recovered on {date})"
            for i, date in enumerate(distractor_dates)
        ]
    else:
        recovery_idx = recovered.index[0]
        recovery_date = df.loc[recovery_idx, "date"]
        recovery_days = int(recovery_idx - window["trough_idx"])
        correct = (
            f"{recovery_days} trading days after {window['trough_date']} "
            f"(recovered on {recovery_date})"
        )
        options = [correct]
        for offset in [-5, -2, 2, 5, 10]:
            days = max(1, recovery_days + offset)
            idx = min(len(df) - 1, window["trough_idx"] + days)
            cand = (
                f"{days} trading days after {window['trough_date']} "
                f"(recovered on {df.loc[idx, 'date']})"
            )
            if cand not in options:
                options.append(cand)
            if len(options) >= 4:
                break

    i = 1
    while len(options) < 4:
        options.append(
            f"{i * 10} trading days after {window['trough_date']} "
            f"(recovered on {df.iloc[min(len(df) - 1, window['trough_idx'] + i * 10)]['date']})"
        )
        i += 1

    rng.shuffle(options)
    return _choice_block(
        "Using the maximum drawdown window, how many trading days after the trough "
        "did portfolio_value recover to the prior peak value?",
        options[:4],
        correct,
    )


def _q_calmar_ratio(
    stats: Dict, initial_cash: float, df: pd.DataFrame, rng: random.Random,
) -> Dict[str, Any]:
    """Calmar-like ratio: annualized return percentage / max drawdown percentage."""
    window = _max_drawdown_window_values(df)
    if window["drawdown_pct"] <= 0:
        raise ValueError("No drawdown; cannot compute Calmar ratio.")
    final = float(stats["net_pnl"]) + initial_cash
    annualized_pct = ((final / initial_cash) ** (365.0 / len(df)) - 1) * 100.0
    ratio = round(annualized_pct / window["drawdown_pct"], 2)
    opts = _numeric_options(ratio, rng)
    return _choice_block(
        "Compute the Calmar-style ratio for this backtest: annualized return "
        "percentage divided by maximum drawdown percentage. What is the ratio?",
        opts,
        f"{ratio:.2f}",
    )


def _q_exposure_adjusted_return(
    stats: Dict, initial_cash: float, df: pd.DataFrame, rng: random.Random,
) -> Dict[str, Any]:
    """Exposure-adjusted ROI: ROI percentage divided by market exposure percentage."""
    exposure_pct = (df["position_size"].astype(float) > 0).mean() * 100.0
    if exposure_pct <= 0:
        raise ValueError("No days in position; cannot compute exposure-adjusted return.")
    roi_pct = float(stats["net_pnl"]) / initial_cash * 100.0
    adjusted = round(roi_pct / exposure_pct, 2)
    opts = _numeric_options(adjusted, rng)
    return _choice_block(
        "Compute exposure-adjusted ROI, defined as overall ROI percentage divided "
        "by market exposure percentage (trading days with a nonzero position / "
        "total trading days). What is the value?",
        opts,
        f"{adjusted:.2f}",
    )


# ---------------------------------------------------------------------------
# Strategy-dependent question builders (require running Backtrader code)
# ---------------------------------------------------------------------------

def _q_position_on_date(df: pd.DataFrame, strategy_name: str, rng: random.Random) -> Dict[str, Any]:
    """Position size on a random date. Must run the strategy to know."""
    row = df.iloc[rng.randrange(len(df))]
    date, pos = row["date"], int(row["position_size"])
    opts = _numeric_options(pos, rng, as_int=True, non_negative=True)
    return _choice_block(
        f"After running the {strategy_name} strategy, how many shares does "
        f"the strategy hold at the close of {date}?",
        opts, str(pos),
    )


def _q_strategy_action_on_date(
    df: pd.DataFrame, strategy_name: str, rng: random.Random,
) -> Dict[str, Any]:
    """Did the strategy buy, sell, or hold on a specific date?
    No indicator values are given — you must run the backtest to find out."""
    buy_dates = set(df[df["buy_shares"] > 0]["date"]) if "buy_shares" in df.columns else set()
    sell_dates = set(df[df["sell_shares"] > 0]["date"]) if "sell_shares" in df.columns else set()

    pool: List[Tuple[str, str]] = []
    for d in buy_dates:
        shares = int(df[df["date"] == d]["buy_shares"].iloc[0])
        pool.append(("buy", d, f"Buy — {shares} shares purchased"))
    for d in sell_dates:
        shares = int(df[df["date"] == d]["sell_shares"].iloc[0])
        pool.append(("sell", d, f"Sell — {shares} shares sold (position closed)"))
    hold_rows = df[~df["date"].isin(buy_dates | sell_dates)]
    hold_sample = hold_rows.sample(n=min(5, len(hold_rows)), random_state=rng.randint(0, 10**9))
    for _, r in hold_sample.iterrows():
        pool.append(("hold", r["date"], "No action — the strategy held its current position"))

    if not pool:
        raise ValueError("No action data available.")

    action_type, date, correct = pool[rng.randrange(len(pool))]

    if action_type == "buy":
        options = [
            correct,
            "Sell — position closed on this day",
            "No action — the strategy held its current position",
            "No action — a buy signal occurred but insufficient cash",
        ]
    elif action_type == "sell":
        options = [
            correct,
            "Buy — shares purchased on this day",
            "No action — the strategy held its current position",
            "No action — a sell signal occurred but no position was held",
        ]
    else:
        options = [
            correct,
            "Buy — shares purchased on this day",
            "Sell — position closed on this day",
            "Buy and Sell — both executed on this day",
        ]

    rng.shuffle(options)
    return _choice_block(
        f"Run the {strategy_name} strategy with the given config. "
        f"What action did the strategy take on {date}?",
        options, correct,
    )


def _q_nth_trade_pnl(df: pd.DataFrame, strategy_name: str, rng: random.Random) -> Dict[str, Any]:
    """Realized P/L of the Nth closed trade. Must run backtest to know trade lifecycle."""
    if "closed_trade_pnl" not in df.columns:
        raise ValueError("No closed_trade_pnl column.")
    trade_rows = df[df["closed_trade_pnl"] != 0].sort_values("date").reset_index(drop=True)
    if trade_rows.empty:
        raise ValueError("No closed trades found.")

    n = rng.randrange(len(trade_rows))
    row = trade_rows.iloc[n]
    pnl = float(row["closed_trade_pnl"])
    date = row["date"]
    ordinal = _ordinal(n + 1)

    opts, correct = _dollar_options(pnl, rng)
    return _choice_block(
        f"After running the {strategy_name} strategy, what was the realized net P/L "
        f"(after commission) of the {ordinal} closed trade (which closed on {date})?",
        opts, correct,
    )


def _q_cash_after_nth_order(df: pd.DataFrame, strategy_name: str, rng: random.Random) -> Dict[str, Any]:
    """Broker cash on the date the Nth order executed. Must run backtest."""
    if "buy_shares" not in df.columns or "sell_shares" not in df.columns:
        raise ValueError("No order columns.")
    order_rows = df[(df["buy_shares"] > 0) | (df["sell_shares"] > 0)].copy()
    if order_rows.empty:
        raise ValueError("No executed orders found.")

    order_rows = order_rows.sort_values("date").reset_index(drop=True)
    n = rng.randrange(len(order_rows))
    row = order_rows.iloc[n]
    cash = float(row["broker_cash"])
    date = row["date"]
    ordinal = _ordinal(n + 1)

    opts, correct = _dollar_options(cash, rng)
    return _choice_block(
        f"After running the {strategy_name} strategy, what was the broker's cash "
        f"balance at the close of the {ordinal} order execution day ({date})?",
        opts, correct,
    )


def _q_signal_diff_with_alt_params(
    runtime_config: Dict, strategy_name: str,
    base_buy_signals: int, base_fast: int, base_slow: int,
    rng: random.Random,
) -> Dict[str, Any]:
    """Compare buy-signal counts between two parameter sets.
    Must run two separate backtests."""
    if not _supports_fast_slow_params(runtime_config):
        raise ValueError(f"{strategy_name} does not use pfast/pslow strategy parameters.")
    alt_params = _make_alt_sma_params(base_fast, base_slow, rng)
    alt_config = dict(runtime_config)
    alt_config["strategy_params"] = {
        **dict(runtime_config.get("strategy_params") or {}),
        **alt_params,
    }
    alt_df = run_backtrader_info(config=alt_config, strategy_name=strategy_name, plot=False)
    alt_buy_signals = int(alt_df["buy_signals"].iloc[-1])

    diff = base_buy_signals - alt_buy_signals
    base_label = f"(pfast={base_fast}, pslow={base_slow})"
    alt_label = f"(pfast={alt_params['pfast']}, pslow={alt_params['pslow']})"

    correct_str = f"{base_buy_signals} vs {alt_buy_signals} (difference of {abs(diff)})"
    options = [correct_str]
    seen = {correct_str}
    for _ in range(20):
        if len(options) >= 4:
            break
        b = max(0, base_buy_signals + rng.choice([-3, -2, -1, 1, 2, 3]))
        a = max(0, alt_buy_signals + rng.choice([-3, -2, -1, 1, 2, 3]))
        cand = f"{b} vs {a} (difference of {abs(b - a)})"
        if cand not in seen:
            seen.add(cand)
            options.append(cand)
    idx = 1
    while len(options) < 4:
        b = base_buy_signals + idx
        a = alt_buy_signals - idx
        cand = f"{b} vs {a} (difference of {abs(b - a)})"
        if cand not in seen:
            seen.add(cand)
            options.append(cand)
        idx += 1

    rng.shuffle(options)
    return _choice_block(
        f"Run the {strategy_name} strategy twice — once with {base_label} and once "
        f"with {alt_label}. How many buy signals does each produce?",
        options[:4], correct_str,
    )


def _q_avg_holding_period(
    df: pd.DataFrame, stats: Dict, strategy_name: str, rng: random.Random,
) -> Dict[str, Any]:
    """Average trading days in position per closed trade. Must run backtest."""
    total_closed = stats["profitable_closed_trades"] + stats["losing_closed_trades"]
    if total_closed == 0:
        raise ValueError("No closed trades; cannot compute average holding period.")
    days_in = int((df["position_size"] > 0).sum())
    avg = round(days_in / total_closed, 2)
    opts = _numeric_options(avg, rng)
    return _choice_block(
        f"After running the {strategy_name} strategy, what is the average number of "
        f"trading days the strategy held a position per closed trade "
        f"({days_in} total days in position across {total_closed} closed trades)?",
        opts, f"{avg:.2f}",
    )


def _ordinal(n: int) -> str:
    if 11 <= (n % 100) <= 13:
        return f"{n}th"
    return f"{n}{['th', 'st', 'nd', 'rd'][min(n % 10, 4)] if n % 10 < 4 else 'th'}"


def _q_golden_cross_count(
    df: pd.DataFrame, fast_period: int, slow_period: int, rng: random.Random,
) -> Dict[str, Any]:
    """Count golden crosses (SMA_fast crossing above SMA_slow) in a sub-period.
    Requires understanding what a golden cross means in the SMA context."""
    diff = df["sma_fast"] - df["sma_slow"]
    prev_diff = diff.shift(1)
    golden = (prev_diff <= 0) & (diff > 0)

    n = len(df)
    if n < 60:
        si, ei = 0, n - 1
    else:
        si = rng.randrange(0, n - 60)
        ei = min(si + rng.randint(60, min(250, n - si)), n - 1)

    start_date = df.iloc[si]["date"]
    end_date = df.iloc[ei]["date"]
    count = int(golden.iloc[si:ei + 1].sum())

    opts = _numeric_options(count, rng, as_int=True, non_negative=True)
    return _choice_block(
        f"Between {start_date} and {end_date}, how many golden crosses "
        f"(SMA({fast_period}) crossing above SMA({slow_period})) occurred?",
        opts, str(count),
    )


def _supports_fast_slow_params(runtime_config: Dict[str, Any]) -> bool:
    params = dict(runtime_config.get("strategy_params") or {})
    return "pfast" in params and "pslow" in params


def _make_alt_sma_params(base_fast: int, base_slow: int, rng: random.Random) -> Dict[str, int]:
    for _ in range(20):
        cand_fast = max(2, base_fast + rng.choice([-5, -3, -2, 2, 3, 5]))
        cand_slow = max(cand_fast + 5, base_slow + rng.choice([-10, -5, 5, 10]))
        if cand_fast != base_fast or cand_slow != base_slow:
            return {"pfast": cand_fast, "pslow": cand_slow}
    return {"pfast": max(2, base_fast + 2), "pslow": max(base_slow + 5, base_fast + 7)}


def _q_sma_comparison(
    base_fast: int, base_slow: int,
    runtime_config: Dict, strategy_name: str,
    base_net_pnl: float, rng: random.Random,
) -> Dict[str, Any]:
    if not _supports_fast_slow_params(runtime_config):
        raise ValueError(f"{strategy_name} does not use pfast/pslow strategy parameters.")
    alt_params = _make_alt_sma_params(base_fast, base_slow, rng)
    alt_config = dict(runtime_config)
    alt_config["strategy_params"] = {
        **dict(runtime_config.get("strategy_params") or {}),
        **alt_params,
    }
    alt_df = run_backtrader_info(config=alt_config, strategy_name=strategy_name, plot=False)
    alt_net_pnl = float(alt_df["net_pnl"].iloc[-1])

    base_label = f"(pfast={base_fast}, pslow={base_slow})"
    alt_label = f"(pfast={alt_params['pfast']}, pslow={alt_params['pslow']})"

    if base_net_pnl >= alt_net_pnl:
        correct = f"{base_label} with net P/L ${base_net_pnl:.2f}"
    else:
        correct = f"{alt_label} with net P/L ${alt_net_pnl:.2f}"

    q = (
        "Under the same config, which SMA parameter set produced the better net performance "
        f"between {base_label} and {alt_label}?"
    )
    opts = list(dict.fromkeys([
        f"{base_label} with net P/L ${base_net_pnl:.2f}",
        f"{alt_label} with net P/L ${alt_net_pnl:.2f}",
        f"{base_label} with net P/L ${alt_net_pnl:.2f}",
        f"{alt_label} with net P/L ${base_net_pnl:.2f}",
    ]))
    while len(opts) < 4:
        opts.append(f"{base_label} with net P/L ${base_net_pnl + len(opts) * 10:.2f}")
    rng.shuffle(opts)
    return _choice_block(q, opts[:4], correct)


# ---------------------------------------------------------------------------
# Scalar extraction (called once per backtest df)
# ---------------------------------------------------------------------------

_STAT_COLS = [
    "buy_signals", "sell_signals", "crossovers",
    "profitable_closed_trades", "losing_closed_trades",
    "net_pnl", "total_commission_fee", "executed_orders",
    "max_initial_shares", "first_close_price",
    "sma_fast_period", "sma_slow_period",
]


def _extract_stats(df: pd.DataFrame) -> Dict[str, Any]:
    last = df.iloc[-1]
    return {col: last[col] for col in _STAT_COLS}


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def _make_factories(
    df: pd.DataFrame,
    stats: Dict[str, Any],
    runtime_config: Dict[str, Any],
    strategy_name: str,
    strategy_params: Dict[str, Any],
    sma_fast_rows: pd.DataFrame,
    sma_slow_rows: pd.DataFrame,
    base_fast: int,
    base_slow: int,
    symbol: str,
    initial_cash: float,
    commission_rate: float,
) -> Dict[str, Any]:
    """Return a {qtype: callable(rng) -> built_question} registry.

    Builders are invoked lazily so expensive types (e.g. those that run a
    second backtest) only execute when actually sampled.
    """
    def _pick(rows: pd.DataFrame, rng: random.Random):
        return rows.iloc[rng.randrange(len(rows))]

    def _sma_either(rng: random.Random) -> Dict[str, Any]:
        if rng.random() < 0.5:
            r = _pick(sma_fast_rows, rng)
            return _q_sma(str(base_fast), r["date"], float(r["sma_fast"]), rng)
        r = _pick(sma_slow_rows, rng)
        return _q_sma(str(base_slow), r["date"], float(r["sma_slow"]), rng)

    def _volume(rng: random.Random) -> Dict[str, Any]:
        r = _pick(df, rng)
        return _q_volume(r["date"], float(r["volume"]), rng)

    def _broker_cash(rng: random.Random) -> Dict[str, Any]:
        r = _pick(df, rng)
        return _q_broker_cash(r["date"], float(r["broker_cash"]), rng)

    def _portfolio_value(rng: random.Random) -> Dict[str, Any]:
        r = _pick(df, rng)
        return _q_portfolio_value(r["date"], float(r["portfolio_value"]), rng)

    return {
        "first_close":             lambda rng: _q_first_close(symbol, float(stats["first_close_price"]), rng),
        "volume":                  _volume,
        "broker_cash":             _broker_cash,
        "portfolio_value":         _portfolio_value,
        "sma":                     _sma_either,
        "max_shares":              lambda rng: _q_max_shares(stats, symbol, initial_cash, rng),
        "position_on_date":        lambda rng: _q_position_on_date(df, strategy_name, rng),
        "crossovers":              lambda rng: _q_crossovers(stats, rng),
        "buy_sell_signals":        lambda rng: _q_buy_sell_signals(stats, rng),

        "first_buy":               lambda rng: _q_first_buy(df, rng),
        "trade_outcomes":          lambda rng: _q_trade_outcomes(stats, rng),
        "net_pnl":                 lambda rng: _q_net_pnl(stats, rng),
        "commission":              lambda rng: _q_commission(stats, strategy_name, commission_rate, rng),
        "total_shares_traded":     lambda rng: _q_total_shares_traded(df, rng),
        "days_in_position":        lambda rng: _q_days_in_position(df, rng),
        "peak_portfolio_date":     lambda rng: _q_peak_portfolio_date(df, rng),
        "best_trade_day":          lambda rng: _q_best_trade_day(df, rng),
        "strategy_action_on_date": lambda rng: _q_strategy_action_on_date(df, strategy_name, rng),
        "nth_trade_pnl":           lambda rng: _q_nth_trade_pnl(df, strategy_name, rng),
        "cash_after_nth_order":    lambda rng: _q_cash_after_nth_order(df, strategy_name, rng),

        "max_drawdown":            lambda rng: _q_max_drawdown(df, rng),
        "roi_percentage":          lambda rng: _q_roi_percentage(stats, initial_cash, rng),
        "win_rate":                lambda rng: _q_win_rate(stats, rng),
        "profit_factor":           lambda rng: _q_profit_factor(df, rng),
        "annualized_return":       lambda rng: _q_annualized_return(stats, initial_cash, df, rng),
        "avg_holding_period":      lambda rng: _q_avg_holding_period(df, stats, strategy_name, rng),
        "golden_cross_count":      lambda rng: _q_golden_cross_count(df, base_fast, base_slow, rng),
        "max_drawdown_window":     lambda rng: _q_max_drawdown_window(df, rng),
        "drawdown_recovery_days":  lambda rng: _q_drawdown_recovery_days(df, rng),
        "calmar_ratio":            lambda rng: _q_calmar_ratio(stats, initial_cash, df, rng),
        "exposure_adjusted_return": lambda rng: _q_exposure_adjusted_return(
            stats, initial_cash, df, rng,
        ),
        "sma_comparison":          lambda rng: _q_sma_comparison(
            base_fast, base_slow, runtime_config, strategy_name,
            float(stats["net_pnl"]), rng,
        ),
        "signal_diff":             lambda rng: _q_signal_diff_with_alt_params(
            runtime_config, strategy_name,
            stats["buy_signals"], base_fast, base_slow, rng,
        ),
    }


_FAST_SLOW_QTYPES = {"sma_comparison", "signal_diff"}


def _qtype_is_compatible(qtype: str, runtime_config: Dict[str, Any]) -> bool:
    """Return whether a template can run under the current strategy config."""
    if qtype in _FAST_SLOW_QTYPES:
        return _supports_fast_slow_params(runtime_config)
    return True


def _compatible_pools_for_levels(
    levels: List[str],
    runtime_config: Dict[str, Any],
) -> Dict[str, List[str]]:
    pools = _types_for_levels(levels)
    return {
        level: [
            qtype for qtype in pool
            if _qtype_is_compatible(qtype, runtime_config)
        ]
        for level, pool in pools.items()
    }


def _remove_qtype_from_pools(
    pools: Dict[str, List[str]],
    qtype: str,
) -> Dict[str, List[str]]:
    return {
        level: [candidate for candidate in pool if candidate != qtype]
        for level, pool in pools.items()
    }


def _plan_qtypes(
    num_questions: int,
    levels: List[str],
    rng: random.Random,
    pools: Optional[Dict[str, List[str]]] = None,
) -> List[str]:
    """Produce a shuffled list of qtypes of length num_questions.

    Splits the quota across `levels` as evenly as possible. Within each
    level the sampling is without replacement when possible, otherwise
    the pool is cycled (with replacement for the overflow).
    """
    counts = _allocate_counts(num_questions, levels)
    pools = pools or _types_for_levels(levels)

    selected: List[str] = []
    for l in levels:
        pool = pools[l]
        n = counts[l]
        if not pool:
            continue
        if n <= len(pool):
            selected.extend(rng.sample(pool, k=n))
        else:
            selected.extend(pool)
            for _ in range(n - len(pool)):
                selected.append(rng.choice(pool))
    rng.shuffle(selected)
    return selected


def _all_qtypes_for_levels(
    levels: List[str],
    pools: Optional[Dict[str, List[str]]] = None,
) -> List[str]:
    """Return each registered template exactly once for the requested levels."""
    pools = pools or _types_for_levels(levels)
    selected: List[str] = []
    for level in levels:
        selected.extend(pools[level])
    return selected


def build_mcq_dataset(
    config: Optional[Dict] = None,
    strategy_name: str = DEFAULT_STRATEGY_NAME,
    num_questions: int = 10,
    rng: Optional[random.Random] = None,
    difficulty: Union[None, str, Iterable[str]] = None,
    qtypes: Optional[List[str]] = None,
):
    """Build an MCQ dataset.

    Parameters
    ----------
    difficulty:
        Which difficulty pool(s) to draw from. Accepts None/"all" (balanced
        across all three levels), a single level ("easy"/"medium"/"hard"),
        a comma-separated string ("easy,medium"), or an iterable of levels.
        When multiple levels are requested, questions are pooled roughly
        equally across them.
    qtypes:
        Optional explicit question-type plan. When provided, it is used as-is
        so callers can build a base pool with one example per template.
    """
    if qtypes is None and num_questions <= 0:
        raise ValueError("num_questions must be a positive integer.")
    if rng is None:
        rng = random.Random()

    levels = _normalize_difficulty(difficulty)

    runtime_config = dict(CONFIG)
    if config:
        runtime_config.update(config)
    _sp_defaults = get_default_params(strategy_name)
    _sp_user = dict(runtime_config.get("strategy_params") or {})
    runtime_config["strategy_params"] = {**_sp_defaults, **_sp_user}

    df = run_backtrader_info(config=runtime_config, strategy_name=strategy_name, plot=False)
    if df.empty:
        raise ValueError("run_backtrader_info returned an empty dataframe.")

    df = df.sort_values("date").reset_index(drop=True)
    for col in ["sma_fast", "sma_slow", "volume", "broker_cash", "portfolio_value"]:
        if col not in df.columns:
            raise ValueError(f"Missing expected column '{col}' from run_backtrader_info output.")

    stats = _extract_stats(df)
    base_fast = int(stats["sma_fast_period"])
    base_slow = int(stats["sma_slow_period"])
    symbol = runtime_config["symbol"]
    initial_cash = float(runtime_config["initial_cash"])
    commission_rate = float(runtime_config["commission_rate"])

    sma_fast_rows = df[df["sma_fast"].notna()]
    sma_slow_rows = df[df["sma_slow"].notna()]
    if sma_fast_rows.empty:
        sma_fast_rows = df
    if sma_slow_rows.empty:
        sma_slow_rows = df

    strategy_params = runtime_config.get("strategy_params", get_default_params(strategy_name))

    factories = _make_factories(
        df=df,
        stats=stats,
        runtime_config=runtime_config,
        strategy_name=strategy_name,
        strategy_params=strategy_params,
        sma_fast_rows=sma_fast_rows,
        sma_slow_rows=sma_slow_rows,
        base_fast=base_fast,
        base_slow=base_slow,
        symbol=symbol,
        initial_cash=initial_cash,
        commission_rate=commission_rate,
    )

    built_entries: List[Tuple[str, Dict[str, Any]]] = []
    skipped: List[Tuple[str, str]] = []

    if qtypes is None:
        compatible_pools = _compatible_pools_for_levels(levels, runtime_config)
        attempts = 0
        max_attempts = max(num_questions * 20, sum(len(p) for p in compatible_pools.values()))
        while len(built_entries) < num_questions and attempts < max_attempts:
            remaining = num_questions - len(built_entries)
            planned_qtypes = _plan_qtypes(
                remaining,
                levels,
                rng,
                pools=compatible_pools,
            )
            if not planned_qtypes:
                break
            for qtype in planned_qtypes:
                attempts += 1
                try:
                    built = factories[qtype](rng)
                except ValueError as exc:
                    # Data-dependent builders may legitimately fail for this
                    # backtest (e.g. no closed trades). Remove that template
                    # for this config and sample another compatible template.
                    skipped.append((qtype, str(exc)))
                    compatible_pools = _remove_qtype_from_pools(compatible_pools, qtype)
                    continue
                built_entries.append((qtype, built))
                if len(built_entries) >= num_questions or attempts >= max_attempts:
                    break
    else:
        invalid = [qtype for qtype in qtypes if qtype not in factories]
        if invalid:
            raise ValueError(f"Unknown question type(s): {invalid}")
        planned_qtypes = []
        for qtype in qtypes:
            if _qtype_is_compatible(qtype, runtime_config):
                planned_qtypes.append(qtype)
            else:
                skipped.append((
                    qtype,
                    "template requires pfast/pslow strategy parameters",
                ))
        for qtype in planned_qtypes:
            try:
                built = factories[qtype](rng)
            except ValueError as exc:
                # Explicit qtype plans are honored as-is; skipped entries are
                # reported instead of replaced with a different template.
                skipped.append((qtype, str(exc)))
                continue
            built_entries.append((qtype, built))

    if not built_entries:
        raise ValueError(
            "No questions could be built for the requested difficulty/configuration. "
            f"Skipped: {skipped}"
        )

    if skipped:
        print(f"[build_mcq_dataset] skipped {len(skipped)} question(s): "
              + ", ".join(f"{t} ({msg})" for t, msg in skipped))

    article = _format_article_from_config(runtime_config, strategy_name)
    instruction = (
        "Use the config and strategy setting for this question. "
        "Run the Backtrader strategy when positions, orders, trades, drawdowns, "
        "or alternate parameters matter; all answers should come from the "
        "resulting backtest dataframe values and the formula stated in the question."
    )

    questions: List[str] = []
    options: List[List[str]] = []
    answers: List[str] = []
    question_texts: List[str] = []
    types: List[str] = []
    difficulties: List[str] = []

    for qtype, built in built_entries:
        om = built["option_map"]
        questions.append(built["question"])
        options.append([om[l] for l in _LETTERS])
        answers.append(built["answer"])
        question_texts.append(
            _render_question_text(article, instruction, built["question"], om)
        )
        types.append(qtype)
        difficulties.append(DIFFICULTY_BY_TYPE[qtype])

    summary = {
        "symbol": symbol,
        "start_date": runtime_config["start_date"],
        "end_date": runtime_config["end_date"],
        "initial_cash": initial_cash,
        "commission_rate": commission_rate,
        "stake": int(runtime_config.get("stake", 1000)),
        "strategy_name": strategy_name,
        "strategy_params": dict(runtime_config.get("strategy_params", get_default_params(strategy_name))),
        "difficulty_levels": levels,
    }

    return {
        "id": f"{symbol}_{strategy_name}_{summary['start_date']}_{summary['end_date']}",
        "article": article,
        "questions": questions,
        "options": options,
        "answers": answers,
        "question_texts": question_texts,
        "types": types,
        "difficulties": difficulties,
        "config": summary,
    }


# ---------------------------------------------------------------------------
# JSONL writer
# ---------------------------------------------------------------------------

def save_mcq_dataset_json(
    output_path: str,
    config: Optional[Dict] = None,
    strategy_name: str = DEFAULT_STRATEGY_NAME,
    num_questions: int = 10,
    seed: Optional[int] = None,
    benchmark: str = "backtrader",
    author: str = "MCQ_test",
    difficulty: Union[None, str, Iterable[str]] = None,
    all_templates: bool = False,
    output_format: str = "jsonl",
):
    rng = random.Random(seed)
    levels = _normalize_difficulty(difficulty)
    qtypes = _all_qtypes_for_levels(levels) if all_templates else None
    dataset = build_mcq_dataset(
        config=config,
        strategy_name=strategy_name,
        num_questions=num_questions,
        rng=rng,
        difficulty=levels,
        qtypes=qtypes,
    )
    dataset_config = dataset["config"]

    records = [
        {
            "question": q,
            "answer": a,
            "benchmark": benchmark,
            "author": author,
            "type": t,
            "difficulty": d,
            "strategy_name": dataset_config["strategy_name"],
            "strategy_params": dataset_config["strategy_params"],
            "config": dataset_config,
        }
        for q, a, t, d in zip(
            dataset["question_texts"],
            dataset["answers"],
            dataset["types"],
            dataset["difficulties"],
        )
    ]

    if output_format not in ("jsonl", "json"):
        raise ValueError("output_format must be either 'jsonl' or 'json'.")

    with open(output_path, "w", encoding="utf-8") as f:
        if output_format == "json":
            json.dump(records, f, ensure_ascii=False, indent=2)
            f.write("\n")
        else:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # Print the difficulty breakdown of what actually got written.
    from collections import Counter
    diff_counts = Counter(dataset["difficulties"])
    breakdown = ", ".join(
        f"{lvl}={diff_counts.get(lvl, 0)}" for lvl in DIFFICULTY_LEVELS
    )
    print(f"Difficulty breakdown: {breakdown} (total={len(records)})")

    return records


if __name__ == "__main__":
    out = save_mcq_dataset_json(
        output_path="Backtrader/backtrader_mcq_dataset.json",
        num_questions=10,
        seed=42,
    )
    print(f"Saved MCQ dataset with {len(out)} questions.")
