#!/usr/bin/env python3
"""Verify that every MCQ answer in a JSONL file matches ground truth.

Re-runs the backtest independently, reads structured config metadata from
new JSONL records when available, falls back to parsing older question text,
computes the correct answer from the backtest DataFrame, and compares against
the stated answer letter.

Usage:
    python backtrader_MCQ_checker.py Backtrader/backtrader_mcq_20.json
"""

import ast
import json
import re
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from backtrader_info import run_backtrader_info
from backtrader_MCQ import DIFFICULTY_BY_TYPE, DIFFICULTY_LEVELS
from strategies import DEFAULT_STRATEGY_NAME


# ---------------------------------------------------------------------------
# Config parser — extract backtest config from the question text
# ---------------------------------------------------------------------------

def _parse_config_from_question(question_text: str) -> Dict[str, Any]:
    """Extract the config block embedded in the question text."""
    config = {}
    for match in re.finditer(r"^- (\w+): (.+)$", question_text, re.MULTILINE):
        key, raw_value = match.group(1), match.group(2).strip()
        if key == "strategy_params":
            config[key] = ast.literal_eval(raw_value)
        elif key in ("initial_cash", "commission_rate"):
            config[key] = float(raw_value)
        elif key in ("stake",):
            config[key] = int(raw_value)
        else:
            config[key] = raw_value
    return config


def _config_from_record(record: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
    """Prefer structured JSONL config, falling back to older question text."""
    if isinstance(record.get("config"), dict):
        config = dict(record["config"])
        strategy_name = (
            record.get("strategy_name")
            or config.pop("strategy_name", None)
            or DEFAULT_STRATEGY_NAME
        )
        if "strategy_name" in config:
            config.pop("strategy_name", None)
        config.pop("difficulty_levels", None)
        return config, strategy_name

    config = _parse_config_from_question(record["question"])
    strategy_name = (
        record.get("strategy_name")
        or config.pop("strategy_name", None)
        or DEFAULT_STRATEGY_NAME
    )
    if "strategy_params" not in config and isinstance(record.get("strategy_params"), dict):
        config["strategy_params"] = dict(record["strategy_params"])
    return config, strategy_name


def _parse_options(question_text: str) -> Dict[str, str]:
    """Extract the A/B/C/D option map from the question text."""
    options = {}
    for match in re.finditer(r"^([A-D])\.\s+(.+)$", question_text, re.MULTILINE):
        options[match.group(1)] = match.group(2).strip()
    return options


def _extract_question_body(question_text: str) -> str:
    """Get the actual question line (after the instruction, before options)."""
    lines = question_text.split("\n")
    for i, line in enumerate(lines):
        if line.startswith("A. "):
            parts = []
            found_text = False
            for j in range(i - 1, -1, -1):
                stripped = lines[j].strip()
                if not stripped:
                    if found_text:
                        break
                    continue
                if stripped.startswith("-") or stripped.startswith("Configuration"):
                    break
                if (
                    stripped.startswith("Use the same config")
                    or stripped.startswith("Use the config")
                    or stripped.startswith("You are answering")
                ):
                    break
                parts.append(stripped)
                found_text = True
            return " ".join(reversed(parts))
    return ""


# ---------------------------------------------------------------------------
# Ground-truth computers — one per question type
# ---------------------------------------------------------------------------

def _check_first_buy(df, options: Dict[str, str]) -> str:
    buys = df[df["buy_shares"] > 0].sort_values("date")
    first = buys.iloc[0]
    truth = f"{int(first['buy_shares'])} shares at close ${first['close']:.2f} on {first['date']}"
    return _find_matching_letter(options, truth)


def _check_buy_sell_signals(df, options: Dict[str, str]) -> str:
    buy_sig = int(df["buy_signals"].iloc[-1])
    sell_sig = int(df["sell_signals"].iloc[-1])
    truth = f"{buy_sig} buys and {sell_sig} sells"
    return _find_matching_letter(options, truth)


def _check_crossovers(df, options: Dict[str, str]) -> str:
    truth = str(int(df["crossovers"].iloc[-1]))
    return _find_matching_letter(options, truth)


def _check_trade_outcomes(df, options: Dict[str, str]) -> str:
    w = int(df["profitable_closed_trades"].iloc[-1])
    l = int(df["losing_closed_trades"].iloc[-1])
    truth = f"{w} profitable and {l} losing"
    return _find_matching_letter(options, truth)


def _check_net_pnl(df, options: Dict[str, str]) -> str:
    truth = f"${float(df['net_pnl'].iloc[-1]):.2f}"
    return _find_matching_letter(options, truth)


def _check_commission(df, options: Dict[str, str]) -> str:
    total = float(df["total_commission_fee"].iloc[-1])
    orders = int(df["executed_orders"].iloc[-1])
    truth = f"${total:.2f} over {orders} orders"
    return _find_matching_letter(options, truth)


def _check_max_shares(df, options: Dict[str, str]) -> str:
    truth = str(int(df["max_initial_shares"].iloc[-1]))
    return _find_matching_letter(options, truth)


def _check_sma(df, question_body: str, options: Dict[str, str]) -> str:
    m = re.search(r"SMA\((\d+)\) on (\S+)", question_body)
    period = int(m.group(1))
    date = m.group(2).rstrip("?")
    fast_period = int(df["sma_fast_period"].iloc[-1])
    col = "sma_fast" if period == fast_period else "sma_slow"
    row = df[df["date"] == date]
    if row.empty:
        return "?"
    truth = f"{float(row[col].iloc[0]):.2f}"
    return _find_matching_letter(options, truth)


def _check_volume(df, question_body: str, options: Dict[str, str]) -> str:
    m = re.search(r"volume on (\S+)", question_body)
    date = m.group(1).rstrip("?")
    row = df[df["date"] == date]
    if row.empty:
        return "?"
    truth = str(int(round(float(row["volume"].iloc[0]))))
    return _find_matching_letter(options, truth)


def _check_broker_cash(df, question_body: str, options: Dict[str, str]) -> str:
    m = re.search(r"broker cash on (\S+)", question_body)
    date = m.group(1).rstrip("?")
    row = df[df["date"] == date]
    if row.empty:
        return "?"
    truth = f"${float(row['broker_cash'].iloc[0]):.2f}"
    return _find_matching_letter(options, truth)


def _check_portfolio_value(df, question_body: str, options: Dict[str, str]) -> str:
    m = re.search(r"portfolio value on (\S+)", question_body)
    date = m.group(1).rstrip("?")
    row = df[df["date"] == date]
    if row.empty:
        return "?"
    truth = f"${float(row['portfolio_value'].iloc[0]):.2f}"
    return _find_matching_letter(options, truth)


def _check_first_close(df, options: Dict[str, str]) -> str:
    truth = f"${float(df['first_close_price'].iloc[-1]):.2f}"
    return _find_matching_letter(options, truth)


def _check_sma_comparison(
    df, config: Dict, strategy_name: str, question_body: str, options: Dict[str, str],
) -> str:
    m = re.search(
        r"\(pfast=(\d+), pslow=(\d+)\) and \(pfast=(\d+), pslow=(\d+)\)",
        question_body,
    )
    base_fast, base_slow = int(m.group(1)), int(m.group(2))
    alt_fast, alt_slow = int(m.group(3)), int(m.group(4))

    base_pnl = float(df["net_pnl"].iloc[-1])

    alt_config = dict(config)
    alt_config["strategy_params"] = {
        **dict(config.get("strategy_params") or {}),
        "pfast": alt_fast,
        "pslow": alt_slow,
    }
    alt_df = run_backtrader_info(
        config=alt_config, strategy_name=strategy_name, plot=False,
    )
    alt_pnl = float(alt_df["net_pnl"].iloc[-1])

    base_label = f"(pfast={base_fast}, pslow={base_slow})"
    alt_label = f"(pfast={alt_fast}, pslow={alt_slow})"

    if base_pnl >= alt_pnl:
        truth = f"{base_label} with net P/L ${base_pnl:.2f}"
    else:
        truth = f"{alt_label} with net P/L ${alt_pnl:.2f}"

    return _find_matching_letter(options, truth)


def _check_max_drawdown(df, options: Dict[str, str]) -> str:
    pv = df["portfolio_value"].astype(float)
    running_peak = pv.cummax()
    drawdown = running_peak - pv
    max_dd = float(drawdown.max())
    truth = f"${max_dd:.2f}"
    return _find_matching_letter(options, truth)


def _check_peak_portfolio_date(df, options: Dict[str, str]) -> str:
    idx = df["portfolio_value"].astype(float).idxmax()
    peak_row = df.loc[idx]
    truth = f"{peak_row['date']} at ${float(peak_row['portfolio_value']):.2f}"
    return _find_matching_letter(options, truth)


def _check_roi_percentage(df, options: Dict[str, str]) -> str:
    net_pnl = float(df["net_pnl"].iloc[-1])
    initial_cash = float(df["initial_cash"].iloc[-1])
    roi = round(net_pnl / initial_cash * 100.0, 2)
    truth = f"{roi:.2f}%"
    return _find_matching_letter(options, truth)


def _check_total_shares_traded(df, options: Dict[str, str]) -> str:
    total_bought = int(df["buy_shares"].sum()) if "buy_shares" in df.columns else 0
    total_sold = int(df["sell_shares"].sum()) if "sell_shares" in df.columns else 0
    total = total_bought + total_sold
    truth = f"{total_bought} bought and {total_sold} sold ({total} total)"
    return _find_matching_letter(options, truth)


def _check_win_rate(df, options: Dict[str, str]) -> str:
    w = int(df["profitable_closed_trades"].iloc[-1])
    l = int(df["losing_closed_trades"].iloc[-1])
    total = w + l
    win_rate = round(w / total * 100.0, 2)
    truth = f"{win_rate:.2f}%"
    return _find_matching_letter(options, truth)


def _check_best_trade_day(df, options: Dict[str, str]) -> str:
    trade_rows = df[df["closed_trade_pnl"] != 0].copy() if "closed_trade_pnl" in df.columns else None
    if trade_rows is None or trade_rows.empty:
        return "?"
    idx = trade_rows["closed_trade_pnl"].idxmax()
    best_row = trade_rows.loc[idx]
    truth = f"{best_row['date']} with P/L ${float(best_row['closed_trade_pnl']):.2f}"
    return _find_matching_letter(options, truth)


def _check_profit_factor(df, options: Dict[str, str]) -> str:
    if "closed_trade_pnl" not in df.columns:
        return "?"
    pnls = df["closed_trade_pnl"]
    gross_profit = float(pnls[pnls > 0].sum())
    gross_loss = float(pnls[pnls < 0].sum())
    if abs(gross_loss) < 1e-9:
        return "?"
    pf = round(gross_profit / abs(gross_loss), 2)
    truth = f"{pf:.2f}"
    return _find_matching_letter(options, truth)


def _check_days_in_position(df, options: Dict[str, str]) -> str:
    in_pos = int((df["position_size"] > 0).sum())
    total = len(df)
    out_pos = total - in_pos
    truth = f"{in_pos} days in position, {out_pos} days out ({total} total trading days)"
    return _find_matching_letter(options, truth)


def _check_annualized_return(df, question_body: str, options: Dict[str, str]) -> str:
    m = re.search(r"(\d+) trading-day", question_body)
    n_days = int(m.group(1)) if m else len(df)
    initial_cash = float(df["initial_cash"].iloc[-1])
    net_pnl = float(df["net_pnl"].iloc[-1])
    final = net_pnl + initial_cash
    ann_return = round(((final / initial_cash) ** (365.0 / n_days) - 1) * 100.0, 2)
    truth = f"{ann_return:.2f}%"
    return _find_matching_letter(options, truth)


# ---------------------------------------------------------------------------
# Strategy-dependent checkers (require running Backtrader code)
# ---------------------------------------------------------------------------

def _check_position_on_date(df, question_body: str, options: Dict[str, str]) -> str:
    m = re.search(r"close of (\S+)\?", question_body)
    if not m:
        return "?"
    date = m.group(1).rstrip("?")
    row = df[df["date"] == date]
    if row.empty:
        return "?"
    truth = str(int(row["position_size"].iloc[0]))
    return _find_matching_letter(options, truth)


def _check_strategy_action_on_date(df, question_body: str, options: Dict[str, str]) -> str:
    m = re.search(r"on (\d{4}-\d{2}-\d{2})\?", question_body)
    if not m:
        return "?"
    date = m.group(1)
    row = df[df["date"] == date]
    if row.empty:
        return "?"
    r = row.iloc[0]
    bought = int(r.get("buy_shares", 0)) > 0
    sold = int(r.get("sell_shares", 0)) > 0
    if bought:
        shares = int(r["buy_shares"])
        truth_prefix = f"Buy — {shares} shares purchased"
    elif sold:
        shares = int(r["sell_shares"])
        truth_prefix = f"Sell — {shares} shares sold"
    else:
        truth_prefix = "No action"

    for letter, text in options.items():
        if text.startswith(truth_prefix):
            return letter
    return "?"


def _check_nth_trade_pnl(df, question_body: str, options: Dict[str, str]) -> str:
    m = re.search(r"closed on (\S+)\)", question_body)
    if not m:
        return "?"
    date = m.group(1).rstrip(")?")
    row = df[df["date"] == date]
    if row.empty or "closed_trade_pnl" not in df.columns:
        return "?"
    pnl = float(row["closed_trade_pnl"].iloc[0])
    truth = f"${pnl:.2f}"
    return _find_matching_letter(options, truth)


def _check_cash_after_nth_order(df, question_body: str, options: Dict[str, str]) -> str:
    m = re.search(r"\((\d{4}-\d{2}-\d{2})\)\?", question_body)
    if not m:
        return "?"
    date = m.group(1)
    row = df[df["date"] == date]
    if row.empty:
        return "?"
    truth = f"${float(row['broker_cash'].iloc[0]):.2f}"
    return _find_matching_letter(options, truth)


def _check_signal_diff(
    df, config: Dict, strategy_name: str, question_body: str, options: Dict[str, str],
) -> str:
    m = re.search(
        r"\(pfast=(\d+), pslow=(\d+)\).*?\(pfast=(\d+), pslow=(\d+)\)",
        question_body,
    )
    if not m:
        return "?"
    base_f, base_s = int(m.group(1)), int(m.group(2))
    alt_f, alt_s = int(m.group(3)), int(m.group(4))

    base_signals = int(df["buy_signals"].iloc[-1])

    alt_config = dict(config)
    alt_config["strategy_params"] = {
        **dict(config.get("strategy_params") or {}),
        "pfast": alt_f,
        "pslow": alt_s,
    }
    alt_df = run_backtrader_info(config=alt_config, strategy_name=strategy_name, plot=False)
    alt_signals = int(alt_df["buy_signals"].iloc[-1])

    diff = abs(base_signals - alt_signals)
    truth = f"{base_signals} vs {alt_signals} (difference of {diff})"
    return _find_matching_letter(options, truth)


def _check_avg_holding_period(df, question_body: str, options: Dict[str, str]) -> str:
    m = re.search(r"(\d+) total days in position across (\d+) closed trades", question_body)
    if not m:
        return "?"
    days_in = int(m.group(1))
    total_closed = int(m.group(2))
    if total_closed == 0:
        return "?"
    avg = round(days_in / total_closed, 2)
    truth = f"{avg:.2f}"
    return _find_matching_letter(options, truth)


def _check_golden_cross_count(
    df, question_body: str, options: Dict[str, str],
) -> str:
    m = re.search(r"Between (\S+) and (\S+)", question_body)
    if not m:
        return "?"
    start_date, end_date = m.group(1).rstrip(","), m.group(2).rstrip(",")

    diff = df["sma_fast"] - df["sma_slow"]
    prev_diff = diff.shift(1)
    golden = (prev_diff <= 0) & (diff > 0)

    mask = (df["date"] >= start_date) & (df["date"] <= end_date)
    count = int(golden[mask].sum())
    truth = str(count)
    return _find_matching_letter(options, truth)


def _max_drawdown_window_values(df) -> Dict[str, Any]:
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
        "drawdown_dollars": drawdown_dollars,
        "drawdown_pct": drawdown_pct,
    }


def _check_max_drawdown_window(df, options: Dict[str, str]) -> str:
    window = _max_drawdown_window_values(df)
    truth = (
        f"{window['peak_date']} to {window['trough_date']}: "
        f"${window['drawdown_dollars']:.2f} ({window['drawdown_pct']:.2f}%)"
    )
    return _find_matching_letter(options, truth)


def _check_drawdown_recovery_days(df, options: Dict[str, str]) -> str:
    window = _max_drawdown_window_values(df)
    after_trough = df.loc[window["trough_idx"] + 1:].copy()
    recovered = after_trough[
        after_trough["portfolio_value"].astype(float) >= window["peak_value"]
    ]
    if recovered.empty:
        truth = f"Not recovered by end of backtest after {window['trough_date']}"
    else:
        recovery_idx = recovered.index[0]
        recovery_date = df.loc[recovery_idx, "date"]
        recovery_days = int(recovery_idx - window["trough_idx"])
        truth = (
            f"{recovery_days} trading days after {window['trough_date']} "
            f"(recovered on {recovery_date})"
        )
    return _find_matching_letter(options, truth)


def _check_calmar_ratio(df, options: Dict[str, str]) -> str:
    window = _max_drawdown_window_values(df)
    if window["drawdown_pct"] <= 0:
        return "?"
    initial_cash = float(df["initial_cash"].iloc[-1])
    final = float(df["net_pnl"].iloc[-1]) + initial_cash
    annualized_pct = ((final / initial_cash) ** (365.0 / len(df)) - 1) * 100.0
    ratio = round(annualized_pct / window["drawdown_pct"], 2)
    truth = f"{ratio:.2f}"
    return _find_matching_letter(options, truth)


def _check_exposure_adjusted_return(df, options: Dict[str, str]) -> str:
    exposure_pct = (df["position_size"].astype(float) > 0).mean() * 100.0
    if exposure_pct <= 0:
        return "?"
    initial_cash = float(df["initial_cash"].iloc[-1])
    roi_pct = float(df["net_pnl"].iloc[-1]) / initial_cash * 100.0
    adjusted = round(roi_pct / exposure_pct, 2)
    truth = f"{adjusted:.2f}"
    return _find_matching_letter(options, truth)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_matching_letter(options: Dict[str, str], truth: str) -> str:
    for letter, text in options.items():
        if text == truth:
            return letter
    return "?"


def _classify_question(question_body: str) -> str:
    q = question_body.lower()
    if "first buy execution" in q:
        return "first_buy"
    if "buy and sell signals" in q:
        return "buy_sell_signals"
    if "crossover events" in q:
        return "crossovers"
    if "profitable and losing" in q:
        return "trade_outcomes"
    if "net profit/loss" in q:
        return "net_pnl"
    if "total commission fee" in q:
        return "commission"
    if "maximum number of shares" in q:
        return "max_shares"
    if re.search(r"SMA\(\d+\) on", q, re.IGNORECASE):
        return "sma"
    if "volume on" in q:
        return "volume"
    if "broker cash on" in q:
        return "broker_cash"
    if "portfolio value on" in q and "highest" not in q:
        return "portfolio_value"
    if "first close price" in q:
        return "first_close"
    if "which sma parameter" in q:
        return "sma_comparison"
    if "maximum portfolio drawdown" in q or "peak-to-trough" in q:
        return "max_drawdown"
    if "highest value" in q and "portfolio" in q:
        return "peak_portfolio_date"
    if "return on investment" in q or "roi" in q:
        return "roi_percentage"
    if "total shares" in q and "traded" in q:
        return "total_shares_traded"
    if "win rate" in q:
        return "win_rate"
    if "most profitable closed trade" in q:
        return "best_trade_day"
    if "profit factor" in q:
        return "profit_factor"
    if "days" in q and "in a position" in q:
        return "days_in_position"
    if "annualized return" in q:
        return "annualized_return"
    if "how many shares does" in q and "hold at the close" in q:
        return "position_on_date"
    if "what action did the strategy take on" in q:
        return "strategy_action_on_date"
    if "realized net p/l" in q and "closed trade" in q:
        return "nth_trade_pnl"
    if "broker" in q and "cash" in q and "order execution day" in q:
        return "cash_after_nth_order"
    if "how many buy signals does each produce" in q:
        return "signal_diff"
    if "average number of trading days" in q and "held a position" in q:
        return "avg_holding_period"
    if "golden cross" in q and ("how many" in q or "occurred" in q):
        return "golden_cross_count"
    if "maximum drawdown window" in q or "which peak-to-trough interval" in q:
        return "max_drawdown_window"
    if "recover to the prior peak value" in q:
        return "drawdown_recovery_days"
    if "calmar-style ratio" in q:
        return "calmar_ratio"
    if "exposure-adjusted roi" in q:
        return "exposure_adjusted_return"
    return "unknown"


# ---------------------------------------------------------------------------
# Main checker
# ---------------------------------------------------------------------------

def check_mcq_file(filepath: str) -> List[Dict[str, Any]]:
    with open(filepath, "r", encoding="utf-8") as f:
        raw = f.read().strip()

    if not raw:
        records = []
    elif raw[0] == "[":
        records = json.loads(raw)
    else:
        records = [json.loads(line) for line in raw.splitlines() if line.strip()]

    if not records:
        print("No questions found.")
        return []

    grouped_records: Dict[str, Dict[str, Any]] = {}
    for i, record in enumerate(records, 1):
        config, strategy_name = _config_from_record(record)
        group_key = json.dumps(
            {"config": config, "strategy_name": strategy_name},
            sort_keys=True,
            default=str,
        )
        if group_key not in grouped_records:
            grouped_records[group_key] = {
                "config": config,
                "strategy_name": strategy_name,
                "items": [],
            }
        grouped_records[group_key]["items"].append((i, record))

    print(f"Found {len(grouped_records)} backtest group(s).")

    results = []
    pass_count = 0
    fail_count = 0

    # Per-difficulty tallies. Includes "unknown" as a safety bucket.
    pass_by_diff: Dict[str, int] = defaultdict(int)
    total_by_diff: Dict[str, int] = defaultdict(int)

    for group_number, group in enumerate(grouped_records.values(), 1):
        config = group["config"]
        strategy_name = group["strategy_name"]
        group_items = group["items"]

        print(f"\nBacktest group {group_number}/{len(grouped_records)}")
        print(f"Parsed config: {config}")
        print(f"Strategy: {strategy_name}")
        print(f"Questions: {len(group_items)}")
        print("Running backtest...")

        df = run_backtrader_info(config=config, strategy_name=strategy_name, plot=False)
        df = df.sort_values("date").reset_index(drop=True)
        print(f"Backtest complete: {len(df)} rows\n")

        for i, record in group_items:
            question_text = record["question"]
            stated_answer = record["answer"]
            options = _parse_options(question_text)
            question_body = _extract_question_body(question_text)

            # Prefer the qtype/difficulty tagged by the builder; fall back to
            # inferring from the question text so older files still work.
            qtype = record.get("type") or _classify_question(question_body)
            difficulty = (
                record.get("difficulty")
                or DIFFICULTY_BY_TYPE.get(qtype, "unknown")
            )

            computed_answer = "?"

            if qtype == "first_buy":
                computed_answer = _check_first_buy(df, options)
            elif qtype == "buy_sell_signals":
                computed_answer = _check_buy_sell_signals(df, options)
            elif qtype == "crossovers":
                computed_answer = _check_crossovers(df, options)
            elif qtype == "trade_outcomes":
                computed_answer = _check_trade_outcomes(df, options)
            elif qtype == "net_pnl":
                computed_answer = _check_net_pnl(df, options)
            elif qtype == "commission":
                computed_answer = _check_commission(df, options)
            elif qtype == "max_shares":
                computed_answer = _check_max_shares(df, options)
            elif qtype == "sma":
                computed_answer = _check_sma(df, question_body, options)
            elif qtype == "volume":
                computed_answer = _check_volume(df, question_body, options)
            elif qtype == "broker_cash":
                computed_answer = _check_broker_cash(df, question_body, options)
            elif qtype == "portfolio_value":
                computed_answer = _check_portfolio_value(df, question_body, options)
            elif qtype == "first_close":
                computed_answer = _check_first_close(df, options)
            elif qtype == "sma_comparison":
                computed_answer = _check_sma_comparison(df, config, strategy_name, question_body, options)
            elif qtype == "max_drawdown":
                computed_answer = _check_max_drawdown(df, options)
            elif qtype == "peak_portfolio_date":
                computed_answer = _check_peak_portfolio_date(df, options)
            elif qtype == "roi_percentage":
                computed_answer = _check_roi_percentage(df, options)
            elif qtype == "total_shares_traded":
                computed_answer = _check_total_shares_traded(df, options)
            elif qtype == "win_rate":
                computed_answer = _check_win_rate(df, options)
            elif qtype == "best_trade_day":
                computed_answer = _check_best_trade_day(df, options)
            elif qtype == "profit_factor":
                computed_answer = _check_profit_factor(df, options)
            elif qtype == "days_in_position":
                computed_answer = _check_days_in_position(df, options)
            elif qtype == "annualized_return":
                computed_answer = _check_annualized_return(df, question_body, options)
            elif qtype == "position_on_date":
                computed_answer = _check_position_on_date(df, question_body, options)
            elif qtype == "strategy_action_on_date":
                computed_answer = _check_strategy_action_on_date(df, question_body, options)
            elif qtype == "nth_trade_pnl":
                computed_answer = _check_nth_trade_pnl(df, question_body, options)
            elif qtype == "cash_after_nth_order":
                computed_answer = _check_cash_after_nth_order(df, question_body, options)
            elif qtype == "signal_diff":
                computed_answer = _check_signal_diff(df, config, strategy_name, question_body, options)
            elif qtype == "avg_holding_period":
                computed_answer = _check_avg_holding_period(df, question_body, options)
            elif qtype == "golden_cross_count":
                computed_answer = _check_golden_cross_count(df, question_body, options)
            elif qtype == "max_drawdown_window":
                computed_answer = _check_max_drawdown_window(df, options)
            elif qtype == "drawdown_recovery_days":
                computed_answer = _check_drawdown_recovery_days(df, options)
            elif qtype == "calmar_ratio":
                computed_answer = _check_calmar_ratio(df, options)
            elif qtype == "exposure_adjusted_return":
                computed_answer = _check_exposure_adjusted_return(df, options)

            match = computed_answer == stated_answer
            status = "PASS" if match else "FAIL"
            if match:
                pass_count += 1
                pass_by_diff[difficulty] += 1
            else:
                fail_count += 1
            total_by_diff[difficulty] += 1

            short_q = question_body[:70] + ("..." if len(question_body) > 70 else "")
            print(
                f"  Q{i:02d} [{status}] [{difficulty}] ({qtype}) "
                f"stated={stated_answer} computed={computed_answer}  {short_q}"
            )

            results.append({
                "index": i,
                "type": qtype,
                "difficulty": difficulty,
                "stated_answer": stated_answer,
                "computed_answer": computed_answer,
                "match": match,
                "question": question_body,
                "strategy_name": strategy_name,
            })

    print(f"\n{'='*60}")
    # Per-difficulty breakdown when the set actually spans multiple levels.
    present_levels = [l for l in DIFFICULTY_LEVELS if total_by_diff.get(l, 0) > 0]
    other_levels = [l for l in total_by_diff if l not in DIFFICULTY_LEVELS]
    ordered_levels = present_levels + other_levels

    if len(ordered_levels) > 1:
        print("Accuracy by difficulty:")
        for lvl in ordered_levels:
            tot = total_by_diff[lvl]
            passed = pass_by_diff.get(lvl, 0)
            pct = (passed / tot * 100.0) if tot else 0.0
            print(f"  {lvl:<8s}: {passed}/{tot} ({pct:.2f}%)")
    elif len(ordered_levels) == 1:
        lvl = ordered_levels[0]
        print(f"Difficulty: {lvl} only")

    total = len(records)
    overall_pct = (pass_count / total * 100.0) if total else 0.0
    print(f"Overall  : {pass_count}/{total} ({overall_pct:.2f}%) — "
          f"{pass_count} PASS, {fail_count} FAIL")

    if fail_count == 0:
        print("\nAll ground-truth answers verified.")
    else:
        print("\nGround-truth mismatches:")
        for r in results:
            if not r["match"]:
                print(f"  Q{r['index']:02d} [{r['difficulty']}] "
                      f"({r['type']}): stated={r['stated_answer']} "
                      f"computed={r['computed_answer']} — {r['question'][:80]}")

    return results


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <mcq_file.json>")
        sys.exit(1)
    check_mcq_file(sys.argv[1])
