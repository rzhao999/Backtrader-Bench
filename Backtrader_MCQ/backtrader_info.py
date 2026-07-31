import json

import backtrader as bt
import pandas as pd
import yfinance as yf

from strategies import (
    DEFAULT_STRATEGY_NAME,
    STRATEGY_MAP,
    SmaCross,
    get_default_params,
    get_strategy_class,
)

# ``SmaCross`` and ``STRATEGY_MAP`` are re-exported above so existing
# external callers (``app.py`` / ``app2.py``) keep working unchanged.

CONFIG = {
    "symbol": "MSFT",
    "start_date": "2011-01-01",
    "end_date": "2013-01-01",
    "commission_rate": 0.001,
    "initial_cash": 50000.0,
    "stake": 1000,
    # strategy_params are merged per strategy_name in run_backtrader_info /
    # build_mcq_dataset — omit here so non-SMA strategies keep their own
    # defaults (pfast/pslow in CONFIG would otherwise override MACD windows).
}


# ---------------------------------------------------------------------------
# Sizer
# ---------------------------------------------------------------------------

class CashCappedFixedSize(bt.Sizer):
    params = (("stake", 1000),)

    def _getsizing(self, comminfo, cash, data, isbuy):
        if not isbuy:
            return self.broker.getposition(data).size

        price = data.close[0]
        if price <= 0:
            return 0

        size = min(int(self.p.stake), int(cash / price))
        while size > 0:
            cost = comminfo.getoperationcost(size, price)
            commission = comminfo.getcommission(size, price)
            if (cost + commission) <= cash:
                break
            size -= 1
        return max(size, 0)


# ---------------------------------------------------------------------------
# Data loading (cached per symbol + date range within a session)
# ---------------------------------------------------------------------------

_price_cache: dict = {}


def load_data(config, *, use_cache=True):
    key = (config["symbol"], config["start_date"], config["end_date"])
    if use_cache and key in _price_cache:
        return _price_cache[key]

    price_df = yf.download(
        config["symbol"],
        start=config["start_date"],
        end=config["end_date"],
        auto_adjust=False,
        progress=False,
    )
    if getattr(price_df.columns, "nlevels", 1) > 1:
        price_df.columns = price_df.columns.get_level_values(0)
    if price_df.empty:
        raise ValueError(f"No data returned for {config['symbol']} from yfinance")

    if use_cache:
        _price_cache[key] = price_df
    return price_df


# ---------------------------------------------------------------------------
# Backtest runner
# ---------------------------------------------------------------------------

def run_backtest(config, price_df):
    strategy_name = config.get("strategy_name", DEFAULT_STRATEGY_NAME)
    try:
        strategy_cls = get_strategy_class(strategy_name)
        strategy_defaults = get_default_params(strategy_name)
    except KeyError as exc:
        raise ValueError(str(exc)) from exc

    strategy_params = {**strategy_defaults, **dict(config.get("strategy_params", {}))}
    # Normal MCQ generation leaves cheat-on-open disabled. The optional
    # ``execute_on_open`` flag is reserved for parity validation scripts.
    cerebro = bt.Cerebro(
        cheat_on_open=bool(strategy_params.get("execute_on_open", False))
    )
    cerebro.broker.setcash(config["initial_cash"])
    cerebro.broker.setcommission(commission=config["commission_rate"])
    cerebro.addsizer(CashCappedFixedSize, stake=config["stake"])
    cerebro.adddata(bt.feeds.PandasData(dataname=price_df))
    cerebro.addstrategy(strategy_cls, **strategy_params)
    strategy = cerebro.run()[0]
    return cerebro, strategy


# ---------------------------------------------------------------------------
# DataFrame builder helpers
# ---------------------------------------------------------------------------

_ZERO_FILL_INT_COLS = [
    "buy_order_count",
    "sell_order_count",
    "buy_shares",
    "sell_shares",
    "closed_trade_count",
    "profitable_closed_trades_day",
    "losing_closed_trades_day",
]

_ZERO_FILL_FLOAT_COLS = [
    "order_commission",
    "closed_trade_pnl",
]


def _build_orders_by_day(strategy):
    buy_df = pd.DataFrame(strategy.buy_executions)
    sell_df = pd.DataFrame(strategy.sell_executions)
    if not buy_df.empty:
        buy_df["side"] = "BUY"
    if not sell_df.empty:
        sell_df["side"] = "SELL"

    orders = pd.concat([buy_df, sell_df], ignore_index=True)
    if orders.empty:
        return pd.DataFrame()

    orders = orders[["date", "side", "size", "price", "commission"]].sort_values("date")
    orders["buy_shares"] = orders["size"].where(orders["side"] == "BUY", 0)
    orders["sell_shares"] = orders["size"].where(orders["side"] == "SELL", 0)
    return orders.groupby("date", as_index=False).agg(
        buy_order_count=("side", lambda s: int((s == "BUY").sum())),
        sell_order_count=("side", lambda s: int((s == "SELL").sum())),
        buy_shares=("buy_shares", "sum"),
        sell_shares=("sell_shares", "sum"),
        order_commission=("commission", "sum"),
    )


def _build_trades_by_day(strategy):
    trades = pd.DataFrame(strategy.closed_trades)
    if trades.empty:
        return pd.DataFrame()

    return (
        trades.groupby("close_date", as_index=False)
        .agg(
            closed_trade_count=("pnl_net", "count"),
            closed_trade_pnl=("pnl_net", "sum"),
            profitable_closed_trades_day=("pnl_net", lambda s: int((s > 0).sum())),
            losing_closed_trades_day=("pnl_net", lambda s: int((s < 0).sum())),
        )
        .rename(columns={"close_date": "date"})
    )


def build_backtest_df(config, strategy, price_df, final_value):
    net_pnl = final_value - config["initial_cash"]
    first_close = float(price_df["Close"].iloc[0])
    max_initial_shares = int(
        config["initial_cash"] / (first_close * (1 + config["commission_rate"]))
    )

    win_trades = 0
    loss_trades = 0
    for trade in strategy.closed_trades:
        if trade["pnl_net"] > 0:
            win_trades += 1
        elif trade["pnl_net"] < 0:
            loss_trades += 1

    backtest_df = pd.DataFrame(strategy.daily_records)

    orders_by_day = _build_orders_by_day(strategy)
    if not orders_by_day.empty:
        backtest_df = backtest_df.merge(orders_by_day, on="date", how="left")

    trades_by_day = _build_trades_by_day(strategy)
    if not trades_by_day.empty:
        backtest_df = backtest_df.merge(trades_by_day, on="date", how="left")

    for col in _ZERO_FILL_INT_COLS:
        if col in backtest_df.columns:
            backtest_df[col] = backtest_df[col].fillna(0).astype(int)
    for col in _ZERO_FILL_FLOAT_COLS:
        if col in backtest_df.columns:
            backtest_df[col] = backtest_df[col].fillna(0.0)

    strategy_params = config.get("strategy_params", {})
    strat_name = config.get("strategy_name", DEFAULT_STRATEGY_NAME)
    strat_defaults = get_default_params(strat_name)
    effective_strategy_params = {**strat_defaults, **dict(strategy_params)}
    sma_fast_period = int(getattr(strategy, "mcq_sma_fast_period"))
    sma_slow_period = int(getattr(strategy, "mcq_sma_slow_period"))

    backtest_df = backtest_df.assign(
        symbol=config["symbol"],
        start_date=config["start_date"],
        end_date=config["end_date"],
        strategy_name=strat_name,
        strategy_params=json.dumps(effective_strategy_params, sort_keys=True),
        initial_cash=config["initial_cash"],
        final_portfolio_value=final_value,
        net_pnl=net_pnl,
        buy_signals=strategy.buy_signal_count,
        sell_signals=strategy.sell_signal_count,
        crossovers=strategy.crossover_count,
        profitable_closed_trades=win_trades,
        losing_closed_trades=loss_trades,
        total_commission_fee=strategy.total_commission,
        executed_orders=strategy.executed_order_count,
        first_close_price=first_close,
        max_initial_shares=max_initial_shares,
        sma_fast_period=sma_fast_period,
        sma_slow_period=sma_slow_period,
    )

    return backtest_df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_backtrader_info(config=None, strategy_name=DEFAULT_STRATEGY_NAME, plot=False):
    runtime_config = dict(CONFIG)
    if config:
        runtime_config.update(config)
    runtime_config["strategy_name"] = strategy_name
    _defaults = get_default_params(strategy_name)
    _user = dict(runtime_config.get("strategy_params") or {})
    runtime_config["strategy_params"] = {**_defaults, **_user}

    price_df = load_data(runtime_config)
    cerebro, strategy = run_backtest(runtime_config, price_df)
    final_value = cerebro.broker.getvalue()
    backtest_df = build_backtest_df(runtime_config, strategy, price_df, final_value)

    if plot:
        figs = cerebro.plot(iplot=False)
        if figs:
            import matplotlib.pyplot as plt
            plt.show()

    return backtest_df


if __name__ == "__main__":
    backtest_df = run_backtrader_info(plot=True)
