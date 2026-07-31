"""Registry of Backtrader strategies available to the MCQ pipeline.

Some strategies were inspired by Zipline examples, but the registered MCQ
defaults use more realistic Backtrader behavior: signals are computed from
known bar data and orders execute on the next bar unless a validation script
explicitly opts into Zipline-parity execution.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Type

import backtrader as bt


# ---------------------------------------------------------------------------
# Shared MCQ instrumentation (orders, trades, daily rows)
# ---------------------------------------------------------------------------


class McqInstrumentedStrategy(bt.Strategy):
    """Order/trade logging + daily OHLCV/SMA rows expected by ``backtrader_info``."""

    params = dict(mcq_sma_fast_period=10, mcq_sma_slow_period=30)

    def _mcq_init_state(self) -> None:
        self.order = None
        self._pending_target_percent = None
        self.buy_signal_count = 0
        self.sell_signal_count = 0
        self.crossover_count = 0
        self.buy_executions = []
        self.sell_executions = []
        self.closed_trades = []
        self.total_commission = 0.0
        self.executed_order_count = 0
        self.daily_records = []

    def _init_mcq_smas(self, fast_period: int | None = None, slow_period: int | None = None) -> None:
        self.mcq_sma_fast_period = max(
            1,
            int(fast_period if fast_period is not None else self.p.mcq_sma_fast_period),
        )
        self.mcq_sma_slow_period = max(
            1,
            int(slow_period if slow_period is not None else self.p.mcq_sma_slow_period),
        )
        self.mcq_sma_fast = bt.ind.SMA(self.data.close, period=self.mcq_sma_fast_period)
        self.mcq_sma_slow = bt.ind.SMA(self.data.close, period=self.mcq_sma_slow_period)

    def _mcq_append_day(self) -> None:
        self.daily_records.append({
            "date": self.datas[0].datetime.date(0).isoformat(),
            "close": float(self.data.close[0]),
            "volume": float(self.data.volume[0]),
            "sma_fast": float(self.mcq_sma_fast[0]),
            "sma_slow": float(self.mcq_sma_slow[0]),
            "broker_cash": float(self.broker.getcash()),
            "portfolio_value": float(self.broker.getvalue()),
            "position_size": int(self.position.size),
        })

    def _mcq_order_target_percent(self, target: float):
        if getattr(self.p, "execute_on_open", False):
            # Opt-in path used by the Zipline comparison script only.
            self._pending_target_percent = float(target)
            return None

        if target <= 0.0:
            return self.order_target_percent(target=0.0)

        # Realistic MCQ path: leave a cash buffer so next-bar market orders are
        # less likely to be rejected if the next open moves against the signal.
        cash = float(self.broker.getcash())
        price = float(self.data.close[0])
        size = int((cash * min(target, 1.0) * 0.95) / price) if price > 0 else 0
        if size <= 0:
            return None
        return self.buy(size=size)

    def _mcq_execute_pending_target_at_open(self):
        if self._pending_target_percent is None or self.order:
            return

        target = self._pending_target_percent
        self._pending_target_percent = None
        open_price = float(self.data.open[0])
        if open_price <= 0:
            return

        position_size = int(self.position.size)
        if target <= 0.0:
            if position_size:
                self.order = self.sell(size=abs(position_size))
            return

        cash = float(self.broker.getcash())
        portfolio_value = cash + position_size * open_price
        target_size = int(portfolio_value * min(target, 1.0) / open_price)
        delta_size = target_size - position_size
        if delta_size <= 0:
            return

        comminfo = self.broker.getcommissioninfo(self.data)
        while delta_size > 0:
            cost = comminfo.getoperationcost(delta_size, open_price)
            commission = comminfo.getcommission(delta_size, open_price)
            if cost + commission <= cash:
                break
            delta_size -= 1

        if delta_size > 0:
            self.order = self.buy(size=delta_size)

    def next_open(self):
        self._mcq_execute_pending_target_at_open()

    def prenext_open(self):
        self._mcq_execute_pending_target_at_open()

    def nextstart_open(self):
        self._mcq_execute_pending_target_at_open()

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status == order.Completed:
            dt = bt.num2date(order.executed.dt).date()
            commission = float(order.executed.comm)
            execution = {
                "date": dt.isoformat(),
                "size": int(abs(order.executed.size)),
                "price": float(order.executed.price),
                "commission": commission,
            }
            if order.isbuy():
                self.buy_executions.append(execution)
            else:
                self.sell_executions.append(execution)

            self.total_commission += commission
            self.executed_order_count += 1

        self.order = None

    def notify_trade(self, trade):
        if trade.isclosed:
            self.closed_trades.append({
                "open_date": bt.num2date(trade.dtopen).date().isoformat(),
                "close_date": bt.num2date(trade.dtclose).date().isoformat(),
                "pnl_net": float(trade.pnlcomm),
            })


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------


class SmaCross(McqInstrumentedStrategy):
    """Classic dual simple-moving-average crossover strategy."""

    params = dict(pfast=10, pslow=30, target_percent=1.0)

    def __init__(self):
        self._mcq_init_state()
        self._init_mcq_smas(self.p.pfast, self.p.pslow)
        self.sma_fast = self.mcq_sma_fast
        self.sma_slow = self.mcq_sma_slow
        self.crossover = bt.ind.CrossOver(self.sma_fast, self.sma_slow)

    def next(self):
        if not self.order:
            if not self.position:
                if self.crossover > 0:
                    self.buy_signal_count += 1
                    self.crossover_count += 1
                    self.order = self._mcq_order_target_percent(float(self.p.target_percent))
            elif self.crossover < 0:
                self.sell_signal_count += 1
                self.crossover_count += 1
                self.order = self._mcq_order_target_percent(0.0)

        self._mcq_append_day()


class RollingWindowMean(McqInstrumentedStrategy):
    """Invest when close is above its rolling-window mean, otherwise hold cash."""

    params = dict(
        rolling_window=90,
        target_percent=1.0,
        execute_on_open=False,
    )

    def __init__(self):
        self._mcq_init_state()
        self._init_mcq_smas()
        rolling_period = max(1, int(self.p.rolling_window))
        self.rolling_mean = bt.ind.SMA(self.data.close, period=rolling_period)

    def next(self):
        if self.order:
            self._mcq_append_day()
            return

        target = (
            float(self.p.target_percent)
            if self.data.close[0] > self.rolling_mean[0]
            else 0.0
        )

        if target > 0.0 and not self.position:
            self.buy_signal_count += 1
            self.crossover_count += 1
            self.order = self._mcq_order_target_percent(target)
        elif target == 0.0 and self.position:
            self.sell_signal_count += 1
            self.crossover_count += 1
            self.order = self._mcq_order_target_percent(0.0)

        self._mcq_append_day()


class ExponentialWeightedMovingAverage(McqInstrumentedStrategy):
    """Invest when short-span EWM is above long-span EWM, otherwise hold cash."""

    params = dict(
        pfast=5,
        pslow=30,
        target_percent=1.0,
        execute_on_open=False,
    )

    def __init__(self):
        self._mcq_init_state()
        self._init_mcq_smas(self.p.pfast, self.p.pslow)
        self.ema_fast = bt.ind.ExponentialMovingAverage(
            self.data.close,
            period=max(1, int(self.p.pfast)),
        )
        self.ema_slow = bt.ind.ExponentialMovingAverage(
            self.data.close,
            period=max(1, int(self.p.pslow)),
        )

    def next(self):
        if self.order:
            self._mcq_append_day()
            return

        if self.ema_fast[0] > self.ema_slow[0] and not self.position:
            self.buy_signal_count += 1
            self.crossover_count += 1
            self.order = self._mcq_order_target_percent(float(self.p.target_percent))
        elif self.ema_fast[0] < self.ema_slow[0] and self.position:
            self.sell_signal_count += 1
            self.crossover_count += 1
            self.order = self._mcq_order_target_percent(0.0)

        self._mcq_append_day()


class RsiStrategy(McqInstrumentedStrategy):
    """Buy when RSI is deeply oversold and sell when RSI is deeply overbought."""

    params = dict(
        mcq_sma_fast_period=12,
        mcq_sma_slow_period=20,
        rsi_period=12,
        rsi_buy_threshold=10,
        rsi_sell_threshold=90,
        target_percent=1.0,
        execute_on_open=False,
    )

    def __init__(self):
        self._mcq_init_state()
        self._init_mcq_smas()
        self.rsi = bt.ind.RSI_Safe(
            self.data.close,
            period=max(1, int(self.p.rsi_period)),
        )

    def next(self):
        if self.order:
            self._mcq_append_day()
            return

        rsi_value = self.rsi[0]
        if rsi_value > float(self.p.rsi_sell_threshold) and self.position:
            self.sell_signal_count += 1
            self.crossover_count += 1
            self.order = self._mcq_order_target_percent(0.0)
        elif rsi_value < float(self.p.rsi_buy_threshold) and not self.position:
            self.buy_signal_count += 1
            self.crossover_count += 1
            self.order = self._mcq_order_target_percent(float(self.p.target_percent))

        self._mcq_append_day()


class MacdCrossoverStrategy(McqInstrumentedStrategy):
    """Buy on MACD crossing above signal and sell on crossing below signal."""

    params = dict(
        pfast=12,
        pslow=26,
        psignal=9,
        target_percent=1.0,
        execute_on_open=False,
    )

    def __init__(self):
        self._mcq_init_state()
        self._init_mcq_smas(self.p.pfast, self.p.pslow)
        self.macd = bt.ind.MACD(
            self.data.close,
            period_me1=max(1, int(self.p.pfast)),
            period_me2=max(1, int(self.p.pslow)),
            period_signal=max(1, int(self.p.psignal)),
        )
        self.macd_cross = bt.ind.CrossOver(self.macd.macd, self.macd.signal)

    def next(self):
        if self.order:
            self._mcq_append_day()
            return

        if self.macd_cross > 0 and not self.position:
            self.buy_signal_count += 1
            self.crossover_count += 1
            self.order = self._mcq_order_target_percent(float(self.p.target_percent))
        elif self.macd_cross < 0 and self.position:
            self.sell_signal_count += 1
            self.crossover_count += 1
            self.order = self._mcq_order_target_percent(0.0)

        self._mcq_append_day()


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


StrategySpec = Dict[str, Any]

STRATEGIES: Dict[str, StrategySpec] = {
    "sma_crossover": {
        "cls": SmaCross,
        "default_params": {"pfast": 10, "pslow": 30, "target_percent": 1.0},
    },
    "rolling_window_mean": {
        "cls": RollingWindowMean,
        "default_params": {
            "rolling_window": 90,
            "target_percent": 1.0,
        },
    },
    "exponential_weighted_moving_average": {
        "cls": ExponentialWeightedMovingAverage,
        "default_params": {
            "pfast": 5,
            "pslow": 30,
            "target_percent": 1.0,
        },
    },
    "rsi_strategy": {
        "cls": RsiStrategy,
        "default_params": {
            "mcq_sma_fast_period": 12,
            "mcq_sma_slow_period": 20,
            "rsi_period": 12,
            "rsi_buy_threshold": 10,
            "rsi_sell_threshold": 90,
            "target_percent": 1.0,
        },
    },
    "macd_crossover": {
        "cls": MacdCrossoverStrategy,
        "default_params": {
            "pfast": 12,
            "pslow": 26,
            "psignal": 9,
            "target_percent": 1.0,
        },
    },
}

DEFAULT_STRATEGY_NAME = "sma_crossover"

STRATEGY_MAP: Dict[str, Type[bt.Strategy]] = {
    name: spec["cls"] for name, spec in STRATEGIES.items()
}

REGISTERED_STRATEGY_NAMES: tuple[str, ...] = tuple(STRATEGIES.keys())


# ---------------------------------------------------------------------------
# Lookup helpers
# ---------------------------------------------------------------------------


def get_strategy_spec(name: str) -> StrategySpec:
    """Return the registry entry for ``name`` or raise ``KeyError``."""
    if name not in STRATEGIES:
        raise KeyError(
            f"Unknown strategy '{name}'. "
            f"Registered: {sorted(STRATEGIES.keys())}"
        )
    return STRATEGIES[name]


def get_strategy_class(name: str) -> Type[bt.Strategy]:
    return get_strategy_spec(name)["cls"]


def get_default_params(name: str) -> Dict[str, Any]:
    """Return a fresh copy of the default params for ``name``."""
    return deepcopy(get_strategy_spec(name)["default_params"])
