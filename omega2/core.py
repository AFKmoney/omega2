"""
OMEGA2 — Core: config, events, logger in one file.

Simplified from omega-hedge-fund's 5 config files + events.py + logger.py
into a single import surface. Same quality, less surface area.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ═══════════════════════════════════════════════════════════════════════
# LOGGER
# ═══════════════════════════════════════════════════════════════════════

class _JsonFormatter(logging.Formatter):
    def format(self, record):
        payload = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "component": getattr(record, "component", "omega2"),
            "msg": record.getMessage(),
        }
        for key in ("symbol", "agent", "regime", "trade_id"):
            val = getattr(record, key, None)
            if val is not None:
                payload[key] = val
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)

_CONFIGURED = {}
def get_logger(name="omega2", level="INFO"):
    if name in _CONFIGURED:
        return _CONFIGURED[name]
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_JsonFormatter())
    logger.addHandler(handler)
    _CONFIGURED[name] = logger
    return logger


# ═══════════════════════════════════════════════════════════════════════
# EVENTS
# ═══════════════════════════════════════════════════════════════════════

class Side(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    FLAT = "FLAT"

class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"

def _uid(prefix):
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


@dataclass(frozen=True)
class MarketEvent:
    symbol: str
    timestamp: str
    last_price: float
    volume_24h: float = 0.0
    bid: float = 0.0
    ask: float = 0.0
    bid_qty: float = 0.0
    ask_qty: float = 0.0
    funding_rate: Optional[float] = None
    source: str = "binance"


@dataclass(frozen=True)
class CrowdSignal:
    """Output of the 4-signal crowd engine fusion."""
    symbol: str
    timestamp: str
    crowd_score: float       # [-1,+1] + = crowd long overcrowded
    conviction: float        # [0,1]
    components: Dict[str, float] = field(default_factory=dict)
    regime_hint: str = "neutral"  # cascade_imminent | euphoria | fear | neutral
    expected_move_bps: float = 0.0


@dataclass(frozen=True)
class Signal:
    """Agent → Risk gate signal."""
    agent: str
    symbol: str
    timestamp: str
    side: Side
    confidence: float
    stop_loss_bps: float = 100.0
    take_profit_bps: float = 200.0
    expected_holding_period_bars: int = 60
    rationale: str = ""
    metadata: Dict = field(default_factory=dict)


@dataclass(frozen=True)
class Order:
    order_id: str = field(default_factory=lambda: _uid("ord"))
    symbol: str = ""
    side: Side = Side.FLAT
    qty: float = 0.0
    order_type: OrderType = OrderType.MARKET
    strategy: str = ""
    metadata: Dict = field(default_factory=dict)


@dataclass(frozen=True)
class Fill:
    order_id: str
    symbol: str
    side: Side
    qty: float
    fill_price: float
    timestamp: str
    slippage_bps: float = 0.0
    fee_paid: float = 0.0


# ═══════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════

def _data_dir():
    d = Path(os.getenv("OMEGA_DATA_DIR", str(Path.home() / ".omega2")))
    d.mkdir(parents=True, exist_ok=True)
    return d


@dataclass
class Config:
    # Venue
    venue: str = "binance"  # auto: okx if OKX creds present
    okx_api_key: str = ""
    okx_api_secret: str = ""
    okx_passphrase: str = ""
    okx_demo: bool = False
    binance_api_key: str = ""
    binance_api_secret: str = ""
    # Symbols
    symbols: tuple = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
    # Risk (THE critical ones — hardened for OMEGA2)
    kelly_fraction: float = 0.25
    max_per_trade_risk_pct: float = 1.0
    max_drawdown_pct: float = 8.0
    max_daily_loss_pct: float = 3.0          # NEW: rolling 24h loss limit
    max_positions: int = 8
    max_per_symbol_notional_pct: float = 20.0  # NEW: max % equity per symbol
    min_confidence: float = 0.50
    min_rr_ratio: float = 2.0                # NEW: reject if R/R < 2
    portfolio_correlation_threshold: float = 0.70
    # Execution
    slippage_bps: float = 15.0               # realistic default
    maker_fee_bps: float = 2.0
    taker_fee_bps: float = 5.0
    funding_interval_hours: int = 8
    # Paper mode
    paper: bool = True
    # Data
    data_dir: Path = field(default_factory=_data_dir)

    @property
    def is_live(self):
        return not self.paper and bool(self.okx_api_key or self.binance_api_key)


def load_config() -> Config:
    """Load from environment with hardened defaults."""
    okx_key = os.getenv("OKX_API_KEY", "")
    okx_sec = os.getenv("OKX_API_SECRET", "")
    okx_pass = os.getenv("OKX_PASSPHRASE", "")
    bin_key = os.getenv("BINANCE_API_KEY", "")
    paper = os.getenv("OMEGA_PAPER", "true").lower() in ("1", "true", "yes")

    venue = "okx" if (okx_key and okx_sec and okx_pass) else "binance"
    symbols = tuple(s.strip().upper() for s in os.getenv("OMEGA_SYMBOLS", "BTCUSDT,ETHUSDT,SOLUSDT").split(","))

    return Config(
        venue=venue,
        okx_api_key=okx_key, okx_api_secret=okx_sec, okx_passphrase=okx_pass,
        okx_demo=os.getenv("OKX_DEMO", "false").lower() in ("1", "true", "yes"),
        binance_api_key=bin_key,
        binance_api_secret=os.getenv("BINANCE_API_SECRET", ""),
        symbols=symbols,
        kelly_fraction=float(os.getenv("OMEGA_RISK_KELLY_FRACTION", "0.25")),
        max_per_trade_risk_pct=float(os.getenv("OMEGA_RISK_PER_TRADE_PCT", "1.0")),
        max_drawdown_pct=float(os.getenv("OMEGA_RISK_MAX_DRAWDOWN_PCT", "8.0")),
        max_daily_loss_pct=float(os.getenv("OMEGA_RISK_MAX_DAILY_LOSS_PCT", "3.0")),
        max_positions=int(os.getenv("OMEGA_RISK_MAX_POSITIONS", "8")),
        min_confidence=float(os.getenv("OMEGA_RISK_MIN_CONFIDENCE", "0.50")),
        min_rr_ratio=float(os.getenv("OMEGA_RISK_MIN_RR", "2.0")),
        paper=paper,
    )
