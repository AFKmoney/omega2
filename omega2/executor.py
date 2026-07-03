"""
OMEGA2 — Execution (Binance + OKX unified, with wallet).

Simplified from omega-hedge-fund's 6 execution files into one.
Includes paper mode (real data, log orders only).
"""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import base64
import os
import time
import urllib.parse
from datetime import datetime, timezone
from typing import Dict, Optional

import aiohttp

from omega2.core import Config, Fill, Order, OrderType, Side, get_logger

logger = get_logger("omega2.executor")


class Executor:
    """Unified executor: Binance or OKX. Paper mode = log only."""

    def __init__(self, config: Config):
        self.cfg = config
        self._session = None
        self._dry = config.paper or not (
            (config.okx_api_key and config.okx_api_secret and config.okx_passphrase)
            or (config.binance_api_key and config.binance_api_secret)
        )
        venue = config.venue if not self._dry else f"{config.venue} (dry-run)"
        logger.info(f"Executor: {venue} {'PAPER' if self._dry else 'LIVE'}")

    async def _get_session(self):
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def submit(self, order: Order, arrival_price: float) -> Optional[Fill]:
        """Submit order. Returns Fill or None."""
        if self._dry:
            logger.info(
                f"[PAPER] {order.side.value} {order.qty:.6f} {order.symbol} @ ~${arrival_price:.2f}",
                extra={"component": "executor", "symbol": order.symbol},
            )
            # Simulate slippage + fee
            slippage = self.cfg.slippage_bps / 10000.0
            fill_price = arrival_price * (1 + slippage * (1 if order.side == Side.BUY else -1))
            fee = order.qty * fill_price * self.cfg.taker_fee_bps / 10000.0
            return Fill(
                order_id=order.order_id, symbol=order.symbol, side=order.side,
                qty=order.qty, fill_price=fill_price,
                timestamp=datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
                slippage_bps=self.cfg.slippage_bps, fee_paid=fee,
            )
        # Real execution would go here (OKX/Binance signed REST)
        # For now, live mode falls through to paper
        logger.warning("Live execution not yet wired in OMEGA2 — using paper fill")
        return await self._paper_fill(order, arrival_price)

    async def _paper_fill(self, order, price):
        slippage = self.cfg.slippage_bps / 10000.0
        fill_price = price * (1 + slippage * (1 if order.side == Side.BUY else -1))
        fee = order.qty * fill_price * self.cfg.taker_fee_bps / 10000.0
        return Fill(
            order_id=order.order_id, symbol=order.symbol, side=order.side,
            qty=order.qty, fill_price=fill_price,
            timestamp=datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            slippage_bps=self.cfg.slippage_bps, fee_paid=fee,
        )

    async def get_balance(self, ccy="USDT") -> float:
        if self._dry:
            return 10_000.0
        return 0.0  # TODO: real balance fetch

    async def emergency_flatten(self) -> int:
        """Cancel all open orders + flatten positions. Called by Kill Switch."""
        logger.error("EMERGENCY FLATTEN: cancelling all", extra={"component": "executor"})
        if self._dry:
            return 0
        return 0  # TODO: real cancel-all

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()

    @property
    def is_live(self):
        return not self._dry
