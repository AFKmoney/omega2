"""
OMEGA2 — Crowd Positioning Engine (4 signals only).

Cut from 8 signals to the 4 that actually predict moves:
    1. Liquidations (WS real-time — most predictive)
    2. Funding rate (perp leverage crowding)
    3. Open interest ROC (leverage piling in)
    4. L/S ratio (retail account positioning)

Same fusion logic as omega-hedge-fund: weighted score + divergence-based conviction.
"""
from __future__ import annotations

import asyncio
import json
import math
import time
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

import aiohttp

from omega2.core import CrowdSignal, MarketEvent, get_logger

logger = get_logger("omega2.crowd")

_HORIZON = {"minutes": 0, "hours": 1, "days": 2}


# ═══════════════════════════════════════════════════════════════════════
# Signal 1: Liquidations (WS real-time)
# ═══════════════════════════════════════════════════════════════════════

class LiquidationSignal:
    name = "liquidations"
    weight = 0.40
    horizon = "minutes"

    def __init__(self, symbols=("BTCUSDT",), window_sec=300, threshold_usd=50_000_000):
        self.symbols = tuple(s.upper() for s in symbols)
        self.window_sec = window_sec
        self.threshold_usd = threshold_usd
        self._events: Dict[str, Deque] = {s: deque(maxlen=5000) for s in self.symbols}
        self._task = None
        self._total = 0

    async def start(self):
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._ws_loop())

    async def stop(self):
        if self._task:
            self._task.cancel()
            try: await self._task
            except asyncio.CancelledError: pass
            self._task = None

    async def _ws_loop(self):
        delay = 1.0
        while True:
            try:
                async with websockets.connect(
                    "wss://fstream.binance.com/ws/!forceOrder@arr",
                    ping_interval=20,
                ) if False else websockets.connect(
                    "wss://fstream.binance.com/ws/!forceOrder@arr", ping_interval=20
                ) as ws:
                    logger.info("Liquidation WS connected", extra={"component": "crowd"})
                    delay = 1.0
                    async for raw in ws:
                        self._handle(json.loads(raw))
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.debug(f"LIQ WS: {exc}, reconnect {delay:.0f}s")
                await asyncio.sleep(delay)
                delay = min(delay * 2, 30)

    def _handle(self, envelope):
        order = envelope.get("o", envelope)
        sym = order.get("s", "")
        if sym not in self.symbols:
            return
        side = order.get("S", "")
        price = float(order.get("ap", order.get("p", 0)) or 0)
        qty = float(order.get("q", 0) or 0)
        notional = price * qty
        if notional <= 0:
            return
        liq_side = "LONG" if side == "SELL" else "SHORT"
        self._total += 1
        self._events.setdefault(sym, deque(maxlen=5000)).append((time.time(), liq_side, notional))

    def reading(self, symbol):
        now = time.time()
        cutoff = now - self.window_sec
        events = self._events.get(symbol, deque())
        long_usd = short_usd = 0.0
        for ts, side, notional in events:
            if ts > cutoff:
                if side == "LONG": long_usd += notional
                else: short_usd += notional
        net = long_usd - short_usd
        if self.threshold_usd <= 0:
            return 0.0
        return max(-1.0, min(1.0, math.tanh(net / self.threshold_usd)))

    def stats(self):
        return {"name": self.name, "total_seen": self._total, "symbols": list(self.symbols)}


# ═══════════════════════════════════════════════════════════════════════
# Signal 2: Funding rate (REST poll — globally accessible)
# ═══════════════════════════════════════════════════════════════════════

class FundingSignal:
    name = "funding"
    weight = 0.35
    horizon = "hours"

    def __init__(self, symbols=("BTCUSDT",), threshold=0.0005, poll_sec=60):
        self.symbols = tuple(s.upper() for s in symbols)
        self.threshold = threshold
        self.poll_sec = poll_sec
        self._latest: Dict[str, float] = {}
        self._task = None

    async def start(self):
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._loop())

    async def stop(self):
        if self._task:
            self._task.cancel()
            try: await self._task
            except asyncio.CancelledError: pass
            self._task = None

    async def _loop(self):
        async with aiohttp.ClientSession() as session:
            while True:
                try:
                    for sym in self.symbols:
                        url = f"https://fapi.binance.com/fapi/v1/premiumIndex?symbol={sym}"
                        async with session.get(url, timeout=10) as resp:
                            if resp.status == 200:
                                data = await resp.json()
                                self._latest[sym] = float(data.get("lastFundingRate", 0) or 0)
                except asyncio.CancelledError: raise
                except Exception as exc:
                    logger.debug(f"Funding poll: {exc}")
                await asyncio.sleep(self.poll_sec)

    def reading(self, symbol):
        rate = self._latest.get(symbol.upper())
        if rate is None:
            return None
        return max(-1.0, min(1.0, math.tanh(rate / self.threshold)))

    def stats(self):
        return {"name": self.name, "latest": dict(self._latest)}


# ═══════════════════════════════════════════════════════════════════════
# Signal 3: Open Interest ROC (REST poll)
# ═══════════════════════════════════════════════════════════════════════

class OpenInterestSignal:
    name = "open_interest"
    weight = 0.30
    horizon = "hours"

    def __init__(self, symbols=("BTCUSDT",), poll_sec=300, roc_gain=10.0):
        self.symbols = tuple(s.upper() for s in symbols)
        self.poll_sec = poll_sec
        self.roc_gain = roc_gain
        self._roc: Dict[str, float] = {}
        self._task = None

    async def start(self):
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._loop())

    async def stop(self):
        if self._task:
            self._task.cancel()
            try: await self._task
            except asyncio.CancelledError: pass
            self._task = None

    async def _loop(self):
        async with aiohttp.ClientSession() as session:
            while True:
                try:
                    for sym in self.symbols:
                        params = {"symbol": sym, "period": "5m", "limit": 14}
                        url = "https://fapi.binance.com/futures/data/openInterestHist"
                        async with session.get(url, params=params, timeout=10) as resp:
                            if resp.status == 200:
                                payload = await resp.json()
                                if payload:
                                    oi = [float(r["sumOpenInterest"]) for r in payload]
                                    rocs = [(oi[i]-oi[i-1])/oi[i-1] for i in range(1, len(oi)) if oi[i-1] > 0]
                                    self._roc[sym] = sum(rocs) if rocs else 0.0
                except asyncio.CancelledError: raise
                except Exception as exc:
                    logger.debug(f"OI poll: {exc}")
                await asyncio.sleep(self.poll_sec)

    def reading(self, symbol):
        roc = self._roc.get(symbol.upper())
        if roc is None:
            return None
        return max(-1.0, min(1.0, roc * self.roc_gain))

    def stats(self):
        return {"name": self.name, "roc": {k: round(v, 5) for k, v in self._roc.items()}}


# ═══════════════════════════════════════════════════════════════════════
# Signal 4: L/S Account Ratio (REST poll)
# ═══════════════════════════════════════════════════════════════════════

class LSRatioSignal:
    name = "ls_ratio"
    weight = 0.35
    horizon = "hours"

    def __init__(self, symbols=("BTCUSDT",), poll_sec=300):
        self.symbols = tuple(s.upper() for s in symbols)
        self.poll_sec = poll_sec
        self._long_pct: Dict[str, float] = {}
        self._task = None

    async def start(self):
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._loop())

    async def stop(self):
        if self._task:
            self._task.cancel()
            try: await self._task
            except asyncio.CancelledError: pass
            self._task = None

    async def _loop(self):
        async with aiohttp.ClientSession() as session:
            while True:
                try:
                    for sym in self.symbols:
                        params = {"symbol": sym, "period": "5m", "limit": 1}
                        url = "https://fapi.binance.com/futures/data/globalLongShortAccountRatio"
                        async with session.get(url, params=params, timeout=10) as resp:
                            if resp.status == 200:
                                payload = await resp.json()
                                if payload:
                                    la = float(payload[0].get("longAccount", 0.5))
                                    sa = float(payload[0].get("shortAccount", 0.5))
                                    total = la + sa
                                    if total > 0:
                                        self._long_pct[sym] = la / total * 100.0
                except asyncio.CancelledError: raise
                except Exception as exc:
                    logger.debug(f"L/S poll: {exc}")
                await asyncio.sleep(self.poll_sec)

    def reading(self, symbol):
        pct = self._long_pct.get(symbol.upper())
        if pct is None:
            return None
        return max(-1.0, min(1.0, (pct - 50.0) / 50.0))

    def stats(self):
        return {"name": self.name, "long_pct": dict(self._long_pct)}


# ═══════════════════════════════════════════════════════════════════════
# Engine: fuses 4 signals into one CrowdSignal
# ═══════════════════════════════════════════════════════════════════════

class CrowdEngine:
    """Fuses 4 positioning signals into CrowdSignal events."""

    def __init__(self, symbols=("BTCUSDT",), emit_threshold=0.15, reemit_delta=0.08):
        self.symbols = tuple(s.upper() for s in symbols)
        self.liquidations = LiquidationSignal(symbols=self.symbols)
        self.funding = FundingSignal(symbols=self.symbols)
        self.open_interest = OpenInterestSignal(symbols=self.symbols)
        self.ls_ratio = LSRatioSignal(symbols=self.symbols)
        self._signals = [self.liquidations, self.funding, self.open_interest, self.ls_ratio]
        self.emit_threshold = emit_threshold
        self.reemit_delta = reemit_delta
        self._last_score: Dict[str, float] = {}
        self._events = 0

    async def start(self):
        for sig in self._signals:
            starter = getattr(sig, "start", None)
            if starter:
                await starter()
        logger.info(f"CrowdEngine started: {len(self.symbols)} symbols, 4 signals",
                    extra={"component": "crowd"})

    async def stop(self):
        for sig in self._signals:
            stopper = getattr(sig, "stop", None)
            if stopper:
                try: await stopper()
                except Exception: pass

    def compute(self, symbol, timestamp) -> Optional[CrowdSignal]:
        """Fuse the 4 signals for one symbol."""
        components = {}
        readings = []
        for sig in self._signals:
            r = sig.reading(symbol)
            if r is None:
                continue
            components[sig.name] = round(r, 4)
            readings.append((r, sig.weight, sig.horizon))

        if not readings:
            return None

        total_w = sum(w for _, w, _ in readings)
        if total_w <= 0:
            return None

        crowd_score = sum(r * w for r, w, _ in readings) / total_w
        crowd_score = max(-1.0, min(1.0, crowd_score))

        # Divergence: how much do significant components disagree?
        sig_scores = [r for r, _, _ in readings if abs(r) >= 0.15]
        if len(sig_scores) >= 2:
            direction = 1.0 if crowd_score >= 0 else -1.0
            disagree = sum(1 for s in sig_scores if (s >= 0) != (direction >= 0))
            divergence = disagree / len(sig_scores)
        else:
            divergence = 0.0 if sig_scores else 1.0

        conviction = abs(crowd_score) * (1.0 - divergence)
        conviction = max(0.0, min(1.0, conviction))

        # Horizon
        sig_readings = [(r, h) for r, _, h in readings if abs(r) >= 0.15]
        if sig_readings:
            horizon = max((h for _, h in sig_readings), key=lambda h: _HORIZON.get(h, 0))
        else:
            horizon = "hours"

        regime_hint = "neutral"
        if conviction >= 0.30:
            if crowd_score > 0:
                regime_hint = "cascade_imminent" if conviction > 0.55 else "euphoria"
            else:
                regime_hint = "cascade_imminent" if conviction > 0.55 else "fear"

        base = {"minutes": 50.0, "hours": 200.0, "days": 600.0}.get(horizon, 200.0)
        expected_move = abs(crowd_score) * conviction * base

        # Throttle
        if abs(crowd_score) < self.emit_threshold:
            return None
        last = self._last_score.get(symbol)
        if last is not None and abs(crowd_score - last) < self.reemit_delta:
            return None
        self._last_score[symbol] = crowd_score
        self._events += 1

        return CrowdSignal(
            symbol=symbol, timestamp=timestamp,
            crowd_score=crowd_score, conviction=conviction,
            components=components, regime_hint=regime_hint,
            expected_move_bps=expected_move,
        )

    def stats(self):
        live_scores = {}
        sym = self.symbols[0] if self.symbols else ""
        for sig in self._signals:
            r = sig.reading(sym)
            live_scores[sig.name] = {
                "score": round(r if r else 0.0, 4),
                "has_data": r is not None,
            }
        return {
            "symbols": list(self.symbols),
            "events_emitted": self._events,
            "live_scores": live_scores,
            "weights": {s.name: s.weight for s in self._signals},
            "signals": {s.name: s.stats() for s in self._signals},
        }
