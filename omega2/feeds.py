"""
OMEGA2 — Market data feeds (Binance + OKX unified).

Simplified from omega-hedge-fund's 4 feed files into one. Same WS logic,
same MarketEvent contract, both venues in one module.
"""
from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime, timezone
from typing import AsyncIterator, Dict, List, Optional

import aiohttp
import websockets

from omega2.core import MarketEvent, get_logger

logger = get_logger("omega2.feeds")


def _ms_to_iso(ms):
    if not ms:
        return datetime.now(timezone.utc).isoformat(timespec="milliseconds")
    return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc).isoformat(timespec="milliseconds")


class BinanceFeed:
    """Binance combined WS stream: trades + depth + ticker + markPrice."""

    def __init__(self, symbols=("BTCUSDT",), depth_levels=20, include_funding=True):
        self.symbols = tuple(s.upper() for s in symbols)
        self.depth_levels = depth_levels
        self.include_funding = include_funding
        self._last_funding: Dict[str, float] = {}
        self._last_book: Dict[str, tuple] = {}

    def _url(self):
        streams = []
        for s in self.symbols:
            sl = s.lower()
            streams.append(f"{sl}@trade")
            streams.append(f"{sl}@depth{self.depth_levels}@100ms")
            streams.append(f"{sl}@ticker")
            if self.include_funding:
                streams.append(f"{sl}@markPrice@1s")
        return f"wss://stream.binance.com:9443/stream?streams={'/'.join(streams)}"

    async def stream(self) -> AsyncIterator[MarketEvent]:
        delay = 1.0
        while True:
            try:
                async with websockets.connect(self._url(), ping_interval=20, ping_timeout=10, max_size=2**22) as ws:
                    logger.info("Binance WS connected", extra={"component": "feeds"})
                    delay = 1.0
                    async for raw in ws:
                        ev = self._parse(raw)
                        if ev:
                            yield ev
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning(f"Binance WS disconnected ({exc}); reconnect in {delay:.0f}s")
                await asyncio.sleep(delay)
                delay = min(delay * 2, 30)

    def _parse(self, raw) -> Optional[MarketEvent]:
        env = json.loads(raw)
        data = env.get("data", env)
        stream = env.get("stream", "")
        et = data.get("e", "")
        sym = data.get("s", data.get("symbol", ""))

        if et == "trade":
            book = self._last_book.get(sym, (0, 0, 0, 0))
            return MarketEvent(
                symbol=sym, timestamp=_ms_to_iso(data.get("T", 0)),
                last_price=float(data["p"]), volume_24h=0.0,
                bid=book[0], ask=book[1], bid_qty=book[2], ask_qty=book[3],
                funding_rate=self._last_funding.get(sym), source="binance_trade",
            )
        elif et == "depthUpdate" or "@depth" in stream:
            bids = data.get("bids") or data.get("b", [])
            asks = data.get("asks") or data.get("a", [])
            bid = float(bids[0][0]) if bids else 0.0
            ask = float(asks[0][0]) if asks else 0.0
            bq = float(bids[0][1]) if bids else 0.0
            aq = float(asks[0][1]) if asks else 0.0
            if bid and ask:
                self._last_book[sym] = (bid, ask, bq, aq)
            last = (bid + ask) / 2 if bid and ask else 0.0
            return MarketEvent(
                symbol=sym, timestamp=_ms_to_iso(data.get("E", 0)),
                last_price=last, volume_24h=0.0, bid=bid, ask=ask,
                bid_qty=bq, ask_qty=aq, funding_rate=self._last_funding.get(sym),
                source="binance_depth",
            )
        elif et == "24hrTicker":
            sym = data["s"]
            bid = float(data["b"]); ask = float(data["a"])
            self._last_book[sym] = (bid, ask, float(data["B"]), float(data["A"]))
            return MarketEvent(
                symbol=sym, timestamp=_ms_to_iso(data["E"]),
                last_price=float(data["c"]), volume_24h=float(data["v"]),
                bid=bid, ask=ask, bid_qty=float(data["B"]), ask_qty=float(data["A"]),
                funding_rate=self._last_funding.get(sym), source="binance_ticker",
            )
        elif et == "markPriceUpdate":
            sym = data.get("s", "")
            rate = float(data.get("r", 0.0))
            self._last_funding[sym] = rate
            book = self._last_book.get(sym, (0, 0, 0, 0))
            return MarketEvent(
                symbol=sym, timestamp=_ms_to_iso(data.get("E", 0)),
                last_price=float(data.get("p", 0)), volume_24h=0.0,
                bid=book[0], ask=book[1], bid_qty=book[2], ask_qty=book[3],
                funding_rate=rate, source="binance_markprice",
            )
        return None


# Funding rate REST poller (fallback when WS funding is geo-blocked)
class FundingPoller:
    """Polls Binance Futures REST for funding rates (globally accessible)."""

    def __init__(self, symbols=("BTCUSDT",), interval_sec=60):
        self.symbols = tuple(s.upper() for s in symbols)
        self.interval_sec = interval_sec
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
                    logger.debug(f"Funding poll failed: {exc}")
                await asyncio.sleep(self.interval_sec)

    def get(self, symbol):
        return self._latest.get(symbol.upper())
