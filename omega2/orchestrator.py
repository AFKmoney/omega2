"""
OMEGA2 — Orchestrator (simplified pipeline, no bypass).

THE critical difference from omega-hedge-fund:
    - Every signal goes through RiskEngine.on_signal() — NO exceptions
    - Kill switch triggers emergency_flatten() — actually closes positions
    - Thesis-exit produces a real close order — no longer a no-op
    - Only 4 crowd signals (cut the 4 weak ones)
"""
from __future__ import annotations

import asyncio
import time
from typing import Dict, List, Optional

from omega2.agents import ContrarianAgent, PPOAgent, LeadLagAgent
from omega2.core import Config, CrowdSignal, Fill, MarketEvent, Order, Side, Signal, get_logger, load_config
from omega2.crowd import CrowdEngine
from omega2.executor import Executor
from omega2.feeds import BinanceFeed
from omega2.risk import RiskEngine

logger = get_logger("omega2.orchestrator")


class Orchestrator:
    """The simplified pipeline. All orders through RiskEngine — no bypass."""

    def __init__(self, config: Config = None):
        self.cfg = config or load_config()
        # Layer 1: Feed
        self.feed = BinanceFeed(symbols=self.cfg.symbols)
        # Layer 1.5: Crowd Engine (4 signals only)
        self.crowd = CrowdEngine(symbols=self.cfg.symbols)
        # Layer 2: Agents
        self.contrarian = ContrarianAgent()
        self.ppo_trend = PPOAgent(symbols=self.cfg.symbols, mode="trend")
        self.ppo_meanrev = PPOAgent(symbols=self.cfg.symbols, mode="meanrev")
        self.leadlag = LeadLagAgent(
            leader="BTCUSDT",
            followers=tuple(s for s in self.cfg.symbols if s != "BTCUSDT"),
        )
        self._agents = [self.contrarian, self.ppo_trend, self.ppo_meanrev, self.leadlag]
        # Layer 4: Risk (THE gate)
        self.risk = RiskEngine(self.cfg)
        # Layer 5: Execution
        self.executor = Executor(self.cfg)
        # State
        self._running = False
        self._signals = 0
        self._orders = 0
        self._fills = 0
        # Open positions for close tracking
        self._open_entries: Dict[str, Fill] = {}

    async def start(self):
        if self._running:
            return
        self._running = True
        await self.crowd.start()
        logger.info(
            f"OMEGA2 starting: {self.cfg.symbols} venue={self.cfg.venue} "
            f"paper={self.cfg.paper}",
            extra={"component": "orchestrator"},
        )
        self._task = asyncio.create_task(self._main_loop())

    async def stop(self):
        self._running = False
        if hasattr(self, "_task"):
            self._task.cancel()
            try: await self._task
            except asyncio.CancelledError: pass
        await self.crowd.stop()
        await self.executor.close()
        logger.info("OMEGA2 stopped")

    async def _main_loop(self):
        """Consume market events → crowd → agents → risk → execute."""
        try:
            async for event in self.feed.stream():
                if not self._running:
                    break
                try:
                    await self._on_market(event)
                except Exception as exc:
                    logger.exception(f"Event error: {exc}")
        except asyncio.CancelledError:
            pass

    async def _on_market(self, event: MarketEvent):
        """Process one market event through the full pipeline."""
        # 1. Feed risk engine (ATR, kill switch, portfolio heat)
        self.risk.update_market(event.symbol, event.last_price)

        # 2. Crowd engine
        crowd = self.crowd.compute(event.symbol, event.timestamp)

        # 3. Collect signals from all agents
        signals: List[Signal] = []
        # Contrarian reacts to crowd events
        if crowd:
            signals.extend(self.contrarian.on_crowd(crowd))
        # PPO agents react to market
        signals.extend(self.ppo_trend.on_market(event))
        signals.extend(self.ppo_meanrev.on_market(event))
        # LeadLag reacts to market
        signals.extend(self.leadlag.on_market(event))

        if not signals:
            return

        # 4. Check kill switch → emergency flatten if just triggered
        if self.risk.kill_switch.is_triggered:
            # Check if we need to flatten (only once per trigger)
            if not hasattr(self, "_flattened") or not self._flattened:
                self._flattened = True
                cancelled = await self.executor.emergency_flatten()
                logger.error(f"Kill switch active — flattened {cancelled} positions")
            return  # Block all new trades
        else:
            self._flattened = False

        # 5. Process each signal through THE gate (RiskEngine)
        for signal in signals:
            self._signals += 1
            price = self.risk.portfolio_heat.last_prices.get(signal.symbol, event.last_price)
            if price <= 0:
                continue
            order = self.risk.on_signal(signal, price)
            if order is None:
                continue
            # 6. Execute
            self._orders += 1
            fill = await self.executor.submit(order, arrival_price=price)
            if fill:
                self._fills += 1
                self._handle_fill(fill)

    def _handle_fill(self, fill: Fill):
        """Track open/close fills."""
        sym = fill.symbol
        if sym in self._open_entries:
            # Closing fill — compute PnL
            entry = self._open_entries.pop(sym)
            direction = 1.0 if entry.side == Side.BUY else -1.0
            pnl_bps = direction * (fill.fill_price - entry.fill_price) / entry.fill_price * 10000
            # Subtract fees
            total_fees_bps = (entry.fee_paid + fill.fee_paid) / (entry.qty * entry.fill_price) * 10000
            pnl_bps_net = pnl_bps - total_fees_bps
            pnl_usd = direction * (fill.fill_price - entry.fill_price) * entry.qty - entry.fee_paid - fill.fee_paid
            self.risk.on_trade_closed(pnl_bps_net, pnl_usd)
            self.risk.portfolio_heat.close(sym)
            logger.info(
                f"Closed {sym}: pnl={pnl_bps_net:+.1f}bps (${pnl_usd:+.2f}) "
                f"fees={total_fees_bps:.1f}bps",
                extra={"symbol": sym},
            )
        else:
            # Opening fill
            self._open_entries[sym] = fill
            logger.info(
                f"Opened {sym} {fill.side.value} qty={fill.qty:.6f} @ ${fill.fill_price:.2f} "
                f"slip={fill.slippage_bps:.1f}bps fee=${fill.fee_paid:.2f}",
                extra={"symbol": sym},
            )

    def stats(self):
        return {
            "running": self._running,
            "venue": self.cfg.venue,
            "paper": self.cfg.paper,
            "signals": self._signals,
            "orders": self._orders,
            "fills": self._fills,
            "risk": self.risk.stats(),
            "crowd": self.crowd.stats(),
        }
