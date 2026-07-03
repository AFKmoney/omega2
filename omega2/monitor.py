"""
OMEGA2 — Monitor (alerting on kill switch, drawdown, daily loss).

Sends alerts via Telegram (free Bot API) or Discord webhook.
Configured via env vars — no-op if not configured.
"""
from __future__ import annotations

import asyncio
import os
from typing import Optional

import aiohttp

from omega2.core import get_logger

logger = get_logger("omega2.monitor")


class Monitor:
    """Sends alerts on critical events. No-op if not configured."""

    def __init__(self):
        self.telegram_token = os.getenv("TELEGRAM_BOT_TOKEN", "")
        self.telegram_chat = os.getenv("TELEGRAM_CHAT_ID", "")
        self.discord_webhook = os.getenv("DISCORD_WEBHOOK", "")
        self._last_alert = {}

    async def alert(self, message: str, level: str = "WARNING"):
        """Send alert to all configured channels."""
        full_msg = f"🤖 OMEGA2 [{level}] {message}"
        logger.info(f"Alert sent: {full_msg}")
        if self.telegram_token and self.telegram_chat:
            await self._send_telegram(full_msg)
        if self.discord_webhook:
            await self._send_discord(full_msg)

    async def _send_telegram(self, text: str):
        try:
            url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
            async with aiohttp.ClientSession() as s:
                async with s.post(url, json={"chat_id": self.telegram_chat, "text": text, "parse_mode": "Markdown"}) as r:
                    if r.status != 200:
                        logger.debug(f"Telegram send failed: {r.status}")
        except Exception as exc:
            logger.debug(f"Telegram alert failed: {exc}")

    async def _send_discord(self, text: str):
        try:
            async with aiohttp.ClientSession() as s:
                async with s.post(self.discord_webhook, json={"content": text}) as r:
                    if r.status != 204:
                        logger.debug(f"Discord send failed: {r.status}")
        except Exception as exc:
            logger.debug(f"Discord alert failed: {exc}")

    def check_and_alert(self, risk_stats: dict):
        """Check risk stats and alert on critical conditions."""
        import time
        now = time.time()
        # Kill switch
        ks = risk_stats.get("kill_switch", {})
        if ks.get("triggered") and now - self._last_alert.get("kill", 0) > 300:  # 5 min dedup
            self._last_alert["kill"] = now
            asyncio.create_task(self.alert(f"🚨 KILL SWITCH: {ks.get('reason','')}", "CRITICAL"))
        # Drawdown
        pnl_pct = risk_stats.get("pnl_pct", 0)
        if pnl_pct < -5 and now - self._last_alert.get("dd", 0) > 600:
            self._last_alert["dd"] = now
            asyncio.create_task(self.alert(f"📉 Drawdown: {pnl_pct:.1f}%", "WARNING"))
