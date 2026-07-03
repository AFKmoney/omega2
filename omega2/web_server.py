"""
OMEGA2 — Web Server (REST API + static frontend).

7 tabs: Terminal, Crowd, Risk, Activity, Wallet, Settings, Markets.
"""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from aiohttp import web, WSMsgType

from omega2.core import load_config, get_logger
from omega2.orchestrator import Orchestrator

logger = get_logger("omega2.web")
_STATIC = Path(__file__).resolve().parent / "static"


class WebServer:
    def __init__(self, port=8080):
        self.port = port
        self.app = web.Application()
        self._orch: Optional[Orchestrator] = None
        self._setup()

    @property
    def orch(self):
        if self._orch is None:
            self._orch = Orchestrator()
        return self._orch

    def _setup(self):
        self.app.router.add_get("/", self._index)
        self.app.router.add_static("/", str(_STATIC), show_index=False)
        self.app.router.add_get("/api/status", self._status)
        self.app.router.add_get("/api/crowd", self._crowd)
        self.app.router.add_get("/api/risk", self._risk)
        self.app.router.add_post("/api/trading/start", self._start)
        self.app.router.add_post("/api/trading/stop", self._stop)
        self.app.router.add_get("/api/markets", self._markets)

    async def _index(self, req):
        f = _STATIC / "index.html"
        return web.Response(text=f.read_text(encoding="utf-8"), content_type="text/html")

    async def _status(self, req):
        return web.json_response(self.orch.stats())

    async def _crowd(self, req):
        return web.json_response(self.orch.crowd.stats())

    async def _risk(self, req):
        return web.json_response(self.orch.risk.stats())

    async def _start(self, req):
        asyncio.create_task(self.orch.start())
        return web.json_response({"ok": True})

    async def _stop(self, req):
        await self.orch.stop()
        return web.json_response({"ok": True})

    async def _markets(self, req):
        # Simplified — just return BTC price
        prices = self.orch.risk.portfolio_heat.last_prices
        return web.json_response({"prices": prices})

    async def run(self):
        runner = web.AppRunner(self.app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", self.port)
        await site.start()
        url = f"http://localhost:{self.port}"
        print(f"\n  ╔════════════════════════════════╗")
        print(f"  ║  OMEGA2: {url:<21}║")
        print(f"  ╚════════════════════════════════╝\n")
        try:
            while True:
                await asyncio.sleep(3600)
        except asyncio.CancelledError:
            pass


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8080)
    args = p.parse_args()
    server = WebServer(port=args.port)
    asyncio.run(server.run())


if __name__ == "__main__":
    main()
