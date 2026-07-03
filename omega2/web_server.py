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
from omega2.web3 import Web3Manager

logger = get_logger("omega2.web")
_STATIC = Path(__file__).resolve().parent / "static"


class WebServer:
    def __init__(self, port=8080):
        self.port = port
        self.app = web.Application()
        self._orch: Optional[Orchestrator] = None
        self._web3 = Web3Manager()
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
        # Web3
        self.app.router.add_get("/api/web3/status", self._web3_status)
        self.app.router.add_post("/api/web3/connect", self._web3_connect)
        self.app.router.add_post("/api/web3/disconnect", self._web3_disconnect)
        self.app.router.add_get("/api/web3/balances", self._web3_balances)

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
        prices = self.orch.risk.portfolio_heat.last_prices
        return web.json_response({"prices": prices})

    # ===== WEB3 =====
    async def _web3_status(self, req):
        return web.json_response(self._web3.stats())

    async def _web3_connect(self, req):
        data = await req.json()
        addr = data.get("address", "")
        wallet_type = data.get("type", "rpc")
        chain = data.get("chain", "ethereum")
        ok = await self._web3.connect(addr, wallet_type, chain)
        return web.json_response({"ok": ok, "wallets": self._web3.list_wallets()})

    async def _web3_disconnect(self, req):
        data = await req.json()
        ok = self._web3.disconnect(data.get("address", ""))
        return web.json_response({"ok": ok})

    async def _web3_balances(self, req):
        addr = req.query.get("address", "")
        chain = req.query.get("chain", "ethereum")
        if not addr:
            all_balances = await self._web3.get_all_wallets_balances()
            return web.json_response(all_balances)
        bals = await self._web3.get_all_balances(addr, chain)
        return web.json_response([
            {"chain": b.chain, "token": b.token, "balance": b.balance, "contract": b.contract}
            for b in bals
        ])

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
