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
from omega2.wallet_scanner import WalletScanner
from omega2.dex_trader import DEXTrader

logger = get_logger("omega2.web")
_STATIC = Path(__file__).resolve().parent / "static"


class WebServer:
    def __init__(self, port=8080):
        self.port = port
        self.app = web.Application()
        self._orch: Optional[Orchestrator] = None
        self._web3 = Web3Manager()
        self._scanner = WalletScanner()
        self._dex = DEXTrader()
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
        self.app.router.add_get("/api/chart/{symbol}", self._chart)
        self.app.router.add_get("/api/wallet/scan", self._wallet_scan)
        self.app.router.add_get("/api/wallet/targets", self._wallet_targets)
        # DEX trading
        self.app.router.add_post("/api/dex/swap", self._dex_swap)
        self.app.router.add_get("/api/dex/pending", self._dex_pending)

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

    async def _chart(self, req):
        symbol = req.match_info["symbol"].upper()
        prices = self.orch.risk.portfolio_heat._prices
        hist = list(prices.get(symbol, []))
        if not hist or len(hist) < 2:
            return web.json_response({"candles": []})
        import time
        candles = []
        for i, p in enumerate(list(hist)[-200:]):
            t = int(time.time() // 15 * 15) - (len(hist) - i) * 15
            candles.append({"t": t, "o": p, "h": p, "l": p, "c": p})
        return web.json_response({"candles": candles})

    async def _wallet_scan(self, req):
        """Full wallet scan: discover ALL tokens across ALL chains."""
        address = req.query.get("address", "")
        if not address:
            return web.json_response({"error": "provide ?address=0x..."}, status=400)
        chains = req.query.get("chains", "ethereum,polygon,bsc,arbitrum,base,optimism").split(",")
        holdings = await self._scanner.scan_wallet(address, chains)
        return web.json_response({
            "address": address,
            "total_tokens": len(holdings),
            "total_usd_value": round(sum(h.usd_value for h in holdings), 2),
            "holdings": [
                {
                    "chain": h.chain, "symbol": h.symbol, "balance": round(h.balance, 6),
                    "usd_price": round(h.usd_price, 4), "usd_value": round(h.usd_value, 2),
                    "tradeable": h.tradeable, "rank_score": round(h.rank_score, 1),
                    "contract": h.contract[:12] + "..." if h.contract else "",
                }
                for h in holdings
            ],
        })

    async def _wallet_targets(self, req):
        """Return the best tokens to trade from connected wallets."""
        address = req.query.get("address", "")
        if not address:
            # Use first connected wallet
            wallets = self._web3.list_wallets()
            if not wallets:
                return web.json_response({"targets": [], "msg": "No wallet connected"})
            address = wallets[0]["address"]
        holdings = await self._scanner.scan_wallet(address)
        targets = self._scanner.get_best_trade_targets(holdings, max_n=5)
        return web.json_response({
            "address": address,
            "targets": [
                {"chain": t.chain, "symbol": t.symbol, "usd_value": round(t.usd_value, 2),
                 "rank_score": round(t.rank_score, 1)}
                for t in targets
            ],
            "msg": f"Top {len(targets)} trade targets from {len(holdings)} holdings",
        })

    async def _dex_swap(self, req):
        """Build a DEX swap tx for the user to sign in MetaMask."""
        data = await req.json()
        swap = self._dex.build_swap(
            chain=data.get("chain", "ethereum"),
            from_token_symbol=data.get("from_token", "USDC"),
            to_token_symbol=data.get("to_token", "WETH"),
            amount_in=float(data.get("amount", 100)),
            slippage_bps=float(data.get("slippage_bps", 50)),
        )
        if swap is None:
            return web.json_response({"ok": False, "error": "Cannot build swap"}, status=400)
        return web.json_response({"ok": True, "swap": {
            "chain": swap.chain, "from": swap.from_token, "to": swap.to_token,
            "amount_in": swap.amount_in, "est_out": round(swap.estimated_amount_out, 6),
            "min_out": round(swap.min_amount_out, 6), "router": swap.router_address,
            "calldata": swap.calldata, "gas": swap.gas_estimate,
        }})

    async def _dex_pending(self, req):
        """Return pending swaps for the frontend."""
        return web.json_response({"swaps": self._dex.get_pending_swaps()})

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
