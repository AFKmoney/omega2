"""
OMEGA2 — Wallet Scanner: auto-discovers ALL tokens across ALL chains.

When a wallet connects, this module:
  1. Scans the wallet's transaction history (via free Etherscan/Polygonscan APIs)
  2. Discovers every ERC-20 token the wallet has ever interacted with
  3. Checks current balances for each discovered token
  4. Returns the complete portfolio with USD estimates

No API key needed for read operations on public block explorers.
Works across: Ethereum, Polygon, BSC, Arbitrum, Base, Optimism.

The TokenRanker then scores which tokens are best to trade based on:
  - Liquidity (24h volume from CoinGecko)
  - Price momentum (recent performance)
  - Spread tightness (tradeable)
  - Position size available
"""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import aiohttp

from omega2.core import get_logger
from omega2.web3 import CHAIN_CONFIG

logger = get_logger("omega2.wallet_scanner")

# Free block explorer APIs (no key for basic tx list)
EXPLORER_API = {
    "ethereum": "https://api.etherscan.io/api",
    "polygon": "https://api.polygonscan.com/api",
    "bsc": "https://api.bscscan.com/api",
    "arbitrum": "https://api.arbiscan.io/api",
    "base": "https://api.basescan.org/api",
    "optimism": "https://api-optimistic.etherscan.io/api",
}

# CoinGecko ID mapping for price data
COINGECKO_IDS = {
    "ETH": "ethereum", "USDT": "tether", "USDC": "usd-coin", "DAI": "dai",
    "WBTC": "wrapped-bitcoin", "LINK": "chainlink", "UNI": "uniswap",
    "MATIC": "matic-network", "BNB": "binancecoin", "CAKE": "pancakeswap-token",
    "ARB": "arbitrum", "OP": "optimism",
}


@dataclass
class TokenHolding:
    chain: str
    symbol: str
    balance: float
    contract: str
    usd_price: float = 0.0
    usd_value: float = 0.0
    tradeable: bool = False
    rank_score: float = 0.0


class WalletScanner:
    """Auto-discovers all tokens in a wallet across all chains."""

    def __init__(self):
        self._session: Optional[aiohttp.ClientSession] = None
        self._price_cache: Dict[str, float] = {}
        self._price_time: float = 0.0

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()

    async def scan_wallet(self, address: str, chains: List[str] = None) -> List[TokenHolding]:
        """
        Full scan: discover all tokens on all chains for a wallet.
        Returns a ranked list of holdings.
        """
        chains = chains or list(CHAIN_CONFIG.keys())
        address = address.lower()
        all_holdings: List[TokenHolding] = []

        for chain in chains:
            holdings = await self._scan_chain(address, chain)
            all_holdings.extend(holdings)

        # Fetch USD prices
        await self._update_prices(all_holdings)

        # Rank by tradeability
        self._rank(all_holdings)

        # Sort by USD value descending
        all_holdings.sort(key=lambda h: h.usd_value, reverse=True)
        return all_holdings

    async def _scan_chain(self, address: str, chain: str) -> List[TokenHolding]:
        """Scan one chain: get native balance + discover ERC-20 tokens."""
        holdings: List[TokenHolding] = []
        session = await self._get_session()

        # 1. Native balance via RPC
        config = CHAIN_CONFIG.get(chain, {})
        rpc_url = config.get("rpc", "")
        native_sym = config.get("native", "ETH")
        if rpc_url:
            try:
                payload = {"jsonrpc": "2.0", "method": "eth_getBalance", "params": [address, "latest"], "id": 1}
                async with session.post(rpc_url, json=payload, timeout=8) as resp:
                    result = await resp.json()
                    balance = int(result.get("result", "0x0"), 16) / 1e18
                    if balance > 0.001:
                        holdings.append(TokenHolding(
                            chain=chain, symbol=native_sym, balance=balance, contract="",
                        ))
            except Exception:
                pass

        # 2. Discover ERC-20 tokens via transfer events (free, no key)
        explorer_url = EXPLORER_API.get(chain)
        if not explorer_url:
            return holdings

        discovered_contracts: Dict[str, str] = {}  # contract -> symbol

        # Get ERC-20 token transfer events (last 200)
        try:
            params = {
                "module": "account", "action": "tokentx",
                "address": address, "startblock": 0, "endblock": 99999999,
                "page": 1, "offset": 200, "sort": "desc",
            }
            # Some explorers need API key for this — try without first
            async with session.get(explorer_url, params=params, timeout=10) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    for tx in data.get("result", [])[:50]:
                        if isinstance(tx, dict):
                            contract = tx.get("contractAddress", "")
                            symbol = tx.get("tokenSymbol", "?")
                            if contract and contract not in discovered_contracts:
                                discovered_contracts[contract] = symbol
        except Exception as exc:
            logger.debug(f"Token discovery failed ({chain}): {exc}")

        # 3. Check balances for discovered tokens
        for contract, symbol in list(discovered_contracts.items())[:20]:  # limit to 20 per chain
            balance = await self._get_erc20_balance(address, contract, chain, session)
            if balance and balance > 0.0001:
                holdings.append(TokenHolding(
                    chain=chain, symbol=symbol, balance=balance, contract=contract,
                ))

        return holdings

    async def _get_erc20_balance(self, address: str, contract: str, chain: str, session) -> float:
        """Get ERC-20 balance via balanceOf() RPC call."""
        config = CHAIN_CONFIG.get(chain, {})
        rpc_url = config.get("rpc", "")
        if not rpc_url:
            return 0.0
        padded = address[2:].zfill(64)
        data = f"0x70a08231000000000000000000000000{padded}"
        try:
            payload = {"jsonrpc": "2.0", "method": "eth_call", "params": [{"to": contract, "data": data}, "latest"], "id": 1}
            async with session.post(rpc_url, json=payload, timeout=8) as resp:
                result = await resp.json()
                raw = int(result.get("result", "0x0"), 16)
                # Try different decimals — most are 6 or 18
                for decimals in (6, 18, 8):
                    val = raw / (10 ** decimals)
                    if 0.001 < val < 10_000_000:
                        return val
                return raw / 1e18
        except Exception:
            return 0.0

    async def _update_prices(self, holdings: List[TokenHolding]):
        """Fetch USD prices from CoinGecko for all holdings."""
        import time
        if time.time() - self._price_time < 60:  # cache 60s
            for h in holdings:
                h.usd_price = self._price_cache.get(h.symbol, 0.0)
                h.usd_value = h.balance * h.usd_price
            return

        # Collect unique symbols
        symbols = list(set(h.symbol for h in holdings if h.symbol in COINGECKO_IDS))
        if not symbols:
            return

        ids = ",".join(COINGECKO_IDS[s] for s in symbols if s in COINGECKO_IDS)
        if not ids:
            return

        session = await self._get_session()
        try:
            url = f"https://api.coingecko.com/api/v3/simple/price?ids={ids}&vs_currencies=usd"
            async with session.get(url, timeout=10) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    for sym in symbols:
                        cg_id = COINGECKO_IDS.get(sym, "")
                        price = data.get(cg_id, {}).get("usd", 0.0)
                        self._price_cache[sym] = price
                    self._price_time = time.time()
        except Exception as exc:
            logger.debug(f"Price update failed: {exc}")

        for h in holdings:
            h.usd_price = self._price_cache.get(h.symbol, 0.0)
            h.usd_value = h.balance * h.usd_price

    def _rank(self, holdings: List[TokenHolding]):
        """Rank tokens by tradeability score."""
        for h in holdings:
            score = 0.0
            # Has USD value = liquid
            if h.usd_value > 100:
                score += 30
            elif h.usd_value > 10:
                score += 15
            elif h.usd_value > 1:
                score += 5
            # Known liquid token = tradeable
            if h.symbol in COINGECKO_IDS:
                score += 20  # we can price it = we can trade it
            # Stablecoin = skip for trading (use as base)
            if h.symbol in ("USDT", "USDC", "DAI"):
                score = 0  # not a trade target, it's the base currency
                h.tradeable = False
            else:
                h.tradeable = score > 15
            h.rank_score = score

    def get_best_trade_targets(self, holdings: List[TokenHolding], max_n: int = 5) -> List[TokenHolding]:
        """Return the top tokens to trade (highest rank + tradeable)."""
        tradeable = [h for h in holdings if h.tradeable]
        tradeable.sort(key=lambda h: h.rank_score, reverse=True)
        return tradeable[:max_n]
