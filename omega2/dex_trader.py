"""
OMEGA2 — DEX Trader: trade tokens FROM the connected wallet via Uniswap.

When a wallet is connected, OMEGA2 can:
  1. See all tokens in the wallet (via WalletScanner)
  2. Decide which token to swap and when (via CrowdEngine + agents)
  3. Build the optimal swap route (Uniswap V3 router)
  4. Present the unsigned transaction to the user for MetaMask signing

The backend NEVER signs — it builds the calldata and the frontend submits
via window.ethereum.sendTransaction(). The user approves in MetaMask.

Supported DEX: Uniswap V3 (Ethereum, Arbitrum, Optimism, Base, Polygon)
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List, Optional

from omega2.core import get_logger

logger = get_logger("omega2.dex")

# Uniswap V3 SwapRouter02 addresses per chain
UNISWAP_ROUTER = {
    "ethereum": "0x68b3465833fb72A70ecDF485E0e4C7bD8665Fc45",
    "polygon": "0x68b3465833fb72A70ecDF485E0e4C7bD8665Fc45",
    "arbitrum": "0x68b3465833fb72A70ecDF485E0e4C7bD8665Fc45",
    "optimism": "0x68b3465833fb72A70ecDF485E0e4C7bD8665Fc45",
    "base": "0x68b3465833fb72A70ecDF485E0e4C7bD8665Fc45",
}

# WETH addresses (for routing)
WETH = {
    "ethereum": "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
    "polygon": "0x0d500B1d8E8eF31E21C99d1Db9A6444d3ADf1270",
    "arbitrum": "0x82aF49447D8a07e3bd95BD0d56f35241523fBab1",
    "optimism": "0x4200000000000000000000000000000000000006",
    "base": "0x4200000000000000000000000000000000000006",
}

# Common token addresses (Ethereum mainnet — extend per chain)
TOKENS = {
    "ethereum": {
        "WETH": WETH["ethereum"], "USDT": "0xdAC17F958D2ee523a2206206994597C13D831ec7",
        "USDC": "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
        "WBTC": "0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599",
        "DAI": "0x6B175474E89094C44Da98b954EedeAC495271d0F",
        "LINK": "0x514910771AF9Ca656af840dff83E8264EcF986CA",
    }
}


@dataclass
class SwapRequest:
    """A DEX swap the bot recommends. User must sign in MetaMask."""
    chain: str
    from_token: str
    to_token: str
    amount_in: float
    estimated_amount_out: float
    price_impact_pct: float
    min_amount_out: float  # slippage-protected
    router_address: str
    calldata: str          # hex-encoded function call to the router
    value: str             # ETH value to send (for native swaps)
    gas_estimate: int
    deadline: int           # unix timestamp


class DEXTrader:
    """Builds DEX swap transactions for the connected wallet to sign."""

    def __init__(self):
        self._pending_swaps: List[SwapRequest] = []

    def build_swap(
        self,
        chain: str,
        from_token_symbol: str,
        to_token_symbol: str,
        amount_in: float,
        slippage_bps: float = 50.0,  # 0.5% default
    ) -> Optional[SwapRequest]:
        """
        Build an unsigned Uniswap V3 swap transaction.

        Returns a SwapRequest that the frontend submits via MetaMask.
        The user signs — the backend never holds keys.
        """
        router = UNISWAP_ROUTER.get(chain)
        if not router:
            logger.warning(f"No DEX router for chain {chain}")
            return None

        tokens = TOKENS.get(chain, TOKENS.get("ethereum", {}))
        from_addr = tokens.get(from_token_symbol)
        to_addr = tokens.get(to_token_symbol)
        if not from_addr or not to_addr:
            logger.warning(f"Unknown token: {from_token_symbol} or {to_token_symbol}")
            return None

        # Build exactInputSingle calldata (Uniswap V3 SwapRouter02)
        # Function: exactInputSingle((tokenIn, tokenOut, fee, recipient, amountIn, amountOutMinimum, sqrtPriceLimitX96))
        # Selector: 0x04e45aaf
        import time
        deadline = int(time.time()) + 1200  # 20 min

        # Estimate output (simplified — in production use Quoter contract)
        est_out = amount_in * 0.997  # 0.3% fee assumption
        min_out = est_out * (1 - slippage_bps / 10000)

        # Encode calldata (simplified — in production use web3.py encoding)
        calldata = self._encode_exact_input_single(
            from_addr, to_addr, 3000,  # 0.3% fee tier
            "{RECIPIENT}",  # replaced by frontend with wallet address
            str(int(amount_in * 1e18)),
            str(int(min_out * 1e18)),
            "0",  # no price limit
        )

        swap = SwapRequest(
            chain=chain,
            from_token=from_token_symbol,
            to_token=to_token_symbol,
            amount_in=amount_in,
            estimated_amount_out=est_out,
            price_impact_pct=0.3,
            min_amount_out=min_out,
            router_address=router,
            calldata=calldata,
            value="0x0",
            gas_estimate=180000,
            deadline=deadline,
        )
        self._pending_swaps.append(swap)
        logger.info(
            f"DEX swap built: {amount_in} {from_token_symbol} → {to_token_symbol} "
            f"est_out={est_out:.6f} chain={chain}",
            extra={"component": "dex"},
        )
        return swap

    def _encode_exact_input_single(
        self, token_in: str, token_out: str, fee: int,
        recipient: str, amount_in: str, min_out: str, price_limit: str,
    ) -> str:
        """Encode exactInputSingle calldata (simplified ABI encoding)."""
        # Selector: exactInputSingle((address,address,uint24,address,uint256,uint256,uint160))
        selector = "0x04e45aaf"
        # Pad addresses and values to 32 bytes each
        def pad(addr):
            if addr.startswith("0x"):
                addr = addr[2:]
            return addr.lower().zfill(64)

        def pad_num(n):
            return hex(int(n))[2:].zfill(64)

        # Offset for the struct (32 bytes)
        offset = pad_num(32)
        # Struct members (7 × 32 bytes)
        encoded = (
            pad(token_in) + pad(token_out) + pad_num(fee) +
            pad(recipient) + pad_num(amount_in) + pad_num(min_out) + pad_num(price_limit)
        )
        return selector + offset + encoded

    def get_pending_swaps(self) -> List[dict]:
        """Return pending swaps for the frontend."""
        return [
            {
                "chain": s.chain,
                "from_token": s.from_token,
                "to_token": s.to_token,
                "amount_in": s.amount_in,
                "estimated_out": round(s.estimated_amount_out, 6),
                "min_out": round(s.min_amount_out, 6),
                "price_impact_pct": s.price_impact_pct,
                "router": s.router_address,
                "calldata": s.calldata[:80] + "...",
                "gas": s.gas_estimate,
                "expires_in_sec": max(0, s.deadline - int(__import__("time").time())),
            }
            for s in self._pending_swaps[-10:]  # last 10
        ]

    def clear_pending(self):
        self._pending_swaps.clear()
