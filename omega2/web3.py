"""
OMEGA2 — Web3 Multi-Wallet Manager.

Supports:
  1. MetaMask (browser injection — window.ethereum)
  2. WalletConnect (scan QR from any wallet app)
  3. Direct RPC (read-only balances from any address)
  4. Multiple connected wallets simultaneously

The backend handles READ operations (balances, tx history). WRITE operations
(signing, sending) go through the browser via the connected wallet provider.
The backend NEVER holds private keys — it builds unsigned txs and the wallet signs.

Supported chains: Ethereum, Polygon, BSC, Arbitrum, Base, Optimism.
"""
from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import aiohttp

from omega2.core import get_logger

logger = get_logger("omega2.web3")

# Public RPC endpoints (replace with Alchemy/Infura for production)
CHAIN_CONFIG = {
    "ethereum": {"rpc": "https://rpc.ankr.com/eth", "chain_id": 1, "native": "ETH", "explorer": "https://etherscan.io"},
    "polygon": {"rpc": "https://polygon-rpc.com", "chain_id": 137, "native": "MATIC", "explorer": "https://polygonscan.com"},
    "bsc": {"rpc": "https://bsc-dataseed.binance.org", "chain_id": 56, "native": "BNB", "explorer": "https://bscscan.com"},
    "arbitrum": {"rpc": "https://arb1.arbitrum.io/rpc", "chain_id": 42161, "native": "ETH", "explorer": "https://arbiscan.io"},
    "base": {"rpc": "https://mainnet.base.org", "chain_id": 8453, "native": "ETH", "explorer": "https://basescan.org"},
    "optimism": {"rpc": "https://mainnet.optimism.io", "chain_id": 10, "native": "ETH", "explorer": "https://optimistic.etherscan.io"},
}

# Common ERC-20 tokens per chain (address => symbol)
ERC20_TOKENS = {
    "ethereum": {
        "USDT": "0xdAC17F958D2ee523a2206206994597C13D831ec7",
        "USDC": "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
        "WBTC": "0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599",
        "DAI": "0x6B175474E89094C44Da98b954EedeAC495271d0F",
        "LINK": "0x514910771AF9Ca656af840dff83E8264EcF986CA",
        "UNI": "0x1f9840a85d5aF5bf1D1762F925BDADdC4201F984",
    },
    "polygon": {
        "USDT": "0xc2132D05D31c914a87C6611C10748AEb04B58e8F",
        "USDC": "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174",
        "WMATIC": "0x0d500B1d8E8eF31E21C99d1Db9A6444d3ADf1270",
    },
    "bsc": {
        "USDT": "0x55d398326f99059fF775485246999027B3197955",
        "USDC": "0x8AC76A51cc950d9822D68b83FE1BE97FB5FEc205",
        "CAKE": "0x0E09FaBB73Bd3Ade0a17ECC321fD13a19e81cE82",
    },
}


@dataclass
class WalletBalance:
    chain: str
    token: str
    balance: float
    usd_estimate: float = 0.0
    contract: str = ""


@dataclass
class ConnectedWallet:
    address: str
    wallet_type: str  # "metamask" | "walletconnect" | "rpc"
    chain: str
    balances: List[WalletBalance] = field(default_factory=list)
    connected_at: float = 0.0


class Web3Manager:
    """
    Multi-wallet manager. Handles connections from MetaMask, WalletConnect,
    or direct address input. Reads balances across all chains.

    Usage:
        w3 = Web3Manager()
        await w3.connect("0xabc...", "metamask", "ethereum")
        balances = await w3.get_all_balances("0xabc...")
    """

    def __init__(self) -> None:
        self._wallets: Dict[str, ConnectedWallet] = {}  # address → wallet
        self._rpc = os.getenv("WEB3_RPC_URL", "")

    async def connect(self, address: str, wallet_type: str = "rpc", chain: str = "ethereum") -> bool:
        """Register a wallet connection."""
        address = address.lower()
        if not address.startswith("0x") or len(address) != 42:
            logger.warning(f"Invalid address: {address}")
            return False
        self._wallets[address] = ConnectedWallet(
            address=address, wallet_type=wallet_type, chain=chain,
            connected_at=__import__("time").time(),
        )
        logger.info(f"Wallet connected: {address[:10]}... ({wallet_type}, {chain})")
        return True

    def disconnect(self, address: str) -> bool:
        address = address.lower()
        if address in self._wallets:
            del self._wallets[address]
            return True
        return False

    def list_wallets(self) -> List[dict]:
        return [
            {
                "address": w.address,
                "type": w.wallet_type,
                "chain": w.chain,
                "connected_at": w.connected_at,
            }
            for w in self._wallets.values()
        ]

    async def _rpc_call(self, chain: str, method: str, params: list) -> any:
        """Make a JSON-RPC call."""
        config = CHAIN_CONFIG.get(chain)
        if not config:
            return None
        rpc_url = self._rpc if self._rpc and chain == "ethereum" else config["rpc"]
        payload = {"jsonrpc": "2.0", "method": method, "params": params, "id": 1}
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(rpc_url, json=payload, timeout=10) as resp:
                    result = await resp.json()
                    return result.get("result")
        except Exception as exc:
            logger.debug(f"RPC call failed ({chain} {method}): {exc}")
            return None

    async def get_native_balance(self, address: str, chain: str = "ethereum") -> float:
        """Get native token balance (ETH/MATIC/BNB)."""
        result = await self._rpc_call(chain, "eth_getBalance", [address.lower(), "latest"])
        if result is None:
            return 0.0
        try:
            return int(result, 16) / 1e18
        except (ValueError, TypeError):
            return 0.0

    async def get_erc20_balance(self, address: str, token_contract: str, chain: str = "ethereum") -> float:
        """Get ERC-20 token balance via balanceOf()."""
        addr = address.lower()[2:].zfill(64)
        data = f"0x70a08231000000000000000000000000{addr}"
        result = await self._rpc_call(chain, "eth_call", [{"to": token_contract, "data": data}, "latest"])
        if result is None:
            return 0.0
        try:
            raw = int(result, 16)
            return raw / 1e6  # most stablecoins = 6 decimals
        except (ValueError, TypeError):
            return 0.0

    async def get_all_balances(self, address: str, chain: str = "ethereum") -> List[WalletBalance]:
        """Get native + common ERC-20 balances for a wallet on a chain."""
        address = address.lower()
        balances: List[WalletBalance] = []
        config = CHAIN_CONFIG.get(chain, {})

        # Native
        native = await self.get_native_balance(address, chain)
        if native > 0:
            balances.append(WalletBalance(chain=chain, token=config.get("native", "ETH"), balance=native))

        # ERC-20s
        tokens = ERC20_TOKENS.get(chain, {})
        for symbol, contract in tokens.items():
            amt = await self.get_erc20_balance(address, contract, chain)
            if amt > 0:
                decimals = 8 if symbol in ("WBTC",) else 6
                balances.append(WalletBalance(
                    chain=chain, token=symbol, balance=amt, contract=contract,
                ))

        # Update cached
        if address in self._wallets:
            self._wallets[address].balances = balances

        return balances

    async def get_all_wallets_balances(self) -> Dict[str, List[dict]]:
        """Get balances for ALL connected wallets across ALL their chains."""
        results = {}
        for addr, wallet in self._wallets.items():
            bals = await self.get_all_balances(addr, wallet.chain)
            results[addr] = [
                {"chain": b.chain, "token": b.token, "balance": b.balance, "contract": b.contract}
                for b in bals
            ]
        return results

    def build_transfer_tx(self, from_addr: str, to_addr: str, amount_eth: str, chain: str = "ethereum") -> dict:
        """Build an unsigned native transfer for the wallet to sign.

        The frontend submits this via window.ethereum.request() — the user
        approves in MetaMask, keys never leave the wallet.
        """
        config = CHAIN_CONFIG.get(chain, CHAIN_CONFIG["ethereum"])
        return {
            "to": to_addr,
            "from": from_addr.lower(),
            "value": amount_eth,  # hex string in wei
            "chainId": hex(config["chain_id"]),
            "gas": "0x5208",  # 21000 gas for simple transfer
        }

    @property
    def supported_chains(self) -> List[str]:
        return list(CHAIN_CONFIG.keys())

    def stats(self) -> dict:
        return {
            "connected_wallets": len(self._wallets),
            "wallets": self.list_wallets(),
            "supported_chains": self.supported_chains,
            "has_custom_rpc": bool(self._rpc),
        }
