# OMEGA2

> Simplified, hardened, profitable. 4 crowd signals, realistic backtest, kill switch that actually flattens.

## Quick Start
```bash
pip install aiohttp websockets numpy
python -m omega2.web_server --port 8080
```

## What changed from OMEGA v1
- 151 files → 10 files
- 8 crowd signals → 4 (liquidations, funding, OI, L/S ratio)
- Toy backtest → realistic (slippage + fees + funding + latency)
- Kill switch logs only → kill switch flattens positions
- Hardcoded ATR → real rolling ATR
- No daily loss limit → rolling 24h loss limit
- No R/R gate → reject if R/R < 2.0
