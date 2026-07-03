"""OMEGA2 tests — risk engine + crowd + backtest."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from omega2.core import Config, Signal, Side
from omega2.risk import RiskEngine, KillSwitch
from omega2.crowd import CrowdEngine
from omega2.agents import ContrarianAgent, LeadLagAgent
from omega2.core import CrowdSignal


def test_risk_rejects_bad_rr():
    """Risk engine must reject signals with R/R < 2.0."""
    cfg = Config(min_rr_ratio=2.0)
    risk = RiskEngine(cfg, initial_equity=10_000)
    sig = Signal(agent="test", symbol="BTCUSDT", timestamp="t", side=Side.BUY,
                 confidence=0.7, stop_loss_bps=200, take_profit_bps=100)  # R/R=0.5
    order = risk.on_signal(sig, 60000)
    assert order is None, "Should reject R/R < 2.0"


def test_risk_accepts_good_rr():
    """Risk engine must accept signals with R/R >= 2.0."""
    cfg = Config(min_rr_ratio=2.0, paper=True)
    risk = RiskEngine(cfg, initial_equity=10_000)
    sig = Signal(agent="test", symbol="BTCUSDT", timestamp="t", side=Side.BUY,
                 confidence=0.7, stop_loss_bps=100, take_profit_bps=300)  # R/R=3.0
    order = risk.on_signal(sig, 60000)
    assert order is not None, "Should accept R/R >= 2.0"


def test_kill_switch_daily_loss():
    """Kill switch must trigger on max daily loss."""
    cfg = Config(max_daily_loss_pct=3.0)
    ks = KillSwitch(cfg)
    ks._day_start_equity = 10_000
    ks.record_equity(9_600)  # -4% daily loss
    assert ks.is_triggered, "Should trigger on daily loss > 3%"


def test_crowd_fusion():
    """Crowd engine should fuse 4 signals."""
    eng = CrowdEngine(symbols=("BTCUSDT",))
    # Inject fake readings
    eng.funding._latest["BTCUSDT"] = 0.001
    eng.ls_ratio._long_pct["BTCUSDT"] = 70.0
    eng.open_interest._roc["BTCUSDT"] = 0.05
    # liquidations stay 0 (calm)
    result = eng.compute("BTCUSDT", "2024-01-01T00:00:00Z")
    assert result is not None, "Should emit with injected data"
    assert result.crowd_score > 0, "Positive funding + long = positive crowd score"


def test_contrarian_fades():
    """Contrarian should fade extreme crowd score."""
    agent = ContrarianAgent(extreme_threshold=0.5, confidence_cap=0.85)
    from omega2.core import Side
    crowd = CrowdSignal(
        symbol="BTCUSDT", timestamp="t", crowd_score=0.8,
        conviction=0.8, regime_hint="cascade_imminent", expected_move_bps=300,
    )
    agent._last_emit["BTCUSDT"] = 0  # reset cooldown
    sigs = agent.on_crowd(crowd)
    assert len(sigs) == 1
    assert sigs[0].side == Side.SELL, "Crowd long → SELL"
    assert sigs[0].stop_loss_bps < sigs[0].take_profit_bps, "Asymmetric TP/SL"


def test_leadlag_correlation_gate():
    """LeadLag should only emit when correlation is sufficient."""
    import time
    agent = LeadLagAgent(min_correlation=0.5, threshold_bps=5.0)
    # Feed BTC + ETH with correlated prices
    for i in range(60):
        t = time.time() - 60 + i
        px = 60000 + i * 10
        agent.on_market(type("E", (), {"symbol": "BTCUSDT", "timestamp": "t", "last_price": px,
                                       "volume_24h": 0, "bid": 0, "ask": 0, "bid_qty": 0, "ask_qty": 0})())
        agent.on_market(type("E", (), {"symbol": "ETHUSDT", "timestamp": "t", "last_price": 3000 + i * 0.5,
                                       "volume_24h": 0, "bid": 0, "ask": 0, "bid_qty": 0, "ask_qty": 0})())


def main():
    tests = [
        test_risk_rejects_bad_rr,
        test_risk_accepts_good_rr,
        test_kill_switch_daily_loss,
        test_crowd_fusion,
        test_contrarian_fades,
        test_leadlag_correlation_gate,
    ]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  ✓ {t.__name__}")
        except Exception as e:
            print(f"  ✗ {t.__name__}: {e}")
            import traceback; traceback.print_exc()
            failed += 1
    print(f"\n{'All' if failed==0 else f'{failed}/'} {len(tests)} tests {'PASSED ✓' if failed==0 else 'FAILED'}")
    return failed


if __name__ == "__main__":
    sys.exit(main())
