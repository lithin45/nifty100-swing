from backtest.costs import compute_trade_costs, round_trip_cost_pct


def test_delivery_costs_breakdown(settings):
    cb = compute_trade_costs(100_000, 105_000, settings.costs)
    # Delivery: STT 0.1% buy + 0.1% sell = 100 + 105 = 205
    assert abs(cb.stt - 205.0) < 1e-6
    # Delivery brokerage is zero by default
    assert cb.brokerage == 0.0
    # Stamp duty on buy only: 0.015% of 100000 = 15
    assert abs(cb.stamp_duty - 15.0) < 1e-6
    # Total is positive and dominated by STT + slippage
    assert cb.total > 200


def test_round_trip_pct_reasonable(settings):
    pct = round_trip_cost_pct(2000.0, settings.costs, quantity=50)
    assert 0.1 < pct < 1.0  # delivery round-trip well under 1%


def test_intraday_uses_flat_brokerage_cap(settings):
    settings.costs.segment = "intraday"
    cb = compute_trade_costs(1_000_000, 1_000_000, settings.costs)
    # 0.03% of 10L = 300 -> capped at flat 20 per leg = 40 total
    assert abs(cb.brokerage - 40.0) < 1e-6
