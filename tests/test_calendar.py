import datetime as dt

from common import calendar_nse as cal


def test_weekend_and_holiday_are_not_trading_days():
    assert not cal.is_trading_day(dt.date(2025, 8, 15))   # Independence Day
    assert not cal.is_trading_day(dt.date(2025, 8, 16))   # Saturday
    assert cal.is_trading_day(dt.date(2025, 8, 14))       # Thursday


def test_add_trading_days_skips_holidays_and_weekends():
    # 2025-08-14 Thu; +1 should skip 15th (holiday) + weekend -> Mon 18th
    assert cal.add_trading_days(dt.date(2025, 8, 14), 1) == dt.date(2025, 8, 18)


def test_next_and_previous_trading_day():
    assert cal.next_trading_day(dt.date(2025, 8, 8)) == dt.date(2025, 8, 11)   # Fri -> Mon
    assert cal.previous_trading_day(dt.date(2025, 8, 11)) == dt.date(2025, 8, 8)


def test_trading_days_until_excludes_holidays():
    # 14th -> 20th: 18,19,20 are trading (15 holiday, 16-17 weekend) = 3
    assert cal.trading_days_until(dt.date(2025, 8, 20), dt.date(2025, 8, 14)) == 3
    assert cal.trading_days_until(dt.date(2025, 8, 14), dt.date(2025, 8, 14)) == 0
