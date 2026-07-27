"""Rate limiter: per-IP hourly window, global daily cap, and window expiry."""

from __future__ import annotations

from app.main import RateLimiter

HOUR = 3_600.0
DAY = 86_400.0


def test_allows_up_to_the_per_ip_cap() -> None:
    limiter = RateLimiter(per_ip_hourly=3, daily_cap=100)
    assert [limiter.check("1.1.1.1", now=1000.0)[0] for _ in range(3)] == [True, True, True]
    allowed, reason = limiter.check("1.1.1.1", now=1000.0)
    assert allowed is False
    assert "3 messages for this hour" in reason


def test_ips_are_counted_independently() -> None:
    limiter = RateLimiter(per_ip_hourly=2, daily_cap=100)
    for _ in range(2):
        limiter.check("1.1.1.1", now=0.0)
    assert limiter.check("1.1.1.1", now=0.0)[0] is False
    assert limiter.check("2.2.2.2", now=0.0)[0] is True


def test_the_hourly_window_rolls() -> None:
    limiter = RateLimiter(per_ip_hourly=2, daily_cap=100)
    limiter.check("1.1.1.1", now=0.0)
    limiter.check("1.1.1.1", now=10.0)
    assert limiter.check("1.1.1.1", now=20.0)[0] is False

    # Just after the first stamp ages out, one slot frees up - not the whole allowance.
    assert limiter.check("1.1.1.1", now=HOUR + 1)[0] is True
    assert limiter.check("1.1.1.1", now=HOUR + 2)[0] is False


def test_the_global_cap_applies_across_ips() -> None:
    limiter = RateLimiter(per_ip_hourly=100, daily_cap=3)
    assert limiter.check("1.1.1.1", now=0.0)[0] is True
    assert limiter.check("2.2.2.2", now=0.0)[0] is True
    assert limiter.check("3.3.3.3", now=0.0)[0] is True

    allowed, reason = limiter.check("4.4.4.4", now=0.0)
    assert allowed is False
    assert "daily message cap" in reason


def test_the_global_cap_is_checked_before_the_per_ip_cap() -> None:
    # A blocked global request must not consume the caller's hourly allowance.
    limiter = RateLimiter(per_ip_hourly=5, daily_cap=1)
    limiter.check("1.1.1.1", now=0.0)
    assert limiter.check("2.2.2.2", now=0.0)[0] is False
    assert limiter.check("2.2.2.2", now=DAY + 1)[0] is True


def test_the_daily_window_rolls() -> None:
    limiter = RateLimiter(per_ip_hourly=100, daily_cap=2)
    limiter.check("1.1.1.1", now=0.0)
    limiter.check("1.1.1.1", now=1.0)
    assert limiter.check("1.1.1.1", now=2.0)[0] is False
    assert limiter.check("1.1.1.1", now=DAY + 2)[0] is True


def test_blocked_requests_are_not_counted() -> None:
    limiter = RateLimiter(per_ip_hourly=1, daily_cap=100)
    limiter.check("1.1.1.1", now=0.0)
    for _ in range(20):
        limiter.check("1.1.1.1", now=0.0)
    # If refusals were recorded, the window would never drain at the expected time.
    assert limiter.check("1.1.1.1", now=HOUR + 1)[0] is True


def test_the_refusal_message_says_when_to_retry() -> None:
    limiter = RateLimiter(per_ip_hourly=1, daily_cap=100)
    limiter.check("1.1.1.1", now=0.0)
    _, reason = limiter.check("1.1.1.1", now=HOUR - 600)
    assert "11 minute" in reason  # 600s remaining, rounded up
