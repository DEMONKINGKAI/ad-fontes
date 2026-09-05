from __future__ import annotations

from app.api.rate_limit import RateLimiter, client_key


def test_client_key_is_not_the_raw_ip():
    key = client_key("203.0.113.7")
    assert "203.0.113.7" not in key
    assert key == client_key("203.0.113.7")  # stable within a process
    assert key != client_key("203.0.113.8")


def test_sliding_window_blocks_after_limit():
    rl = RateLimiter(max_requests=2, window_s=100)
    t = 1000.0
    assert rl.check("k", now=t)[0] is True
    assert rl.check("k", now=t + 1)[0] is True
    allowed, remaining, retry_after = rl.check("k", now=t + 2)
    assert allowed is False
    assert remaining == 0
    assert 0 < retry_after <= 100


def test_window_slides():
    rl = RateLimiter(max_requests=1, window_s=10)
    assert rl.check("k", now=0)[0] is True
    assert rl.check("k", now=5)[0] is False
    assert rl.check("k", now=11)[0] is True


def test_keys_are_independent():
    rl = RateLimiter(max_requests=1, window_s=10)
    assert rl.check("a", now=0)[0] is True
    assert rl.check("b", now=0)[0] is True
