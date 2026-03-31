from __future__ import annotations

import unittest

from trading_bot.circuit_breaker import CircuitBreaker


class _Clock:
    def __init__(self, now: float = 100.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class CircuitBreakerTests(unittest.TestCase):
    def test_closed_open_half_open_closed_cycle(self) -> None:
        clock = _Clock()
        breaker = CircuitBreaker(2, 30.0, clock=clock)

        self.assertTrue(breaker.allow_request())
        self.assertFalse(breaker.record_failure("boom-1"))
        self.assertTrue(breaker.allow_request())
        self.assertTrue(breaker.record_failure("boom-2"))
        self.assertEqual(breaker.state, "open")
        self.assertFalse(breaker.allow_request())

        clock.advance(31.0)
        self.assertTrue(breaker.allow_request())
        self.assertEqual(breaker.state, "half_open")

        breaker.record_success()
        self.assertEqual(breaker.state, "closed")
        self.assertEqual(breaker.failure_count, 0)

    def test_half_open_failure_reopens_breaker(self) -> None:
        clock = _Clock()
        breaker = CircuitBreaker(1, 10.0, clock=clock)

        self.assertTrue(breaker.record_failure("boom"))
        self.assertEqual(breaker.state, "open")
        clock.advance(11.0)
        self.assertTrue(breaker.allow_request())
        self.assertEqual(breaker.state, "half_open")
        self.assertTrue(breaker.record_failure("boom-again"))
        self.assertEqual(breaker.state, "open")
