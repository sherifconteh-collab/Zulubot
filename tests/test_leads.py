from __future__ import annotations

import unittest

from trading_bot.leads import WATCH_REASONS, extract_leads


class ExtractLeadsTests(unittest.TestCase):
    def test_returns_empty_when_result_is_none_or_not_dict(self) -> None:
        self.assertEqual(extract_leads("crypto", None), [])
        self.assertEqual(extract_leads("crypto", "not-a-dict"), [])  # type: ignore[arg-type]
        self.assertEqual(extract_leads("crypto", []), [])  # type: ignore[arg-type]

    def test_returns_empty_when_no_decisions(self) -> None:
        self.assertEqual(extract_leads("crypto", {}), [])
        self.assertEqual(extract_leads("crypto", {"decisions": []}), [])

    def test_only_skip_buy_with_whitelisted_reason_becomes_lead(self) -> None:
        result = {
            "decisions": [
                {"strategy": "crypto_sentiment_funding", "symbol": "BTCUSDT", "action": "buy", "reason": "fear_threshold"},
                {"strategy": "crypto_sentiment_funding", "symbol": "ETHUSDT", "action": "skip_buy", "reason": "position_size_zero", "signals": {"fear_greed": 22}},
                {"strategy": "crypto_sentiment_funding", "symbol": "SOLUSDT", "action": "skip_buy", "reason": "ai_rejected", "signals": {}},
                {"strategy": "crypto_sentiment_funding", "symbol": "DOGEUSDT", "action": "skip_buy", "reason": "neutral_fear_greed"},
                {"strategy": "crypto_sentiment_funding", "symbol": "*", "action": "skip_buy", "reason": "position_size_zero"},
                {"strategy": "crypto_sentiment_funding", "symbol": "ADAUSDT", "action": "hold", "reason": "neutral_fear_greed"},
            ]
        }

        leads = extract_leads("crypto", result)

        symbols = [lead["symbol"] for lead in leads]
        self.assertEqual(symbols, ["ETHUSDT", "SOLUSDT"])
        self.assertEqual(leads[0]["reason"], "position_size_zero")
        self.assertEqual(leads[0]["signals"], {"fear_greed": 22})
        self.assertEqual(leads[0]["strategy"], "crypto")

    def test_dedupes_repeated_symbols(self) -> None:
        result = {
            "decisions": [
                {"symbol": "BTC-USD", "action": "skip_buy", "reason": "ai_rejected"},
                {"symbol": "BTC-USD", "action": "skip_buy", "reason": "position_size_zero"},
            ]
        }

        leads = extract_leads("predictions", result)

        self.assertEqual(len(leads), 1)
        self.assertEqual(leads[0]["symbol"], "BTC-USD")

    def test_stocks_includes_top_targets_not_already_bought(self) -> None:
        result = {
            "buys": ["AAPL"],
            "top_targets": ["AAPL", "MSFT", "GOOG"],
            "decisions": [
                {"symbol": "NVDA", "action": "skip_buy", "reason": "max_positions_reached"},
            ],
        }

        leads = extract_leads("stocks", result)
        symbols = [lead["symbol"] for lead in leads]

        # NVDA from decisions, then MSFT and GOOG from top_targets (AAPL excluded — already bought).
        self.assertIn("NVDA", symbols)
        self.assertIn("MSFT", symbols)
        self.assertIn("GOOG", symbols)
        self.assertNotIn("AAPL", symbols)

        msft = next(lead for lead in leads if lead["symbol"] == "MSFT")
        self.assertEqual(msft["reason"], "top_target_not_bought")

    def test_stocks_top_target_does_not_duplicate_decision_lead(self) -> None:
        result = {
            "top_targets": ["NVDA"],
            "decisions": [
                {"symbol": "NVDA", "action": "skip_buy", "reason": "ai_rejected"},
            ],
        }

        leads = extract_leads("stocks", result)
        self.assertEqual([lead["symbol"] for lead in leads], ["NVDA"])
        self.assertEqual(leads[0]["reason"], "ai_rejected")

    def test_global_state_reasons_excluded(self) -> None:
        result = {
            "decisions": [
                {"symbol": "ETHUSDT", "action": "skip_buy", "reason": "kill_switch_active"},
                {"symbol": "BTCUSDT", "action": "skip_buy", "reason": "risk_guardrails"},
            ]
        }
        self.assertEqual(extract_leads("crypto", result), [])

    def test_watch_reasons_constant_is_immutable_set(self) -> None:
        self.assertIsInstance(WATCH_REASONS, frozenset)
        self.assertIn("position_size_zero", WATCH_REASONS)


if __name__ == "__main__":
    unittest.main()
