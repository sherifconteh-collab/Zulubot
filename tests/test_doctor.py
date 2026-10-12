import unittest

from trading_bot.doctor import FAIL, INFO, OK, WARN, diagnose, render, run


def levels(env):
    return [(f.level, f.text) for f in diagnose(env)]


def texts(env, level=None):
    return " | ".join(t for lv, t in levels(env) if level is None or lv == level)


class DoctorTests(unittest.TestCase):
    def test_empty_environment_is_safe_paper_with_nothing_blocking(self):
        found = diagnose({})
        self.assertEqual(found[0].level, OK)
        self.assertIn("PAPER", found[0].text)
        self.assertFalse([f for f in found if f.level in (FAIL, WARN)])

    def test_live_without_kalshi_key_warns_that_orders_stay_paper(self):
        self.assertIn("orders stay on paper", texts({"PAPER_MODE": "false"}, WARN))

    def test_live_with_a_key_says_real_orders(self):
        env = {"PAPER_MODE": "false", "KALSHI_KEY_ID": "abc",
               "KALSHI_PRIVATE_KEY": "-----BEGIN RSA PRIVATE KEY-----\\nxx\\n-----END RSA PRIVATE KEY-----"}
        self.assertIn("REAL orders", texts(env, WARN))

    def test_malformed_kalshi_key_is_a_failure(self):
        env = {"KALSHI_KEY_ID": "abc", "KALSHI_PRIVATE_KEY": "not a key"}
        self.assertIn("does not look like a PEM key", texts(env, FAIL))

    def test_half_configured_telegram_is_a_failure(self):
        self.assertIn("TELEGRAM_CHAT_ID is missing", texts({"TELEGRAM_BOT_TOKEN": "123:abc"}, FAIL))
        self.assertIn("TELEGRAM_BOT_TOKEN is missing", texts({"TELEGRAM_CHAT_ID": "42"}, FAIL))

    def test_example_placeholders_count_as_missing(self):
        env = {"TELEGRAM_BOT_TOKEN": "your_bot_token_from_botfather", "TELEGRAM_CHAT_ID": "your_chat_id"}
        text = texts(env)
        self.assertIn("Telegram: not set", text)
        self.assertIn("Still the example text", text)

    def test_coinbase_trading_needs_keys_but_data_does_not(self):
        self.assertIn("Coinbase: data only", texts({}, INFO))
        self.assertIn("COINBASE_API_KEY", texts({"COINBASE_TRADING_ENABLED": "true"}, FAIL))

    def test_railway_without_storage_dir_is_only_a_note(self):
        self.assertIn("reset on every redeploy", texts({"RAILWAY_ENVIRONMENT": "production"}, INFO))
        self.assertNotIn("reset on every redeploy",
                         texts({"RAILWAY_ENVIRONMENT": "production", "BOT_STORAGE_DIR": "/data"}))

    def test_unused_fmp_key_is_called_out(self):
        self.assertIn("FMP_API_KEY", texts({"FMP_API_KEY": "realkey123"}, INFO))

    def test_secret_values_are_never_printed(self):
        env = {"KALSHI_KEY_ID": "SECRET-ID-123", "TELEGRAM_BOT_TOKEN": "999:SECRETTOKEN",
               "TELEGRAM_CHAT_ID": "424242", "ANTHROPIC_API_KEY": "sk-ant-SECRET", "FMP_API_KEY": "FMPSECRET"}
        out = render(diagnose(env))
        for secret in ("SECRET-ID-123", "SECRETTOKEN", "424242", "sk-ant-SECRET", "FMPSECRET"):
            self.assertNotIn(secret, out)

    def test_exit_code_is_nonzero_only_for_failures(self):
        import contextlib
        import io
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(run({}), 0)
            self.assertEqual(run({"TELEGRAM_BOT_TOKEN": "1:a"}), 1)


if __name__ == "__main__":
    unittest.main()
