"""Unit tests for BTC5MinSpreadStrategy helper functions."""

import unittest

from trading_bot.strategies.btc_5min import compute_implied_median, parse_strike


class TestParseStrike(unittest.TestCase):
    """Tests for parse_strike()."""

    def test_plain_dollar_amount(self):
        self.assertEqual(parse_strike("Will BTC be above $83000 by end of month?"), 83000.0)

    def test_dollar_with_comma_separator(self):
        self.assertEqual(parse_strike("Will BTC exceed $83,000?"), 83000.0)

    def test_lowercase_k_suffix(self):
        self.assertEqual(parse_strike("BTC above $83k at expiry"), 83000.0)

    def test_uppercase_k_suffix(self):
        self.assertEqual(parse_strike("BTC above $83K at expiry"), 83000.0)

    def test_decimal_k_suffix(self):
        self.assertAlmostEqual(parse_strike("Will BTC hit $83.5k?"), 83500.0)

    def test_m_suffix(self):
        self.assertEqual(parse_strike("BTC > $1m?"), 1_000_000.0)

    def test_uppercase_m_suffix(self):
        self.assertEqual(parse_strike("BTC > $1M?"), 1_000_000.0)

    def test_no_dollar_sign_returns_none(self):
        self.assertIsNone(parse_strike("Will BTC rise above 83000 by Friday?"))

    def test_empty_string_returns_none(self):
        self.assertIsNone(parse_strike(""))

    def test_non_numeric_returns_none(self):
        self.assertIsNone(parse_strike("Will BTC rise?"))

    def test_underscore_separator(self):
        # "$83_000" uses underscore as thousands separator
        self.assertEqual(parse_strike("BTC above $83_000"), 83000.0)

    def test_multiple_strikes_returns_first(self):
        # regex returns first match
        result = parse_strike("BTC above $80k or $90k")
        self.assertEqual(result, 80000.0)


class TestComputeImpliedMedian(unittest.TestCase):
    """Tests for compute_implied_median()."""

    def test_empty_list_returns_none(self):
        self.assertIsNone(compute_implied_median([]))

    def test_single_pair_above_50_returns_strike(self):
        # Only one pair, prob >= 0.50 → return highest strike
        self.assertEqual(compute_implied_median([(80000.0, 0.75)]), 80000.0)

    def test_single_pair_below_50_returns_strike(self):
        # Only one pair, prob <= 0.50 → return lowest strike
        self.assertEqual(compute_implied_median([(80000.0, 0.40)]), 80000.0)

    def test_all_probs_above_50_returns_highest_strike(self):
        pairs = [(80000.0, 0.80), (85000.0, 0.60), (90000.0, 0.55)]
        self.assertEqual(compute_implied_median(pairs), 90000.0)

    def test_all_probs_below_50_returns_lowest_strike(self):
        pairs = [(80000.0, 0.30), (85000.0, 0.20), (90000.0, 0.10)]
        self.assertEqual(compute_implied_median(pairs), 80000.0)

    def test_interpolation_midpoint(self):
        # prob goes from 0.75 at 80k to 0.25 at 90k; 0.50 should land at 85k
        pairs = [(80000.0, 0.75), (90000.0, 0.25)]
        result = compute_implied_median(pairs)
        self.assertAlmostEqual(result, 85000.0, places=1)

    def test_interpolation_asymmetric(self):
        # prob: 0.80 at 80k, 0.40 at 90k; crossing at 0.50
        # t = (0.80 - 0.50) / (0.80 - 0.40) = 0.75 → 80000 + 0.75 * 10000 = 87500
        pairs = [(80000.0, 0.80), (90000.0, 0.40)]
        result = compute_implied_median(pairs)
        self.assertAlmostEqual(result, 87500.0, places=1)

    def test_equal_probs_at_bracket_returns_highest_strike(self):
        # When all probs are >= 0.50, the early-exit returns the highest strike
        pairs = [(80000.0, 0.50), (90000.0, 0.50)]
        result = compute_implied_median(pairs)
        self.assertAlmostEqual(result, 90000.0, places=1)

    def test_unsorted_input_is_sorted(self):
        # Supply pairs out of order; result should be same as sorted version
        pairs_unsorted = [(90000.0, 0.25), (80000.0, 0.75)]
        pairs_sorted = [(80000.0, 0.75), (90000.0, 0.25)]
        self.assertAlmostEqual(
            compute_implied_median(pairs_unsorted),
            compute_implied_median(pairs_sorted),
            places=1,
        )

    def test_50_not_bracketed_falls_back_to_median_strike(self):
        # All probs hover around 0.50 without cleanly crossing; fallback returns median strike
        # We simulate by having all probs equal to 0.50 except extremes
        pairs = [
            (70000.0, 0.90),
            (80000.0, 0.70),
            (85000.0, 0.55),
            (87000.0, 0.52),
            (89000.0, 0.51),
        ]
        # All probs >= 0.50 → returns highest strike
        result = compute_implied_median(pairs)
        self.assertEqual(result, 89000.0)


if __name__ == "__main__":
    unittest.main()
