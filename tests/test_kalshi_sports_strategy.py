"""Tests for the ESPN-driven Kalshi sports strategy (all network calls mocked)."""

from __future__ import annotations

import logging
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

from trading_bot.broker import PaperBroker
from trading_bot.clients_espn import (
    EspnGame,
    EspnTeam,
    american_to_prob,
    devig_two_way,
    parse_predictor,
    parse_record,
    parse_scoreboard,
)
from trading_bot.config import KalshiSportsConfig
from trading_bot.models import PortfolioState
from trading_bot.sports_model import (
    Signal,
    SourceCalibration,
    combine,
    evaluate_side,
    kalshi_fee,
    microprice,
)
from trading_bot.strategies.kalshi_sports import (
    KalshiSportsStrategy,
    _settlement_yes_value,
    build_signals,
    match_market_to_game,
    record_home_prob,
)

_NOW = datetime(2026, 10, 4, 12, 0, tzinfo=timezone.utc)


def _team(tid, location, name, abbr, record="8-4") -> EspnTeam:
    wp, gp = parse_record(record)
    return EspnTeam(tid, name, location, f"{location} {name}", name, abbr, record, wp, gp)


def _game(**overrides) -> EspnGame:
    defaults = dict(
        game_id="401",
        league="football/nfl",
        start_time=_NOW + timedelta(hours=5),
        state="pre",
        home=_team("1", "Baltimore", "Ravens", "BAL"),
        away=_team("2", "Buffalo", "Bills", "BUF"),
        home_moneyline=-110.0,
        away_moneyline=-110.0,
    )
    defaults.update(overrides)
    return EspnGame(**defaults)


def _market(side_team="Buffalo", yes_ask=0.40, no_ask=0.62, **overrides) -> dict:
    m = {
        "ticker": f"KXNFLGAME-26OCT04BUFBAL-{side_team[:3].upper()}",
        "event_ticker": "KXNFLGAME-26OCT04BUFBAL",
        "title": "Buffalo at Baltimore Winner?",
        "yes_sub_title": side_team,
        "yes_ask_dollars": str(yes_ask),
        "yes_bid_dollars": str(yes_ask - 0.01),
        "no_ask_dollars": str(no_ask),
        "no_bid_dollars": str(no_ask - 0.03),
        "volume": 5000,
        "status": "active",
    }
    m.update(overrides)
    return m


def _config(**overrides) -> KalshiSportsConfig:
    defaults = dict(enabled=True, leagues=["football/nfl"], min_trade_usd=1.0)
    defaults.update(overrides)
    return KalshiSportsConfig(**defaults)


def _strategy(config, games, markets, *, kalshi=None):
    logger = logging.getLogger("test_kalshi_sports")
    kalshi = kalshi or MagicMock()
    kalshi.get_markets.return_value = (markets, None)
    espn = MagicMock()
    espn.get_scoreboard.return_value = games
    espn.enrich_game.side_effect = lambda g: g
    return KalshiSportsStrategy(
        config=config, kalshi_client=kalshi, espn_client=espn, logger=logger,
    ), kalshi


def _run(strategy, state=None):
    state = state or PortfolioState(cash=10_000.0)
    state.peak_equity = state.cash
    broker = PaperBroker(True, logging.getLogger("test_kalshi_sports.broker"))
    result = strategy.run(state, broker=broker, risk_manager=MagicMock(), now=_NOW)
    return state, broker, result


class MathTests(unittest.TestCase):
    def test_american_odds(self):
        self.assertAlmostEqual(american_to_prob(-200), 2 / 3)
        self.assertAlmostEqual(american_to_prob(150), 0.4)
        self.assertIsNone(american_to_prob(None))

    def test_devig_sums_to_one(self):
        p = devig_two_way(-150, 130)
        self.assertAlmostEqual(p + devig_two_way(130, -150), 1.0)
        self.assertGreater(p, 0.5)

    def test_parse_record(self):
        self.assertEqual(parse_record("10-3"), (10 / 13, 13))
        self.assertEqual(parse_record("8-4-2")[1], 14)
        self.assertEqual(parse_record("garbage"), (None, 0))

    def test_record_model_favours_better_home_team(self):
        game = _game(home=_team("1", "A", "Aces", "ACE", "12-2"), away=_team("2", "B", "Bees", "BEE", "3-11"))
        self.assertGreater(record_home_prob(game, 0.55), 0.7)

    def test_equal_records_give_home_prior(self):
        game = _game(home=_team("1", "A", "Aces", "ACE", "7-7"), away=_team("2", "B", "Bees", "BEE", "7-7"))
        self.assertAlmostEqual(record_home_prob(game, 0.55), 0.55, places=3)

    def test_signals_use_book_not_records_when_line_exists(self):
        game = _game(home=_team("1", "A", "Aces", "ACE", "9-1"), away=_team("2", "B", "Bees", "BEE", "1-9"),
                     home_moneyline=300.0, away_moneyline=-380.0)
        names = {s.name for s in build_signals(game, "home", _market("Baltimore"), _config(), SourceCalibration())}
        self.assertIn("sportsbook", names)
        self.assertNotIn("record", names)

    def test_records_only_as_fallback(self):
        game = _game(home_moneyline=None, away_moneyline=None, predictor_home_prob=0.6)
        names = {s.name for s in build_signals(game, "home", _market("Baltimore"), _config(), SourceCalibration())}
        self.assertEqual(names, {"kalshi_book", "espn_predictor", "record"})

    def test_signal_flips_for_away_team(self):
        game = _game(home_moneyline=-200.0, away_moneyline=170.0)
        home = {s.name: s.p for s in build_signals(game, "home", _market("Baltimore"), _config(), SourceCalibration())}
        away = {s.name: s.p for s in build_signals(game, "away", _market("Buffalo"), _config(), SourceCalibration())}
        self.assertAlmostEqual(home["sportsbook"] + away["sportsbook"], 1.0)


class ModelTests(unittest.TestCase):
    def test_fee_matches_kalshi_formula(self):
        self.assertAlmostEqual(kalshi_fee(0.50), 0.02)   # 0.07*.25 = .0175 -> rounds up to $0.02
        self.assertAlmostEqual(kalshi_fee(0.90), 0.01)   # .0063 -> $0.01
        self.assertEqual(kalshi_fee(0.0), 0.0)

    def test_microprice_leans_toward_thin_side(self):
        self.assertAlmostEqual(microprice(0.40, 1000, 0.44, 10), 0.4396, places=3)  # heavy bid -> near ask
        self.assertIsNone(microprice(0.0, 5, 0.44, 5))

    def test_precise_source_dominates(self):
        post = combine([Signal("sharp", 0.60, 0.02), Signal("noisy", 0.40, 0.10)])
        self.assertAlmostEqual(post.p, 0.60, delta=0.02)

    def test_agreement_tightens_and_disagreement_widens(self):
        agree = combine([Signal("a", 0.60, 0.03), Signal("b", 0.60, 0.03)])
        clash = combine([Signal("a", 0.80, 0.03), Signal("b", 0.40, 0.03)])
        self.assertLess(agree.sd, 0.03)
        self.assertGreater(clash.sd, 0.10)
        self.assertFalse(agree.conflict)
        self.assertTrue(clash.conflict)

    def test_normal_sharp_vs_market_gap_is_not_a_conflict(self):
        self.assertFalse(combine([Signal("book", 0.58, 0.025), Signal("kalshi", 0.54, 0.03)]).conflict)

    def test_outlier_source_dropped_with_three_sources(self):
        post = combine([Signal("a", 0.55, 0.03), Signal("b", 0.56, 0.03), Signal("bad", 0.95, 0.03)])
        self.assertEqual(post.dropped, ["bad"])
        self.assertAlmostEqual(post.p, 0.555, delta=0.02)

    def test_no_bet_when_edge_eaten_by_fee(self):
        post = combine([Signal("book", 0.52, 0.02)])
        self.assertIsNone(evaluate_side(post, "yes", 0.52, 0.50, fee_rate=0.07, kelly_fraction=0.25))

    def test_stake_scales_with_edge_to_noise(self):
        solid = combine([Signal("book", 0.62, 0.02)])
        shaky = combine([Signal("book", 0.62, 0.12)])
        a = evaluate_side(solid, "yes", solid.p, 0.55, fee_rate=0.07, kelly_fraction=0.25)
        b = evaluate_side(shaky, "yes", shaky.p, 0.55, fee_rate=0.07, kelly_fraction=0.25)
        self.assertGreater(a.stake_fraction, 3 * b.stake_fraction)
        self.assertGreater(a.confidence, b.confidence)

    def test_calibration_learns_which_source_is_better(self):
        cal = SourceCalibration()
        for i in range(60):
            cal.record_forecast(f"T{i}", {"sportsbook": 0.9, "espn_predictor": 0.1})
            cal.record_outcome(f"T{i}", 1.0)
        self.assertLess(cal.sd("sportsbook"), 0.03)
        self.assertGreater(cal.sd("espn_predictor"), 0.2)

    def test_calibration_persists_and_handles_ties(self):
        import tempfile, pathlib
        path = pathlib.Path(tempfile.mkdtemp()) / "cal.json"
        cal = SourceCalibration(path)
        cal.record_forecast("T1", {"sportsbook": 0.5})
        cal.save()
        again = SourceCalibration(path)
        self.assertTrue(again.record_outcome("T1", 0.5))
        self.assertEqual(again.stats["sportsbook"]["n"], 1)
        self.assertFalse(again.record_outcome("unknown", 1.0))

    def test_settlement_value_handles_tie_and_open_markets(self):
        self.assertEqual(_settlement_yes_value({"status": "finalized", "result": "yes", "settlement_value_dollars": "1.0000"}), 1.0)
        self.assertEqual(_settlement_yes_value({"status": "finalized", "result": "", "settlement_value_dollars": "0.5000"}), 0.5)
        self.assertIsNone(_settlement_yes_value({"status": "active", "result": ""}))


class ParsingTests(unittest.TestCase):
    def test_scoreboard_and_predictor(self):
        payload = {"events": [{
            "id": "401", "date": "2026-10-04T17:00Z",
            "status": {"type": {"state": "pre"}},
            "competitions": [{
                "competitors": [
                    {"homeAway": "home", "team": {"id": "1", "displayName": "Baltimore Ravens", "name": "Ravens",
                                                  "location": "Baltimore", "abbreviation": "BAL"},
                     "records": [{"type": "total", "summary": "3-1"}]},
                    {"homeAway": "away", "team": {"id": "2", "displayName": "Buffalo Bills", "name": "Bills",
                                                  "location": "Buffalo", "abbreviation": "BUF"},
                     "records": [{"type": "total", "summary": "2-2"}]},
                ],
                "odds": [{"provider": {"name": "DraftKings"},
                          "homeTeamOdds": {"moneyLine": -140}, "awayTeamOdds": {"moneyLine": 120}}],
            }],
        }]}
        games = parse_scoreboard(payload, "football/nfl")
        self.assertEqual(len(games), 1)
        g = games[0]
        self.assertEqual((g.home.abbreviation, g.away.abbreviation), ("BAL", "BUF"))
        self.assertEqual((g.home_moneyline, g.away_moneyline), (-140.0, 120.0))
        self.assertEqual(g.odds_provider, "DraftKings")
        self.assertAlmostEqual(
            parse_predictor({"predictor": {"homeTeam": {"gameProjection": "62.0"},
                                           "awayTeam": {"gameProjection": "38.0"}}}), 0.62)


class MatchingTests(unittest.TestCase):
    def test_match_by_name_and_yes_team(self):
        game = _game()
        self.assertEqual(match_market_to_game(_market("Buffalo"), [game]), (game, "away"))
        self.assertEqual(match_market_to_game(_market("Baltimore"), [game]), (game, "home"))

    def test_match_by_ticker_when_names_missing(self):
        game = _game()
        market = _market("", title="", yes_sub_title="")
        market["ticker"] = "KXNFLGAME-26OCT04BUFBAL-BAL"
        self.assertEqual(match_market_to_game(market, [game]), (game, "home"))

    def test_match_kalshi_real_shape_titles(self):
        # Real Kalshi titles are just "<Team> wins"; matching must use the event ticker.
        game = _game(home=_team("1", "Los Angeles", "Rams", "LAR"), away=_team("2", "Buffalo", "Bills", "BUF"))
        market = {"ticker": "KXNFLGAME-26OCT12BUFLAR-LAR", "event_ticker": "KXNFLGAME-26OCT12BUFLAR",
                  "title": "Los Angeles R wins", "yes_sub_title": "Los Angeles R", "no_sub_title": "Los Angeles R"}
        self.assertEqual(match_market_to_game(market, [game]), (game, "home"))

    def test_series_games_disambiguated_by_time(self):
        g1 = _game(game_id="1", start_time=_NOW + timedelta(hours=5))
        g2 = _game(game_id="2", start_time=_NOW + timedelta(hours=29))
        market = _market("Baltimore", expected_expiration_time=(g2.start_time + timedelta(hours=3)).isoformat())
        self.assertEqual(match_market_to_game(market, [g1, g2]), (g2, "home"))

    def test_unrelated_market_not_matched(self):
        market = _market(title="Kansas City at Denver Winner?", yes_sub_title="Denver",
                         event_ticker="KXNFLGAME-26OCT04KCDEN", ticker="KXNFLGAME-26OCT04KCDEN-DEN")
        self.assertIsNone(match_market_to_game(market, [_game()]))


class StrategyTests(unittest.TestCase):
    def test_buys_underpriced_side_in_paper(self):
        # Book + ESPN say Baltimore (home) ~65%; Kalshi YES on Baltimore asks only 0.50.
        game = _game(home_moneyline=-190.0, away_moneyline=160.0, predictor_home_prob=0.66)
        markets = [_market("Baltimore", yes_ask=0.57, no_ask=0.45), _market("Buffalo", yes_ask=0.57, no_ask=0.45)]
        strat, _ = _strategy(_config(), [game], markets)
        state, _, result = _run(strat)

        self.assertEqual(result["status"], "ok")
        self.assertEqual(len(result["buys"]), 1, "one bet per game")
        pos = state.positions[result["buys"][0]]
        self.assertEqual(pos.strategy, "kalshi_sports")
        self.assertEqual(pos.meta["pick"], "Baltimore Ravens")
        self.assertEqual(pos.meta["venue"], "kalshi")
        self.assertTrue(result["predictions"])
        self.assertLess(state.cash, 10_000.0)

    def test_skips_when_no_edge(self):
        game = _game()  # even game, fair price
        strat, _ = _strategy(_config(), [game], [_market("Baltimore", yes_ask=0.52, no_ask=0.50)])
        state, _, result = _run(strat)
        self.assertEqual(result["buys"], [])
        self.assertEqual(state.positions, {})

    def test_skips_started_and_far_future_games(self):
        for game in (_game(state="in"), _game(start_time=_NOW + timedelta(days=5)),
                     _game(start_time=_NOW + timedelta(minutes=1))):
            game.home_moneyline, game.away_moneyline = -300.0, 250.0
            strat, _ = _strategy(_config(), [game], [_market("Baltimore", yes_ask=0.40, no_ask=0.62)])
            _, _, result = _run(strat)
            self.assertEqual(result["buys"], [])
            self.assertEqual(result["markets_matched"], 0)

    def test_conflicting_sources_do_not_trigger_a_big_bet(self):
        # Book says Baltimore ~80%, Kalshi prices it 30%: far more likely a bad
        # match/quote than a free lunch. A significance test flags the conflict and the bet is skipped.
        game = _game(home_moneyline=-400.0, away_moneyline=320.0)
        strat, _ = _strategy(_config(), [game], [_market("Baltimore", yes_ask=0.30, no_ask=0.72)])
        state, _, result = _run(strat)
        self.assertEqual(result["buys"], [])
        self.assertIn("sources_conflict", [d.get("reason") for d in result["decisions"]])

    def test_stake_limited_by_depth_at_the_ask(self):
        game = _game(home_moneyline=-190.0, away_moneyline=160.0, predictor_home_prob=0.66)
        market = _market("Baltimore", yes_ask=0.57, no_ask=0.45, yes_ask_size_fp="10.0", yes_bid_size_fp="10.0")
        strat, _ = _strategy(_config(), [game], [market])
        state, _, result = _run(strat)
        if result["buys"]:
            self.assertLessEqual(state.positions[result["buys"][0]].quantity, 10)

    def test_no_sportsbook_line_means_no_bet(self):
        game = _game(home_moneyline=None, away_moneyline=None, predictor_home_prob=0.75)
        strat, _ = _strategy(_config(), [game], [_market("Baltimore", yes_ask=0.40, no_ask=0.62)])
        _, _, result = _run(strat)
        self.assertEqual(result["buys"], [])
        self.assertIn("no_sportsbook_line", [d.get("reason") for d in result["decisions"]])

    def test_paper_entry_includes_fee(self):
        game = _game(home_moneyline=-190.0, away_moneyline=160.0, predictor_home_prob=0.66)
        strat, _ = _strategy(_config(), [game], [_market("Baltimore", yes_ask=0.57, no_ask=0.45)])
        state, _, result = _run(strat)
        pos = state.positions[result["buys"][0]]
        self.assertAlmostEqual(pos.entry_price, pos.meta["ask"] + pos.meta["fee"])

    def test_tie_settles_at_fifty_cents(self):
        game = _game(home_moneyline=-190.0, away_moneyline=160.0, predictor_home_prob=0.66)
        strat, kalshi = _strategy(_config(), [game], [_market("Baltimore", yes_ask=0.57, no_ask=0.45)])
        state, _, result = _run(strat)
        symbol = result["buys"][0]
        kalshi.get_market.return_value = {"status": "finalized", "result": "", "settlement_value_dollars": "0.5000"}
        strat.espn.get_scoreboard.return_value = []
        _run(strat, state)
        self.assertNotIn(symbol, state.positions)
        self.assertEqual(result["buys"][0], symbol)

    def test_forecasts_are_scored_after_settlement(self):
        game = _game(home_moneyline=-190.0, away_moneyline=160.0)
        strat, kalshi = _strategy(_config(), [game], [_market("Baltimore", yes_ask=0.57, no_ask=0.45)])
        _run(strat)
        self.assertTrue(strat.calibration.pending)
        kalshi.get_market.return_value = {"status": "finalized", "result": "yes", "settlement_value_dollars": "1.0000"}
        strat.espn.get_scoreboard.return_value = []
        _run(strat)
        self.assertIn("sportsbook", strat.calibration.stats)

    def test_does_not_bet_same_game_twice(self):
        game = _game(home_moneyline=-190.0, away_moneyline=160.0)
        strat, _ = _strategy(_config(), [game], [_market("Baltimore", yes_ask=0.57, no_ask=0.45)])
        state = PortfolioState(cash=10_000.0)
        _, _, first = _run(strat, state)
        _, _, second = _run(strat, state)
        self.assertEqual(len(first["buys"]), 1)
        self.assertEqual(second["buys"], [])

    def test_resolved_win_closes_at_one_dollar(self):
        game = _game(home_moneyline=-190.0, away_moneyline=160.0)
        strat, kalshi = _strategy(_config(), [game], [_market("Baltimore", yes_ask=0.57, no_ask=0.45)])
        state, broker, result = _run(strat)
        symbol = result["buys"][0]
        side = state.positions[symbol].meta["side"]
        cash_after_buy = state.cash

        kalshi.get_market.return_value = {"status": "finalized", "result": side}
        strat.espn.get_scoreboard.return_value = []
        _, _, result2 = _run(strat, state)
        self.assertEqual(result2["resolved_closed"], [symbol])
        self.assertNotIn(symbol, state.positions)
        self.assertGreater(state.cash, cash_after_buy)
        self.assertGreater(state.realized_pnl, 0)

    def test_resolved_loss_closes_at_zero(self):
        game = _game(home_moneyline=-190.0, away_moneyline=160.0)
        strat, kalshi = _strategy(_config(), [game], [_market("Baltimore", yes_ask=0.57, no_ask=0.45)])
        state, _, result = _run(strat)
        symbol = result["buys"][0]
        side = state.positions[symbol].meta["side"]
        kalshi.get_market.return_value = {"status": "settled", "result": "no" if side == "yes" else "yes"}
        strat.espn.get_scoreboard.return_value = []
        _run(strat, state)
        self.assertNotIn(symbol, state.positions)
        self.assertLess(state.realized_pnl, 0)

    def test_open_market_not_closed(self):
        game = _game(home_moneyline=-190.0, away_moneyline=160.0)
        strat, kalshi = _strategy(_config(), [game], [_market("Baltimore", yes_ask=0.57, no_ask=0.45)])
        state, _, result = _run(strat)
        kalshi.get_market.return_value = {"status": "active", "result": ""}
        strat.espn.get_scoreboard.return_value = []
        _, _, result2 = _run(strat, state)
        self.assertEqual(result2["resolved_closed"], [])
        self.assertIn(result["buys"][0], state.positions)

    def test_kill_switch_blocks_entries(self):
        game = _game(home_moneyline=-190.0, away_moneyline=160.0)
        strat, _ = _strategy(_config(), [game], [_market("Baltimore", yes_ask=0.57, no_ask=0.45)])
        state = PortfolioState(cash=10_000.0)
        state.kill_switch_engaged = True
        _, _, result = _run(strat, state)
        self.assertEqual(result["status"], "halted")
        self.assertEqual(result["buys"], [])

    def test_disabled(self):
        strat, _ = _strategy(_config(enabled=False), [], [])
        _, _, result = _run(strat)
        self.assertEqual(result["status"], "disabled")

    def test_exposure_cap_limits_total_bets(self):
        games = [
            _game(game_id=str(i),
                  home=_team(str(i), f"City{i}", f"Team{i}", f"H{i}X"),
                  away=_team(str(i + 100), f"Town{i}", f"Club{i}", f"A{i}X"),
                  home_moneyline=-190.0, away_moneyline=160.0)
            for i in range(5)
        ]
        markets = [
            _market(f"City{i}", yes_ask=0.57, no_ask=0.45,
                    title=f"Town{i} at City{i} Winner?",
                    ticker=f"KXNFLGAME-26OCT04A{i}XH{i}X-H{i}X",
                    event_ticker=f"KXNFLGAME-26OCT04A{i}XH{i}X")
            for i in range(5)
        ]
        strat, _ = _strategy(_config(capital_allocation_pct=0.003, max_position_pct=0.03), games, markets)
        state, broker, result = _run(strat)
        exposure = sum(p.quantity * p.entry_price for p in state.positions.values())
        self.assertLessEqual(exposure, 10_000 * 0.003 + 1.0)
        self.assertLess(len(result["buys"]), 5)


if __name__ == "__main__":
    unittest.main()


class OrderBookTests(unittest.TestCase):
    """Kalshi books list only bids; asks are the complement of the opposite side."""

    def test_new_layout_yes_ask_from_no_bid(self):
        from trading_bot.clients_kalshi import parse_top_of_book
        book = {"orderbook_fp": {
            "yes_dollars": [["0.4100", "50.0"], ["0.4300", "6385.10"]],
            "no_dollars": [["0.5300", "20.0"], ["0.5500", "10001.00"]],
        }}
        top = parse_top_of_book(book)
        self.assertEqual((top["yes_bid"], top["no_bid"]), (0.43, 0.55))
        self.assertAlmostEqual(top["yes_ask"], 0.45)
        self.assertAlmostEqual(top["no_ask"], 0.57)
        self.assertEqual(top["yes_ask_size"], 10001.0)

    def test_legacy_cents_layout(self):
        from trading_bot.clients_kalshi import parse_top_of_book
        top = parse_top_of_book({"orderbook": {"yes": [[40, 10], [43, 5]], "no": [[52, 7], [55, 3]]}})
        self.assertAlmostEqual(top["yes_ask"], 0.45)
        self.assertAlmostEqual(top["no_ask"], 0.57)

    def test_empty_side_gives_zero(self):
        from trading_bot.clients_kalshi import parse_top_of_book
        top = parse_top_of_book({"orderbook_fp": {"yes_dollars": [], "no_dollars": [["0.5000", "1"]]}})
        self.assertEqual(top["no_ask"], 0.0)
        self.assertAlmostEqual(top["yes_ask"], 0.5)

    def test_latency_arb_reads_new_layout(self):
        from trading_bot.strategies.latency_arb import LatencyArbStrategy
        strat = LatencyArbStrategy.__new__(LatencyArbStrategy)
        strat.logger = logging.getLogger("t")
        strat.kalshi = MagicMock()
        strat.kalshi.get_orderbook.return_value = {"orderbook_fp": {
            "yes_dollars": [["0.4300", "5"]], "no_dollars": [["0.5500", "5"]]}}
        yes_ask, no_ask = strat._get_market_prices("X")
        self.assertAlmostEqual(yes_ask, 0.45)
        self.assertAlmostEqual(no_ask, 0.57)
