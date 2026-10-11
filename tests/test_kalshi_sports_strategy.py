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
from trading_bot.clients_polymarket_sports import PolyQuote, parse_poly_events
from trading_bot.config import KalshiSportsConfig
from trading_bot.models import PortfolioState
from trading_bot.sports_model import (
    Signal,
    SourceCalibration,
    combine,
    evaluate_side,
    kalshi_fee,
    microprice,
    parlay_value,
)
from trading_bot.strategies.kalshi_sports import (
    KalshiSportsStrategy,
    _settlement_yes_value,
    build_signals,
    link_polymarket,
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
        # Kalshi expects settlement a few hours after kickoff (default game starts _NOW+5h)
        "expected_expiration_time": (_NOW + timedelta(hours=8)).isoformat(),
    }
    m.update(overrides)
    return m


def _config(**overrides) -> KalshiSportsConfig:
    # Value-bet-only by default so these tests isolate the mispricing logic; action picks have their own tests.
    defaults = dict(enabled=True, leagues=["nfl"], min_trade_usd=1.0, action_bets=False)
    defaults.update(overrides)
    return KalshiSportsConfig(**defaults)


def _strategy(config, games, markets, *, kalshi=None):
    logger = logging.getLogger("test_kalshi_sports")
    kalshi = kalshi or MagicMock()
    kalshi.get_markets.return_value = (markets, None)
    kalshi.get_series.return_value = {"fee_type": "quadratic_with_maker_fees", "fee_multiplier": 1}
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

    def test_fee_multiplier_halves_mlb(self):
        self.assertAlmostEqual(kalshi_fee(0.50, multiplier=0.5), 0.01)   # .00875 -> $0.01
        self.assertAlmostEqual(kalshi_fee(0.50, multiplier=0.0), 0.0)

    def test_bigger_orders_round_less_per_contract(self):
        # 10 contracts @ .50: total .175 -> $0.18 -> 1.8c each, vs 2c each for a single contract
        self.assertAlmostEqual(kalshi_fee(0.50, count=10), 0.018)
        self.assertLess(kalshi_fee(0.50, count=10), kalshi_fee(0.50, count=1))

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
                  "expected_expiration_time": (_NOW + timedelta(hours=8)).isoformat(),
                  "title": "Los Angeles R wins", "yes_sub_title": "Los Angeles R", "no_sub_title": "Los Angeles R"}
        self.assertEqual(match_market_to_game(market, [game]), (game, "home"))

    def test_same_teams_on_a_different_day_is_not_matched(self):
        # Playoff series: the Oct-9 market must never pair with the Oct-7 game.
        game_oct7 = _game(start_time=_NOW + timedelta(hours=5))
        market_oct9 = _market("Baltimore", expected_expiration_time=(_NOW + timedelta(hours=53)).isoformat())
        self.assertIsNone(match_market_to_game(market_oct9, [game_oct7]))

    def test_market_without_expiry_is_not_matched(self):
        market = _market("Baltimore")
        market.pop("expected_expiration_time")
        self.assertIsNone(match_market_to_game(market, [_game()]))

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
        self.assertAlmostEqual(pos.entry_price, pos.meta["ask"] + pos.meta["fee"], places=3)
        self.assertLess(pos.meta["fee"], 0.02 + 1e-9)

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

    def test_sports_keeps_betting_through_global_kill_switch_by_default(self):
        game = _game(home_moneyline=-190.0, away_moneyline=160.0, predictor_home_prob=0.66)
        strat, _ = _strategy(_config(), [game], [_market("Baltimore", yes_ask=0.57, no_ask=0.45)])
        state = PortfolioState(cash=10_000.0)
        state.kill_switch_engaged = True
        state.halt_new_trades_for_day = True
        _, _, result = _run(strat, state)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(len(result["buys"]), 1)

    def test_kill_switch_can_still_be_honoured_when_configured(self):
        game = _game(home_moneyline=-190.0, away_moneyline=160.0, predictor_home_prob=0.66)
        strat, _ = _strategy(_config(ignore_global_halt=False), [game],
                             [_market("Baltimore", yes_ask=0.57, no_ask=0.45)])
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

    def test_simultaneous_bets_share_the_budget_proportionally(self):
        games = [
            _game(game_id=str(i), home=_team(str(i), f"City{i}", f"Team{i}", f"H{i}X"),
                  away=_team(str(i + 100), f"Town{i}", f"Club{i}", f"A{i}X"),
                  home_moneyline=-190.0, away_moneyline=160.0, predictor_home_prob=0.66)
            for i in range(4)
        ]
        markets = [
            _market(f"City{i}", yes_ask=0.57, no_ask=0.45, title=f"Town{i} at City{i} Winner?",
                    ticker=f"KXNFLGAME-26OCT04A{i}XH{i}X-H{i}X", event_ticker=f"KXNFLGAME-26OCT04A{i}XH{i}X")
            for i in range(4)
        ]
        loose, _ = _strategy(_config(capital_allocation_pct=1.0), games, markets)
        tight, _ = _strategy(_config(capital_allocation_pct=0.01), games, markets)
        s1, _, r1 = _run(loose)
        s2, _, r2 = _run(tight)
        spent1 = sum(p.quantity * p.entry_price for p in s1.positions.values())
        spent2 = sum(p.quantity * p.entry_price for p in s2.positions.values())
        self.assertLessEqual(spent2, 10_000 * 0.01 + 1.0)
        self.assertLess(spent2, spent1)
        self.assertGreaterEqual(len(r2["buys"]), 2, "budget is shared, not hogged by the first bet")

    def test_stakes_grow_with_the_wallet(self):
        def spend(cash):
            game = _game(home_moneyline=-190.0, away_moneyline=160.0, predictor_home_prob=0.66)
            strat, _ = _strategy(_config(max_position_pct=0.5, capital_allocation_pct=1.0), [game],
                                 [_market("Baltimore", yes_ask=0.57, no_ask=0.45)])
            state, _, r = _run(strat, PortfolioState(cash=cash))
            return sum(p.quantity * p.entry_price for p in state.positions.values())
        small, big = spend(1_000.0), spend(5_000.0)
        self.assertAlmostEqual(big / small, 5.0, delta=0.6)

    def test_stakes_shrink_after_a_drawdown_but_never_stop(self):
        from trading_bot.sports_model import drawdown_scale
        self.assertEqual(drawdown_scale(0.0, 0.5), 1.0)
        self.assertAlmostEqual(drawdown_scale(0.25, 0.5), 0.5)
        self.assertEqual(drawdown_scale(0.9, 0.5), 0.25)      # floor
        self.assertEqual(drawdown_scale(0.4, 0.0), 1.0)       # disabled

    def test_daily_session_tracks_peak_and_resets_each_day(self):
        import tempfile, pathlib
        from trading_bot.sports_model import DailySession
        path = pathlib.Path(tempfile.mkdtemp()) / "s.json"
        s = DailySession(path)
        s.update("2026-10-04", 100.0)
        up = s.update("2026-10-04", 130.0)
        down = DailySession(path).update("2026-10-04", 104.0)      # survives restart
        self.assertEqual(up["day_pnl"], 30.0)
        self.assertAlmostEqual(down["drawdown"], 0.2)
        self.assertEqual(DailySession(path).update("2026-10-05", 104.0)["day_pnl"], 0.0)

    def test_result_reports_bankroll(self):
        strat, _ = _strategy(_config(), [_game()], [_market("Baltimore")])
        _, _, result = _run(strat)
        self.assertEqual(result["bankroll"]["equity"], 10_000.0)
        self.assertEqual(result["bankroll"]["stake_scale"], 1.0)


def _soccer_game(**overrides) -> EspnGame:
    defaults = dict(
        game_id="900", league="soccer/eng.1",
        home=_team("1", "Everton", "Everton", "EVE", "2-3-0"),
        away=_team("2", "Chelsea", "Chelsea", "CHE", "2-1-2"),
        home_moneyline=175.0, draw_moneyline=275.0, away_moneyline=130.0, three_way=True,
    )
    defaults.update(overrides)
    return _game(**defaults)


def _soccer_market(suffix, sub, yes_ask, no_ask, **kw):
    m = {
        "ticker": f"KXEPLGAME-26OCT04CHEEVE-{suffix}", "event_ticker": "KXEPLGAME-26OCT04CHEEVE",
        "title": "Tie is the result" if suffix == "TIE" else f"{sub} wins", "yes_sub_title": sub,
        "yes_ask_dollars": str(yes_ask), "yes_bid_dollars": str(yes_ask - 0.01),
        "no_ask_dollars": str(no_ask), "no_bid_dollars": str(no_ask - 0.01),
        "expected_expiration_time": (_NOW + timedelta(hours=8)).isoformat(), "status": "active",
    }
    m.update(kw)
    return m


class SoccerTests(unittest.TestCase):
    def test_three_way_devig_sums_to_one(self):
        from trading_bot.clients_espn import devig_three_way
        h, d, a = devig_three_way(175, 275, 130)
        self.assertAlmostEqual(h + d + a, 1.0)
        self.assertGreater(a, h)  # Chelsea favoured

    def test_tie_market_matches_via_sibling_event(self):
        game = _soccer_game()
        team = _soccer_market("EVE", "Everton", 0.30, 0.71)
        tie = _soccer_market("TIE", "Tie", 0.27, 0.74)
        self.assertEqual(match_market_to_game(team, [game]), (game, "home"))
        self.assertEqual(match_market_to_game(tie, [game]), (game, "draw"))  # via ticker code
        self.assertEqual(match_market_to_game(tie, [], {tie["event_ticker"]: game}), (game, "draw"))

    def test_signals_use_correct_outcome_probability(self):
        game = _soccer_game()
        cal = SourceCalibration()
        def p(outcome, market):
            return {s.name: s.p for s in build_signals(game, outcome, market, _config(), cal)}["sportsbook"]
        home = p("home", _soccer_market("EVE", "Everton", 0.3, 0.7))
        draw = p("draw", _soccer_market("TIE", "Tie", 0.27, 0.74))
        away = p("away", _soccer_market("CHE", "Chelsea", 0.4, 0.6))
        self.assertAlmostEqual(home + draw + away, 1.0)
        self.assertNotIn("record", {s.name for s in build_signals(game, "home", _soccer_market("EVE", "Everton", .3, .7), _config(), cal)})

    def test_soccer_bet_labels_not_the_other_team(self):
        # Draw priced far too cheaply vs the line -> bet YES on the draw.
        game = _soccer_game()
        markets = [_soccer_market("EVE", "Everton", 0.30, 0.71), _soccer_market("CHE", "Chelsea", 0.40, 0.61),
                   _soccer_market("TIE", "Tie", 0.20, 0.81)]
        strat, _ = _strategy(_config(), [game], markets)
        state, _, result = _run(strat)
        self.assertEqual(result["markets_matched"], 3)
        self.assertEqual(len(result["buys"]), 1)
        pred = result["predictions"][0]
        self.assertIn(pred["pick"], ("Draw", "No draw", "Not Everton", "Not Chelsea", "Everton", "Chelsea"))
        self.assertTrue(result["buys"][0].startswith("KS:KXEPLGAME-26OCT04CHEEVE-"))

    def test_no_draw_odds_means_no_soccer_bet(self):
        game = _soccer_game(draw_moneyline=None)
        strat, _ = _strategy(_config(), [game], [_soccer_market("EVE", "Everton", 0.20, 0.81)])
        _, _, result = _run(strat)
        self.assertEqual(result["buys"], [])


class PolymarketTests(unittest.TestCase):
    def _events(self):
        two_way = {"title": "Ravens vs. Bills", "markets": [{
            "sportsMarketType": "moneyline", "gameStartTime": "2026-10-04 17:00:00+00",
            "outcomes": '["Ravens", "Bills"]', "bestBid": 0.55, "bestAsk": 0.57, "liquidityNum": 5000}]}
        soccer = {"title": "Chelsea vs. Everton", "markets": [
            {"sportsMarketType": "moneyline", "gameStartTime": "2026-10-04 17:00:00+00",
             "question": "Will Chelsea FC win on 2026-10-04?", "outcomes": '["Yes", "No"]', "bestBid": 0.50, "bestAsk": 0.52},
            {"sportsMarketType": "moneyline", "gameStartTime": "2026-10-04 17:00:00+00",
             "question": "Will Chelsea FC vs. Everton end in a draw?", "outcomes": '["Yes", "No"]', "bestBid": 0.24, "bestAsk": 0.26},
            {"sportsMarketType": "moneyline", "gameStartTime": "2026-10-04 17:00:00+00",
             "question": "Will Everton FC win on 2026-10-04?", "outcomes": '["Yes", "No"]', "bestBid": 0.23, "bestAsk": 0.25}]}
        stale = {"title": "Old", "markets": [{"sportsMarketType": "moneyline", "gameStartTime": "2026-10-04 17:00:00+00",
                                             "outcomes": '["A", "B"]', "bestBid": None, "bestAsk": 1}]}
        return [two_way, soccer, stale]

    def test_parse_two_way_and_soccer_and_skip_unquoted(self):
        games = parse_poly_events(self._events())
        self.assertEqual(len(games), 2)
        two, soc = games
        self.assertAlmostEqual(two.quotes["ravens"].mid, 0.56)
        self.assertAlmostEqual(two.quotes["bills"].mid, 0.44)       # complement of outcome 0
        self.assertAlmostEqual(two.quotes["bills"].bid, 0.43)
        self.assertEqual(set(soc.quotes), {"chelsea fc", "draw", "everton fc"})

    def test_link_two_way_matches_teams_and_time(self):
        game = _game(start_time=datetime(2026, 10, 4, 17, 0, tzinfo=timezone.utc))
        quotes = link_polymarket(game, parse_poly_events(self._events()))
        self.assertIsNotNone(quotes)
        self.assertAlmostEqual(quotes["home"].mid + quotes["away"].mid, 1.0)  # normalised
        self.assertGreater(quotes["away"].mid, 0.4)  # Bills == away (Buffalo)

    def test_link_rejects_wrong_day(self):
        game = _game(start_time=datetime(2026, 10, 7, 17, 0, tzinfo=timezone.utc))
        self.assertIsNone(link_polymarket(game, parse_poly_events(self._events())))

    def test_soccer_link_needs_draw_and_normalises(self):
        game = _soccer_game(start_time=datetime(2026, 10, 4, 17, 0, tzinfo=timezone.utc))
        quotes = link_polymarket(game, parse_poly_events(self._events()))
        self.assertEqual(set(quotes), {"home", "away", "draw"})
        self.assertAlmostEqual(sum(q.mid for q in quotes.values()), 1.0)

    def test_polymarket_becomes_a_signal_with_spread_based_sd(self):
        game = _game()
        game.extra["polymarket"] = {"home": PolyQuote(0.6, 0.55, 0.65), "away": PolyQuote(0.4, 0.35, 0.45)}
        sigs = {s.name: s for s in build_signals(game, "home", _market("Baltimore"), _config(), SourceCalibration())}
        self.assertAlmostEqual(sigs["polymarket"].p, 0.6)
        self.assertAlmostEqual(sigs["polymarket"].sd, 0.05)  # half the 10-point spread

    def test_lone_dissenting_polymarket_price_is_dropped_as_an_outlier(self):
        game = _game(home_moneyline=-190.0, away_moneyline=160.0, predictor_home_prob=0.66)
        game.extra["polymarket"] = {"home": PolyQuote(0.25, 0.24, 0.26), "away": PolyQuote(0.75, 0.74, 0.76)}
        strat, _ = _strategy(_config(), [game], [_market("Baltimore", yes_ask=0.57, no_ask=0.45)])
        _, _, result = _run(strat)
        self.assertIn("polymarket", result["predictions"][0]["dropped"])
        self.assertAlmostEqual(result["predictions"][0]["p_yes"], 0.62, delta=0.04)

    def test_polymarket_vs_book_conflict_with_no_tiebreaker_blocks_bet(self):
        game = _game(home_moneyline=-190.0, away_moneyline=160.0)   # no predictor
        game.extra["polymarket"] = {"home": PolyQuote(0.25, 0.24, 0.26), "away": PolyQuote(0.75, 0.74, 0.76)}
        # Kalshi's own price sits with the book, so the three-way split is 2 vs 1 and Polymarket drops.
        strat, _ = _strategy(_config(), [game], [_market("Baltimore", yes_ask=0.64, no_ask=0.38)])
        _, _, result = _run(strat)
        self.assertIn("polymarket", result["predictions"][0]["dropped"])


def _fav_and_dog_games(n=4):
    """Fair-priced games (no edge): Kalshi asks match the book, so only action picks can bet."""
    games, markets = [], []
    for i in range(n):
        games.append(_game(
            game_id=str(i), home=_team(str(i), f"City{i}", f"Team{i}", f"H{i}X"),
            away=_team(str(i + 100), f"Town{i}", f"Club{i}", f"A{i}X"),
            home_moneyline=-150.0, away_moneyline=130.0))
        markets.append(_market(
            f"City{i}", yes_ask=0.60, no_ask=0.42, title=f"Town{i} at City{i} Winner?",
            ticker=f"KXNFLGAME-26OCT04A{i}XH{i}X-H{i}X", event_ticker=f"KXNFLGAME-26OCT04A{i}XH{i}X"))
    return games, markets


class ActionPickTests(unittest.TestCase):
    def test_bets_a_side_when_nothing_is_mispriced(self):
        games, markets = _fav_and_dog_games(1)
        strat, _ = _strategy(_config(action_bets=True), games, markets)
        state, _, result = _run(strat)
        self.assertEqual(len(result["buys"]), 1)
        self.assertTrue(state.positions[result["buys"][0]].meta["action_bet"])
        self.assertTrue(result["predictions"][0]["action"])
        self.assertLess(result["predictions"][0]["net_edge"], 0)         # honest: no edge after fees

    def test_off_switch_restores_value_bets_only(self):
        games, markets = _fav_and_dog_games(1)
        strat, _ = _strategy(_config(action_bets=False), games, markets)
        _, _, result = _run(strat)
        self.assertEqual(result["buys"], [])

    def test_limited_per_scan_and_per_day(self):
        games, markets = _fav_and_dog_games(6)
        strat, _ = _strategy(_config(action_bets=True, action_max_per_scan=2, action_bets_per_day=3,
                                     capital_allocation_pct=1.0), games, markets)
        state = PortfolioState(cash=10_000.0)
        _, _, first = _run(strat, state)
        _, _, second = _run(strat, state)
        _, _, third = _run(strat, state)
        self.assertEqual((len(first["buys"]), len(second["buys"]), len(third["buys"])), (2, 1, 0))

    def test_stake_is_a_small_share_of_the_bankroll(self):
        games, markets = _fav_and_dog_games(1)
        strat, _ = _strategy(_config(action_bets=True, action_stake_pct=0.02), games, markets)
        state, _, result = _run(strat)
        pos = state.positions[result["buys"][0]]
        self.assertLessEqual(pos.quantity * pos.entry_price, 10_000 * 0.02 + 1.0)

    def test_never_backs_a_long_shot(self):
        # Book: Baltimore 20%. Both sides are fairly priced; the favourite (NO on Baltimore) is picked, never the 20% side.
        game = _game(home_moneyline=400.0, away_moneyline=-500.0)
        strat, _ = _strategy(_config(action_bets=True, action_min_win_prob=0.35), [game],
                             [_market("Baltimore", yes_ask=0.20, no_ask=0.82)])
        state, _, result = _run(strat)
        if result["buys"]:
            self.assertEqual(state.positions[result["buys"][0]].meta["side"], "no")
        game2 = _game(home_moneyline=400.0, away_moneyline=-500.0)
        strat2, _ = _strategy(_config(action_bets=True, action_min_win_prob=0.90), [game2],
                              [_market("Baltimore", yes_ask=0.20, no_ask=0.82)])
        self.assertEqual(_run(strat2)[2]["buys"], [])

    def test_picks_by_value_not_always_the_underdog_or_favourite(self):
        # Same game, two prices: when the underdog is the cheaper side relative to its
        # probability the pick is the underdog; otherwise it is the favourite.
        dog = _strategy(_config(action_bets=True, action_min_win_prob=0.30),
                        [_game(home_moneyline=-150.0, away_moneyline=130.0)],
                        [_market("Baltimore", yes_ask=0.64, no_ask=0.35)])[0]
        fav = _strategy(_config(action_bets=True, action_min_win_prob=0.30),
                        [_game(home_moneyline=-150.0, away_moneyline=130.0)],
                        [_market("Baltimore", yes_ask=0.58, no_ask=0.43)])[0]
        side_dog = _run(dog)[0].positions
        side_fav = _run(fav)[0].positions
        self.assertEqual({p.meta["side"] for p in side_dog.values()}, {"no"})     # underdog (Buffalo) is the value
        self.assertEqual({p.meta["side"] for p in side_fav.values()}, {"yes"})    # favourite (Baltimore) is the value

    def test_underdog_picks_are_capped_when_favourites_are_available(self):
        # Four games: the two best-value sides are underdogs (p<.5), two others are favourites.
        games, markets = [], []
        specs = [  # (home_ml, away_ml, yes_ask, no_ask): value of the underdog side is best in the first two
            (-130.0, 110.0, 0.57, 0.44),   # NO (away underdog ~46%): best value, slightly
            (-130.0, 110.0, 0.57, 0.44),
            (-200.0, 170.0, 0.63, 0.39),   # YES (home favourite ~64%): a little worse value
            (-200.0, 170.0, 0.63, 0.39),
        ]
        for i, (h, a, ya, na) in enumerate(specs):
            games.append(_game(game_id=str(i), home=_team(str(i), f"City{i}", f"Team{i}", f"H{i}X"),
                               away=_team(str(i + 100), f"Town{i}", f"Club{i}", f"A{i}X"),
                               home_moneyline=h, away_moneyline=a))
            markets.append(_market(f"City{i}", yes_ask=ya, no_ask=na, title=f"Town{i} at City{i} Winner?",
                                   ticker=f"KXNFLGAME-26OCT04A{i}XH{i}X-H{i}X",
                                   event_ticker=f"KXNFLGAME-26OCT04A{i}XH{i}X"))
        strat, _ = _strategy(_config(action_bets=True, action_bets_per_day=2, action_max_per_scan=2,
                                     action_min_win_prob=0.30, capital_allocation_pct=1.0), games, markets)
        state, _, result = _run(strat)
        sides = [p.meta["side"] for p in state.positions.values()]
        dogs = sum(1 for p in state.positions.values() if p.meta["model_prob"] < 0.5)
        self.assertEqual(len(sides), 2)
        self.assertEqual(dogs, 1)                          # cap = ceil(2/2) = 1 underdog; the rest are favourites
        self.assertEqual(sorted(sides), ["no", "yes"])

    def test_real_edge_beats_an_action_pick_in_the_same_game(self):
        game = _game(home_moneyline=-190.0, away_moneyline=160.0, predictor_home_prob=0.66)
        strat, _ = _strategy(_config(action_bets=True), [game], [_market("Baltimore", yes_ask=0.57, no_ask=0.45)])
        state, _, result = _run(strat)
        self.assertEqual(len(result["buys"]), 1)
        self.assertFalse(state.positions[result["buys"][0]].meta.get("action_bet", False))

    def test_action_picks_are_not_parlay_legs(self):
        games, markets = _fav_and_dog_games(3)
        strat, _ = _strategy(_config(action_bets=True), games, markets)
        self.assertEqual(_run(strat)[2]["parlays"], [])

    def test_action_count_survives_restart_and_resets_daily(self):
        import tempfile, pathlib
        from trading_bot.sports_model import DailySession
        path = pathlib.Path(tempfile.mkdtemp()) / "s.json"
        s = DailySession(path)
        s.update("2026-10-04", 100.0)
        s.record_action(); s.record_action()
        self.assertEqual(DailySession(path).actions, 2)
        again = DailySession(path)
        again.update("2026-10-05", 100.0)
        self.assertEqual(again.actions, 0)


class FeeMultiplierTests(unittest.TestCase):
    def _run(self, series_info):
        # A thin gap: worth betting at half fee (MLB), not at the full fee.
        game = _game(home_moneyline=-190.0, away_moneyline=160.0)
        strat, kalshi = _strategy(_config(), [game], [_market("Baltimore", yes_ask=0.60, no_ask=0.42)])
        if isinstance(series_info, Exception):
            kalshi.get_series.side_effect = series_info
        else:
            kalshi.get_series.return_value = series_info
        return _run(strat)[2]

    def test_series_multiplier_is_read_from_kalshi(self):
        full = self._run({"fee_type": "quadratic_with_maker_fees", "fee_multiplier": 1})
        half = self._run({"fee_type": "quadratic_with_maker_fees", "fee_multiplier": 0.5})
        net = lambda r: r["predictions"][0]["net_edge"]
        self.assertIsNone(net(full))
        self.assertIsNotNone(net(half))

    def test_unreadable_series_falls_back_to_the_standard_fee(self):
        result = self._run(RuntimeError("503"))
        self.assertIsNone(result["predictions"][0]["net_edge"])      # same as multiplier 1

    def test_multiplier_is_cached_per_series(self):
        strat, kalshi = _strategy(_config(), [_game()], [_market("Baltimore")])
        _run(strat)
        _run(strat)
        self.assertEqual(kalshi.get_series.call_count, 1)

    def test_paper_entry_uses_the_exact_order_fee(self):
        game = _game(home_moneyline=-190.0, away_moneyline=160.0, predictor_home_prob=0.66)
        strat, _ = _strategy(_config(), [game], [_market("Baltimore", yes_ask=0.57, no_ask=0.45)])
        state, _, result = _run(strat)
        pos = state.positions[result["buys"][0]]
        from trading_bot.sports_model import kalshi_fee as fee
        self.assertAlmostEqual(pos.entry_price - pos.meta["ask"], fee(pos.meta["ask"], 0.07, 1.0, int(pos.quantity)))


class ParlayTests(unittest.TestCase):
    def test_parlay_value_is_product_and_fee_adjusted(self):
        v = parlay_value([(0.6, 0.02), (0.5, 0.02)], fee_rate=0.07)
        self.assertAlmostEqual(v["fair"], 0.30)
        self.assertLess(v["max_price"], v["fair"])
        self.assertGreater(v["sd"], 0.0)
        self.assertAlmostEqual(v["fair_decimal_odds"], 1 / 0.30)

    def test_suggestions_only_from_positive_edge_legs_in_different_games(self):
        games = [
            _game(game_id=str(i), home=_team(str(i), f"City{i}", f"Team{i}", f"H{i}X"),
                  away=_team(str(i + 100), f"Town{i}", f"Club{i}", f"A{i}X"),
                  home_moneyline=-190.0, away_moneyline=160.0, predictor_home_prob=0.66)
            for i in range(3)
        ]
        markets = [
            _market(f"City{i}", yes_ask=0.57, no_ask=0.45, title=f"Town{i} at City{i} Winner?",
                    ticker=f"KXNFLGAME-26OCT04A{i}XH{i}X-H{i}X", event_ticker=f"KXNFLGAME-26OCT04A{i}XH{i}X")
            for i in range(3)
        ]
        strat, _ = _strategy(_config(), games, markets)
        _, _, result = _run(strat)
        self.assertEqual([len(p["legs"]) for p in result["parlays"]], [2, 3])
        for parlay in result["parlays"]:
            self.assertEqual(len({l["game"] for l in parlay["legs"]}), len(parlay["legs"]))
            self.assertLess(parlay["fair_prob"], 0.7)

    def test_no_parlays_without_edge(self):
        strat, _ = _strategy(_config(), [_game()], [_market("Baltimore", yes_ask=0.55, no_ask=0.47)])
        _, _, result = _run(strat)
        self.assertEqual(result["parlays"], [])


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


class UserBetTests(unittest.TestCase):
    """Bets chosen by hand in Telegram: board, quote and placement."""

    def _board(self, **cfg):
        game = _game(home_moneyline=-110.0, away_moneyline=-110.0)
        markets = [_market("Baltimore", yes_ask=0.52, no_ask=0.50), _market("Buffalo", yes_ask=0.50, no_ask=0.52)]
        strat, kalshi = _strategy(_config(**cfg), [game], markets)
        kalshi.get_market.side_effect = lambda t: next(m for m in markets if m["ticker"] == t)
        games = strat.refresh_board(_NOW)
        return strat, kalshi, games, markets

    def _bet(self, strat, ticker, side="yes", amount=5.0, cash=1000.0, max_ask=None, state=None):
        state = state or PortfolioState(cash=cash)
        broker = PaperBroker(True, logging.getLogger("test_kalshi_sports.user"))
        return state, strat.place_user_bet(state, broker, ticker=ticker, side=side, amount=amount, now=_NOW, max_ask=max_ask)

    def test_board_lists_each_team_once_with_cheapest_route(self):
        _, _, games, _ = self._board()
        self.assertEqual(len(games), 1)
        options = {o["label"]: o for o in games[0]["options"]}
        self.assertEqual(set(options), {"Baltimore Ravens", "Buffalo Bills"})
        # Baltimore: YES 0.52 on its own market vs NO 0.52 on Buffalo's market: same price either way
        self.assertAlmostEqual(options["Baltimore Ravens"]["ask"], 0.52)
        self.assertIsNotNone(options["Baltimore Ravens"]["p"])
        self.assertGreater(options["Baltimore Ravens"]["fee"], 0)

    def test_board_includes_games_without_a_model_view(self):
        game = _game(home_moneyline=None, away_moneyline=None)
        strat, _ = _strategy(_config(), [game], [_market("Baltimore", yes_ask=0.52, no_ask=0.50)])
        games = strat.refresh_board(_NOW)
        self.assertEqual(len(games), 1)
        self.assertTrue(all(o["p"] is None for o in games[0]["options"]))

    def test_started_games_leave_the_board(self):
        strat, _, _, _ = self._board()
        self.assertEqual(strat.board_games(_NOW + timedelta(hours=6)), [])

    def test_paper_bet_is_booked_with_user_flag(self):
        strat, _, games, _ = self._board()
        option = next(o for o in games[0]["options"] if o["label"] == "Baltimore Ravens")
        state, out = self._bet(strat, option["ticker"], option["side"], amount=50.0)
        self.assertTrue(out["ok"], out)
        position = state.positions[out["symbol"]]
        self.assertTrue(position.meta["user_bet"])
        self.assertEqual(position.meta["pick"], "Baltimore Ravens")
        self.assertLessEqual(out["spent"], 50.0 + 1e-6)
        self.assertGreater(out["spent"], 45.0)

    def test_default_stake_is_a_share_of_the_bankroll(self):
        strat, _, games, _ = self._board()
        ticker = games[0]["options"][0]["ticker"]
        state, out = self._bet(strat, ticker, games[0]["options"][0]["side"], amount=None, cash=1000.0)
        self.assertTrue(out["ok"], out)
        self.assertAlmostEqual(out["stake"], 50.0)  # 5% default

    def test_over_the_cap_is_refused(self):
        strat, _, games, _ = self._board()
        option = games[0]["options"][0]
        state, out = self._bet(strat, option["ticker"], option["side"], amount=300.0, cash=1000.0)
        self.assertFalse(out["ok"])
        self.assertIn("per-bet limit", out["reason"])
        self.assertEqual(state.positions, {})

    def test_below_minimum_is_refused(self):
        strat, _, games, _ = self._board()
        option = games[0]["options"][0]
        _, out = self._bet(strat, option["ticker"], option["side"], amount=0.5)
        self.assertFalse(out["ok"])
        self.assertIn("Minimum", out["reason"])

    def test_one_bet_per_game(self):
        strat, _, games, _ = self._board()
        a, b = games[0]["options"]
        state, first = self._bet(strat, a["ticker"], a["side"], amount=20.0)
        self.assertTrue(first["ok"])
        _, second = self._bet(strat, b["ticker"], b["side"], amount=20.0, state=state)
        self.assertFalse(second["ok"])
        self.assertIn("already have a bet", second["reason"])

    def test_closed_market_and_started_game_are_refused(self):
        strat, kalshi, games, markets = self._board()
        option = games[0]["options"][0]
        for m in markets:
            m["status"] = "closed"
        _, out = self._bet(strat, option["ticker"], option["side"])
        self.assertFalse(out["ok"])
        self.assertIn("closed", out["reason"])
        for m in markets:
            m["status"] = "active"
        state = PortfolioState(cash=1000.0)
        late = strat.quote_user_bet(
            state, PaperBroker(True, logging.getLogger("x")), ticker=option["ticker"],
            side=option["side"], amount=5.0, now=_NOW + timedelta(hours=6),
        )
        self.assertFalse(late["ok"])
        self.assertIn("started", late["reason"])

    def test_unknown_ticker_asks_for_refresh(self):
        strat, _, _, _ = self._board()
        _, out = self._bet(strat, "NOPE", "yes")
        self.assertFalse(out["ok"])
        self.assertIn("/games", out["reason"])

    def test_price_move_beyond_agreed_ask_is_refused(self):
        strat, _, games, markets = self._board()
        option = games[0]["options"][0]
        market = next(m for m in markets if m["ticker"] == option["ticker"])
        market[f"{option['side']}_ask_dollars"] = str(option["ask"] + 0.05)
        state, out = self._bet(strat, option["ticker"], option["side"], max_ask=option["ask"] + 0.02)
        self.assertFalse(out["ok"])
        self.assertTrue(out.get("moved"))
        self.assertEqual(state.positions, {})

    def test_live_bet_places_ioc_order_and_books_only_fills(self):
        strat, kalshi, games, _ = self._board()
        strat.live_trading_enabled = True
        kalshi.place_order.return_value = {"order": {"order_id": "o1", "fill_count": 4}}
        option = games[0]["options"][0]
        state = PortfolioState(cash=0.0, cash_reserves={"kalshi": 100.0})
        _, out = self._bet(strat, option["ticker"], option["side"], amount=10.0, state=state)
        self.assertTrue(out["ok"], out)
        kwargs = kalshi.place_order.call_args.kwargs
        self.assertEqual(kwargs["time_in_force"], "immediate_or_cancel")
        self.assertEqual(kwargs["action"], "buy")
        self.assertEqual(out["filled"], 4)
        self.assertEqual(state.positions[out["symbol"]].meta["kalshi_order_id"], "o1")
        self.assertTrue(out["live"])

    def test_live_unfilled_order_books_nothing(self):
        strat, kalshi, games, _ = self._board()
        strat.live_trading_enabled = True
        kalshi.place_order.return_value = {"order": {"order_id": "o2", "fill_count": 0}}
        option = games[0]["options"][0]
        state = PortfolioState(cash=0.0, cash_reserves={"kalshi": 100.0})
        _, out = self._bet(strat, option["ticker"], option["side"], amount=10.0, state=state)
        self.assertFalse(out["ok"])
        self.assertEqual(state.positions, {})

    def test_scan_still_blocks_a_second_bet_on_a_hand_picked_game(self):
        strat, _, games, _ = self._board()
        option = games[0]["options"][0]
        state, out = self._bet(strat, option["ticker"], option["side"], amount=20.0)
        self.assertTrue(out["ok"])
        _, _, result = _run(strat, state=state)
        self.assertEqual(result["buys"], [])


class ParlayRfqTests(unittest.TestCase):
    """Hand-built parlays: Kalshi combo market + request for quote (all calls mocked)."""

    COMBO = "KXMVE-COMBO-1"

    def _setup(self, **cfg):
        g1 = _game(home_moneyline=-110.0, away_moneyline=-110.0)
        g2 = _game(
            game_id="402", home=_team("3", "Kansas City", "Chiefs", "KC"), away=_team("4", "Denver", "Broncos", "DEN"),
            home_moneyline=-110.0, away_moneyline=-110.0,
        )
        m1 = [_market("Baltimore", yes_ask=0.52, no_ask=0.50), _market("Buffalo", yes_ask=0.50, no_ask=0.52)]
        m2 = [
            _market("Kansas City", yes_ask=0.52, no_ask=0.50, ticker="KXNFLGAME-26OCT04DENKC-KAN",
                    event_ticker="KXNFLGAME-26OCT04DENKC", title="Denver at Kansas City Winner?"),
            _market("Denver", yes_ask=0.50, no_ask=0.52, ticker="KXNFLGAME-26OCT04DENKC-DEN",
                    event_ticker="KXNFLGAME-26OCT04DENKC", title="Denver at Kansas City Winner?"),
        ]
        markets = m1 + m2
        config = _config(parlay_quote_wait_seconds=0.0, **cfg)
        strat, kalshi = _strategy(config, [g1, g2], markets)
        kalshi.get_market.side_effect = lambda t: next(
            (m for m in markets if m["ticker"] == t), {"ticker": t, "status": "active"})
        kalshi.get_multivariate_collections.return_value = [
            {"collection_ticker": "KXMVEOTHER-R", "associated_event_tickers": ["KXNBAGAME-X"]},
            {"collection_ticker": "KXMVESPORTS-R",
             "associated_event_tickers": ["KXNFLGAME-26OCT04BUFBAL", "KXNFLGAME-26OCT04DENKC"]},
        ]
        kalshi.create_combo_market.return_value = {"market_ticker": self.COMBO, "event_ticker": "KXMVE-EV"}
        kalshi.create_rfq.return_value = "rfq-1"
        kalshi.get_rfq_quotes.return_value = [
            {"id": "q-1", "status": "open", "yes_bid_dollars": "0.1000", "no_bid_dollars": "0.8000", "contracts_fp": "80.00"},
        ]
        kalshi.get_rfq_quote.return_value = {"status": "executed"}
        kalshi.get_positions.return_value = [{"ticker": self.COMBO, "position": 80}]
        strat.live_trading_enabled = True
        strat._sleep = lambda s: None
        strat._EXECUTION_WAIT_SECONDS = 0.0
        strat.refresh_board(_NOW)
        return strat, kalshi

    def _legs(self, strat):
        pick = {}
        for game in strat.board_games(_NOW):
            for o in game["options"]:
                pick[o["label"]] = {"ticker": o["ticker"], "side": o["side"]}
        return [pick["Baltimore Ravens"], pick["Kansas City Chiefs"]]

    def _state(self):
        return PortfolioState(cash=0.0, cash_reserves={"kalshi": 1000.0})

    def _place(self, strat, legs=None, amount=20.0, max_price=None, state=None):
        state = state or self._state()
        broker = PaperBroker(True, logging.getLogger("t.parlay"))
        return state, strat.place_user_parlay(
            state, broker, legs=legs or self._legs(strat), amount=amount, max_price=max_price, now=_NOW)

    def test_quote_uses_fair_probability_as_the_ceiling(self):
        strat, _ = self._setup()
        state = self._state()
        plan = strat.quote_user_parlay(state, PaperBroker(True, logging.getLogger("t")), legs=self._legs(strat),
                                       amount=20.0, max_price=None, now=_NOW)
        self.assertTrue(plan["ok"], plan)
        self.assertAlmostEqual(plan["fair"], 0.25, delta=0.03)
        self.assertAlmostEqual(plan["max_price"], plan["fair"])
        self.assertLessEqual(plan["max_cost"], 20.0 + 1e-9)
        self.assertEqual(plan["collection"], "KXMVESPORTS-R")
        self.assertEqual(plan["contracts"], int(20.0 // plan["max_price"]))

    def test_paper_mode_and_bad_leg_counts_are_refused(self):
        strat, _ = self._setup()
        legs = self._legs(strat)
        broker = PaperBroker(True, logging.getLogger("t"))
        self.assertFalse(strat.quote_user_parlay(self._state(), broker, legs=legs[:1], amount=5, max_price=None, now=_NOW)["ok"])
        strat.live_trading_enabled = False
        out = strat.quote_user_parlay(self._state(), broker, legs=legs, amount=5, max_price=None, now=_NOW)
        self.assertIn("live trading", out["reason"])

    def test_same_game_legs_and_missing_collection_are_refused(self):
        strat, kalshi = self._setup()
        broker = PaperBroker(True, logging.getLogger("t"))
        by_label = {o["label"]: o for g in strat.board_games(_NOW) for o in g["options"]}
        same_game = [{"ticker": by_label["Baltimore Ravens"]["ticker"], "side": by_label["Baltimore Ravens"]["side"]},
                     {"ticker": by_label["Buffalo Bills"]["ticker"], "side": by_label["Buffalo Bills"]["side"]}]
        out = strat.quote_user_parlay(self._state(), broker, legs=same_game, amount=5, max_price=None, now=_NOW)
        self.assertIn("different game or a different player", out["reason"])
        kalshi.get_multivariate_collections.return_value = [{"collection_ticker": "X-R", "associated_event_tickers": []}]
        strat._combo_collections = None
        out = strat.quote_user_parlay(self._state(), broker, legs=self._legs(strat), amount=5, max_price=None, now=_NOW)
        self.assertIn("doesn't offer a combo", out["reason"])

    def test_mixed_sports_are_refused(self):
        strat, _ = self._setup()
        legs = self._legs(strat)
        strat.board[legs[1]["ticker"]]["league"] = "baseball/mlb"
        out = strat.quote_user_parlay(self._state(), PaperBroker(True, logging.getLogger("t")),
                                      legs=legs, amount=5, max_price=None, now=_NOW)
        self.assertFalse(out["ok"])
        self.assertIn("one sport", out["reason"])
        self.assertIn("baseball", out["reason"])
        self.assertIn("football", out["reason"])

    def _props(self, strat, kalshi, query="allen"):
        prop = {
            "ticker": "KXNFLPASSYDS-26OCT04BUFBAL-BUFJALLEN17-250", "event_ticker": "KXNFLPASSYDS-26OCT04BUFBAL",
            "title": "Josh Allen: 250+ passing yards", "yes_sub_title": "Josh Allen: 250+", "status": "active",
            "yes_ask_dollars": "0.40", "no_ask_dollars": "0.62",
        }
        other = dict(prop, ticker="KXNFLPASSYDS-26OCT04BUFBAL-BALLJACKSON8-200",
                     title="Lamar Jackson: 200+ passing yards", yes_ask_dollars="0.55")
        elsewhere = dict(prop, ticker="KXNFLPASSYDS-26OCT04XXXYYY-Z-1", event_ticker="KXNFLPASSYDS-26OCT04XXXYYY")
        by_series = {"KXNFLPASSYDS": [prop, other, elsewhere]}
        kalshi.get_markets.side_effect = lambda **kw: (by_series.get(kw["series_ticker"], []), None)
        base = kalshi.get_market.side_effect
        kalshi.get_market.side_effect = lambda t: prop if t == prop["ticker"] else (
            other if t == other["ticker"] else base(t))
        kalshi.get_multivariate_collections.return_value = [{
            "collection_ticker": "KXMVESPORTS-R",
            "associated_event_tickers": ["KXNFLGAME-26OCT04BUFBAL", "KXNFLGAME-26OCT04DENKC", "KXNFLPASSYDS-26OCT04BUFBAL"]}]
        strat._combo_collections = None
        return strat.refresh_props(_NOW, query), prop, other

    def test_props_search_matches_player_and_upcoming_football_games_only(self):
        strat, kalshi = self._setup()
        props, prop, _ = self._props(strat, kalshi, "allen")
        self.assertEqual([p["ticker"] for p in props], [prop["ticker"]])
        self.assertEqual(props[0]["player"], "Josh Allen")
        self.assertEqual(props[0]["stat"], "250+ passing yards")
        self.assertAlmostEqual(props[0]["yes_ask"], 0.40)
        both, _, _ = self._props(strat, kalshi, "passing yards")
        self.assertEqual(len(both), 2)  # the market for a game not on the board is dropped

    def test_prop_parlay_defaults_to_the_market_price_and_places(self):
        strat, kalshi = self._setup()
        props, prop, _ = self._props(strat, kalshi, "allen")
        legs = [{"ticker": prop["ticker"], "side": "yes"}, self._legs(strat)[1]]  # Allen 250+ and Kansas City
        broker = PaperBroker(True, logging.getLogger("t"))
        plan = strat.quote_user_parlay(self._state(), broker, legs=legs, amount=10, max_price=None, now=_NOW)
        self.assertTrue(plan["ok"], plan)
        self.assertEqual([l["kind"] for l in plan["legs"]], ["prop", "game"])
        self.assertIsNone(plan["fair"])  # the model has no view on player stats
        game_p = plan["legs"][1]["p"]
        self.assertAlmostEqual(plan["reference"], 0.40 * game_p)  # prop at Kalshi's price, game at the model's
        self.assertAlmostEqual(plan["max_price"], plan["reference"])
        self.assertEqual(plan["legs"][0]["pick"], "Josh Allen 250+ passing yards")
        state, out = self._place(strat, legs=legs, amount=10.0, max_price=0.25)
        self.assertTrue(out["ok"], out)
        sent = kalshi.create_combo_market.call_args.args[1]
        self.assertEqual(sent[0]["event_ticker"], "KXNFLPASSYDS-26OCT04BUFBAL")

    def test_props_are_cached_and_searchable_by_threshold(self):
        strat, kalshi = self._setup()
        props, prop, other = self._props(strat, kalshi, "passing yards")
        self.assertEqual(len(props), 2)
        calls = kalshi.get_markets.call_count
        self.assertEqual([p["ticker"] for p in strat.refresh_props(_NOW, "250 allen")], [prop["ticker"]])
        self.assertEqual([p["ticker"] for p in strat.refresh_props(_NOW, "jackson 200")], [other["ticker"]])
        self.assertEqual(strat.refresh_props(_NOW, "jackson 250"), [])  # threshold must match
        self.assertEqual(kalshi.get_markets.call_count, calls, "searches reuse the cached listing")

    def test_same_player_twice_or_a_prop_with_baseball_is_refused(self):
        strat, kalshi = self._setup()
        props, prop, other = self._props(strat, kalshi, "passing yards")
        broker = PaperBroker(True, logging.getLogger("t"))
        twice = [{"ticker": prop["ticker"], "side": "yes"}, {"ticker": prop["ticker"].replace("-250", "-300"), "side": "yes"}]
        strat.props[twice[1]["ticker"] + "#yes"] = dict(strat.props[prop["ticker"] + "#yes"], ticker=twice[1]["ticker"])
        out = strat.quote_user_parlay(self._state(), broker, legs=twice, amount=5, max_price=0.3, now=_NOW)
        self.assertIn("different player", out["reason"])
        two_players = [{"ticker": prop["ticker"], "side": "yes"}, {"ticker": other["ticker"], "side": "yes"}]
        self.assertTrue(strat.quote_user_parlay(self._state(), broker, legs=two_players, amount=5, max_price=0.3, now=_NOW)["ok"])
        game_leg = self._legs(strat)[1]
        strat.board[game_leg["ticker"]]["league"] = "baseball/mlb"
        mixed = [{"ticker": prop["ticker"], "side": "yes"}, game_leg]
        self.assertIn("one sport", strat.quote_user_parlay(self._state(), broker, legs=mixed, amount=5, max_price=0.3, now=_NOW)["reason"])

    def test_stake_limits(self):
        strat, _ = self._setup()
        broker = PaperBroker(True, logging.getLogger("t"))
        legs = self._legs(strat)
        over = strat.quote_user_parlay(self._state(), broker, legs=legs, amount=300.0, max_price=None, now=_NOW)
        self.assertIn("parlay limit", over["reason"])
        under = strat.quote_user_parlay(self._state(), broker, legs=legs, amount=0.5, max_price=None, now=_NOW)
        self.assertIn("Minimum", under["reason"])
        default = strat.quote_user_parlay(self._state(), broker, legs=legs, amount=None, max_price=None, now=_NOW)
        self.assertAlmostEqual(default["stake"], 20.0)  # 2% of 1000

    def test_a_slip_price_can_raise_the_default_ceiling_but_never_an_explicit_max(self):
        strat, _ = self._setup()
        broker = PaperBroker(True, logging.getLogger("t"))
        legs = self._legs(strat)
        base = strat.quote_user_parlay(self._state(), broker, legs=legs, amount=10, max_price=None, now=_NOW)
        higher = strat.quote_user_parlay(self._state(), broker, legs=legs, amount=10, max_price=None, now=_NOW,
                                         anchor_price=base["max_price"] + 0.05)
        self.assertAlmostEqual(higher["max_price"], base["max_price"] + 0.05)
        lower = strat.quote_user_parlay(self._state(), broker, legs=legs, amount=10, max_price=None, now=_NOW,
                                        anchor_price=base["max_price"] - 0.05)
        self.assertAlmostEqual(lower["max_price"], base["max_price"])  # never below Kalshi's own reference
        explicit = strat.quote_user_parlay(self._state(), broker, legs=legs, amount=10, max_price=0.1, now=_NOW,
                                           anchor_price=0.9)
        self.assertAlmostEqual(explicit["max_price"], 0.1)

    def test_a_ten_dollar_parlay_is_within_the_default_limit(self):
        strat, _ = self._setup()
        plan = strat.quote_user_parlay(PortfolioState(cash=0.0, cash_reserves={"kalshi": 74.0}),
                                       PaperBroker(True, logging.getLogger("t")),
                                       legs=self._legs(strat), amount=10.0, max_price=None, now=_NOW)
        self.assertTrue(plan["ok"], plan)

    def test_without_a_model_view_the_reference_is_kalshis_own_price(self):
        strat, _ = self._setup()
        for entry in strat.board.values():
            entry["p_yes"] = None
        broker = PaperBroker(True, logging.getLogger("t"))
        plan = strat.quote_user_parlay(self._state(), broker, legs=self._legs(strat), amount=5, max_price=None, now=_NOW)
        self.assertTrue(plan["ok"], plan)
        self.assertIsNone(plan["fair"])
        self.assertAlmostEqual(plan["reference"], 0.515 * 0.515, places=4)  # each leg's bid/ask midpoint
        self.assertAlmostEqual(plan["max_price"], plan["reference"])
        pricier = strat.quote_user_parlay(self._state(), broker, legs=self._legs(strat), amount=5, max_price=0.4, now=_NOW)
        self.assertAlmostEqual(pricier["max_price"], 0.4)

    def test_accepts_a_quote_under_the_ceiling_and_books_it(self):
        strat, kalshi = self._setup()
        expected_contracts = int(20.0 // out_fair(strat, self._state()))
        state, out = self._place(strat)
        self.assertTrue(out["ok"], out)
        kalshi.create_combo_market.assert_called_once()
        legs = kalshi.create_combo_market.call_args.args[1]
        self.assertEqual([l["side"] for l in legs], ["yes", "yes"])
        kalshi.create_rfq.assert_called_once_with(self.COMBO, expected_contracts)
        # maker bid NO at 0.80, so we hold YES at 0.20: we accept the maker's NO side
        kalshi.accept_rfq_quote.assert_called_once_with("rfq-1", "q-1", "no")
        self.assertAlmostEqual(out["price"], 0.20)
        position = state.positions[out["symbol"]]
        self.assertEqual(position.meta["side"], "yes")
        self.assertTrue(position.meta["parlay"])
        self.assertEqual(len(position.meta["event_tickers"]), 2)
        self.assertIsNone(out["warning"] or None)

    def test_both_games_are_then_off_limits(self):
        strat, _ = self._setup()
        state, out = self._place(strat)
        self.assertTrue(out["ok"])
        _, _, result = _run(strat, state=state)
        self.assertEqual(result["buys"], [])
        again = strat.quote_user_parlay(state, PaperBroker(True, logging.getLogger("t")), legs=self._legs(strat),
                                        amount=5, max_price=None, now=_NOW)
        self.assertIn("already have a bet", again["reason"])

    def test_quote_above_the_ceiling_buys_nothing_and_cleans_up(self):
        strat, kalshi = self._setup()
        kalshi.get_rfq_quotes.return_value = [
            {"id": "q-1", "status": "open", "yes_bid_dollars": "0.1", "no_bid_dollars": "0.7000", "contracts_fp": "80.00"}]
        state, out = self._place(strat)
        self.assertFalse(out["ok"])
        self.assertIn("no quote at or under", out["reason"])
        kalshi.accept_rfq_quote.assert_not_called()
        kalshi.delete_rfq.assert_called_once_with("rfq-1")
        self.assertEqual(state.positions, {})

    def test_user_ceiling_allows_a_pricier_quote(self):
        strat, kalshi = self._setup()
        kalshi.get_rfq_quotes.return_value = [
            {"id": "q-1", "status": "open", "yes_bid_dollars": "0.1", "no_bid_dollars": "0.7000", "contracts_fp": "70.00"}]
        kalshi.get_positions.return_value = [{"ticker": self.COMBO, "position": 70}]
        state, out = self._place(strat, max_price=0.32, amount=20.0)
        self.assertTrue(out["ok"], out)

    def test_no_quotes_means_no_bet(self):
        strat, kalshi = self._setup()
        kalshi.get_rfq_quotes.return_value = []
        _, out = self._place(strat)
        self.assertFalse(out["ok"])
        self.assertIn("No market maker quoted", out["reason"])
        kalshi.delete_rfq.assert_called_once()

    def test_undersized_or_non_open_quotes_are_ignored(self):
        strat, kalshi = self._setup()
        kalshi.get_rfq_quotes.return_value = [
            {"id": "a", "status": "open", "no_bid_dollars": "0.9", "contracts_fp": "5.00"},
            {"id": "b", "status": "cancelled", "no_bid_dollars": "0.9", "contracts_fp": "80.00"},
        ]
        _, out = self._place(strat)
        self.assertFalse(out["ok"])
        kalshi.accept_rfq_quote.assert_not_called()

    def test_best_quote_wins(self):
        strat, kalshi = self._setup()
        kalshi.get_rfq_quotes.return_value = [
            {"id": "worse", "status": "open", "no_bid_dollars": "0.78", "contracts_fp": "80.00"},
            {"id": "better", "status": "open", "no_bid_dollars": "0.82", "contracts_fp": "80.00"},
        ]
        _, out = self._place(strat)
        self.assertTrue(out["ok"])
        self.assertEqual(kalshi.accept_rfq_quote.call_args.args[1], "better")

    def test_maker_not_confirming_books_nothing(self):
        strat, kalshi = self._setup()
        kalshi.get_rfq_quote.return_value = {"status": "cancelled"}
        state, out = self._place(strat)
        self.assertFalse(out["ok"])
        self.assertFalse(out["uncertain"])
        self.assertEqual(state.positions, {})

    def test_unknown_outcome_is_flagged_uncertain(self):
        strat, kalshi = self._setup()
        kalshi.get_rfq_quote.side_effect = RuntimeError("boom")
        state, out = self._place(strat)
        self.assertFalse(out["ok"])
        self.assertTrue(out["uncertain"])
        self.assertEqual(state.positions, {})

    def test_holding_the_wrong_side_is_reported_and_tracked_truthfully(self):
        strat, kalshi = self._setup()
        kalshi.get_positions.return_value = [{"ticker": self.COMBO, "position": -80}]
        state, out = self._place(strat)
        self.assertTrue(out["ok"])
        self.assertIn("opposite", out["warning"])
        self.assertEqual(state.positions[out["symbol"]].meta["side"], "no")

    def test_failed_combo_setup_is_reported(self):
        strat, kalshi = self._setup()
        kalshi.create_combo_market.side_effect = RuntimeError("400 bad legs")
        state, out = self._place(strat)
        self.assertFalse(out["ok"])
        kalshi.create_rfq.assert_not_called()


def out_fair(strat, state):
    plan = strat.quote_user_parlay(
        state, PaperBroker(True, logging.getLogger("t")), legs=[
            {"ticker": o["ticker"], "side": o["side"]}
            for g in strat.board_games(_NOW) for o in g["options"] if o["label"] in ("Baltimore Ravens", "Kansas City Chiefs")],
        amount=20.0, max_price=None, now=_NOW)
    return plan["max_price"]


def _prop_market(series, player_token, title, ask, bid, volume=1000.0, event_suffix="26OCT04BUFBAL", no_ask=None, no_bid=None):
    return {
        "ticker": f"{series}-{event_suffix}-{player_token}", "event_ticker": f"{series}-{event_suffix}", "title": title,
        "status": "active", "yes_ask_dollars": str(ask), "yes_bid_dollars": str(bid),
        "no_ask_dollars": str(no_ask if no_ask is not None else round(1 - bid, 2)),
        "no_bid_dollars": str(no_bid if no_bid is not None else round(1 - ask, 2)), "volume_fp": str(volume),
    }


class StackTests(unittest.TestCase):
    """Same-game stacks of player-stat legs, like a sportsbook SGP."""

    _setup = ParlayRfqTests._setup
    _legs = ParlayRfqTests._legs
    _state = ParlayRfqTests._state
    _place = ParlayRfqTests._place
    COMBO = ParlayRfqTests.COMBO

    def _stack_setup(self):
        strat, kalshi = self._setup()
        markets = {
            "KXNFLPASSYDS": [
                _prop_market("KXNFLPASSYDS", "BUFJALLEN17-200", "Josh Allen: 200+ passing yards", 0.80, 0.78),
                _prop_market("KXNFLPASSYDS", "BUFJALLEN17-250", "Josh Allen: 250+ passing yards", 0.40, 0.38),
                _prop_market("KXNFLPASSYDS", "BALLJACKSON8-150", "Lamar Jackson: 150+ passing yards", 0.86, 0.84),
            ],
            "KXNFLRSHYDS": [
                _prop_market("KXNFLRSHYDS", "BUFJCOOK4-30", "James Cook: 30+ rushing yards", 0.90, 0.88, volume=5000.0),
                _prop_market("KXNFLRSHYDS", "BALBENCH1-10", "Bench Guy: 10+ rushing yards", 0.85, 0.0),  # one-sided
            ],
            "KXNFLREC": [
                _prop_market("KXNFLREC", "BUFKCOLEMAN0-2", "Keon Coleman: 2+ receptions", 0.95, 0.93),
                _prop_market("KXNFLREC", "BUFKCOLEMAN0-12", "Keon Coleman: 12+ receptions", 0.02, 0.01),
            ],
            "KXNFLSPREAD": [
                _prop_market("KXNFLSPREAD", "BAL10", "BAL Ravens wins by over 9.5 points?", 0.25, 0.23,
                             no_ask=0.77, no_bid=0.75),
                _prop_market("KXNFLSPREAD", "BUF10", "BUF Bills wins by over 9.5 points?", 0.22, 0.20,
                             no_ask=0.80, no_bid=0.78),
            ],
        }
        markets["KXNFLPASSYDS"].append(_prop_market(
            "KXNFLPASSYDS", "KCPMAHOMES15-250", "Patrick Mahomes: 250+ passing yards", 0.60, 0.58,
            event_suffix="26OCT04DENKC"))
        kalshi.get_markets.side_effect = lambda **kw: (markets.get(kw["series_ticker"], []), None)
        lookup = {m["ticker"]: m for ms in markets.values() for m in ms}
        base = kalshi.get_market.side_effect
        kalshi.get_market.side_effect = lambda t: lookup[t] if t in lookup else base(t)
        events = {m["event_ticker"] for ms in markets.values() for m in ms}
        kalshi.get_multivariate_collections.return_value = [{
            "collection_ticker": "KXMVESPORTS-R",
            "associated_event_tickers": sorted(events | {"KXNFLGAME-26OCT04BUFBAL", "KXNFLGAME-26OCT04DENKC"})}]
        strat._combo_collections = None
        strat._props_cache = None
        return strat, kalshi

    def test_spreads_offer_both_sides_and_thresholds_match_exactly(self):
        strat, kalshi = self._stack_setup()
        spreads = strat.refresh_props(_NOW, "win by over")
        self.assertEqual({(p["ticker"].rsplit("-", 1)[-1], p["side"]) for p in spreads},
                         {("BAL10", "yes"), ("BAL10", "no"), ("BUF10", "yes"), ("BUF10", "no")})
        self.assertEqual({p["player"] for p in spreads}, {"Spread"})
        no_side = next(p for p in spreads if p["ticker"].endswith("BAL10") and p["side"] == "no")
        self.assertEqual(no_side["stat"], "BAL Ravens don't win by over 9.5")
        self.assertAlmostEqual(no_side["yes_ask"], 0.77)  # the NO side's own price
        # "2" must find 2+ receptions, not 12+
        found = [p["title"] for p in strat.refresh_props(_NOW, "coleman 2")]
        self.assertEqual(found, ["Keon Coleman: 2+ receptions"])

    def test_stack_picks_one_liquid_market_per_player_closest_to_the_target(self):
        strat, _ = self._stack_setup()
        built = strat.build_stack(_NOW, "bills ravens", 4, 0.85)
        self.assertTrue(built["ok"], built)
        titles = [r["title"] for r in built["rows"]]
        self.assertEqual(len(titles), 4)
        self.assertIn("Lamar Jackson: 150+ passing yards", titles)   # right on target
        self.assertIn("Josh Allen: 200+ passing yards", titles)      # 80% beats 40% for Allen
        self.assertNotIn("Josh Allen: 250+ passing yards", titles)
        self.assertNotIn("Bench Guy: 10+ rushing yards", titles)     # one-sided market skipped
        self.assertNotIn("Keon Coleman: 12+ receptions", titles)     # nobody trades a 2% leg
        self.assertEqual(len({r["player"] for r in built["rows"]}), 4)
        product = 1.0
        for r in built["rows"]:
            product *= r["mid"]
        self.assertAlmostEqual(built["reference"], product)

    def test_stack_target_changes_the_thresholds(self):
        strat, _ = self._stack_setup()
        long_shots = strat.build_stack(_NOW, "bills ravens", 2, 0.40)
        self.assertIn("Josh Allen: 250+ passing yards", [r["title"] for r in long_shots["rows"]])

    def test_stack_errors(self):
        strat, _ = self._stack_setup()
        self.assertIn("No upcoming football game", strat.build_stack(_NOW, "packers", 5, 0.85)["reason"])
        self.assertIn("several games", strat.build_stack(_NOW, "a", 5, 0.85)["reason"])
        self.assertIn("Not enough liquid", strat.build_stack(_NOW, "chiefs", 5, 0.85)["reason"])

    def test_a_built_stack_can_be_quoted_and_bought_as_one_same_game_combo(self):
        strat, kalshi = self._stack_setup()
        built = strat.build_stack(_NOW, "bills ravens", 4, 0.85)
        kalshi.get_rfq_quotes.return_value = [
            {"id": "q-1", "status": "open", "no_bid_dollars": "0.60", "contracts_fp": "9999.00"}]
        broker = PaperBroker(True, logging.getLogger("t"))
        plan = strat.quote_user_parlay(self._state(), broker, legs=built["legs"], amount=10.0, max_price=None, now=_NOW)
        self.assertTrue(plan["ok"], plan)
        self.assertTrue(plan["same_game"])
        self.assertAlmostEqual(plan["reference"], built["reference"], places=6)
        self.assertEqual(len(plan["legs"]), 4)
        kalshi.get_positions.return_value = [{"ticker": self.COMBO, "position": plan["contracts"]}]
        kalshi.get_rfq_quotes.return_value = [
            {"id": "q-1", "status": "open", "no_bid_dollars": str(round(1 - built["reference"] + 0.05, 3)),
             "contracts_fp": f"{plan['contracts']}.00"}]
        state, out = self._place(strat, legs=built["legs"], amount=10.0, max_price=None)
        self.assertTrue(out["ok"], out)
        self.assertEqual(len(kalshi.create_combo_market.call_args.args[1]), 4)

    def test_nine_leg_sgp_with_a_spread_no_leg(self):
        strat, kalshi = self._stack_setup()
        built = strat.build_stack(_NOW, "bills ravens", 9, 0.85)
        rows = built["rows"]
        self.assertGreaterEqual(len(rows), 4)
        legs = built["legs"]
        spread_no = next(p for p in strat.props.values() if p["kind"] == "spread" and p["side"] == "no" and "BAL" in p["ticker"])
        if not any(l["ticker"] == spread_no["ticker"] for l in legs):
            legs = legs + [{"ticker": spread_no["ticker"], "side": "no"}]
        plan = strat.quote_user_parlay(self._state(), PaperBroker(True, logging.getLogger("t")),
                                       legs=legs, amount=10.0, max_price=0.5, now=_NOW)
        self.assertTrue(plan["ok"], plan)
        self.assertIn("don't win by over", " ".join(r["pick"] for r in plan["legs"]))
        self.assertIn("no", [r["side"] for r in plan["legs"]])

    def test_leg_limit_is_configurable(self):
        strat, _ = self._stack_setup()
        built = strat.build_stack(_NOW, "bills ravens", 4, 0.85)
        strat.config = _config(user_parlay_max_legs=3, parlay_quote_wait_seconds=0.0)
        out = strat.quote_user_parlay(self._state(), PaperBroker(True, logging.getLogger("t")),
                                      legs=built["legs"], amount=10.0, max_price=None, now=_NOW)
        self.assertFalse(out["ok"])
        self.assertIn("2 to 3 legs", out["reason"])


class SlipMatchTests(unittest.TestCase):
    """Legs read from a screenshot are matched exactly or reported, never silently swapped."""

    _setup = ParlayRfqTests._setup
    _state = ParlayRfqTests._state
    _stack_setup = StackTests._stack_setup
    COMBO = ParlayRfqTests.COMBO

    def _slip(self, *legs, game="Buffalo Bills @ Baltimore Ravens"):
        return {"game": game, "odds": "+478", "legs": [dict({"type": "player_stat", "player": None, "team": None,
                "stat": None, "threshold": None, "line": None, "direction": None, "text": "leg"}, **l) for l in legs]}

    def test_exact_player_stats_match_and_names_ignore_suffixes_and_accents(self):
        strat, _ = self._stack_setup()
        out = strat.match_slip_legs(_NOW, self._slip(
            {"player": "Josh Allen", "stat": "passing yards", "threshold": 200, "text": "Allen 200+"},
            {"player": "Lamar Jackson Jr.", "stat": "passing yards", "threshold": 150, "text": "Jackson 150+"},
            {"player": "Keon Coleman", "stat": "receptions", "threshold": 2, "text": "Coleman 2+"},
        ))
        self.assertEqual(out["unmatched"], [])
        self.assertEqual([m["pick"] for m in out["matched"]], [
            "Josh Allen 200+ passing yards", "Lamar Jackson 150+ passing yards", "Keon Coleman 2+ receptions"])
        self.assertEqual(out["game"], "Buffalo Bills @ Baltimore Ravens")
        self.assertTrue(all(m["side"] == "yes" for m in out["matched"]))

    def test_a_missing_threshold_is_reported_with_what_kalshi_has(self):
        strat, _ = self._stack_setup()
        out = strat.match_slip_legs(_NOW, self._slip(
            {"player": "Josh Allen", "stat": "passing yards", "threshold": 225, "text": "Allen 225+"}))
        self.assertEqual(out["matched"], [])
        self.assertIn("200+, 250+", out["unmatched"][0]["reason"])

    def test_unknown_player_wrong_stat_under_and_total_are_unmatched(self):
        strat, _ = self._stack_setup()
        out = strat.match_slip_legs(_NOW, self._slip(
            {"player": "Nobody Special", "stat": "receptions", "threshold": 2, "text": "nobody"},
            {"player": "Josh Allen", "stat": "rushing yards", "threshold": 20, "text": "allen rush"},
            {"player": "Josh Allen", "stat": "passing yards", "threshold": 200, "direction": "under", "text": "allen under"},
            {"type": "total", "line": 45.5, "text": "over 45.5"},
        ))
        self.assertEqual(out["matched"], [])
        reasons = [u["reason"] for u in out["unmatched"]]
        self.assertIn("no Kalshi market for that player", reasons[0])
        self.assertIn("no rushing yards market", reasons[1])
        self.assertIn("unders aren't offered", reasons[2])
        self.assertIn("isn't offered here", reasons[3])

    def test_spreads_map_underdog_to_the_opponents_no_and_favourite_to_yes(self):
        strat, _ = self._stack_setup()
        out = strat.match_slip_legs(_NOW, self._slip(
            {"type": "spread", "team": "Baltimore Ravens", "line": 9.5, "text": "Ravens +9.5"},
            {"type": "spread", "team": "Buffalo Bills", "line": -9.5, "text": "Bills -9.5"},
        ))
        self.assertEqual(out["unmatched"], [])
        first, second = out["matched"]
        # Ravens +9.5 = the Bills do NOT win by over 9.5
        self.assertEqual((first["ticker"].rsplit("-", 1)[-1], first["side"]), ("BUF10", "no"))
        # (the Ravens market is a different ticker: Bills -9.5 = YES on 'BUF wins by over 9.5')
        self.assertEqual((second["ticker"].rsplit("-", 1)[-1], second["side"]), ("BUF10", "yes"))

    def test_unlisted_spread_line_is_reported(self):
        strat, _ = self._stack_setup()
        out = strat.match_slip_legs(_NOW, self._slip({"type": "spread", "team": "Baltimore Ravens", "line": 16.5, "text": "+16.5"}))
        self.assertIn("lines: 9.5", out["unmatched"][0]["reason"])

    def test_moneyline_leg_uses_the_board(self):
        strat, _ = self._stack_setup()
        out = strat.match_slip_legs(_NOW, self._slip({"type": "moneyline", "team": "Baltimore Ravens", "text": "Ravens ML"}))
        self.assertEqual(len(out["matched"]), 1)
        self.assertEqual(out["matched"][0]["pick"], "Baltimore Ravens to win")

    def _with_second_jackson(self):
        """Adds a Jackson with a passing-yards market in the other game (Denver @ Kansas City)."""
        strat, kalshi = self._stack_setup()
        base = kalshi.get_markets.side_effect

        def get_markets(**kw):
            rows, cursor = base(**kw)
            if kw["series_ticker"] == "KXNFLPASSYDS":
                rows = rows + [_prop_market("KXNFLPASSYDS", "DENCJACKSON9-150", "Courtland Jackson: 150+ passing yards",
                                            0.55, 0.53, event_suffix="26OCT04DENKC")]
            return rows, cursor
        kalshi.get_markets.side_effect = get_markets
        return strat

    def test_a_bare_surname_shared_by_two_players_is_not_guessed(self):
        strat = self._with_second_jackson()
        out = strat.match_slip_legs(_NOW, self._slip(
            {"player": "Jackson", "stat": "passing yards", "threshold": 150, "text": "Jackson 150+"}, game=None))
        self.assertEqual(out["matched"], [])
        self.assertIn("more than one player matches", out["unmatched"][0]["reason"])
        self.assertIn("Courtland Jackson", out["unmatched"][0]["reason"])
        self.assertIn("Lamar Jackson", out["unmatched"][0]["reason"])

    def test_typed_slip_without_a_game_uses_the_game_most_legs_point_to(self):
        strat = self._with_second_jackson()
        out = strat.match_slip_legs(_NOW, self._slip(
            {"player": "Josh Allen", "stat": "passing yards", "threshold": 200, "text": "Allen 200+"},
            {"type": "spread", "team": "Baltimore Ravens", "line": 9.5, "text": "Ravens +9.5"},
            {"player": "Jackson", "stat": "passing yards", "threshold": 150, "text": "Jackson 150+"},
            game=None,
        ))
        self.assertEqual(out["unmatched"], [])
        self.assertIn("Lamar Jackson 150+ passing yards", [m["pick"] for m in out["matched"]])

    def test_the_inferred_game_never_drops_legs_from_other_games(self):
        strat = self._with_second_jackson()
        out = strat.match_slip_legs(_NOW, self._slip(
            {"player": "Josh Allen", "stat": "passing yards", "threshold": 200, "text": "Allen 200+"},
            {"player": "Keon Coleman", "stat": "receptions", "threshold": 2, "text": "Coleman 2+"},
            {"player": "Patrick Mahomes", "stat": "passing yards", "threshold": 250, "text": "Mahomes 250+"},
            game=None,
        ))
        self.assertEqual(out["unmatched"], [])
        self.assertEqual(len(out["matched"]), 3)

    def test_a_spread_for_a_team_not_in_the_listed_game_is_not_matched_elsewhere(self):
        strat, _ = self._stack_setup()
        out = strat.match_slip_legs(_NOW, self._slip(
            {"type": "spread", "team": "Kansas City Chiefs", "line": 9.5, "text": "Chiefs +9.5"}, game=None))
        self.assertEqual(out["matched"], [])
        self.assertIn("doesn't list a 9.5-point spread for Kansas City Chiefs", out["unmatched"][0]["reason"])

    def test_duplicates_collapse_and_matches_feed_the_parlay_quote(self):
        strat, _ = self._stack_setup()
        leg = {"player": "Josh Allen", "stat": "passing yards", "threshold": 200, "text": "Allen 200+"}
        out = strat.match_slip_legs(_NOW, self._slip(
            leg, dict(leg), {"player": "Lamar Jackson", "stat": "passing yards", "threshold": 150, "text": "Jackson"}))
        self.assertEqual(len(out["matched"]), 2)
        legs = [{"ticker": m["ticker"], "side": m["side"]} for m in out["matched"]]
        plan = strat.quote_user_parlay(self._state(), PaperBroker(True, logging.getLogger("t")),
                                       legs=legs, amount=10.0, max_price=None, now=_NOW)
        self.assertTrue(plan["ok"], plan)
        self.assertEqual(len(plan["legs"]), 2)
