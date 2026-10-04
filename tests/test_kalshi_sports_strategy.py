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
from trading_bot.strategies.kalshi_sports import (
    KalshiSportsStrategy,
    estimate_home_win_prob,
    kelly_notional,
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

    def test_records_alone_rejected_when_market_signal_required(self):
        game = _game(home_moneyline=None, away_moneyline=None)
        self.assertIsNone(estimate_home_win_prob(game, _config()))
        self.assertIsNotNone(estimate_home_win_prob(game, _config(require_market_signal=False)))

    def test_blend_uses_sportsbook_and_predictor(self):
        game = _game(home_moneyline=-200.0, away_moneyline=170.0, predictor_home_prob=0.70)
        prob, comps = estimate_home_win_prob(game, _config())
        self.assertEqual(set(comps), {"sportsbook", "espn_predictor", "record"})
        self.assertTrue(0.55 < prob < 0.75)

    def test_kelly_capped(self):
        cfg = _config(kelly_fraction=1.0, max_position_pct=0.03)
        self.assertAlmostEqual(kelly_notional(0.3, 0.5, 10_000, cfg), 300.0)
        self.assertEqual(kelly_notional(-0.1, 0.5, 10_000, cfg), 0.0)


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

    def test_unrelated_market_not_matched(self):
        market = _market(title="Kansas City at Denver Winner?", yes_sub_title="Denver",
                         event_ticker="KXNFLGAME-26OCT04KCDEN", ticker="KXNFLGAME-26OCT04KCDEN-DEN")
        self.assertIsNone(match_market_to_game(market, [_game()]))


class StrategyTests(unittest.TestCase):
    def test_buys_underpriced_side_in_paper(self):
        # Book + ESPN say Baltimore (home) ~65%; Kalshi YES on Baltimore asks only 0.50.
        game = _game(home_moneyline=-190.0, away_moneyline=160.0, predictor_home_prob=0.66)
        markets = [_market("Baltimore", yes_ask=0.50, no_ask=0.52), _market("Buffalo", yes_ask=0.50, no_ask=0.52)]
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

    def test_implausible_edge_is_rejected(self):
        game = _game(home_moneyline=-400.0, away_moneyline=320.0)
        strat, _ = _strategy(_config(), [game], [_market("Baltimore", yes_ask=0.30, no_ask=0.72)])
        _, _, result = _run(strat)
        self.assertEqual(result["buys"], [])
        self.assertIn("edge_implausible", [d.get("reason") for d in result["decisions"]])

    def test_illiquid_market_skipped(self):
        game = _game(home_moneyline=-190.0, away_moneyline=160.0)
        strat, _ = _strategy(_config(), [game], [_market("Baltimore", yes_ask=0.55, volume=3)])
        _, _, result = _run(strat)
        self.assertEqual(result["buys"], [])

    def test_does_not_bet_same_game_twice(self):
        game = _game(home_moneyline=-190.0, away_moneyline=160.0)
        strat, _ = _strategy(_config(), [game], [_market("Baltimore", yes_ask=0.50, no_ask=0.52)])
        state = PortfolioState(cash=10_000.0)
        _, _, first = _run(strat, state)
        _, _, second = _run(strat, state)
        self.assertEqual(len(first["buys"]), 1)
        self.assertEqual(second["buys"], [])

    def test_resolved_win_closes_at_one_dollar(self):
        game = _game(home_moneyline=-190.0, away_moneyline=160.0)
        strat, kalshi = _strategy(_config(), [game], [_market("Baltimore", yes_ask=0.50, no_ask=0.52)])
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
        strat, kalshi = _strategy(_config(), [game], [_market("Baltimore", yes_ask=0.50, no_ask=0.52)])
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
        strat, kalshi = _strategy(_config(), [game], [_market("Baltimore", yes_ask=0.50, no_ask=0.52)])
        state, _, result = _run(strat)
        kalshi.get_market.return_value = {"status": "active", "result": ""}
        strat.espn.get_scoreboard.return_value = []
        _, _, result2 = _run(strat, state)
        self.assertEqual(result2["resolved_closed"], [])
        self.assertIn(result["buys"][0], state.positions)

    def test_kill_switch_blocks_entries(self):
        game = _game(home_moneyline=-190.0, away_moneyline=160.0)
        strat, _ = _strategy(_config(), [game], [_market("Baltimore", yes_ask=0.50, no_ask=0.52)])
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
            _market(f"City{i}", yes_ask=0.50, no_ask=0.52,
                    title=f"Town{i} at City{i} Winner?",
                    ticker=f"KXNFLGAME-26OCT04A{i}XH{i}X-H{i}X",
                    event_ticker=f"KXNFLGAME-26OCT04A{i}XH{i}X")
            for i in range(5)
        ]
        strat, _ = _strategy(_config(capital_allocation_pct=0.04, max_position_pct=0.03), games, markets)
        state, broker, result = _run(strat)
        exposure = sum(p.quantity * p.entry_price for p in state.positions.values())
        self.assertLessEqual(exposure, 10_000 * 0.04 + 1.0)
        self.assertLess(len(result["buys"]), 5)


if __name__ == "__main__":
    unittest.main()
