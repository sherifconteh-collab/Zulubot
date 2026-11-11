from __future__ import annotations

import logging
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, PropertyMock, patch

from trading_bot.clients_kalshi import KalshiClient
from trading_bot.config import KalshiConfig, _parse_shard_targets
from trading_bot.service import TradingBotService
from trading_bot.telegram_bot import _shard_allocation_line


def _client(pages):
    client = KalshiClient(logger=logging.getLogger("t"))
    patcher = patch.object(KalshiClient, "can_trade", new_callable=PropertyMock, return_value=True)
    patcher.start()
    client._get = MagicMock(side_effect=pages) if isinstance(pages, list) else MagicMock(return_value=pages)
    client._post = MagicMock(return_value={})
    return client, patcher


class ClientTests(unittest.TestCase):
    def test_reads_the_target_allocation(self):
        client, patcher = _client({"allocations": [{"exchange_index": 3, "percent": 20}, {"exchange_index": 0, "percent": 80}],
                                   "resting_margin_reservation": "sum"})
        self.addCleanup(patcher.stop)
        self.assertEqual(client.get_target_allocation(), [(0, 80), (3, 20)])
        self.assertEqual(client._get.call_args.args[0], "/portfolio/target_balance_allocation")

    def test_an_empty_allocation_means_off_and_a_reply_without_the_key_is_an_error(self):
        client, patcher = _client({"allocations": [], "resting_margin_reservation": "sum"})
        self.addCleanup(patcher.stop)
        self.assertEqual(client.get_target_allocation(), [])
        client._get.return_value = {"error": "x"}
        with self.assertRaises(RuntimeError):
            client.get_target_allocation()

    def test_sets_the_allocation_in_kalshis_shape_and_refuses_a_bad_total(self):
        client, patcher = _client({})
        self.addCleanup(patcher.stop)
        client.set_target_allocation({3: 15, 0: 60, 1: 15, 2: 10})
        path, body = client._post.call_args.args
        self.assertEqual(path, "/portfolio/target_balance_allocation")
        self.assertEqual(body, {"allocations": [{"exchange_index": 0, "percent": 60}, {"exchange_index": 1, "percent": 15},
                                                {"exchange_index": 2, "percent": 10}, {"exchange_index": 3, "percent": 15}]})
        with self.assertRaises(ValueError):
            client.set_target_allocation({0: 50, 1: 40})
        self.assertEqual(client._post.call_count, 1)


class ConfigTests(unittest.TestCase):
    def test_parses_shard_targets_and_falls_back_on_anything_odd(self):
        self.assertEqual(_parse_shard_targets("0:70, 1:30"), ((0, 70), (1, 30)))
        default = KalshiConfig().shard_targets
        for bad in ("", "0:50,1:40", "a:b", "0:100,0:0", "0:120,1:-20", "0-100"):
            self.assertEqual(_parse_shard_targets(bad), default, bad)
        self.assertEqual(sum(p for _, p in default), 100)
        self.assertTrue(KalshiConfig().auto_rebalance)


def _service(paper=False, auto=True, can_trade=True, **client):
    svc = TradingBotService.__new__(TradingBotService)
    svc.config = SimpleNamespace(paper_mode=paper, timezone=__import__("zoneinfo").ZoneInfo("UTC"),
                                 kalshi=KalshiConfig(auto_rebalance=auto))
    svc.logger = logging.getLogger("t")
    svc.kalshi_client = SimpleNamespace(can_trade=can_trade, **client)
    return svc


class EnsureAllocationTests(unittest.TestCase):
    def test_sets_the_default_split_when_none_exists_and_confirms_it(self):
        get = MagicMock(side_effect=[[], [(0, 60), (1, 15), (2, 10), (3, 15)]])
        put = MagicMock()
        status = _service(get_target_allocation=get, set_target_allocation=put).ensure_shard_allocation()
        self.assertEqual(status["state"], "set")
        put.assert_called_once_with({0: 60, 1: 15, 2: 10, 3: 15})

    def test_never_overwrites_a_split_the_owner_already_set(self):
        put = MagicMock()
        status = _service(get_target_allocation=MagicMock(return_value=[(0, 100)]), set_target_allocation=put).ensure_shard_allocation()
        self.assertEqual((status["state"], status["targets"]), ("yours", [(0, 100)]))
        put.assert_not_called()

    def test_does_nothing_in_paper_mode_without_keys_or_when_switched_off(self):
        put = MagicMock()
        for svc in (_service(paper=True, set_target_allocation=put, get_target_allocation=put),
                    _service(can_trade=False, set_target_allocation=put, get_target_allocation=put),
                    _service(auto=False, set_target_allocation=put, get_target_allocation=put)):
            svc.ensure_shard_allocation()
        put.assert_not_called()

    def test_disabled_management_does_not_claim_remote_rebalancing_is_off(self):
        api = MagicMock()
        status = _service(auto=False, get_target_allocation=api, set_target_allocation=api).ensure_shard_allocation()
        self.assertEqual(status["state"], "unmanaged")
        line = _shard_allocation_line(status)
        self.assertIn("bot management disabled", line)
        self.assertIn("Kalshi may still rebalance", line)
        self.assertIn("kalshi.com/account/exchange-indexes", line)
        api.assert_not_called()

    def test_a_refusal_or_an_unconfirmed_setting_is_reported_not_hidden(self):
        boom = MagicMock(side_effect=RuntimeError("403 forbidden"))
        status = _service(get_target_allocation=MagicMock(return_value=[]), set_target_allocation=boom).ensure_shard_allocation()
        self.assertEqual(status["state"], "failed")
        self.assertIn("403", status["error"])
        status = _service(get_target_allocation=MagicMock(return_value=[]), set_target_allocation=MagicMock()).ensure_shard_allocation()
        self.assertEqual(status["state"], "failed")


class LineTests(unittest.TestCase):
    def test_each_state_reads_plainly(self):
        self.assertIn("set by the bot (#0 60%, #1 40%)", _shard_allocation_line({"state": "set", "targets": [(0, 60), (1, 40)]}))
        self.assertIn("your own split", _shard_allocation_line({"state": "yours", "targets": [(0, 100)]}))
        failed = _shard_allocation_line({"state": "failed", "error": "403", "targets": []})
        self.assertIn("NOT on", failed)
        self.assertIn("kalshi.com/account/exchange-indexes", failed)
        self.assertEqual(_shard_allocation_line({}), "")
        self.assertEqual(_shard_allocation_line({"state": "paper"}), "")


if __name__ == "__main__":
    unittest.main()
