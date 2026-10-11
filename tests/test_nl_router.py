"""Plain-text routing: the model only picks a whitelisted action; arguments are cleaned."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from trading_bot.integrations import nl_router


def _client(tool_input):
    client = MagicMock()
    client.messages.create.return_value = SimpleNamespace(
        content=[SimpleNamespace(type="tool_use", input=tool_input)])
    return client


class ValidateTests(unittest.TestCase):
    def test_unknown_action_becomes_a_plain_reply(self):
        route = nl_router.validate_route({"action": "wire_money", "amount": 5})
        self.assertEqual(route["action"], "reply")

    def test_arguments_are_cleaned_and_bounded(self):
        route = nl_router.validate_route({
            "action": "parlay", "legs": ["  hurts   200 passing ", "", "browns"] + ["x"] * 30,
            "amount": "5", "max_cents": 250, "leg_count": 99, "leg_percent": 85, "team": "a" * 500})
        self.assertEqual(route["legs"][:2], ["hurts 200 passing", "browns"])
        self.assertEqual(len(route["legs"]), 12)
        self.assertEqual(route["amount"], 5.0)
        self.assertIsNone(route["max_cents"])      # 250 cents isn't a price
        self.assertIsNone(route["leg_count"])
        self.assertEqual(route["leg_percent"], 85.0)
        self.assertEqual(len(route["team"]), 200)

    def test_buying_is_not_an_action(self):
        for action in ("place", "buy", "order", "bet_now", "confirm_all"):
            self.assertNotIn(action, nl_router.ACTIONS)
        self.assertNotIn("kill", nl_router.ACTIONS)


class RouteMessageTests(unittest.TestCase):
    def test_forces_the_routing_tool_and_returns_the_validated_route(self):
        client = _client({"action": "bet", "team": "cleveland browns", "amount": 5})
        route = nl_router.route_message("put five bucks on the browns", pending=None, api_key="", model="m", client=client)
        self.assertEqual((route["action"], route["team"], route["amount"]), ("bet", "cleveland browns", 5.0))
        kwargs = client.messages.create.call_args.kwargs
        self.assertEqual(kwargs["tool_choice"], {"type": "tool", "name": "route"})
        self.assertEqual(kwargs["model"], "m")
        self.assertIn("No bet is waiting", kwargs["messages"][0]["content"])

    def test_pending_bet_is_given_as_context(self):
        client = _client({"action": "cancel"})
        nl_router.route_message("nah", pending="parlay of A + B, $2.00", api_key="", model="m", client=client)
        self.assertIn("parlay of A + B", client.messages.create.call_args.kwargs["messages"][0]["content"])

    def test_missing_key_and_bad_replies_are_friendly_errors(self):
        with self.assertRaises(RuntimeError) as ctx:
            nl_router.route_message("hi", pending=None, api_key="", model="m")
        self.assertIn("ANTHROPIC_API_KEY", str(ctx.exception))
        empty = MagicMock()
        empty.messages.create.return_value = SimpleNamespace(content=[SimpleNamespace(type="text", text="hi")])
        with self.assertRaises(RuntimeError):
            nl_router.route_message("hi", pending=None, api_key="", model="m", client=empty)
        boom = MagicMock()
        boom.messages.create.side_effect = ConnectionError("down")
        with self.assertRaises(RuntimeError):
            nl_router.route_message("hi", pending=None, api_key="", model="m", client=boom)


if __name__ == "__main__":
    unittest.main()
