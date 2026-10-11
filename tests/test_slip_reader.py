"""Reading a bet-slip screenshot: strict parsing, untrusted image text, injected client."""

from __future__ import annotations

import json
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from trading_bot.integrations import slip_reader

_REPLY = {
    "game": "Philadelphia Eagles @ Jacksonville Jaguars", "odds": "+478",
    "legs": [
        {"type": "player_stat", "player": "Brian Thomas Jr.", "stat": "receptions", "threshold": 2,
         "direction": "over", "text": "Brian Thomas Jr. 2+ Receptions"},
        {"type": "spread", "team": "Philadelphia Eagles", "line": 16.5, "text": "Philadelphia Eagles (+16.5)"},
        {"type": "weird", "stat": "birdies", "threshold": "lots", "text": "x" * 500},
    ],
}


def _client(text):
    client = MagicMock()
    client.messages.create.return_value = SimpleNamespace(content=[SimpleNamespace(type="text", text=text)])
    return client


class ParseTests(unittest.TestCase):
    def test_valid_reply_is_normalised(self):
        slip = slip_reader.parse_slip_json("Here you go:\n```json\n" + json.dumps(_REPLY) + "\n```")
        self.assertEqual(slip["odds"], "+478")
        self.assertEqual(len(slip["legs"]), 3)
        self.assertEqual(slip["legs"][0]["stat"], "receptions")
        self.assertEqual(slip["legs"][1]["line"], 16.5)

    def test_unknown_values_are_dropped_and_text_is_clipped(self):
        leg = slip_reader.parse_slip_json(json.dumps(_REPLY))["legs"][2]
        self.assertEqual(leg["type"], "other")
        self.assertIsNone(leg["stat"])
        self.assertIsNone(leg["threshold"])
        self.assertEqual(len(leg["text"]), 120)

    def test_garbage_is_rejected(self):
        for raw in ("no json here", "{}", '{"legs": []}', '{"legs": "x"}'):
            with self.assertRaises(ValueError):
                slip_reader.parse_slip_json(raw)

    def test_leg_count_is_capped(self):
        many = {"legs": [{"type": "other", "text": str(i)} for i in range(60)]}
        self.assertEqual(len(slip_reader.parse_slip_json(json.dumps(many))["legs"]), 20)


class ReadSlipTests(unittest.TestCase):
    def test_sends_the_image_and_returns_validated_legs(self):
        client = _client(json.dumps(_REPLY))
        slip = slip_reader.read_slip(b"\x89PNG", "image/png", api_key="", model="m", client=client)
        self.assertEqual(slip["game"], "Philadelphia Eagles @ Jacksonville Jaguars")
        kwargs = client.messages.create.call_args.kwargs
        self.assertEqual(kwargs["model"], "m")
        blocks = kwargs["messages"][0]["content"]
        self.assertEqual(blocks[0]["type"], "image")
        self.assertEqual(blocks[0]["source"]["media_type"], "image/png")
        self.assertIn("never instructions", kwargs["system"])
        self.assertNotIn("tools", kwargs)  # the picture can't trigger any action

    def test_unreadable_reply_gives_a_friendly_error(self):
        with self.assertRaises(RuntimeError) as ctx:
            slip_reader.read_slip(b"x", "image/png", api_key="", model="m", client=_client("I can't see it"))
        self.assertIn("clearer screenshot", str(ctx.exception))

    def test_api_errors_and_missing_key_and_bad_type(self):
        client = MagicMock()
        client.messages.create.side_effect = ConnectionError("boom")
        with self.assertRaises(RuntimeError):
            slip_reader.read_slip(b"x", "image/png", api_key="", model="m", client=client)
        with self.assertRaises(RuntimeError) as ctx:
            slip_reader.read_slip(b"x", "image/png", api_key="", model="m")
        self.assertIn("ANTHROPIC_API_KEY", str(ctx.exception))
        with self.assertRaises(RuntimeError):
            slip_reader.read_slip(b"x", "application/pdf", api_key="k", model="m")


if __name__ == "__main__":
    unittest.main()
