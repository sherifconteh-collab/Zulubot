from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from zoneinfo import ZoneInfo

from trading_bot.models import PortfolioState
from trading_bot.state import StateStore


class StateStoreAtomicSaveTests(unittest.TestCase):
    def test_save_replaces_state_atomically_and_cleans_tmp_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            state_path = Path(temp_dir) / "state.json"
            store = StateStore(state_path, 1000.0, ZoneInfo("UTC"))
            state = PortfolioState(
                cash=123.45,
                cash_reserves={"kalshi": 50.0},
                ai_monthly_usage={"month": "2026-03", "cost": 3.21},
            )

            store.save(state)

            self.assertTrue(state_path.exists())
            self.assertFalse((Path(temp_dir) / "state.json.tmp").exists())

            loaded = store.load()
            self.assertAlmostEqual(loaded.cash, 123.45)
            self.assertEqual(loaded.cash_reserves["kalshi"], 50.0)
            self.assertEqual(loaded.ai_monthly_usage["month"], "2026-03")
