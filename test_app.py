"""Smoke test: the hosted UI must run the invented ZIP end to end."""

from __future__ import annotations

import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest


class DemoRunTest(unittest.TestCase):
    def test_demo_produces_source_timed_cross_episode_promo(self) -> None:
        app = AppTest.from_file(str(Path(__file__).with_name("app.py")), default_timeout=30).run()
        self.assertFalse(app.exception)
        app.button[0].click().run()
        self.assertFalse(app.exception)
        result = app.session_state["promo_result"]
        self.assertEqual(result["report"]["episodes"], 2)
        self.assertEqual(result["validation"]["failures"], [])
        self.assertGreaterEqual(len(result["ranked"]), 1)
        first = result["ranked"][0]
        self.assertEqual([segment["episode"] for segment in first["segments"]], [1, 2])
        self.assertTrue(first["hook"] and first["cliffhanger"] and first["why_it_works"] and first["post_first_reason"])
        self.assertEqual(len(app.get("download_button")), 3)


if __name__ == "__main__":
    unittest.main()
