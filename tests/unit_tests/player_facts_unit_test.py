"""Unit tests for agent.player_facts. Offline: curated + cache, fake Wikipedia fetcher."""
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from agent import player_facts as pf

WIKI = {
    "title": "Some Player",
    "extract": "Some Player is a defender. He plays for a club.",
    "description": "footballer",
    "content_urls": {"desktop": {"page": "http://wiki/x"}},
}


def fake_fetcher(url):
    return WIKI


class CuratedTests(unittest.TestCase):
    def test_full_name(self):
        facts = pf.facts_for("Lionel Messi")
        self.assertIsNotNone(facts)
        self.assertIn("club", facts)
        self.assertIn("tendency", facts)

    def test_last_name_match(self):
        self.assertIsNotNone(pf.facts_for("Messi"))
        self.assertEqual(pf.facts_for("Messi")["role"], pf.CURATED["lionel messi"]["role"])

    def test_unknown_returns_none(self):
        self.assertIsNone(pf.facts_for("Nonexistent Player"))

    def test_note_text(self):
        self.assertIn("club:", pf.note_text(pf.facts_for("Lionel Messi")))
        self.assertEqual(pf.note_text(None), "")


class PrewarmTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._patch = mock.patch.object(pf, "CACHE_DIR", Path(self._tmp.name))
        self._patch.start()

    def tearDown(self):
        self._patch.stop()
        self._tmp.cleanup()

    def test_prewarm_writes_cache_then_facts_for_reads_it(self):
        written = pf.prewarm(["Some Player"], "en", fetcher=fake_fetcher)
        self.assertEqual(written, 1)
        facts = pf.facts_for("Some Player", "en")
        self.assertIsNotNone(facts)
        self.assertIn("defender", facts["note"])          # condensed from the Wikipedia extract
        self.assertEqual(facts["source"], "http://wiki/x")

    def test_prewarm_skips_curated_players(self):
        # Messi is curated -> prewarm must not fetch/overwrite him.
        self.assertEqual(pf.prewarm(["Lionel Messi"], "en", fetcher=fake_fetcher), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
