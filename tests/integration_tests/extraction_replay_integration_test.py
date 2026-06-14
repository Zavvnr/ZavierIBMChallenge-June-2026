"""Integration: data_extraction caches a match, then data_replayer streams it back in order."""
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from data_extraction import loader
from data_replayer.replayer import replay

# Deliberately out of index order to prove the replayer re-orders on read.
EVENTS = [
    {"index": 3, "period": 1, "timestamp": "00:00:30.000", "type": {"name": "Shot"},
     "shot": {"outcome": {"name": "Goal"}}, "team": {"name": "Argentina"}},
    {"index": 1, "period": 1, "timestamp": "00:00:05.000", "type": {"name": "Pass"},
     "team": {"name": "Argentina"}},
    {"index": 2, "period": 1, "timestamp": "00:00:18.000", "type": {"name": "Pass"},
     "team": {"name": "Argentina"}},
]


class ExtractionToReplay(unittest.TestCase):
    def test_cache_then_load_then_replay_in_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(loader, "CACHE_DIR", Path(tmp)), \
                 mock.patch.object(loader, "download_events", return_value=EVENTS), \
                 mock.patch.object(loader, "download_lineups", return_value=[]), \
                 mock.patch.object(loader, "find_match_meta", return_value=None):
                dest = loader.cache_match(3869685)
                self.assertTrue((dest / "events.json").exists())
                events = loader.load_cached_events(3869685)

            streamed = list(replay(events, speed=0))
            self.assertEqual([e["index"] for e in streamed], [1, 2, 3])
            self.assertEqual(len(streamed), len(EVENTS))

    def test_attribution_written_next_to_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(loader, "CACHE_DIR", Path(tmp)), \
                 mock.patch.object(loader, "download_events", return_value=EVENTS), \
                 mock.patch.object(loader, "download_lineups", return_value=[]), \
                 mock.patch.object(loader, "find_match_meta", return_value=None):
                dest = loader.cache_match(3869685)
            self.assertIn("StatsBomb", (dest / "ATTRIBUTION.txt").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
