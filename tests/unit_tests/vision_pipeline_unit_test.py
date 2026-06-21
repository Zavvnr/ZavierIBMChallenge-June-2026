"""Tier A flow: vision events save/load round-trip + replay through the agent. Offline."""
import tempfile
import unittest
from pathlib import Path

from vision_model.pipeline import demo_events, load_events, save_events, _teams


class SaveLoadTests(unittest.TestCase):
    def test_round_trip(self):
        events = demo_events()
        with tempfile.TemporaryDirectory() as d:
            path = str(Path(d) / "ev.json")
            self.assertEqual(save_events(events, path), path)
            self.assertTrue(Path(path).exists())
            self.assertEqual(load_events(path), events)   # JSON round-trip preserves the events

    def test_teams_from_events(self):
        evs = [{"team": {"name": "Blue"}}, {"team": {"name": "Red"}}, {"team": {"name": "Blue"}}]
        self.assertEqual(_teams(evs), ("Blue", "Red"))
        self.assertEqual(_teams([]), ("", ""))


class ReplayTests(unittest.TestCase):
    def test_saved_events_drive_the_agent(self):
        from agent.commentary_agent import CommentaryAgent
        from data_replayer.replayer import replay
        with tempfile.TemporaryDirectory() as d:
            path = str(Path(d) / "ev.json")
            save_events(demo_events(), path)
            events = load_events(path)
        agent = CommentaryAgent(mock=True)
        lines = [agent.handle(ev) for ev in replay(events, speed=0.0)]
        self.assertTrue(any(lines), "saved vision events produced no commentary")


if __name__ == "__main__":
    unittest.main(verbosity=2)
