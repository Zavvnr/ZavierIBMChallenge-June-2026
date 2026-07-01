"""Offline tests for the commentary pre-cache: store/load, pacing, and /api/stream replay."""
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from data_pipeline import commentary_cache as cc


class StoreTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_round_trip(self):
        payloads = [{"minute": 0, "second": 2, "text": "uno"},
                    {"minute": 0, "second": 9, "text": "dos"}]
        cc.save_run("sample", "es", True, True, payloads, cache_dir=self.dir)
        self.assertEqual(cc.load_run("sample", "es", True, True, cache_dir=self.dir), payloads)

    def test_missing_returns_none(self):
        self.assertIsNone(cc.load_run("sample", "es", True, True, cache_dir=self.dir))

    def test_key_includes_voice_and_tts_mode(self):
        a = cc.cache_path("sample", "es", True, True, cache_dir=self.dir).name
        b = cc.cache_path("sample", "es", False, False, cache_dir=self.dir).name
        self.assertIn("2v-tts", a)
        self.assertIn("1v-notts", b)
        self.assertNotEqual(a, b)


class PaceTests(unittest.TestCase):
    def _payloads(self):
        return [{"minute": 0, "second": 0}, {"minute": 0, "second": 10},
                {"minute": 0, "second": 40}]

    def test_speed_zero_is_instant(self):
        slept = []
        out = list(cc.pace(self._payloads(), speed=0, sleeper=slept.append))
        self.assertEqual(len(out), 3)
        self.assertEqual(slept, [])                          # no waiting

    def test_positive_speed_waits_by_clock_delta(self):
        slept = []
        list(cc.pace(self._payloads(), speed=10, sleeper=slept.append))
        self.assertEqual([round(s, 3) for s in slept], [1.0, 3.0])  # 10s/10 and 30s/10


class StreamReplayTests(unittest.TestCase):
    def test_api_stream_replays_cache_without_the_model(self):
        from web.app import create_app
        canned = [{"minute": 0, "second": 2, "text": "Bienvenidos", "event_type": ""},
                  {"minute": 0, "second": 9, "text": "Buen pase", "event_type": "Pass"}]
        events = [{"type": {"name": "Pass"}, "team": {"name": "Argentina"}}]
        with mock.patch("data_pipeline.commentary_cache.load_run", return_value=canned), \
             mock.patch("web.app._load_events", return_value=events):
            client = create_app().test_client()
            resp = client.get(
                "/api/stream?match=sample&language=es&mock=0&two_speakers=1&tts=1&speed=0")
            body = resp.get_data(as_text=True)
        self.assertIn("Bienvenidos", body)
        self.assertIn("Buen pase", body)
        self.assertIn("event: done", body)


class PrecacheTests(unittest.TestCase):
    def test_limit_stops_early_and_saves_after_every_line(self):
        from web import app as webapp
        saves = []

        def fake_save(match, lang, two, tts, payloads, **kw):
            saves.append(len(payloads))                 # checkpoint size at each save
            return "cache/path.json"

        with mock.patch("web.app._load_events",
                        return_value=[{"type": {"name": "Pass"}, "team": {"name": "A"}}]), \
             mock.patch("web.app._sse_payload", side_effect=lambda it: {"line": it}), \
             mock.patch("data_pipeline.commentary_pipeline.stream_commentary",
                        return_value=iter([1, 2, 3, 4, 5])), \
             mock.patch("data_pipeline.commentary_cache.save_run", side_effect=fake_save):
            path, count = webapp.precache_commentary(limit=3)
        self.assertEqual(count, 3)                       # stopped at the limit
        self.assertEqual(saves, [1, 2, 3])               # saved incrementally after each line


if __name__ == "__main__":
    unittest.main(verbosity=2)
