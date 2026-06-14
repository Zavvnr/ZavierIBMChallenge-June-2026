"""Unit tests for data_extraction.loader (StatsBomb caching + API fetch). All offline."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from data_extraction import loader


class FakeResp:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise loader.requests.HTTPError(f"status {self.status_code}")

    def json(self):
        return self._payload


class ConstantsTests(unittest.TestCase):
    def test_demo_match_constants(self):
        self.assertEqual(loader.DEFAULT_DEMO_MATCH_ID, 3869685)
        self.assertEqual(loader.WORLD_CUP_2022, {"competition_id": 43, "season_id": 106})
        self.assertIn(3869685, loader.DEMO_MATCHES)

    def test_attribution_credits_statsbomb(self):
        self.assertIn("StatsBomb", loader.STATSBOMB_ATTRIBUTION)


class FetchTests(unittest.TestCase):
    def test_get_json_returns_decoded_body(self):
        with mock.patch.object(loader.requests, "get", return_value=FakeResp({"ok": 1})) as g:
            self.assertEqual(loader._get_json("http://x"), {"ok": 1})
            g.assert_called_once()

    def test_download_events_hits_events_url(self):
        with mock.patch.object(loader.requests, "get", return_value=FakeResp([{"id": 1}])) as g:
            out = loader.download_events(3869685)
        self.assertEqual(out, [{"id": 1}])
        self.assertIn("/events/3869685.json", g.call_args.args[0])

    def test_list_matches_hits_matches_url(self):
        with mock.patch.object(loader.requests, "get", return_value=FakeResp([{"match_id": 7}])) as g:
            loader.list_matches(43, 106)
        self.assertIn("/matches/43/106.json", g.call_args.args[0])

    def test_find_match_meta_matches_id(self):
        rows = [{"match_id": 1}, {"match_id": 3869685, "home_score": 3}]
        with mock.patch.object(loader, "list_matches", return_value=rows):
            meta = loader.find_match_meta(3869685, 43, 106)
        self.assertEqual(meta["home_score"], 3)

    def test_find_match_meta_missing_returns_none(self):
        with mock.patch.object(loader, "list_matches", return_value=[{"match_id": 1}]):
            self.assertIsNone(loader.find_match_meta(999, 43, 106))


class CacheTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._cache = Path(self._tmp.name)
        self._patch = mock.patch.object(loader, "CACHE_DIR", self._cache)
        self._patch.start()

    def tearDown(self):
        self._patch.stop()
        self._tmp.cleanup()

    def test_match_dir_under_cache(self):
        self.assertEqual(loader.match_dir(42), self._cache / "42")

    def test_cache_match_writes_files(self):
        with mock.patch.object(loader, "download_events", return_value=[{"e": 1}]), \
             mock.patch.object(loader, "download_lineups", return_value=[{"l": 1}]), \
             mock.patch.object(loader, "find_match_meta", return_value={"home_score": 3}):
            dest = loader.cache_match(3869685)
        self.assertTrue((dest / "events.json").exists())
        self.assertTrue((dest / "lineups.json").exists())
        self.assertTrue((dest / "meta.json").exists())
        self.assertTrue((dest / "ATTRIBUTION.txt").exists())
        self.assertEqual(json.loads((dest / "events.json").read_text(encoding="utf-8")), [{"e": 1}])

    def test_load_cached_events_roundtrip(self):
        d = loader.match_dir(3869685)
        d.mkdir(parents=True)
        (d / "events.json").write_text(json.dumps([{"e": 1}]), encoding="utf-8")
        self.assertEqual(loader.load_cached_events(3869685), [{"e": 1}])

    def test_load_cached_events_missing_raises(self):
        with self.assertRaises(FileNotFoundError):
            loader.load_cached_events(123456)

    def test_load_cached_meta_missing_returns_none(self):
        self.assertIsNone(loader.load_cached_meta(123456))


class FetchEventsTests(unittest.TestCase):
    def test_uses_cache_when_present(self):
        with mock.patch.object(loader, "load_cached_events", return_value=[{"c": 1}]), \
             mock.patch.object(loader, "download_events") as dl:
            out = loader.fetch_events(3869685)
        self.assertEqual(out, [{"c": 1}])
        dl.assert_not_called()                       # no network when cached

    def test_downloads_and_caches_on_miss(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(loader, "CACHE_DIR", Path(tmp)), \
                 mock.patch.object(loader, "download_events", return_value=[{"d": 1}]) as dl:
                out = loader.fetch_events(3869685)
                self.assertEqual(out, [{"d": 1}])
                dl.assert_called_once()
                self.assertEqual(loader.load_cached_events(3869685), [{"d": 1}])  # cached for next time

    def test_none_uses_default_demo_match(self):
        with mock.patch.object(loader, "load_cached_events", side_effect=FileNotFoundError), \
             mock.patch.object(loader, "download_events", return_value=[]) as dl, \
             mock.patch.object(loader, "_write_json"):
            loader.fetch_events(None)
        dl.assert_called_once_with(loader.DEFAULT_DEMO_MATCH_ID)


if __name__ == "__main__":
    unittest.main(verbosity=2)
