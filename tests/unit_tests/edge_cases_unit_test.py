"""Additional edge-case unit tests: error paths and boundaries across all modules. Offline."""
import base64
import json
import os
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from data_extraction import loader
from data_replayer import replayer
from agent.mcp_client import MongoMCPContextClient, build_context_query, _format_context
from data_pipeline import commentary_pipeline as cp
from data_pipeline import live_cv_pipeline as cv
from text_to_speech import speak
from text_to_speech import mutilingual_speaker as ms
from spike import go_no_go as gng


# --------------------------------------------------------------------------- #
# data_extraction.loader
# --------------------------------------------------------------------------- #
class _Resp:
    def __init__(self, payload=None, status=200):
        self._payload, self.status_code = payload, status
    def raise_for_status(self):
        if self.status_code >= 400:
            raise loader.requests.HTTPError(f"status {self.status_code}")
    def json(self):
        return self._payload


class LoaderEdgeTests(unittest.TestCase):
    def test_get_json_raises_on_http_error(self):
        with mock.patch.object(loader.requests, "get", return_value=_Resp(status=500)):
            with self.assertRaises(loader.requests.HTTPError):
                loader._get_json("http://x")

    def test_cache_match_without_meta_writes_no_meta_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(loader, "CACHE_DIR", Path(tmp)), \
                 mock.patch.object(loader, "download_events", return_value=[{"e": 1}]), \
                 mock.patch.object(loader, "download_lineups", return_value=[]), \
                 mock.patch.object(loader, "find_match_meta", return_value=None):
                dest = loader.cache_match(42)
            self.assertFalse((dest / "meta.json").exists())
            self.assertTrue((dest / "events.json").exists())

    def test_find_match_meta_falls_back_to_world_cup(self):
        def fake_list(comp, season):
            return [{"match_id": 3869685, "home_score": 3}] if (comp, season) == (43, 106) else []
        with mock.patch.object(loader, "list_matches", side_effect=fake_list):
            meta = loader.find_match_meta(3869685)  # no comp/season -> WC2022 fallback
        self.assertEqual(meta["home_score"], 3)


# --------------------------------------------------------------------------- #
# data_replayer.replayer
# --------------------------------------------------------------------------- #
class ReplayerEdgeTests(unittest.TestCase):
    def test_empty_stream(self):
        self.assertEqual(list(replayer.replay([], speed=0)), [])

    def test_no_negative_wait_on_out_of_order_timestamps(self):
        waits = []
        events = [{"index": 1, "timestamp": "00:00:10.000"},
                  {"index": 2, "timestamp": "00:00:05.000"}]  # earlier ts after
        list(replayer.replay(events, speed=10.0, sleep=waits.append))
        self.assertEqual(waits, [])  # max(0, -5)/speed = 0 -> no sleep

    def test_missing_timestamp_defaults(self):
        out = list(replayer.replay([{"index": 1}, {"index": 2}], speed=0))
        self.assertEqual(len(out), 2)


# --------------------------------------------------------------------------- #
# agent.mcp_client
# --------------------------------------------------------------------------- #
class FakeCursor(list):
    def limit(self, n):
        return FakeCursor(self[:n])

class FakeCollection:
    def __init__(self, vector_docs=None, text_docs=None):
        self._v, self._t = vector_docs or [], text_docs or []
        self.aggregate_called = False
    def aggregate(self, pipeline):
        self.aggregate_called = True
        return list(self._v)
    def find(self, *a, **k):
        return FakeCursor(self._t)


class McpEdgeTests(unittest.TestCase):
    def test_query_includes_pass_recipient_and_technique(self):
        ev = {"type": {"name": "Pass"}, "team": {"name": "Argentina"}, "player": {"name": "De Paul"},
              "pass": {"recipient": {"name": "Messi"}, "technique": {"name": "Through Ball"}}}
        q = build_context_query(ev, {})
        self.assertIn("Messi", q)
        self.assertIn("Through Ball", q)

    def test_query_includes_substitution_replacement(self):
        ev = {"type": {"name": "Substitution"}, "team": {"name": "France"},
              "substitution": {"replacement": {"name": "Kolo Muani"}}}
        self.assertIn("Kolo Muani", build_context_query(ev, {}))

    def test_format_context_skips_empty_docs(self):
        self.assertEqual(_format_context([{"kind": "player", "name": "", "text": ""}]), {})

    def test_vector_empty_falls_back_to_text_search(self):
        col = FakeCollection(vector_docs=[], text_docs=[{"kind": "player", "name": "Messi", "text": "Captain."}])
        c = MongoMCPContextClient(collection_handle=col, embedder=lambda q: [0.1] * 768)
        ev = {"type": {"name": "Shot"}, "team": {"name": "Argentina"}, "player": {"name": "Messi"}}
        out = c.fetch_event_context(ev, {"score": "1-0"})
        self.assertTrue(col.aggregate_called)          # vector search was attempted
        self.assertEqual(out, {"players": ["Messi: Captain."]})  # then text fallback won


# --------------------------------------------------------------------------- #
# data_pipeline
# --------------------------------------------------------------------------- #
class PipelineEdgeTests(unittest.TestCase):
    def test_tempo_for_handles_bad_event(self):
        item = types.SimpleNamespace(kind="call", event="not-a-dict")
        self.assertEqual(cp.tempo_for(item), 1.0)  # importance() raises -> intensity 0

    def test_as_dict_includes_turn_audio(self):
        da = ms.DialogueAudio([ms.TurnAudio("lead", "Goal!", audio_bytes=b"A")])
        sr = speak.SpeechResult(text="Goal!", language="en", provider="multi")
        out = cp.CommentaryOutput(event={}, text="Goal!", speech=sr, dialogue_audio=da)
        d = out.as_dict()
        self.assertIn("turn_audio", d)
        self.assertTrue(d["audio_ready"])
        self.assertEqual(d["turn_audio"][0]["speaker"], "lead")

    def test_live_cv_keeps_zero_confidence(self):
        det = cv.Detection(label="pass", confidence=0.0)  # 0.0 is falsy -> not filtered
        self.assertEqual(len(cv.LiveEventAdapter().to_events([det], seconds=0)), 1)


# --------------------------------------------------------------------------- #
# text_to_speech
# --------------------------------------------------------------------------- #
class _FlakyTransport:
    def __init__(self):
        self.n = 0
    def __call__(self, url, payload, key):
        self.n += 1
        if self.n == 1:
            raise RuntimeError("bad voice")
        return {"audioContent": base64.b64encode(b"A").decode("ascii")}


class GoogleTtsEdgeTests(unittest.TestCase):
    def test_voice_fallback_after_first_failure(self):
        sp = speak.GoogleCloudSpeaker(transport=_FlakyTransport(), api_key="x")
        with tempfile.TemporaryDirectory() as tmp:
            r = sp.synthesize("hi", language="en-US", output_dir=Path(tmp))
        self.assertTrue(r.has_audio())  # second voice option succeeded

    def test_no_audio_content_degrades(self):
        sp = speak.GoogleCloudSpeaker(transport=lambda u, p, k: {}, api_key="x")
        r = sp.synthesize("hi", language="en-US")
        self.assertFalse(r.has_audio())
        self.assertIn("TTS failed", r.skipped_reason)

    def test_speaking_rate_omitted_when_none(self):
        captured = []
        def t(url, payload, key):
            captured.append(payload)
            return {"audioContent": base64.b64encode(b"A").decode("ascii")}
        with tempfile.TemporaryDirectory() as tmp:
            speak.GoogleCloudSpeaker(transport=t, api_key="x").synthesize(
                "hi", language="en-US", output_dir=Path(tmp))
        self.assertNotIn("speakingRate", captured[0]["audioConfig"])

    def test_multispeaker_empty_turns(self):
        da = ms.MultiSpeakerSpeaker(mock=True).synthesize_dialogue([])
        self.assertFalse(da.has_audio())
        self.assertEqual(da.segments, [])


# --------------------------------------------------------------------------- #
# spike.go_no_go
# --------------------------------------------------------------------------- #
class SpikeEdgeTests(unittest.TestCase):
    def test_describe_dribble_and_goalkeeper(self):
        dribble = {"type": {"name": "Dribble"}, "team": {"name": "Argentina"},
                   "dribble": {"outcome": {"name": "Complete"}}}
        self.assertIn("Complete", gng.describe_event(dribble))
        gk = {"type": {"name": "Goal Keeper"}, "team": {"name": "France"},
              "goalkeeper": {"type": {"name": "Save"}}}
        self.assertIn("Save", gng.describe_event(gk))

    def test_build_prompt_unknown_language_uses_code(self):
        self.assertIn("xx", gng.build_prompt([{"type": {"name": "Shot"}}], "xx"))

    def test_select_window_out_of_range(self):
        self.assertEqual(gng.select_window([{"type": {"name": "Pass"}}], 100, 5, dense=False), [])


# --------------------------------------------------------------------------- #
# web.app
# --------------------------------------------------------------------------- #
class WebEdgeTests(unittest.TestCase):
    def setUp(self):
        from web.app import create_app
        from web import app as webapp
        self.webapp = webapp
        self.client = create_app().test_client()

    def test_matches_lists_cached_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp) / "3869685"
            d.mkdir()
            (d / "events.json").write_text("[]", encoding="utf-8")
            (d / "meta.json").write_text(json.dumps({
                "home_team": {"home_team_name": "Argentina"},
                "away_team": {"away_team_name": "France"},
                "home_score": 3, "away_score": 3, "match_date": "2022-12-18"}), encoding="utf-8")
            with mock.patch.object(self.webapp, "CACHE_DIR", Path(tmp)):
                data = self.client.get("/api/matches").get_json()
        ids = {m["id"] for m in data}
        self.assertIn("sample", ids)
        self.assertIn("3869685", ids)

    def test_match_label_handles_corrupt_meta(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp) / "777"
            d.mkdir()
            (d / "meta.json").write_text("{not valid json", encoding="utf-8")
            with mock.patch.object(self.webapp, "CACHE_DIR", Path(tmp)):
                self.assertEqual(self.webapp._match_label("777"), "Match 777")


if __name__ == "__main__":
    unittest.main(verbosity=2)
