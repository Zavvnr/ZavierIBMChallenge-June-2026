"""Unit tests for web.app (Flask UI). All offline — no Granite/TTS/network calls."""
import json
import os
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import agent.prompts as prompts_mod
from web.app import (create_app, _first_notable_event, _match_label,
                     _match_context, _granite_line)
from web import app as webapp


class _Resp:
    def __init__(self, content):
        self.choices = [types.SimpleNamespace(message=types.SimpleNamespace(content=content))]

class FakeClient:
    def __init__(self, content="GOL!"):
        comp = types.SimpleNamespace(create=lambda **kw: _Resp(content))
        self.chat = types.SimpleNamespace(completions=comp)


class NotableEventTests(unittest.TestCase):
    def test_prefers_goal(self):
        events = [{"type": {"name": "Shot"}, "shot": {"outcome": {"name": "Saved"}}},
                  {"type": {"name": "Shot"}, "shot": {"outcome": {"name": "Goal"}}, "id": "g"}]
        self.assertEqual(_first_notable_event(events).get("id"), "g")
    def test_first_shot_when_no_goal(self):
        events = [{"type": {"name": "Pass"}},
                  {"type": {"name": "Shot"}, "shot": {"outcome": {"name": "Saved"}}, "id": "s"}]
        self.assertEqual(_first_notable_event(events).get("id"), "s")
    def test_midpoint_when_no_shots(self):
        events = [{"type": {"name": "Pass"}}, {"type": {"name": "Pass"}, "id": "mid"}, {"type": {"name": "Pass"}}]
        self.assertEqual(_first_notable_event(events).get("id"), "mid")
    def test_empty_returns_none(self):
        self.assertIsNone(_first_notable_event([]))


class MatchMetaTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._patch = mock.patch.object(webapp, "CACHE_DIR", Path(self._tmp.name))
        self._patch.start()
    def tearDown(self):
        self._patch.stop(); self._tmp.cleanup()

    def _write_meta(self, match_id, meta):
        d = Path(self._tmp.name) / match_id
        d.mkdir(parents=True)
        (d / "meta.json").write_text(json.dumps(meta), encoding="utf-8")

    def test_match_label_from_meta(self):
        self._write_meta("3869685", {
            "home_team": {"home_team_name": "Argentina"}, "away_team": {"away_team_name": "France"},
            "home_score": 3, "away_score": 3, "match_date": "2022-12-18"})
        self.assertEqual(_match_label("3869685"), "Argentina 3-3 France (2022-12-18)")

    def test_match_label_fallback(self):
        self.assertEqual(_match_label("999"), "Match 999")

    def test_match_context_from_meta(self):
        self._write_meta("3869685", {
            "competition": {"competition_name": "FIFA World Cup"},
            "home_team": {"home_team_name": "Argentina"}, "away_team": {"away_team_name": "France"}})
        ctx = _match_context("3869685")
        self.assertEqual(ctx["competition"], "FIFA World Cup")
        self.assertEqual(ctx["home"], "Argentina")

    def test_match_context_sample_is_empty(self):
        self.assertEqual(_match_context("sample"), {})


class GraniteLineTests(unittest.TestCase):
    def test_granite_line_uses_client(self):
        fake = FakeClient("Gol de Messi!")
        with mock.patch("agent.granite_client.build_granite_client", return_value=fake), \
             mock.patch("agent.granite_client.model_id", return_value="granite-x"):
            line = _granite_line({"minute": 80, "type": {"name": "Shot"}, "player": {"name": "Messi"}}, "es")
        self.assertEqual(line, "Gol de Messi!")


class EndpointTests(unittest.TestCase):
    def setUp(self):
        self.client = create_app().test_client()

    def test_index_served(self):
        self.assertEqual(self.client.get("/").status_code, 200)

    def test_matches_lists_sample(self):
        with mock.patch.object(webapp, "CACHE_DIR", Path(tempfile.gettempdir()) / "nope_cache_xyz"):
            data = self.client.get("/api/matches").get_json()
        self.assertTrue(any(m["id"] == "sample" for m in data))

    def test_tts_without_key_returns_503(self):
        env = {k: v for k, v in os.environ.items() if k not in ("GOOGLE_TTS_API_KEY", "GOOGLE_API_KEY")}
        with mock.patch.dict(os.environ, env, clear=True):
            r = self.client.get("/api/tts?text=hello&language=en-US")
        self.assertEqual(r.status_code, 503)

    def test_languages_uses_prompts_names(self):
        original = getattr(prompts_mod, "LANGUAGE_NAMES", None)
        prompts_mod.LANGUAGE_NAMES = {"en": "English", "es": "Spanish"}
        try:
            r = self.client.get("/api/languages")
            self.assertEqual(r.status_code, 200)
            codes = {x["code"] for x in r.get_json()}
            self.assertEqual(codes, {"en", "es"})
        finally:
            if original is None:
                del prompts_mod.LANGUAGE_NAMES
            else:
                prompts_mod.LANGUAGE_NAMES = original


if __name__ == "__main__":
    unittest.main(verbosity=2)
