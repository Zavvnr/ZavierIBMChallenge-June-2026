"""Unit tests for web.app (Flask UI). All offline — no Granite/TTS/network calls."""
import json
import os
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import agent.prompts as prompts_mod
from data_extraction.lineups import PlayerSlot, TeamLineup
from web.app import (
    create_app, _first_notable_event, _match_label, _match_context, _teams_from_events,
)
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


class TeamsFromEventsTests(unittest.TestCase):
    def test_from_starting_xi(self):
        events = [
            {"type": {"name": "Starting XI"}, "team": {"name": "Argentina"}},
            {"type": {"name": "Starting XI"}, "team": {"name": "France"}},
            {"type": {"name": "Pass"}, "team": {"name": "Argentina"}},
        ]
        self.assertEqual(_teams_from_events(events), ("Argentina", "France"))

    def test_fallback_to_first_two_teams(self):
        events = [{"type": {"name": "Pass"}, "team": {"name": "Brazil"}},
                  {"type": {"name": "Pass"}, "team": {"name": "Croatia"}}]
        self.assertEqual(_teams_from_events(events), ("Brazil", "Croatia"))

    def test_empty_events(self):
        self.assertEqual(_teams_from_events([]), ("", ""))


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


class LineupProfileEndpointTests(unittest.TestCase):
    def setUp(self):
        self.client = create_app().test_client()

    def test_lineup_endpoint(self):
        team = TeamLineup(
            team="Argentina", formation="442", manager="Scaloni",
            starting_xi=[PlayerSlot(name="Lionel Messi", number=10,
                                    position="Center Forward", is_captain=True)],
            substitutes=[PlayerSlot(name="Paulo Dybala", number=21)],
        )
        with mock.patch("data_extraction.lineups.fetch_lineups", return_value=[team]):
            r = self.client.get("/api/lineup?match=sample&language=es")
        self.assertEqual(r.status_code, 200)
        data = r.get_json()
        self.assertEqual(data["teams"][0]["team"], "Argentina")
        self.assertIn("<svg", data["teams"][0]["svg"])
        self.assertEqual(data["labels"]["manager"], "Entrenador")   # localised (es)

    def test_lineup_empty_is_ok(self):
        with mock.patch("data_extraction.lineups.fetch_lineups", return_value=[]):
            r = self.client.get("/api/lineup?match=sample")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.get_json()["teams"], [])

    def test_profile_endpoint(self):
        canned = {"name": "Lionel Messi", "language": "es-ES", "photo_url": "http://img",
                  "source_url": "http://wiki", "grounded": True, "profile": "Un futbolista."}
        with mock.patch("profiles.profile_builder.build_profile", return_value=canned):
            r = self.client.get("/api/profile?player=Lionel%20Messi&language=es")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.get_json()["profile"], "Un futbolista.")

    def test_profile_missing_player_returns_400(self):
        self.assertEqual(self.client.get("/api/profile").status_code, 400)


class VisionMatchTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        ev = Path(self._tmp.name) / "events.json"
        ev.write_text(json.dumps([{
            "index": 1, "type": {"name": "Shot"}, "team": {"name": "Blue"},
            "period": 1, "minute": 1, "second": 0, "timestamp": "00:01:00.000",
            "shot": {"outcome": {"name": "Goal"}}}]), encoding="utf-8")
        self._patch = mock.patch.object(webapp, "VISION_EVENTS", ev)
        self._patch.start()

    def tearDown(self):
        self._patch.stop()
        self._tmp.cleanup()

    def test_matches_lists_vision_when_file_exists(self):
        with mock.patch.object(webapp, "CACHE_DIR", Path(self._tmp.name) / "nocache"):
            data = create_app().test_client().get("/api/matches").get_json()
        self.assertTrue(any(m["id"] == "vision" for m in data))

    def test_load_events_reads_vision_file(self):
        from web.app import _load_events
        events = _load_events("vision")
        self.assertEqual(events[0]["team"]["name"], "Blue")


class StartupPrewarmTests(unittest.TestCase):
    def test_demo_player_names_collects_xi_and_subs(self):
        from web.app import _demo_player_names
        team = TeamLineup(
            team="Argentina", formation="442", manager="Scaloni",
            starting_xi=[PlayerSlot(name="Lionel Messi", number=10)],
            substitutes=[PlayerSlot(name="Paulo Dybala", number=21)],
        )
        with mock.patch("data_extraction.lineups.fetch_lineups", return_value=[team]):
            names = _demo_player_names("sample")
        self.assertEqual(names, ["Lionel Messi", "Paulo Dybala"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
