"""Unit tests for data_pipeline (commentary_pipeline + live_cv_pipeline). All offline."""
import os
import types
import unittest
from unittest import mock

from text_to_speech.speak import SpeechResult
from data_pipeline import commentary_pipeline as cp
from data_pipeline import live_cv_pipeline as cv


class TempoTests(unittest.TestCase):
    def test_goal_is_fastest(self):
        item = types.SimpleNamespace(kind="goal", event={"type": {"name": "Shot"}})
        self.assertEqual(cp.tempo_for(item), 1.30)  # base 1.0 + span 0.30 * intensity 1.0

    def test_quiet_pass_near_base(self):
        item = types.SimpleNamespace(kind="call", event={"type": {"name": "Pass"}, "pass": {}})
        self.assertAlmostEqual(cp.tempo_for(item), 1.036, places=3)


class EnvFloatTests(unittest.TestCase):
    def test_reads_and_falls_back(self):
        with mock.patch.dict(os.environ, {"REPLAY_SPEED": "2.5"}):
            self.assertEqual(cp._env_float("REPLAY_SPEED", 0.0), 2.5)
        with mock.patch.dict(os.environ, {"REPLAY_SPEED": "nope"}):
            self.assertEqual(cp._env_float("REPLAY_SPEED", 9.0), 9.0)
        env = {k: v for k, v in os.environ.items() if k != "REPLAY_SPEED"}
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertEqual(cp._env_float("REPLAY_SPEED", 7.0), 7.0)


class CommentaryOutputTests(unittest.TestCase):
    def test_audio_ready_reflects_speech(self):
        no_audio = SpeechResult(text="hi", language="en", provider="fake")
        out = cp.CommentaryOutput(event={"minute": 1, "second": 2}, text="hi", speech=no_audio)
        self.assertFalse(out.audio_ready())

        with_audio = SpeechResult(text="hi", language="en", provider="fake", audio_bytes=b"A")
        out2 = cp.CommentaryOutput(event={"minute": 1, "second": 2}, text="hi", speech=with_audio)
        self.assertTrue(out2.audio_ready())

    def test_as_dict_shape(self):
        sr = SpeechResult(text="Goal!", language="es", provider="fake")
        out = cp.CommentaryOutput(
            event={"minute": 23, "second": 11, "type": {"name": "Shot"}}, text="Goal!", speech=sr)
        d = out.as_dict()
        self.assertEqual(d["minute"], 23)
        self.assertEqual(d["event_type"], "Shot")
        self.assertEqual(d["text"], "Goal!")
        self.assertEqual(d["language"], "es")
        self.assertFalse(d["audio_ready"])


class _FakeItem:
    def __init__(self, text, kind="call", event=None):
        self.text, self.kind, self.event, self.turns = text, kind, event or {}, []

    def as_dict(self):
        return {"kind": self.kind, "speaker": "lead", "text": self.text, "turns": []}


class _FakeAgent:
    language = "en"

    def __init__(self):
        self.seen = []

    def opening(self, competition="", home="", away="", briefing=""):
        return _FakeItem("Welcome", kind="opening")

    def handle_item(self, ev):
        self.seen.append(ev)
        if (ev.get("type") or {}).get("name") == "Shot":
            return _FakeItem("Shot!", kind="call", event=ev)
        return None


class _FakeSpeaker:
    def synthesize(self, text, language="en", speaking_rate=None):
        return SpeechResult(text=text, language=language, provider="fake")


class StreamCommentaryTests(unittest.TestCase):
    def test_stream_yields_opening_then_commented_events(self):
        events = [
            {"index": 1, "timestamp": "00:00:01.000", "type": {"name": "Pass"}},
            {"index": 2, "timestamp": "00:00:02.000", "type": {"name": "Shot"}},
        ]
        outs = list(cp.stream_commentary(
            events, agent=_FakeAgent(), speaker=_FakeSpeaker(), speed=0))
        self.assertEqual([o.text for o in outs], ["Welcome", "Shot!"])  # Pass produced no line


class LiveCvAdapterTests(unittest.TestCase):
    def test_label_to_type_mapping(self):
        self.assertEqual(cv._LABEL_TO_TYPE["shot"], "Shot")
        self.assertEqual(cv._LABEL_TO_TYPE["goal"], "Shot")
        self.assertEqual(cv._LABEL_TO_TYPE["pass"], "Pass")

    def test_to_events_shot_and_clock(self):
        det = cv.Detection(label="shot", team="Argentina", player="Messi",
                           location=[104, 38], confidence=0.9)
        evs = cv.LiveEventAdapter().to_events([det], seconds=65)
        self.assertEqual(len(evs), 1)
        ev = evs[0]
        self.assertEqual(ev["type"]["name"], "Shot")
        self.assertEqual((ev["minute"], ev["second"]), (1, 5))
        self.assertEqual(ev["shot"]["outcome"]["name"], "Saved")
        self.assertEqual(ev["location"], [104, 38])

    def test_to_events_goal_outcome(self):
        det = cv.Detection(label="goal", team="France", player="Mbappe", confidence=0.9)
        ev = cv.LiveEventAdapter().to_events([det], seconds=0)[0]
        self.assertEqual(ev["shot"]["outcome"]["name"], "Goal")

    def test_to_events_pass_end_location(self):
        det = cv.Detection(label="pass", team="Argentina", player="De Paul",
                           location=[60, 40], end_location=[88, 30], confidence=0.9)
        ev = cv.LiveEventAdapter().to_events([det], seconds=0)[0]
        self.assertEqual(ev["type"]["name"], "Pass")
        self.assertEqual(ev["pass"]["end_location"], [88, 30])

    def test_low_confidence_dropped(self):
        det = cv.Detection(label="pass", confidence=0.1)  # below min_confidence 0.4
        self.assertEqual(cv.LiveEventAdapter().to_events([det], seconds=0), [])

    def test_live_event_stream_from_mock_cv(self):
        evs = list(cv.live_event_stream(cv.MockFrameSource(n=2), cv.MockVisionEventDetector()))
        self.assertEqual(len(evs), 2)
        self.assertTrue(all("type" in e for e in evs))


if __name__ == "__main__":
    unittest.main(verbosity=2)
