"""Unit tests for spike.go_no_go (the go/no-go commentary spike). All offline."""
import types
import unittest
from unittest import mock

from spike import go_no_go as gng


class _Resp:
    def __init__(self, content):
        self.choices = [types.SimpleNamespace(message=types.SimpleNamespace(content=content))]

class _Completions:
    def __init__(self, content):
        self.content = content; self.calls = []
    def create(self, **kw):
        self.calls.append(kw); return _Resp(self.content)

class FakeClient:
    def __init__(self, content="GOL!"):
        self.chat = types.SimpleNamespace(completions=_Completions(content))


class ConstantsTests(unittest.TestCase):
    def test_language_names(self):
        self.assertEqual(gng.LANGUAGE_NAMES["en"], "English")
        self.assertIn("es", gng.LANGUAGE_NAMES)

    def test_skip_types(self):
        self.assertIn("Pressure", gng.SKIP_TYPES)
        self.assertIn("Carry", gng.SKIP_TYPES)


class DescribeEventTests(unittest.TestCase):
    def test_pass_with_detail(self):
        ev = {"minute": 23, "second": 11, "type": {"name": "Pass"},
              "team": {"name": "Argentina"}, "player": {"name": "Rodrigo De Paul"},
              "pass": {"recipient": {"name": "Lionel Messi"}, "outcome": {"name": "Complete"}}}
        s = gng.describe_event(ev)
        self.assertIn("[23:11] Pass", s)
        self.assertIn("Argentina", s)
        self.assertIn("to Lionel Messi", s)

    def test_shot_with_xg(self):
        ev = {"minute": 80, "second": 5, "type": {"name": "Shot"}, "team": {"name": "Argentina"},
              "player": {"name": "Messi"}, "shot": {"outcome": {"name": "Goal"},
              "body_part": {"name": "Left Foot"}, "statsbomb_xg": 0.27}}
        s = gng.describe_event(ev)
        self.assertIn("Shot", s); self.assertIn("Goal", s); self.assertIn("xG 0.27", s)

    def test_basic_event(self):
        ev = {"minute": 5, "second": 0, "type": {"name": "Foul Committed"}, "team": {"name": "France"}}
        self.assertIn("[05:00] Foul Committed", gng.describe_event(ev))


class WindowTests(unittest.TestCase):
    def test_select_window_slices(self):
        events = [{"type": {"name": "Pass"}} for _ in range(10)]
        self.assertEqual(len(gng.select_window(events, 2, 3, dense=False)), 3)

    def test_dense_drops_low_signal(self):
        events = [{"type": {"name": "Pressure"}}, {"type": {"name": "Shot"}}, {"type": {"name": "Carry"}}]
        out = gng.select_window(events, 0, 3, dense=True)
        self.assertEqual([e["type"]["name"] for e in out], ["Shot"])


class PromptTests(unittest.TestCase):
    def test_build_prompt_contains_rules_and_feed(self):
        events = [{"minute": 1, "second": 0, "type": {"name": "Shot"}, "team": {"name": "Argentina"}}]
        p = gng.build_prompt(events, "es")
        self.assertIn("Spanish", p)
        self.assertIn("invent", p.lower())
        self.assertIn("Shot", p)


class LoadEventsTests(unittest.TestCase):
    def test_load_events_delegates_to_api_fetch(self):
        with mock.patch("data_extraction.loader.fetch_events", return_value=[{"x": 1}]) as fe:
            self.assertEqual(gng.load_events(None), [{"x": 1}])
            fe.assert_called_once_with(None)

    def test_load_events_passes_match_id(self):
        with mock.patch("data_extraction.loader.fetch_events", return_value=[]) as fe:
            gng.load_events(3869685)
            fe.assert_called_once_with(3869685)


class CallGraniteTests(unittest.TestCase):
    def test_call_granite_uses_shared_client(self):
        fake = FakeClient("GOL! Messi marca.")
        with mock.patch("agent.granite_client.build_granite_client", return_value=fake), \
             mock.patch("agent.granite_client.model_id", return_value="granite-x"):
            out = gng.call_granite("PROMPT")
        self.assertEqual(out, "GOL! Messi marca.")
        call = fake.chat.completions.calls[0]
        self.assertEqual(call["model"], "granite-x")
        self.assertEqual(call["messages"][0]["content"], "PROMPT")


if __name__ == "__main__":
    unittest.main(verbosity=2)
