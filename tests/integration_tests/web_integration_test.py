"""Integration: the Flask web layer driving the real pipeline (SSE stream) and Granite highlight.

External events and the Granite endpoint are faked; the web -> pipeline -> agent wiring is real.
"""
import unittest
from unittest import mock

from tests.integration_tests._helpers import ensure_prompts, build_up_with_goal, FakeGranite
from web.app import create_app
from web import app as webapp


class WebStreamIntegration(unittest.TestCase):
    def setUp(self):
        ensure_prompts()
        self.client = create_app().test_client()

    def test_stream_emits_commentary_and_done(self):
        with mock.patch.object(webapp, "_load_events", return_value=build_up_with_goal()):
            r = self.client.get("/api/stream?match=sample&mock=1&speed=0")
            body = r.get_data(as_text=True)
        self.assertEqual(r.status_code, 200)
        self.assertIn("data:", body)            # at least one SSE commentary line
        self.assertIn("event: done", body)      # stream terminates cleanly
        self.assertIn("GOAL", body.upper())     # the goal was narrated through the stack

    def test_stream_missing_match_is_graceful(self):
        with mock.patch.object(webapp, "_load_events",
                               side_effect=FileNotFoundError("Match 999 is not cached.")):
            r = self.client.get("/api/stream?match=999")
            body = r.get_data(as_text=True)
        self.assertEqual(r.status_code, 200)
        self.assertIn("streamerror", body)

    def test_ask_via_granite(self):
        fake = FakeGranite("Offside under Law 11 — ahead of the second-last defender.")
        with mock.patch.object(webapp, "_load_events", return_value=build_up_with_goal()), \
             mock.patch("agent.granite_client.build_granite_client", return_value=fake), \
             mock.patch("agent.granite_client.model_id", return_value="granite-x"), \
             mock.patch("context.retrieve.retrieve", return_value=[{"text": "Law 11 — Offside", "score": 0.9}]):
            r = self.client.get("/api/ask?q=why+was+it+offside&match=sample&language=en")
        self.assertEqual(r.status_code, 200)
        data = r.get_json()
        self.assertIn("Offside", data["answer"])
        self.assertEqual(data["via"], "ibm-granite")

    def test_ask_requires_question(self):
        self.assertEqual(self.client.get("/api/ask?match=sample").status_code, 400)


if __name__ == "__main__":
    unittest.main(verbosity=2)
