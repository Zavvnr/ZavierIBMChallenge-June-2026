"""Integration: replayer -> real CommentaryAgent -> pipeline -> TTS (+ live-CV source swap).

Uses the real components end to end; only the Granite endpoint is faked. Requires the
minimal agent.prompts stub (see _helpers.ensure_prompts) until agent.prompts is built.
"""
import unittest

from tests.integration_tests._helpers import ensure_prompts, build_up_with_goal, FakeGranite
from data_pipeline.commentary_pipeline import stream_commentary
from data_pipeline import live_cv_pipeline as cv
from agent.commentary_agent import CommentaryAgent


class MockModeEndToEnd(unittest.TestCase):
    def setUp(self):
        ensure_prompts()

    def test_pipeline_narrates_the_goal(self):
        outs = list(stream_commentary(build_up_with_goal(), mock=True, speed=0))
        self.assertGreaterEqual(len(outs), 2)                      # opening + >=1 line
        self.assertTrue(any("GOAL" in o.text.upper() for o in outs))

    def test_skip_type_events_produce_no_lines(self):
        filler = [{"index": i, "timestamp": f"00:00:{i:02d}.000", "type": {"name": "Pressure"}}
                  for i in range(5)]
        outs = list(stream_commentary(filler, mock=True, speed=0))
        # Opening may appear (event == {}); no event-driven lines for pure skip types.
        self.assertEqual([o.text for o in outs if o.event], [])

    def test_score_tracked_after_goal(self):
        agent = CommentaryAgent(language="en", mock=True)
        list(stream_commentary(build_up_with_goal(), agent=agent, speed=0))
        self.assertEqual(agent.state.score.get("Argentina"), 1)


class FakeGraniteEndToEnd(unittest.TestCase):
    def setUp(self):
        ensure_prompts()

    def test_real_agent_uses_injected_granite_client(self):
        fake = FakeGranite("Vamos Argentina!")
        agent = CommentaryAgent(language="en", mock=False, client=fake)
        outs = list(stream_commentary(build_up_with_goal(), agent=agent, speed=0))
        self.assertTrue(outs)
        self.assertTrue(all(o.text == "Vamos Argentina!" for o in outs))
        self.assertGreaterEqual(fake.calls, 1)                     # Granite actually called


class TwoSpeakerEndToEnd(unittest.TestCase):
    def setUp(self):
        ensure_prompts()

    def test_goal_yields_lead_and_analyst_turns(self):
        agent = CommentaryAgent(language="en", mock=True, two_speakers=True)
        outs = list(stream_commentary(build_up_with_goal(), agent=agent, speed=0))
        multi = [o for o in outs if o.item and len(o.item.turns) > 1]
        self.assertTrue(multi)
        speakers = {t.speaker for o in multi for t in o.item.turns}
        self.assertIn("analyst", speakers)


class LiveCvEndToEnd(unittest.TestCase):
    def setUp(self):
        ensure_prompts()

    def test_vision_events_flow_through_pipeline(self):
        events = cv.live_event_stream(cv.MockFrameSource(n=3), cv.MockVisionEventDetector())
        outs = list(stream_commentary(events, mock=True, speed=0))
        self.assertTrue(any((o.event.get("type") or {}).get("name") == "Shot" for o in outs))


if __name__ == "__main__":
    unittest.main(verbosity=2)
