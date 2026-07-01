"""Unit tests for the SoccerNet GSR adapter (vision_model.soccernet_gsr).

Fully offline: a tiny synthetic Labels-GameState.json fixture, no torch / OpenCV and no
real SoccerNet download. The headline test proves GSR positions flow through the existing
event inference and drive CommentaryAgent(mock=True) unchanged.
"""
import unittest

from vision_model.detector import BALL, PLAYER
from vision_model.schema import PITCH_LENGTH, PITCH_WIDTH
from vision_model.soccernet_gsr import build_script, events_from_labels, gsr_to_statsbomb


def _ann(image_id, role, x_m, y_m, *, track_id=None, team=None, jersey=None):
    """One GSR annotation dict carrying a bottom-middle pitch point (metres)."""
    return {
        "image_id": image_id,
        "track_id": track_id,
        "attributes": {"role": role, "team": team, "jersey": jersey},
        "bbox_pitch": {"x_bottom_middle": x_m, "y_bottom_middle": y_m},
    }


def _labels(annotations, n_frames=3):
    """Wrap annotations in a minimal Labels-GameState.json structure."""
    images = [{"image_id": i + 1, "file_name": f"{i + 1:06d}.jpg"} for i in range(n_frames)]
    return {"info": {"version": "1.3"}, "images": images, "annotations": annotations}


class CoordinateTransformTests(unittest.TestCase):
    def test_corners_and_centre(self):
        self.assertEqual(gsr_to_statsbomb(-52.5, -34.0), (0.0, 0.0))
        self.assertEqual(gsr_to_statsbomb(52.5, 34.0), (PITCH_LENGTH, PITCH_WIDTH))
        self.assertEqual(gsr_to_statsbomb(0.0, 0.0), (60.0, 40.0))

    def test_off_pitch_is_clamped(self):
        x, y = gsr_to_statsbomb(999.0, -999.0)
        self.assertTrue(0.0 <= x <= PITCH_LENGTH and 0.0 <= y <= PITCH_WIDTH)


class BuildScriptTests(unittest.TestCase):
    def test_roles_and_teams_mapped(self):
        anns = [
            _ann(1, "ball", 0.0, 0.0),
            _ann(1, "player", 0.0, 0.0, track_id=10, team="left", jersey=10),
            _ann(1, "referee", 5.0, 5.0, track_id=99),        # skipped
            _ann(1, "other", 6.0, 6.0, track_id=98),          # skipped
        ]
        script, names, (home, away) = build_script(
            _labels(anns), team_names={"left": "Argentina", "right": "France"})
        kinds = sorted(d.cls for d in script[0])
        self.assertEqual(kinds, [BALL, PLAYER])               # referee/other dropped
        player = [d for d in script[0] if d.cls == PLAYER][0]
        self.assertEqual(player.team, "Argentina")
        self.assertEqual(names[10], "#10")
        self.assertEqual((home, away), ("Argentina", "France"))

    def test_default_team_names(self):
        anns = [_ann(1, "player", 0.0, 0.0, track_id=1, team="right")]
        script, _, (home, away) = build_script(_labels(anns))
        self.assertEqual((home, away), ("Left Team", "Right Team"))
        self.assertEqual(script[0][0].team, "Right Team")

    def test_missing_pitch_point_skipped(self):
        anns = [{"image_id": 1, "track_id": 1,
                 "attributes": {"role": "player", "team": "left"}, "bbox_pitch": None}]
        script, _, _ = build_script(_labels(anns))
        self.assertEqual(script[0], [])


class EventsTests(unittest.TestCase):
    """A scripted pass-then-goal, expressed in GSR metres, mirrors the stub demo."""

    def _pass_then_goal_labels(self):
        # gsr(0,0)->sb(60,40); gsr(35,0.85)->sb(100,41); gsr(52.325,0)->sb(119.8,40).
        a_xy, b_xy = (0.0, 0.0), (35.0, 0.85)
        anns = []
        for img in (1, 2, 3):
            anns.append(_ann(img, "player", *a_xy, track_id=10, team="left", jersey=10))
            anns.append(_ann(img, "player", *b_xy, track_id=7, team="left", jersey=7))
        anns.append(_ann(1, "ball", 0.875, 0.0))      # ball at #10
        anns.append(_ann(2, "ball", 35.0, 0.85))      # ball arrives at #7  -> Pass
        anns.append(_ann(3, "ball", 52.325, 0.0))     # ball in the goal mouth -> Shot/Goal
        return _labels(anns)

    def test_pass_and_goal_inferred(self):
        events = events_from_labels(self._pass_then_goal_labels())
        types = [e["type"]["name"] for e in events]
        self.assertIn("Pass", types)
        self.assertIn("Shot", types)
        pass_ev = next(e for e in events if e["type"]["name"] == "Pass")
        self.assertEqual(pass_ev["player"]["name"], "#10")
        self.assertEqual(pass_ev["pass"]["recipient"]["name"], "#7")
        goal_ev = next(e for e in events if e["type"]["name"] == "Shot")
        self.assertEqual(goal_ev["shot"]["outcome"]["name"], "Goal")

    def test_events_drive_commentary_agent(self):
        from agent.commentary_agent import CommentaryAgent
        from data_replayer.replayer import replay
        events = events_from_labels(self._pass_then_goal_labels())
        agent = CommentaryAgent(mock=True)
        lines = [agent.handle(ev) for ev in replay(events, speed=0.0)]
        self.assertTrue(any(lines), "GSR events produced no commentary")


if __name__ == "__main__":
    unittest.main(verbosity=2)
