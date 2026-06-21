"""Unit tests for vision_model. Offline: no torch/ultralytics/OpenCV, no video.

The headline test is SchemaCompatibilityTests — it proves stub vision output flows
through data_replayer.replay() and produces commentary from CommentaryAgent(mock=True),
i.e. the vision events are drop-in compatible with the existing agent.
"""
import unittest

from vision_model import schema
from vision_model.detector import BALL, PLAYER, Detection, FrameDetections, StubDetector, build_detector
from vision_model.events import EventBuilder, nearest_player
from vision_model.pitch import LinearPitchMapper
from vision_model.pipeline import demo_events


class SchemaTests(unittest.TestCase):
    def test_seconds_to_clock_first_half(self):
        c = schema.seconds_to_clock(65, period=1)
        self.assertEqual((c["period"], c["minute"], c["second"]), (1, 1, 5))
        self.assertEqual(c["timestamp"], "00:01:05.000")

    def test_seconds_to_clock_second_half_continues_from_45(self):
        c = schema.seconds_to_clock(30, period=2)
        self.assertEqual((c["period"], c["minute"], c["second"]), (2, 45, 30))

    def test_make_event_shape(self):
        ev = schema.make_event(
            3, "Shot", team="Blue", player="P7", location=(100, 41), elapsed_s=10,
            **{"shot": schema.shot("Goal", end_location=(119.8, 40), xg=0.4)},
        )
        self.assertEqual(ev["index"], 3)
        self.assertEqual(ev["type"]["name"], "Shot")
        self.assertEqual(ev["team"]["name"], "Blue")
        self.assertEqual(ev["location"], [100.0, 41.0])
        self.assertEqual(ev["shot"]["outcome"]["name"], "Goal")
        self.assertEqual(ev["shot"]["end_location"], [119.8, 40.0])
        self.assertEqual(ev["shot"]["statsbomb_xg"], 0.4)

    def test_completed_pass_has_no_outcome_key(self):
        p = schema.pass_(recipient="P7", end_location=(100, 41))
        self.assertNotIn("outcome", p)               # StatsBomb: complete passes omit outcome
        self.assertEqual(p["recipient"]["name"], "P7")
        self.assertEqual(schema.pass_(outcome="Incomplete")["outcome"]["name"], "Incomplete")


class DetectorTests(unittest.TestCase):
    def test_stub_detector_replays_script(self):
        det = build_detector("stub", script=[[Detection(BALL)], []])
        self.assertEqual(len(det.detect(0)), 1)
        self.assertEqual(det.detect(1), [])
        self.assertEqual(det.detect(99), [])         # out of range -> empty

    def test_unknown_kind_raises(self):
        with self.assertRaises(ValueError):
            build_detector("magic")


class PitchTests(unittest.TestCase):
    def test_scales_pixels_into_pitch(self):
        mapper = LinearPitchMapper(1000, 500)
        self.assertEqual(mapper.to_pitch(500, 250), (60.0, 40.0))
        x, y = mapper.to_pitch(5000, -10)            # clamped
        self.assertTrue(0.0 <= x <= schema.PITCH_LENGTH and 0.0 <= y <= schema.PITCH_WIDTH)

    def test_map_frame_keeps_preset_pitch_coords(self):
        mapper = LinearPitchMapper(1000, 500)
        fd = FrameDetections(0, 0.0, [
            Detection(PLAYER, bbox=(0, 0, 100, 100)),     # no xy_pitch -> filled
            Detection(BALL, xy_pitch=(10.0, 10.0)),       # preset -> kept
        ])
        mapper.map_frame(fd)
        self.assertIsNotNone(fd.detections[0].xy_pitch)
        self.assertEqual(fd.detections[1].xy_pitch, (10.0, 10.0))


class EventInferenceTests(unittest.TestCase):
    def _frame(self, i, ball_xy, players):
        dets = [Detection(BALL, xy_pitch=ball_xy)]
        dets += [Detection(PLAYER, track_id=t, team="Blue", xy_pitch=xy) for t, xy in players]
        return FrameDetections(i, i * 0.04, dets)

    def test_nearest_player_within_reach(self):
        fd = self._frame(0, (50, 40), [(1, (51, 40)), (2, (80, 40))])
        self.assertEqual(nearest_player(fd).track_id, 1)

    def test_nearest_player_none_when_too_far(self):
        fd = self._frame(0, (50, 40), [(1, (60, 40))])   # 10 units > POSSESSION_RADIUS
        self.assertIsNone(nearest_player(fd))

    def test_same_team_handover_is_a_pass(self):
        builder = EventBuilder()
        builder.observe(self._frame(0, (61, 40), [(10, (60, 40)), (7, (100, 41))]))
        builder.observe(self._frame(1, (100, 41), [(10, (60, 40)), (7, (100, 41))]))
        self.assertEqual(len(builder.events), 1)
        ev = builder.events[0]
        self.assertEqual(ev["type"]["name"], "Pass")
        self.assertEqual(ev["player"]["name"], "Player 10")
        self.assertEqual(ev["pass"]["recipient"]["name"], "Player 7")


class SchemaCompatibilityTests(unittest.TestCase):
    """The integration guarantee: stub vision events feed the real agent unchanged."""

    def test_demo_events_drive_the_commentary_agent(self):
        from data_replayer.replayer import replay
        from agent.commentary_agent import CommentaryAgent

        events = demo_events()
        self.assertTrue(events, "demo produced no events")

        required = {"index", "type", "period", "minute", "second", "timestamp"}
        for ev in events:
            self.assertTrue(required <= set(ev), f"missing keys in {ev}")

        type_names = [e["type"]["name"] for e in events]
        self.assertIn("Pass", type_names)
        self.assertIn("Shot", type_names)

        ordered = list(replay(events, speed=0.0))
        self.assertEqual([e["index"] for e in ordered], sorted(e["index"] for e in events))

        agent = CommentaryAgent(mock=True)
        lines = [agent.handle(ev) for ev in replay(events, speed=0.0)]
        self.assertTrue(any(lines), "agent produced no commentary from vision events")


class FormationPreprocessTests(unittest.TestCase):
    """Tests for vision_model.formation.preprocess (no torch needed)."""

    def test_normalises_pitch_coordinates(self):
        from vision_model.formation import preprocess, N_PLAYERS
        from vision_model.schema import PITCH_LENGTH, PITCH_WIDTH
        # One player at the centre of the pitch.
        features = preprocess([(PITCH_LENGTH / 2, PITCH_WIDTH / 2)])
        self.assertAlmostEqual(features[0], 0.5)
        self.assertAlmostEqual(features[1], 0.5)

    def test_sorts_players_deepest_first(self):
        from vision_model.formation import preprocess
        # Forward at x=100, defender at x=20 — after preprocess, x[0] should be < x[2].
        features = preprocess([(100, 40), (20, 40)])
        self.assertLess(features[0], features[2])

    def test_pads_short_list_with_sentinel(self):
        from vision_model.formation import preprocess, N_PLAYERS, _SENTINEL
        features = preprocess([(60, 40)])          # only 1 player
        self.assertEqual(len(features), N_PLAYERS * 2)
        # The remaining 9 pairs should be the sentinel value.
        self.assertEqual(features[2], _SENTINEL)
        self.assertEqual(features[3], _SENTINEL)

    def test_truncates_to_n_players(self):
        from vision_model.formation import preprocess, N_PLAYERS
        many = [(float(i), 40.0) for i in range(15)]
        features = preprocess(many)
        self.assertEqual(len(features), N_PLAYERS * 2)

    def test_flips_x_for_right_to_left_team(self):
        from vision_model.formation import preprocess
        # A player at x=20 (deep in their own half) attacking right→left
        # should appear as x≈0.83 in feature space (flipped).
        features_ltr = preprocess([(20, 40)], attack_left_to_right=True)
        features_rtl = preprocess([(20, 40)], attack_left_to_right=False)
        self.assertAlmostEqual(features_ltr[0] + features_rtl[0], 1.0, places=5)


class FormationPredictorTests(unittest.TestCase):
    """Tests for StubFormationPredictor and build_predictor (no torch needed)."""

    def test_stub_returns_scripted_formation(self):
        from vision_model.formation import StubFormationPredictor
        stub = StubFormationPredictor(["4-4-2"])
        self.assertEqual(stub.predict([]), "4-4-2")

    def test_stub_cycles_round_robin(self):
        from vision_model.formation import StubFormationPredictor
        stub = StubFormationPredictor(["4-4-2", "4-3-3"])
        results = [stub.predict([]) for _ in range(4)]
        self.assertEqual(results, ["4-4-2", "4-3-3", "4-4-2", "4-3-3"])

    def test_stub_accepts_position_list_as_positional(self):
        from vision_model.formation import StubFormationPredictor
        stub = StubFormationPredictor(["4-3-3"])
        positions = [(20, 15), (20, 38), (20, 62), (20, 85),
                     (50, 25), (50, 50), (50, 75),
                     (80, 20), (80, 50), (80, 80)]
        self.assertEqual(stub.predict(positions), "4-3-3")

    def test_build_predictor_returns_stub_by_default(self):
        from vision_model.formation import build_predictor, StubFormationPredictor
        p = build_predictor("stub")
        self.assertIsInstance(p, StubFormationPredictor)

    def test_build_predictor_raises_for_unknown_kind(self):
        from vision_model.formation import build_predictor
        with self.assertRaises(ValueError):
            build_predictor("magic")


class TacticsTests(unittest.TestCase):
    """Tests for vision_model.tactics (no torch needed)."""

    _ATK = "Blue"
    _DEF = "Red"

    def _make_fd(self, frame_idx, ball_xy, attackers, defenders):
        """Build a FrameDetections with attacker and defender players."""
        from vision_model.detector import Detection, FrameDetections, BALL, PLAYER
        dets = [Detection(BALL, xy_pitch=ball_xy)]
        for tid, xy in attackers:
            dets.append(Detection(PLAYER, track_id=tid, team=self._ATK, xy_pitch=xy))
        for tid, xy in defenders:
            dets.append(Detection(PLAYER, track_id=tid, team=self._DEF, xy_pitch=xy))
        return FrameDetections(frame_idx, frame_idx * 0.04, dets)

    def test_crowd_box_detected_when_three_attackers_in_area(self):
        from vision_model.tactics import analyse, PENALTY_AREA_X
        # Three attackers inside the 18-yard box.
        fd = self._make_fd(
            0, (110, 40),
            [(1, (103, 25)), (2, (105, 40)), (3, (108, 55))],
            [(10, (95, 30)), (11, (95, 50))],
        )
        report = analyse([fd] * 8, self._ATK, self._DEF)
        patterns = [o.pattern for o in report.observations]
        self.assertIn("crowd_box", patterns)

    def test_crowd_box_not_detected_with_too_few_attackers(self):
        from vision_model.tactics import analyse
        fd = self._make_fd(0, (80, 40), [(1, (103, 30))], [(10, (90, 40))])
        report = analyse([fd] * 8, self._ATK, self._DEF)
        patterns = [o.pattern for o in report.observations]
        self.assertNotIn("crowd_box", patterns)

    def test_man_marking_detected_when_defender_shadows_attacker(self):
        from vision_model.tactics import analyse, MAN_MARK_RADIUS, MIN_MARK_FRAMES
        # Defender stays within marking radius across many frames.
        frames = [
            self._make_fd(i, (60, 40), [(1, (60 + i * 0.5, 40))], [(10, (61 + i * 0.5, 40))])
            for i in range(MIN_MARK_FRAMES + 2)
        ]
        report = analyse(frames, self._ATK, self._DEF)
        patterns = [o.pattern for o in report.observations]
        self.assertIn("man_marking", patterns)

    def test_drag_defenders_detected_when_attacker_pulls_followers(self):
        from vision_model.tactics import analyse
        # Attacker moves 15 units forward; two defenders follow in the same direction.
        frame0 = self._make_fd(
            0, (50, 40),
            [(1, (50, 40))],
            [(10, (55, 35)), (11, (55, 45))],
        )
        frame1 = self._make_fd(
            1, (75, 40),
            [(1, (75, 40))],              # attacker moved +25 x
            [(10, (70, 35)), (11, (70, 45))],  # defenders followed
        )
        report = analyse([frame0, frame1], self._ATK, self._DEF)
        patterns = [o.pattern for o in report.observations]
        self.assertIn("drag_defenders", patterns)

    def test_analyse_empty_frames_returns_empty_report(self):
        from vision_model.tactics import analyse
        report = analyse([], self._ATK, self._DEF)
        self.assertEqual(report.observations, [])
        self.assertIsNone(report.key_movement)

    def test_analyse_passes_formation_strings_through(self):
        from vision_model.tactics import analyse
        report = analyse([], self._ATK, self._DEF,
                         formation_before="4-4-2", formation_after="4-3-3")
        self.assertEqual(report.formation_before, "4-4-2")
        self.assertEqual(report.formation_after, "4-3-3")

    def test_key_movement_is_highest_confidence_pattern(self):
        from vision_model.tactics import analyse
        # Crowd box (3 attackers in area, high confidence) + limited other signals.
        fd = self._make_fd(
            0, (110, 40),
            [(1, (103, 25)), (2, (105, 40)), (3, (108, 55))],
            [(10, (95, 30)), (11, (95, 50))],
        )
        report = analyse([fd] * 10, self._ATK, self._DEF)
        if report.observations:
            best = max(report.observations, key=lambda o: o.confidence)
            self.assertEqual(report.key_movement, best.pattern)


class TrainerDataTests(unittest.TestCase):
    """Tests for vision_model.trainer that run without torch."""

    def test_canonical_positions_all_have_n_players(self):
        from vision_model.trainer import _CANONICAL
        from vision_model.formation import N_PLAYERS
        for formation, positions in _CANONICAL.items():
            self.assertEqual(
                len(positions), N_PLAYERS,
                f"_CANONICAL[{formation!r}] has {len(positions)} players, expected {N_PLAYERS}",
            )

    def test_generate_dataset_correct_size(self):
        from vision_model.trainer import generate_dataset
        from vision_model.formation import FORMATIONS
        n = 10
        X, y = generate_dataset(n_per_formation=n, seed=0)
        self.assertEqual(len(X), len(FORMATIONS) * n)
        self.assertEqual(len(y), len(X))

    def test_generate_dataset_feature_vector_length(self):
        from vision_model.trainer import generate_dataset
        from vision_model.formation import N_PLAYERS
        X, _ = generate_dataset(n_per_formation=5, seed=1)
        self.assertTrue(all(len(row) == N_PLAYERS * 2 for row in X))

    def test_generate_dataset_is_reproducible(self):
        from vision_model.trainer import generate_dataset
        X1, y1 = generate_dataset(n_per_formation=5, seed=42)
        X2, y2 = generate_dataset(n_per_formation=5, seed=42)
        self.assertEqual(X1, X2)
        self.assertEqual(y1, y2)

    def test_generate_dataset_different_seeds_differ(self):
        from vision_model.trainer import generate_dataset
        X1, _ = generate_dataset(n_per_formation=5, seed=1)
        X2, _ = generate_dataset(n_per_formation=5, seed=2)
        self.assertNotEqual(X1, X2)

    def test_labels_cover_all_formations(self):
        from vision_model.trainer import generate_dataset
        from vision_model.formation import FORMATIONS
        _, y = generate_dataset(n_per_formation=5, seed=0)
        self.assertEqual(set(y), set(range(len(FORMATIONS))))


class TacticalContextPipelineTests(unittest.TestCase):
    """Tests for pipeline.with_tactical_context (no torch needed)."""

    def _goal_event(self):
        from vision_model.schema import make_event, shot
        return make_event(
            1, "Shot", team="Blue", player="Player 7",
            location=(100, 40), elapsed_s=10,
            **{"shot": shot("Goal", end_location=(119.8, 40))},
        )

    def _pass_event(self):
        from vision_model.schema import make_event, pass_
        return make_event(
            2, "Pass", team="Blue", player="Player 10",
            location=(60, 40), elapsed_s=8,
            **{"pass": pass_(recipient="Player 7", end_location=(100, 40))},
        )

    def _frames(self):
        """A few FrameDetections spanning 0–12 seconds."""
        from vision_model.detector import Detection, FrameDetections, BALL, PLAYER
        frames = []
        for i in range(30):  # 30 frames, ~0.4 s each
            t = i * 0.4
            dets = [
                Detection(BALL, xy_pitch=(60 + i * 2, 40)),
                Detection(PLAYER, track_id=7, team="Blue", xy_pitch=(95 + i * 0.5, 40)),
                Detection(PLAYER, track_id=10, team="Red", xy_pitch=(90 + i * 0.3, 35)),
            ]
            frames.append(FrameDetections(i, t, dets))
        return frames

    def test_goal_event_gets_vision_context(self):
        from vision_model.pipeline import with_tactical_context
        events = [self._goal_event()]
        enriched = with_tactical_context(events, self._frames(), "Blue", "Red")
        self.assertIn("vision_context", enriched[0])

    def test_vision_context_has_expected_keys(self):
        from vision_model.pipeline import with_tactical_context
        events = [self._goal_event()]
        enriched = with_tactical_context(events, self._frames(), "Blue", "Red")
        ctx = enriched[0]["vision_context"]
        for key in ("formation_before", "formation_after", "tactical_patterns", "key_movement"):
            self.assertIn(key, ctx)

    def test_pass_event_not_enriched(self):
        from vision_model.pipeline import with_tactical_context
        events = [self._pass_event()]
        enriched = with_tactical_context(events, self._frames(), "Blue", "Red")
        self.assertNotIn("vision_context", enriched[0])

    def test_original_events_not_mutated(self):
        from vision_model.pipeline import with_tactical_context
        ev = self._goal_event()
        original_keys = set(ev.keys())
        with_tactical_context([ev], self._frames(), "Blue", "Red")
        self.assertEqual(set(ev.keys()), original_keys)

    def test_video_to_events_return_frames_flag(self):
        """return_frames=True should give (events, frames) tuple."""
        from vision_model.pipeline import video_to_events, build_demo
        frame_iter, detector = build_demo()
        result = video_to_events(frame_iter, detector=detector, return_frames=True)
        self.assertIsInstance(result, tuple)
        events, frames = result
        self.assertIsInstance(events, list)
        self.assertIsInstance(frames, list)
        self.assertTrue(len(frames) > 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
