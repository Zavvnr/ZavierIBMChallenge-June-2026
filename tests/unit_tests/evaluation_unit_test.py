"""Unit tests for the evaluation harness. Offline: mock agent, no Granite, no network."""
import unittest

from evaluation import metrics
from evaluation.harness import evaluate

PASS_EVENT = {
    "index": 1, "period": 1, "minute": 10, "second": 0, "timestamp": "00:10:00.000",
    "type": {"name": "Pass"}, "team": {"name": "Argentina"},
    "player": {"name": "Lionel Messi"}, "location": [60, 40],
    "pass": {"recipient": {"name": "Julian Alvarez"}, "end_location": [100, 40]},
}
GOAL_EVENT = {
    "index": 2, "period": 1, "minute": 11, "second": 0, "timestamp": "00:11:00.000",
    "type": {"name": "Shot"}, "team": {"name": "Argentina"},
    "player": {"name": "Julian Alvarez"}, "location": [105, 40],
    "shot": {"outcome": {"name": "Goal"}},
}


class MetricTests(unittest.TestCase):
    def test_is_goal_event(self):
        self.assertTrue(metrics.is_goal_event(GOAL_EVENT))
        self.assertFalse(metrics.is_goal_event(PASS_EVENT))
        self.assertTrue(metrics.is_goal_event({"type": {"name": "Own Goal For"}}))

    def test_claims_goal(self):
        self.assertTrue(metrics.claims_goal("GOAL! He scores!"))
        self.assertFalse(metrics.claims_goal("A neat pass into midfield."))

    def test_match_vocab(self):
        vocab = metrics.match_vocab([PASS_EVENT])
        self.assertIn("argentina", vocab)
        self.assertIn("messi", vocab)
        self.assertIn("alvarez", vocab)        # recipient tokens are included

    def test_unknown_names_flags_out_of_vocab(self):
        vocab = {"messi", "argentina"}
        self.assertEqual(metrics.unknown_names("What a strike from Ronaldo", vocab), ["Ronaldo"])
        self.assertEqual(metrics.unknown_names("What a strike from Messi", {"messi"}), [])

    def test_line_violations(self):
        vocab = metrics.match_vocab([PASS_EVENT, GOAL_EVENT])
        self.assertEqual(metrics.line_violations("Messi slides it to Alvarez", PASS_EVENT, vocab), [])
        bad = metrics.line_violations("What a goal from Ronaldo", PASS_EVENT, vocab)
        self.assertIn("invented_goal", bad)
        self.assertIn("unknown_name:Ronaldo", bad)

    def test_faithfulness_rate(self):
        vocab = {"messi", "argentina"}
        records = [
            ("Messi drives forward", PASS_EVENT),       # clean
            ("What a goal from Ronaldo", PASS_EVENT),   # invented goal + unknown name
        ]
        result = metrics.faithfulness(records, vocab)
        self.assertEqual(result["rate"], 0.5)
        self.assertEqual(result["total"], 2)
        self.assertEqual(len(result["violations"]), 1)

    def test_latency_stats(self):
        stats = metrics.latency_stats([10.0, 20.0, 30.0, 40.0])
        self.assertEqual(stats["count"], 4)
        self.assertEqual(stats["mean_ms"], 25.0)
        self.assertEqual(stats["median_ms"], 25.0)
        self.assertGreaterEqual(stats["p95_ms"], 30.0)
        self.assertEqual(metrics.latency_stats([])["count"], 0)


class HarnessTests(unittest.TestCase):
    def test_evaluate_offline_match(self):
        report = evaluate([PASS_EVENT, GOAL_EVENT], language="en", mock=True,
                          languages=("en", "es", "fr"))
        self.assertGreaterEqual(report.n_lines, 1)
        self.assertEqual(report.faithfulness["rate"], 1.0)   # mock lines are faithful by construction
        self.assertEqual(report.latency["count"], report.n_lines)
        self.assertEqual(set(report.coverage), {"en", "es", "fr"})
        self.assertTrue(all(report.coverage.values()))       # every language emits commentary


if __name__ == "__main__":
    unittest.main(verbosity=2)
