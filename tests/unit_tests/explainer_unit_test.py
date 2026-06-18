"""Unit tests for agent.explainer (the third-commentator Q&A engine). Offline."""
import types
import unittest

from agent import explainer


def _fake_client(content):
    create = lambda **kw: types.SimpleNamespace(
        choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=content))])
    return types.SimpleNamespace(chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create)))


class BuildPromptTests(unittest.TestCase):
    def test_includes_question_event_and_laws(self):
        ev = {"minute": 80, "second": 5, "type": {"name": "Shot"}, "team": {"name": "Argentina"},
              "shot": {"outcome": {"name": "Disallowed"}}}
        p = explainer.build_explainer_prompt("why disallowed?", event=ev,
                                             laws=[{"text": "Law 11 — Offside"}], state={"score": "3-3"})
        self.assertIn("why disallowed?", p)
        self.assertIn("Shot", p)
        self.assertIn("Law 11 — Offside", p)
        self.assertIn("3-3", p)

    def test_no_laws_note(self):
        p = explainer.build_explainer_prompt("q", event=None, laws=[])
        self.assertIn("none retrieved", p.lower())


class ExplainTests(unittest.TestCase):
    def test_grounds_in_retrieved_laws(self):
        seen = {}
        def retriever(q, k=4):
            seen["q"], seen["k"] = q, k
            return [{"text": "Law 11 — Offside", "score": 0.9}]
        out = explainer.explain("why was the goal disallowed?",
                                 event={"type": {"name": "Shot"}}, language="en",
                                 client=_fake_client("Offside under Law 11."), retriever=retriever)
        self.assertEqual(out, "Offside under Law 11.")
        self.assertEqual(seen["q"], "why was the goal disallowed?")

    def test_degrades_when_retrieval_fails(self):
        def boom(q, k=4):
            raise RuntimeError("no index built")
        out = explainer.explain("explain that", client=_fake_client("From the event: a shot."),
                                 retriever=boom)
        self.assertEqual(out, "From the event: a shot.")  # still answered despite retrieval failure


if __name__ == "__main__":
    unittest.main(verbosity=2)
