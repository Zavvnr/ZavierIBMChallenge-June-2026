"""Unit tests for agent.match_briefing + the opening's use of it. Fully offline:
a fake Wikipedia fetcher and a fake Granite client — no network, no LM Studio.
"""
import unittest

from agent import match_briefing


def fake_fetcher(url):
    return {
        "title": "2022 FIFA World Cup Final",
        "extract": ("The 2022 FIFA World Cup final was contested by Argentina and France. "
                    "Argentina won on penalties after a 3-3 draw."),
        "description": "Football match",
        "content_urls": {"desktop": {"page": "https://en.wikipedia.org/wiki/Final"}},
    }


def none_fetcher(url):
    return None


class BriefingForTests(unittest.TestCase):
    def test_curated_demo_match(self):
        brief = match_briefing.briefing_for(match_id="3869685", use_cache=False)
        self.assertIn("Argentina", brief["stakes"])
        self.assertIn("Messi", " ".join(brief["storylines"]))

    def test_wikipedia_fallback_for_uncurated_match(self):
        brief = match_briefing.briefing_for(
            match_id="999", competition="FIFA World Cup", stage="Final", year="2022",
            fetcher=fake_fetcher, use_cache=False)
        self.assertIsNotNone(brief)
        self.assertIn("Argentina", brief["stakes"])
        self.assertEqual(brief["source_url"], "https://en.wikipedia.org/wiki/Final")

    def test_none_when_wikipedia_empty(self):
        brief = match_briefing.briefing_for(
            match_id="999", competition="FIFA World Cup", stage="Final", year="2022",
            fetcher=none_fetcher, use_cache=False)
        self.assertIsNone(brief)

    def test_none_when_no_competition_to_search(self):
        self.assertIsNone(match_briefing.briefing_for(match_id="999", fetcher=fake_fetcher,
                                                      use_cache=False))


class NoteTextTests(unittest.TestCase):
    def test_condenses_stakes_and_two_storylines(self):
        note = match_briefing.note_text({
            "stakes": "Big stakes.",
            "storylines": ["Story one.", "Story two.", "Story three."]})
        self.assertIn("Big stakes.", note)
        self.assertIn("Story one.", note)
        self.assertIn("Story two.", note)
        self.assertNotIn("Story three.", note)          # capped at two storylines

    def test_empty_for_no_briefing(self):
        self.assertEqual(match_briefing.note_text(None), "")


class OpeningBriefingTests(unittest.TestCase):
    """The opening must feed the briefing into the Granite prompt (so it can be grounded)."""

    class _CaptureClient:
        def __init__(self):
            self.prompts = []
            outer = self

            class _Comp:
                def create(self, **kwargs):
                    outer.prompts.append(
                        " ".join(m.get("content", "") for m in kwargs.get("messages", [])))
                    msg = type("M", (), {"content": "Bienvenidos a la final."})()
                    return type("R", (), {"choices": [type("C", (), {"message": msg})()]})()

            self.chat = type("Chat", (), {"completions": _Comp()})()

    def test_opening_injects_briefing_into_prompt(self):
        from agent.commentary_agent import CommentaryAgent
        cap = self._CaptureClient()
        agent = CommentaryAgent(language="es", mock=False, client=cap)
        item = agent.opening(competition="World Cup", home="Argentina", away="France",
                             briefing="Messi's likely last World Cup.")
        self.assertIsNotNone(item)
        self.assertTrue(item.text)
        self.assertTrue(any("Messi's likely last World Cup" in p for p in cap.prompts),
                        "briefing was not passed into the opening prompt")


if __name__ == "__main__":
    unittest.main(verbosity=2)
