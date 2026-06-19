"""Unit tests for the agent.prompts loader (composes the .md fragments). Offline."""
import unittest

from agent import prompts


class LanguageMetaTests(unittest.TestCase):
    def test_language_names_and_codes(self):
        self.assertEqual(prompts.LANGUAGE_NAMES["en"], "English")
        self.assertIn("es", prompts.SUPPORTED_LANGUAGE_CODES)
        self.assertEqual(prompts.SUPPORTED_LANGUAGE_CODES, sorted(prompts.LANGUAGE_NAMES))
        self.assertEqual(len(prompts.SUPPORTED_LANGUAGE_CODES), 12)   # Granite's supported set
        self.assertNotIn("id", prompts.LANGUAGE_NAMES)                # Indonesian is outside the 12

    def test_normalize_language(self):
        self.assertEqual(prompts.normalize_language("es"), "es-ES")    # bare -> locale
        self.assertEqual(prompts.normalize_language("en-GB"), "en-GB")  # locale kept as-is
        self.assertEqual(prompts.normalize_language(None), "en-US")

    def test_display_name(self):
        self.assertEqual(prompts.language_display_name("es-ES"), "Spanish")
        self.assertEqual(prompts.language_display_name("ja"), "Japanese")


class SystemPromptTests(unittest.TestCase):
    def test_composes_fragments_with_language(self):
        sp = prompts.system_prompt("es")
        self.assertIn("Spanish", sp)        # language fragment substituted ({language_name})
        self.assertIn("only", sp.lower())   # faithfulness guardrail wording present
        self.assertGreater(len(sp), 200)    # fragments actually loaded from the .md files

    def test_explainer_prompt_uses_agent_explainer_fragment(self):
        ep = prompts.explainer_system_prompt("en")
        self.assertIn("third", ep.lower())  # agent_explainer.md describes the third agent


class EventPromptTests(unittest.TestCase):
    def test_faithful_and_has_no_comment_escape(self):
        ev = {"type": {"name": "Shot"}, "team": {"name": "Argentina"},
              "player": {"name": "Messi"}, "shot": {"outcome": {"name": "Goal"}}}
        p = prompts.build_event_prompt(ev, {"score": "1-0", "clock": "80:00"})
        self.assertIn("Shot", p)
        self.assertIn("Argentina", p)
        self.assertIn("Goal", p)
        self.assertIn("NO_COMMENT", p)

    def test_includes_context_when_given(self):
        p = prompts.build_event_prompt({"type": {"name": "Pass"}}, {}, {"players": ["Messi: captain"]})
        self.assertIn("CONTEXT", p)


class VoiceCoverageTests(unittest.TestCase):
    def test_every_supported_language_has_a_tts_voice(self):
        from text_to_speech import speak
        for code in prompts.SUPPORTED_LANGUAGE_CODES:
            locale = prompts.normalize_language(code)
            self.assertIn(locale, speak.DEFAULT_VOICES, f"no TTS voice for {code} ({locale})")


if __name__ == "__main__":
    unittest.main(verbosity=2)
