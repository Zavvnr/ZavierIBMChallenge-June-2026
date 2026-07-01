"""Per-event grounding for the two-voice crew: it must be told the real teams and that
the match is already in play, so it never invents a fixture (Barcelona/Real Madrid) or
re-welcomes viewers on every line. Offline — no Granite.
"""
import unittest

from agent.commentary_agent import MatchState
from agent.commentary_crew import LEAD, CommentaryCrew, TurnPlan


class StateExposesTeamsTests(unittest.TestCase):
    def test_as_prompt_dict_includes_team_names(self):
        d = MatchState(home_team="Argentina", away_team="France").as_prompt_dict()
        self.assertEqual(d["home_team"], "Argentina")
        self.assertEqual(d["away_team"], "France")

    def test_agent_state_exposes_supplied_teams(self):
        from agent.commentary_agent import CommentaryAgent
        agent = CommentaryAgent(language="es", mock=True,
                                home_team="Argentina", away_team="France")
        d = agent.state.as_prompt_dict()
        self.assertEqual((d["home_team"], d["away_team"]), ("Argentina", "France"))


class CrewPromptGroundingTests(unittest.TestCase):
    def _prompt(self, state, kind="call", speakers=(LEAD,)):
        crew = CommentaryCrew(language="es-ES", mock=False, generate=lambda p: "")
        return crew.build_prompt({"type": {"name": "Pass"}}, state,
                                 TurnPlan(kind, list(speakers)), {})

    def test_prompt_names_the_real_fixture(self):
        prompt = self._prompt({"home_team": "Argentina", "away_team": "France", "clock": "00:02"})
        self.assertIn("Argentina vs France", prompt)

    def test_prompt_forbids_re_welcoming(self):
        prompt = self._prompt({"home_team": "Argentina", "away_team": "France", "clock": "00:02"})
        self.assertIn("already", prompt.lower())          # "match is ALREADY under way"
        self.assertIn("do not welcome", prompt.lower())

    def test_prompt_has_no_stale_gemini_reference(self):
        prompt = self._prompt({"home_team": "Argentina", "away_team": "France"},
                              kind="goal", speakers=("lead", "analyst"))
        self.assertNotIn("Gemini", prompt)


class EventDetailTests(unittest.TestCase):
    """The crew must see the actual event (player, recipient, zone), not just its type,
    and be told to use fresh wording — otherwise the small model repeats one template."""

    def _rich_event(self):
        return {
            "type": {"name": "Pass"}, "team": {"name": "Argentina"},
            "player": {"name": "Mac Allister"}, "location": [85, 40],
            "pass": {"recipient": {"name": "Messi"}},
        }

    def test_describe_event_names_player_recipient_and_zone(self):
        from agent.commentary_crew import _describe_event
        desc = _describe_event(self._rich_event())
        self.assertIn("Mac Allister", desc)
        self.assertIn("Argentina", desc)
        self.assertIn("Messi", desc)                  # the pass recipient
        self.assertIn("final third", desc)            # x=85 -> attacking third

    def test_prompt_carries_event_specifics_and_forbids_repetition(self):
        crew = CommentaryCrew(language="es-ES", mock=False, generate=lambda p: "")
        state = {"home_team": "Argentina", "away_team": "France", "clock": "00:02"}
        prompt = crew.build_prompt(self._rich_event(), state, TurnPlan("call", [LEAD]), {})
        self.assertIn("Mac Allister", prompt)          # the real event, not just "Pass"
        self.assertIn("Messi", prompt)
        self.assertIn("fresh wording", prompt.lower())
        self.assertIn("recent_lines", prompt.lower())


class RepetitionGuardTests(unittest.TestCase):
    def test_too_similar_catches_near_duplicate(self):
        from agent.commentary_agent import _too_similar
        recent = ["Un pase de Tchouameni a Giroud en el medio del campo"]
        self.assertTrue(_too_similar(
            "Un pase de Tchouameni a Giroud en el medio del campo ahora", recent))
        self.assertFalse(_too_similar("Rabiot recupera el balon en defensa", recent))

    def _agent(self):
        from agent.commentary_agent import CommentaryAgent
        return CommentaryAgent(language="es", mock=False, two_speakers=True)  # dedup runs (live)

    def test_immediate_repeat_is_dropped(self):
        from agent.commentary_crew import LEAD, Turn
        agent = self._agent()
        line = "Un pase de Tchouameni a Giroud en el medio del campo"
        agent._dedupe_turns([Turn(LEAD, line)])                       # spoken once
        kept = [t.text for t in agent._dedupe_turns(
            [Turn(LEAD, line), Turn(LEAD, "Rabiot recupera el balon en defensa")])]
        self.assertNotIn(line, kept)                                  # near-immediate repeat blocked
        self.assertIn("Rabiot recupera el balon en defensa", kept)    # a distinct line is kept

    def test_recurrence_after_a_few_lines_is_allowed(self):
        from agent.commentary_crew import LEAD, Turn
        agent = self._agent()
        style = "Que jugada de futbol increible en el medio del campo"
        agent._dedupe_turns([Turn(LEAD, style)])
        for filler in ("El portero argentino despeja con seguridad total",
                       "Mbappe corre por la banda derecha muy rapido",
                       "Falta dura cometida cerca del area grande ahora"):
            agent._dedupe_turns([Turn(LEAD, filler)])                 # push `style` out of the window
        again = [t.text for t in agent._dedupe_turns([Turn(LEAD, style)])]
        self.assertIn(style, again)                                   # natural recurrence is allowed


class FaithfulnessTests(unittest.TestCase):
    def test_prompt_forbids_inventing_goals(self):
        crew = CommentaryCrew(language="es-ES", mock=False, generate=lambda p: "")
        prompt = crew.build_prompt({"type": {"name": "Foul Committed"}},
                                   {"home_team": "Argentina", "away_team": "France"},
                                   TurnPlan("call", [LEAD]), {})
        self.assertIn("never claim a goal", prompt.lower())


if __name__ == "__main__":
    unittest.main(verbosity=2)
