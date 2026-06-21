"""Unit tests for data_extraction.lineups. Offline: synthetic StatsBomb JSON, no network."""
import unittest

from data_extraction import lineups as L

LINEUPS = [
    {"team_id": 1, "team_name": "Argentina", "lineup": [
        {"player_id": 10, "player_name": "Lionel Andrés Messi", "player_nickname": "Lionel Messi",
         "jersey_number": 10, "positions": [{"position": "Center Forward"}], "captain": True},
        {"player_id": 2, "player_name": "Nahuel Molina", "jersey_number": 26,
         "positions": [{"position": "Right Back"}]},
        {"player_id": 99, "player_name": "Bench Player", "jersey_number": 7, "positions": []},
    ]},
    {"team_id": 2, "team_name": "France", "lineup": [
        {"player_id": 30, "player_name": "Kylian Mbappé", "jersey_number": 10,
         "positions": [{"position": "Left Wing"}]},
    ]},
]
EVENTS = [
    {"type": {"name": "Starting XI"}, "team": {"name": "Argentina"}, "tactics": {
        "formation": 442, "lineup": [
            {"jersey_number": 26, "player": {"id": 2, "name": "Nahuel Molina"},
             "position": {"id": 2, "name": "Right Back"}},
            {"jersey_number": 10, "player": {"id": 10, "name": "Lionel Messi"},
             "position": {"id": 23, "name": "Center Forward"}},
        ]}},
]
META = {"home_team": {"home_team_name": "Argentina", "managers": [{"name": "Lionel Scaloni"}]},
        "away_team": {"away_team_name": "France", "managers": [{"name": "Didier Deschamps"}]}}


class FormationTests(unittest.TestCase):
    def test_formation_lines(self):
        self.assertEqual(L.formation_lines(442), [4, 4, 2])
        self.assertEqual(L.formation_lines("4-2-3-1"), [4, 2, 3, 1])
        self.assertEqual(L.formation_lines(None), [4, 4, 2])   # missing -> fallback
        self.assertEqual(L.formation_lines(999), [4, 4, 2])    # doesn't sum to 10 -> fallback


class ParseTests(unittest.TestCase):
    def setUp(self):
        self.arg = L.parse_team_lineup("Argentina", LINEUPS, EVENTS, META)

    def test_formation_and_manager(self):
        self.assertEqual(self.arg.formation, "442")
        self.assertEqual(self.arg.manager, "Lionel Scaloni")

    def test_starting_xi_and_captain(self):
        self.assertEqual(len(self.arg.starting_xi), 2)
        messi = next(p for p in self.arg.starting_xi if p.name == "Lionel Messi")  # nickname preferred
        self.assertTrue(messi.is_captain)
        self.assertEqual(messi.number, 10)
        self.assertEqual(self.arg.starting_xi[0].position, "Right Back")  # sorted by position id

    def test_substitutes(self):
        self.assertEqual([p.name for p in self.arg.substitutes], ["Bench Player"])

    def test_parse_both_teams(self):
        teams = L.parse_lineups(LINEUPS, EVENTS, META)
        self.assertEqual([t.team for t in teams], ["Argentina", "France"])
        france = teams[1]
        self.assertEqual(france.formation, "")          # no Starting XI event for France
        self.assertEqual(france.manager, "Didier Deschamps")


class SvgAndLabelTests(unittest.TestCase):
    def test_svg_contains_players(self):
        svg = L.formation_svg(L.parse_team_lineup("Argentina", LINEUPS, EVENTS, META), "en")
        self.assertTrue(svg.startswith("<svg"))
        self.assertIn("Messi", svg)
        self.assertIn("(C)", svg)                       # captain marker
        self.assertIn("442", svg)

    def test_labels_localised(self):
        self.assertEqual(L.labels_for("ja")["manager"], "監督")
        self.assertEqual(L.labels_for("es-ES")["captain"], "Capitán")
        self.assertEqual(L.labels_for("zz")["manager"], "Manager")   # fallback to English


if __name__ == "__main__":
    unittest.main(verbosity=2)
