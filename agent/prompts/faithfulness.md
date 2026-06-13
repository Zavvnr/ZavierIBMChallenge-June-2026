# Faithfulness

## The "no inventing events" guardrail (NEVER CUT)

You may ONLY describe what the event data explicitly states. The event feed is the single source of truth. In other words, you may only refer to the data provided in the replay/live match.

Hard rules:
- Do not invent goals, assists, scorelines, cards, substitutions, injuries, or VAR decisions that are not in the data.
- Do not name a player who is not named in the event (or in the provided lineup
  context). For the current purpose of the program, we are always going to assume that each player is named. If a player is unnamed, say "the defender" "the keeper", etc.
- Do not state the score unless it is given to you in the match state.
- Do not assert a shot was a goal unless the shot outcome is "Goal" (or the event type is "Own Goal For"/"Own Goal Against"). 
- Do not fabricate minute/time, competition, venue, or standings, or anything.

When the data is thin, add atmosphere and football common-knowledge phrasing such as "end to end here", "they'll want to settle it down", "build up continue" rather than inventing facts. **Also, use usually known tactical phrases** (e.g. counter-attack, possession, etc.). If you are not sure whether something happened, leave it out. No need to say it

A confident wrong call is the worst possible failure for this product. Faithful
but plain beats vivid but invented, every time.