"""Pure scoring functions for the MATE evaluation harness — no I/O, no model calls.

Faithfulness here is a deliberately simple, conservative heuristic suitable for an MVP:
flag a line that claims a goal on a non-goal event, or that uses a Title-Case name not
present in the match's own vocabulary. It biases toward precision (few false alarms) — it
can miss a hallucinated name that opens a sentence — so a 100% score means "no detected
violations", not a proof of perfection. See README for the metric's limits.
"""
from __future__ import annotations

import re
from statistics import mean, median
from typing import Iterable, Sequence

# Words that assert a goal was scored.
GOAL_WORDS = ("goal", "scores", "scored", "nets ", "into the net", "back of the net")

# Title-Case words that are never player/team names — kept out of the unknown-name check.
_STOP = {
    "The", "A", "An", "And", "But", "Or", "So", "Now", "Here", "There", "What", "That",
    "This", "Goal", "Save", "Saved", "Yes", "Oh", "Well", "Up", "In", "On", "Out", "He",
    "She", "It", "They", "We", "You", "Welcome", "Let", "Off", "Penalty", "Corner",
    "Free", "Kick", "Shot", "Pass", "Cross", "Foul", "Offside", "Half", "Full", "Time",
    "Substitution", "Card", "Yellow", "Red", "Booking", "Goalkeeper", "Keeper",
}
_STOP_LOWER = {w.lower() for w in _STOP}  # compared case-insensitively (e.g. emphatic "GOAL")


def _name_tokens(text: str) -> set:
    """Lowercased word tokens from a name (e.g. 'Lionel Messi' -> {'lionel', 'messi'})."""
    return {t for t in re.findall(r"[a-zà-ÿ]+", text.lower()) if t}


def match_vocab(events: Iterable[dict]) -> set:
    """All team/player name terms seen in a match, lowercased (full names + tokens)."""
    vocab: set = set()
    for ev in events:
        etype = (ev.get("type") or {}).get("name") or ""
        if etype:
            vocab |= _name_tokens(etype)  # football terms (Pass, Foul Committed, ...) are legit vocab
        for key in ("team", "player"):
            name = (ev.get(key) or {}).get("name") or ""
            if name:
                vocab.add(name.lower())
                vocab |= _name_tokens(name)
        recipient = ((ev.get("pass") or {}).get("recipient") or {}).get("name") or ""
        if recipient:
            vocab.add(recipient.lower())
            vocab |= _name_tokens(recipient)
    return vocab


def is_goal_event(event: dict) -> bool:
    """True if the event is an actual goal (a shot finished as Goal, or an own goal)."""
    etype = (event.get("type") or {}).get("name", "")
    if etype == "Shot":
        return ((event.get("shot") or {}).get("outcome") or {}).get("name") == "Goal"
    return etype in ("Own Goal For", "Own Goal Against")


def claims_goal(line: str) -> bool:
    """True if a commentary line asserts that a goal was scored."""
    low = line.lower()
    return any(word in low for word in GOAL_WORDS)


def unknown_names(line: str, vocab: set) -> list:
    """Title-Case tokens that look like names but aren't in the match vocab (heuristic).

    Skips each sentence's first word (capitalised by grammar) and a stoplist of common
    capitalised words, to keep false positives low.
    """
    found = []
    for sentence in re.split(r"[.!?]\s*", line):
        words = re.findall(r"[A-Za-zÀ-ÿ']+", sentence)
        for i, word in enumerate(words):
            if i == 0:
                continue  # sentence-initial capital is grammar, not necessarily a name
            if word.isupper():
                continue  # ALL-CAPS is emphasis (GOAL!, WHAT A STRIKE), not a name
            if word[:1].isupper() and len(word) >= 3 and word.lower() not in _STOP_LOWER:
                if word.lower() not in vocab:
                    found.append(word)
    return found


def line_violations(line: str, event: dict, vocab: set) -> list:
    """Faithfulness violations for one (line, event): invented goal and/or unknown names."""
    violations = []
    if claims_goal(line) and not is_goal_event(event):
        violations.append("invented_goal")
    violations += [f"unknown_name:{name}" for name in unknown_names(line, vocab)]
    return violations


def faithfulness(records: Sequence[tuple], vocab: set) -> dict:
    """Score [(line, event), ...] -> {rate, total, clean, violations}.

    ``rate`` is the fraction of lines with zero detected faithfulness violations.
    """
    total = len(records)
    violations = []
    clean = 0
    for line, event in records:
        line_v = line_violations(line, event, vocab)
        if line_v:
            violations.append({"line": line, "violations": line_v})
        else:
            clean += 1
    rate = (clean / total) if total else 1.0
    return {"rate": round(rate, 4), "total": total, "clean": clean, "violations": violations}


def latency_stats(latencies_ms: Sequence[float]) -> dict:
    """Summary statistics for per-line latencies, in milliseconds."""
    if not latencies_ms:
        return {"count": 0, "mean_ms": 0.0, "median_ms": 0.0, "p95_ms": 0.0}
    ordered = sorted(latencies_ms)
    p95 = ordered[min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))]
    return {
        "count": len(latencies_ms),
        "mean_ms": round(mean(latencies_ms), 1),
        "median_ms": round(median(latencies_ms), 1),
        "p95_ms": round(p95, 1),
    }
