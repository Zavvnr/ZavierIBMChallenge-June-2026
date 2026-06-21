"""
evaluation — a small, offline-mockable harness for scoring MATE's commentary.

It answers the questions the challenge cares about, with real numbers instead of "TBD":
  - faithfulness: does a line invent a goal, or name a player/team the match never had?
  - language coverage: do all supported languages actually emit commentary?
  - latency: wall-clock time per generated line.

`metrics.py` holds the pure scoring functions (no I/O, no model). `harness.py` runs the
real CommentaryAgent over a match and aggregates the scores; with `mock=True` it needs no
Granite, so it runs in CI and gives a baseline. Run it with:

    python -m evaluation.harness --mock --all-languages          # offline baseline
    python -m evaluation.harness --match-id 3869685 --language es # real Granite (LM Studio)

Faithfulness is a conservative heuristic, not a proof — see metrics.py / README for limits.
"""
