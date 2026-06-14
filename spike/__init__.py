"""
spike — the throwaway go/no-go check.

Answers ONE question before any pipeline is built: "If we hand IBM Granite ~15-20
real match events, is the commentary actually good?" Deliberately minimal — no
replayer, no agent, no MCP, no TTS. Events in -> one Granite call -> commentary
out -> a human eyeballs it against a short checklist.

Modules
-------
go_no_go.py
    Selects a dense window of events, formats one faithful line each, builds a
    "no inventing" prompt, and calls Granite (or just prints the prompt with
    ``--mock``). CLI::

        python spike/go_no_go.py --mock                       # offline: print the prompt
        python spike/go_no_go.py --language es                # Spanish, real Granite call
        python spike/go_no_go.py --match-id 3869685 --start 1500 --count 18

    A real match needs caching first: ``python -m data_extraction.loader --match-id <id>``.

Note: ``spike/sample_events.json`` is an ILLUSTRATIVE, StatsBomb-shaped sample so
the spike runs with no download (player/team ids are not canonical). For a real
go/no-go, point ``--match-id`` at a downloaded match.
"""
