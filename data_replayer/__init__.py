"""
data_replayer — accelerated event-stream playback.

Replays a cached StatsBomb event stream in accelerated real time so the commentary
agent receives events with realistic pacing (a 3-second build-up takes 3
match-seconds) instead of all at once. It is transport-only: it never decides what
is interesting (that is the agent's job) and never mutates events.

Pacing comes from each event's ``timestamp`` ("HH:MM:SS.mmm") within its period;
the half-time gap is skipped (no sleep when ``period`` changes); and ``sleep`` is
injectable so tests run instantly.

Modules
-------
replayer.py
    ``replay(events, speed=..., sleep=...)`` yields events one at a time with the
    right wait between them. ``speed`` is match-seconds per real second
    (0 = no waiting). CLI::

        python -m data_replayer.replayer --match-id 3869685 --speed 60
"""
