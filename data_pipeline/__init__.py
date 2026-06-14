"""
data_pipeline — wiring the pieces into one stream.

Ties the project together: events enter from data_replayer, the Granite-powered
commentary agent turns them into text (lead + analyst), and a speaker optionally
turns that text into audio. Going live later is a source swap, not a rewrite.

Modules
-------
commentary_pipeline.py
    ``stream_commentary(...)`` -> an iterator of ``CommentaryOutput`` (source event
    + text + optional speech). The agent and speaker are injectable, and TTS
    defaults to a no-op so the pipeline runs credential-free. The CLI emits one
    JSON object per line::

        python -m data_pipeline.commentary_pipeline --sample --mock

live_cv_pipeline.py
    SCAFFOLD for live computer-vision commentary. The hard part of live AI
    commentary is perception (tracking + action recognition + pitch homography);
    by isolating that behind ``VisionEventDetector`` and mapping its output to the
    SAME event-dict schema the replayer emits (``LiveEventAdapter``), going live
    becomes a source swap — everything downstream (agent, pacing, context, TTS) is
    unchanged. The CV pieces are stubs; a Mock* implementation lets the seam run
    end-to-end (fake detections -> real commentary) without any model.
"""
