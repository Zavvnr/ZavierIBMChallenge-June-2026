"""
web — the Flask UI for MATE.

Thin glue over data_pipeline.stream_commentary: pick a language + match, then watch
commentary stream in near-real-time with optional synced audio. All the real work
(replay -> context -> Granite agent -> TTS) lives in the modules it calls; this
package only serves the page and exposes a few JSON/SSE endpoints.

Modules
-------
app.py
    ``create_app()`` builds the Flask app; module-level ``app`` is the WSGI entry
    point (e.g. ``gunicorn web.app:app``). Endpoints:

        GET /                  single-page UI (web/static/index.html)
        GET /api/languages     [{code, name}] for the language picker
        GET /api/matches       [{id, label}] cached matches (plus "sample")
        GET /api/stream        Server-Sent Events: one commentary line per event
        GET /api/audio/<file>  serves a generated mp3 from the TTS step
        GET /api/tts           synthesize one line on demand (audio/mpeg)
        GET /api/ask           on-demand explainer answer (RAG over the Laws) via Granite
        GET /api/lineup        line-ups + formation diagram (StatsBomb)
        GET /api/profile       grounded player profile (Wikipedia + Granite)

    Run locally:  ``python -m web.app``  (serves http://localhost:8080).
    ``.env`` is loaded only in ``main()``; Cloud Run injects real env vars directly.

static/
    index.html (markup + behaviour) and style.css (formatting, split out so styling
    can be tweaked without touching the page).
"""
