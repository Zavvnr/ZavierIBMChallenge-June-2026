"""
google_text_to_speech — commentary text -> audio.

Granite writes the words; this package only voices them (Granite is a text model
and does not synthesize speech). Synthesis uses the Google Cloud Text-to-Speech
REST API authenticated with GOOGLE_TTS_API_KEY. Every path is FAIL-SAFE: no key,
a network error, or a bad voice degrades to a text-only result (with
``skipped_reason`` set) rather than raising, so the live commentary loop never
breaks because audio failed.

Modules
-------
speak.py
    Single-speaker synthesis. ``NoOpSpeaker`` (default, no audio) and
    ``GoogleCloudSpeaker`` (real mp3 via REST). ``build_speaker(enabled, provider)``
    chooses between them; ``speaking_rate`` lets the agent map event intensity to
    delivery tempo. CLI::

        python -m google_text_to_speech.speak --text "Goal!" --language es-ES

mutilingual_speaker.py
    Two-voice (lead + analyst) dialogue rendered SEQUENTIALLY with a distinct voice
    per role (overlapping synthetic voices are unintelligible — it's the lead's
    call, THEN the analyst). Delegates each turn to GoogleCloudSpeaker. Factory:
    ``build_multispeaker_speaker(language=...)``.
"""
