"""
profiles — grounded, multilingual player profiles (the player-explainer feature).

`wiki_client` fetches factual player summaries + photos from Wikipedia's public REST API
(per-language, no API key, cached, fail-safe). `profile_builder` combines that with the
player's StatsBomb position / match involvement and asks Granite to write the profile in
the target language, strictly from the provided facts — no invented clubs or honours.

    python -m profiles.profile_builder --player "Lionel Messi" --language es

Player photos/bios are not in StatsBomb, so this is the project's one external data source
beyond StatsBomb and the Laws PDF. It was added with explicit consent and uses only
public, key-less Wikipedia endpoints (no secrets, no .env changes).
"""
