"""
data_extraction — StatsBomb Open Data ingestion.

Downloads and caches a single StatsBomb match (events + lineups + best-effort
metadata) under the repo-root ``data/cache/<match_id>/`` so the rest of the
pipeline (data_replayer, agent) can run repeatedly against a real match without
re-hitting the network. The full nested event JSON is preserved on purpose — that
is the structure the commentary agent reads.

Modules
-------
loader.py
    Fetch + cache a match, and discover what is available. Key functions:
    ``cache_match`` / ``load_cached_events`` / ``load_cached_meta`` plus the
    discovery helpers ``list_competitions`` / ``list_matches``. CLI::

        python -m data_extraction.loader --list-competitions
        python -m data_extraction.loader --list-matches --competition-id 43 --season-id 106
        python -m data_extraction.loader --match-id 3869685   # World Cup 2022 Final (default demo)

Attribution: match data is provided by StatsBomb Open Data
(https://github.com/statsbomb/open-data); their public data user agreement
requires crediting StatsBomb as the source for any derived output.
"""
