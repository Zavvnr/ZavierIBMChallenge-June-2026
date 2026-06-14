"""
Unit tests for the agent directory, focused on the granite_client and mcp_client modules.

Run from the repo root:
    python -m unittest tests.unit_test -v
    # or: python -m pytest tests/unit_test.py

These cover the Granite config resolution, OpenAI-compatible
client construction, and the MongoDB context client's embedding + fail-safe paths,
including the SystemExit-degradation fix. They need no live Granite endpoint and no
MongoDB. Agent/crew/dead_air coverage can be added once agent.prompts exposes its API.
"""
import os
import unittest

from agent import granite_client as gc
from agent.mcp_client import (
    NoOpContextClient, MongoMCPContextClient, build_context_client,
    build_context_query, _format_context,
)

try:
    import openai  # noqa: F401
    _HAS_OPENAI = True
except ImportError:
    _HAS_OPENAI = False


def _clear_env(*names):
    return {n: os.environ.pop(n, None) for n in names}

def _restore_env(saved):
    for n, v in saved.items():
        if v is None:
            os.environ.pop(n, None)
        else:
            os.environ[n] = v


class GraniteClientConfigTests(unittest.TestCase):
    def setUp(self):
        self._saved = _clear_env("GRANITE_MODEL_ID", "GRANITE_EMBED_MODEL",
                                 "GRANITE_BASE_URL", "GRANITE_API_KEY")

    def tearDown(self):
        _restore_env(self._saved)

    def test_model_id_default(self):
        self.assertEqual(gc.model_id(), "granite-4-h-tiny")

    def test_model_id_env_override(self):
        os.environ["GRANITE_MODEL_ID"] = "granite-test"
        self.assertEqual(gc.model_id(), "granite-test")

    def test_embed_model_default(self):
        self.assertEqual(gc.embed_model_id(), "granite-embedding-278m-multilingual")

    def test_embed_model_env_override(self):
        os.environ["GRANITE_EMBED_MODEL"] = "embed-test"
        self.assertEqual(gc.embed_model_id(), "embed-test")

    def test_build_client_requires_base_url(self):
        with self.assertRaises(SystemExit):
            gc.build_granite_client()

    @unittest.skipUnless(_HAS_OPENAI, "openai not installed")
    def test_build_client_constructs_with_default_key(self):
        os.environ["GRANITE_BASE_URL"] = "http://localhost:1234/v1"
        client = gc.build_granite_client()
        self.assertIn("localhost:1234/v1", str(client.base_url))
        self.assertEqual(client.api_key, "lm-studio")  # LM Studio placeholder default

    @unittest.skipUnless(_HAS_OPENAI, "openai not installed")
    def test_build_client_uses_env_api_key(self):
        os.environ["GRANITE_BASE_URL"] = "http://localhost:1234/v1"
        os.environ["GRANITE_API_KEY"] = "secret-key"
        self.assertEqual(gc.build_granite_client().api_key, "secret-key")


class McpQueryAndFormatTests(unittest.TestCase):
    def test_noop_returns_empty(self):
        self.assertEqual(NoOpContextClient().fetch_event_context({"type": {"name": "Shot"}}), {})

    def test_build_context_client_toggle(self):
        self.assertIsInstance(build_context_client(False), NoOpContextClient)
        self.assertIsInstance(build_context_client(True), MongoMCPContextClient)

    def test_build_context_query_shot(self):
        ev = {"type": {"name": "Shot"}, "team": {"name": "Argentina"},
              "player": {"name": "Lionel Messi"},
              "shot": {"outcome": {"name": "Goal"}, "body_part": {"name": "Left Foot"}}}
        q = build_context_query(ev, {"score": "1-0", "clock": "23:11"})
        for token in ("23:11", "1-0", "Shot", "Argentina", "Lionel Messi", "Goal", "Left Foot"):
            self.assertIn(token, q)

    def test_build_context_query_empty(self):
        self.assertEqual(build_context_query({}, {}), "")

    def test_format_context_groups(self):
        docs = [
            {"kind": "player", "name": "Messi", "text": "Captain."},
            {"kind": "team", "name": "Argentina", "text": "2022 champions."},
            {"kind": "term", "name": "Offside", "text": "Law 11."},
        ]
        out = _format_context(docs)
        self.assertEqual(out["players"], ["Messi: Captain."])
        self.assertEqual(out["teams"], ["Argentina: 2022 champions."])
        self.assertEqual(out["glossary"], ["Offside: Law 11."])

    def test_format_context_empty(self):
        self.assertEqual(_format_context([]), {})


class FakeCursor(list):
    def limit(self, n):
        return FakeCursor(self[:n])

class FakeCollection:
    """Minimal stand-in for a pymongo collection."""
    def __init__(self, vector_docs=None, text_docs=None):
        self._vector_docs = vector_docs or []
        self._text_docs = text_docs or []
        self.aggregate_called = False
    def aggregate(self, pipeline):
        self.aggregate_called = True
        return list(self._vector_docs)
    def find(self, *a, **k):
        return FakeCursor(self._text_docs)


class McpRetrievalTests(unittest.TestCase):
    def setUp(self):
        self._saved = _clear_env("MONGODB_URI", "GRANITE_BASE_URL", "GRANITE_EMBED_MODEL")

    def tearDown(self):
        _restore_env(self._saved)

    def test_embed_model_filled_from_env(self):
        self.assertEqual(MongoMCPContextClient().embed_model, "granite-embedding-278m-multilingual")
        self.assertEqual(MongoMCPContextClient(embed_model="custom").embed_model, "custom")

    def test_not_configured_without_uri_or_handle(self):
        self.assertFalse(MongoMCPContextClient().is_configured())

    def test_fetch_returns_empty_without_collection(self):
        ev = {"type": {"name": "Shot"}, "team": {"name": "Argentina"}, "player": {"name": "Messi"}}
        self.assertEqual(MongoMCPContextClient().fetch_event_context(ev, {"score": "1-0"}), {})

    def test_injected_embedder_drives_vector_search(self):
        col = FakeCollection(vector_docs=[{"kind": "player", "name": "Messi", "text": "Captain."}])
        c = MongoMCPContextClient(collection_handle=col, embedder=lambda q: [0.1] * 768)
        ev = {"type": {"name": "Shot"}, "team": {"name": "Argentina"}, "player": {"name": "Messi"}}
        out = c.fetch_event_context(ev, {"score": "1-0"})
        self.assertTrue(col.aggregate_called)
        self.assertEqual(out, {"players": ["Messi: Captain."]})

    def test_embed_failure_degrades_to_text_search(self):
        # Mongo configured, but Granite unavailable -> build_granite_client raises
        # SystemExit; _embed must catch it (the fix) and fall back to keyword search.
        col = FakeCollection(text_docs=[{"kind": "player", "name": "Messi", "text": "Captain."}])
        c = MongoMCPContextClient(collection_handle=col)  # no embedder, no GRANITE_BASE_URL
        self.assertIsNone(c._embed("messi goal"))          # must NOT raise SystemExit
        ev = {"type": {"name": "Shot"}, "team": {"name": "Argentina"}, "player": {"name": "Messi"}}
        out = c.fetch_event_context(ev, {"score": "1-0"})
        self.assertEqual(out, {"players": ["Messi: Captain."]})
        self.assertFalse(col.aggregate_called)             # vector search skipped when embed is None


if __name__ == "__main__":
    unittest.main(verbosity=2)
