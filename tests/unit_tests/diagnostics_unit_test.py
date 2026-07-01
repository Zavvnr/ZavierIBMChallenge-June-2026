"""Unit tests for the diagnostics 'doctor'. Fully offline: the Granite client is a fake,
the network is mocked, and file checks use temp dirs. No LM Studio / TTS / StatsBomb calls.
"""
import os
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from diagnostics import (
    CheckResult, _probe_openai, check_granite, check_laws_index, check_statsbomb,
    check_tts_key, check_vision_events, overall_status, run_all,
)


class FakeGranite:
    """A stand-in for the OpenAI-compatible client: lists the model ids it's given."""

    def __init__(self, ids):
        page = types.SimpleNamespace(data=[types.SimpleNamespace(id=i) for i in ids])
        self.models = types.SimpleNamespace(list=lambda: page)

    def with_options(self, **_kwargs):
        return self


class GraniteCheckTests(unittest.TestCase):
    def setUp(self):
        self._p1 = mock.patch("diagnostics.model_id", return_value="chat-x")
        self._p2 = mock.patch("diagnostics.embed_model_id", return_value="embed-y")
        self._p1.start(); self._p2.start()

    def tearDown(self):
        self._p1.stop(); self._p2.stop()

    def test_both_models_loaded(self):
        res = check_granite(FakeGranite(["chat-x", "embed-y"]))
        self.assertEqual([r.status for r in res], ["ok", "ok"])
        self.assertTrue(res[0].required)            # chat is required for real commentary

    def test_loose_match_handles_quant_and_namespace(self):
        res = check_granite(FakeGranite(["ibm/chat-x-GGUF", "embed-y-Q8_0"]))
        self.assertEqual([r.status for r in res], ["ok", "ok"])

    def test_chat_loaded_embed_missing(self):
        res = check_granite(FakeGranite(["chat-x"]))
        self.assertEqual(res[0].status, "ok")
        self.assertEqual(res[1].status, "warn")     # embeddings only degrade the explainer

    def test_no_matching_model_fails_chat(self):
        res = check_granite(FakeGranite(["llama-3-8b"]))
        self.assertEqual(res[0].status, "fail")
        self.assertTrue(res[0].required)

    def test_server_unreachable_is_fail(self):
        class _Raising:
            def list(self): raise ConnectionError("refused")

        class Boom:
            models = _Raising()
            def with_options(self, **_k): return self

        res = check_granite(Boom())
        self.assertEqual(res[0].status, "fail")

    def test_client_build_failure_is_handled(self):
        with mock.patch("diagnostics._probe_openai", return_value=(True, "")), \
             mock.patch("diagnostics.build_granite_client", side_effect=SystemExit("no url")):
            res = check_granite()
        self.assertEqual(res[0].status, "fail")
        self.assertIn("client unavailable", res[0].detail)

    def test_broken_openai_reports_real_cause(self):
        broken = (False, "openai present but import failed (ImportError: DLL load failed) "
                         "— likely a corrupted install")
        with mock.patch("diagnostics._probe_openai", return_value=broken):
            res = check_granite()                       # client=None -> probe runs first
        self.assertEqual(res[0].status, "fail")
        self.assertTrue(res[0].required)
        self.assertIn("import failed", res[0].detail)


class OpenAiProbeTests(unittest.TestCase):
    def test_missing(self):
        with mock.patch("importlib.util.find_spec", return_value=None):
            importable, detail = _probe_openai()
        self.assertFalse(importable)
        self.assertIn("not installed", detail)

    def test_present_but_broken_surfaces_real_error(self):
        with mock.patch("importlib.util.find_spec", return_value=object()), \
             mock.patch("importlib.import_module", side_effect=ImportError("DLL load failed")):
            importable, detail = _probe_openai()
        self.assertFalse(importable)
        self.assertIn("import failed", detail)
        self.assertIn("corrupted", detail)

    def test_importable(self):
        with mock.patch("importlib.util.find_spec", return_value=object()), \
             mock.patch("importlib.import_module", return_value=object()):
            importable, detail = _probe_openai()
        self.assertTrue(importable)
        self.assertEqual(detail, "")


class TtsKeyTests(unittest.TestCase):
    def test_present(self):
        with mock.patch.dict(os.environ, {"GOOGLE_TTS_API_KEY": "x"}):
            self.assertEqual(check_tts_key().status, "ok")

    def test_absent(self):
        env = {k: v for k, v in os.environ.items() if k != "GOOGLE_TTS_API_KEY"}
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertEqual(check_tts_key().status, "warn")

    def test_secret_value_never_leaks_into_detail(self):
        with mock.patch.dict(os.environ, {"GOOGLE_TTS_API_KEY": "super-secret-123"}):
            self.assertNotIn("super-secret-123", check_tts_key().detail)


class FileCheckTests(unittest.TestCase):
    def test_laws_index_built(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            (p / "laws_chunks.json").write_text("[]", encoding="utf-8")
            (p / "laws_vectors.npy").write_bytes(b"\x00")
            self.assertEqual(check_laws_index(p).status, "ok")

    def test_laws_index_missing(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(check_laws_index(Path(d)).status, "warn")

    def test_vision_events_present(self):
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "events.json"
            f.write_text("[]", encoding="utf-8")
            self.assertEqual(check_vision_events(f).status, "ok")

    def test_vision_events_absent(self):
        self.assertEqual(check_vision_events(Path("/no/such/events.json")).status, "warn")


class StatsbombTests(unittest.TestCase):
    def test_reachable(self):
        import requests
        resp = types.SimpleNamespace(status_code=200, close=lambda: None)
        with mock.patch.object(requests, "get", return_value=resp):
            self.assertEqual(check_statsbomb().status, "ok")

    def test_unreachable_degrades_to_warn(self):
        import requests
        with mock.patch.object(requests, "get", side_effect=OSError("no network")):
            self.assertEqual(check_statsbomb().status, "warn")


class AggregateTests(unittest.TestCase):
    def test_overall_status_picks_worst(self):
        self.assertEqual(overall_status([CheckResult("a", "ok")]), "ok")
        self.assertEqual(overall_status([CheckResult("a", "ok"), CheckResult("b", "warn")]), "warn")
        self.assertEqual(overall_status([CheckResult("a", "warn"), CheckResult("b", "fail")]), "fail")

    def test_run_all_is_offline_safe_and_covers_every_check(self):
        import requests
        with mock.patch("diagnostics.model_id", return_value="m"), \
             mock.patch("diagnostics.embed_model_id", return_value="e"), \
             mock.patch.object(requests, "get", side_effect=OSError("offline")):
            results = run_all(FakeGranite(["m", "e"]))
        names = {r.name for r in results}
        self.assertTrue(
            {"Granite chat", "Granite embeddings", "Google TTS key",
             "StatsBomb data", "Laws index", "Vision clip"} <= names)


class HealthEndpointTests(unittest.TestCase):
    def test_health_returns_aggregated_json(self):
        from web.app import create_app
        canned = [CheckResult("Granite chat", "fail", "down", required=True),
                  CheckResult("Vision clip", "ok", "present")]
        with mock.patch("diagnostics.run_all", return_value=canned):
            r = create_app().test_client().get("/api/health")
        self.assertEqual(r.status_code, 200)
        data = r.get_json()
        self.assertEqual(data["status"], "fail")
        self.assertIn("Granite chat", {c["name"] for c in data["checks"]})


if __name__ == "__main__":
    unittest.main(verbosity=2)
