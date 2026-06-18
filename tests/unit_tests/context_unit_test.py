"""Unit tests for the Laws knowledge base (context.retrieve + context.ingest_laws). Offline."""
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from context import retrieve as rtv
from context import ingest_laws as ing


class FakeEmbedClient:
    """Stands in for the Granite embeddings client; returns fixed vectors."""
    def __init__(self, vectors):
        data = [types.SimpleNamespace(embedding=v) for v in vectors]
        self.embeddings = types.SimpleNamespace(create=lambda model, input: types.SimpleNamespace(data=data))


class EmbedTextsTests(unittest.TestCase):
    def test_embed_texts_returns_vectors(self):
        with mock.patch("agent.granite_client.build_granite_client",
                        return_value=FakeEmbedClient([[1.0, 0.0], [0.0, 1.0]])), \
             mock.patch("agent.granite_client.embed_model_id", return_value="granite-embed"):
            self.assertEqual(rtv.embed_texts(["a", "b"]), [[1.0, 0.0], [0.0, 1.0]])


class RetrieveTests(unittest.TestCase):
    def _write_index(self, tmp, chunks, vectors):
        (Path(tmp) / "laws_chunks.json").write_text(json.dumps({"chunks": chunks}), encoding="utf-8")
        np.save(Path(tmp) / "laws_vectors.npy", np.asarray(vectors, dtype="float32"))

    def test_returns_most_similar_chunk(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._write_index(tmp, ["offside rule", "throw-in rule", "penalty rule"],
                              [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
            with mock.patch.object(rtv, "embed_texts", return_value=[[0.9, 0.1, 0.0]]):
                out = rtv.retrieve("when is a player offside?", k=1, index_dir=Path(tmp))
        self.assertEqual(out[0]["text"], "offside rule")
        self.assertGreater(out[0]["score"], 0.5)

    def test_respects_k(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._write_index(tmp, ["a", "b", "c"], [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
            with mock.patch.object(rtv, "embed_texts", return_value=[[1.0, 0.0, 0.0]]):
                out = rtv.retrieve("q", k=2, index_dir=Path(tmp))
        self.assertEqual(len(out), 2)

    def test_no_index_returns_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(rtv.retrieve("q", index_dir=Path(tmp) / "missing"), [])


class BuildIndexTests(unittest.TestCase):
    def test_build_index_writes_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            pdf = Path(tmp) / "laws.pdf"
            pdf.write_text("x", encoding="utf-8")
            with mock.patch.object(ing, "parse_chunks", return_value=["c1", "c2"]), \
                 mock.patch.object(ing, "embed_texts", return_value=[[1.0, 0.0], [0.0, 1.0]]), \
                 mock.patch("agent.granite_client.embed_model_id", return_value="granite-embed"):
                out = ing.build_index(pdf_path=pdf, index_dir=Path(tmp) / "index")
            self.assertTrue((out / "laws_chunks.json").exists())
            self.assertTrue((out / "laws_vectors.npy").exists())
            data = json.loads((out / "laws_chunks.json").read_text(encoding="utf-8"))
            self.assertEqual(data["chunks"], ["c1", "c2"])
            self.assertEqual(data["dims"], 2)

    def test_build_index_missing_pdf_raises(self):
        with self.assertRaises(SystemExit):
            ing.build_index(pdf_path=Path("/no/such/laws.pdf"))

    def test_round_trip_build_then_retrieve(self):
        with tempfile.TemporaryDirectory() as tmp:
            pdf = Path(tmp) / "laws.pdf"
            pdf.write_text("x", encoding="utf-8")
            idx = Path(tmp) / "index"
            with mock.patch.object(ing, "parse_chunks", return_value=["offside", "throw-in"]), \
                 mock.patch.object(ing, "embed_texts", return_value=[[1.0, 0.0], [0.0, 1.0]]), \
                 mock.patch("agent.granite_client.embed_model_id", return_value="granite-embed"):
                ing.build_index(pdf_path=pdf, index_dir=idx)
            with mock.patch.object(rtv, "embed_texts", return_value=[[0.95, 0.05]]):
                out = rtv.retrieve("offside?", k=1, index_dir=idx)
        self.assertEqual(out[0]["text"], "offside")


if __name__ == "__main__":
    unittest.main(verbosity=2)
