"""Optional real-model regression against public raw vectors, never a DB assertion."""
import csv
import json
import os
from pathlib import Path
import secrets
import unittest
from unittest.mock import patch

import numpy as np
from fastapi.testclient import TestClient
from api.app import create_app


@unittest.skipUnless(os.getenv("CONTRACTCHECK_TEST_BGE") == "1", "Opt in with cached BGE-M3")
class SavedVectorsTest(unittest.TestCase):
    def test_original_law_and_precedent_vectors(self):
        root = Path(__file__).resolve().parents[2]
        with (root/"law-embedding/data/law_embeddings.jsonl").open(encoding="utf8") as f:
            laws = [json.loads(next(f)) for _ in range(3)]
        path = root/"precedent-search/data/processed"
        with (path/"precedent_chunks_reviewed_with_urls.csv").open(encoding="utf-8-sig") as f:
            cases = list(csv.DictReader(f))
        with np.load(path/"precedent_embeddings_reviewed.npz", allow_pickle=False) as archive:
            saved = dict(zip(archive["record_ids"].tolist(), archive["vectors"], strict=True))
        key = secrets.token_hex(24)
        env = {"CONTRACTCHECK_EMBED_ENABLED":"1", "CONTRACTCHECK_INTERNAL_API_KEY":key,
               "CONTRACTCHECK_BATCH_MODE":"off", "HF_HUB_OFFLINE":"1"}
        with patch.dict(os.environ, env):
            os.environ.pop("CONTRACTCHECK_LEGACY_FOLD", None)
            os.environ.pop("CONTRACTCHECK_MODEL_DIR", None)
            with TestClient(create_app(), headers={"X-Internal-Api-Key":key}) as c:
                self.assertIsNone(c.app.state.legacy_predictor)
                cases = [r for r in cases if c.app.state.embedder.count_tokens(r["text"]) <= 512][:3]
                for rows, field, refs in [
                    (laws, "embedding_text", [r["embedding"] for r in laws]),
                    (cases, "text", [saved[r["record_id"]] for r in cases]),
                ]:
                    res = c.post("/embed", json={"texts":[r[field] for r in rows], "normalize":True})
                    self.assertEqual(res.status_code, 200)
                    vectors = np.array(res.json()["vectors"])
                    self.assertEqual(vectors.shape, (3, 1024))
                    self.assertTrue(np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-4))
                    cosine = (vectors * refs).sum(axis=1) / (
                        np.linalg.norm(vectors, axis=1) * np.linalg.norm(refs, axis=1))
                    self.assertTrue(np.all(cosine > 0.999))
