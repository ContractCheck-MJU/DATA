import os
import secrets
from threading import Lock
import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from fastapi.testclient import TestClient

from api.app import create_app
from api.internal import MINIBATCH


class Encoder:
    def __init__(self):
        self.sizes = []

    def count_tokens(self, text):
        return len(text) + 2

    def encode(self, texts):
        self.sizes.append(len(texts))
        return np.eye(1024, dtype=np.float32)[[int(s) for s in texts]]


class InternalTests(unittest.TestCase):
    def setUp(self):
        self.key = secrets.token_hex(24)
        self.headers = {"X-Internal-Api-Key": self.key, "X-ContractCheck-Mode": "research-only"}
        self.embed = {"texts": ["0"], "normalize": True}
        self.classify = {"clauses": [{"id": "11", "text": "0"}]}
        self.encoder = Encoder()
        self.predictor = SimpleNamespace(version="cv_20261116_seed42/fold_1", max_length=192, lock=Lock(),
                                         tokenizer=Mock(return_value={"input_ids": [[0], [1]]}))

    @contextmanager
    def client(self, mode="both", key=True):
        env = {"CONTRACTCHECK_INTERNAL_API_KEY": self.key} if key else {}
        if mode in ("both", "classify", "legacy"):
            env["CONTRACTCHECK_LEGACY_FOLD"] = "1"
        if mode in ("both", "classify"):
            env["CONTRACTCHECK_BATCH_MODE"] = "research-demo"
        if mode in ("both", "embed"):
            env["CONTRACTCHECK_EMBED_ENABLED"] = "1"
        with patch.dict(os.environ, env, clear=True), \
             patch("api.app.load_embedder", return_value=self.encoder) as embed_loader, \
             patch("api.legacy.LegacyPredictor", return_value=self.predictor) as classifier, \
             TestClient(create_app(), headers=self.headers) as client:
            self.assertEqual(embed_loader.called, mode in ("both", "embed"))
            self.assertEqual(classifier.called, mode in ("both", "classify", "legacy"))
            yield client

    def test_auth_on_every_inference_route(self):
        routes = ["/classify", "/embed", "/v1/clauses/classify", "/dev/v1/legacy/clauses/classify"]
        with self.client() as client:
            client.headers.clear()
            for route in routes:
                for headers in [{}, {"X-Internal-Api-Key": "wrong"}]:
                    self.assertEqual(client.post(route, json={}, headers=headers).status_code, 401)
        with self.client(key=False) as client:
            for route in routes:
                self.assertEqual(client.post(route, json={}).status_code, 503)

    def test_mode_boundaries(self):
        with self.client("classify") as client:
            self.assertEqual(client.post("/embed", json=self.embed).status_code, 503)
            del client.headers["X-ContractCheck-Mode"]
            self.assertEqual(client.post("/classify", json=self.classify).status_code, 403)
        for mode in ("embed", "off", "legacy"):
            with self.client(mode) as client:
                self.assertEqual(client.post("/classify", json=self.classify).status_code, 503)

    def test_batch_validation(self):
        with self.client() as c:
            for clauses in [[], self.classify["clauses"]*2, [{"id": " ", "text":"x"}],
                            [{"id": 1, "text":"x"}], [{"id":"1", "text":"   "}],
                            [{"id":str(i), "text":"x"} for i in range(65)]]:
                self.assertEqual(c.post("/classify", json={"clauses": clauses}).status_code, 422)
            for body in [{"texts":[]}, {"texts":["0"]*65}, {"texts":[" "]}, {"texts":["0"], "normalize":False},
                         {"texts":["0"], "normalize":1}, {"texts":["0"*511]}]:
                self.assertEqual(c.post("/embed", json=body).status_code, 422)

    def test_embed_order_and_minibatches(self):
        with self.client("embed") as c:
            texts = [str(i) for i in reversed(range(64))]
            response = c.post("/embed", json={"texts":texts})
            self.assertEqual(response.status_code, 200)
            vectors = np.array(response.json()["vectors"])
            self.assertEqual(vectors.shape, (64, 1024))
            self.assertEqual(vectors.argmax(axis=1).tolist(), list(reversed(range(64))))
            self.assertLessEqual(max(self.encoder.sizes), MINIBATCH)

    def test_embed_malformed_output_fails_entire_request(self):
        with self.client() as c:
            for rows in [np.zeros((1,1024)), np.ones((1,1023)), np.ones((2,1024)),
                         np.full((1,1024),np.nan), np.full((1,1024),np.inf)]:
                with patch.object(self.encoder, "encode", return_value=rows):
                    response = c.post("/embed", json=self.embed)
                    self.assertEqual(response.status_code, 500)
                    self.assertNotIn("vectors", response.json())

    def test_max_score_and_weak_label_survive_grouping(self):
        clauses = [{"id":str(i), "text":"text"} for i in reversed(range(64))]
        with self.client() as c, patch("api.internal.window_scores", return_value=[.2,.93,.3,.4]*32):
            response = c.post("/classify", json={"clauses":clauses})
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertIs(data["research_only"], True)
            rows = data["results"]
            self.assertEqual([r["id"] for r in rows], [x["id"] for x in clauses])
            self.assertEqual(rows[0]["probs"], {"RISK_CANDIDATE": .93})
            self.assertEqual(rows[0]["weak_label_name"], "BEFORE_CORRECTION_SUMMARY_WEAK")
            self.assertEqual(rows[1]["probs"], {})
            self.assertIsNone(rows[1]["riskType"])
            self.assertEqual(rows[1]["before_summary_model_score"], .4)
            self.assertEqual(rows[1]["windows"], 2)
            self.predictor.tokenizer.assert_called_with("text", add_special_tokens=True, max_length=192,
                                                       truncation=True, stride=32, return_overflowing_tokens=True)

    def test_window_budgets_and_error_privacy(self):
        with self.client() as c:
            for count,n in [(17,1),(16,17)]:
                self.predictor.tokenizer.return_value = {"input_ids":[[1]]*count}
                response = c.post("/classify", json={"clauses":[{"id":str(i),"text":"test"} for i in range(n)]})
                self.assertEqual(response.status_code,422)
            self.predictor.tokenizer.side_effect = RuntimeError("sensitive-test-input")
            response = c.post("/classify",json=self.classify)
            self.assertEqual(response.status_code,500)
            self.assertNotIn("sensitive-test-input",response.text)
            response = c.post("/embed",json={"texts":[{"sample":"sensitive-test-input"}]})
            self.assertEqual(response.status_code,422)
            self.assertNotIn("sensitive-test-input",response.text)
