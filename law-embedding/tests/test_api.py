"""법령 검색 API 테스트. 모델을 내려받지 않고 가짜 인코더로 계약(스키마)과 동작을 검증한다."""
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from api.app import LawSearchService, RetrievalFix, create_app

DIM = 1024


def unit(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.normal(size=DIM).astype(np.float32)
    return v / np.linalg.norm(v)


class FakeEmbedder:
    """질의 문자열별로 고정 벡터를 돌려준다."""

    name = "local"

    def __init__(self, mapping):
        self.mapping = mapping
        self.calls = []

    def count_tokens(self, text):
        return len(text.split())

    def encode(self, texts):
        self.calls.append(list(texts))
        return np.stack([self.mapping.get(t, unit(999)) for t in texts])


def make_rows():
    base = [unit(1), unit(1) * 0.9 + unit(2) * 0.1, unit(3)]
    base = [v / np.linalg.norm(v) for v in base]
    meta = [
        ("LAW_A", "LAW_A@1:aaa:bbb:0-10", "민법", "제640조", "차임연체와 해지"),
        ("LAW_A", "LAW_A@1:aaa:bbb:10-20", "민법", "제640조", "차임연체와 해지"),
        ("LAW_B", "LAW_B@1:ccc:bbb:0-10", "주택임대차보호법", "제6조의2", "묵시적 갱신의 경우 계약의 해지"),
    ]
    rows = []
    for vec, (doc, chunk, law, article, title) in zip(base, meta):
        rows.append({
            "document_key": doc, "chunk_key": chunk, "law_name": law, "article": article,
            "article_title": title, "instrument_type": "ACT",
            "source_url": "https://www.law.go.kr/LSW/lsInfoP.do?lsiSeq=1",
            "source_version_seq": "1", "source_effective_date": "2026-01-02",
            "text": "조문 본문 " * 200, "review_flags": [], "application_caution": None,
            "embedding": vec.tolist(),
        })
    return rows, np.stack(base)


@pytest.fixture
def client_and_embedder():
    rows, vectors = make_rows()
    embedder = FakeEmbedder({"차임 연체": vectors[0], "묵시적 갱신": vectors[2]})
    service = LawSearchService(rows, vectors, None, embedder)
    with TestClient(create_app(service)) as client:
        yield client, embedder, rows


def test_health_reports_corpus_and_no_db(client_and_embedder):
    client, _, _ = client_and_embedder
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["documents"] == 2 and body["chunks"] == 3
    assert body["db_ingested"] is False
    assert client.get("/health/ready").json()["status"] == "ready"


def test_search_returns_contract_fields(client_and_embedder):
    client, _, _ = client_and_embedder
    body = client.post("/search/laws", json={"query": "차임 연체", "top_k": 2}).json()
    assert body["count"] == 2
    assert body["ranking_method"] == "max_chunk_cosine_similarity"
    assert "위험 확률" in body["score_note"]
    first = body["results"][0]
    assert first["law_name"] == "민법" and first["article"] == "제640조"
    assert first["rank"] == 1 and 0.99 <= first["score"] <= 1.0
    for key in ("document_key", "chunk_key", "source_url", "source_version_seq", "text", "review_flags"):
        assert key in first


def test_same_article_is_not_duplicated(client_and_embedder):
    client, _, _ = client_and_embedder
    body = client.post("/search/laws", json={"query": "차임 연체", "top_k": 5}).json()
    keys = [r["document_key"] for r in body["results"]]
    assert keys == ["LAW_A", "LAW_B"]
    assert len(set(keys)) == len(keys)


def test_text_is_truncated_by_request(client_and_embedder):
    client, _, _ = client_and_embedder
    body = client.post("/search/laws", json={"query": "차임 연체", "text_chars": 100}).json()
    assert all(len(r["text"]) <= 100 for r in body["results"])


def test_blank_and_unknown_fields_are_rejected(client_and_embedder):
    client, _, _ = client_and_embedder
    assert client.post("/search/laws", json={"query": "   "}).status_code == 422
    assert client.post("/search/laws", json={"query": "차임 연체", "top_k": 99}).status_code == 422
    assert client.post("/search/laws", json={"query": "차임 연체", "unknown": 1}).status_code == 422


def test_long_query_is_rejected_instead_of_truncated(client_and_embedder):
    client, _, _ = client_and_embedder
    response = client.post("/search/laws", json={"query": "단어 " * 600})
    assert response.status_code == 422
    assert "512" in response.json()["detail"]


def test_apply_fix_changes_ranking_vectors():
    rows, vectors = make_rows()
    center = vectors.mean(axis=0)
    centered = vectors - center
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    fix = RetrievalFix(center, vt[:1])
    embedder = FakeEmbedder({"차임 연체": vectors[0]})
    service = LawSearchService(rows, vectors, fix, embedder)
    with TestClient(create_app(service)) as client:
        off = client.post("/search/laws", json={"query": "차임 연체", "apply_fix": False}).json()
        on = client.post("/search/laws", json={"query": "차임 연체", "apply_fix": True}).json()
    assert off["retrieval_fix_applied"] is False
    assert on["retrieval_fix_applied"] is True
    # 질의가 1위 문서 벡터와 동일하면 양쪽 모두 1.0이므로, 다른 문서의 점수 변화로 확인한다.
    assert on["results"][0]["score"] == off["results"][0]["score"] == 1.0
    assert on["results"][1]["score"] != off["results"][1]["score"]
    assert not np.allclose(service.fixed_vectors, service.raw_vectors, atol=1e-4)
    assert np.allclose(np.linalg.norm(service.fixed_vectors, axis=1), 1.0, atol=1e-5)


def test_apply_fix_requested_without_parameters_is_422(client_and_embedder):
    client, _, _ = client_and_embedder
    response = client.post("/search/laws", json={"query": "차임 연체", "apply_fix": True})
    assert response.status_code == 422
    assert "보정" in response.json()["detail"]


def test_embed_endpoint_shares_model(client_and_embedder):
    client, embedder, _ = client_and_embedder
    body = client.post("/embed", json={"texts": ["차임 연체", "묵시적 갱신"]}).json()
    assert body["dimension"] == 1024 and body["normalized"] is True
    assert len(body["vectors"]) == 2 and len(body["vectors"][0]) == 1024
    assert embedder.calls[-1] == ["차임 연체", "묵시적 갱신"]
    assert client.post("/embed", json={"texts": ["  "]}).status_code == 422


def test_service_rejects_missing_embedding(tmp_path):
    rows, _ = make_rows()
    rows[0].pop("embedding")
    path = tmp_path / "broken.jsonl"
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="임베딩이 없는"):
        LawSearchService.from_files(path, None, FakeEmbedder({}))


def test_unconfigured_app_reports_503():
    with TestClient(create_app(None)) as client:
        assert client.get("/health").json()["status"] == "unavailable"
        assert client.post("/search/laws", json={"query": "차임 연체"}).status_code == 503
