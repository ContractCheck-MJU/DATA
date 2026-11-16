"""ContractCheck: reviewed precedent vector search API (NPZ, BGE-M3).

Usage (from inner ContractCheck_RAG_Starter directory):
  python -m pip install fastapi uvicorn
  python -m uvicorn scripts.06_precedent_search_api:app --host 127.0.0.1 --port 8001

Note: the module filename begins with a digit, so python -m uvicorn may use
importlib, which supports numeric module components, but a safer command is:
  python scripts/06_precedent_search_api.py
"""
from __future__ import annotations

import csv
import hashlib
from functools import lru_cache
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

BASE = Path(__file__).resolve().parents[1]
CSV_PATH = BASE / "data/processed/precedent_chunks_reviewed_with_urls.csv"
NPZ_PATH = BASE / "data/processed/precedent_embeddings_reviewed.npz"

app = FastAPI(title="ContractCheck 판례 검색 API", version="0.1.0")


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=5000, description="계약 조항 또는 질문")
    top_k: int = Field(default=3, ge=1, le=16)
    domain: str | None = Field(default=None, description="선택적 분야 필터. 예: residential_rental")


@lru_cache(maxsize=1)
def load_resources():
    if not CSV_PATH.is_file() or not NPZ_PATH.is_file():
        raise FileNotFoundError("검수 CSV 또는 임베딩 NPZ 파일을 찾을 수 없습니다.")

    with CSV_PATH.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    with np.load(NPZ_PATH, allow_pickle=False) as z:
        vectors = np.asarray(z["vectors"], dtype=np.float32)
        record_ids = z["record_ids"].tolist()
        model_name = str(z["model_name"].item())
        expected_hash = str(z["csv_sha256"].item())

    actual_hash = hashlib.sha256(CSV_PATH.read_bytes()).hexdigest()
    if actual_hash != expected_hash:
        raise ValueError("CSV가 임베딩 이후 변경됐습니다. 서로 일치하는 파일이 필요합니다.")
    if [r["record_id"] for r in rows] != record_ids:
        raise ValueError("CSV와 임베딩의 record_id 순서가 일치하지 않습니다.")
    if vectors.ndim != 2 or vectors.shape[0] != len(rows):
        raise ValueError("임베딩 배열 크기와 CSV 행 개수가 맞지 않습니다.")

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    return rows, vectors, model, model_name


@app.get("/health")
def health():
    return {
        "status": "ok" if CSV_PATH.is_file() and NPZ_PATH.is_file() else "missing_files",
        "csv_exists": CSV_PATH.is_file(),
        "embeddings_exists": NPZ_PATH.is_file(),
    }


@app.post("/search/precedents")
def search_precedents(body: SearchRequest):
    query = body.query.strip()
    if not query:
        raise HTTPException(status_code=422, detail="빈 검색어는 사용할 수 없습니다.")
    try:
        rows, vectors, model, model_name = load_resources()
    except (OSError, KeyError, ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    query_vector = model.encode(query, normalize_embeddings=True, convert_to_numpy=True)
    scores = vectors @ np.asarray(query_vector, dtype=np.float32)

    # Existing baseline: maximum chunk score per unique precedent (source_id).
    chosen = set()
    results = []
    for ix in np.argsort(scores)[::-1]:
        row = rows[int(ix)]
        if body.domain and row.get("domain") != body.domain:
            continue
        source_id = row["source_id"]
        if source_id in chosen:
            continue
        chosen.add(source_id)
        results.append({
            "rank": len(results) + 1,
            "case_no": row.get("case_no", ""),
            "source_id": source_id,
            "record_id": row.get("record_id", ""),
            "source_title": row.get("source_title", ""),
            "source_url": row.get("source_url", ""),
            "domain": row.get("domain", ""),
            "court": row.get("court", ""),
            "decided_at": row.get("decided_at", ""),
            "chunk_section": row.get("chunk_section", ""),
            "score": round(float(scores[ix]), 4),
            "text": row.get("text", "")[:500],
        })
        if len(results) >= body.top_k:
            break

    return {
        "query": query,
        "model": model_name,
        "ranking_method": "max_chunk_cosine_similarity",
        "score_note": "검색 유사도이며 법적 위험 확률이나 법률적 적합성 판정이 아닙니다.",
        "count": len(results),
        "results": results,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)
