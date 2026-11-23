"""ContractCheck: 법령 벡터 검색 API (BGE-M3, 사전 생성 벡터 사용).

- 이미 만들어 둔 법령 임베딩(법령 95건 / 청크 165개)을 읽어 검색만 수행한다. 재임베딩하지 않는다.
- 질의 문장만 런타임에 인코딩한다.
- 선택적으로 검색 보정(코퍼스 중심화 + 상위 주성분 제거)을 적용한다.
- 점수는 코사인 유사도이며 위험확률이나 법적 판단이 아니다.

실행:
  python -m pip install -r requirements.txt
  python -m uvicorn app:app --host 127.0.0.1 --port 8002

환경변수:
  LAW_EMBEDDING_JSONL     기본 ../data/law_embeddings.jsonl
  LAW_RETRIEVAL_FIX_DIR   기본 ../retrieval_fix (없으면 보정 없이 동작)
  LAW_EMBED_BACKEND       local(기본) | http
  LAW_EMBED_URL           LAW_EMBED_BACKEND=http 일 때 호출할 /embed 주소
  LAW_MODEL_DEVICE        cpu | cuda (기본 자동)
"""
from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

MODEL_ID = "BAAI/bge-m3"
MODEL_REVISION = "3069def033ce91d907258f9a830e442610dbfe0b"
VECTOR_DIM = 1024
MAX_QUERY_TOKENS = 512

BASE = Path(__file__).resolve().parent
DEFAULT_JSONL = BASE.parent / "data" / "law_embeddings.jsonl"
DEFAULT_FIX_DIR = BASE.parent / "retrieval_fix"

SCORE_NOTE = "검색 유사도이며 법적 위험 확률이나 법률적 적합성 판정이 아닙니다."
APPLICATION_NOTE = "적용 조건(지역·계약 시점·건물 유형 등)과 참조 조문은 별도로 확인해야 합니다."


class SearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    query: str = Field(min_length=1, max_length=5000, description="계약 조항 또는 질문")
    top_k: int = Field(default=5, ge=1, le=16)
    apply_fix: bool | None = Field(default=None, description="검색 보정 사용 여부. 생략하면 서버 기본값")
    text_chars: int = Field(default=500, ge=100, le=4000, description="응답 본문 길이 상한")


class EmbedRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    texts: list[str] = Field(min_length=1, max_length=32)


class RetrievalFix:
    """문서 벡터만으로 계산한 후처리 보정. 정답 라벨을 쓰지 않는다."""

    def __init__(self, center: np.ndarray, components: np.ndarray):
        self.center = np.asarray(center, dtype=np.float32).reshape(-1)
        self.components = np.atleast_2d(np.asarray(components, dtype=np.float32))
        if self.center.shape != (VECTOR_DIM,) or self.components.shape[1] != VECTOR_DIM:
            raise ValueError("보정 파라미터 형상이 벡터 차원과 다릅니다")

    @classmethod
    def from_dir(cls, directory: Path):
        center_path = directory / "fix_center.npy"
        pc_path = directory / "fix_components_2.npy"
        if not center_path.is_file() or not pc_path.is_file():
            return None
        return cls(np.load(center_path, allow_pickle=False), np.load(pc_path, allow_pickle=False))

    def apply(self, mat: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(np.asarray(mat, dtype=np.float32)) - self.center
        for component in self.components:
            x = x - np.outer(x @ component, component)
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        if not np.all(norms > 1e-6):
            raise ValueError("보정 후 영벡터가 생겼습니다")
        return (x / norms).astype(np.float32)


class LocalEmbedder:
    """질의 인코딩용. 문서 벡터와 동일한 모델·revision·CLS 풀링·L2 정규화를 사용한다."""

    name = "local"

    def __init__(self, device: str | None = None):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION, use_fast=True)
        if not self.tokenizer.is_fast:
            raise RuntimeError("fast tokenizer가 필요합니다")
        self.model = AutoModel.from_pretrained(MODEL_ID, revision=MODEL_REVISION, torch_dtype=dtype)
        if int(self.model.config.hidden_size) != VECTOR_DIM:
            raise RuntimeError("모델 차원이 저장된 벡터와 다릅니다")
        self.model.to(self.device).eval()
        self.lock = Lock()

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"])

    def encode(self, texts: list[str]) -> np.ndarray:
        import torch.nn.functional as F

        enc = self.tokenizer(texts, padding=True, truncation=False, return_tensors="pt")
        if enc["input_ids"].shape[1] > 8192:
            raise ValueError("모델 입력 길이를 넘었습니다")
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with self.lock, self.torch.inference_mode():
            hidden = self.model(**enc).last_hidden_state
        vectors = F.normalize(hidden[:, 0, :].float(), p=2, dim=1).cpu().numpy().astype(np.float32)
        if not np.isfinite(vectors).all():
            raise ValueError("질의 벡터에 비유한값이 있습니다")
        return vectors


class HttpEmbedder:
    """RAM이 부족한 서버에서 모델 인스턴스를 공유하기 위한 원격 인코딩."""

    name = "http"

    def __init__(self, url: str, timeout: float = 30.0):
        import httpx

        self.url = url
        self.client = httpx.Client(timeout=timeout)

    def count_tokens(self, text: str) -> int:
        return len(text)  # 원격 모드에서는 토크나이저가 없어 길이 상한만 확인한다

    def encode(self, texts: list[str]) -> np.ndarray:
        response = self.client.post(self.url, json={"texts": texts})
        response.raise_for_status()
        vectors = np.asarray(response.json()["vectors"], dtype=np.float32)
        if vectors.shape != (len(texts), VECTOR_DIM) or not np.isfinite(vectors).all():
            raise ValueError("원격 임베딩 응답 형상이 올바르지 않습니다")
        return vectors


class LawSearchService:
    def __init__(self, rows: list[dict], vectors: np.ndarray, fix: RetrievalFix | None, embedder):
        if vectors.ndim != 2 or vectors.shape != (len(rows), VECTOR_DIM):
            raise ValueError("벡터 배열과 청크 수가 맞지 않습니다")
        if not np.isfinite(vectors).all():
            raise ValueError("저장된 벡터에 비유한값이 있습니다")
        self.rows = rows
        self.raw_vectors = vectors
        self.fix = fix
        self.embedder = embedder
        self.fixed_vectors = fix.apply(vectors) if fix else None
        self.documents = len({r["document_key"] for r in rows})

    @classmethod
    def from_files(cls, jsonl_path: Path, fix_dir: Path | None, embedder):
        rows = [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not rows:
            raise ValueError("임베딩 파일이 비어 있습니다")
        missing = [r.get("chunk_key") for r in rows if "embedding" not in r or r.get("embedding") is None]
        if missing:
            raise ValueError(f"임베딩이 없는 청크가 있습니다: {len(missing)}개")
        vectors = np.asarray([r["embedding"] for r in rows], dtype=np.float32)
        fix = RetrievalFix.from_dir(fix_dir) if fix_dir else None
        return cls(rows, vectors, fix, embedder)

    def search(self, query: str, top_k: int, apply_fix: bool, text_chars: int) -> dict:
        query = query.strip()
        if not query:
            raise ValueError("빈 검색어는 사용할 수 없습니다")
        if apply_fix and self.fixed_vectors is None:
            raise ValueError("보정 파라미터가 설치되지 않아 apply_fix=true를 쓸 수 없습니다")
        tokens = self.embedder.count_tokens(query)
        if self.embedder.name == "local" and tokens > MAX_QUERY_TOKENS:
            raise ValueError(f"질의가 {MAX_QUERY_TOKENS}토큰을 넘습니다. 조항을 하나씩 보내세요")

        qvec = self.embedder.encode([query])[0]
        doc_vectors = self.raw_vectors
        if apply_fix:
            qvec = self.fix.apply(qvec)[0]
            doc_vectors = self.fixed_vectors

        scores = doc_vectors @ qvec
        if not np.isfinite(scores).all():
            raise ValueError("검색 점수에 비유한값이 있습니다")

        results, seen = [], set()
        for idx in np.argsort(-scores):
            row = self.rows[int(idx)]
            document_key = row["document_key"]
            if document_key in seen:
                continue
            seen.add(document_key)
            results.append({
                "rank": len(results) + 1,
                "document_key": document_key,
                "chunk_key": row["chunk_key"],
                "law_name": row.get("law_name"),
                "article": row.get("article"),
                "article_title": row.get("article_title"),
                "instrument_type": row.get("instrument_type"),
                "source_url": row["source_url"],
                "source_version_seq": row.get("source_version_seq"),
                "source_effective_date": row.get("source_effective_date"),
                "score": round(float(scores[idx]), 4),
                "text": (row.get("text") or "")[:text_chars],
                "review_flags": row.get("review_flags", []),
                "application_caution": row.get("application_caution"),
            })
            if len(results) >= top_k:
                break

        return {
            "query": query,
            "model": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "ranking_method": "max_chunk_cosine_similarity",
            "retrieval_fix_applied": bool(apply_fix),
            "score_note": SCORE_NOTE,
            "application_note": APPLICATION_NOTE,
            "count": len(results),
            "results": results,
        }


def build_embedder():
    backend = os.environ.get("LAW_EMBED_BACKEND", "local").lower()
    if backend == "http":
        url = os.environ.get("LAW_EMBED_URL")
        if not url:
            raise ValueError("LAW_EMBED_BACKEND=http 이면 LAW_EMBED_URL이 필요합니다")
        return HttpEmbedder(url)
    if backend != "local":
        raise ValueError(f"알 수 없는 LAW_EMBED_BACKEND: {backend}")
    return LocalEmbedder(os.environ.get("LAW_MODEL_DEVICE"))


def create_app(service: LawSearchService | None = None, default_apply_fix: bool = True):
    @asynccontextmanager
    async def lifespan(application: FastAPI):
        application.state.service = service
        application.state.startup_error = None
        if service is None:
            try:
                jsonl = Path(os.environ.get("LAW_EMBEDDING_JSONL", DEFAULT_JSONL))
                fix_dir = Path(os.environ.get("LAW_RETRIEVAL_FIX_DIR", DEFAULT_FIX_DIR))
                application.state.service = LawSearchService.from_files(jsonl, fix_dir, build_embedder())
            except Exception as exc:  # 설정 오류를 조용히 숨기지 않고 상태로 드러낸다
                application.state.startup_error = f"{type(exc).__name__}: {exc}"
        yield

    application = FastAPI(title="ContractCheck 법령 검색 API", version="0.1.0", lifespan=lifespan)
    application.state.default_apply_fix = default_apply_fix

    def get_service() -> LawSearchService:
        current = getattr(application.state, "service", None)
        if current is None:
            raise HTTPException(503, application.state.startup_error or "법령 임베딩이 로드되지 않았습니다")
        return current

    @application.get("/health")
    def health():
        current = getattr(application.state, "service", None)
        return {
            "status": "ok" if current else "unavailable",
            "error": getattr(application.state, "startup_error", None),
            "documents": current.documents if current else 0,
            "chunks": len(current.rows) if current else 0,
            "retrieval_fix_available": bool(current and current.fixed_vectors is not None),
            "embed_backend": current.embedder.name if current else None,
            "db_ingested": False,
        }

    @application.get("/health/ready")
    def ready():
        get_service()
        return {"status": "ready"}

    @application.post("/search/laws")
    def search_laws(body: SearchRequest):
        current = get_service()
        apply_fix = application.state.default_apply_fix if body.apply_fix is None else body.apply_fix
        if apply_fix and current.fixed_vectors is None:
            apply_fix = False if body.apply_fix is None else apply_fix
        try:
            return current.search(body.query, body.top_k, apply_fix, body.text_chars)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        except Exception as exc:
            raise HTTPException(500, f"{type(exc).__name__}: {exc}") from exc

    @application.post("/embed")
    def embed(body: EmbedRequest):
        """같은 서버에서 다른 서비스가 모델을 중복 적재하지 않도록 인코딩을 공유한다."""
        current = get_service()
        texts = [t.strip() for t in body.texts]
        if any(not t for t in texts):
            raise HTTPException(422, "빈 문자열은 인코딩할 수 없습니다")
        try:
            vectors = current.embedder.encode(texts)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        return {"model": MODEL_ID, "model_revision": MODEL_REVISION, "pooling": "CLS_last_hidden_state",
                "normalized": True, "dimension": VECTOR_DIM, "vectors": vectors.tolist()}

    return application


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8002)
