"""Internal BE wire contract. Classification is available for research demos only."""
import importlib
import os
import sys
from pathlib import Path
from typing import Annotated, Literal

import numpy as np
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator

from api.legacy import LABELS, SCORES_NOTE

router = APIRouter()
MAX_ITEMS = 64
MINIBATCH = 4
STRIDE = 32
MAX_WINDOWS = 16
MAX_BATCH_WINDOWS = 256
MAX_EMBED_TOKENS = 512
ClauseText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=10000)]


class WireRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class Clause(WireRequest):
    id: str = Field(min_length=1, max_length=128)
    text: ClauseText

    @field_validator("id")
    @classmethod
    def valid_id(cls, value):
        if value != value.strip() or not value:
            raise ValueError("id must be nonblank without outer whitespace")
        return value


class Batch(WireRequest):
    clauses: list[Clause] = Field(min_length=1, max_length=MAX_ITEMS)

    @field_validator("clauses")
    @classmethod
    def distinct_ids(cls, value):
        if len({entry.id for entry in value}) != len(value):
            raise ValueError("duplicate clause id")
        return value


class EmbedRequest(WireRequest):
    texts: list[ClauseText] = Field(min_length=1, max_length=MAX_ITEMS)
    normalize: bool = True

    @field_validator("normalize")
    @classmethod
    def normalized_only(cls, value):
        if not value:
            raise ValueError("only L2-normalized embeddings are supported")
        return value


class ResearchItem(BaseModel):
    id: str
    isRisky: bool
    riskType: Literal["RISK_CANDIDATE"] | None
    probs: dict[str, float]
    modelVersion: str
    weak_label: Literal[0, 1]
    weak_label_name: str
    before_summary_model_score: float
    windows: int


class ResearchBatch(BaseModel):
    results: list[ResearchItem]
    research_only: Literal[True] = True
    warning: str = SCORES_NOTE


class Vectors(BaseModel):
    vectors: list[list[float]]


def load_embedder():
    # Reuse the pinned law encoder; do not start the search app or load its corpus.
    data_root = Path(__file__).resolve().parents[2]
    if not (data_root / "law-embedding/api/app.py").is_file():
        raise RuntimeError("Embedding mode requires the DATA checkout with law-embedding/api")
    if str(data_root) not in sys.path:
        sys.path.append(str(data_root))
    encoder = importlib.import_module("law-embedding.api.app").LocalEmbedder
    device = os.getenv("CONTRACTCHECK_EMBED_DEVICE", "cpu")
    if device not in {"cpu", "cuda"}:
        raise ValueError("CONTRACTCHECK_EMBED_DEVICE must be cpu or cuda")
    return encoder(device)


def window_scores(model, windows):
    """Forward at most MINIBATCH windows at once; avoid padding all 64 clauses together."""
    scores = []
    with model.lock, model.torch.inference_mode():
        for start in range(0, len(windows), MINIBATCH):
            inputs = model.tokenizer.pad(
                {"input_ids": windows[start:start + MINIBATCH]}, padding=True, return_tensors="pt"
            )
            logits = model.model(**inputs).logits
            scores.extend(logits.softmax(dim=-1)[:, 1].tolist())
    return scores


def classify_batch(model, clauses):
    windows, counts = [], []
    for clause in clauses:
        # The single and batch routes share a fast tokenizer with mutable padding/stride settings.
        with model.lock:
            encoded = model.tokenizer(
                clause.text, add_special_tokens=True, max_length=model.max_length,
                truncation=True, stride=STRIDE, return_overflowing_tokens=True,
            )["input_ids"]
        count = len(encoded)
        if count > MAX_WINDOWS or len(windows) + count > MAX_BATCH_WINDOWS:
            raise HTTPException(422, detail={"code": "WINDOW_LIMIT_EXCEEDED"})
        windows.extend(encoded)
        counts.append(count)

    scores = np.asarray(window_scores(model, windows))
    if (scores.shape != (len(windows),) or not np.isfinite(scores).all()
            or (scores < 0).any() or (scores > 1).any()):
        raise HTTPException(500, detail={"code": "INVALID_MODEL_OUTPUT"})
    results = []
    start = 0
    for clause, count in zip(clauses, counts, strict=True):
        score = float(scores[start:start + count].max())
        start += count
        label = int(score > 0.5)
        results.append(ResearchItem(
            id=clause.id, isRisky=bool(label), riskType="RISK_CANDIDATE" if label else None,
            probs={"RISK_CANDIDATE": score} if label else {}, modelVersion=model.version,
            weak_label=label, weak_label_name=LABELS[label],
            before_summary_model_score=score, windows=count,
        ))
    return ResearchBatch(results=results)


@router.post("/classify")
def classify(body: Batch, request: Request) -> ResearchBatch:
    state = request.app.state
    if not state.demo or state.legacy_predictor is None:
        raise HTTPException(503, detail={"code": "RESEARCH_CLASSIFIER_DISABLED"})
    if request.headers.getlist("X-ContractCheck-Mode") != ["research-only"]:
        raise HTTPException(403, detail={"code": "RESEARCH_MODE_REQUIRED"})
    return classify_batch(state.legacy_predictor, body.clauses)


@router.post("/embed")
def embed(body: EmbedRequest, request: Request) -> Vectors:
    encoder = request.app.state.embedder
    if encoder is None:
        raise HTTPException(503, detail={"code": "EMBEDDING_DISABLED"})
    with request.app.state.embed_request_lock:
        return encode_batch(encoder, body)


def encode_batch(encoder, body: EmbedRequest) -> Vectors:
    if any(encoder.count_tokens(text) > MAX_EMBED_TOKENS for text in body.texts):
        raise HTTPException(422, detail={"code": "EMBED_TOKEN_LIMIT_EXCEEDED", "max_tokens": MAX_EMBED_TOKENS})
    vectors = []
    for start in range(0, len(body.texts), MINIBATCH):
        texts = body.texts[start:start + MINIBATCH]
        rows = np.asarray(encoder.encode(texts))
        if (rows.shape != (len(texts), 1024) or not np.isfinite(rows).all()
                or not np.allclose(np.linalg.norm(rows, axis=1), 1, rtol=0, atol=1e-4)):
            raise HTTPException(500, detail={"code": "INVALID_EMBEDDING_OUTPUT"})
        vectors.extend(rows.tolist())
    return Vectors(vectors=vectors)
