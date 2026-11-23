#!/usr/bin/env python3
"""법령 의미검색 + 검색 보정(코퍼스 중심화 + 상위 주성분 2개 제거).

- 문서 벡터는 기존 산출물(data/law_embeddings.jsonl, data/law_vectors.npy)을 그대로 사용한다. 재임베딩하지 않는다.
- 보정 변환(중심벡터, 주성분)은 문서 벡터만으로 계산했으며 정답 라벨을 쓰지 않았다.
- 점수는 코사인 유사도이며 위험확률이나 법적 판단이 아니다.

사용 예:
    python scripts/03_apply_retrieval_fix.py --jsonl data/law_embeddings.jsonl \
        --fix-dir retrieval_fix --query "월세를 한 번만 연체해도 즉시 계약을 해지한다."
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

MODEL_ID = "BAAI/bge-m3"
MODEL_REVISION = "3069def033ce91d907258f9a830e442610dbfe0b"


def load_corpus(jsonl_path):
    rows = [json.loads(l) for l in Path(jsonl_path).read_text(encoding="utf-8").splitlines() if l.strip()]
    vectors = np.array([r["embedding"] for r in rows], dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape[1] != 1024:
        raise ValueError(f"벡터 형상 이상: {vectors.shape}")
    if not np.isfinite(vectors).all():
        raise ValueError("벡터에 비유한값이 있습니다")
    return rows, vectors


class SearchFix:
    """중심벡터와 주성분을 적용해 허브(아무 질의에나 상위에 뜨는 조문) 현상을 줄인다."""

    def __init__(self, center, components):
        self.center = np.asarray(center, dtype=np.float32).reshape(-1)
        self.components = np.asarray(components, dtype=np.float32)
        if self.center.shape != (1024,) or self.components.shape[1] != 1024:
            raise ValueError("보정 파라미터 형상 불일치")

    @classmethod
    def from_dir(cls, fix_dir):
        fix_dir = Path(fix_dir)
        return cls(np.load(fix_dir / "fix_center.npy"), np.load(fix_dir / "fix_components_2.npy"))

    @classmethod
    def fit(cls, vectors, remove_pc=2):
        center = vectors.mean(axis=0)
        _, _, vt = np.linalg.svd(vectors - center, full_matrices=False)
        return cls(center, vt[:remove_pc])

    def apply(self, mat):
        x = np.atleast_2d(np.asarray(mat, dtype=np.float32)) - self.center
        for u in self.components:
            x = x - np.outer(x @ u, u)
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        if not np.all(norms > 1e-6):
            raise ValueError("보정 후 영벡터가 생겼습니다")
        return (x / norms).astype(np.float32)


class LawSearch:
    def __init__(self, rows, vectors, fix=None, device=None):
        self.rows = rows
        self.fix = fix
        self.doc_vectors = fix.apply(vectors) if fix else vectors
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION, use_fast=True)
        self.model = AutoModel.from_pretrained(MODEL_ID, revision=MODEL_REVISION, torch_dtype=dtype)
        self.model.to(self.device).eval()

    def encode(self, text):
        enc = self.tokenizer([text], return_tensors="pt", truncation=False)
        if enc["input_ids"].shape[1] > 512:
            raise ValueError("질의가 512토큰을 넘습니다. 조항 하나씩 입력하세요")
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with torch.inference_mode():
            hidden = self.model(**enc).last_hidden_state
        return F.normalize(hidden[:, 0, :].float(), p=2, dim=1).cpu().numpy().astype(np.float32)[0]

    def search(self, query, top_k=5):
        query = (query or "").strip()
        if not query:
            raise ValueError("검색할 계약 문구를 입력하세요")
        qvec = self.encode(query)
        if self.fix:
            qvec = self.fix.apply(qvec)[0]
        scores = self.doc_vectors @ qvec
        if not np.isfinite(scores).all():
            raise ValueError("검색 점수에 비유한값이 있습니다")
        results, seen = [], set()
        for idx in np.argsort(-scores):
            row = self.rows[int(idx)]
            if row["document_key"] in seen:
                continue
            seen.add(row["document_key"])
            results.append({
                "rank": len(results) + 1,
                "score": float(scores[idx]),
                "law_name": row.get("law_name"),
                "article": row.get("article"),
                "article_title": row.get("article_title"),
                "text": row["text"],
                "source_url": row["source_url"],
                "document_key": row["document_key"],
                "chunk_key": row["chunk_key"],
                "review_flags": row.get("review_flags", []),
            })
            if len(results) >= top_k:
                break
        return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", required=True, help="law_embeddings.jsonl 경로")
    parser.add_argument("--fix-dir", default=None, help="보정 파라미터 폴더 (생략하면 보정 없이 검색)")
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    rows, vectors = load_corpus(args.jsonl)
    fix = SearchFix.from_dir(args.fix_dir) if args.fix_dir else None
    engine = LawSearch(rows, vectors, fix)
    print("보정 적용:", bool(fix), "| 문서 청크:", len(rows))
    print("점수는 코사인 유사도이며 위험확률이 아닙니다. 적용 조건과 참조 조문은 따로 확인하세요.\n")
    for r in engine.search(args.query, args.top_k):
        print(f'{r["rank"]}. {r["law_name"]} {r["article"]} / 유사도 {r["score"]:.4f}')
        print(r["text"][:200])
        print("공식 출처:", r["source_url"], "\n")


if __name__ == "__main__":
    main()
