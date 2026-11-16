#!/usr/bin/env python3
"""BGE-M3 dense CLS embeddings, cosine retrieval, and local export. No DB writes."""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

MODEL_ID = "BAAI/bge-m3"
MODEL_REVISION = "3069def033ce91d907258f9a830e442610dbfe0b"

EXAMPLE_QUERIES = [
    "계약 기간 중 모든 수리비와 수선비는 임차인이 부담한다.",
    "새로운 세입자가 구해져야 보증금을 돌려준다고 합니다.",
    "월세를 한 번만 연체해도 즉시 계약을 해지한다.",
    "묵시적으로 갱신된 계약을 해지하려면 언제 효력이 발생하나요?",
    "집주인이 실제 거주한다는 이유로 계약갱신을 거절했습니다.",
    "임차인은 전입신고를 하지 않기로 한다.",
    "퇴거할 때 도배와 장판을 전부 새것으로 교체해야 한다.",
    "임대인이 필요할 때마다 월세를 일방적으로 인상한다.",
    "보증금 일부를 월세로 바꿀 때 적용하는 전환율은 얼마인가요?",
    "소액임차인이 보증금 중 일정액을 우선변제받는 금액 기준을 알고 싶습니다.",
    "보증금을 돌려받지 못한 채 이사해야 하는데 임차권등기를 신청하고 싶습니다.",
    "관리비에 장기수선충당금이 포함되어 있는데 세입자가 대신 낸 돈을 돌려받을 수 있나요?",
]


class DenseLawSearch:
    def __init__(self, chunks):
        self.chunks = chunks
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION, use_fast=True)
        if not self.tokenizer.is_fast:
            raise RuntimeError("fast tokenizer required")
        print("모델 다운로드/로딩 시작:", MODEL_ID, "device:", self.device, flush=True)
        self.model = AutoModel.from_pretrained(MODEL_ID, revision=MODEL_REVISION, torch_dtype=self.dtype)
        self.model.to(self.device).eval()
        if int(self.model.config.hidden_size) != 1024:
            raise RuntimeError("BGE-M3 차원 불일치")
        print("모델 로딩 완료", flush=True)
        self.vectors = None

    def encode(self, texts, batch_size=None):
        batch_size = batch_size or (8 if self.device.type == "cuda" else 2)
        output = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            encoded = self.tokenizer(batch, padding=True, truncation=False, return_tensors="pt")
            if encoded["input_ids"].shape[1] > 8192:
                raise ValueError("모델 입력 길이 초과; 자동 절단하지 않음")
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            with torch.inference_mode():
                hidden = self.model(**encoded).last_hidden_state
                dense = F.normalize(hidden[:, 0, :].float(), p=2, dim=1)
            output.append(dense.cpu().numpy().astype(np.float32, copy=False))
            if len(texts) > 16:
                print("임베딩 진행:", min(start + batch_size, len(texts)), "/", len(texts), flush=True)
        return np.concatenate(output, axis=0)

    def search(self, query, top_k=5):
        query = query.strip()
        if not query:
            raise ValueError("검색할 계약 문구를 입력하세요")
        qn = len(self.tokenizer(query, add_special_tokens=True, truncation=False)["input_ids"])
        if qn > 512:
            raise ValueError("검색 문구가 512토큰을 넘습니다. 조항 하나씩 입력하세요")
        if self.vectors is None:
            raise RuntimeError("문서 임베딩이 아직 없습니다")
        qvec = self.encode([query])[0]
        if not np.isfinite(qvec).all() or not np.isclose(np.linalg.norm(qvec), 1.0, atol=1e-5):
            raise ValueError("검색어 벡터 유한값/정규화 검증 실패")
        scores = self.vectors @ qvec
        if not np.isfinite(scores).all():
            raise ValueError("검색 점수 비유한값")
        selected, seen = [], set()
        for idx in np.argsort(-scores):
            c = self.chunks[int(idx)]
            if c["document_key"] in seen:
                continue
            seen.add(c["document_key"])
            selected.append({"rank": len(selected) + 1, "query": query, "score": float(scores[idx]), "document_key": c["document_key"], "chunk_key": c["chunk_key"], "law_name": c.get("law_name"), "article": c.get("article"), "article_title": c.get("article_title"), "source_effective_date": c.get("source_effective_date"), "source_url": c["source_url"], "text": c["text"], "context_spans": c.get("context_spans", []), "review_flags": c.get("review_flags", [])})
            if len(selected) >= top_k:
                break
        return selected


def build(input_file, output_dir):
    t0 = time.time()
    input_bytes = Path(input_file).read_bytes()
    chunks = [json.loads(line) for line in input_bytes.decode("utf-8-sig").splitlines() if line.strip()]
    if len(chunks) != 165 or len({c["chunk_key"] for c in chunks}) != 165:
        raise ValueError("현재 고정 입력은 중복 없는 청크165개여야 합니다")
    if len({c["document_key"] for c in chunks}) != 95 or not all(c["document_key"].startswith("LAW_") for c in chunks):
        raise ValueError("법령95개 범위 불일치")
    if not all(c["tokenizer_revision"] == MODEL_REVISION for c in chunks):
        raise ValueError("청킹과 임베딩 토크나이저 판본이 다릅니다")
    engine = DenseLawSearch(chunks)
    texts = [c["embedding_text"] for c in chunks]
    token_counts = [len(engine.tokenizer(t, add_special_tokens=True, truncation=False)["input_ids"]) for t in texts]
    if token_counts != [c["token_count"] for c in chunks]:
        raise ValueError("실제 임베딩 입력 토큰 수 불일치")
    engine.vectors = engine.encode(texts)
    vectors = engine.vectors
    if vectors.shape != (165, 1024) or not np.isfinite(vectors).all():
        raise ValueError("벡터 형상/유한값 검증 실패")
    norms = np.linalg.norm(vectors, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-5):
        raise ValueError("벡터 L2 정규화 실패")
    if np.any(np.linalg.norm(vectors, axis=1) == 0):
        raise ValueError("영벡터")
    probe = engine.encode([texts[0]])[0]
    probe_score = float(probe @ vectors[0])
    if not np.isfinite(probe).all() or not np.isfinite(probe_score) or probe_score < 0.99:
        raise ValueError("같은 입력 재인코딩 일관성 실패")
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    np.save(outdir / "법령_벡터.npy", vectors, allow_pickle=False)
    index = [{"vector_row": i, "chunk_key": c["chunk_key"], "document_key": c["document_key"], "law_name": c.get("law_name"), "article": c.get("article"), "source_version_seq": c.get("source_version_seq"), "source_url": c["source_url"]} for i, c in enumerate(chunks)]
    (outdir / "법령_벡터인덱스.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in index), encoding="utf-8")
    with (outdir / "법령_임베딩.jsonl").open("w", encoding="utf-8") as handle:
        for i, c in enumerate(chunks):
            r = dict(c)
            r.update({"vector_row": i, "embedding": vectors[i].tolist(), "embedding_status": "generated", "embedding_model": MODEL_ID, "embedding_model_revision": MODEL_REVISION, "embedding_pooling": "CLS_last_hidden_state", "embedding_normalized": True, "embedding_storage_dtype": "float32", "db_id": None, "db_document_id": None})
            handle.write(json.dumps(r, ensure_ascii=False) + "\n")
    results = [r for q in EXAMPLE_QUERIES for r in engine.search(q, 5)]
    columns = ["query", "rank", "score", "law_name", "article", "article_title", "source_effective_date", "source_url", "document_key", "chunk_key", "text", "review_flags"]
    with (outdir / "법령_검색예시.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for r in results:
            writer.writerow({k: ";".join(r[k]) if k == "review_flags" else r[k] for k in columns})
    (outdir / "법령_검색예시.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    readable = ["법령 의미검색 예시", "점수는 코사인 유사도이며 위험확률/법적 적합성 확률이 아닙니다.", "자동 검색 순위 예시이며 정답 검수/정확도 평가를 마친 결과가 아닙니다.", ""]
    for q in EXAMPLE_QUERIES:
        readable += ["=" * 60, "검색 문구: " + q]
        for r in [r for r in results if r["query"] == q]:
            readable += ["", str(r["rank"]) + ". " + str(r["law_name"]) + " " + str(r["article"]) + " / 유사도 " + format(r["score"], ".4f"), "출처: " + r["source_url"], r["text"]]
        readable.append("")
    (outdir / "법령_검색예시.txt").write_text("\n".join(readable), encoding="utf-8-sig")
    meta = {"created_at": datetime.now(timezone.utc).isoformat(), "source_chunks_sha256": hashlib.sha256(input_bytes).hexdigest(), "source_documents": 95, "vector_count": len(vectors), "vector_dimension": 1024, "vector_shape": list(vectors.shape), "vector_dtype": str(vectors.dtype), "finite_values": bool(np.isfinite(vectors).all()), "zero_vectors": int(np.sum(norms == 0)), "l2_norm_min": float(norms.min()), "l2_norm_max": float(norms.max()), "same_input_reencode_cosine": probe_score, "embedding_model": MODEL_ID, "revision": MODEL_REVISION, "pooling": "CLS_last_hidden_state", "normalized": True, "input_field": "embedding_text", "max_input_tokens": max(token_counts), "token_counts_match_chunk_audit": True, "device": str(engine.device), "model_dtype": str(engine.dtype), "transformers_version": importlib.metadata.version("transformers"), "torch_version": torch.__version__, "example_queries": len(EXAMPLE_QUERIES), "example_results": len(results), "search_score_semantics": "cosine similarity, not risk probability", "retrieval_relevance_human_validated": False, "legal_applicability_verified": False, "db_ingested": False, "elapsed_seconds": round(time.time() - t0, 2)}
    (outdir / "법령_임베딩검증.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    loaded = np.load(outdir / "법령_벡터.npy", allow_pickle=False)
    if not np.array_equal(loaded, vectors):
        raise ValueError("NPY 저장 재개방 검증 실패")
    names = ["법령_벡터.npy", "법령_벡터인덱스.jsonl", "법령_임베딩.jsonl", "법령_검색예시.csv", "법령_검색예시.json", "법령_검색예시.txt", "법령_임베딩검증.json"]
    archive = outdir.parent / "법령_임베딩결과.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as package:
        for name in names:
            package.write(outdir / name, name)
    with zipfile.ZipFile(archive) as package:
        if package.testzip() is not None:
            raise ValueError("ZIP 검증 실패")
    print(json.dumps(meta, ensure_ascii=False, indent=2), flush=True)
    return engine, meta, archive


def show_search(engine):
    import ipywidgets as widgets
    from IPython.display import display, clear_output
    textbox = widgets.Text(value="계약 기간 중 모든 수리비는 임차인이 부담한다.", description="계약 문구:", layout=widgets.Layout(width="95%"))
    button = widgets.Button(description="관련 법령 검색", button_style="primary")
    output = widgets.Output()
    def on_click(_):
        with output:
            clear_output(wait=True)
            try:
                rows = engine.search(textbox.value, 5)
                print("점수는 유사도이며 위험확률이 아닙니다. 적용조건과 참조조문은 별도로 확인하세요.\n")
                for r in rows:
                    print(str(r["rank"]) + ". " + str(r["law_name"]) + " " + str(r["article"]) + " / 유사도 " + format(r["score"], ".4f"))
                    print(r["text"])
                    print("공식 출처:", r["source_url"])
                    print()
            except Exception as exc:
                print(type(exc).__name__ + ": " + str(exc))
    button.on_click(on_click)
    display(textbox, button, output)
    on_click(None)
    return textbox, button, output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="embedding_output")
    args = parser.parse_args()
    build(args.input, args.output)
