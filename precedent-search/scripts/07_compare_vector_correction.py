"""Compare BGE-M3 precedent retrieval with corpus-centering / PC removal.

Run from precedent-search root (or any working directory):
    python scripts/07_compare_vector_correction.py

Reads existing reviewed CSV, saved NPZ and the original 20-question evaluation
file. Does not overwrite the original embeddings, search API, or evaluation.
"""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
CSV_PATH = DATA / "precedent_chunks_reviewed_with_urls.csv"
NPZ_PATH = DATA / "precedent_embeddings_reviewed.npz"
EVAL_SCRIPT = ROOT / "scripts" / "04_evaluate_retrieval.py"
OUT_CSV = DATA / "vector_correction_comparison.csv"
OUT_JSON = DATA / "vector_correction_comparison.json"


def unit_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    if np.any(norms < 1e-12):
        raise ValueError("변환 후 길이가 0인 벡터가 있습니다.")
    return x / norms


def load_test_questions():
    if not EVAL_SCRIPT.is_file():
        raise SystemExit(f"평가 스크립트 없음: {EVAL_SCRIPT}")
    spec = importlib.util.spec_from_file_location("existing_precedent_eval", EVAL_SCRIPT)
    if spec is None or spec.loader is None:
        raise SystemExit("기존 평가 스크립트를 읽을 수 없습니다.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.TESTS


def evaluate(name, doc_vectors, query_vectors, rows, tests):
    scores = query_vectors @ doc_vectors.T
    detail_rows = []
    hit1 = hit3 = 0
    rr_sum = 0.0
    t04_rank = None
    t04_top3 = None
    for i, (test_id, query, gold, category) in enumerate(tests):
        case_ranks = []
        seen = set()
        # The search API uses the maximum chunk score per source_id.
        for ix in np.argsort(scores[i])[::-1]:
            r = rows[int(ix)]
            sid = r["source_id"]
            if sid in seen:
                continue
            seen.add(sid)
            case_ranks.append((r["case_no"], float(scores[i, int(ix)])))
        names = [c for c, _ in case_ranks]
        rank = names.index(gold) + 1 if gold in names else None
        hit1 += int(rank == 1)
        hit3 += int(rank is not None and rank <= 3)
        rr_sum += (1.0 / rank) if rank is not None and rank <= 3 else 0.0
        if test_id == "T04":
            t04_rank = rank
            t04_top3 = names[:3]
        detail_rows.append({
            "method": name,
            "test_id": test_id,
            "category": category,
            "expected_case_no": gold,
            "correct_rank_all": rank if rank is not None else "not_found",
            "top1": names[0] if len(names) > 0 else "",
            "top2": names[1] if len(names) > 1 else "",
            "top3": names[2] if len(names) > 2 else "",
        })
    n = len(tests)
    return {
        "method": name,
        "hit1_count": hit1,
        "hit3_count": hit3,
        "hit_at_1": round(hit1 / n, 4),
        "hit_at_3": round(hit3 / n, 4),
        "mrr_at_3": round(rr_sum / n, 4),
        "T04_rank": t04_rank,
        "T04_top3": t04_top3,
    }, detail_rows


def main():
    tests = load_test_questions()
    if not CSV_PATH.is_file() or not NPZ_PATH.is_file():
        raise SystemExit("판례 CSV 또는 NPZ가 없습니다. precedent-search 폴더의 data/processed를 확인하세요.")
    with CSV_PATH.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    with np.load(NPZ_PATH, allow_pickle=False) as z:
        docs = np.asarray(z["vectors"], dtype=np.float64)
        ids = z["record_ids"].tolist()
        model_name = str(z["model_name"].item())
        checksum = str(z["csv_sha256"].item())
    if hashlib.sha256(CSV_PATH.read_bytes()).hexdigest() != checksum:
        raise SystemExit("CSV 해시와 NPZ 내 해시가 일치하지 않습니다. 서로 맞는 파일이 필요합니다.")
    if [r["record_id"] for r in rows] != ids or docs.shape[0] != len(rows):
        raise SystemExit("CSV와 임베딩의 개수 또는 record_id 순서가 다릅니다.")
    if docs.shape[1] != 1024:
        raise SystemExit(f"예상과 다른 임베딩 차원: {docs.shape}")
    missing = {t[2] for t in tests} - {r["case_no"] for r in rows}
    if missing:
        raise SystemExit("테스트 정답 판례 누락: " + ", ".join(sorted(missing)))

    print(f"데이터: {len({r['source_id'] for r in rows})}건 / {len(rows)}청크 / 평가 질문 {len(tests)}개")
    print(f"모델: {model_name} / 문서 임베딩 재생성 없음")
    print("질문 벡터 생성 중...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    queries = model.encode(
        [t[1] for t in tests], batch_size=4,
        normalize_embeddings=True, convert_to_numpy=True,
        show_progress_bar=True,
    ).astype(np.float64)
    if queries.shape[1] != docs.shape[1]:
        raise SystemExit("질문/문서 벡터 차원이 다릅니다.")

    variants = [("baseline_max", unit_rows(docs), unit_rows(queries))]
    # Center and PCA are fitted exclusively to the document corpus.
    # Apply the *same* mean and PC projections to both documents and queries.
    corpus_mean = docs.mean(axis=0, keepdims=True)
    centered_docs = docs - corpus_mean
    centered_queries = queries - corpus_mean
    _, singular_values, vh = np.linalg.svd(centered_docs, full_matrices=False)
    if len(singular_values) < 3 or singular_values[1] < 1e-10:
        raise SystemExit("주성분을 계산하기에 충분한 유효 문서 벡터가 없습니다.")
    for n_pc in (0, 1, 2):
        if n_pc:
            pc = vh[:n_pc].T
            corrected_docs = centered_docs - (centered_docs @ pc) @ pc.T
            corrected_queries = centered_queries - (centered_queries @ pc) @ pc.T
        else:
            corrected_docs = centered_docs
            corrected_queries = centered_queries
        variants.append((
            f"center_pc{n_pc}", unit_rows(corrected_docs), unit_rows(corrected_queries)
        ))

    reports = []
    all_details = []
    for name, vdocs, vqueries in variants:
        report, details = evaluate(name, vdocs, vqueries, rows, tests)
        reports.append(report)
        all_details.extend(details)

    print("\n========== 판례 벡터 후처리 비교 (동일 20개 질문) ==========")
    print(f"{'방식':<16} {'Hit@1':>13} {'Hit@3':>13} {'MRR@3':>9} {'T04 순위':>9}")
    for r in reports:
        print(
            f"{r['method']:<16} "
            f"{r['hit1_count']:>2}/{len(tests)} ({r['hit_at_1']:.0%}) "
            f"{r['hit3_count']:>2}/{len(tests)} ({r['hit_at_3']:.0%}) "
            f"{r['mrr_at_3']:>9.3f} {str(r['T04_rank']) + '위':>9}"
        )
    print("\nT04 상위 3건:")
    for r in reports:
        print(f"- {r['method']}: {' / '.join(r['T04_top3'])}")

    # New filenames only. Baseline CSV/NPZ/API and original evaluation remain unchanged.
    with OUT_CSV.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_details[0].keys()))
        writer.writeheader()
        writer.writerows(all_details)
    OUT_JSON.write_text(json.dumps({
        "n_cases": len({r["source_id"] for r in rows}),
        "n_chunks": len(rows),
        "n_questions": len(tests),
        "model": model_name,
        "methods": reports,
        "note": "내부 20개 질문. 실서비스 법률 정확도 아님. 우수한 점수만으로 자동 적용 금지."
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n비교 결과 저장:")
    print("-", OUT_CSV)
    print("-", OUT_JSON)
    print("\n기존 임베딩 및 API는 수정되지 않았습니다.")


if __name__ == "__main__":
    main()
