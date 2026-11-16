r"""ContractCheck 판례 RAG: 사례 단위 검색 정확도 검증.

실행 위치: 프로젝트 최상위 폴더(안쪽 ContractCheck_RAG_Starter)
  ..\.venv\Scripts\python.exe scripts\04_evaluate_retrieval.py

기존 CSV/임베딩은 읽기만 합니다. 보고서만 새로 생성합니다.
검증은 '정답 판례를 Top-k에서 찾아내는지'에 관한 것이며,
법률적 적용 가능성/검색 결과의 완전한 정확성을 검증하지 않습니다.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data" / "processed"
DEFAULT_CSV = DATA / "precedent_chunks_reviewed_with_urls.csv"
DEFAULT_NPZ = DATA / "precedent_embeddings_reviewed.npz"

# 금지: 사건번호를 질문에 직접 넣으면 쉽게 정답이 나오므로 넣지 않았음.
# 하나의 쟁점과 여러 판례가 연결될 수 있으므로, 아래는 '대표 정답 판례' 기반의 초기 시험셋임.
TESTS = [
    ("T01", "신탁된 건물은 등기 후에도 위탁자가 소유자인가요? 수탁자에게 소유권이 완전히 넘어가나요?", "2012다33174", "소유권·신탁"),
    ("T02", "건물을 사면서 기존 임차인의 보증금 반환채무를 인수하고 매매대금에서 공제했으면 매도인은 자동으로 책임을 벗어나나요?", "2012다84370", "보증금 채무인수"),
    ("T03", "입주하고 나서 심한 결로와 곰팡이가 발생했습니다. 집주인의 수선의무와 세입자의 하자 통지의무가 궁금합니다.", "2014나13609", "임대인 수선의무"),
    ("T04", "공인중개사가 집주인에게 월세계약만 위임받고 마음대로 전세계약을 맺어 보증금을 빼돌린 경우 중개사와 공제협회가 책임지나요?", "2009가합3360", "대리권·공제책임"),
    ("T05", "다가구주택 전세계약 중개 당시 다른 세입자들의 선순위 보증금 자료를 집주인이 주지 않았다면 중개사는 어떻게 설명해야 하나요?", "2020가단5104753", "선순위 보증금·책임"),
    ("T06", "임대인이 자료 제출을 거부한 사실과 선순위 세입자가 많다는 설명을 확인설명서에 쓰고 세입자도 들었다고 서명했다면 중개사 책임은 어떻게 판단하나요?", "2023나72693", "중개사 책임 부정 사례"),
    ("T07", "집주인이 아닌 매수인에게서 집을 임차했는데 나중에 원래 매매계약이 해제되면 전입한 임차인은 임차권을 새 소유자에게 주장할 수 있나요?", "2023다201218, 201225", "임대권한·대항력"),
    ("T08", "신탁등기된 부동산을 중개하면서 공인중개사가 신탁원부를 보여주지 않고 수탁자 승낙이 필요하다는 점도 설명하지 않았다면 책임이 있나요?", "2023다224327", "신탁물건 중개"),
    ("T09", "세입자가 계약갱신요구를 한 뒤 갱신된 임대차 시작 전에 해지 통보를 했으면 해지 효력의 3개월은 언제부터 계산하나요?", "2023다258672", "갱신 후 해지"),
    ("T10", "계약갱신을 한 임차인은 새 계약기간이 시작되기 전에는 해지 통보를 할 수 없다고 약정해도 되는 걸까요?", "2023다258672", "계약조항형·갱신"),
    ("T11", "다가구주택을 중개할 때 공인중개사는 다른 임차인의 보증금과 임대차 기간을 구체적으로 확인하고 설명해야 하나요?", "2023다259743", "다가구 중개 설명의무"),
    ("T12", "중개인은 근저당만 알려주고 다른 세입자들의 선순위 임차보증금 규모는 확인하지 않아도 된다는 계약 조항이 있습니다.", "2023다259743", "계약조항형·설명의무"),
    ("T13", "세입자가 임차권등기명령 신청비용을 집주인에게 청구할 때 다른 채무와 상계하는 방식으로도 받을 수 있나요?", "2024다221455", "임차권등기 비용"),
    ("T14", "다세대주택 여러 호실에 공동근저당권이 설정되어 있으면 중개사는 계약하려는 호실 외에 다른 호실의 보증금도 확인해줘야 하나요?", "2024다305087", "공동근저당·중개"),
    ("T15", "여러 주택이 공동담보로 잡혀 있어도 중개사는 내가 빌리는 호실의 등기부만 설명하면 된다는 특약이 있습니다.", "2024다305087", "계약조항형·공동담보"),
    ("T16", "임차인이 전입신고를 하고 살다가 먼저 이사하고 나중에 임차권등기를 마쳤다면 이전 대항력이 소급해서 회복되나요?", "2024다326398", "대항력·점유상실"),
    ("T17", "임대관리회사가 세입자 보증금으로 돌려막기하면서 그 사정을 숨긴 채 보증금 반환 책임을 지겠다고 계약했다면 형사상 사기가 될 수 있나요?", "2025도4777", "형사·보증금 사기"),
    ("T18", "주택을 1년만 빌리기로 했는데 계약이 묵시적으로 갱신됐습니다. 짧은 기간 약정을 이유로 또 새로운 갱신을 주장할 수 있나요?", "96다5551, 5568", "단기계약·묵시갱신"),
    ("T19", "임대인이 새 세입자를 구해 받은 돈으로 보증금을 돌려주겠다고 합의했는데 새 사람이 안 구해지면 끝없이 안 돌려줘도 되나요?", "97나3537", "보증금 반환 유예"),
    ("T20", "새로운 임차인이 들어올 때까지 기존 임차인의 보증금은 반환하지 않는다는 약정을 했습니다. 반환을 무기한 미룰 수 있나요?", "97나3537", "계약조항형·보증금"),
]


def top_cases(scores, rows, k):
    """실습용 03_search_cases.py와 동일하게 동일 사건은 한 번만 제시."""
    seen = set()
    result = []
    for ix in np.argsort(scores)[::-1]:
        row = rows[int(ix)]
        case = row["case_no"]
        if case in seen:
            continue
        seen.add(case)
        result.append((case, float(scores[int(ix)]), row["record_id"]))
        if len(result) >= k:
            break
    return result


def main():
    ap = argparse.ArgumentParser(description="판례 RAG Hit@1 / Hit@3 / MRR@3 평가")
    ap.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    ap.add_argument("--npz", type=Path, default=DEFAULT_NPZ)
    ap.add_argument("--batch-size", type=int, default=4)
    args = ap.parse_args()
    for p in [args.csv, args.npz]:
        if not p.is_file():
            raise SystemExit(f"파일 없음: {p}")

    with args.csv.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit("CSV가 비어 있음")

    with np.load(args.npz, allow_pickle=False) as z:
        vectors = z["vectors"].astype(np.float32)
        ids = z["record_ids"].tolist()
        model_name = str(z["model_name"].item())
        checksum = str(z["csv_sha256"].item())

    actual_checksum = hashlib.sha256(args.csv.read_bytes()).hexdigest()
    if checksum != actual_checksum:
        raise SystemExit("CSV와 임베딩의 SHA256이 달라요. 임베딩을 만들 때 사용한 원본 CSV를 확인해줘.")
    if ids != [row["record_id"] for row in rows]:
        raise SystemExit("임베딩의 record_id 순서가 CSV와 달라요.")
    if len(rows) != vectors.shape[0]:
        raise SystemExit("CSV의 청크 수와 임베딩 벡터 수가 달라요.")
    case_set = {r["case_no"] for r in rows}
    missing = sorted({t[2] for t in TESTS} - case_set)
    if missing:
        raise SystemExit("정답으로 지정한 판례가 현재 CSV에 없어요: " + ", ".join(missing))

    print(f"데이터 확인 완료: {len(case_set)}개 판례, {len(rows)}개 청크, 벡터 차원 {vectors.shape[1]}")
    print(f"임베딩 모델: {model_name} / 평가 질문: {len(TESTS)}개")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    query_vectors = model.encode(
        [t[1] for t in TESTS], batch_size=args.batch_size,
        normalize_embeddings=True, convert_to_numpy=True,
        show_progress_bar=True,
    ).astype(np.float32)
    if query_vectors.shape[1] != vectors.shape[1]:
        raise SystemExit("질문 벡터 차원과 저장된 판례 벡터 차원이 달라요.")

    scores_matrix = vectors @ query_vectors.T
    output = []
    for j, (qid, query, gold, category) in enumerate(TESTS):
        ranked = top_cases(scores_matrix[:, j], rows, 3)
        names = [x[0] for x in ranked]
        rank = names.index(gold) + 1 if gold in names else None
        item = {
            "test_id": qid, "category": category, "query": query,
            "expected_case_no": gold, "hit_at_1": int(rank == 1),
            "hit_at_3": int(rank is not None), "rank_in_top3": rank or "미검색",
        }
        for i in range(3):
            case, sim, rid = ranked[i] if i < len(ranked) else ("", "", "")
            item[f"top{i+1}_case_no"] = case
            item[f"top{i+1}_score"] = round(sim, 4) if isinstance(sim, float) else ""
            item[f"top{i+1}_record_id"] = rid
        output.append(item)

    n = len(output)
    h1 = sum(x["hit_at_1"] for x in output)
    h3 = sum(x["hit_at_3"] for x in output)
    mrr3 = sum(1 / x["rank_in_top3"] if isinstance(x["rank_in_top3"], int) else 0 for x in output) / n
    summary = {
        "evaluation_type": "seed_case_level_retrieval",
        "n_queries": n, "n_cases": len(case_set), "n_chunks": len(rows),
        "hit_at_1": round(h1 / n, 4), "hit_at_3": round(h3 / n, 4),
        "mrr_at_3": round(mrr3, 4),
        "hit_at_1_count": h1, "hit_at_3_count": h3,
        "model": model_name, "csv_sha256": checksum,
        "caveat": "질문은 현재 판례 데이터에서 구성한 초기 테스트. 실제 서비스의 일반 정확도나 법률적 타당성을 의미하지 않음",
    }

    out_csv = DATA / "retrieval_eval_results.csv"
    out_json = DATA / "retrieval_eval_summary.json"
    out_md = DATA / "retrieval_eval_report.md"
    with out_csv.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(output[0]))
        w.writeheader()
        w.writerows(output)
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    report = [
        "# ContractCheck 판례 검색 적중률 초기 평가",
        "",
        f"- 데이터: {len(case_set)}개 사건 / {len(rows)}개 청크 / 질문 {n}개",
        f"- **Hit@1: {h1}/{n} = {h1/n:.1%}**",
        f"- **Hit@3: {h3}/{n} = {h3/n:.1%}**",
        f"- **MRR@3: {mrr3:.3f}**",
        "",
        "기준: 질문별 대표 정답 사건이 상위 1/3개 '사건'에 포함되는지 확인. 같은 사건의 여러 청크는 한 번만 계산.",
        "테스트 질문은 현 보유 판례를 참고해 만들었으므로 독립적인 일반화 성능평가가 아닌 1차 점검용임.",
        "",
        "|번호|대표 정답|Top1|Top2|Top3|정답 순위|",
        "|---|---|---|---|---|---|",
    ]
    for x in output:
        report.append(f"|{x['test_id']}|{x['expected_case_no']}|{x['top1_case_no']}|{x['top2_case_no']}|{x['top3_case_no']}|{x['rank_in_top3']}|")
    report += ["", "## Top3에 정답이 없는 질문", ""]
    misses = [x for x in output if not x["hit_at_3"]]
    if not misses:
        report.append("- 없음")
    for x in misses:
        report.append(f"- **{x['test_id']}** {x['query']}")
        report.append(f"  - 정답: {x['expected_case_no']}; 반환: {x['top1_case_no']}, {x['top2_case_no']}, {x['top3_case_no']}")
    report += ["", "## 해석 주의", "", "- 코사인 유사도는 법적 위험 확률이 아님", "- Top3 적중률은 질문의 대표 정답 사건 검색 여부만 측정하며 판례 법리의 정확한 적용까지 확인하지 않음", "- 질문과 유사한 다른 판례도 적절할 수 있으므로 미검색 사례는 사람이 확인 필요", "- 실제 계약조항을 통한 별도 평가와 검색 실패 사례 분석이 다음 단계", ""]
    out_md.write_text("\n".join(report), encoding="utf-8")

    print("\n========== 1차 검색 정확도 결과 ==========")
    print(f"Hit@1 : {h1}/{n} = {h1/n:.1%} (첫 번째 사건이 정답)")
    print(f"Hit@3 : {h3}/{n} = {h3/n:.1%} (상위 세 사건 안에 정답)")
    print(f"MRR@3 : {mrr3:.3f} (정답 순위까지 반영)")
    print(f"Top3 미적중 질문: {', '.join(x['test_id'] for x in misses) if misses else '없음'}")
    print("\n저장한 파일:")
    for path in (out_csv, out_json, out_md):
        print("-", path)
    print("\n주의: 본 평가는 검색 적중률 점검이지, 법률 판단의 정확성·실서비스 품질을 확정하는 테스트는 아님")

if __name__ == "__main__":
    main()
