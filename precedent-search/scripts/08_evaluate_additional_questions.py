"""새 질문 15개에 대해 baseline vs corpus centering 평가. 기존 20개 및 API 보존.
실행: python precedent-search/scripts/08_evaluate_additional_questions.py
"""
from __future__ import annotations
import csv, hashlib, json
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data'/'processed'
TEST_CSV=ROOT/'data'/'evaluation'/'precedent_additional_questions_v1.csv'
DOC_CSV=DATA/'precedent_chunks_reviewed_with_urls.csv'
NPZ=DATA/'precedent_embeddings_reviewed.npz'
OUT_CSV=DATA/'additional_evaluation_details.csv'
OUT_JSON=DATA/'additional_evaluation_summary.json'

def unit_rows(a):
    a=np.asarray(a,dtype=np.float64)
    norms=np.linalg.norm(a,axis=1,keepdims=True)
    if np.any(norms < 1e-12):
        raise ValueError('보정 후 길이가 0인 벡터 발생')
    return a/norms

def case_rank(doc_scores, docs):
    # 원래 검색 API와 동일하게, 중복된 판례는 최고 유사 청크 한 개로 대표
    found=[]; seen=set()
    for idx in np.argsort(doc_scores)[::-1]:
        row=docs[int(idx)]; sid=row['source_id']
        if sid in seen: continue
        seen.add(sid)
        found.append((row['case_no'], float(doc_scores[int(idx)])))
    return found

def evaluate(method, dvec, qvec, docs, tests):
    scores=qvec @ dvec.T
    details=[]; ranks=[]
    for j,t in enumerate(tests):
        ranked=case_rank(scores[j],docs)
        cases=[c for c,_ in ranked]
        rank=cases.index(t['expected_case_no'])+1
        ranks.append(rank)
        details.append({
          'method':method, 'test_id':t['test_id'], 'query':t['query'],
          'expected_case_no':t['expected_case_no'], 'gold_review_status':t['gold_review_status'],
          'rank':rank, 'top1':cases[0] if len(cases)>0 else '',
          'top2':cases[1] if len(cases)>1 else '',
          'top3':cases[2] if len(cases)>2 else '',
          'hit_at_1':int(rank==1), 'hit_at_3':int(rank<=3)
        })
    n=len(tests)
    summary={'method':method,'n_questions':n,'hit_at_1':sum(r==1 for r in ranks)/n,
         'hit_at_3':sum(r<=3 for r in ranks)/n,
         'mrr_at_3':sum(1/r if r<=3 else 0 for r in ranks)/n,
         'hit1_count':sum(r==1 for r in ranks), 'hit3_count':sum(r<=3 for r in ranks)}
    return summary,details

def main():
    for p in (TEST_CSV,DOC_CSV,NPZ):
        if not p.is_file(): raise SystemExit('필요한 파일이 없습니다: '+str(p))
    with TEST_CSV.open('r',encoding='utf-8-sig',newline='') as f:
        tests=list(csv.DictReader(f))
    with DOC_CSV.open('r',encoding='utf-8-sig',newline='') as f:
        docs=list(csv.DictReader(f))
    if len(tests)!=15 or len({t['test_id'] for t in tests})!=len(tests):
        raise SystemExit('추가 질문 파일은 15개 고유 질문이어야 합니다.')
    with np.load(NPZ,allow_pickle=False) as z:
        doc_vecs=np.asarray(z['vectors'],dtype=np.float64)
        record_ids=z['record_ids'].tolist()
        model_name=str(z['model_name'].item())
        checksum=str(z['csv_sha256'].item())
    if checksum!=hashlib.sha256(DOC_CSV.read_bytes()).hexdigest():
        raise SystemExit('CSV/NPZ 해시 불일치: 원본 판례 파일을 수정하지 마세요.')
    if record_ids!=[d['record_id'] for d in docs] or doc_vecs.shape!=(len(docs),1024):
        raise SystemExit('CSV와 임베딩의 ID·형상 불일치')
    cases={d['case_no'] for d in docs}
    missing={t['expected_case_no'] for t in tests}-cases
    if missing: raise SystemExit('검수용 정답 판례 누락: '+str(missing))
    # 사전에 저장한 대표 정답은 검색 결과로 재지정하지 않음
    print(f'판례 {len(cases)}건 / 청크 {len(docs)}개 / 신규 질문 {len(tests)}개')
    print('주의: 새 질문의 대표 정답은 데이터 기반 임시 라벨이며 법률 전문가 검수 전입니다.')
    print('모델 불러오는 중:',model_name)
    from sentence_transformers import SentenceTransformer
    model=SentenceTransformer(model_name)
    query_vecs=model.encode([t['query'] for t in tests],batch_size=4,
        normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=True).astype(np.float64)
    if query_vecs.shape!=(len(tests),doc_vecs.shape[1]):
        raise SystemExit('질문/문서 벡터 형상 불일치')

    # 실험 중 보정 통계는 문서 벡터만으로 계산. 질문에 동일한 평균을 적용.
    corpus_mean=doc_vecs.mean(axis=0,keepdims=True)
    variants=[('baseline_max',unit_rows(doc_vecs),unit_rows(query_vecs)),
              ('center_pc0',unit_rows(doc_vecs-corpus_mean),unit_rows(query_vecs-corpus_mean))]
    summaries={}; all_details=[]; by_method={}
    for name,dvec,qvec in variants:
        summary,details=evaluate(name,dvec,qvec,docs,tests)
        summaries[name]=summary; by_method[name]={r['test_id']:r for r in details}
        all_details+=details
    print('\n===== 새 질문 15개 평가: 기존 방식 vs 중심화 =====')
    print(f"{'방식':<16} {'Hit@1':>18} {'Hit@3':>18} {'MRR@3':>10}")
    for name,s in summaries.items():
        print(f"{name:<16} {s['hit1_count']:>2}/15 ({s['hit_at_1']:>5.1%})     {s['hit3_count']:>2}/15 ({s['hit_at_3']:>5.1%})     {s['mrr_at_3']:>8.3f}")
    print('\n===== 질문별 순위 변화 (정답 판례 순위) =====')
    improved=worsened=unchanged=0
    for t in tests:
        tid=t['test_id']; before=by_method['baseline_max'][tid]['rank']; after=by_method['center_pc0'][tid]['rank']
        if after<before: improved+=1
        elif after>before: worsened+=1
        else: unchanged+=1
        print(f'{tid}: {before:>2}위 -> {after:>2}위 | 기대: {t["expected_case_no"]}')
    print(f'개선 {improved} / 악화 {worsened} / 동일 {unchanged}')
    with OUT_CSV.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(all_details[0].keys()))
        writer.writeheader();writer.writerows(all_details)
    OUT_JSON.write_text(json.dumps({
        'dataset':'additional_case_paraphrase_v1',
        'gold_status':'provisional_needs_team_review',
        'n_cases':len(cases),'n_chunks':len(docs),'n_questions':len(tests),
        'model':model_name,'results':summaries,
        'rank_changes':{'improved':improved,'worsened':worsened,'unchanged':unchanged},
        'caveats':['기존 보유 판례를 근거로 수동 작성한 신규 문항. 독립 외부 검증셋이 아님',
                   '판례 대표 정답은 법률 전문가 검증 전',
                   '높은 Hit@k 수치는 법률적 적합성이나 실서비스 정확도가 아님',
                   'API 및 기존 임베딩은 변경하지 않음']},ensure_ascii=False,indent=2),encoding='utf-8')
    print('\n저장:');print('-',OUT_CSV);print('-',OUT_JSON)
    print('기존 임베딩, API 및 최초 20문항 평가 결과는 변경하지 않았습니다.')

if __name__=='__main__':
    main()
