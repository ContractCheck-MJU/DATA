"""실습용 벡터 검색. 검색 점수는 법적 위험 확률이 아니다."""
import argparse,csv,hashlib
from pathlib import Path
import numpy as np

BASE=Path(__file__).resolve().parents[1]
CSV=BASE/'data/processed/precedent_chunks_reviewed_with_urls.csv'
NPZ=BASE/'data/processed/precedent_embeddings_reviewed.npz'

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--query',required=True,help='검색할 계약서 문구')
    ap.add_argument('--top-k',type=int,default=3)
    args=ap.parse_args()
    with CSV.open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    with np.load(NPZ,allow_pickle=False) as z:
        vectors=z['vectors']; ids=z['record_ids'].tolist()
        model_name=str(z['model_name'].item());checksum=str(z['csv_sha256'].item())
    if hashlib.sha256(CSV.read_bytes()).hexdigest()!=checksum:
        raise SystemExit('CSV가 임베딩 이후 변경되었습니다. 02_embed_cases.py를 다시 실행하세요.')
    assert [r['record_id'] for r in rows]==ids,'CSV/임베딩 record_id 불일치'
    from sentence_transformers import SentenceTransformer
    model=SentenceTransformer(model_name)
    query_vector=model.encode(args.query,normalize_embeddings=True,convert_to_numpy=True)
    scores=vectors @ query_vector
    # 동일 판례의 중복 청크는 결과창에서 한 번만 표시
    chosen=set();total=0
    for ix in np.argsort(scores)[::-1]:
        r=rows[int(ix)]
        if r['source_id'] in chosen:continue
        chosen.add(r['source_id']);total+=1
        print(f'\n{total}. {r["case_no"]} ({r["chunk_section"]}) | 코사인 유사도 {scores[ix]:.4f}')
        print('출처:',r['source_url'] or '(국가법령정보센터 원문 URL 미입력)')
        print('검수:',r['review_status'])
        print('근거 후보:',r['text'][:380])
        if total>=args.top_k:break
    print('\n참고: 높은 검색 유사도 != 판례 적용 가능성/법적 위법 확정')

if __name__=='__main__':main()

