"""판례 청크를 BGE-M3 벡터로 변환하여 별도 .npz 파일에 보관한다."""
import argparse
import csv
import hashlib
from pathlib import Path
import numpy as np

BASE=Path(__file__).resolve().parents[1]
CSV=BASE/'data/processed/precedent_chunks_reviewed_with_urls.csv'
OUT=BASE/'data/processed/precedent_embeddings_reviewed.npz'

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--model',default='BAAI/bge-m3',help='예: BAAI/bge-m3')
    ap.add_argument('--batch-size',type=int,default=4)
    args=ap.parse_args()
    if not CSV.exists():raise SystemExit('먼저 01_prepare_cases.py를 실행하세요.')
    with CSV.open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    if not rows:raise SystemExit('청크가 없습니다. PDF 및 manual_review.csv를 점검하세요.')
    from sentence_transformers import SentenceTransformer
    print('임베딩 모델 불러오는 중:',args.model)
    model=SentenceTransformer(args.model)
    vectors=model.encode([r['text'] for r in rows],batch_size=args.batch_size,
                         normalize_embeddings=True,show_progress_bar=True,
                         convert_to_numpy=True).astype('float32')
    checksum=hashlib.sha256(CSV.read_bytes()).hexdigest()
    np.savez_compressed(OUT,vectors=vectors,
                        record_ids=np.array([r['record_id'] for r in rows]),
                        model_name=np.array(args.model),csv_sha256=np.array(checksum))
    print('완료:',OUT,'벡터 크기:',vectors.shape)
    print('아직 pgvector DB에 저장한 것은 아닙니다.')

if __name__=='__main__':main()

