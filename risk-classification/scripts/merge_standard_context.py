"""Apply *technical* source-text proposals to a DRAFT review queue. No label approval.
Run from the project root: .\\.venv\\Scripts\\python.exe scripts\\merge_standard_context.py
"""
from pathlib import Path
import csv

project=Path(__file__).resolve().parents[1]
queue=project/'data'/'review'/'risk_clause_review_queue.csv'
patch=project/'reports'/'standard_clauses_context_review_20261110.csv'
dest=project/'data'/'review'/'risk_clause_review_queue_context_DRAFT.csv'

if not queue.exists(): raise SystemExit(f'원본 검수 대기열 없음: {queue}')
if not patch.exists(): raise SystemExit(f'문구 검토 CSV 없음: {patch}')

def read_csv(path):
    with path.open('r',encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f); return list(reader),reader.fieldnames

queue_rows,headers=read_csv(queue)
patch_rows,_=read_csv(patch)
if len(patch_rows)!=17: raise SystemExit('보완 후보 행 수가 17개가 아닙니다.')
if len(queue_rows)!=38: raise SystemExit('원본 검수 대기열 행 수가 38개가 아닙니다. 원본을 점검하세요.')
patch_map={r['record_id']:r for r in patch_rows}
if len(patch_map)!=17: raise SystemExit('중복 record_id가 있습니다.')
changed=0
for row in queue_rows:
    id=row['record_id']
    if id in patch_map:
        p=patch_map[id]
        if row['text'] != p['original_text']:
            raise SystemExit(f'원본과 비교 CSV가 불일치합니다: {id}. 수정본 생성 중단.')
        if row.get('approved_label','').strip():
            raise SystemExit(f'이미 라벨 승인된 행은 자동 수정하지 않습니다: {id}')
        if p['suggested_text'] != row['text']:
            row['text']=p['suggested_text']
            changed+=1
        old=row.get('review_comment','')
        row['review_comment']=(old+' | ' if old else '')+'[자동 작성 기술 수정안/미승인] '+p['technical_observation']
        row['review_status']='needs_review'

# Keep original header and all 38 rows.
with dest.open('x',encoding='utf-8-sig',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=headers,extrasaction='ignore')
    writer.writeheader();writer.writerows(queue_rows)
print(f'기술 검토용 초안 생성: {dest}')
print(f'원본 38개 유지, text 수정 제안 반영 {changed}개, approved_label 변경 0개')
print('주의: 원본 risk_clause_review_queue.csv 파일은 변경하지 않았습니다.')
