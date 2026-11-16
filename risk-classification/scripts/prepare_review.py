"""Create blank decisions and original/DRAFT comparison; never overwrite review work."""
import csv
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ml.data_contract import ROOT, read_rows, row_digest
from scripts.export_approved import QUEUE, DECISIONS

FIELDS = ['record_id', 'source_row_sha256', 'original_text', 'draft_text', 'context_note',
          'source_type', 'domain', 'group_id', 'source_url', 'proposed_label',
          'reviewed_text', 'reviewed_text_sha256', 'source_locator', 'source_snapshot_sha256',
          'source_verified', 'context_verified', 'reviewer_1', 'reviewer_1_label',
          'reviewer_2', 'reviewer_2_label', 'legal_basis', 'applicability_note',
          'reviewed_at', 'review_status', 'approved_label', 'hold_reason']


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--queue', type=Path, default=QUEUE)
    p.add_argument('--out', type=Path, default=DECISIONS)
    a = p.parse_args()
    if not a.queue.is_file():
        p.error('Queue CSV not found; copy the header template and add authorized local records')
    draft = {r['record_id']: r for r in read_rows(ROOT/'data/review/risk_clause_review_queue_context_DRAFT.csv')}
    context = {r['record_id']: r for r in read_rows(ROOT/'reports/standard_clauses_context_review_20261110.csv')}
    rows = []
    for r in read_rows(a.queue):
        y = {k: r.get(k, '') for k in FIELDS[:10]}
        y.update(source_row_sha256=row_digest(r), original_text=r['text'],
                 draft_text=draft.get(r['record_id'], {}).get('text', ''),
                 context_note=context.get(r['record_id'], {}).get('related_context_note', ''),
                 review_status='needs_review')
        rows.append(y)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open('x', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS); w.writeheader(); w.writerows(rows)
    print(f'검수표 생성: {a.out}; 라벨/검수 증거는 모두 빈 상태')


if __name__ == '__main__':
    main()
