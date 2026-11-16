"""Export a separate dataset from human decisions bound to exact source/text hashes."""
import argparse
import csv
import json
import sys
from datetime import date
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ml.data_contract import ROOT, APPROVED, COLUMNS, SYNTHETIC, read_rows, digest, row_digest, validate_rows

QUEUE = ROOT / 'data/review/risk_clause_review_queue.csv'
DECISIONS = ROOT / 'data/review/human_decisions.csv'


def export(queue, decisions, out):
    if not Path(queue).exists() or not Path(decisions).exists():
        raise ValueError('검수표가 없습니다. scripts/prepare_review.py 실행 후 사람이 검수하세요.')
    rows, votes = read_rows(queue), read_rows(decisions)
    mapping = {r['record_id']: r for r in rows}
    if len(mapping) != len(rows) or len({v['record_id'] for v in votes}) != len(votes):
        raise ValueError('duplicate record_id')
    approved = []
    for v in votes:
        if v.get('review_status') != 'approved':
            if v.get('approved_label', '').strip():
                raise ValueError(f"{v['record_id']}: non-approved row has a final label")
            continue
        r = mapping.get(v['record_id'])
        if not r or v.get('source_row_sha256') != row_digest(r):
            raise ValueError('source row changed; review again')
        required = ('reviewed_text', 'reviewer_1', 'reviewer_2', 'reviewed_at',
                    'legal_basis', 'applicability_note', 'source_locator', 'source_snapshot_sha256')
        if any(not v.get(k, '').strip() for k in required):
            raise ValueError(f"{r['record_id']}: review evidence incomplete")
        if v['reviewer_1'].strip() == v['reviewer_2'].strip():
            raise ValueError('two distinct reviewers required')
        date.fromisoformat(v['reviewed_at'])
        if v.get('source_verified') != 'yes' or v.get('context_verified') != 'yes':
            raise ValueError('source and context must be human verified')
        if v.get('reviewed_text_sha256') != row_digest({'text': v['reviewed_text']}):
            raise ValueError('reviewed text hash mismatch; confirm exact reviewed wording')
        if len(v['source_snapshot_sha256']) != 64 or any(c not in '0123456789abcdefABCDEF' for c in v['source_snapshot_sha256']):
            raise ValueError('source snapshot requires SHA256 hex')
        if not (v.get('reviewer_1_label') == v.get('reviewer_2_label') == v.get('approved_label') in {'0', '1'}):
            raise ValueError('reviewer disagreement or missing final label')
        y = {k: r.get(k, '') for k in COLUMNS}
        y.update(text=v['reviewed_text'], approved_label=v['approved_label'],
                 evaluation_policy='train_only' if r['source_type'] == SYNTHETIC else 'grouped_source_only')
        approved.append(y)
    errors = validate_rows(approved)
    if errors:
        raise ValueError('; '.join(errors))
    out = Path(out)
    if any(out.resolve().is_relative_to((ROOT/f'data/{folder}').resolve()) for folder in ('raw', 'review')):
        raise ValueError('raw/review directories are protected output destinations')
    if out.resolve() in {Path(queue).resolve(), Path(decisions).resolve(), (ROOT/'data/approved/approved_clauses.csv').resolve()}:
        raise ValueError('protected input path')
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS); w.writeheader(); w.writerows(approved)
    out.with_suffix('.manifest.json').write_text(json.dumps({
        'approved_sha256': digest(out), 'inputs': {str(Path(p).resolve()): digest(p) for p in (queue, decisions)},
        'approved_rows': len(approved), 'policy': 'human-reviewed-v1'
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    return approved


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--queue', type=Path, default=QUEUE)
    p.add_argument('--decisions', type=Path, default=DECISIONS)
    p.add_argument('--out', type=Path, default=APPROVED)
    a = p.parse_args()
    try:
        rows = export(a.queue, a.decisions, a.out)
    except ValueError as e:
        p.exit(2, f'BLOCKED: {e}\n')
    print(f'승인 {len(rows)}개; 출력 {a.out}; 원본 및 기존 승인 파일 보존')


if __name__ == '__main__':
    main()
