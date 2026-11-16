#!/usr/bin/env python3
"""Technical-only audit. It NEVER changes labels or approval status.

Run from anywhere: python scripts/audit_data_solo.py
Reports are reproducible from the existing review queue and original dataset.
No web requests; source PDFs were checked separately and recorded in a human-readable table.
"""
from __future__ import annotations
import csv
import json
import re
import difflib
from collections import Counter
from datetime import date
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / 'data/review/risk_clause_review_queue.csv'
LEGACY = ROOT / 'data/raw/clause_classification_original.csv'
TECH = ROOT / 'data/review/technical_source_review.csv'
OUT = ROOT / 'reports'


def read_csv(path):
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        wr = csv.DictWriter(f, fieldnames=fieldnames)
        wr.writeheader()
        wr.writerows(rows)


def normalize(text):
    return re.sub(r'[^가-힣a-zA-Z0-9]', '', text.casefold())


def compare_text(a, b):
    return difflib.SequenceMatcher(None, normalize(a), normalize(b), autojunk=False).ratio()


def audit():
    q = read_csv(REVIEW)
    legacy = read_csv(LEGACY)
    tech = read_csv(TECH)
    ids = [r['record_id'] for r in q]
    if len(ids) != len(set(ids)):
        raise ValueError('record_id가 중복되어 있습니다.')
    assert len(q) == len(tech) == 38
    assert {r['record_id'] for r in tech} == set(ids)
    # Only metadata and text checks: does NOT imply legal approval.
    invalid = []
    for r in q:
        if not r['text'].strip() or not r['group_id'].strip():
            invalid.append((r['record_id'], 'missing_text_or_group'))
        if r['source_url'] and (urlsplit(r['source_url']).scheme not in {'https', 'http'} or not urlsplit(r['source_url']).netloc):
            invalid.append((r['record_id'], 'invalid_url_syntax'))
    exact_internal = []
    exact_legacy = []
    similarities = []
    norm_original = {}
    for old in legacy:
        norm_original.setdefault(normalize(old['text']), []).append(old['sample_id'])
    normalized = {r['record_id']: normalize(r['text']) for r in q}
    for r in q:
        for legacy_id in norm_original.get(normalized[r['record_id']], []):
            exact_legacy.append({'review_id': r['record_id'], 'original_sample_id': legacy_id})
    for i, a in enumerate(q):
        for b in q[i+1:]:
            x, y = a['record_id'], b['record_id']
            if normalized[x] == normalized[y]:
                exact_internal.append((x, y))
            score = compare_text(a['text'], b['text'])
            if score >= 0.56:
                similarities.append({'review_id_a': x, 'review_id_b': y,
                                     'text_similarity': f'{score:.4f}',
                                     'same_source_group': 'yes' if a['group_id'] == b['group_id'] else 'no',
                                     'handling': 'manual_near_duplicate_review_only'})
    similarities.sort(key=lambda r: -float(r['text_similarity']))
    write_csv(OUT / 'possible_near_duplicates.csv',
              ['review_id_a', 'review_id_b', 'text_similarity', 'same_source_group', 'handling'], similarities)
    write_csv(OUT / 'existing_exact_duplicates.csv', ['review_id', 'original_sample_id'], exact_legacy)
    counts = Counter(r['source_type'] for r in q)
    groups = Counter(r['group_id'] for r in q)
    tech_status = Counter(r['technical_text_status'] for r in tech)
    info = {
        'run_date': str(date.today()),
        'review_queue_count': len(q),
        'legacy_sample_count': len(legacy),
        'source_category_counts': dict(counts),
        'source_group_counts': dict(groups),
        'source_group_total': len(groups),
        'exact_duplicates_within_review': len(exact_internal),
        'exact_duplicates_with_original_samples': len(exact_legacy),
        'near_duplicate_pairs_threshold_0_56_including_exact': len(similarities),
        'technical_text_status_counts': dict(tech_status),
        'approved_new_samples': sum(r['review_status'] == 'approved' and r['approved_label'] in ('0','1') for r in q),
        'invalid_fields': invalid,
        'IMPORTANT': 'Technical checks only. Verified URL/PDF text does not mean legal risk label is approved. Near-duplicate similarity is only a review hint.',
    }
    with (OUT / 'solo_audit_summary.json').open('w', encoding='utf-8') as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    print(json.dumps(info, ensure_ascii=False, indent=2))
    return info


if __name__ == '__main__':
    audit()
