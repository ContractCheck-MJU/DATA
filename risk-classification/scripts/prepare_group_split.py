#!/usr/bin/env python3
"""Only human-approved true contract clauses may enter validation.

Zero-approved rows are a normal state; prints a blocking reason and writes no split.
Split at source group level FIRST; synthetic examples can go only to training.
"""
from __future__ import annotations
import csv
import itertools
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ml.data_contract import (APPROVED, validate_rows, approved_errors, cross_split_errors, digest)

ROOT = Path(__file__).resolve().parents[1]
INPUT = APPROVED
SPLITS = ROOT / 'data/splits'
REPORT = ROOT / 'reports/split_readiness.json'
REAL_TYPES = {'official_standard_clause_excerpt', 'verified_real_contract_clause'}
TRAIN_ONLY_TYPES = {'synthetic_contract_style_draft'}
COLUMNS = ['record_id','text','approved_label','source_type','domain','group_id','source_url','parent_record_id','evaluation_policy']


def read_rows(path):
    if not path.exists():
        return []
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def select_groups(reals):
    # This function requires minimum independent data; it never splits the same source across train/val.
    groups = sorted({r['group_id'] for r in reals})
    if not reals:
        return None, '승인된 실제 계약서 문장 0개: 라벨 검수 및 팀 승인 전에는 학습 불가'
    if len(groups) < 4:
        return None, '독립적인 실제 계약서 출처 그룹 4개 이상 필요'
    counts = Counter(r['approved_label'] for r in reals)
    if min(counts.get('0', 0), counts.get('1', 0)) < 5 or len(reals) < 20:
        return None, '각 라벨 5개 이상이면서 실제 문장 총 20개 이상 필요'
    by_group = defaultdict(list)
    for r in reals:
        by_group[r['group_id']].append(r)
    # both labels must occur in at least two independent groups, otherwise not generalizable
    for label in ('0','1'):
        if sum(any(r['approved_label'] == label for r in rr) for rr in by_group.values()) < 2:
            return None, f'라벨 {label}을 포함하는 독립 출처가 2개 이상 필요'
    candidates = []
    # combinations capped to keep the preparation command lightweight
    if len(groups) <= 15:
        subsets = itertools.chain.from_iterable(itertools.combinations(groups, n) for n in range(1, len(groups)-1))
    else:
        rng = random.Random(42)
        subsets = (tuple(sorted(rng.sample(groups, max(1, round(len(groups)*.25))))) for _ in range(2000))
    for subset in subsets:
            n = len(subset)
            vg = set(subset)
            valid = [r for r in reals if r['group_id'] in vg]
            train = [r for r in reals if r['group_id'] not in vg]
            if set(r['approved_label'] for r in valid) != {'0','1'}:
                continue
            if set(r['approved_label'] for r in train) != {'0','1'}:
                continue
            if len(valid) < 5 or len(train) < 10:
                continue
            # Lightweight unit fixtures omit source_type; production rows never do.
            if reals and 'source_type' in reals[0]:
                if any({r['approved_label'] for r in split if r['source_type']=='verified_real_contract_clause'} != {'0','1'} for split in (train, valid)):
                    continue
            val_rate = len(valid)/len(reals)
            candidates.append((abs(val_rate - .25), n, tuple(sorted(subset))))
    if not candidates:
        return None, '그룹을 분리하면 학습/검증 양쪽에 정상·위험 문장을 확보할 수 없음'
    vg = set(min(candidates)[2])
    return vg, ''


def readiness():
    SPLITS.mkdir(parents=True, exist_ok=True)
    data = read_rows(INPUT)
    errors = approved_errors(INPUT)
    real = [r for r in data if r['source_type'] in REAL_TYPES]
    synthetic = [r for r in data if r['source_type'] in TRAIN_ONLY_TYPES]
    val_groups, block = select_groups(real)
    train = [r for r in real if r['group_id'] not in (val_groups or set())]
    valid = [r for r in real if r['group_id'] in (val_groups or set())]
    train.extend(r for r in synthetic if r['group_id'] not in (val_groups or set()))
    if val_groups is not None and not errors:
        errors.extend(cross_split_errors(train, valid))
    if errors:
        block = '잘못 승인된 데이터가 있음: '+ '; '.join(errors[:4])
    info = {'approved_rows': len(data), 'real_clause_rows': len(real), 'synthetic_train_only_rows': len(synthetic),
            'unique_real_source_groups': len({r['group_id'] for r in real}),
            'status': 'READY' if val_groups is not None and not errors else 'BLOCKED',
            'reason': block, 'validation_group_ids': sorted(val_groups or []),
            'policy': 'Real source-group split; human-approved contract clause excerpts only; no summaries in evaluation; synthetic only train.'}
    if val_groups is not None and not errors:
        train = [r for r in real if r['group_id'] not in val_groups]
        valid = [r for r in real if r['group_id'] in val_groups]
        # If synthetic and its originating source is validated, never add it to train.
        train.extend(r for r in synthetic if r['group_id'] not in val_groups)
        assert not set(r['group_id'] for r in valid) & set(r['group_id'] for r in train)
        for filename, rows in [('train.csv',train),('validation.csv',valid)]:
            with (SPLITS/filename).open('w',encoding='utf-8-sig',newline='') as f:
                w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction='ignore');w.writeheader();w.writerows(rows)
        info['train_rows'] = len(train); info['validation_rows'] = len(valid)
        (SPLITS/'manifest.json').write_text(json.dumps({
            'sha256': {p.name: digest(p) for p in (INPUT, SPLITS/'train.csv', SPLITS/'validation.csv')},
            'seed': 42, 'evaluation': 'development_validation_not_independent_test',
            'excluded_synthetic_ids': [r['record_id'] for r in synthetic if r['group_id'] in val_groups]
        }, indent=2), encoding='utf-8')
    else:
        # Prevent reuse of stale splits from an earlier run.
        for file in (SPLITS/'train.csv', SPLITS/'validation.csv', SPLITS/'manifest.json'):
            if file.exists(): file.unlink()
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    with REPORT.open('w',encoding='utf-8') as f:
        json.dump(info,f,ensure_ascii=False,indent=2)
    print(json.dumps(info,ensure_ascii=False,indent=2))
    return info


if __name__ == '__main__':
    raise SystemExit(0 if readiness()['status']=='READY' else 2)
