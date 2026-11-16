"""Shared fail-closed provenance and leakage checks; no automatic legal labels."""
import csv
import hashlib
import json
import re
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APPROVED = ROOT / 'data/approved/reviewed_clauses.csv'
REAL_TYPES = {'official_standard_clause_excerpt', 'verified_real_contract_clause'}
SYNTHETIC = 'synthetic_contract_style_draft'
COLUMNS = ['record_id', 'text', 'approved_label', 'source_type', 'domain', 'group_id',
           'source_url', 'parent_record_id', 'evaluation_policy']


def read_rows(path):
    if not Path(path).exists():
        return []
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def row_digest(row):
    return hashlib.sha256(json.dumps(row, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def normalize(text):
    return re.sub(r'\s+', '', unicodedata.normalize('NFKC', text).casefold())


def validate_rows(rows):
    errors, ids, texts, urls = [], {}, {}, {}
    for r in rows:
        rid = r.get('record_id', '')
        for key in COLUMNS:
            if key not in r or (key != 'parent_record_id' and not r[key].strip()):
                errors.append(f'{rid}: missing {key}')
        if rid in ids:
            errors.append(f'{rid}: duplicate record_id')
        ids[rid] = r
        if r.get('approved_label') not in {'0', '1'}:
            errors.append(f'{rid}: invalid approved_label')
        if r.get('domain') not in {'residential_rental', 'registered_private_rental'}:
            errors.append(f'{rid}: unsupported domain; resolve team scope before export')
        typ = r.get('source_type')
        if typ not in REAL_TYPES | {SYNTHETIC}:
            errors.append(f'{rid}: summaries/non-contract sources forbidden')
        if r.get('evaluation_policy') != ('train_only' if typ == SYNTHETIC else 'grouped_source_only'):
            errors.append(f'{rid}: invalid evaluation_policy')
        key = normalize(r.get('text', ''))
        if key in texts:
            errors.append(f'{rid}: duplicate text with {texts[key]} (human review required)')
        texts[key] = rid
        if typ in REAL_TYPES:
            url, group = r.get('source_url', '').strip(), r.get('group_id')
            if url in urls and urls[url] != group:
                errors.append(f'{rid}: same source_url assigned multiple groups')
            urls[url] = group
    for r in rows:
        parent = r.get('parent_record_id')
        if r.get('source_type') == SYNTHETIC or parent:
            p = ids.get(parent)
            if not p or p.get('source_type') not in REAL_TYPES:
                errors.append(f"{r.get('record_id')}: parent must be an approved real clause")
            elif p['group_id'] != r.get('group_id') or p['domain'] != r.get('domain'):
                errors.append(f"{r['record_id']}: parent group/domain mismatch")
    return errors


def cross_split_errors(train, validation):
    errors = []
    if {r['group_id'] for r in train} & {r['group_id'] for r in validation}:
        errors.append('source group leakage')
    if any(r['source_type'] not in REAL_TYPES for r in validation):
        errors.append('validation must contain real clauses only')
    for a in train:
        for b in validation:
            if SequenceMatcher(None, normalize(a['text']), normalize(b['text']), autojunk=False).ratio() >= .90:
                errors.append(f"near-duplicate leakage requires review: {a['record_id']} / {b['record_id']}")
    return errors


def approved_errors(path=APPROVED):
    path = Path(path)
    manifest_path = path.with_suffix('.manifest.json')
    if not path.exists() or not manifest_path.exists():
        return ['approved export/manifest missing; complete human review and export']
    try:
        m = json.loads(manifest_path.read_text(encoding='utf-8'))
        if m['approved_sha256'] != digest(path):
            return ['approved export changed after approval export']
        for source, expected in m['inputs'].items():
            if digest(source) != expected:
                return ['review inputs changed; re-export and re-split']
    except (OSError, ValueError, KeyError):
        return ['invalid approval provenance manifest']
    return validate_rows(read_rows(path))


def preflight_files(train_path, valid_path, approved=APPROVED):
    tr, va = read_rows(train_path), read_rows(valid_path)
    errors = approved_errors(approved) + validate_rows(tr + va)
    approved_map = {r['record_id']: r for r in read_rows(approved)}
    if any(r != approved_map.get(r.get('record_id')) for r in tr + va):
        errors.append('split row does not match current approved export')
    try:
        m = json.loads((Path(train_path).parent/'manifest.json').read_text(encoding='utf-8'))
        for path in (Path(approved), Path(train_path), Path(valid_path)):
            if m['sha256'][path.name] != digest(path):
                errors.append('stale split fingerprint')
    except (OSError, ValueError, KeyError):
        errors.append('split manifest missing or invalid')
    if len(tr) < 10 or len(va) < 5:
        errors.append('minimum train=10 / validation=5 not met')
    for name, rows in [('train', tr), ('validation', va)]:
        if {r.get('approved_label') for r in rows} != {'0', '1'}:
            errors.append(f'{name}: both classes required')
        actual = [r for r in rows if r.get('source_type') == 'verified_real_contract_clause']
        if {r.get('approved_label') for r in actual} != {'0', '1'}:
            errors.append(f'{name}: both labels of verified actual contracts required')
    if not validate_rows(tr + va):
        errors += cross_split_errors(tr, va)
    return tr, va, errors
