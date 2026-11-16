"""Regression tests use temporary fictional fixtures, never production labels."""
import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from ml.data_contract import (COLUMNS, digest, row_digest, validate_rows, cross_split_errors,
                              preflight_files, approved_errors)
from scripts.export_approved import export
from scripts import prepare_group_split as splitting
from ml.compare_models import compare


def row(i=0, group='g0', label='0', **kwargs):
    r = dict(zip(COLUMNS, [f'r{i}', f'fixture clause {i}', label, 'verified_real_contract_clause',
                          'residential_rental', group, f'https://example.org/{group}', '', 'grouped_source_only']))
    r.update(kwargs)
    return r


def write_csv(path, rows, fields=None):
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields or list(rows[0])); w.writeheader(); w.writerows(rows)


def vote(r):
    return dict(record_id=r['record_id'], source_row_sha256=row_digest(r), review_status='approved',
                reviewed_text=r['text'], reviewed_text_sha256=row_digest({'text':r['text']}),
                approved_label=r['approved_label'], reviewer_1='fixture_a', reviewer_2='fixture_b',
                reviewer_1_label=r['approved_label'], reviewer_2_label=r['approved_label'],
                reviewed_at='2026-11-15', legal_basis='TEST ONLY', applicability_note='TEST ONLY',
                source_locator='fixture p1', source_snapshot_sha256='a'*64,
                source_verified='yes', context_verified='yes')


class ApprovalTests(unittest.TestCase):
    def test_export_and_stale_evidence(self):
        with tempfile.TemporaryDirectory() as d:
            q, v, out = [Path(d)/n for n in ('q.csv','v.csv','out.csv')]
            r = row(); decision = vote(r)
            write_csv(q, [r]); write_csv(v, [decision])
            self.assertEqual(len(export(q,v,out)), 1)
            self.assertFalse(approved_errors(out))
            decision['reviewed_text'] = 'changed after review'
            write_csv(v, [decision])
            self.assertTrue(approved_errors(out))
            with self.assertRaises(ValueError): export(q,v,out)

    def test_invalid_approvals_fail_closed(self):
        mutations = [{'approved_label':''}, {'reviewer_2':'fixture_a'}, {'source_verified':'no'},
                     {'context_verified':'no'}, {'reviewer_2_label':'1'}, {'legal_basis':''},
                     {'source_row_sha256':'wrong'}, {'source_snapshot_sha256':'wrong'}]
        for change in mutations:
            with self.subTest(change=change), tempfile.TemporaryDirectory() as d:
                q,v,out = [Path(d)/n for n in ('q.csv','v.csv','out.csv')]
                r=row(); decision=vote(r); decision.update(change)
                write_csv(q,[r]); write_csv(v,[decision])
                with self.assertRaises(ValueError): export(q,v,out)
                self.assertFalse(out.exists())

    def test_pending_not_exported(self):
        with tempfile.TemporaryDirectory() as d:
            q,v,out = [Path(d)/n for n in ('q.csv','v.csv','out.csv')]
            r=row(); decision=vote(r); decision.update(review_status='needs_review',approved_label='')
            write_csv(q,[r]);write_csv(v,[decision])
            self.assertEqual(export(q,v,out),[])

    def test_summary_types_and_duplicates_rejected(self):
        self.assertTrue(validate_rows([row(source_type='ftc_correction_summary_before_after')]))
        self.assertTrue(validate_rows([row(), row(1,text='fixture clause 0')]))
        self.assertTrue(validate_rows([row(group_id='')]))
        self.assertTrue(validate_rows([row(),row(1,group='g1',source_url='https://example.org/g0')]))

    def test_synthetic_lineage(self):
        parent=row()
        child=row(1,source_type='synthetic_contract_style_draft',evaluation_policy='train_only',parent_record_id='r0')
        self.assertFalse(validate_rows([parent,child]))
        self.assertTrue(validate_rows([child]))
        child['group_id']='different'
        self.assertTrue(validate_rows([parent,child]))

    def test_cross_split_near_duplicates_and_synthetic(self):
        self.assertTrue(cross_split_errors([row()], [row(1,group='g1',text='fixture clause 0 ')]))
        self.assertTrue(cross_split_errors([row()], [row(1,source_type='synthetic_contract_style_draft')]))

    def test_many_groups_supported(self):
        rows=[row(i*2+j,group=f'g{i}',label=str(j)) for i in range(20) for j in range(2)]
        groups, reason=splitting.select_groups(rows)
        self.assertIsNotNone(groups,reason)

    def test_standard_only_not_ready(self):
        rows=[row(i*2+j,group=f'g{i}',label=str(j),source_type='official_standard_clause_excerpt') for i in range(20) for j in range(2)]
        self.assertIsNone(splitting.select_groups(rows)[0])

    def test_positive_split_and_tamper(self):
        import random
        rng=random.Random(12)
        rows=[row(i*4+j,group=f'g{i}',label=str(j%2),text=''.join(rng.choices('가나다라마바사아자차카타파하',k=50)))
              for i in range(6) for j in range(4)]
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); q=root/'q.csv';v=root/'v.csv';out=root/'approved.csv';sp=root/'splits'
            write_csv(q,rows);write_csv(v,[vote(r) for r in rows]);export(q,v,out)
            with patch.multiple(splitting, INPUT=out,SPLITS=sp,REPORT=root/'report.json'):
                self.assertEqual(splitting.readiness()['status'],'READY')
            tr,va,errors=preflight_files(sp/'train.csv',sp/'validation.csv',out)
            self.assertFalse(errors,errors)
            tr[0]['approved_label']='1' if tr[0]['approved_label']=='0' else '0'
            write_csv(sp/'train.csv',tr)
            self.assertTrue(preflight_files(sp/'train.csv',sp/'validation.csv',out)[2])

    def test_comparison_rejects_legacy_or_different_splits(self):
        with self.assertRaises(ValueError): compare({'macro_f1':.9},{'macro_f1':.8})
        with self.assertRaises(ValueError): compare({'train_sha256':'a'},{'train_sha256':'b'})


if __name__=='__main__':
    unittest.main()
