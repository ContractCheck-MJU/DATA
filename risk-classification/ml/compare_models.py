"""Refuse misleading comparisons across different datasets or split versions."""
import argparse
import json
from pathlib import Path


def compare(a, b):
    for field in ('train_sha256', 'validation_sha256', 'evaluation_scope'):
        if not a.get(field) or a[field] != b.get(field):
            raise ValueError('Cannot compare: different/missing ' + field)
    keys = ['macro_f1', 'precision_macro', 'recall_macro', 'precision_risk', 'recall_risk',
            'confusion_matrix_true_rows_pred_columns_0_1']
    return {'scope': a['evaluation_scope'], 'runs': [dict(model=x['config']['model'], **{k: x[k] for k in keys}) for x in (a, b)]}


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('tfidf', type=Path); p.add_argument('roberta', type=Path)
    a = p.parse_args()
    print(json.dumps(compare(json.loads(a.tfidf.read_text(encoding='utf-8')),
                             json.loads(a.roberta.read_text(encoding='utf-8'))), ensure_ascii=False, indent=2))
