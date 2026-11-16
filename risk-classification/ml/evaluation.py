"""Identical evaluation schema for approved-data TF-IDF and RoBERTa."""
import csv
import json
from importlib.metadata import version
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score
from ml.data_contract import digest


def metrics(rows, predictions):
    truth = [int(r['approved_label']) for r in rows]
    result = {
        'rows': len(rows),
        'macro_f1': float(f1_score(truth, predictions, labels=[0, 1], average='macro', zero_division=0)),
        'precision_macro': float(precision_score(truth, predictions, labels=[0, 1], average='macro', zero_division=0)),
        'recall_macro': float(recall_score(truth, predictions, labels=[0, 1], average='macro', zero_division=0)),
        'precision_risk': float(precision_score(truth, predictions, zero_division=0)),
        'recall_risk': float(recall_score(truth, predictions, zero_division=0)),
        'confusion_matrix_true_rows_pred_columns_0_1': confusion_matrix(truth, predictions, labels=[0, 1]).tolist(),
        'classification_report': classification_report(truth, predictions, labels=[0, 1], output_dict=True, zero_division=0)
    }
    return result


def save_evaluation(out, rows, predictions, scores, train_path, valid_path, config):
    out.mkdir(parents=True, exist_ok=True)
    result = metrics(rows, predictions)
    result.update(config=config, train_sha256=digest(train_path), validation_sha256=digest(valid_path),
                  evaluation_scope='development_validation_not_independent_test',
                  warning='Model score is not legal risk probability; no service certification.',
                  versions={p: version(p) for p in ('numpy', 'scikit-learn', 'joblib')})
    result['by_domain'] = {}
    result['by_source_type'] = {}
    for field, dest in [('domain', 'by_domain'), ('source_type', 'by_source_type')]:
        for key in sorted({r[field] for r in rows}):
            idx = [i for i, r in enumerate(rows) if r[field] == key]
            result[dest][key] = metrics([rows[i] for i in idx], [predictions[i] for i in idx])
    (out/'evaluation.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    with (out/'validation_predictions.csv').open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.writer(f); w.writerow(['record_id', 'group_id', 'approved_label', 'predicted', 'model_score', 'correct'])
        for r, pred, score in zip(rows, predictions, scores):
            w.writerow([r['record_id'], r['group_id'], r['approved_label'], int(pred), float(score), int(r['approved_label']) == pred])
    return result
