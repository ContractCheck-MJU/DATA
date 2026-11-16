"""Same immutable split and metrics as the RoBERTa run; legacy baseline unchanged."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ml.data_contract import ROOT
from ml.train_klue_roberta import preflight, TRAIN, VALID


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--check-only', action='store_true')
    p.add_argument('--output', type=Path, default=ROOT/'artifacts/tfidf_reviewed')
    a = p.parse_args()
    tr, va, errors = preflight()
    if errors:
        p.exit(2, 'BLOCKED: ' + '; '.join(errors) + '\n')
    if a.check_only:
        print('READY'); return
    if a.output.exists():
        p.exit(2, 'Output exists; choose a new --output directory\n')
    import joblib
    from ml.train_baseline import new_pipeline
    from ml.evaluation import save_evaluation
    model = new_pipeline().fit([r['text'] for r in tr], [int(r['approved_label']) for r in tr])
    texts = [r['text'] for r in va]
    result = save_evaluation(a.output, va, model.predict(texts).tolist(), model.predict_proba(texts)[:, 1].tolist(),
                             TRAIN, VALID, {'model': 'tfidf_logistic', 'seed': 42, 'threshold': .5})
    joblib.dump(model, a.output/'model.joblib')
    print('Validation Macro-F1:', result['macro_f1'])


if __name__ == '__main__':
    main()
