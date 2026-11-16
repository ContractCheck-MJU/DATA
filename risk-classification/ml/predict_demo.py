#!/usr/bin/env python3
"""Legacy summary-text diagnostic CLI. NOT for legal decisions/service use.
Run: python ml/predict_demo.py '임대인은 ...'
"""
from __future__ import annotations
import json,sys
from pathlib import Path
import joblib
ROOT=Path(__file__).resolve().parents[1]
MODEL=ROOT/'reports/exploratory_baseline.joblib'

def predict(text):
 if not text.strip():raise ValueError('문장이 비어 있습니다.')
 # joblib uses pickle. Load only files that YOU trained or explicitly trust.
 mdl=joblib.load(MODEL)
 p=float(mdl.predict_proba([text])[:,1][0])
 return {'is_risky_candidate':bool(p>=0.5),'probability_model_score':round(p,4),'risk_type':None,'risk_level':None,
         'model_version':'tfidf-correction-summary-weak-v0',
         'warning':'Research baseline only. Trained on correction summaries, not verified real contracts. Never treat as legal determination or calibrated probability.'}
if __name__=='__main__':
 if len(sys.argv)<2:raise SystemExit("사용법: python ml/predict_demo.py '계약서 조항 문장'")
 print(json.dumps(predict(' '.join(sys.argv[1:])),ensure_ascii=False,indent=2))
