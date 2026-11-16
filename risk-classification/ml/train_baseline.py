#!/usr/bin/env python3
"""Educational grouped baseline, NOT a legal risk classifier for deployment.

Run at project root: python ml/train_baseline.py
Requires: scikit-learn>=1.2, joblib
The labels distinguish pre/post correction case summaries, not true lease clauses.
"""
from __future__ import annotations
import csv, json
from pathlib import Path
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import classification_report, confusion_matrix, f1_score, recall_score, precision_score, accuracy_score
import joblib

ROOT=Path(__file__).resolve().parents[1]
EXPERIMENTS=[('legacy_all',ROOT/'data'/'raw'/'clause_classification_original.csv'),('exclude_rule_flags',ROOT/'data'/'examples'/'weak_label_exploratory.csv')]

def read_data(path, mode):
 with path.open(encoding='utf-8-sig',newline='') as f:rr=list(csv.DictReader(f))
 if mode=='legacy_all':
  with (ROOT/'data'/'raw'/'clause_pairs_original.csv').open(encoding='utf-8-sig',newline='') as f:pr=list(csv.DictReader(f))
  mapping={r['clause_id']:r['doc_id'] for r in pr}
  for r in rr:r['doc_id']=mapping[r['clause_id']]
  for r in rr:r['label_weak']=r['label']
 return rr

def new_pipeline():
 return Pipeline([
   ('tfidf',TfidfVectorizer(analyzer='char',ngram_range=(2,4),min_df=1,sublinear_tf=True,max_features=20000)),
   ('classifier',LogisticRegression(class_weight='balanced',max_iter=2000,random_state=42))
 ])

def evaluate(rows,tag):
 xs=np.array([r['text'] for r in rows],dtype=object)
 ys=np.array([int(r['label_weak']) for r in rows],dtype=int)
 groups=np.array([int(r['doc_id']) for r in rows],dtype=int)
 cv=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=42)
 predicted=np.full(len(xs),-1,dtype=int)
 folds=np.full(len(xs),-1,dtype=int)
 for fold,(tr,te) in enumerate(cv.split(xs,ys,groups),1):
  assert len(set(groups[tr])&set(groups[te]))==0, 'Document leakage detected'
  if len(set(ys[tr]))<2:raise RuntimeError('Training fold lacks a class')
  model=new_pipeline();model.fit(xs[tr],ys[tr]);predicted[te]=model.predict(xs[te]);folds[te]=fold
 assert np.all(predicted>=0) and np.all(folds>0)
 m={'experiment':tag,'sample_count':len(xs),'source_doc_count':len(set(groups)),
    'class_0':int(np.sum(ys==0)),'class_1':int(np.sum(ys==1)),'f1_macro':float(f1_score(ys,predicted,average='macro',zero_division=0)),
    'f1_risk_1':float(f1_score(ys,predicted,average='binary',zero_division=0)),
    'recall_risk_1':float(recall_score(ys,predicted,pos_label=1,zero_division=0)),
    'precision_risk_1':float(precision_score(ys,predicted,pos_label=1,zero_division=0)),
    'accuracy':float(accuracy_score(ys,predicted)),
    'confusion_matrix_rows_true_0_1_cols_pred_0_1':confusion_matrix(ys,predicted,labels=[0,1]).tolist(),
    'note':'Weak-label before/after correction case SUMMARY classification only; NOT real contract detection. 13 groups; unstable estimates.'}
 out=[]
 for r,yhat,fold in zip(rows,predicted,folds):
  out.append({'experiment':tag,'fold':int(fold),'sample_id':r['sample_id'],'clause_id':r['clause_id'],
             'doc_id':r['doc_id'],'text':r['text'],'label_true_weak':r['label_weak'],'predicted':int(yhat),
             'correct':bool(yhat==int(r['label_weak']))})
 return m,out

def main():
 metrics=[];preds=[]
 for tag,path in EXPERIMENTS:
  rows=read_data(path,tag)
  m,pr=evaluate(rows,tag)
  metrics.append(m);preds.extend(pr)
  print(json.dumps(m,ensure_ascii=False,indent=2))
  if tag=='exclude_rule_flags':
   model=new_pipeline().fit(np.array([r['text'] for r in rows]),np.array([int(r['label_weak']) for r in rows]))
   # joblib uses pickle: load only from trusted artifact, do not expose as a production legal classifier
   joblib.dump(model,ROOT/'reports'/'exploratory_baseline.joblib')
 with (ROOT/'reports'/'baseline_metrics.json').open('w',encoding='utf-8') as f: json.dump({'runs':metrics},f,ensure_ascii=False,indent=2)
 with (ROOT/'reports'/'cross_validated_predictions.csv').open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(preds[0]));w.writeheader();w.writerows(preds)

if __name__=='__main__': main()
