#!/usr/bin/env python3
"""Run only diagnostic experiments on correction summaries and UNAPPROVED official/synthetic candidates.
This is NOT legal risk evaluation; source groups are kept disjoint for legacy holdout.
"""
from __future__ import annotations
import csv,json
from collections import Counter
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,f1_score,precision_score,recall_score
from sklearn.pipeline import Pipeline
ROOT=Path(__file__).resolve().parents[1]
def rows(path):
 with path.open('r',newline='',encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def model():return Pipeline([('tfidf',TfidfVectorizer(analyzer='char',ngram_range=(2,4),max_features=20000,sublinear_tf=True)),('classifier',LogisticRegression(class_weight='balanced',max_iter=2000,random_state=42))])
def main():
 legacy=rows(ROOT/'data/raw/clause_classification_original.csv')
 pairs=rows(ROOT/'data/raw/clause_pairs_original.csv')
 doc={r['clause_id']:r['doc_id'] for r in pairs}
 # HOLDOUT entire document 5, no leak, but labels are still correction summaries
 train=[r for r in legacy if doc[r['clause_id']]!='5']
 test=[r for r in legacy if doc[r['clause_id']]=='5']
 assert len(train)==135 and len(test)==8
 assert {doc[r['clause_id']] for r in train}.isdisjoint({doc[r['clause_id']] for r in test})
 m=model().fit([r['text'] for r in train],[int(r['label']) for r in train])
 yy=[int(r['label']) for r in test]
 yp=m.predict([r['text'] for r in test]).tolist();probs=m.predict_proba([r['text'] for r in test])[:,1].tolist()
 diagnosis={'legacy_heldout_doc5':{'train_samples':len(train),'test_samples':len(test),'train_source_groups':12,'held_out_source_group':5,'accuracy':accuracy_score(yy,yp),'macro_f1':f1_score(yy,yp,average='macro',zero_division=0),'confusion_0_1':confusion_matrix(yy,yp,labels=[0,1]).tolist(),'warning':'Test samples are pre/post-correction SUMMARIES; no true real-contract legal performance can be inferred.'}}
 pred=[]
 for r,y,p in zip(test,yp,probs):pred.append({'subset':'legacy_heldout_doc5','record_id':r['sample_id'],'text':r['text'],'candidate_label':r['label'],'predicted_risky':y,'p_risky_model_score':round(p,4),'status':'legacy_summary_weak','training_leakage':'none_from_doc5'})
 # Hold out 17 official standard candidate snippets and 11 unverified synthetic from training
 q=rows(ROOT/'data/review/risk_clause_review_queue.csv')
 train_all=model().fit([r['text'] for r in legacy],[int(r['label']) for r in legacy])
 for source_type,subset in [('official_standard_clause_excerpt','official_standard_candidates'),('synthetic_contract_style_draft','synthetic_drafts')]:
  sub=[r for r in q if r['source_type']==source_type]
  pred_y=train_all.predict([r['text'] for r in sub]).tolist()
  pred_p=train_all.predict_proba([r['text'] for r in sub])[:,1].tolist()
  diagnosis[subset]={'sample_count':len(sub),'candidate_label':sub[0]['proposed_label'] if sub else None,'predicted_risky_count':sum(pred_y),'predicted_safe_count':len(pred_y)-sum(pred_y),'warning':'Candidate labels have not been human reviewed. This is a source-shift diagnostic, NOT held-out accuracy or a legal finding.'}
  for r,y,p in zip(sub,pred_y,pred_p):pred.append({'subset':subset,'record_id':r['record_id'],'text':r['text'],'candidate_label':r['proposed_label'],'predicted_risky':y,'p_risky_model_score':round(p,4),'status':r['review_status'],'training_leakage':'not_in_training'})
 (ROOT/'reports/source_shift_diagnostic.json').write_text(json.dumps(diagnosis,ensure_ascii=False,indent=2),encoding='utf-8')
 with (ROOT/'reports/source_shift_predictions.csv').open('w',newline='',encoding='utf-8-sig') as f:
  w=csv.DictWriter(f,fieldnames=list(pred[0]));w.writeheader();w.writerows(pred)
 print(json.dumps(diagnosis,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
