"""Isolated weak-label research experiment. Never calls or bypasses approved training.

Reuses the original baseline loader, estimator and exact five-fold SGKF algorithm.
Fixed epoch budget (no selection on held-out scores); each fold starts pretrained.
"""
import argparse
import csv
import gc
import json
import os
import platform
import random
import sys
import time
from collections import Counter
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('HF_HOME', str(ROOT/'.cache/huggingface'))
os.environ.setdefault('HF_HUB_OFFLINE', '1')
from ml.train_baseline import read_data, new_pipeline
from ml.data_contract import digest
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score, accuracy_score

BASE = Path(__file__).resolve().parent
WARNING = 'Weak-label correction SUMMARY experiment only. Not actual lease risk detection or legal ground truth.'


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def write_csv(path, rows):
    # Keep headers even when no errors occur.
    fields = ['sample_id','clause_id','doc_id','fold','text','label_true_weak','predicted','model_score','correct']
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)


def metrics(truth, pred):
    return {'sample_count':len(truth), 'macro_f1':float(f1_score(truth,pred,labels=[0,1],average='macro',zero_division=0)),
            'precision_macro':float(precision_score(truth,pred,labels=[0,1],average='macro',zero_division=0)),
            'recall_macro':float(recall_score(truth,pred,labels=[0,1],average='macro',zero_division=0)),
            'accuracy':float(accuracy_score(truth,pred)),
            'classification_report':classification_report(truth,pred,labels=[0,1],output_dict=True,zero_division=0),
            'confusion_matrix_true_rows_pred_cols_0_1':confusion_matrix(truth,pred,labels=[0,1]).tolist(),
            'warning':WARNING}


def load_protocol():
    source = ROOT/'data/raw/clause_classification_original.csv'
    manifest = json.loads((BASE/'fold_assignments.json').read_text(encoding='utf-8'))
    if digest(source) != manifest['original_sha256']:
        raise ValueError('Legacy CSV does not match the archived training input')
    rows = read_data(source,'legacy_all')
    if len(rows)!=143 or len({r['sample_id'] for r in rows})!=143:
        raise ValueError('This version of the protocol requires the original 143 unique samples')
    y=np.array([int(r['label_weak']) for r in rows])
    groups=np.array([int(r['doc_id']) for r in rows])
    folds=list(StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=42).split(np.zeros(len(rows)),y,groups))
    seen=[]; assignments={}
    for fold,(tr,va) in enumerate(folds,1):
        if set(groups[tr]) & set(groups[va]):
            raise ValueError('Source leakage')
        if set(y[tr])!={0,1} or set(y[va])!={0,1}:
            raise ValueError('Both weak classes required in every partition')
        seen.extend(va.tolist())
        assignments.update({rows[i]['sample_id']:fold for i in va})
    if sorted(seen)!=list(range(len(rows))):
        raise ValueError('OOF coverage invalid')
    if assignments!=manifest['assignments']:
        raise ValueError('Split differs from the recorded TF-IDF baseline; investigate before comparing')
    return rows, y, folds, assignments


def protected_hashes():
    files=[]
    for folder in ('data','reports','ml','scripts','api'):
        files.extend(p for p in (ROOT/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    return {str(p.relative_to(ROOT)):digest(p) for p in sorted(files)}


def prediction_rows(rows, indices, fold, predictions, scores):
    return [dict(sample_id=rows[i]['sample_id'],clause_id=rows[i]['clause_id'],doc_id=rows[i]['doc_id'],
                 fold=fold,text=rows[i]['text'],label_true_weak=int(rows[i]['label_weak']),
                 predicted=int(p),model_score=float(s),correct=int(p)==int(rows[i]['label_weak']))
            for i,p,s in zip(indices,predictions,scores)]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['smoke','cv'],required=True)
    p.add_argument('--run-name',required=True)
    p.add_argument('--epochs',type=int,default=3)
    p.add_argument('--batch-size',type=int,default=4)
    p.add_argument('--lr',type=float,default=2e-5)
    p.add_argument('--max-length',type=int,default=192)
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    args=p.parse_args()
    if min(args.epochs,args.batch_size,args.threads)<1 or args.lr<=0 or not 8<=args.max_length<=512:
        p.error('Invalid training settings')
    if Path(args.run_name).name!=args.run_name or args.run_name in ('.','..'):
        p.error('run-name must be a single directory name')
    out=BASE/'runs'/args.run_name
    out.mkdir(parents=True,exist_ok=False)
    before=protected_hashes()
    write_json(out/'protected_input_hashes.json',before)
    start=time.perf_counter()
    rows,y,folds,assignments=load_protocol()
    write_json(out/'split_manifest.json',{'algorithm':'StratifiedGroupKFold','n_splits':5,'seed':42,
        'group_key':'doc_id','assignments':assignments,'matches_existing_tfidf':True,
        'warning':WARNING,'original_sha256':digest(ROOT/'data/raw/clause_classification_original.csv')})
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    device='cuda' if args.device=='auto' and torch.cuda.is_available() else 'cpu' if args.device=='auto' else args.device
    if device=='cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA is unavailable; use CPU')
    cache=ROOT/'.cache/klue-roberta-base'
    provenance=json.loads((cache/'download_provenance.json').read_text(encoding='utf-8'))
    tokenizer=AutoTokenizer.from_pretrained(cache,local_files_only=True)
    encoded=tokenizer([r['text'] for r in rows],truncation=False)
    lengths=[len(x) for x in encoded['input_ids']]
    if max(lengths)>args.max_length:
        raise ValueError(f'Max token length {max(lengths)} > {args.max_length}; increase --max-length; no silent truncation')
    config=dict(vars(args),device_used=device,pretrained=provenance,
                epoch_selection='fixed final epoch, NOT best held-out epoch',trainable='all parameters',
                optimizer='AdamW',weight_decay=.01,gradient_clip_norm=1.0,padding='dynamic',threshold=.5,
                token_lengths={'min':min(lengths),'max':max(lengths),'median':float(np.median(lengths))},
                python=platform.python_version(),platform=platform.platform(),logical_cpus=os.cpu_count(),
                cpu=os.environ.get('PROCESSOR_IDENTIFIER',''),
                packages={k:version(k) for k in ('torch','transformers','numpy','scikit-learn')},
                warning=WARNING)
    write_json(out/'config.json',config)
    collate=DataCollatorWithPadding(tokenizer)

    def loader(indices, shuffle=False):
        records=[dict({k:v[i] for k,v in encoded.items()},labels=int(y[i])) for i in indices]
        return DataLoader(records,batch_size=args.batch_size,shuffle=shuffle,collate_fn=collate,num_workers=0)

    def evaluate(model, indices):
        model.eval();pred=[];scores=[];loss=0.
        with torch.inference_mode():
            for b in loader(indices):
                b={k:v.to(device) for k,v in b.items()}
                output=model(**b)
                loss+=float(output.loss)*len(b['labels'])
                pred.extend(output.logits.argmax(-1).cpu().tolist())
                scores.extend(output.logits.softmax(-1)[:,1].cpu().tolist())
        return dict(metrics(y[indices],pred),loss=loss/len(indices)),pred,scores

    all_roberta=[];all_tfidf=[];fold_results=[]
    for fold,(tr,va) in enumerate(folds,1):
        if args.mode=='smoke':
            tr=tr[:8];va=va[:8]
        fold_dir=out/f'fold_{fold}';fold_dir.mkdir()
        random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
        print(f'Fold {fold}: train={len(tr)}, validation={len(va)}, device={device}',flush=True)
        tfidf=new_pipeline().fit([rows[i]['text'] for i in tr],y[tr])
        tp=tfidf.predict([rows[i]['text'] for i in va]);ts=tfidf.predict_proba([rows[i]['text'] for i in va])[:,1]
        tf_rows=prediction_rows(rows,va,fold,tp,ts);all_tfidf.extend(tf_rows)
        tf_metrics=metrics(y[va],tp)
        write_json(fold_dir/'tfidf_metrics.json',tf_metrics)
        write_csv(fold_dir/'tfidf_predictions.csv',tf_rows)
        model=AutoModelForSequenceClassification.from_pretrained(cache,local_files_only=True,num_labels=2,
              id2label={0:'AFTER_CORRECTION_SUMMARY_WEAK',1:'BEFORE_CORRECTION_SUMMARY_WEAK'},
              label2id={'AFTER_CORRECTION_SUMMARY_WEAK':0,'BEFORE_CORRECTION_SUMMARY_WEAK':1}).to(device)
        optim=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=.01)
        history=[]
        for epoch in range(1,(1 if args.mode=='smoke' else args.epochs)+1):
            tick=time.perf_counter();model.train();loss=0.;count=0
            for step,b in enumerate(loader(tr,shuffle=True),1):
                b={k:v.to(device) for k,v in b.items()}
                optim.zero_grad(set_to_none=True)
                output=model(**b)
                if not torch.isfinite(output.loss):
                    raise RuntimeError('Non-finite training loss')
                output.loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);optim.step()
                loss+=float(output.loss.detach())*len(b['labels']);count+=len(b['labels'])
                if step%5==0:
                    print(f'fold={fold} epoch={epoch} step={step} loss={loss/count:.4f}',flush=True)
            train_metrics,_,_=evaluate(model,tr)
            val_metrics,pred,scores=evaluate(model,va)
            history.append({'epoch':epoch,'optimization_loss':loss/count,'train_eval':train_metrics,
                            'validation':val_metrics,'seconds':time.perf_counter()-tick,
                            'train_minus_val_macro_f1':train_metrics['macro_f1']-val_metrics['macro_f1']})
            write_json(fold_dir/'learning_curve.json',history)
            print(f'fold={fold} epoch={epoch} train_F1={train_metrics["macro_f1"]:.4f} val_F1={val_metrics["macro_f1"]:.4f} seconds={history[-1]["seconds"]:.1f}',flush=True)
        result_rows=prediction_rows(rows,va,fold,pred,scores);all_roberta.extend(result_rows)
        write_json(fold_dir/'roberta_metrics.json',val_metrics)
        write_csv(fold_dir/'roberta_predictions.csv',result_rows)
        write_csv(fold_dir/'roberta_errors.csv',[r for r in result_rows if not r['correct']])
        checkpoint=fold_dir/'checkpoint'
        model.save_pretrained(checkpoint,safe_serialization=True);tokenizer.save_pretrained(checkpoint)
        write_json(checkpoint/'EXPLORATORY_ONLY.json',dict(warning=WARNING,fold=fold,
                   service_eligible=False,source_type='ftc_correction_summary_before_after',config=config))
        # Verify actual saved weights load and reproduce held-out predictions.
        del optim,output,model;gc.collect()
        if device=='cuda':torch.cuda.empty_cache()
        reloaded=AutoModelForSequenceClassification.from_pretrained(checkpoint,local_files_only=True).to(device)
        _,reload_pred,reload_scores=evaluate(reloaded,va)
        if reload_pred!=pred or not np.allclose(reload_scores,scores,atol=1e-6):
            raise RuntimeError('Checkpoint roundtrip mismatch')
        del reloaded;gc.collect()
        fold_results.append({'fold':fold,'train_rows':len(tr),'validation_rows':len(va),
              'train_groups':sorted({rows[i]['doc_id'] for i in tr}),
              'validation_groups':sorted({rows[i]['doc_id'] for i in va}),
              'tfidf':tf_metrics,'roberta':val_metrics,'checkpoint_roundtrip':'PASS',
              'final_train_minus_val_macro_f1':history[-1]['train_minus_val_macro_f1']})
        write_json(out/'fold_results.json',fold_results)
        if args.mode=='smoke':break
    for name,pr in [('roberta',all_roberta),('tfidf',all_tfidf)]:
        write_csv(out/f'{name}_oof_predictions.csv',pr)
        write_csv(out/f'{name}_errors.csv',[r for r in pr if not r['correct']])
    if args.mode=='cv' and {r['sample_id'] for r in all_roberta}!={r['sample_id'] for r in rows}:
        raise RuntimeError('Full CV coverage mismatch')
    changed=[p for p,h in before.items() if digest(ROOT/p)!=h]
    if changed:raise RuntimeError(f'Protected input changed: {changed}')
    summary={'mode':args.mode,'warning':WARNING,'seconds_total':time.perf_counter()-start,
             'config':config,'folds':fold_results,'protected_files_unchanged':len(before)}
    for name,pr in [('tfidf',all_tfidf),('roberta',all_roberta)]:
        summary[name]=metrics([r['label_true_weak'] for r in pr],[r['predicted'] for r in pr])
        summary[name]['error_count']=sum(not r['correct'] for r in pr)
        summary[name]['fold_macro_f1_mean']=float(np.mean([f[name]['macro_f1'] for f in fold_results]))
        summary[name]['fold_macro_f1_std']=float(np.std([f[name]['macro_f1'] for f in fold_results]))
    write_json(out/'summary.json',summary)
    print(json.dumps({k:summary[k] for k in ('mode','seconds_total','tfidf','roberta')},ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    main()
