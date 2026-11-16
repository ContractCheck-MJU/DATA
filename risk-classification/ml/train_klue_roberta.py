#!/usr/bin/env python3
"""KLUE-RoBERTa binary research prototype (NOT a legal opinion engine).

Preflight requires non-empty approved group-held-out train/validation CSV files.
Dependencies are imported lazily; --check-only works without torch/transformers.
"""
from __future__ import annotations
import argparse
import csv
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ml.data_contract import preflight_files
TRAIN=ROOT/'data/splits/train.csv'
VALID=ROOT/'data/splits/validation.csv'


def read_split(path):
    if not path.exists(): return []
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def preflight():
    return preflight_files(TRAIN, VALID)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--check-only',action='store_true')
    p.add_argument('--epochs',type=int,default=2)
    p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--lr',type=float,default=2e-5)
    p.add_argument('--max-length',type=int,default=192)
    p.add_argument('--model',default='klue/roberta-base')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--output',type=Path,default=ROOT/'artifacts/klue_roberta_risk_research')
    args=p.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0 or not 8 <= args.max_length <= 512:
        p.error('epochs/batch-size/lr must be positive; max-length must be 8..512')
    tr,va,errors=preflight()
    status={'train':len(tr),'validation':len(va),'train_class_counts':dict(Counter(r['approved_label'] for r in tr)),
            'validation_class_counts':dict(Counter(r['approved_label'] for r in va)), 'issues':errors}
    print(json.dumps(status,ensure_ascii=False,indent=2))
    if errors:
        print('학습 보류: 검수와 출처별 분할을 완료한 뒤 다시 실행하세요.')
        raise SystemExit(2)
    if args.check_only:
        print('학습 가능 조건 확인 완료. 실제 가중치 학습은 수행하지 않았습니다.')
        return
    if args.output.exists():
        p.error('Output exists; choose a new --output directory')
    # Do not even load 400MB+ HF weights until approvals and split satisfy guards.
    try:
        import torch
        from torch.utils.data import Dataset, DataLoader
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        from sklearn.metrics import classification_report, confusion_matrix, f1_score
    except ImportError as exc:
        raise SystemExit(f'모델용 의존성 설치 필요: python -m pip install -r ml/requirements-transformers.txt ({exc})')
    import numpy as np
    from ml.evaluation import save_evaluation
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device=torch.device('cuda' if torch.cuda.is_available() else 'mps' if hasattr(torch.backends,'mps') and torch.backends.mps.is_available() else 'cpu')
    print('학습 장치:',device)
    tokenizer=AutoTokenizer.from_pretrained(args.model)
    model=AutoModelForSequenceClassification.from_pretrained(args.model,num_labels=2,id2label={0:'NORMAL_CANDIDATE',1:'RISK_CANDIDATE'},label2id={'NORMAL_CANDIDATE':0,'RISK_CANDIDATE':1}).to(device)
    if args.max_length > model.config.max_position_embeddings - 2:
        p.error('max-length exceeds model position limit')
    # Never silently cut off a legal exception at the end of a clause.
    if any(len(tokenizer(r['text'], truncation=False)['input_ids']) > args.max_length for r in tr+va):
        p.error('Clause exceeds max-length; review segmentation/context or increase max-length')
    class ClauseSet(Dataset):
        def __init__(self, rows):
            self.rows=rows
            self.encoded=tokenizer([r['text'] for r in rows],padding='max_length',truncation=True,max_length=args.max_length)
        def __len__(self):return len(self.rows)
        def __getitem__(self,idx):
            sample={k:torch.tensor(v[idx],dtype=torch.long) for k,v in self.encoded.items()}
            sample['labels']=torch.tensor(int(self.rows[idx]['approved_label']),dtype=torch.long)
            return sample
    trainloader=DataLoader(ClauseSet(tr),batch_size=args.batch_size,shuffle=True)
    validloader=DataLoader(ClauseSet(va),batch_size=args.batch_size,shuffle=False)
    optim=torch.optim.AdamW(model.parameters(),lr=args.lr)
    out=args.output
    best=-1.0
    history=[]
    for epoch in range(1,args.epochs+1):
        model.train();losses=[]
        for batch in trainloader:
            batch={k:v.to(device) for k,v in batch.items()}
            optim.zero_grad(set_to_none=True)
            output=model(**batch)
            output.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            losses.append(float(output.loss.detach().cpu()))
        model.eval();pred=[];truth=[];scores=[]
        with torch.no_grad():
            for batch in validloader:
                truth.extend(batch['labels'].tolist())
                batch={k:v.to(device) for k,v in batch.items()}
                logits=model(**batch).logits
                pred.extend(logits.argmax(dim=-1).cpu().tolist())
                scores.extend(logits.softmax(dim=-1)[:,1].cpu().tolist())
        print(f'epoch {epoch}: train_loss={sum(losses)/len(losses):.4f}, val_macro_f1={f1_score(truth,pred,average="macro",zero_division=0):.4f}')
        score=float(f1_score(truth,pred,average='macro',zero_division=0))
        history.append({'epoch':epoch,'train_loss':sum(losses)/len(losses),'macro_f1':score})
        if score > best:
            best=score
            config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
            config.update(best_epoch=epoch,device=str(device),threshold=.5,
                          torch_version=torch.__version__,resolved_revision=getattr(model.config,'_commit_hash',None))
            save_evaluation(out,va,pred,scores,TRAIN,VALID,config)
            model.save_pretrained(out,safe_serialization=True);tokenizer.save_pretrained(out)
            (out/'inference_config.json').write_text(json.dumps({'max_length':args.max_length,
                'model_version':out.name,'threshold':.5,'research_only':True}),encoding='utf-8')
    (out/'history.json').write_text(json.dumps(history,indent=2),encoding='utf-8')
    print('실험용 모델 저장:',out)

if __name__=='__main__':main()
