"""Summarize completed out-of-fold runs, learning curves and data caveats."""
import argparse
import csv
import json
import sys
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from ml.data_contract import read_rows, normalize
from experiments.legacy_roberta.run import load_protocol, metrics, write_json


def main():
    p=argparse.ArgumentParser()
    p.add_argument('run',type=Path)
    a=p.parse_args();run=a.run.resolve()
    if not run.is_relative_to(Path(__file__).resolve().parent/'runs'):
        p.error('Only separate legacy experiment run directories are allowed')
    summary=json.loads((run/'summary.json').read_text(encoding='utf-8'))
    if summary['mode']!='cv':p.error('A completed full CV run is required')
    rp=read_rows(run/'roberta_oof_predictions.csv');tp=read_rows(run/'tfidf_oof_predictions.csv')
    rows,_,_,_=load_protocol()
    errors={name:{r['sample_id'] for r in pr if int(r['predicted'])!=int(r['label_true_weak'])}
            for name,pr in [('roberta',rp),('tfidf',tp)]}
    curves=[]
    for f in summary['folds']:
        curve=json.loads((run/f'fold_{f["fold"]}'/'learning_curve.json').read_text(encoding='utf-8'))
        curves.append({'fold':f['fold'],'epochs':len(curve),
            'train_f1':[e['train_eval']['macro_f1'] for e in curve],
            'validation_f1':[e['validation']['macro_f1'] for e in curve],
            'train_loss':[e['train_eval']['loss'] for e in curve],
            'validation_loss':[e['validation']['loss'] for e in curve],
            'final_gap':curve[-1]['train_minus_val_macro_f1'],
            'validation_loss_rises_while_train_loss_falls':any(
                b['validation']['loss']>a['validation']['loss'] and b['train_eval']['loss']<a['train_eval']['loss']
                for a,b in zip(curve,curve[1:]))})
    near=[]
    for i,x in enumerate(rows):
        for y in rows[i+1:]:
            if x['doc_id']==y['doc_id']:continue
            sim=SequenceMatcher(None,normalize(x['text']),normalize(y['text']),autojunk=False).ratio()
            if sim>=.9:near.append({'a':x['sample_id'],'b':y['sample_id'],'similarity':sim})
    flagged={'49_a','50_a','51_a','52_a'}
    subsets={}
    for name,pr in [('roberta',rp),('tfidf',tp)]:
        subsets[name]={}
        for doc in sorted({r['doc_id'] for r in pr},key=int):
            selected=[r for r in pr if r['doc_id']==doc]
            subsets[name][doc]=metrics([int(r['label_true_weak']) for r in selected],[int(r['predicted']) for r in selected])
    audit={'learning_curves':curves,'source_group_metrics':subsets,
        'cross_source_near_duplicates_threshold_0_90':near,
        'both_models_wrong':sorted(errors['roberta']&errors['tfidf']),
        'roberta_only_wrong':sorted(errors['roberta']-errors['tfidf']),
        'tfidf_only_wrong':sorted(errors['tfidf']-errors['roberta']),
        'known_questionable_after_labels':[r for r in rp if r['sample_id'] in flagged],
        'confident_errors_ge_0_90':[r for r in rp if int(r['predicted'])!=int(r['label_true_weak'])
                                  and max(float(r['model_score']),1-float(r['model_score']))>=.90],
        'warning':'Source-specific metrics and curves are descriptive post-hoc diagnostics, not model selection or legal validation.'}
    write_json(run/'diagnostics.json',audit)
    lines=['# 기존 공정위 요약문 KLUE-RoBERTa 탐색 실험 결과','',
           '**실제 학습 완료. 아래 수치는 공정위 시정 전·후 요약문의 약한 라벨 분류이며 실제 계약서 위험 탐지 성능이 아니다.**','',
           '## 동일 분할의 전체 OOF 비교','',
           '|모델|Macro-F1|Macro Precision|Macro Recall|오분류|','|---|---:|---:|---:|---:|']
    for key,label in [('tfidf','TF-IDF + Logistic Regression'),('roberta','KLUE-RoBERTa-base')]:
        m=summary[key]
        lines.append(f'|{label}|{m["macro_f1"]:.4f}|{m["precision_macro"]:.4f}|{m["recall_macro"]:.4f}|{m["error_count"]}/143|')
    lines+=['','143개 모두 정확히 한 번씩 미학습 출처 검증에 사용했다. OOF 전체 Macro-F1은 fold별 F1의 단순 평균과 다르다. fold 평균±표준편차는 summary.json에 별도 기록했다.','',
            '|모델|약한 라벨|Precision|Recall|F1|','|---|---|---:|---:|---:|']
    for name in ('tfidf','roberta'):
        for label in ('0','1'):
            m=summary[name]['classification_report'][label]
            lines.append(f'|{name}|{label}|{m["precision"]:.4f}|{m["recall"]:.4f}|{m["f1-score"]:.4f}|')
    lines+=['','0=시정 후 요약문 약라벨, 1=시정 전 요약문 약라벨. 법률상 적법/위법의 검증된 정답이 아니다.','',
            '혼동행렬은 행=정답 약라벨, 열=예측, 순서=[0,1].','']
    for name in ('tfidf','roberta'):
        lines.append(f'- {name}: `{summary[name]["confusion_matrix_true_rows_pred_cols_0_1"]}`')
    lines+=['','## 과적합 및 출처별 변동','',
            '|Fold|train/val 수|TF-IDF F1|RoBERTa F1|RoBERTa train F1|train−val 격차|','|---|---|---:|---:|---:|---:|']
    for f,c in zip(summary['folds'],curves):
        lines.append(f'|{f["fold"]}|{f["train_rows"]}/{f["validation_rows"]}|{f["tfidf"]["macro_f1"]:.4f}|{f["roberta"]["macro_f1"]:.4f}|{c["train_f1"][-1]:.4f}|{c["final_gap"]:.4f}|')
    trend=[c['fold'] for c in curves if c['validation_loss_rises_while_train_loss_falls']]
    lines+=['',f'학습 loss 감소 중 검증 loss 증가가 관찰된 fold: {trend}. 격차와 곡선은 과적합/출처 이동의 징후이며 원인을 단정하지 않는다. 학습 F1이 높아도 작은 출처별 검증 수에서는 편차가 크다.',
            f'검증 점수로 epoch를 선택하지 않고 사전에 정한 {summary["config"]["epochs"]} epoch의 마지막 가중치를 사용했다. 단일 seed 실험이고 별도 최종 test는 없다.',
            f'예측한 클래스의 모델 확률이 0.90 이상인데 약라벨 기준으로 틀린 사례가 {len(audit["confident_errors_ge_0_90"])}개다. 높은 확률을 법적 판단의 확신으로 해석하면 안 된다.','',
            '## 환경과 설정','',
            f'- 모델: klue/roberta-base, revision `{summary["config"]["pretrained"]["revision"]}`.',
            f'- 환경: Python {summary["config"]["python"]}, {summary["config"]["device_used"]}, CPU threads={summary["config"]["threads"]}.',
            f'- 패키지: `{summary["config"]["packages"]}`.',
            '- 5-fold StratifiedGroupKFold, doc_id 그룹, split seed=42. 기존 TF-IDF 기록과 sample별 fold 배정 동일.',
            f'- seed={summary["config"]["seed"]}, {summary["config"]["epochs"]} epoch/fold, batch={summary["config"]["batch_size"]}, AdamW lr={summary["config"]["lr"]}, weight_decay=0.01, gradient_clip=1.0. 전체 파라미터 fine-tuning.',
            f'- max_length={summary["config"]["max_length"]}, 동적 패딩. 실제 토큰 길이: {summary["config"]["token_lengths"]}. 절단 0개.',
            f'- 전체 CV 실행 시간: {summary["seconds_total"]:.1f}초(다운로드·사전 smoke 제외). 각 fold 체크포인트 재로딩 예측 일치 확인.',
            f'- 원본 데이터·기존 보고서·승인 파이프라인 등 {summary["protected_files_unchanged"]}개 파일의 실행 전후 해시 일치. 승인 데이터 경로와 최종 서비스 모델 경로에는 쓰지 않았다.','',
            '## 발견한 데이터 문제와 해석 제한','',
            '- 기존 original_README에는 139개 수정본 설명이 있으나 실제 제공 CSV는 143개다. 요청한 143개를 그대로 사용하고 이를 기록했다.',
            '- 49_a~52_a는 이미 검수 필요로 표시된 시정권고 방향 문구이다. 이를 정상 계약 조항으로 해석하면 안 된다. 이 실험에서 삭제하거나 재라벨링하지 않았다.',
            '- 13개 출처에는 수리·선수·연습생·플랫폼 등 다른 계약 영역이 섞인다. doc_id=5의 주택임대차 관련 8개도 실제 계약서 문구가 아닌 요약문이다.',
            '- 시정 전·후 서술 방식, 삭제·수정 안내 등 문체 단서로 약라벨을 맞출 수 있다. 모델 점수 향상이 법률 이해 향상을 의미하지 않는다.',
            f'- 출처 간 정규화 유사도 0.90 이상 문장 쌍: {len(near)}개. 이 문자열 검사는 의미 중복이나 원출처 계보 문제를 모두 찾지 못한다.',
            '- 모델 score는 보정되지 않은 분류 출력이다. 법적 위험 확률로 해석하지 않는다.','',
            '## 다음 개선 사항','',
            '1. 팀과 대상 계약서·정상/위험 기준을 정하고 실제 계약 조항을 수집·검수한다. 기존 승인 로직은 유지한다.',
            '2. 기존 검수 제외 규칙을 적용한 114개 별도 실험과 여러 seed 반복으로 라벨 잡음·학습 불안정성을 확인한다. 이번 143개 결과와 혼합하지 않는다.',
            '3. 추가 학습률·epoch 선택이 필요하면 그룹 기반 내부 검증으로 선택하고 외부 fold를 평가에만 사용한다. 현재 validation 결과를 반복 탐색에 사용하면 편향된다.',
            '4. 실제 계약서 독립 test와 도메인별 평가를 확보하기 전 서비스 모델로 승격하지 않는다.','',
            '## 파일','',
            '- summary.json: 전체 OOF 및 fold별 지표, 환경/설정.',
            '- roberta_errors.csv / tfidf_errors.csv: 원문 요약문과 정답 약라벨·예측·score를 포함한 오분류.',
            '- diagnostics.json: 출처별 지표, 공통/모델별 오류, 의심 라벨, 곡선 분석.',
            '- fold_*/learning_curve.json: epoch별 train/validation loss와 F1.',
            '- fold_*/checkpoint/: 탐색 전용 모델. EXPLORATORY_ONLY.json 표기; 서비스용 inference_config.json 없음.',
            '- split_manifest.json / protected_input_hashes.json: 분할 및 원본 보존 증거.','']
    (run/'RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
    print(run/'RESULTS.md')


if __name__=='__main__':main()
