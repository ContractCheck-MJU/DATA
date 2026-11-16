# 사람 검수와 최종 학습 준비

현재 승인된 실제 계약 조항은 0건이다. 공정위 요약문, 보도자료 요약문, 합성 후보, DRAFT의 제안 라벨은 승인 라벨로 사용하지 않는다.

1. 대상 계약 범위와 이진 라벨 기준을 팀에서 합의한다. 정상·위험 실제 문구를 여러 독립 계약서에서 확보하고 개인정보를 제거한다. 출처, 원문 위치, 스냅샷, 문맥, 원계약 출처 그룹을 보관한다.
2. `templates/review_queue.header.csv`를 Git에서 제외된 `data/review/risk_clause_review_queue.csv`로 복사하고 승인된 수집 범위 안에서 후보를 작성한다. `python scripts/prepare_review.py`로 별도의 `human_decisions.csv`를 만든다.
3. 두 검수자가 문구, 출처·문맥, 적용 가능성, 판단 근거, 0/1 라벨을 확인한다. 근거가 부족하거나 판단이 다르면 보류한다. `python scripts/review_hashes.py`의 검수 문구 해시와 원본 스냅샷 SHA-256을 기록한다. 두 검수자가 일치하는 경우에만 사람이 승인한다.
4. 아래 명령으로 승인본과 그룹 분할을 준비한다. 승인본이나 증거 파일이 바뀌면 다시 내보내고 분할한다. 같은 출처·부모의 조항은 분리하지 않으며 합성은 조건을 충족할 때만 학습에 포함한다.

```text
python scripts/export_approved.py
python scripts/prepare_group_split.py
python ml/train_klue_roberta.py --check-only
python ml/train_tfidf_reviewed.py --check-only
```

승인 자료가 없어 `BLOCKED`와 종료 코드 2가 나오는 것은 정상이다. READY가 확인된 환경에서만 `python ml/train_klue_roberta.py`와 `python ml/train_tfidf_reviewed.py`로 동일 분할을 학습하고, `ml/compare_models.py`로 두 모델의 `evaluation.json`을 비교한다. 서비스 검증은 별도의 독립 출처 테스트셋에서 진행한다.
