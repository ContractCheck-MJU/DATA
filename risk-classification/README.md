# ContractCheck 위험 조항 분류

공정위 시정 사례 요약문 143개를 약라벨로 학습한 탐색 실험과, 사람 검수를 거친 데이터만 사용하는 별도의 학습 파이프라인이다. 탐색 모델의 실제 계약서 위험 판정 성능은 검증되지 않았다.

| 경로 | 내용 |
|---|---|
| `api/` | FastAPI 및 개발용 RoBERTa 추론 |
| `ml/` | TF-IDF 베이스라인, 승인 데이터 검증·학습·평가, 의존성 |
| `scripts/` | 데이터 대조·검수 양식·승인 내보내기·그룹 분할 |
| `experiments/legacy_roberta/` | 탐색 학습 및 동일한 5-fold 설정 |
| `templates/` | 행이 없는 CSV 헤더 양식 |
| `docs/` | API 명세, 평가 요약, 승인 절차 및 데이터 제외 목록 |

## 설치와 테스트

저장소를 복제한 뒤 `risk-classification` 디렉터리에서 실행한다. 아래는 Windows PowerShell 기준이다. macOS나 Linux에서는 `python3.12 -m venv .venv`로 만들고 `.venv/bin/python`을 사용한다.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r ml/requirements-lock-windows-py312.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Windows CPU에서는 PyTorch 설치 경로가 너무 길어지지 않도록 짧은 경로에 복제한다. 그 외 환경은 `ml/requirements.txt`, `ml/requirements-transformers.txt`, `api/requirements.txt`를 사용한다. 비공개 입력이나 모델을 요구하는 테스트는 해당 파일이 없으면 건너뛴다.

## 백엔드 개발용 API

**설치 위치를 먼저 선택한다.** 저장소에서 실행할 때는 위의 `risk-classification/` 설치 절차를 사용하고 아래에 표시한 fold 1 체크포인트를 배치한다. 모델 ZIP만 사용할 때는 압축을 풀고 `ContractCheck_Backend_Fold1/` 안에서 가상환경을 만든 뒤 `.\.venv\Scripts\python.exe -m pip install -r requirements.txt`로 설치한다. 두 환경 모두 아래 서버 실행 명령을 사용한다.

[Spring Boot 연동 가이드](docs/BACKEND_API.md)에서 설치·호출 예시와 오류 처리를 확인한다. [Java 클라이언트 예제](spring-client/README.md)와 [팀 환경 연동 체크리스트](docs/INTEGRATION_CHECKLIST.md)로 개발 서버 연결을 점검할 수 있다.

별도로 받은 `ContractCheck_Backend_Fold1.zip`에서 아래 **체크포인트 폴더 전체**를 꺼내 동일한 상대 경로에 둔다. 기존 전달용 ZIP만으로도 독립 실행할 수 있다. 체크포인트와 ZIP은 Git 추적 대상에서 제외된다.

```text
experiments/legacy_roberta/runs/cv_20261116_seed42/fold_1/checkpoint/
  EXPLORATORY_ONLY.json
  config.json
  model.safetensors
  special_tokens_map.json
  tokenizer_config.json
  tokenizer.json
  vocab.txt
```

```powershell
Remove-Item Env:CONTRACTCHECK_MODEL_DIR -ErrorAction SilentlyContinue
$env:CONTRACTCHECK_LEGACY_FOLD = '1'
.\.venv\Scripts\python.exe -m uvicorn api.app:app --host 127.0.0.1 --port 8000
```

Spring Boot에서 `GET /dev/v1/legacy/health/ready`와 `POST /dev/v1/legacy/clauses/classify`를 사용한다. 요청·응답, 오류 및 예시는 [API 명세](docs/LEGACY_INFERENCE_API.md)를 참고한다. 모델 점수는 약라벨 출력이며 운영 위험도 필드와 분리한다. 승인 모델 경로 `/v1/clauses/classify`는 별도이며 승인 모델 미설정 시 503을 반환한다.

## 학습과 재현

- [탐색 결과](docs/LEGACY_EVALUATION.md)와 [데이터 공개 점검](docs/DATA_RELEASE.md)을 먼저 읽는다.
- 별도 승인을 받아 기존 실험 CSV를 전달받았다면 원래 이름으로 `data/raw/clause_classification_original.csv`, `data/raw/clause_pairs_original.csv`에 둔다. TF-IDF 전체 명령에는 `data/examples/weak_label_exploratory.csv`도 필요하다. `reports/`를 만든 뒤 `python ml/train_baseline.py`를 실행한다.
- 탐색 RoBERTa는 `python experiments/legacy_roberta/download_model.py`와 `python experiments/legacy_roberta/run.py --mode cv --run-name cv_reproduction --epochs 3 --batch-size 4 --threads 4`로 재현한다. 원래 데이터 SHA-256과 그룹별 fold 배정을 확인한다.
- [승인 데이터 검수·학습 절차](docs/REVIEWED_TRAINING.md)에 따라 별도 데이터를 준비한다. 승인본과 분할 검증을 통과해야 학습한다.
