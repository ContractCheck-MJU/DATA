# ContractCheck 탐색 모델 API 연동 가이드

기존 FastAPI 앱의 `/dev/v1/legacy` 경로에서 KLUE-RoBERTa 체크포인트로 추론한다. 143개 학습 자료는 공정위 시정 사례 **요약문**이고, 그중 각 fold의 학습 분할을 사용했다. `0`은 시정 후 요약문, `1`은 시정 전 요약문의 약라벨이다. 모델 점수가 높아도 법적 위험이나 판단의 신뢰도를 보장하지 않는다.

## 체크포인트 선택과 시작

연동 테스트의 재현성을 위해 **번호가 가장 작은 fold 1**을 기준으로 사용한다. 검증 결과는 선택에 사용하지 않았다. 5개 모델은 서로 다른 출처 그룹에서 평가했으므로 검증 점수로 최종 모델 순위를 정하기 어렵다. `CONTRACTCHECK_LEGACY_FOLD`로 1~5 중 하나를 명시하며, 결과에 fold와 실험 버전이 포함된다. 모델을 바꾸려면 환경 변수를 변경하고 서버를 재시작한다.

모델 경로는 `experiments/legacy_roberta/runs/cv_20261116_seed42/fold_<N>/checkpoint/`이다. 해당 폴더의 `model.safetensors`, `config.json`, 토크나이저 파일들과 `EXPLORATORY_ONLY.json`이 필요하다. 가중치는 Git 제외 파일이라 다른 PC에서는 체크포인트 폴더를 별도로 전달받아 같은 위치에 둔다. 서버가 시작할 때 로컬에서 한 번 로드하고 CPU에서 추론한다. 잘못된 fold 번호나 누락·불일치 체크포인트는 시작 오류로 처리한다.

프로젝트 루트 PowerShell에서 실행한다. 새 환경에서의 의존성 설치는 `python -m pip install -r ml/requirements-lock-windows-py312.txt`를 참고한다.

```powershell
Remove-Item Env:CONTRACTCHECK_MODEL_DIR -ErrorAction SilentlyContinue
$env:CONTRACTCHECK_LEGACY_FOLD = '1'
$env:OMP_NUM_THREADS = '4'
$env:MKL_NUM_THREADS = '4'
.\.venv\Scripts\python.exe -m uvicorn api.app:app --host 127.0.0.1 --port 8000
```

Spring Boot를 같은 PC에서 실행할 때 base URL은 `http://127.0.0.1:8000`이다. `GET /dev/v1/legacy/health/ready`에서 HTTP 200과 `model_version`을 확인한 뒤 호출한다. Swagger UI는 `/docs`, OpenAPI JSON은 `/openapi.json`에서 확인할 수 있다. `/health/ready`는 승인 모델 준비 상태를 계속 표현한다.

## 단건 분류 명세

`POST /dev/v1/legacy/clauses/classify` · `Content-Type: application/json`

```json
{"clause_id":"demo-1","text":"사업자가 계약 내용을 일방적으로 변경할 수 있도록 규정함."}
```

`clause_id`는 호출자가 정하는 1~128자 문자열이고, `text`는 1~10,000자 문자열이다. 양쪽 공백을 제거한 뒤 검증한다. 추가 필드는 허용하지 않는다. 입력 한도는 특수 토큰을 포함해 **192 모델 토큰**이며, 초과하면 전문을 자르는 대신 422를 반환한다.

아래 응답은 fold 1에서 위 요청을 실행한 결과다.

```json
{
  "clause_id": "demo-1",
  "model_version": "cv_20261116_seed42/fold_1",
  "fold": 1,
  "weak_label": 1,
  "weak_label_name": "BEFORE_CORRECTION_SUMMARY_WEAK",
  "before_summary_model_score": 0.9959288239479065,
  "input_tokens": 17,
  "research_only": true,
  "warning": "Model trained on FTC correction case summaries with weak labels. Scores are uncalibrated class outputs and do not measure legal risk or reliability."
}
```

`weak_label`은 두 클래스 중 점수가 높은 쪽이고, `before_summary_model_score`는 예측 라벨에 관계없이 항상 클래스 1의 softmax 출력이다. `clause_id`를 요청과 응답의 연결 키로 사용한다. Spring Boot DTO는 `weak_label`을 정수, `before_summary_model_score`를 실수로 받고 snake_case 필드 이름을 매핑한다. 탐색 결과에는 별도 DTO를 사용하고 `is_risky`, `risk_level` 같은 운영 필드에 저장하지 않는다.

| 상태 | 경우 | 호출 측 처리 |
|---|---|---|
| 200 | 모델 준비 완료 또는 추론 성공 | 개발 DTO로 파싱 |
| 422 | 공백, 누락, 잘못된 타입, 추가 필드 등 | 요청을 수정. 기본 FastAPI `detail` 배열 |
| 422 | 토큰 한도 초과 | `detail.code=TOKEN_LIMIT_EXCEEDED`, `detail.max_tokens=192`. 보낼 문장 범위를 재검토 |
| 503 | 개발 모델 설정 없음 | `detail.code=LEGACY_MODEL_DISABLED`. 배포 설정 확인 |
| 500 | 모델 내부 오류 | 서버 로그 확인. 비정상 점수는 `detail.code=INVALID_MODEL_OUTPUT` |

PowerShell 호출 예시:

```powershell
$payload = @{clause_id='demo-1'; text='사업자가 계약 내용을 일방적으로 변경할 수 있도록 규정함.'} | ConvertTo-Json
Invoke-RestMethod -Uri 'http://127.0.0.1:8000/dev/v1/legacy/clauses/classify' `
  -Method Post -ContentType 'application/json; charset=utf-8' `
  -Body ([Text.Encoding]::UTF8.GetBytes($payload))
```

## 검증 결과와 남은 연동

`api/app.py`는 개발 라우터를 등록하고 시작 시 별도 모델 상태를 초기화한다. `api/legacy.py`는 체크포인트 검사, 토큰화, 추론 및 개발 응답을 담당한다. 공개 테스트 [test_legacy_api.py](../tests/test_legacy_api.py)는 별도 제공된 fold 1 체크포인트가 있으면 위 `demo-1` 예제 **한 건**의 약라벨과 클래스 1 점수를 대조한다. 응답 ID·버전·연구용 표시, 잘못된 입력의 422, 승인 모델 경로의 503도 확인한다. 가중치가 없는 환경에서는 실제 추론 테스트를 건너뛰고, 모델 비활성 상태와 잘못된 fold 설정 등은 별도로 검사한다.

Spring Boot 쪽에서 개발 DTO와 HTTP 클라이언트를 구현하고, 개발 환경의 주소·접근 제어·타임아웃을 정한 뒤 동일 요청으로 통합 테스트한다. 서버를 서로 다른 호스트에 놓는 경우 서비스 주소와 방화벽 설정이 필요하다. 개발 API에는 인증이 구현돼 있지 않으므로 승인된 개발망에서 예제 텍스트로만 연동을 확인한다. 유효성 오류에 입력값이 포함될 수 있어 오류 본문을 그대로 로그에 남기지 않는다.
