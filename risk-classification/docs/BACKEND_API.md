# Spring Boot에서 fold 1 연구 모델 호출하기

Spring Boot가 FastAPI에 조항 한 건을 보내고 같은 `clause_id`로 응답을 연결한다. 모델은 공정위 시정 전후 요약문 143건을 사용한 탐색 실험의 fold 1이다. 실제 주택임대차계약서 위험 판정 성능은 검증되지 않았다.

## 1. 실행 환경 선택

최신 내부 서버는 [배치 API 명세와 모드별 실행 안내](INTERNAL_BATCH_API.md)에 따라 설치한다. 기존 모델 ZIP에서는 fold 1 체크포인트만 사용한다. 아래 단건 예제에서도 서버와 호출자에 같은 `CONTRACTCHECK_INTERNAL_API_KEY`를 주입한다.

## 2. 서버 실행과 단건 호출

설치한 폴더에서 서버를 실행한다. 아래 주소는 Python과 호출자가 같은 호스트에 있을 때 사용한다.

```powershell
Remove-Item Env:CONTRACTCHECK_MODEL_DIR -ErrorAction SilentlyContinue
$env:CONTRACTCHECK_LEGACY_FOLD = '1'
.\.venv\Scripts\python.exe -m uvicorn api.app:app --host 127.0.0.1 --port 8000
```

두 번째 PowerShell 창에서 준비 상태를 확인하고 더미 문장으로 호출한다.

```powershell
$base = 'http://127.0.0.1:8000'
Invoke-RestMethod "$base/dev/v1/legacy/health/ready"
$body = @{ clause_id = 'integration-dummy-001'; text = '개발 연동 확인용 입력입니다.' } | ConvertTo-Json
Invoke-RestMethod "$base/dev/v1/legacy/clauses/classify" -Method Post -Headers @{'X-Internal-Api-Key'=$env:CONTRACTCHECK_INTERNAL_API_KEY} -ContentType 'application/json; charset=utf-8' -Body ([Text.Encoding]::UTF8.GetBytes($body))
```

요청 본문은 `{"clause_id":"integration-dummy-001","text":"개발 연동 확인용 입력입니다."}`이다. 양끝 공백 제거 후 ID는 1~128자, 본문은 1~10,000자다. 특수 토큰을 포함해 192토큰을 초과하거나 `domain` 등 추가 필드를 보내면 422를 반환한다.

fold 1의 더미 응답 예시는 다음과 같다.

```json
{
  "clause_id": "integration-dummy-001",
  "model_version": "cv_20261116_seed42/fold_1",
  "fold": 1,
  "weak_label": 0,
  "weak_label_name": "AFTER_CORRECTION_SUMMARY_WEAK",
  "before_summary_model_score": 0.13432887196540833,
  "input_tokens": 9,
  "research_only": true,
  "warning": "Model trained on FTC correction case summaries with weak labels. Scores are uncalibrated class outputs and do not measure legal risk or reliability."
}
```

`weak_label` 0은 시정 **후**, 1은 시정 **전** 요약문 클래스다. 점수는 예측 라벨에 관계없이 클래스 1의 softmax 출력이다. Spring 응답에는 `research_only=true`와 버전을 유지한다. 계약의 정상·위험 여부, 법률 위험 확률 또는 운영 DB의 확정 위험 등급으로 변환하면 안 된다.

## 3. Spring에서 처리할 상태

| 상황 | 호출 측 처리 |
|---|---|
| 준비 확인 200 | `GET /dev/v1/legacy/health/ready`의 버전과 fold를 확인한다. |
| 분류 성공 200 | 요청 ID와 응답 ID, 버전·fold·약라벨·연구용 표시를 검증한다. |
| 401 | 내부 키 누락 또는 불일치. 서버와 호출자 설정을 확인한다. |
| 422 | 일반 입력 오류는 `detail` 배열이다. 토큰 초과는 `detail.code=TOKEN_LIMIT_EXCEEDED`이다. 입력값이 포함될 수 있어 오류 본문을 로그에 남기지 않는다. |
| 503 | `LEGACY_MODEL_DISABLED`이면 fold 환경 변수를 확인한다. 승인 모델용 `/health/ready`, `/v1/clauses/classify`는 별도 경로이며 승인 모델 미설정 시 503이다. |
| 연결 실패·타임아웃·500 | 분석 실패로 처리하고 주소·포트·서버 로그를 확인한다. 실패를 약라벨 0으로 대체하지 않는다. 잘못된 fold나 누락된 체크포인트는 서버 시작 실패로 나타날 수 있다. |

[Java 예제](../spring-client/README.md)는 연결 3초, 응답 30초를 개발 초기값으로 사용한다. 팀의 Spring Boot 버전과 실행망에 맞춰 설정하고 [연동 체크리스트](INTEGRATION_CHECKLIST.md)로 확인한다.
