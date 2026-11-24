# BE 내부 추론 API · 이슈 #8

이 서버는 조항 배치 분류와 BGE-M3 원본 벡터 인코딩을 제공한다. **임베딩은 독립적으로 연동할 수 있다. 분류는 연구 시연 전용이다.** 현재 BE 분석 서비스에 시연 응답을 보내면 약라벨 0이 SAFE로 해석된다. 별도 연구 DTO·호출 경로를 합의할 때까지 기존 운영 호출에는 시연 헤더를 설정하지 않는다.

## 최신 코드 실행

DATA 체크아웃의 `risk-classification/`에서 실행한다. 이전 모델 ZIP은 **가중치 전달용**이다. ZIP의 `experiments/legacy_roberta/runs/cv_20261116_seed42/fold_1/checkpoint/` 전체를 체크아웃의 같은 상대 경로에 복사한다. 예전 ZIP의 Python 서버에는 새 인증과 배치 API가 없다. 임베딩만 사용할 때는 체크포인트가 필요 없다. 임베딩 모드는 같은 체크아웃의 `../law-embedding/api/app.py`에서 인코더 클래스만 재사용한다.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r ml/requirements-lock-windows-py312.txt
.\.venv\Scripts\python.exe -m pip check
```

서버와 호출자에 합의한 비밀을 환경변수로 주입한다. 서버 변수는 `CONTRACTCHECK_INTERNAL_API_KEY`, 현 BE 변수는 `INFERENCE_API_KEY`이다. 키 미설정·공백·BE 기본값 `unset`은 차단한다. 개발용 단건과 승인 모델용 단건을 포함한 **모든 추론 POST**에 `X-Internal-Api-Key`가 필요하다. 상태 GET에는 조항 입력이나 예측 결과가 없다.

이 인증 범위는 `risk-classification`의 `api.app:app`이다. 이전 ZIP의 서버와 별도로 기동한 법령 검색 서버의 `/embed`에는 이 검사가 적용되지 않는다. BE 두 주소 모두 이 문서의 인증 서버를 가리키게 하고 구버전·검색 전용 서버를 동일 내부 추론 경로에 노출하지 않는다.

각 PowerShell 세션에서 아래 초기 설정 후 한 가지 구성을 선택한다.

```powershell
if (-not $env:CONTRACTCHECK_INTERNAL_API_KEY) { throw '내부 키를 환경변수에 주입하세요.' }
Remove-Item Env:CONTRACTCHECK_MODEL_DIR,Env:CONTRACTCHECK_LEGACY_FOLD -ErrorAction SilentlyContinue
$env:CONTRACTCHECK_BATCH_MODE = 'off'
$env:CONTRACTCHECK_EMBED_ENABLED = '0'
$env:CONTRACTCHECK_EMBED_DEVICE = 'cpu'
```

| 구성 | 초기 설정에 추가할 환경변수 |
|---|---|
| 서버 · 분류 전용 | `CONTRACTCHECK_LEGACY_FOLD=1`, `CONTRACTCHECK_BATCH_MODE=research-demo` |
| PC · 임베딩 전용 | `CONTRACTCHECK_EMBED_ENABLED=1` |
| 로컬 통합 | 위 세 변수 모두 설정 |

예를 들어 PowerShell에서 `$env:CONTRACTCHECK_EMBED_ENABLED = '1'`로 지정한다. BGE-M3는 활성화했을 때만 로드한다. 첫 로딩은 고정 리비전의 모델 다운로드 또는 사전 준비한 Hugging Face 캐시가 필요하다. 캐시가 있으면 `HF_HUB_OFFLINE=1`로 실행할 수 있다. `CONTRACTCHECK_EMBED_DEVICE=cuda`는 CUDA 사용이 가능한 PyTorch 설치에서 선택한다. fold 1은 CPU로 동작한다. 승인 모델용 `CONTRACTCHECK_MODEL_DIR`은 독립 설정이다.

```powershell
.\.venv\Scripts\python.exe -m uvicorn api.app:app --host 127.0.0.1 --port 8000 --workers 1 --no-access-log
```

### 호출 예시

호출 세션에도 같은 비밀을 주입한다. 두 모델을 다른 장비에서 구동할 때 아래 주소를 각각 해당 개발망 주소로 설정한다.

```powershell
$classBase = 'http://127.0.0.1:8000'
$embedBase = 'http://127.0.0.1:8000'
$headers = @{'X-Internal-Api-Key'=$env:CONTRACTCHECK_INTERNAL_API_KEY}
$demo = $headers.Clone()
$demo['X-ContractCheck-Mode'] = 'research-only'
Invoke-RestMethod "$classBase/classify" -Method Post -Headers $demo -ContentType 'application/json' -Body '{"clauses":[{"id":"11","text":"demo clause"}]}'
Invoke-RestMethod "$embedBase/embed" -Method Post -Headers $headers -ContentType 'application/json' -Body '{"texts":["demo clause"],"normalize":true}'
```

## POST /classify

1~64개 조항을 순서대로 처리한다. ID는 요청 안에서 고유한 1~128자 문자열이며 양끝 공백을 거절한다. 본문은 공백 제거 후 1~10,000자다. 응답 개수·순서·ID를 보존하며 한 항목이라도 실패하면 요청 전체가 실패한다.

실제 더미 입력 `개발 연동 확인용 입력입니다.`에 대한 단건 배치 응답에서 본 구조다.

```json
{
  "results": [{
    "id": "0", "isRisky": false, "riskType": null, "probs": {},
    "modelVersion": "cv_20261116_seed42/fold_1",
    "weak_label": 0, "weak_label_name": "AFTER_CORRECTION_SUMMARY_WEAK",
    "before_summary_model_score": 0.13432887196540833, "windows": 1
  }],
  "research_only": true,
  "warning": "Model trained on FTC correction case summaries with weak labels. Scores are uncalibrated class outputs and do not measure legal risk or reliability."
}
```

0/1은 공정위 **시정 후/시정 전 요약문** 약라벨이다. 시연 모드에서만 클래스 1을 `isRisky=true`, `riskType=RISK_CANDIDATE`, `probs.RISK_CANDIDATE=before_summary_model_score`로 직렬화한다. 이 값은 보정되지 않은 클래스 출력이다. 모델은 유형·법률 위험 확률·위험 등급을 학습하지 않았다.

**긴 조항.** fold 1의 빠른 토크나이저로 특수 토큰 포함 192토큰 창을 만들고 내용 토큰 32개를 겹친다. 마지막 창까지 최대 4개씩 패딩·추론하며, 어느 구간에서든 시정 전 표현이 나타나는지 탐색하기 위해 클래스 1 점수의 최댓값을 사용한다. 0.5 초과가 약라벨 1이다. 긴 조항일수록 최대 점수가 커질 기회가 많고 예외 문구가 다른 창에 위치할 수 있다. 이 집계의 실제 계약 정확도는 미평가이며 기존 OOF 성능을 적용할 수 없다. 조항당 16창 또는 요청 전체 256창 초과는 422다. 개발 단건 API는 192토큰 초과 시 기존대로 422를 반환한다.

## POST /embed

`texts`는 1~64개, 각 1~10,000자, 특수 토큰 포함 최대 512토큰이다. `normalize`는 생략 또는 JSON `true`만 허용한다. 같은 순서로 4개씩 인코딩해 `{"vectors":[...]}`를 반환한다. 모든 행의 1,024차원·유한성·L2 노름(절대 오차 0.0001)을 확인한다. 한 문장이 한도를 넘으면 잘라내거나 다른 벡터로 대체하지 않고 422로 응답한다. 같은 조항을 양쪽 API에 보낼 때 두 토크나이저의 길이 제한을 각각 적용한다.

- **공통 원본 공간.** 법령의 LocalEmbedder와 같은 `BAAI/bge-m3` 리비전 `3069def033ce91d907258f9a830e442610dbfe0b`, fast tokenizer, 마지막 은닉층 CLS, L2 정규화를 사용한다. 중심화·PCA는 적용하지 않는다.
- **저장 벡터.** 법령은 임베딩 입력인 `embedding_text`로 대조한다. 판례 생성 코드는 SentenceTransformer로 `text`를 인코딩하고 정규화하며 모델 이름만 기록한다. NPZ에는 리비전·토크나이저·최대 길이·풀링 설정이 기록되어 있지 않다. 확인한 로컬 SentenceTransformer 구성은 CLS·최대 8192토큰이다. 짧은 표본의 수치 일치는 코퍼스 전체나 생성 당시 환경 동일성을 증명하지 않는다.
- **검색 단계.** BE 적재 스크립트는 법령의 원본 JSONL과 판례 원본 NPZ를 읽는다. 현재 DB의 실제 적재본은 별도로 확인해야 한다. 원본 벡터와 비교할 때는 원본 질의 벡터를 사용한다. DATA 검색 결과를 재현하려면 법령은 법령 평균·상위 주성분 2개 제거, 판례는 판례 평균만 제거하고 **각각 문서와 질의 양쪽을 재정규화**한다.

## 오류와 BE 합의 사항

| 상태 | 처리 |
|---|---|
| 401 | 내부 키 누락·불일치. 설정 확인 |
| 403 | 서버는 시연 모드지만 연구용 호출 헤더가 없음. 기존 운영 호출이 시연 응답을 받지 않도록 차단 |
| 422 | 본문·중복 ID·배치/토큰/창 제한 또는 정규화 옵션 오류. 입력 단위 확인 |
| 503 | 키 미설정 또는 해당 모델/시연 기능 비활성. 승인 모델 경로도 미설정 시 503 |
| 500 | 모델 오류·출력 이상. 부분 응답이나 비위험 라벨로 대체하지 않음 |

1. **종혁님 · 연구 결과 격리.** BE 현재 DTO/Jackson은 추가 필드를 읽어 버릴 수 있다. 현 매퍼는 음성을 확률 0으로 변환하고 RiskLevelResolver는 SAFE, 양성은 점수와 근거에 따라 CAUTION/HIGH로 매핑한다. 운영 클라이언트에 연구 헤더를 일괄 설정하지 않는다. 개발 전용 호출·DTO만 `research_only=true`를 요구하고, 등급 산정·운영 결과 저장 흐름에 진입하지 않게 분리한다. 운영 DTO에서도 누락된 출처 표시를 승인 근거로 취급하지 않는다.
2. **종혁님 · 두 주소와 장애 처리.** 현 HttpInferenceClient는 단일 `INFERENCE_BASE_URL`/RestClient로 두 요청을 보낸다. 분류·임베딩별 base URL과 transport를 주입하는 변경을 제안한다. `INFERENCE_CLASSIFY_BASE_URL`·`INFERENCE_EMBED_BASE_URL`은 합의할 이름이다. 실제 BE 기본 제한은 64건, 시도당 10초, 최대 3회다. 4xx는 요청 수정, 5xx·연결 오류 재시도 정책은 비활성 상태 및 긴 배치 비용을 고려해 결정한다.
3. **DB 대조.** 운영 전 DB의 문서 버전·행 식별자·원본/보정 여부를 확인하고 같은 입력·같은 문서 단위로 벡터와 순위를 비교한다. 저장·재시도·관측 로그에는 원문, 응답 본문, 인증 헤더를 남기지 않는다. API 키의 전송 보안과 서로 다른 장비 간 접근 제어는 BE 배포 범위다.

## 로컬 검증 기록

Windows CPU(Ryzen 7 7800X3D), 모델 프로세스 1개, OMP/MKL 스레드 4개, 짧은 더미 문장 반복. 아래 값은 모델 로드 후 로컬 HTTP 호출 한 번의 경과 시간, 누적 CPU 시간 차이, 해당 시점까지 프로세스의 최대 작업 집합이다. 측정 클라이언트가 같은 프로세스에서 직렬화·역직렬화한 비용을 포함한다.

| 전용 구성 | 입력 수 | 경과 시간 | CPU 시간 | 최대 작업 집합 |
|---|---:|---:|---:|---:|
| fold 1 | 1 / 64 | 0.130 / 0.712초 | 0.297 / 2.703초 | 727 / 732 MiB |
| BGE-M3 | 1 / 64 | 0.264 / 2.574초 | 0.922 / 9.922초 | 1805 / 1818 MiB |

동시 활성 구성에서 두 API에 각각 한 건 호출 후 최대 2142 MiB였다. 전용 구성에서는 반대 모델 미적재를 확인했다. 법령 3행·판례 3행의 저장 **원본** 벡터와 실수 오차를 감안한 코사인 범위는 각각 0.99994~1.00000, 약 1.00000이었다. 서버 사양, CUDA, 운영 DB, 동시 사용자 부하는 별도 검증 대상이다.

BE `f9d5ec7`의 요청·응답 레코드와 InferenceResponseMapper 사본을 BE의 Spring Boot 4.1.2 BOM(Jackson 3.1.7)으로 검사했다. 실제 배치의 양성·음성 두 행과 1,024차원 임베딩을 읽었고, 현 DTO가 연구 표시를 보존하지 않는 동작을 확인했다. 이 검사는 팀 BE 애플리케이션 전체나 DB 쓰기를 실행하지 않는다.

가중치를 배치하고 `python -m unittest discover -s tests -v`로 검사한다. 실제 BGE 저장 벡터 테스트는 고정 리비전 캐시를 준비하고 `CONTRACTCHECK_TEST_BGE=1`을 지정한다. 모의 테스트는 실데이터 없이 인증·한도·오류·선택 실행을 검사한다.
