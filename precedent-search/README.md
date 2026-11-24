
# ContractCheck 판례 검색 RAG

## 1. 프로젝트 개요

계약 조항 또는 질문을 입력하면 관련 판례를 검색하는 RAG 검색 모듈입니다.

- 임베딩 모델: BAAI/bge-m3
- 판례 데이터: 16건, 62개 청크
- 벡터 차원: 1024
- 검색 방식: 코퍼스 중심화 후 코사인 유사도 기반, 판례별 최고 청크 점수
- API 프레임워크: FastAPI
- 벡터 저장 방식: NPZ

## 2. 검색 정확도 평가

20개 질문을 이용한 1차 내부 평가 결과입니다.

| 지표 | 결과 |
|---|---|
| Hit@1 | 85% (17/20) |
| Hit@3 | 95% (19/20) |
| MRR@3 | 0.900 |

평가 결과는 `data/processed` 폴더에 저장되어 있습니다.

검색 적중률은 법률 판단의 정확도를 의미하지 않습니다. 현재 데이터는 추가 법률 검수가 필요한 실습용 자료입니다.

### 벡터 후처리 보정 및 추가 평가

BGE-M3 임베딩에 코퍼스 중심화(문서 벡터 평균 제거)를 적용했습니다.
문서 및 질문 벡터에 동일한 중심화와 재정규화를 적용하며,
기존 임베딩 파일을 재생성하거나 모델을 추가 학습하지 않습니다.

| 평가 | 지표 | 보정 전 | 보정 후 |
|---|---|---|---|
| 기존 20문항 | Hit@1 | 85% | 85% |
| 기존 20문항 | Hit@3 | 95% | 100% |
| 기존 20문항 | MRR@3 | 0.900 | 0.925 |
| 추가 15문항 | Hit@1 | 93.3% | 100% |
| 추가 15문항 | Hit@3 | 100% | 100% |
| 추가 15문항 | MRR@3 | 0.967 | 1.000 |

추가 평가 문항의 정답은 보유 판례를 기준으로 작성한
임시 라벨이며 법률 전문가 검수를 거치지 않았습니다.
본 결과는 내부 검색 적중률로, 실서비스 정확도를 보장하지 않습니다.

## 3. 실행 방법

### 3.1 필요한 라이브러리 설치

Python 가상환경을 준비한 뒤 다음 명령어를 실행합니다.

```bash
python -m pip install -r requirements.txt
```

### 3.2 API 서버 실행

`precedent-search` 폴더에서 실행합니다.

```bash
python scripts/06_precedent_search_api.py
```

정상적으로 실행되면 다음 주소에서 API 문서를 확인할 수 있습니다.

http://127.0.0.1:8001/docs

## 4. 판례 검색 API

### 요청 주소

`POST /search/precedents`

### 요청 예시

```json
{
  "query": "임차인이 계약갱신 후 해지를 통보하면 언제 효력이 발생하나요?",
  "top_k": 3
}
```

- `query`: 검색할 계약 조항 또는 질문
- `top_k`: 반환할 판례 수 (기본값 3)
- `domain`: 선택적 판례 분야 필터

### 응답 데이터

- `case_no`: 판례번호
- `source_id`: 판례 식별자
- `record_id`: 검색된 청크 식별자
- `source_title`: 사건명
- `source_url`: 공식 판례 URL
- `court`: 법원
- `decided_at`: 판결일
- `domain`: 판례 분야
- `chunk_section`: 판례 내 청크 구분
- `score`: 코사인 유사도
- `text`: 관련 판례 내용 (최대 500자)

검색 유사도는 법적 위험 확률이나 판례 적용 가능성에 대한 판정이 아닙니다.

## 5. 주요 파일

### scripts

- `02_embed_cases_reviewed.py`: 판례 임베딩 생성
- `03_search_cases_reviewed.py`: 터미널 기반 판례 검색
- `04_evaluate_retrieval.py`: 검색 정확도 평가
- `06_precedent_search_api.py`: FastAPI 판례 검색 API

### data/processed

- `precedent_chunks_reviewed_with_urls.csv`: 판례 청크 및 메타데이터
- `precedent_embeddings_reviewed.npz`: BGE-M3 임베딩 벡터
- `retrieval_eval_results.csv`: 질문별 평가 결과
- `retrieval_eval_summary.json`: 정확도 평가 요약
- `retrieval_eval_report.md`: 평가 보고서

CSV와 NPZ는 서로 일치하는 버전을 사용해야 합니다.

## 6. 백엔드 연동

백엔드에서 `POST /search/precedents`를 호출하여 관련 판례를 검색할 수 있습니다.

현재 API는 로컬 주소 `127.0.0.1:8001`에서 실행되므로 다른 컴퓨터나 서버에서 호출하려면 실행 환경 및 네트워크 설정이 필요합니다.

본 모듈은 판례 검색 전용이며, 법령 검색 및 위험 조항 분류 API와는 별도로 동작합니다.
