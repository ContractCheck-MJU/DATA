# 법령 검색 API

계약 조항 문구를 받아 관련 법령 조문을 돌려주는 검색 API입니다. 판례 검색 API(`precedent-search`)와 같은 구조로 맞췄고, **백엔드가 분류·판례·법령 API를 각각 호출하는 구조**를 전제로 합니다.

- 이미 만들어 둔 법령 임베딩(법령 95건 / 청크 165개)을 읽어 검색만 수행합니다. **재임베딩·재학습 없음**
- 런타임에 인코딩하는 것은 **질의 문장 하나**뿐입니다.
- 점수는 코사인 유사도이며 **위험 확률이나 위법 판정이 아닙니다.**

## 실행

`law-embedding/api` 폴더에서 실행합니다.

```bash
pip install -r requirements.txt
python -m uvicorn app:app --host 127.0.0.1 --port 8002
```

포트는 판례(8001)와 겹치지 않게 **8002**를 기본으로 잡았습니다. 문서는 `http://127.0.0.1:8002/docs`에서 볼 수 있습니다.

### 환경변수

| 변수 | 기본값 | 설명 |
|---|---|---|
| `LAW_EMBEDDING_JSONL` | `../data/law_embeddings.jsonl` | 청크 + 벡터 파일 |
| `LAW_RETRIEVAL_FIX_DIR` | `../retrieval_fix` | 검색 보정 파라미터 폴더. 없으면 보정 없이 동작 |
| `LAW_EMBED_BACKEND` | `local` | `local` 또는 `http` |
| `LAW_EMBED_URL` | - | `http` 모드에서 호출할 `/embed` 주소 |
| `LAW_MODEL_DEVICE` | 자동 | `cpu` 또는 `cuda` |

## 엔드포인트

### `POST /search/laws`

요청:

```json
{ "query": "차임을 1회라도 연체하면 임대인은 즉시 계약을 해지할 수 있다.", "top_k": 5 }
```

| 필드 | 필수 | 설명 |
|---|---|---|
| `query` | O | 계약 조항 또는 질문. 1~5000자, BGE-M3 기준 512토큰 이하 |
| `top_k` | X | 기본 5, 최대 16 |
| `apply_fix` | X | 검색 보정 사용 여부. 생략하면 서버 기본값(보정 사용) |
| `text_chars` | X | 응답 본문 길이 상한. 기본 500 |

응답:

```json
{
  "query": "...",
  "model": "BAAI/bge-m3",
  "model_revision": "3069def033ce91d907258f9a830e442610dbfe0b",
  "ranking_method": "max_chunk_cosine_similarity",
  "retrieval_fix_applied": true,
  "score_note": "검색 유사도이며 법적 위험 확률이나 법률적 적합성 판정이 아닙니다.",
  "application_note": "적용 조건(지역·계약 시점·건물 유형 등)과 참조 조문은 별도로 확인해야 합니다.",
  "count": 5,
  "results": [
    {
      "rank": 1,
      "document_key": "LAW_CIVIL_640",
      "chunk_key": "LAW_CIVIL_640@284415:...:0-73",
      "law_name": "민법",
      "article": "제640조",
      "article_title": "차임연체와 해지",
      "instrument_type": "ACT",
      "source_url": "https://www.law.go.kr/LSW/lsInfoP.do?lsiSeq=284415",
      "source_version_seq": "284415",
      "source_effective_date": "2026-01-02",
      "score": 0.7566,
      "text": "제640조(차임연체와 해지) ...",
      "review_flags": [],
      "application_caution": null
    }
  ]
}
```

같은 조문의 여러 청크 중 **가장 높은 점수**를 그 조문 점수로 쓰고, 결과에서 조문이 중복되지 않게 합칩니다. 판례 API의 `max_chunk_cosine_similarity`와 같은 방식입니다.

### `POST /embed`

```json
{ "texts": ["계약 조항 문구"] }
```

1024차원 정규화 벡터를 돌려줍니다. **서버 메모리가 부족할 때 다른 서비스가 모델을 중복 적재하지 않도록** 두었습니다.

### `GET /health`, `GET /health/ready`

로드된 문서 수·청크 수, 보정 파라미터 유무, 인코딩 백엔드, 기동 실패 사유를 보여줍니다. 임베딩이 로드되지 않으면 검색은 **503**을 반환하고 조용히 빈 결과를 주지 않습니다.

## 서버 메모리 (2 vCPU / RAM 4GB / GPU 없음)

BGE-M3 가중치만 약 2.27GB입니다. **법령 API와 판례 API가 각각 모델을 올리면 4GB 안에 들어가기 어렵습니다.** 둘 중 하나를 고르면 됩니다.

1. 한쪽만 모델을 올리고, 다른 쪽은 `LAW_EMBED_BACKEND=http`로 그 `/embed`를 호출
2. 임베딩 전용 프로세스를 하나 띄우고 두 검색 API가 모두 그쪽을 호출

문서 벡터는 이미 파일에 있어서 **메모리를 차지하는 건 사실상 모델 하나**입니다. 165 × 1024 float32는 약 0.6MB입니다.

## 백엔드 DB 매핑

응답에 들어 있는 안정적인 식별자는 다음과 같습니다.

| 필드 | 성격 | 쓰임 |
|---|---|---|
| `document_key` | 조문 단위 문자열 키 (예: `LAW_CIVIL_640`) | 법령 조문 테이블의 자연키 |
| `chunk_key` | 청크 단위 키. 법령 버전·원문 해시·청킹 규칙 해시 포함 | 인용 단위 추적 |
| `source_version_seq` | 국가법령정보센터 법령 버전 일련번호 | 개정 시 버전 구분 |
| `source_url` | 공식 출처 | 화면 링크 |

- DB의 BIGINT id는 적재할 때 생성하면 되고, **`document_key`에 유니크 제약**을 걸어 매핑하는 방식을 제안합니다. 임베딩 파일의 `db_id`, `db_document_id`는 현재 전부 `null`입니다.
- 화면에 인용할 원문은 `text`, 조문 표기는 `law_name` + `article`을 쓰면 됩니다.
- `review_flags`는 단서·준용·항 참조 때문에 **추가 검토가 필요하다는 표시**이고 오류 표시가 아닙니다.

## 검색 보정

기본으로 켜져 있습니다. 코퍼스 중심화 + 상위 주성분 2개 제거 후 재정규화하는 후처리이고, 질의 24개 평가에서 Hit@1 62.5% → 70.8%, Hit@3 70.8% → 83.3%였습니다. 자세한 내용은 `../evaluation/retrieval_eval_report.md`를 보세요.

보정 파라미터는 **현재 법령 165청크로 계산한 값**이라 코퍼스가 바뀌면 다시 계산해야 합니다. 끄려면 요청에 `"apply_fix": false`를 넣으면 됩니다.

## 테스트

`law-embedding` 폴더에서 실행합니다.

```bash
pip install pytest
python -m pytest tests -q
```

모델을 내려받지 않고 가짜 인코더로 스키마·중복 제거·입력 검증·보정 토글·503 처리를 확인합니다. (11개 통과)

## 아직 안 한 것

- 벡터 DB(pgvector) 적재
- 실제 서버 배포 및 프로세스 관리
- 판례 벡터에 대한 동일 보정 적용
- 검색 결과의 법적 적합성 검수
