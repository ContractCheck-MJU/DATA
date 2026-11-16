# 법령 임베딩 (RAG 검색용 데이터)

임대차 계약서 위험 분석 서비스의 **RAG 근거 제공**에 쓰려고 만든 법령 데이터입니다.
공식 법령 원문을 수집 → 조문 단위 청킹 → BGE-M3 임베딩까지 마친 상태이고, **벡터 DB 적재는 하지 않았습니다.**

- 법령 문서 **95건** → 청크 **165개** → 벡터 **165 × 1024**
- 임베딩 모델: `BAAI/bge-m3` (revision `3069def033ce91d907258f9a830e442610dbfe0b`)
- 풀링: CLS(last hidden state), L2 정규화 완료 → **내적 = 코사인 유사도**
- 담당: 서제현 (법령 파트) / 판례·위험조항 분류는 별도 파트

---

## 1. 폴더 구조

```
law-embedding/
├── README.md
├── data/
│   ├── 법령_임베딩.jsonl         # 핵심 산출물: 청크 메타데이터 + 1024차원 벡터 (165줄, 4.4MB)
│   ├── 법령_벡터.npy             # 벡터만 분리 (float32, shape 165×1024)
│   ├── 법령_벡터인덱스.jsonl      # npy의 행 번호 ↔ chunk_key 대응표
│   ├── 법령_chunks.jsonl         # 임베딩 전 청킹 결과 (벡터 없음)
│   └── 법령_원문95건.jsonl        # 청킹 입력이 된 법령 원문 95건
├── validation/
│   ├── 법령_임베딩검증.json       # 벡터 개수·차원·정규화·재현성 검증 결과
│   ├── 법령_청킹검증.json         # 청킹 검증 결과 (원문 복원 일치 등)
│   ├── 법령_문서별청킹통계.json    # 문서별 청크 수·토큰 수
│   └── 법령_청크검수.csv          # 청크별 검토 표시 목록
├── examples/
│   ├── 법령_검색예시.csv          # 계약 문구 12개 × 상위 5개 검색 결과 (60행)
│   └── 법령_검색예시.txt          # 같은 내용을 사람이 읽기 좋은 형태로
└── code/
    ├── 법령_청킹.py               # 원문 → 청크 생성 + 검증
    ├── 법령_임베딩.py             # 청크 → 벡터 생성 + 검색 + 검증
    ├── 법령_임베딩_검색_Colab.ipynb # Colab에서 재현·검색 테스트용 (출력 비움)
    └── requirements.txt
```

**백엔드에서 쓸 파일은 `data/법령_임베딩.jsonl` 하나면 됩니다.** 나머지는 재현·검증·참고용입니다.

---

## 2. `법령_임베딩.jsonl` 주요 필드

한 줄 = 청크 1개(JSON 1개). 주요 필드만 적습니다.

| 필드 | 설명 |
|---|---|
| `chunk_key` | 청크 고유 키 (`document_key@버전:원문해시:규칙해시:시작-끝`) |
| `document_key` | 법령 문서 키 (예: `LAW_HOUSING_2`) |
| `chunk_no` | 문서 내 청크 순번 |
| `text` | **공식 원문 그대로의 부분 문자열** (사용자에게 보여줄 근거 원문) |
| `embedding_text` | **실제 임베딩에 넣은 문자열** (법령명·조문 제목 등 문맥이 앞에 붙음) |
| `embedding` | 1024개 float (L2 정규화 완료) |
| `vector_row` | `법령_벡터.npy`의 행 번호 |
| `law_name` / `article` / `article_title` | 법령명 / 조문 번호 / 조문 제목 |
| `source_url` | 국가법령정보센터 공식 출처 |
| `source_version_seq` / `source_effective_date` | 법령 버전 일련번호 / 시행일 |
| `instrument_type` | `ACT`(법률), 시행령 등 구분 |
| `token_count` | BGE-M3 토크나이저 기준 토큰 수 (최대 507) |
| `review_flags` | 단서·준용·항 참조 등 **추가 검토 표시** (오류 표시가 아님) |
| `reference_mentions` | 본문에 나온 참조 조문 후보 (자동 해결하지 않음) |
| `application_caution` | 해당 조문 적용 시 확인해야 할 조건 메모 |
| `db_id` / `db_document_id` | DB 적재 시 채울 자리 (현재 모두 `null`) |

> `text`와 `embedding_text`는 다릅니다. **화면에 인용할 때는 `text`**, 검색 입력 재현에는 `embedding_text`를 쓰세요.

---

## 3. 사용 방법

### 3-1. 그대로 읽어서 검색하기 (벡터 DB 없이)

165개 규모라 numpy만으로 충분히 동작합니다.

```python
import json, numpy as np, torch, torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

MODEL_ID, REV = "BAAI/bge-m3", "3069def033ce91d907258f9a830e442610dbfe0b"

rows = [json.loads(l) for l in open("data/법령_임베딩.jsonl", encoding="utf-8") if l.strip()]
vectors = np.array([r["embedding"] for r in rows], dtype=np.float32)   # (165, 1024)

tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=REV, use_fast=True)
model = AutoModel.from_pretrained(MODEL_ID, revision=REV).eval()

def embed(text: str) -> np.ndarray:
    enc = tok([text], return_tensors="pt", truncation=False)
    with torch.inference_mode():
        h = model(**enc).last_hidden_state
    return F.normalize(h[:, 0, :].float(), p=2, dim=1)[0].numpy()

def search(query: str, top_k: int = 5):
    scores = vectors @ embed(query)          # 정규화되어 있으므로 내적 = 코사인
    for i in np.argsort(-scores)[:top_k]:
        r = rows[int(i)]
        print(f'{scores[i]:.4f}  {r["law_name"]} {r["article"]}  {r["source_url"]}')
        print(r["text"][:120])

search("월세를 한 번만 연체해도 즉시 계약을 해지한다.")
```

### 3-2. 벡터 DB(pgvector 등)에 넣는 경우

- 컬럼 타입은 `vector(1024)`
- 거리 연산자는 **코사인** 기준으로 맞추면 됩니다 (이미 L2 정규화 완료라 내적도 동일)
- 검색 쿼리도 **같은 모델·같은 revision·같은 CLS 풀링 + L2 정규화**로 임베딩해야 합니다. 다르면 점수가 전부 틀어집니다.

### 3-3. 재현

`code/법령_임베딩_검색_Colab.ipynb`를 Colab에서 `런타임 → 모두 실행` 하면 모델을 내려받아 임베딩을 다시 만들고, 검색 입력칸으로 직접 테스트할 수 있습니다. GPU(T4) 기준 임베딩 생성에 약 37초 걸렸습니다. PC 설치나 유료 API 키는 필요 없습니다.

---

## 4. 검증한 내용

| 항목 | 결과 |
|---|---|
| 청크 → 원문 복원 | 95건 전부 원문과 일치 |
| 청크 토큰 수 | 최대 507 (목표 512 초과 0건, 모델 한계 8192 초과 0건) |
| 벡터 형상 | 165 × 1024, float32 |
| 비정상 값 | NaN/inf 0, 영벡터 0 |
| L2 노름 | 0.99999994 ~ 1.00000012 |
| 같은 입력 재인코딩 | 코사인 0.9999989 (재현성 확인) |
| 청킹 토큰 수 ↔ 임베딩 입력 토큰 수 | 일치 |

상세 수치는 `validation/` 폴더의 JSON을 보시면 됩니다.

---

## 5. 주의사항 / 아직 안 한 것

- **유사도 점수는 위험 확률이 아닙니다.** 코사인 유사도일 뿐이고, 점수가 높다고 해당 조항이 위법·무효라는 뜻이 아닙니다.
- **검색 관련성은 아직 사람이 검수하지 않았습니다.** 실제로 일부 질의에서 직접 관련 없는 조문이 1위로 나오는 것을 확인했습니다. (예: "임대인이 월세를 일방적으로 인상한다" → 민법 제626조가 1위, 더 관련 있는 주택임대차보호법 제7조는 3위) 개선 여지가 있는 상태로 공유합니다.
- **판례는 포함되어 있지 않습니다.** 이 레포 폴더는 법령 95건만 다룹니다.
- **벡터 DB 적재는 하지 않았습니다.** `db_id`, `db_document_id`는 전부 `null`입니다.
- **참조 조문 자동 해결은 하지 않았습니다.** `reference_mentions`는 후보일 뿐이라, "제2조"가 같은 법인지 인용된 다른 법인지 확인되지 않았습니다.
- `review_flags`가 붙은 청크 99개는 **오류가 아니라** 단서·준용·항 참조 때문에 추가 검토가 필요하다는 표시입니다.
- 조문 적용 여부는 지역 조례, 계약 시점, 담보물권 취득 시점, 건물 유형 등에 따라 달라집니다. **법적 판단 근거로 바로 쓰지 말고 조건을 확인해야 합니다.**

---

## 6. 백엔드 연동 메모

- ERD 설계대로 **분류 API와 법령 검색은 각각 호출**하는 구조를 전제로 했습니다. 이 데이터는 분류 모델의 학습 라벨이 아니라 **검색 근거 원문**입니다.
- 사용자에게 보여줄 인용문은 `text`, 출처 링크는 `source_url`, 조문 표기는 `law_name` + `article`을 쓰면 됩니다.
- 검색 API가 필요하면 위 3-1 코드를 그대로 감싸면 됩니다. 아직 API 서버는 만들지 않았습니다.
