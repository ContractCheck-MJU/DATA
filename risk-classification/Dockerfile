# 위험 분류 추론 서버(FastAPI) 배포 이미지. CPU 전용이다.
# 모델 가중치는 이미지에 넣지 않는다. 실행할 때 체크포인트 폴더를 아래 경로에 읽기 전용으로 연결한다.
#   /app/experiments/legacy_roberta/runs/cv_20261116_seed42/fold_<N>/checkpoint
# 빌드: risk-classification 폴더를 컨텍스트로 사용한다.
#   docker build -t contractcheck-inference risk-classification
FROM python:3.12-slim

# 헬스체크(Coolify)에 curl이 필요하다.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# 버전은 전달용 모델 패키지(ContractCheck_Backend_Fold1)의 requirements.txt와 같다.
# PyTorch는 CPU 전용 휠을 받는다. 기본 휠은 GPU 라이브러리를 포함해 이미지가 수 GB 커진다.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.15.0 \
    && pip install --no-cache-dir \
        fastapi==0.163.1 \
        pydantic==2.14.3 \
        uvicorn==0.54.0 \
        transformers==4.57.6 \
        numpy==2.5.4 \
        safetensors==0.8.0

WORKDIR /app
COPY api ./api

# 서버 CPU(2코어)를 다른 서비스와 나눠 쓰므로 PyTorch 스레드를 제한한다.
# CONTRACTCHECK_LEGACY_FOLD가 있으면 시작할 때 해당 fold 체크포인트를 읽고, 없거나 깨졌으면 시작에 실패한다.
ENV PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=2 \
    MKL_NUM_THREADS=2 \
    CONTRACTCHECK_LEGACY_FOLD=1

RUN useradd --create-home --uid 10001 app
USER app

EXPOSE 8000
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
