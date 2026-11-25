"""Draft internal inference API. No trained model -> 503, never fallback to summaries."""
import json
import os
import secrets
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock
from typing import Literal
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator
from api.legacy import legacy_router, load_legacy_from_env
from api.internal import router as internal_router, load_embedder


class ClauseRequest(BaseModel):
    model_config = ConfigDict(extra='forbid', str_strip_whitespace=True)
    clause_id: str = Field(min_length=1, max_length=128)
    text: str = Field(min_length=1, max_length=10000)
    domain: Literal['residential_rental', 'registered_private_rental']

    @field_validator('text')
    @classmethod
    def nonblank(cls, value):
        if not value.strip():
            raise ValueError('text must not be blank')
        return value


class ClauseResponse(BaseModel):
    clause_id: str
    is_risky_candidate: bool
    model_version: str
    needs_human_review: Literal[True] = True
    risk_type: None = None
    risk_level: None = None
    probability_model_score: None = None
    research_only: Literal[True] = True


class LocalPredictor:
    def __init__(self, path):
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        self.torch = torch
        self.config = json.loads((path/'inference_config.json').read_text(encoding='utf-8'))
        evaluation = json.loads((path/'evaluation.json').read_text(encoding='utf-8'))
        if evaluation.get('evaluation_scope') != 'development_validation_not_independent_test':
            raise ValueError('Unsupported artifact provenance')
        self.domains = set(evaluation['by_domain'])
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(path, local_files_only=True).eval()
        if self.model.config.id2label != {0:'NORMAL_CANDIDATE',1:'RISK_CANDIDATE'}:
            raise ValueError('Unexpected class mapping')
        self.lock = Lock()

    def predict(self, text, domain):
        if domain not in self.domains:
            raise ValueError('Domain not represented in validation')
        inputs = self.tokenizer(text, return_tensors='pt', truncation=False)
        if inputs['input_ids'].shape[1] > self.config['max_length']:
            raise ValueError('Clause exceeds model token limit; review clause segmentation')
        with self.lock, self.torch.inference_mode():
            score = self.model(**inputs).logits.softmax(dim=-1)[0, 1].item()
        return bool(score >= self.config['threshold']), self.config['model_version']


def create_app(predictor=None):
    @asynccontextmanager
    async def lifespan(app):
        app.state.api_key = os.getenv("CONTRACTCHECK_INTERNAL_API_KEY", "")
        app.state.demo = os.getenv("CONTRACTCHECK_BATCH_MODE", "off") == "research-demo"
        if os.getenv("CONTRACTCHECK_BATCH_MODE", "off") not in {"off", "research-demo"}:
            raise ValueError("Unknown CONTRACTCHECK_BATCH_MODE")
        if app.state.demo and os.getenv("CONTRACTCHECK_LEGACY_FOLD") != "1":
            raise ValueError("Batch demo requires the fold 1 checkpoint")
        embedding = os.getenv("CONTRACTCHECK_EMBED_ENABLED", "0")
        if embedding not in {"0", "1"}:
            raise ValueError("CONTRACTCHECK_EMBED_ENABLED must be 0 or 1")
        app.state.predictor = predictor
        path = os.environ.get('CONTRACTCHECK_MODEL_DIR')
        if predictor is None and path:
            # A configured broken model fails startup instead of hiding deployment errors.
            app.state.predictor = LocalPredictor(Path(path))
        app.state.legacy_predictor = load_legacy_from_env()
        app.state.embedder = load_embedder() if embedding == "1" else None
        app.state.embed_request_lock = Lock()
        yield

    app = FastAPI(title='ContractCheck research inference (draft)', version='0.1.0', lifespan=lifespan)
    app.include_router(legacy_router)
    app.include_router(internal_router)

    @app.middleware("http")
    async def protect_inference(request, call_next):
        if request.method == "POST":
            expected = request.app.state.api_key
            if not expected.strip() or expected == "unset":
                return JSONResponse({"detail": {"code": "INTERNAL_KEY_NOT_CONFIGURED"}}, status_code=503)
            supplied = request.headers.getlist("x-internal-api-key")
            if len(supplied) != 1 or not secrets.compare_digest(supplied[0].encode(), expected.encode()):
                return JSONResponse({"detail": {"code": "UNAUTHORIZED"}}, status_code=401)
            try:
                return await call_next(request)
            except Exception:
                # Third-party tokenizer/model exception messages can contain request text.
                return JSONResponse({"detail": {"code": "INFERENCE_FAILED"}}, status_code=500)
        return await call_next(request)

    @app.exception_handler(RequestValidationError)
    async def invalid_input(request: Request, error: RequestValidationError):
        # Validation details may include the rejected input, even for malformed JSON.
        return JSONResponse({"detail": [{"type": e["type"]} for e in error.errors()]}, status_code=422)

    @app.get('/health/live')
    def live():
        return {'status': 'alive'}

    @app.get('/health/ready')
    def ready():
        if app.state.predictor is None:
            raise HTTPException(503, 'No reviewed-data model configured')
        return {'status': 'ready', 'research_only': True}

    @app.post('/v1/clauses/classify', response_model=ClauseResponse)
    def classify(request: ClauseRequest):
        if app.state.predictor is None:
            raise HTTPException(503, 'No reviewed-data model configured')
        try:
            risky, version = app.state.predictor.predict(request.text, request.domain)
        except ValueError as e:
            raise HTTPException(422, str(e)) from e
        return ClauseResponse(clause_id=request.clause_id, is_risky_candidate=risky, model_version=version)

    return app


app = create_app()
