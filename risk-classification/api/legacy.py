"""Opt-in inference for existing legacy weak-label fold checkpoints."""
import json
import math
import os
from pathlib import Path
from threading import Lock
from typing import Literal

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

RUN = Path(__file__).resolve().parents[1] / 'experiments/legacy_roberta/runs/cv_20261116_seed42'
LABELS = {0: 'AFTER_CORRECTION_SUMMARY_WEAK', 1: 'BEFORE_CORRECTION_SUMMARY_WEAK'}
SCORES_NOTE = ('Model trained on FTC correction case summaries with weak labels. '
               'Scores are uncalibrated class outputs and do not measure legal risk or reliability.')


class LegacyRequest(BaseModel):
    model_config = ConfigDict(extra='forbid', str_strip_whitespace=True)
    clause_id: str = Field(min_length=1, max_length=128)
    text: str = Field(min_length=1, max_length=10000)


class LegacyResponse(BaseModel):
    clause_id: str
    model_version: str
    fold: int = Field(ge=1, le=5)
    weak_label: Literal[0, 1]
    weak_label_name: Literal['AFTER_CORRECTION_SUMMARY_WEAK', 'BEFORE_CORRECTION_SUMMARY_WEAK']
    before_summary_model_score: float = Field(ge=0, le=1)
    input_tokens: int
    research_only: Literal[True] = True
    warning: str = SCORES_NOTE


class LegacyPredictor:
    def __init__(self, fold: int):
        path = RUN / f'fold_{fold}' / 'checkpoint'
        marker = json.loads((path/'EXPLORATORY_ONLY.json').read_text(encoding='utf-8'))
        config = marker['config']
        if (marker.get('service_eligible') is not False or marker.get('fold') != fold
                or marker.get('source_type') != 'ftc_correction_summary_before_after'
                or config.get('mode') != 'cv' or config.get('run_name') != RUN.name
                or config.get('pretrained', {}).get('model_id') != 'klue/roberta-base'):
            raise ValueError('Unexpected legacy checkpoint provenance')
        self.max_length = config['max_length']
        if type(self.max_length) is not int or not 8 <= self.max_length <= 512:
            raise ValueError('Invalid legacy checkpoint token limit')

        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        self.model, loading = AutoModelForSequenceClassification.from_pretrained(
            path, local_files_only=True, use_safetensors=True, output_loading_info=True,
        )
        if any(loading.get(key) for key in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')):
            raise ValueError('Legacy checkpoint does not contain the complete expected model')
        if self.model.config.model_type != 'roberta' or self.model.config.id2label != LABELS:
            raise ValueError('Unexpected legacy architecture or weak-label mapping')
        self.model.eval()
        self.torch = torch
        self.fold = fold
        self.version = f'{RUN.name}/fold_{fold}'
        self.lock = Lock()

    def classify(self, request: LegacyRequest) -> LegacyResponse:
        with self.lock, self.torch.inference_mode():
            inputs = self.tokenizer(request.text, return_tensors='pt', truncation=False)
            length = inputs['input_ids'].shape[1]
            if length > self.max_length:
                raise HTTPException(422, detail={'code': 'TOKEN_LIMIT_EXCEEDED', 'max_tokens': self.max_length})
            logits = self.model(**inputs).logits[0]
            label = int(logits.argmax().item())
            score = logits.softmax(dim=-1)[1].item()
        if not math.isfinite(score):
            raise HTTPException(500, detail={'code': 'INVALID_MODEL_OUTPUT'})
        return LegacyResponse(clause_id=request.clause_id, model_version=self.version, fold=self.fold,
                              weak_label=label, weak_label_name=LABELS[label],
                              before_summary_model_score=score, input_tokens=length)


def load_legacy_from_env():
    fold = os.environ.get('CONTRACTCHECK_LEGACY_FOLD')
    if fold is None:
        return None
    if fold not in ('1', '2', '3', '4', '5'):
        raise ValueError('CONTRACTCHECK_LEGACY_FOLD must be 1, 2, 3, 4, or 5')
    return LegacyPredictor(int(fold))


legacy_router = APIRouter(prefix='/dev/v1/legacy', tags=['Development / legacy weak labels'])


def require_legacy(request: Request) -> LegacyPredictor:
    if request.app.state.legacy_predictor is None:
        raise HTTPException(503, detail={'code': 'LEGACY_MODEL_DISABLED'})
    return request.app.state.legacy_predictor


@legacy_router.get('/health/ready', responses={503: {'description': 'Legacy model disabled'}})
def legacy_ready(request: Request):
    predictor = require_legacy(request)
    return {'status': 'ready', 'model_version': predictor.version, 'fold': predictor.fold,
            'max_tokens': predictor.max_length, 'research_only': True}


@legacy_router.post('/clauses/classify', response_model=LegacyResponse,
                    responses={503: {'description': 'Legacy model disabled'}})
def legacy_classify(payload: LegacyRequest, request: Request):
    return require_legacy(request).classify(payload)
