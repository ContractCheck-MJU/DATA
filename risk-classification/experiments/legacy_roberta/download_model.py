"""Download only the official pretrained checkpoint into the project-local cache."""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault('HF_HOME', str(ROOT/'.cache/huggingface'))
os.environ.setdefault('HF_HUB_DISABLE_XET', '1')


def main():
    from huggingface_hub import HfApi, snapshot_download
    revision = HfApi().model_info('klue/roberta-base').sha
    dest = ROOT/'.cache/klue-roberta-base'
    snapshot_download('klue/roberta-base', revision=revision, local_dir=dest,
                      allow_patterns=['config.json', 'model.safetensors', 'tokenizer*.json',
                                      'special_tokens_map.json', 'vocab.txt'])
    if not (dest/'model.safetensors').exists():
        raise RuntimeError('Expected safetensors checkpoint missing')
    (dest/'download_provenance.json').write_text(json.dumps({
        'model_id':'klue/roberta-base', 'revision':revision,
        'purpose':'legacy summary weak-label experiment only'
    }, indent=2), encoding='utf-8')
    print(dest, revision, flush=True)


if __name__ == '__main__':
    main()
