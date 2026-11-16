"""Offline runtime check: tiny random network only, no contract training/download."""
import json
import platform
import tempfile
from importlib.metadata import version
from pathlib import Path


def main():
    import torch
    from transformers import RobertaConfig, RobertaForSequenceClassification
    torch.manual_seed(42)
    model=RobertaForSequenceClassification(RobertaConfig(vocab_size=32,hidden_size=16,
        num_hidden_layers=1,num_attention_heads=2,intermediate_size=32,num_labels=2,max_position_embeddings=32))
    inputs={'input_ids':torch.tensor([[0,5,6,2],[0,8,9,2]]),'attention_mask':torch.ones(2,4,dtype=torch.long)}
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4)
    loss=model(**inputs,labels=torch.tensor([0,1])).loss
    loss.backward();optimizer.step();model.eval()
    with tempfile.TemporaryDirectory() as d:
        model.save_pretrained(d,safe_serialization=True)
        loaded=RobertaForSequenceClassification.from_pretrained(d,local_files_only=True).eval()
        with torch.inference_mode():
            assert torch.allclose(model(**inputs).logits, loaded(**inputs).logits)
    result={'python':platform.python_version(),'platform':platform.platform(),
            'packages':{p:version(p) for p in ('torch','transformers','scikit-learn','fastapi','uvicorn','httpx')},
            'cuda_available':torch.cuda.is_available(),'torch_cuda':torch.version.cuda,
            'offline_random_network_forward_backward_save_reload':'PASS',
            'klue_weights_downloaded':False,'approved_contract_training_performed':False}
    root=Path(__file__).resolve().parents[1]
    (root/'reports/environment_check.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
