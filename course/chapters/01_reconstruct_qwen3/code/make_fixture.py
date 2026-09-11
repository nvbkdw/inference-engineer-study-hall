"""Create a tiny randomly initialized HF Qwen3 checkpoint for offline loader practice."""
import argparse
import json
from pathlib import Path
import torch
from transformers import Qwen3Config,Qwen3ForCausalLM
from lab import Config


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.out.exists() and any(a.out.iterdir()):
        p.error('output directory must be empty')
    torch.manual_seed(42)
    tiny=Config()
    c=Qwen3Config(hidden_size=tiny.hidden,intermediate_size=tiny.intermediate,
                  num_hidden_layers=tiny.layers,num_attention_heads=tiny.q_heads,
                  num_key_value_heads=tiny.kv_heads,head_dim=tiny.head_dim,
                  vocab_size=tiny.vocab,rms_norm_eps=tiny.eps,tie_word_embeddings=False,
                  rope_parameters={'rope_type':'default','rope_theta':tiny.theta})
    model=Qwen3ForCausalLM(c).eval()
    model.save_pretrained(a.out, max_shard_size='20KB')
    (a.out/'tokens.json').write_text(json.dumps(dict(kind='random tiny fixture; no text quality',
        prompt_ids=[1,4,9,16,25,36,49], continuation_ids=[64,81,2]))+'\n')
    print(f'Created local-tiny-v1 in {a.out}; random weights are not a pretrained model.')


if __name__=='__main__':
    main()
