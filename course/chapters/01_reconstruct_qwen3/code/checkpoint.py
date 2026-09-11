"""Strict, shard-at-a-time loader for the supported dense, unscaled Qwen3 structure."""
import json
from pathlib import Path
from safetensors import safe_open
import torch
from lab import Config,TinyQwen


def configuration(snapshot):
    data=json.loads((Path(snapshot)/'config.json').read_text())
    if data.get('model_type')!='qwen3' or data.get('tie_word_embeddings') or data.get('attention_bias'):
        raise ValueError('only untied, bias-free dense Qwen3 is supported')
    if data.get('hidden_act','silu')!='silu' or data.get('quantization_config') or data.get('use_sliding_window'):
        raise ValueError('unsupported activation, quantized checkpoint, or sliding window')
    if any(kind!='full_attention' for kind in data.get('layer_types',[])):
        raise ValueError('all layers must use full attention')
    rope=data.get('rope_parameters') or data.get('rope_scaling') or {}
    if rope.get('rope_type',rope.get('type','default'))!='default':
        raise ValueError('scaled RoPE requires a separate implementation')
    c=Config(data['num_hidden_layers'],data['hidden_size'],data['intermediate_size'],
             data['num_attention_heads'],data['num_key_value_heads'],data['head_dim'],
             data['vocab_size'],data['rms_norm_eps'],rope.get('rope_theta',data.get('rope_theta',1e6)))
    if c.q_heads%c.kv_heads or c.head_dim%2:
        raise ValueError('invalid GQA/rotary dimensions')
    return c,data


def mapping(c):
    names={'model.embed_tokens.weight':'embed.weight','model.norm.weight':'norm.weight','lm_head.weight':'head.weight'}
    suffixes={'input_layernorm':'input_norm','post_attention_layernorm':'post_norm',
              'self_attn.q_norm':'q_norm','self_attn.k_norm':'k_norm'}
    suffixes.update({f'self_attn.{name}_proj':name for name in ['q','k','v','o']})
    suffixes.update({f'mlp.{name}_proj':name for name in ['gate','up','down']})
    for layer in range(c.layers):
        for original,destination in suffixes.items():
            names[f'model.layers.{layer}.{original}.weight']=f'layers.{layer}.{destination}.weight'
    return names


def load_custom(snapshot,device='cpu',dtype=torch.float32,audit_only=False):
    snapshot=Path(snapshot)
    c,data=configuration(snapshot)
    with torch.device('meta'):
        model=TinyQwen(c).to(dtype=dtype)
    names=mapping(c)
    params=dict(model.named_parameters())
    index=snapshot/'model.safetensors.index.json'
    if index.exists():
        indexed=json.loads(index.read_text())['weight_map']
        filenames=sorted(set(indexed.values()))
    else:
        indexed=None
        filenames=['model.safetensors']
    seen,rows=set(),[]
    for filename in filenames:
        file=snapshot/filename
        if file.resolve().parent!=snapshot.resolve():
            raise ValueError('shards must be files in the snapshot directory')
        with safe_open(file,framework='pt',device='cpu') as shard:
            for name in shard.keys():
                if name in seen or name not in names:
                    raise ValueError(f'duplicate or unmapped checkpoint tensor: {name}')
                if indexed is not None and indexed.get(name)!=filename:
                    raise ValueError(f'index/header disagreement: {name}')
                shape=tuple(shard.get_slice(name).get_shape())
                if shape!=tuple(params[names[name]].shape):
                    raise ValueError(f'shape mismatch for {name}: {shape} vs {tuple(params[names[name]].shape)}')
                stored_dtype=shard.get_slice(name).get_dtype()
                if stored_dtype not in ['F32','F16','BF16']:
                    raise ValueError(f'unsupported storage dtype: {stored_dtype}')
                seen.add(name)
                rows.append(dict(checkpoint_name=name,engine_name=names[name],shape=list(shape),
                                 numel=params[names[name]].numel(),stored_dtype=stored_dtype,shard=filename))
    if seen!=set(names) or (indexed is not None and seen!=set(indexed)):
        raise ValueError(f'checkpoint coverage mismatch; missing={sorted(set(names)-seen)}')
    if sum(row['numel'] for row in rows)!=c.parameters():
        raise ValueError('parameter inventory disagrees with architecture')
    if audit_only:
        return None,rows
    model.to_empty(device=device)
    params=dict(model.named_parameters())
    with torch.no_grad():
        for filename in filenames:
            with safe_open(snapshot/filename,framework='pt',device='cpu') as shard:
                for name in shard.keys():
                    params[names[name]].copy_(shard.get_tensor(name))
    return model.eval(),rows
