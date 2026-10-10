import torch
from transformers import DynamicCache, StaticCache

def create_prefill_inputs(batch_size, seq_len, hidden_size, dtype, device):
    pseudo_input = torch.randn(batch_size, seq_len, hidden_size, dtype=dtype, device=device)
    pseudo_position_ids = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0).expand(batch_size, -1)
    return pseudo_input, pseudo_position_ids


def create_decode_inputs_without_cache(batch_size, current_seq_len, hidden_size, dtype, device):
    pseudo_input = torch.randn(batch_size, 1, hidden_size, dtype=dtype, device=device)
    pseudo_position_ids = torch.arange(current_seq_len, current_seq_len + 1, dtype=torch.long, device=device).unsqueeze(0).expand(batch_size, -1)
    prefill_hidden_states = torch.randn(batch_size, current_seq_len, hidden_size, dtype=dtype, device=device)
    return pseudo_input, pseudo_position_ids, prefill_hidden_states


def create_decode_inputs_with_cache(batch_size, current_seq_len, hidden_size, dtype, device):
    pseudo_input = torch.randn(batch_size, 1, hidden_size, dtype=dtype, device=device)
    pseudo_cache_positions = torch.arange(current_seq_len, current_seq_len + 1, dtype=torch.long, device=device)
    pseudo_position_ids = pseudo_cache_positions.unsqueeze(0)
    return pseudo_input, pseudo_position_ids, pseudo_cache_positions


def get_position_embeddings(hidden_states, position_ids, rotary_emb, layer_type):
    return rotary_emb(hidden_states, position_ids, layer_type=layer_type)


def initialize_kv_cache(config, max_cache_len):
    # return DynamicCache(config=config)
    return StaticCache(config=config, max_cache_len=max_cache_len)

@torch.inference_mode()
def prefill_forward(layer, hidden_states, position_ids, position_embeddings, past_key_values):
    # Create cache_position for prefill stage (0 to seq_len-1)
    cache_position = torch.arange(hidden_states.shape[1], device=hidden_states.device)
    return layer(
        hidden_states=hidden_states, 
        position_embeddings=position_embeddings,
        position_ids=position_ids,
        past_key_values=past_key_values,
        cache_position=cache_position,
        use_kv_cache=True
    )

@torch.inference_mode()
def decode_forward(layer, hidden_states, position_ids, position_embeddings, use_kv_cache, past_key_values, cache_positions, prefill_hidden_states):
    return layer(
        hidden_states=hidden_states, 
        position_embeddings=position_embeddings, 
        position_ids=position_ids,
        past_key_values=past_key_values,
        cache_positions=cache_positions,
        use_kv_cache=use_kv_cache,
        prefill_hidden_states=prefill_hidden_states
    )

@torch.inference_mode()
def multi_layer_prefill_forward(num_layers, layer, hidden_states, position_ids, position_embeddings, past_key_values):
    for _ in range(num_layers):
        hidden_states = prefill_forward(layer, hidden_states, position_ids, position_embeddings, past_key_values)
    return hidden_states

@torch.inference_mode()
def multi_layer_decode_forward(num_layers, layer, hidden_states, position_ids, position_embeddings, use_kv_cache, past_key_values, cache_positions, prefill_hidden_states):
    for _ in range(num_layers):
        hidden_states = decode_forward(layer, hidden_states, position_ids, position_embeddings, use_kv_cache, past_key_values, cache_positions, prefill_hidden_states)
    return hidden_states