"""Build Chapter 1 Qwen3 tiny from the first two layers of pinned real 8B weights."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

from huggingface_hub import snapshot_download
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoTokenizer

SOURCE_REPO = 'Qwen/Qwen3-8B'
SOURCE_REVISION = 'b968826d9c46dd6066d109eabc6255188de91218'
MODEL_REVISION = f'{SOURCE_REPO}@{SOURCE_REVISION}:layers-0-1'
COURSE_ROOT = Path(__file__).resolve().parents[3]
SOURCE_SNAPSHOT = COURSE_ROOT / 'models/qwen3-8b-source'
PRACTICE_SNAPSHOT = COURSE_ROOT / 'models/qwen3-tiny'
SELECTED_LAYERS = (0, 1)


def selected_weight(name):
    return name in {'model.embed_tokens.weight', 'model.norm.weight', 'lm_head.weight'} or any(
        name.startswith(f'model.layers.{i}.') for i in SELECTED_LAYERS)


def make_fixture(out=PRACTICE_SNAPSHOT, config=None, *, source_snapshot=SOURCE_SNAPSHOT, download=True):
    """Copy selected real tensors unchanged; preserve tokenizer and source provenance.

    The first call downloads only source shards containing selected tensors.
    Subsequent calls can use download=False with the cached source files.
    An existing matching extracted checkpoint is reused; unrelated outputs fail.
    """
    out, source_snapshot = Path(out), Path(source_snapshot)
    provenance_path = out / 'provenance.json'
    if out.exists() and any(out.iterdir()):
        if provenance_path.is_file():
            provenance = json.loads(provenance_path.read_text())
            if provenance.get('model_revision') == MODEL_REVISION:
                if all((out / name).is_file() for name in provenance['output_shards']) and (out / 'tokens.json').is_file():
                    return out
        raise FileExistsError(f'Preserve existing output; choose an empty directory: {out}')
    if download:
        snapshot_download(SOURCE_REPO, revision=SOURCE_REVISION, local_dir=source_snapshot,
                          allow_patterns=['*.json', '*.jinja', '*.txt'])
    data = json.loads((source_snapshot / 'config.json').read_text())
    index = json.loads((source_snapshot / 'model.safetensors.index.json').read_text())
    selected = {name: filename for name, filename in index['weight_map'].items() if selected_weight(name)}
    if len(selected) != 25:
        raise ValueError(f'Expected 25 tensors for the two-layer practice model, got {len(selected)}')
    source_shards = sorted(set(selected.values()))
    if download:
        snapshot_download(SOURCE_REPO, revision=SOURCE_REVISION, local_dir=source_snapshot,
                          allow_patterns=source_shards)
    if config is not None:
        expected = (2, data['hidden_size'], data['intermediate_size'], data['num_attention_heads'],
                    data['num_key_value_heads'], data['head_dim'], data['vocab_size'])
        actual = tuple(getattr(config, key) for key in
                       ('layers', 'hidden', 'intermediate', 'q_heads', 'kv_heads', 'head_dim', 'vocab'))
        if actual != expected:
            raise ValueError('Practice dimensions must match the first two real Qwen3-8B layers.')
    out.mkdir(parents=True, exist_ok=True)
    weight_map, total_bytes, output_hashes = {}, 0, {}
    for number, source_name in enumerate(source_shards, 1):
        destination_name = f'model-{number:05d}-of-{len(source_shards):05d}.safetensors'
        with safe_open(source_snapshot / source_name, framework='pt', device='cpu') as shard:
            tensors = {name: shard.get_tensor(name) for name in selected if selected[name] == source_name}
            total_bytes += sum(t.numel() * t.element_size() for t in tensors.values())
            save_file(tensors, str(out / destination_name), metadata={'format': 'pt'})
        del tensors
        with (out / destination_name).open('rb') as stream:
            output_hashes[destination_name] = hashlib.file_digest(stream, 'sha256').hexdigest()
        weight_map.update({name: destination_name for name in selected if selected[name] == source_name})
    data['num_hidden_layers'] = 2
    if 'layer_types' in data:
        data['layer_types'] = data['layer_types'][:2]
    (out / 'config.json').write_text(json.dumps(data, indent=2) + '\n')
    (out / 'model.safetensors.index.json').write_text(json.dumps(
        {'metadata': {'total_size': total_bytes}, 'weight_map': weight_map}, indent=2) + '\n')
    for filename in ['tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json',
                     'added_tokens.json', 'vocab.json', 'merges.txt', 'chat_template.jinja', 'generation_config.json']:
        if (source_snapshot / filename).is_file():
            shutil.copy2(source_snapshot / filename, out / filename)
    tokenizer = AutoTokenizer.from_pretrained(out, local_files_only=True)
    messages = [{'role': 'user', 'content': 'Explain why a KV cache speeds up decoding.'}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=True, return_dict=False,
                                          add_generation_prompt=True, enable_thinking=False)
    continuation_text = 'A KV cache retains'
    continuation = tokenizer.encode(continuation_text, add_special_tokens=False)
    (out / 'tokens.json').write_text(json.dumps({
        'kind': 'real Qwen3 tokenizer; teacher-forced two-layer practice',
        'messages': messages, 'enable_thinking': False, 'add_generation_prompt': True,
        'tokenizer_revision': SOURCE_REVISION, 'continuation_text': continuation_text,
        'prompt_ids': prompt, 'continuation_ids': continuation,
    }, indent=2) + '\n')
    provenance_path.write_text(json.dumps({
        'model_revision': MODEL_REVISION, 'source_repo': SOURCE_REPO,
        'source_revision': SOURCE_REVISION, 'selected_layers': list(SELECTED_LAYERS),
        'source_tensors': selected, 'output_shards': output_hashes,
        'weight_transformation': 'none; tensor values and storage dtypes copied unchanged',
        'scope': 'Two-layer truncation for implementation practice; not full-model quality',
    }, indent=2) + '\n')
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=PRACTICE_SNAPSHOT)
    parser.add_argument('--source-snapshot', type=Path, default=SOURCE_SNAPSHOT)
    parser.add_argument('--offline', action='store_true', help='Use already downloaded source shards')
    args = parser.parse_args()
    path = make_fixture(args.out, source_snapshot=args.source_snapshot, download=not args.offline)
    print(f'Prepared {MODEL_REVISION} in {path}')


if __name__ == '__main__':
    main()
