"""Checkpoint preparation and verification utilities for the Chapter 1 notebooks.

The candidate class is supplied by the notebook. The shared loader audits its
parameters; this module owns saved-logit comparison and the tiny timing sweep.
Real checkpoint downloads require an explicit pinned revision and download=True.
"""
import ast
import csv
import gc
import hashlib
import json
import math
import re
import sys
import time
import uuid
from types import ModuleType
from datetime import datetime, timezone
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

import torch
import transformers
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM

from checkpoint import configuration, load_custom
from make_fixture import make_fixture, PRACTICE_SNAPSHOT, MODEL_REVISION

CODE_DIR = Path(__file__).resolve().parent
COURSE_ROOT = CODE_DIR.parents[2]
if str(COURSE_ROOT) not in sys.path:
    sys.path.insert(0, str(COURSE_ROOT))
from shared.experiment import begin, finish, plot, time_cuda


def load_notebook_implementation(notebook_path=CODE_DIR / 'lab.ipynb'):
    """Reuse imports/classes/functions in tagged model cells, skipping lab actions.

    This executes trusted local notebook definitions. Checkpoint downloads,
    demonstrations, verification runs, and measurements are not executed.
    """
    notebook_path = Path(notebook_path).resolve()
    source = notebook_path.read_bytes()
    notebook = json.loads(source)
    module = ModuleType(f'_qwen3_notebook_{uuid.uuid4().hex}')
    module.__file__ = str(notebook_path)
    sys.modules[module.__name__] = module  # dataclasses resolves the defining module.
    try:
        for cell in notebook['cells']:
            if cell['cell_type'] != 'code' or 'qwen3-definition' not in cell.get('metadata', {}).get('tags', []):
                continue
            tree = ast.parse(''.join(cell['source']))
            tree.body = [node for node in tree.body if isinstance(
                node, (ast.Import, ast.ImportFrom, ast.ClassDef, ast.FunctionDef))]
            exec(compile(tree, f"{notebook_path}:{cell['id']}", 'exec'), module.__dict__)
        for name in ('Config', 'Block', 'TinyQwen3', 'model_weight_name_mapping'):
            if not hasattr(module, name):
                raise ValueError(f'Missing notebook definition {name}; check qwen3-definition cell tags.')
        module.TinyQwen3.__notebook_source_path__ = str(notebook_path)
        module.TinyQwen3.__notebook_source_sha256__ = hashlib.sha256(source).hexdigest()
        return module
    except Exception:
        sys.modules.pop(module.__name__, None)
        raise


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def write_csv(path, rows):
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def create_practice_checkpoint(output_root=None):
    """Prepare/reuse the first two real 8B layers; first use downloads source shards."""
    snapshot = PRACTICE_SNAPSHOT if output_root is None else Path(output_root) / 'tiny'
    return make_fixture(snapshot)


def describe_checkpoint(snapshot):
    """Print model metadata, layer names, and weight headers without loading weights."""
    snapshot = Path(snapshot)
    config, data = configuration(snapshot)
    fields = ['model_type', 'architectures', 'num_hidden_layers', 'hidden_size',
              'intermediate_size', 'num_attention_heads', 'num_key_value_heads',
              'head_dim', 'vocab_size', 'hidden_act', 'rms_norm_eps', 'tie_word_embeddings']
    metadata = {key: data.get(key) for key in fields}
    metadata.update(rope_theta=config.theta, parameters=config.parameters())
    if (snapshot / 'provenance.json').is_file():
        metadata['provenance'] = {k: v for k, v in json.loads((snapshot / 'provenance.json').read_text()).items()
                                  if k in ['source_repo', 'source_revision', 'selected_layers', 'scope']}
    layers = ['model.embed_tokens', *[f'model.layers.{i}' for i in range(config.layers)],
              'model.norm', 'lm_head']
    index = snapshot / 'model.safetensors.index.json'
    filenames = (sorted(set(json.loads(index.read_text())['weight_map'].values()))
                 if index.exists() else ['model.safetensors'])
    weights = []
    for filename in filenames:
        shard_path = snapshot / filename
        if shard_path.resolve().parent != snapshot.resolve():
            raise ValueError('Shards must be files in the snapshot directory.')
        with safe_open(shard_path, framework='pt', device='cpu') as shard:
            for name in shard.keys():
                header = shard.get_slice(name)
                weights.append(dict(name=name, shape=header.get_shape(),
                                    dtype=header.get_dtype(), shard=filename))
    weights.sort(key=lambda row: row['name'])
    print('Checkpoint:', snapshot)
    print('\nModel metadata:')
    print(json.dumps(metadata, indent=2))
    print('\nLayers (embedding, decoder stack, final norm, vocabulary head):')
    for layer in layers:
        print(' ', layer)
    print(f'\nCheckpoint weights ({len(weights)} tensors in {len(filenames)} shards):')
    for row in weights:
        print(f"  {row['name']:<52} {str(row['shape']):<14} {row['dtype']}")
    return dict(metadata=metadata, layers=layers, weights=weights)


def download_model_checkpoint(model_case, revision, destination):
    """Download Qwen3-8B/32B weights and tokenizer files at a resolved commit.

    Uses existing Hugging Face authentication/cache settings; no credentials are
    printed. Files are materialized in destination for the strict shard loader.
    """
    from huggingface_hub import snapshot_download

    if model_case not in {'8b', '32b'}:
        raise ValueError('Downloads support only the real 8B and 32B models.')
    if not isinstance(revision, str) or not re.fullmatch(r'[0-9a-fA-F]{40}', revision):
        raise ValueError('Use the resolved 40-character checkpoint commit, not a branch name.')
    return Path(snapshot_download(
        repo_id=f'Qwen/Qwen3-{model_case.upper()}', revision=revision,
        local_dir=Path(destination), allow_patterns=['*.json', '*.safetensors', '*.jinja', '*.txt'],
    ))


def create_text_fixture(snapshot, revision, output_path):
    """Save local-tokenizer inputs and their provenance for both execution paths."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    messages = [{'role': 'user', 'content': 'Explain why a KV cache speeds up decoding.'}]
    continuation = 'A KV cache retains'
    fixture = dict(
        kind='real text, non-thinking, teacher-forced fixture',
        tokenizer_revision=revision, messages=messages, continuation_text=continuation,
        enable_thinking=False, add_generation_prompt=True,
        rendered_prompt=tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False),
        prompt_ids=tokenizer.apply_chat_template(
            messages, tokenize=True, return_dict=False,
            add_generation_prompt=True, enable_thinking=False),
        continuation_ids=tokenizer.encode(continuation, add_special_tokens=False),
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and json.loads(output_path.read_text()) != fixture:
        raise FileExistsError(f'Preserve existing token fixture: {output_path}')
    write_json(output_path, fixture)
    read_fixture(snapshot, output_path)
    return output_path


def read_fixture(snapshot, tokens_path):
    config, data = configuration(snapshot)
    fixture = json.loads(Path(tokens_path).read_text())
    prompt, continuation = fixture['prompt_ids'], fixture['continuation_ids']
    if not isinstance(prompt, list) or not isinstance(continuation, list):
        raise ValueError('Token IDs must be lists.')
    if not prompt or any(type(i) != int or i < 0 or i >= config.vocab for i in prompt + continuation):
        raise ValueError('A nonempty prompt and valid integer token IDs are required.')
    if len(prompt) + len(continuation) > data.get('max_position_embeddings', 32768):
        raise ValueError('Fixture exceeds declared context capacity.')
    return config, fixture


def inspect_logits(output, snapshot, top_k=5):
    """Print top logits at each saved position; return the full CPU tensor."""
    from transformers import AutoTokenizer

    logits = load_file(str(Path(output) / 'logits.safetensors'))['logits']
    if not torch.isfinite(logits).all():
        raise ValueError('Saved logits contain nonfinite values.')
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    print('Saved logits [positions, vocabulary]:', tuple(logits.shape))
    for position, row in enumerate(logits):
        values, indices = row.topk(top_k)
        print(f'Position {position}:', [
            dict(token_id=index, text=tokenizer.decode([index]), logit=value)
            for index, value in zip(indices.tolist(), values.tolist())])
    return logits


class VerificationRun:
    """One immutable output directory for a notebook checkpoint comparison.

    model_factory must accept a Config and return the notebook candidate, with
    forward(ids, cache, decode=True). The factory is retained, never a loaded model.
    name_map_factory is supplied explicitly by the notebook and maps checkpoint
    tensor names to the candidate's named_parameters() keys.
    """
    def __init__(self, model_factory, name_map_factory, *, tiny_config=None, model_case='tiny',
                 revision=None, device='cuda:0', dtype='fp32', tokens_path=None,
                 snapshot=None, output_root=None, notebook_path=None):
        if model_case not in {'tiny', '8b', '32b'}:
            raise ValueError('Choose tiny, 8b, or 32b.')
        if dtype not in {'fp32', 'bf16'}:
            raise ValueError('Choose fp32 or bf16.')
        if model_case != 'tiny' and (not isinstance(revision, str)
                                    or not re.fullmatch(r'[0-9a-fA-F]{40}', revision)):
            raise ValueError('Record the real checkpoint commit before preparing the snapshot.')
        self.model_factory = model_factory
        self.name_map_factory = name_map_factory
        self.tiny_config = tiny_config
        self.model_case = model_case
        self.model_revision = MODEL_REVISION if model_case == 'tiny' else revision
        self.device, self.dtype_name = device, dtype
        run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid.uuid4().hex[:8]
        self.run_root = Path(output_root or COURSE_ROOT / 'results/p1-notebook') / run_id
        self.run_root.mkdir(parents=True, exist_ok=False)
        self.snapshot = Path(snapshot) if snapshot is not None else (
            PRACTICE_SNAPSHOT if model_case == 'tiny' else COURSE_ROOT / f'models/qwen3-{model_case}')
        self.tokens_path = Path(tokens_path) if tokens_path is not None else (
            self.snapshot / 'tokens.json' if model_case == 'tiny' else COURSE_ROOT / 'results/qwen_tokens.json')
        self.notebook_path = Path(notebook_path or CODE_DIR / 'lab.ipynb')

    def prepare_checkpoint(self, *, download=False):
        """Prepare two-layer tiny or use/download a full pinned snapshot and saved IDs."""
        if self.model_case == 'tiny':
            if not (self.snapshot / 'config.json').is_file():
                make_fixture(self.snapshot, self.tiny_config, download=download)
        elif download:
            self.snapshot = download_model_checkpoint(self.model_case, self.model_revision, self.snapshot)
        if not (self.snapshot / 'config.json').is_file() or not self.tokens_path.is_file():
            raise FileNotFoundError('Prepare the local checkpoint and tokens.json; see the checkpoint preparation steps in lab.ipynb.')
        read_fixture(self.snapshot, self.tokens_path)
        print('Snapshot:', self.snapshot)
        print('Artifacts:', self.run_root)
        return self.snapshot

    def _load_custom(self, snapshot, device='cpu', dtype=torch.float32, audit_only=False):
        return load_custom(snapshot, device, dtype, audit_only,
                           model_factory=self.model_factory, name_map_factory=self.name_map_factory)

    def audit(self):
        """Check all candidate weights on meta before allocating their storage."""
        config, fixture = read_fixture(self.snapshot, self.tokens_path)
        _, inventory = self._load_custom(self.snapshot, audit_only=True)
        b = 4 if self.dtype_name == 'fp32' else 2
        prediction = dict(parameters=config.parameters(), parameter_bytes=config.parameters() * b,
                          final_kv_bytes=config.kv_bytes(
                              len(fixture['prompt_ids']) + len(fixture['continuation_ids']), b))
        write_json(self.run_root / 'inventory.json', inventory)
        write_json(self.run_root / 'memory_prediction.json', prediction)
        print(f'Audited {len(inventory)} tensors without allocating full model storage.')
        print(json.dumps(prediction, indent=2))
        return prediction

    def generate_logits(self, backend):
        """Save forced-history logits, releasing model/cache before returning.

        CUDA intervals are cold diagnostics; explicit CPU execution is untimed.
        """
        out = self.run_root / ('reference' if backend == 'transformers' else 'custom')
        if backend not in {'transformers', 'custom'}:
            raise ValueError('Unknown backend.')
        out = Path(out)
        if out.exists() and any(out.iterdir()):
            raise FileExistsError(f'Preserve existing results: {out}')
        c, tokens = read_fixture(self.snapshot, self.tokens_path)
        prompt_ids, forced_ids = tokens['prompt_ids'], tokens['continuation_ids']
        device = torch.device(self.device)
        dtype = torch.float32 if self.dtype_name == 'fp32' else torch.bfloat16
        if device.type not in {'cuda', 'cpu'}:
            raise ValueError('Use Spark CUDA, or explicitly select CPU for untimed correctness.')
        if device.type == 'cuda':
            if not torch.cuda.is_available():
                raise RuntimeError('CUDA is unavailable. Select a Spark-compatible kernel; no automatic CPU fallback.')
            torch.cuda.set_device(device)
            torch.cuda.reset_peak_memory_stats(device)
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        else:
            torch.set_num_threads(1)
        out.mkdir(parents=True, exist_ok=True)
        notebook_source = self.notebook_path.read_bytes()
        (out / 'notebook_source.ipynb').write_bytes(notebook_source)
        implementation_path = Path(getattr(self.model_factory, '__notebook_source_path__', self.notebook_path))
        implementation_source = implementation_path.read_bytes()
        expected_hash = getattr(self.model_factory, '__notebook_source_sha256__', None)
        if expected_hash and hashlib.sha256(implementation_source).hexdigest() != expected_hash:
            raise ValueError('Implementation notebook changed; reload its definitions before running.')
        (out / 'implementation_notebook.ipynb').write_bytes(implementation_source)
        manifest = dict(
            backend=backend, snapshot=str(self.snapshot.resolve()), model_revision=self.model_revision,
            config_sha256=hashlib.sha256((self.snapshot / 'config.json').read_bytes()).hexdigest(),
            fixture_sha256=hashlib.sha256(self.tokens_path.read_bytes()).hexdigest(),
            fixture_kind=tokens.get('kind', 'unspecified'), dtype=self.dtype_name, device=str(device),
            torch=torch.__version__, transformers=transformers.__version__,
            safetensors=version('safetensors'), accelerate=version('accelerate'),
            prompt_tokens=len(prompt_ids), continuation_tokens=len(forced_ids),
            sampling='teacher-forced supplied continuation', attention='eager dense reference',
            timing=('cold, synchronized diagnostic; not a steady-state benchmark'
                    if device.type == 'cuda' else 'untimed CPU correctness only'),
            source_sha256={
                self.notebook_path.name: hashlib.sha256(notebook_source).hexdigest(),
                implementation_path.name: hashlib.sha256(implementation_source).hexdigest(),
                **{name: hashlib.sha256((CODE_DIR / name).read_bytes()).hexdigest()
                   for name in ['notebook_utils.py', 'checkpoint.py', 'make_fixture.py']},
            },
        )
        if device.type == 'cuda':
            manifest['gpu_name'] = torch.cuda.get_device_name(device)
            manifest['cuda_runtime'] = torch.version.cuda
        write_json(out / 'manifest.json', manifest)

        def sync():
            if device.type == 'cuda':
                torch.cuda.synchronize(device)

        model = cache = output = logits = ids_tensor = None
        try:
            sync()
            start = time.perf_counter() if device.type == 'cuda' else None
            _, rows_inventory = self._load_custom(self.snapshot, audit_only=True)
            write_json(out / 'inventory.json', rows_inventory)
            write_json(out / 'memory_prediction.json', dict(
                parameters=c.parameters(), parameter_bytes=c.parameters() * dtype.itemsize,
                final_kv_bytes=c.kv_bytes(len(prompt_ids) + len(forced_ids), dtype.itemsize),
            ))
            if backend == 'custom':
                model, _ = self._load_custom(self.snapshot, device, dtype)
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    self.snapshot, local_files_only=True, dtype=dtype,
                    device_map={'': str(device)}, attn_implementation='eager',
                ).eval()
            sync()
            loading_ms = 1000 * (time.perf_counter() - start) if start is not None else None
            rows, saved = [], []
            with torch.inference_mode():
                for step, ids in enumerate([prompt_ids] + [[token] for token in forced_ids]):
                    ids_tensor = torch.tensor([ids], device=device)
                    sync()
                    start = time.perf_counter() if device.type == 'cuda' else None
                    if backend == 'custom':
                        logits, cache = model(ids_tensor, cache, decode=True)
                    else:
                        output = model(input_ids=ids_tensor, past_key_values=cache,
                                       use_cache=True, logits_to_keep=1)
                        logits, cache = output.logits, output.past_key_values
                    sync()
                    elapsed = 1000 * (time.perf_counter() - start) if start is not None else None
                    saved.append(logits[0, -1].float().cpu())
                    rows.append(dict(step=step, phase='prefill' if step == 0 else 'forced_decode',
                                     processed_tokens=len(prompt_ids) + step, diagnostic_ms=elapsed))
            save_file({'logits': torch.stack(saved)}, str(out / 'logits.safetensors'))
            write_csv(out / 'diagnostic_timing.csv', rows)
            summary = dict(
                saved_positions=len(saved), vocabulary=c.vocab, loading_ms=loading_ms,
                parameters=sum(p.numel() for p in model.parameters()),
                scope='Selected-position numerical verification; no steady-state performance claim',
            )
            if device.type == 'cuda':
                summary.update(allocated_bytes=torch.cuda.memory_allocated(device),
                               reserved_bytes=torch.cuda.memory_reserved(device),
                               peak_allocated_bytes=torch.cuda.max_memory_allocated(device))
            write_json(out / 'summary.json', summary)
            print(f'Saved {len(saved)} complete vocabulary vectors to {out}')
            return out
        finally:
            # Drop all GPU tensor owners, including the HF output object and its cache.
            model = cache = output = logits = ids_tensor = None
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()

    def compare(self, *, rtol, atol):
        """Compare every saved logit with matching identities and save the report."""
        output = self.run_root / 'comparison.json'
        if output.exists():
            raise FileExistsError('Preserve the accepted comparison; start a fresh run.')
        result = compare_checkpoint_logits(self.run_root / 'reference', self.run_root / 'custom', rtol, atol)
        write_json(output, result)
        print(json.dumps({k: v for k, v in result.items() if k != 'per_position'}, indent=2))
        return result


def compare_checkpoint_logits(reference, candidate, rtol, atol, *, display=True):
    """Accept matching-identity full logits only within declared tolerances."""
    if not all(math.isfinite(v) and v >= 0 for v in (rtol, atol)):
        raise ValueError('Finite nonnegative tolerances are required.')
    ref_meta = json.loads((Path(reference) / 'manifest.json').read_text())
    new_meta = json.loads((Path(candidate) / 'manifest.json').read_text())
    for key in ['model_revision', 'config_sha256', 'fixture_sha256', 'dtype']:
        if ref_meta[key] != new_meta[key]:
            raise ValueError(f'cannot compare different {key}')
    ref = load_file(str(Path(reference) / 'logits.safetensors'))['logits']
    new = load_file(str(Path(candidate) / 'logits.safetensors'))['logits']
    if ref.ndim != 2 or ref.numel() == 0 or new.shape != ref.shape:
        raise ValueError('Expected matching nonempty [positions, vocabulary] logits.')
    if not torch.isfinite(ref).all() or not torch.isfinite(new).all():
        raise ValueError('Logits must be finite.')
    torch.testing.assert_close(new, ref, rtol=rtol, atol=atol)
    summary = dict(correctness='passed', positions=ref.shape[0], vocabulary=ref.shape[1],
                   max_abs_error=float((new-ref).abs().max()),
                   relative_l2=float((new-ref).norm()/ref.norm().clamp_min(1e-20)))
    per_position = [dict(
        position=i, reference_top_id=int(ref[i].argmax()), custom_top_id=int(new[i].argmax()),
        reference_top_logit=float(ref[i, ref[i].argmax()]),
        custom_logit_at_reference_top_id=float(new[i, ref[i].argmax()]),
        max_abs_error=float((new[i] - ref[i]).abs().max()),
    ) for i in range(ref.shape[0])]
    if display:
        print('position  ref top  custom top   ref logit    custom logit   max |error|')
        for row in per_position:
            print(f"{row['position']:8d} {row['reference_top_id']:8d} {row['custom_top_id']:11d} "
                  f"{row['reference_top_logit']:11.8f} {row['custom_logit_at_reference_top_id']:13.8f} "
                  f"{row['max_abs_error']:12.3e}")
    return dict(summary, rtol=rtol, atol=atol, per_position=per_position,
                comparator='notebook_utils.compare_checkpoint_logits')


@torch.inference_mode()
def check_cache(model_factory, name_map_factory, *, device='cuda:0',
                partitions=((1,)*11, (3, 1, 7), (5, 6))):
    """Check the notebook model's full, incremental, and chunked cached logits."""
    if not partitions or any(not chunks or sum(chunks) != 11 or
                             any(type(size) is not int or size <= 0 for size in chunks)
                             for chunks in partitions):
        raise ValueError('Each partition must contain positive chunks totaling 11 tokens.')
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    c, _ = configuration(PRACTICE_SNAPSHOT)
    model = cache = ids = full = logits = pieces = None
    try:
        model, _ = load_custom(PRACTICE_SNAPSHOT, device=device, dtype=torch.float32,
                               model_factory=model_factory, name_map_factory=name_map_factory)
        assert sum(p.numel() for p in model.parameters()) == c.parameters()
        ids = torch.randint(0, c.vocab, (2, 11), device=device)
        full, _ = model(ids)
        for chunks in partitions:
            cache, pieces, start = None, [], 0
            for size in chunks:
                logits, cache = model(ids[:, start:start+size], cache)
                pieces.append(logits)
                start += size
            check_cache_logits(torch.cat(pieces, 1), full)
        print('PASS: notebook parameter count, incremental decode, unequal chunk equivalence')
    finally:
        model = cache = ids = full = logits = pieces = _ = None
        gc.collect()
        if torch.device(device).type == 'cuda':
            torch.cuda.empty_cache()


def run_cache_experiment(out, *, model_factory, name_map_factory, repeats=3, device='cuda:0', seed=42,
                         lengths=(128, 256, 512, 1024, 2048)):
    """Measure tiny CUDA decode and save predictions, raw data, and figures."""
    lengths = tuple(lengths)
    if not lengths or any(type(length) is not int or length <= 0 for length in lengths):
        raise ValueError('Context lengths must be positive integers.')
    if len(set(lengths)) != len(lengths):
        raise ValueError('Context lengths must be distinct.')
    args = begin('01', 'Qwen3 tiny CUDA decode and logical cache storage', [
        'First two real Qwen3-8B layers in FP32; truncated-model timing',
        'Logical tensor bytes exclude allocator/workspace',
        'Each timed region produces exactly one next-position logit',
    ], argv=['--out', str(out), '--repeats', str(repeats),
             '--device', device, '--seed', str(seed)])
    manifest = json.loads((args.out / 'manifest.json').read_text())
    source_path = Path(getattr(model_factory, '__notebook_source_path__', CODE_DIR / 'lab.ipynb'))
    source = source_path.read_bytes()
    (args.out / 'implementation_notebook.ipynb').write_bytes(source)
    manifest['source_sha256'][str(source_path.relative_to(COURSE_ROOT))] = hashlib.sha256(source).hexdigest()
    write_json(args.out / 'manifest.json', manifest)
    c, _ = configuration(PRACTICE_SNAPSHOT)
    write_json(args.out / 'prediction.json', dict(
        model='Qwen3 tiny (course)', config=asdict(c), model_revision=MODEL_REVISION,
        cache_bytes_per_token=c.kv_bytes(1, 4), lengths=lengths,
        hypothesis='Cache bytes are linear in processed length; recomputation grows faster than cached decode.'))
    model = cache = ids = a = b = cached = full = None
    try:
        model, _ = load_custom(PRACTICE_SNAPSHOT, device=args.device, dtype=torch.float32,
                               model_factory=model_factory, name_map_factory=name_map_factory)
        memory, rows, largest_error, errors = [], [], 0., []
        with torch.inference_mode():
            for length in lengths:
                ids = torch.randint(c.vocab, (1, length+1), device=args.device)
                _, cache = model(ids[:, :length])
                actual = sum(t.numel()*t.element_size() for pair in cache for t in pair)
                expected = c.kv_bytes(length, 4)
                assert actual == expected
                memory.extend(dict(length=length, kind=kind, cache_bytes=value) for kind, value in
                              [('predicted logical bytes', expected), ('counted tensor bytes', actual)])
                cached = lambda: model(ids[:, length:], cache, decode=True)
                full = lambda: model(ids, decode=True)
                a, b = cached()[0], full()[0][:, -1:]
                largest_error = max(largest_error, float((a-b).abs().max()))
                errors.append(dict(length=length, **check_cache_logits(a, b)))
                for name, fn in [('cached', cached), ('recomputed', full)]:
                    for repeat, ms in enumerate(time_cuda(fn, args.repeats)):
                        rows.append(dict(length=length, method=name, repeat=repeat, measured_ms=ms))
        write_csv(args.out / 'memory.csv', memory)
        plot(args.out / 'memory.svg', memory, 'length', 'cache_bytes', 'kind',
             'P1 logical KV accounting (CUDA FP32)')
        plot(args.out / 'latency.svg', rows, 'length', 'measured_ms', 'method',
             'P1 Qwen3 tiny CUDA decode: median and range')
        finish(args, rows, dict(correctness='passed', max_abs_logit_error=largest_error,
                               cache_error_metrics=errors, exact_cache_byte_matches=len(lengths),
                               scope='Chapter 1 Qwen3 tiny only'))
        return args.out
    finally:
        model = cache = ids = a = b = cached = full = None
        # The prefill output and last timing closure also retain tensors/locals.
        _ = fn = None
        gc.collect()
        torch.cuda.empty_cache()


def check_cache_logits(candidate, reference):
    """Normwise FP32 cache equivalence for real-weight, differently shaped GEMMs.

    Near-zero logits need a scale-aware absolute error. Require both relative L2
    <= 2e-6 and max error / max reference magnitude <= 5e-6 per vocabulary row.
    The independent checkpoint comparison retains its elementwise criterion.
    See shared/VALIDATION.md for FP64 calibration of this separate cache gate.
    """
    if candidate.shape != reference.shape or not torch.isfinite(candidate).all() or not torch.isfinite(reference).all():
        raise AssertionError('Cache logits must have matching shapes and finite values')
    error = (candidate.double() - reference.double()).flatten(0, -2)
    target = reference.double().flatten(0, -2)
    relative_l2 = float((error.norm(dim=-1) / target.norm(dim=-1).clamp_min(1e-20)).max())
    max_scaled = float((error.abs().amax(dim=-1) / target.abs().amax(dim=-1).clamp_min(1e-20)).max())
    assert relative_l2 <= 2e-6 and max_scaled <= 5e-6, (relative_l2, max_scaled)
    return dict(relative_l2=relative_l2, max_scaled_error=max_scaled)
