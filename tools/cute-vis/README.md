# CuteViz

Write, compile, and inspect your own CuTeDSL kernels in a local Python workbench. CuteViz records explicit **compile-time observations** and links layouts, tensor views, copies, and dense MMA operands to their source. Opening an existing capture needs neither CUDA nor CuTeDSL.

## Open the Python workbench

```bash
uv sync --locked --extra capture
uv run --extra capture cuteviz serve --open
```

At <http://127.0.0.1:8765>, edit Python in the **Code** tab and click **Compile & capture** (Ctrl/⌘+Enter). The bundled copy example is ready to compile. Choose the MMA example and set `sm_80`, `sm_90a`, or `sm_100a` to inspect the corresponding operand model. Successful runs open the **Inspect** tab automatically; each run offers a `.cuteviz.json` download.

The editor includes Python syntax highlighting, indentation, undo, search, `.py` import/export, and a draft saved in browser local storage. Set **Entry function** to your `@cute.jit` function (including dotted names such as `operation.entry`). Put probes in the compiled code as shown below. The optional **Arguments factory** names a normal Python function that creates the compilation arguments:

```python
import cutlass
import cutlass.cute as cute
import cuteviz

@cute.jit
def entry(rows: cutlass.Constexpr):
    tile = cute.make_layout((rows, 8), stride=(8, 1))
    cuteviz.inspect("tile", tile)

def make_args():
    return (4,)
```

An entry may launch any of your `@cute.kernel` functions during compilation. For tensor arguments, `make_args()` can create `cute.runtime.make_fake_tensor(...)` objects or runtime pointer placeholders, as in the bundled copy example. Return `{"args": (a, b), "kwargs": {"options": "..."}}` to pass keyword arguments or compiler options. An absent default `make_args` means zero arguments; a custom factory name must exist. Class instances and imported entries are supported through named objects; entry and factory fields do not accept Python expressions.

The server uses its own Python environment. Install CuteViz with the capture extra in the environment containing your kernel's dependencies, then start `cuteviz serve` there. CuTeDSL **4.8.0** is required for editor compilation; a missing or different compiler disables compilation with a diagnostic while keeping playback available.

```bash
# Use an existing project for local imports and relative paths.
cuteviz serve --workdir /path/to/project --timeout 180 --open
# Choose where source and capture files are saved.
cuteviz serve --output-dir /path/to/runs
# Open portable captures with Python execution disabled.
cuteviz serve kernel.cuteviz.json --read-only --open
```

Each run starts a fresh process with `CUTE_DSL_NO_CACHE=1`, imports the edited module, calls its argument factory, and runs `cute.compile(entry, *args, **kwargs)` inside a capture context. `if __name__ == "__main__"` driver blocks do not run. The target field sets `CUTE_DSL_ARCH` before importing CuTe; leaving it empty uses the environment/compiler default. Target compilation is subject to CuTe's capabilities and the installed toolchain; it does not expand the set of mappings CuteViz can inspect.

Runs save `kernel.py`, `config.json`, and `kernel.cuteviz.json` under `WORKDIR/.cuteviz/<run-id>/` by default. The browser displays the last 65,536 characters of compiler output. Errors preserve the last displayed capture, and a partial capture can be inspected explicitly. Draft changes are marked when they differ from the selected run. Cancellation or timeout terminates the process group; a capture is only available if it was already saved. The default timeout is 120 seconds, with one active compile and up to 100 runs per server session. Session history is in memory; files remain on disk after restart and can be opened with `cuteviz serve <capture>`.

This is a local trusted-code workbench: Python imports, top-level code, and argument factories execute with your user permissions and can allocate resources or perform I/O. The app does not call the compiled kernel, but your Python code can. Fresh processes isolate compiler state, **not** filesystem or network access. The server binds to loopback and checks Host, Origin, and a session token for execution requests. Opening a capture never executes its embedded source. Keep the service local; it is not a multi-user execution sandbox.

## Try the viewer

From this directory, with [uv](https://docs.astral.sh/uv/) installed:

```bash
uv sync --locked
uv run cuteviz serve examples/captures/copy.cuteviz.json --open
```

The viewer runs at <http://127.0.0.1:8765>. Select `load_A` to compare copy source and destination, `nested` to explore hierarchical modes, or `swizzled_sA` to compare offsets. Select a grid cell to see its coordinate calculation and source location. Arrow keys move the cell selection; thread, warp, warpgroup, and value filters link the views.

In Inspect mode, select an object in the left sidebar to show it and its directly related objects on the **tensor canvas**, implemented with [Svelte Flow](https://svelteflow.dev/). Related objects come from explicit parent/child and observed-operand relationships. Selecting an operand includes its copy/MMA operation and sibling operands. Clicking a canvas cell updates the inspection without changing the sidebar’s chosen group. Drag the background to pan, use **+ / −** or Ctrl/⌘+scroll to zoom, and drag panel titles to arrange them. **Fit all** frames this related group; **Focus selected** frames the selected probe and its thread/value panel. A minimap, **Expand canvas**, and **Hide source** provide more room for exploring larger captures.

Tensor cards expand to fit the **full 2D slice**. Drag the **bottom-right corner** to shrink or enlarge a card; its complete grid scales with it, with no internal tensor scrollbar or manual paging. Resize controls reuse Svelte Flow's [NodeResizeControl](https://svelteflow.dev/api-reference/components/node-resize-control). **Arrange** preserves resized dimensions. Pan and zoom the shared canvas, double-click a grid to inspect that region, or use **Go to cell** to reach a distant row and column.

Visible cell mappings load automatically in the largest applicable batches, each capped at 4,096 cells. Up to 16,384 cell details are rendered per visible card. When more cells fit in the viewport, the card shows a labeled full-slice overview; zooming in restores exact coordinates, offsets, ownership, and selection. The full shape always occupies the card. Extremely large shapes use normalized drawing coordinates to stay within browser limits. Axis, slice, and color settings survive panning off-screen and back.

Clicking a cell highlights related copy/MMA coordinates across the canvas. Explicit copy operand snapshots link in both directions when their coordinate domains match. Thread/value selections highlight captured operation ownership and its explicit copy snapshots. **Links** is off by default, leaving cell highlights visible. Turn it on to show red selection links and dashed observed-operand relationships. Matching shapes alone, declared textual transforms, and MMA fragment snapshots do not imply a coordinate mapping. Ownership remains unknown where it was not captured, and value indices remain local to each operand.

Prebuilt captures for `sm_80`, `sm_90a`, and `sm_100a` are also in `examples/captures/`. Each contains the source used to generate it. These are compile-time observations. `gemm.cuteviz.json` captures a real one-warp GEMM; the Hopper/Blackwell examples demonstrate fragment construction without executing MMA.

## Probe your kernel

Install CuteViz in the environment that compiles your kernel. The capture adapter is pinned to **nvidia-cutlass-dsl 4.8.0**:

```bash
uv sync --locked --extra capture
# Or, in an existing compatible Python environment:
python -m pip install -e '.[capture]'
```

Add probes inside your existing `@cute.kernel` or `@cute.jit` function:

```python
import cuteviz

# These observe existing objects and return None.
cuteviz.inspect("gA", gA)
cuteviz.inspect_copy("load_A", tiled_copy, src=gA, dst=sA)
cuteviz.inspect_mma("mma", tiled_mma, a=tCrA, b=tCrB, c=acc)
```

In the driver, wrap compilation:

```python
with cuteviz.capture("kernel.cuteviz.json", parameters={"tile_mnk": [128, 64, 64]}):
    compiled = cute.compile(entrypoint, *args)
```

Then run:

```bash
cuteviz serve kernel.cuteviz.json --open
```

Optional `target=` and `parameters=` record caller-supplied compilation metadata; otherwise the adapter reads the compiler target. These options do not alter compilation. A tensor probe does not infer thread ownership. `inspect_copy` and `inspect_mma` provide that context explicitly. Source/destination and A/B/C arguments are optional; supplied tensors are also captured as separate, linked snapshots.

To declare ancestry, use an earlier unique probe label or ID:

```python
cuteviz.inspect("tile", gA_tile, parent="gA", transform="local_tile((64,32), (1,0))")
```

The transform is explanatory text, never executable code. Repeated labels receive distinct IDs; ambiguous parent labels produce a diagnostic. Probes outside a capture are no-ops. Nested captures are isolated using context-local sessions.

## Follow a tile back to global memory

Probing a supported `cute.local_tile` or tensor slice automatically captures its backing tensor and a compiler-derived coordinate transform. You do not need to probe the original tensor separately:

```python
block_m, block_n, _ = cute.arch.block_idx()
tile = cute.local_tile(global_tensor, (8, 16), (block_m, block_n))
cuteviz.inspect("cta_tile", tile)
```

Select `cta_tile` in the sidebar to show the tile and its backing chain on the canvas. Blue highlights mark the tile's footprint in the backing tensor; orange highlights follow a selected cell in either direction. **Inspection coordinates** lets you choose assumed `blockIdx.x/y/z` or unresolved tile indices. These are inputs to the visualization, not recorded GPU execution. For CTA `(1, 2)`, local cell `(1, 2)` in this example maps to global cell `(9, 34)`. **FOLLOW THE COORDINATE** displays each step, its calculation, and offsets relative to that backing view. **Show in canvas** centers and magnifies the corresponding cell, including in a large tensor. **Links** remains optional.

Choose the **tiles** starter in the Code tab, or open the portable demonstration:

```bash
cuteviz serve examples/captures/tiles.cuteviz.json --open
```

The pinned adapter supports flat static backing shapes, rectangular unit-stride tilers, projections, remainder modes, slices, nested tile chains, and CTA indices with constant affine arithmetic. It recovers the actual input tensor from compiler IR and applies the same operation to an identity-coordinate tensor in a detached compiler context. It does not reconstruct arbitrary kernel dataflow. Unsupported transforms retain the backing snapshot with a reason; dynamic backing extents, gathered tilers, and opaque partition/descriptor ancestry remain unresolved. Out-of-bounds mappings are labeled and their footprint is clipped to the backing shape. Logical ancestry does not invent thread ownership or runtime pointer addresses.

New captures use schema `1.1`; schema `1.0` captures still open. Recompile older captures to obtain backing transforms—textual `parent`/`transform` declarations alone cannot recover them.

## Refresh a capture

Run the driver in a fresh process:

```bash
CUTE_DSL_NO_CACHE=1 python your_driver.py
```

With the pinned compiler, `cute.compile(...)` already sets `compile_only=True` and `no_cache=True`. Calling an existing compiled callable does not rerun probes. Some JIT cache paths still trace before reusing compiled artifacts; the tool does not assume every cache hit skips probes.

An empty capture is written with a diagnostic and raises `EmptyCaptureError`; it is never reported as a successful inspection. A compilation failure writes the observations obtained so far and re-raises the original error. Unsupported probes preserve surrounding observations.

## What is supported

| Object | Inspection |
| --- | --- |
| Layout | Flat/nested shape and stride trees, zero/negative strides and aliases, scalar-output composition, layout swizzles |
| Tensor | Shape, dtype, memory space, relative offsets, declared ancestry, supported compiler-derived tile/slice backing coordinates |
| TiledCopy | Source and destination thread/value partitions within one tile; logical element correspondence, including `ldmatrix` redistribution |
| Ampere dense MMA | FP16/BF16 `(16,8,8)` and `(16,8,16)`; register operand and accumulator thread/value indices |
| Hopper dense WGMMA | FP16/BF16 `(64,N,16)`, N=8…256 by 8; shared or register A, shared B, register C; warpgroup participation |
| Blackwell dense tcgen05 | FP16/BF16 `(M,N,16)`, M=64/128, N=8…256 by 8; single CTA; shared/tensor A, shared B, tensor C |

FP16 supports FP16 or FP32 accumulation; BF16 supports FP32 accumulation. Compiler-derived layouts parameterize tiling rather than hard-coding a particular atom. Target validation and restrictions are documented in [the validation matrix](docs/validation.md). `tcgen05` support does not mean every Blackwell device supports these instructions; the GB10/SM121 target is not the validated `sm_100a` target family.

Thread/value indices are logical fragments, not physical register allocation. Tensor-memory placement uses **data path, 32-bit column, and bit offset**, not byte addresses or thread ownership. MMA shared operands remain collective storage. A supplied partition/descriptor does not establish the backing tensor's matrix-to-byte mapping; inspect that backing tensor separately.

Position-dependent pointer swizzles are recorded separately. Their logical offsets remain queryable, but physical byte placement is unresolved without the runtime base phase. Ordinary composed-layout swizzles support concrete offset and bank-placement views. Symbolic extents cannot produce a finite grid; probe a static slice. Tuple-output/basis-stride compositions and unknown operations remain explicit partial fields with reasons.

The viewer fetches visible cell details in requests capped at 4,096 cells while preserving the full grid extent on the canvas. Ownership inversion is bounded to 262,144 thread/value entries. See [the capture and API contract](docs/capture-format.md) for schema limits.

## Shared-memory banks

The grid offers byte-based **placement** under a declared 32-bank, four-byte model and assumed base displacement. It does not infer conflict counts from colors.

The access lab accepts an explicit scalar 32-bit warp access group. Same-word reads broadcast, inactive lanes are excluded, and distinct words in the same bank serialize. Same-word writes are reported as an unsupported data race. Vector, descriptor, and tensor-core accesses receive no scalar conflict predictions. These are model results, not measured performance.

## Examples and development

```bash
# Real global → shared → global copy, compiled for the selected target.
CUTE_DSL_ARCH=sm_80 uv run --extra capture python examples/inspect_copy.py
CUTE_DSL_ARCH=sm_80 uv run --extra capture python examples/inspect_gemm.py
# Real compiler fragment construction; these examples do not launch MMA.
CUTE_DSL_ARCH=sm_90a uv run --extra capture python examples/inspect_mma.py
CUTE_DSL_ARCH=sm_100a uv run --extra capture python examples/inspect_mma.py

# Python checks, including cross-target compiler oracles when installed.
uv run --extra capture pytest -q
# Optional GPU copy neutrality and GEMM correctness checks on the local GPU.
CUTEVIZ_GPU_TEST=1 uv run --extra capture pytest -q

# Frontend development, build, and browser interaction checks.
cd frontend
npm ci
npm run check
npm run build
npx playwright install chromium
npm test
```

`npm run dev` proxies API requests to a running `cuteviz serve` instance on port 8765. `npm run build` writes the bundled frontend to `src/cuteviz/static`. Rebuild it after frontend edits. The built assets are included in source and wheel distributions so playback does not require Node.

```bash
uv build
python -m pip install dist/cuteviz-0.1.0-py3-none-any.whl
cuteviz validate examples/captures/copy.cuteviz.json
```

Deferred: automatic instrumentation, runtime tensor values, physical registers, pipeline timing, TMA execution simulation, sparse/block-scaled or multi-CTA MMA, and performance predictions. Beginner usability sessions require participants and have not been conducted; a repeatable [exercise protocol](docs/usability.md) is provided.
