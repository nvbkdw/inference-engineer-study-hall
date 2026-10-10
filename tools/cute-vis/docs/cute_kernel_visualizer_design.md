# CuteViz implementation design

CuteViz is a local inspection tool for important objects in a user's CuTeDSL kernel. Learners add explicit probes next to those objects and compile through the built-in Python editor or their existing driver. The resulting portable capture synchronizes embedded source, logical coordinates, storage, and operation-specific ownership in a browser.

## Observation model

`inspect`, `inspect_copy`, and `inspect_mma` are CuTeDSL `dsl_user_op` extensions. They return no value and replace no kernel operations. Observations occur during the compiler's meta stage, including symbolic control flow; they are not execution history. This distinction follows [NVIDIA's compilation model](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/dsl_code_generation.html).

A context-local capture serializes objects immediately. Temporary compiler algebra runs in a detached MLIR module, so extraction does not insert operations into the user's kernel. No live compiler objects are retained. Repeated probes receive distinct, deterministic IDs within a source snapshot. A stack-derived call span points into embedded source. Optional textual parent relationships remain learner-declared. Supported local-tile/slice parents are recovered separately from the observed tensor's compiler IR, with explicit derived provenance.

Empty observations produce an actionable exception and an empty diagnostic capture. Failures preserve preceding observations. The pinned compiler's `cute.compile` disables its compile cache; calling a compiled callable skips probes. The documented refresh command additionally disables caches in a fresh process.

## Data and evaluation

The versioned capture contains compiler/target metadata, caller-supplied parameters, source snapshots, probe IDs/spans, layout expression trees, tensor metadata, copy/MMA operands, and relationships. Per-field observations distinguish known, symbolic, unsupported, and unavailable data. The schema contains no executable expressions.

Nested modes survive serialization. The Python evaluator adapts [jduprat/tensor-layouts](https://github.com/jduprat/tensor-layouts) to CuteViz's `left(offset + right(coordinate))` composition convention. Mapping queries flatten mode paths only for axis selection; the full hierarchy is retained. Aliased addresses and multiple owners are not collapsed into a bijection.

CuTe's actual tiled source/destination and A/B/C TV descriptors supply ownership. A tensor snapshot alone implies no copy/MMA ownership. Copy correspondence follows logical elements in the compiler reference tile; source/destination owners may differ, as with `ldmatrix`. Register fragments show thread/value indices, never physical register allocation.

For supported `cute.local_tile` and `cute.slice` operations, the adapter recovers the actual input tensor and applies the compiler operation to an identity-coordinate tensor in the detached module. It snapshots an affine logical-coordinate transform with explicit runtime parameters. Direct CTA indices and constant affine arithmetic retain their meaning; other dynamic indices become named inspection inputs. Flat static parents and rectangular unit-stride tilers are supported. Unsupported transforms preserve their backing tensor and reason. The CPU composes these transforms for selected coordinates and bounded footprints without materializing the original array. Capture schema `1.1` adds these descriptors; `1.0` remains readable.

## Architecture adapters

Basic layout/tensor inspection is independent of architecture. Dense FP16/BF16 operation adapters are checked against compiler version 4.8.0:

- Ampere `mma.sync` uses warp-level logical thread/value ownership.
- Hopper WGMMA distinguishes shared/register A, shared B, and register C with 128-thread warpgroup participation. See [NVIDIA's warpgroup API](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/cute_nvgpu_warpgroup.html).
- Blackwell `tcgen05` is restricted to a single CTA, with shared/tensor A, shared B, and tensor C. Tensor-memory data paths and 32-bit columns are derived from compiler fragment layouts; element widths and subword positions are explicit. See [NVIDIA's tcgen05 API](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/cute_nvgpu_tcgen05.html).

Descriptor fragments do not establish physical backing addresses. Position-dependent pointer swizzles additionally require a base phase; the capture retains the swizzle while marking physical placement unresolved. These semantics follow NVIDIA's [swizzled pointer definition](https://github.com/NVIDIA/cutlass/blob/main/include/cute/pointer_swizzle.hpp).

Unsupported operations retain useful operand snapshots and an explanatory diagnostic. Support for the `sm_100a` tcgen05 target is not a claim about every device marketed as Blackwell.

## Local service and browser

FastAPI serves capture, mapping, bank analysis, and compilation-job APIs, plus bundled Svelte/TypeScript/Vite assets. The CLI binds to loopback. Opening a capture neither imports CuTeDSL nor needs a GPU, and never evaluates captured strings.

The CodeMirror Python editor submits source, entry name, argument-factory name, and optional target. A fresh Python process imports the source from a real file, constructs arguments, and calls `cute.compile` inside a capture context. Compilation has bounded logs, a timeout, cancellation, explicit partial outcomes, and persistent per-run source/capture files. Immutable capture IDs keep mapping requests tied to the selected run. Failed compiles retain the preceding visualization; drafts persist in local storage. The server's Python environment must contain the pinned compiler and the user's dependencies.

Browser compilation supersedes the original deferred feature. It executes trusted local Python, including imports and argument factories, with the local user's privileges. Processes isolate compiler state, not privileges. Host/Origin checks and a session token protect the loopback execution API; `--read-only` disables it. Captures remain inert portable data.

One browser selection links source, object, coordinate, thread, warp/warpgroup, value index, and operand. SVG grids provide hierarchical axis/slice controls, keyboard navigation, labels, persistent legends, and swizzle comparison. Copy views compare both operands. MMA views show A/B/C together, linking their shared M/N/K axes. An explanation panel gives concrete coordinate arithmetic, and a thread/value table preserves multiple owners.

The left sidebar selects the canvas group: the chosen probe, its immediate incoming/outgoing relationships, and complete copy/MMA operand groups when an operand is chosen. Derived backing chains are followed to their original tensor without expanding unrelated descendants. Canvas cell selection does not change group membership. These views occupy a Svelte Flow canvas with a movable thread/value panel, pan/zoom, minimap, optional relationship edges, fit/focus actions, and expanded mode. The **Links** toggle defaults off; cell highlights work with either setting. Blue footprints show a tile's region in its backing tensors, while orange selections propagate logical coordinates in both directions. Inspection inputs choose assumed CTA/tile indices without recompiling. The coordinate explanation lists each ancestor and provides navigation to its cell; out-of-bounds results are explicit.

Tensor nodes size themselves for their full 2D slice, using the library's bottom-right `NodeResizeControl` to scale the complete grid when resized. There are no internal tensor scrollbars or manual pages. Canvas arrangement accounts for varying card sizes and preserves resized dimensions. SVG drawing coordinates are normalized for extremely large shapes. Cell details render in a bounded SVG near the camera; the canvas rebases its origin at large translations. Coordinate-based pointer picking supplements native SVG hit testing at extreme zoom. These controls avoid browser coordinate clamping while keeping the full logical extent. The visible intersection determines mapping requests, split into batches of at most 4,096 cells. Up to 16,384 visible cell details render per card; denser views show a labeled full-slice overview until zoomed in. Double-click and **Go to cell** center and magnify any coordinate. Off-screen nodes are virtualized, and per-view settings survive remounting.

Library review selected [Svelte Flow custom nodes](https://svelteflow.dev/learn/customization/custom-nodes) and its [viewport/virtualization API](https://svelteflow.dev/api-reference/svelte-flow) because they support the existing Svelte controls, SVG grids, and node connections. [Konva](https://konvajs.org/docs/overview.html) provides a Canvas 2D scene graph and [PixiJS](https://pixijs.com/8.x/guides/getting-started/intro) provides GPU rendering; those are useful for raster-heavy scenes but would require rebuilding this tool's form, keyboard, and accessibility interactions. Svelte Flow owns viewport and graph rendering; Python continues to own mapping math. The npm lock pins the selected dependency version.

Every grid query is limited to 4,096 cells. CPU TV inversion is bounded and cached. Unsupported or symbolic fields explain why a view is partial.

## Bank analysis

Placement uses explicit byte offsets and a declared bank width/count. Scalar conflict analysis requires one explicitly supplied 32-bit access per active lane of one warp. Same-word reads broadcast, distinct words sharing a bank serialize, and same-word writes are reported separately as a race. Vector/descriptor/tensor-core transactions are outside the scalar model. This follows the distinctions in [CUDA's shared-memory guidance](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#shared-memory-and-memory-banks).

## Validation and scope

The implementation includes portable captures, compiler partition oracles, independent layout/swizzle/TMEM expectations, IR/PTX neutrality checks, an optional real GPU copy check, API tests, and browser interaction tests. Current evidence and gaps are recorded in [validation.md](validation.md). Human usability is a separate acceptance exercise described in [usability.md](usability.md).

Automatic instrumentation, runtime values, physical registers, pipeline timing, TMA simulation, sparse/block-scaled/multi-CTA MMA, and performance prediction remain deferred.
