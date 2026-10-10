# Capture format and local API

Schema `1.1` is defined by Pydantic models in `src/cuteviz/model.py`; `1.0` remains readable. `docs/capture.schema.json` is generated from those models. Unknown keys, schema versions, invalid dimensions, duplicate IDs, dangling relationships, and invalid source spans are rejected. Imports are limited to 32 MiB, 5,000 objects, 100 source snapshots, and 1,000,000 characters per source. Static extents are positive signed 64-bit integers; nested modes are limited to depth 16.

## Fields

A `Capture` contains `schema_version`, `producer_version`, `observation: "compile-time"`, compiler/target observations, parameters, embedded sources, objects, diagnostics, and outcome (`complete`, `partial`, or `empty`). Outcome describes capture collection; a complete capture can still contain runtime-dependent fields.

An `Observation` contains `status`, `value`, and `reason`. Non-known statuses require a reason. Dynamic shape/stride leaves use observations directly, while static leaves are integers. Source IDs identify embedded content and its original path; the service never reads that path when playing back a capture.

Objects have unique IDs, labels, source spans, fields, views, and relationships. Repeated labels do not overwrite objects. Relationships distinguish `declared_parent`, captured `operand`, and compiler-derived `derived_parent`. Textual `transform` fields are descriptions, never code.

A `derived_parent` may contain a `mapping` and a `mapping_status` observation. The mapping has `operation` (`cute.local_tile` or `cute.slice`), `origin`, `matrix`, and named `parameters`, each with a label and coefficient vector. It expresses **parent logical coordinates = origin + matrix × child logical coordinates + parameter terms**. Coordinates address flattened leaf modes, not byte addresses. Dimensions must match the referenced views; coefficients are signed 64-bit integers, ranks and parameter counts are at most 32, and derived ancestry must be acyclic with at most 16 edges. Automatically recovered parents have a provenance field and the originating probe's source span. Unsupported transforms keep the relationship and backing snapshot, with an absent mapping and an explicit reason.

Layouts have one of three expression kinds:

- `affine`: nested `shape` and `stride` trees.
- `swizzle`: CuTe `bits`, `base`, and `shift`.
- `compose`: `left(offset + right(coordinate))`, with a scalar integer offset.

Views distinguish their logical shape from their storage layout, TV layout, observed operand object, and ownership status. For MMA, the logical tile and supplied fragment are separate views with different domains. Register operands expose logical ownership, shared operands expose collective storage, and tensor-memory operands expose data-path/column/bit placement. A tensor's `pointer_swizzle` is distinct from a layout swizzle; unresolved runtime phase prevents physical byte-placement claims.

Offsets use explicit units:

- `element_offset`: elements relative to the captured view's iterator (before a pointer swizzle if present).
- `byte_offset`: bytes relative to the view's base, only when dtype and address mapping are known.
- `fragment_index`: a thread-local logical register-fragment value index, with no physical address claim.
- `tmem`: `{data_path, column, bit_offset}` with 32-bit columns, relative to the allocation.

## Mapping query

```http
POST /api/mapping/query
Content-Type: application/json

{"object_id":"probe-…","role":"C","axes":[0,1],"fixed":{},
 "start":[0,0],"extent":[8,8],"selection":{"role":"C","coordinate":[1,2]}}
```

Axes index the flattened leaf-mode list, whose paths (`0.0`, `0.1`, `1`, etc.) are returned in `dimensions`. `fixed` supplies coordinates for non-visible axes; omitted ones default to zero. A one-axis grid has one column. `start` is the row/column origin; `extent` is clipped at the shape boundary and may request no more than 4,096 cells.

The response includes cells, full logical coordinates, offset observations, all owners, selected state, copy correspondence links, concrete explanations, and unswizzled offsets where available. Selections can use a coordinate/role or thread/value/warp/warpgroup filters. MMA coordinates link along shared M/N/K axes. Copy coordinates link by logical element while ownership can change.

`selection.object_id` optionally identifies the source probe when querying another tensor on the shared canvas. Coordinate bounds are checked in that source view. Within an operation, the existing copy/MMA rules apply. Across probes, explicit copy `observed_object` links with matching shapes propagate coordinates or thread/value highlights; known derived ancestry additionally links logical coordinates in both directions, preserving aliases. Plain tensors retain their original ownership metadata; selection does not manufacture ownership. Unrelated equal-shaped tensors, declared textual ancestry, and MMA matrix-to-fragment domains are not equated. Omitting `selection.object_id` preserves the original per-object behavior. The canvas requests visible windows in batches capped at 4,096 cells and renders at most 16,384 detailed cells per card. Denser views request only footprint metadata and show a full-shape overview until zoomed in.

Supply `bindings`, for example `{"blockIdx.x":1,"blockIdx.y":2}`, to resolve runtime tile indices for inspection. Values are integers within JavaScript's exact integer range; at most 64 bindings are accepted. Missing bindings leave ancestry unresolved with a diagnostic. The browser explicitly defaults inspection indices to zero; the API never invents missing values. `focus_object_id` identifies the sidebar-selected object whose footprint should be highlighted. Responses add `footprints` (projected start/extent and rectangular exactness), `mapping_diagnostics`, cell `in_focus`, and per-cell `coordinate_mappings` with the backing coordinate, calculation, offsets, and in-bounds flag. Unsupported footprint shapes do not receive a false rectangular overlay. No original tensor grid is materialized to compute a footprint. Offsets remain relative to each respective view; runtime allocation addresses are not recovered.

Ownership inversion is limited to 262,144 thread/value entries. Probe a smaller operation tile if this limit is reached. Symbolic extents cannot be enumerated and return an explanatory 422 response. Symbolic strides retain a grid with unresolved addresses. Unknown object IDs return 404. Mapping and bank queries do not execute strings or accept arbitrary expression evaluation.

Both `GET /api/capture` and `POST /api/mapping/query` accept `?capture_id=<run-id>` to select an immutable workbench result. Without this parameter they use the capture supplied to the CLI. No capture returns 404; compiling never silently replaces the default.

## Compilation jobs

`GET /api/workbench` returns availability, compiler version, Python executable, working/output directories, timeout, starter sources, and the session token. Source editing is always available; execution requires the pinned compiler and is disabled by `--read-only`.

```http
POST /api/compilations
Content-Type: application/json
X-CuteViz-Token: <session token>

{"source":"...Python module...", "entry":"entry", "args_factory":"make_args", "target":"sm_80"}
```

The response is 202 with a run ID and status. Source is limited to 256,000 characters. Entry/factory names are dotted Python identifiers; the factory may be empty. Target may be empty or an `sm_` target name. Unknown fields are rejected. Only one compilation can be active; a second submission returns 409. The editor does not upload dependencies; local imports resolve from the configured working directory.

| Endpoint | Result |
| --- | --- |
| `GET /api/compilations` | Current session runs, newest first |
| `GET /api/compilations/{id}` | Status, last 65,536 characters of logs, errors, capture availability, and output path |
| `POST /api/compilations/{id}/cancel` | Request termination (requires session token); poll until terminal |
| `GET /api/compilations/{id}/source` | Exact submitted source and compilation settings |
| `GET /api/compilations/{id}/capture` | Download validated `.cuteviz.json`, including partial/empty results when saved |

Statuses are `queued`, `running`, `succeeded`, `failed`, `cancelled`, and `timed_out`. `succeeded` requires a successful compiler exit and a nonempty valid capture. Capture outcome and job status are distinct. Cancel/timeout may have no saved capture. Job IDs are server-generated, and routes accept no filesystem paths. Unknown jobs and absent captures return 404.

The compilation endpoint intentionally executes trusted Python in a subprocess with the local user's permissions. Loopback Host checks, same-origin checks, and the session token prevent foreign browser origins from submitting source. Playback never executes embedded source, and the server process does not import CuTe/CUDA. See the README for the argument-factory contract and execution limits.

## Bank query

```http
POST /api/analysis/banks
Content-Type: application/json

{"mode":"scalar-warp","operation":"read","bank_count":32,
 "bank_width_bytes":4,"access_width_bytes":4,"base_byte_offset":0,
 "accesses":[{"lane":0,"byte_offset":0},
             {"lane":1,"byte_offset":128},
             {"lane":2,"byte_offset":0,"active":false}]}
```

`mode: "placement"` instead accepts up to 4,096 `byte_offsets` and reports all touched banks for each declared access width. It returns no conflict count. Scalar mode accepts at most one four-byte-aligned access per lane, requires the 32×4-byte model, and returns serialization rounds, extra distinct-word bank transactions, broadcast groups, and assumptions. Same-word writes return `unsupported` with a race explanation. Neither endpoint predicts execution time.
