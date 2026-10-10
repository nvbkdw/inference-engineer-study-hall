<script lang="ts">
  import { onMount, tick } from "svelte";
  import TensorCanvas from "./TensorCanvas.svelte";
  import { relatedObjects } from "./canvas";
  import { SvelteFlowProvider } from "@xyflow/svelte";
  import {
    api,
    show,
    type Capture,
    type CapturedObject,
    type Selection,
    type Cell,
    type Mapping,
  } from "./api";
  let { capture, captureId = "" }: { capture: Capture; captureId?: string } =
    $props();
  let search = $state("");
  let canvasRootId = $state("");
  let bindings = $state<Record<string, number>>({});
  let canvas = $state<{
    reveal: (objectId: string, role: string, coordinate: number[]) => void;
  }>();
  const coordinateParameters = $derived([
    ...new Map(
      capture.objects.flatMap((object) =>
        object.relationships.flatMap((relation) =>
          Object.entries(relation.mapping?.parameters ?? {}).map(
            ([name, value]) => [name, value.label] as const,
          ),
        ),
      ),
    ).entries(),
  ]);
  $effect(() => {
    for (const [name] of coordinateParameters)
      if (!(name in bindings)) bindings[name] = 0;
  });
  let showSource = $state(true);
  let objectId = $state(""),
    selection = $state<Selection | null>(null),
    selectedCell = $state<Cell | null>(null),
    activeRole = $state("");
  let mappings = $state<Record<string, Mapping>>({});
  let thread = $state<number | undefined>(),
    warp = $state<number | undefined>(),
    warpgroup = $state<number | undefined>(),
    value = $state<number | undefined>();
  let offsets = $state("0, 4, 8, 12, 16, 20, 24, 28"),
    operation = $state("read"),
    bankReport = $state<any>(null),
    bankError = $state("");
  const current = $derived(capture?.objects.find((o) => o.id === objectId));
  const canvasObjects = $derived(relatedObjects(capture.objects, canvasRootId));
  const canvasRoot = $derived(
    capture.objects.find((object) => object.id === canvasRootId),
  );
  const source = $derived(
    capture?.sources.find((s) => s.id === current?.source?.source_id),
  );
  const filtered = $derived(
    capture?.objects.filter((o) =>
      `${o.label} ${o.kind}`.toLowerCase().includes(search.toLowerCase()),
    ) ?? [],
  );
  const activeMap = $derived(mappings[`${objectId}:${activeRole}`]);
  let coordinateError = $state("");
  $effect(() => {
    coordinateError = "";
    if (
      selection?.coordinate &&
      (!selection.role || selection.role === activeRole) &&
      (!selection.object_id || selection.object_id === objectId)
    ) {
      const coordinate = selection.coordinate;
      const controller = new AbortController();
      // Keep the explanation current even when its tensor is outside the viewport.
      api<Mapping>(
        `/api/mapping/query${captureId ? `?capture_id=${encodeURIComponent(captureId)}` : ""}`,
        {
          object_id: objectId,
          role: activeRole,
          axes: coordinate.length === 1 ? [0] : [0, 1],
          start: [coordinate[0], coordinate[1] ?? 0],
          fixed: Object.fromEntries(
            coordinate.map((value, axis) => [axis, value]).slice(2),
          ),
          extent: [1, 1],
          bindings,
        },
        controller.signal,
      )
        .then((result) => {
          if (!controller.signal.aborted) selectedCell = result.cells[0];
        })
        .catch((error) => {
          if (!controller.signal.aborted)
            coordinateError =
              error instanceof Error ? error.message : String(error);
        });
      return () => controller.abort();
    }
  });
  const ownerRows = $derived(
    activeMap?.cells.flatMap((c) => c.owners.map((o) => ({ ...o, cell: c }))) ??
      [],
  );
  $effect(() => {
    const span = current?.source;
    if (span)
      tick().then(() => {
        const panel = document.querySelector(".source-view");
        const line = panel?.querySelector(".source-active");
        if (panel && line)
          panel.scrollTop +=
            line.getBoundingClientRect().top -
            panel.getBoundingClientRect().top -
            100;
      });
  });
  onMount(() => {
    if (capture.objects.length) {
      objectId = capture.objects[0].id;
      canvasRootId = objectId;
      activeRole = capture.objects[0].views[0]?.role ?? "";
    }
  });
  function choose(obj: CapturedObject) {
    canvasRootId = obj.id;
    objectId = obj.id;
    selection = null;
    selectedCell = null;
    activeRole = obj.views[0]?.role ?? "";
    thread = warp = warpgroup = value = undefined;
  }
  function select(role: string, cell: Cell) {
    activeRole = role;
    selectedCell = cell;
    selection = { object_id: objectId, role, coordinate: cell.coordinate };
    thread = warp = warpgroup = value = undefined;
  }
  function activate(obj: CapturedObject, role: string) {
    if (objectId !== obj.id) {
      selectedCell = null;
      if (selection?.coordinate) selection = null;
    }
    objectId = obj.id;
    activeRole = role;
  }
  function selectCanvas(obj: CapturedObject, role: string, cell: Cell) {
    objectId = obj.id;
    select(role, cell);
  }
  function recordMapping(
    obj: CapturedObject,
    role: string,
    mapping: Mapping | null,
  ) {
    const key = `${obj.id}:${role}`;
    if (mapping) mappings[key] = mapping;
    else delete mappings[key];
  }
  function selectOwners() {
    selection = { thread, warp, warpgroup, value };
    selectedCell = null;
  }
  function selectLine(line: number) {
    const obj = capture?.objects.find(
      (o) =>
        o.source &&
        o.source.source_id === source?.id &&
        o.source.line <= line &&
        o.source.end_line >= line,
    );
    if (obj) choose(obj);
  }
  async function banks() {
    try {
      const entries = offsets.split(",").map((x) => x.trim());
      if (
        entries.length > 32 ||
        entries.some((x) => x !== "-" && !/^\d+$/.test(x))
      )
        throw new Error(
          "Enter up to 32 byte offsets separated by commas. Use - for an inactive lane.",
        );
      bankReport = await api("/api/analysis/banks", {
        mode: "scalar-warp",
        operation,
        accesses: entries.map((x, lane) => ({
          lane,
          byte_offset: x === "-" ? 0 : Number(x),
          active: x !== "-",
        })),
      });
      bankError = "";
    } catch (e) {
      bankError = String(e);
      bankReport = null;
    }
  }
</script>

{#snippet ownerPanel()}
  {#if current}
    <section
      class="detail-panel ownership-panel"
      aria-label="Thread value table"
    >
      <div class="panel-heading">
        <h2>Thread / value</h2>
        <select aria-label="Active operand" bind:value={activeRole}
          >{#each current.views as v}<option value={v.role}>{v.role}</option
            >{/each}</select
        >
      </div>
      <p class="muted">
        Select a thread/value to highlight matching captured ownership across
        the canvas. Value indices are local to each operand.
      </p>
      <div class="table-scroll">
        <table>
          <thead
            ><tr
              ><th>Thread</th><th>Warp</th><th>Value</th><th>Coordinate</th></tr
            ></thead
          ><tbody
            >{#each ownerRows as owner}<tr
                class:row-selected={owner.cell.selected}
                ><td
                  ><button
                    aria-label={`Select thread ${owner.thread} value ${owner.value}`}
                    onclick={() => {
                      thread = owner.thread;
                      value = owner.value;
                      warp = warpgroup = undefined;
                      selectOwners();
                    }}>{owner.thread}</button
                  ></td
                ><td>{owner.warp}</td><td>{owner.value}</td><td
                  ><button onclick={() => select(activeRole, owner.cell)}
                    >({owner.cell.coordinate.join(", ")})</button
                  ></td
                ></tr
              >{/each}</tbody
          >
        </table>
        {#if !ownerRows.length}<p class="field-note">
            {current.views.find((v) => v.role === activeRole)?.ownership
              .reason ?? "No ownership in this viewport."}
          </p>{/if}
      </div>
    </section>
  {/if}
{/snippet}

<div class="app-shell" class:source-hidden={!showSource}>
  <aside class="sidebar" hidden={!showSource}>
    <div class="sidebar-heading">
      <span class="eyebrow">CAPTURED OBJECTS</span><span class="count"
        >{capture.objects.length}</span
      >
    </div>
    <input
      class="search"
      aria-label="Search captured objects"
      placeholder="Search tensors, copies, MMA…"
      bind:value={search}
    />
    <nav aria-label="Captured objects" class="object-list">
      {#each filtered as obj}<button
          class:active={canvasRootId === obj.id}
          onclick={() => choose(obj)}
          ><span class={`kind ${obj.kind}`}
            >{obj.kind.slice(0, 1).toUpperCase()}</span
          ><span class="object-name"
            >{obj.label}<small
              >{obj.kind}{obj.source ? ` · line ${obj.source.line}` : ""}</small
            ></span
          >{#if obj.diagnostics.length}<span title="Partial inspection">!</span
            >{/if}</button
        >{/each}
    </nav>
    <div class="source-title">
      <span class="eyebrow">EMBEDDED SOURCE</span><span class="muted"
        >{source?.path.split("/").pop() ?? "Unavailable"}</span
      >
    </div>
    <div class="source-view" aria-label="Source viewer">
      {#if source}{#each source.text.split("\n") as line, i}<button
            class:source-active={current?.source &&
              i + 1 >= current.source.line &&
              i + 1 <= current.source.end_line}
            onclick={() => selectLine(i + 1)}
            aria-label={`Source line ${i + 1}`}
            ><span>{i + 1}</span><code>{line || " "}</code></button
          >{/each}{:else}<p class="muted">
          This capture has no source snapshot.
        </p>{/if}
    </div>
    <div class="capture-meta">
      <span class="eyebrow">CAPTURE DETAILS</span>
      <p>{show(capture.target)} · {capture.outcome}</p>
      <p>{show(capture.compiler)}</p>
      <small>Schema {capture.schema_version} · Captured data only</small
      >{#if Object.keys(capture.parameters).length}<pre>{JSON.stringify(
            capture.parameters,
            null,
            2,
          )}</pre>{/if}
    </div>
  </aside>
  <main class="workspace">
    {#each capture.diagnostics as diagnostic}<p class="notice">
        {diagnostic}
      </p>{/each}
    {#if current}
      <div class="workspace-title">
        <div>
          <p class="eyebrow">EXPLORE A KERNEL OBJECT</p>
          <h1>{current.label}</h1>
          <p class="muted">
            {current.kind === "mma"
              ? show(current.fields.family)
              : current.kind === "copy"
                ? "Source → destination, linked by logical element"
                : "Choose a coordinate to follow its storage mapping"}
          </p>
        </div>
        <div class="inspect-heading-actions">
          <button onclick={() => (showSource = !showSource)}
            >{showSource ? "Hide source" : "Show source"}</button
          ><span class="tag">{current.kind}</span>
        </div>
      </div>
      {#each current.diagnostics as diagnostic}<p class="notice">
          {diagnostic}
        </p>{/each}
      {#if current.fields.execution_group}<div class="group-note">
          <span class="eyebrow">EXECUTION GROUP</span><span
            >{current.fields.execution_group.value.kind} · {current.fields
              .execution_group.value.size}
            {current.fields.execution_group.value.size === 1
              ? "issuing thread"
              : "threads"}{#if current.fields.execution_group.value.note}<br
              /><small>{current.fields.execution_group.value.note}</small
              >{/if}</span
          >
        </div>{/if}
      {#if current.relationships.length}<div class="relationships">
          {#each current.relationships as relation}<button
              onclick={() => {
                const o = capture?.objects.find(
                  (o) => o.id === relation.target,
                );
                if (o) choose(o);
              }}
              >{relation.kind === "declared_parent"
                ? "Declared parent"
                : relation.kind === "derived_parent"
                  ? "Backing tensor"
                  : "Observed operand"}: {capture.objects.find(
                (o) => o.id === relation.target,
              )?.label}
              {relation.transform ? `· ${relation.transform}` : ""} ↗</button
            >{/each}
        </div>{/if}
      <div class="selection-bar">
        <span class="eyebrow">LINKED SELECTION</span><label
          >Thread<input
            aria-label="Select thread"
            type="number"
            min="0"
            bind:value={thread}
            oninput={() => queueMicrotask(selectOwners)}
          /></label
        ><label
          >Warp<input
            aria-label="Select warp"
            type="number"
            min="0"
            bind:value={warp}
            oninput={() => queueMicrotask(selectOwners)}
          /></label
        ><label
          >Warpgroup<input
            aria-label="Select warpgroup"
            type="number"
            min="0"
            bind:value={warpgroup}
            oninput={() => queueMicrotask(selectOwners)}
          /></label
        ><label
          >Value index<input
            aria-label="Select value index"
            type="number"
            min="0"
            bind:value
            oninput={() => queueMicrotask(selectOwners)}
          /></label
        ><button
          onclick={() => {
            selection = null;
            selectedCell = null;
            thread = warp = warpgroup = value = undefined;
          }}>Clear</button
        >
      </div>
      {#if coordinateParameters.length}<div
          class="coordinate-bindings"
          aria-label="Tile inspection coordinates"
        >
          <span class="eyebrow">INSPECTION COORDINATES</span>
          {#each coordinateParameters as [name, label]}<label
              >{label}<input
                type="number"
                min="0"
                step="1"
                aria-label={`Inspection ${label}`}
                value={bindings[name] ?? 0}
                onchange={(event) => {
                  bindings = {
                    ...bindings,
                    [name]: Math.max(
                      0,
                      Math.trunc(Number(event.currentTarget.value) || 0),
                    ),
                  };
                }}
              /></label
            >{/each}
          <small
            >Assumed CTA / tile indices for this visualization, not recorded GPU
            execution.</small
          >
        </div>{/if}
      {#key canvasRootId}<SvelteFlowProvider>
          <TensorCanvas
            bind:this={canvas}
            objects={canvasObjects}
            scopeLabel={canvasRoot?.label ?? "Selected object"}
            focusObjectId={canvasRootId}
            {bindings}
            {captureId}
            {objectId}
            {activeRole}
            {selection}
            {mappings}
            onactivate={activate}
            onselect={selectCanvas}
            onmapping={recordMapping}
            owners={ownerPanel}
          />
        </SvelteFlowProvider>{/key}
      <div class="detail-panels canvas-details">
        <section
          class="detail-panel explanation"
          aria-label="Mapping explanation"
        >
          <p class="eyebrow">FOLLOW THE COORDINATE</p>
          <h2>
            {selectedCell
              ? `${selection?.role ?? activeRole} (${selectedCell.coordinate.join(", ")})`
              : "Select a cell to inspect it"}
          </h2>
          {#if coordinateError}<p role="alert">{coordinateError}</p>{/if}
          {#if selectedCell && !coordinateError}<p class="explanation-text">
              {selectedCell.explanation}
            </p>
            {#if selectedCell.coordinate_mappings?.length}<div
                class="coordinate-trail"
                aria-label="Backing coordinate mapping"
              >
                <div>
                  <strong>{current.label}</strong><code
                    >({selectedCell.coordinate.join(", ")})</code
                  >
                </div>
                {#each selectedCell.coordinate_mappings as mapping}<span
                    aria-hidden="true">↓</span
                  >
                  <div>
                    <strong>{mapping.label}</strong>
                    <button
                      onclick={() =>
                        canvas?.reveal(
                          mapping.object_id,
                          mapping.role,
                          mapping.coordinate,
                        )}
                      disabled={!mapping.in_bounds}>Show in canvas</button
                    ><code>({mapping.coordinate.join(", ")})</code>
                    <small>{mapping.calculation.join("; ")}</small>
                    <small
                      >Backing element offset: {show(mapping.element_offset)} · byte
                      offset: {show(mapping.byte_offset)}</small
                    >
                    {#if !mapping.in_bounds}<small class="out-of-bounds"
                        >Outside the backing tensor — requires a kernel
                        predicate.</small
                      >{/if}
                  </div>{/each}
              </div>{/if}
            <dl>
              {#if selectedCell.fragment_index?.status === "known"}<dt>
                  Fragment index
                </dt>
                <dd>{show(selectedCell.fragment_index)}</dd>{/if}
              <dt>Element offset</dt>
              <dd>{show(selectedCell.element_offset)}</dd>
              <dt>Byte offset</dt>
              <dd>{show(selectedCell.byte_offset)}</dd>
              <dt>Tensor memory</dt>
              <dd>{show(selectedCell.tmem)}</dd>
            </dl>
            {#each selectedCell.links as link}<p class="copy-link">
                → {link.role} ({link.coordinate.join(", ")}) · thread {link.thread},
                value {link.value}
              </p>{/each}{:else}<p class="muted">
              The selected coordinate, its source location, and related operands
              stay together. Offsets are relative to the captured view; runtime
              base pointers remain unresolved.
            </p>{/if}
        </section>
      </div>
      <details class="banks">
        <summary
          >Shared-memory access lab <span class="muted"
            >Explicit scalar 32-bit warp model</span
          ></summary
        >
        <p>
          Declare one byte offset per lane, separated by commas. Use <code
            >-</code
          > for inactive lanes. Base byte offset is 0; offsets must include any base
          displacement. The model uses 32 four-byte banks.
        </p>
        <label
          >Lane byte offsets<textarea
            aria-label="Lane byte offsets"
            bind:value={offsets}></textarea></label
        >
        <div class="bank-actions">
          <label
            >Access<select aria-label="Access operation" bind:value={operation}
              ><option value="read">Read (same-word broadcast)</option><option
                value="write">Write</option
              ></select
            ></label
          ><button class="primary" onclick={banks}
            >Analyze declared access</button
          >
        </div>
        {#if bankError}<p class="notice">
            {bankError}
          </p>{/if}{#if bankReport}<div class="bank-result" aria-live="polite">
            {#if bankReport.status === "known"}<strong
                >{bankReport.serialization_rounds} serialization round(s)</strong
              ><span
                >{bankReport.extra_bank_transactions} extra bank transaction(s) ·
                {bankReport.active_lanes} active lanes · {bankReport.broadcasts
                  .length} broadcast group(s)</span
              >{:else}<p>{bankReport.reason}</p>{/if}
            <p>{bankReport.assumptions.join(". ")}.</p>
          </div>{/if}
      </details>
      <p class="footer-note">
        Observations describe compilation, including symbolic control flow. They
        are not GPU execution history or runtime tensor values.
      </p>
    {:else}<h1>No probes captured</h1>
      <p>
        Add inspection probes and compile again with the cache controls
        described above.
      </p>{/if}
  </main>
</div>
