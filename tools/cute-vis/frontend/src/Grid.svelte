<script lang="ts">
  import { getContext, onDestroy, onMount, tick, untrack } from "svelte";
  import { useSvelteFlow } from "@xyflow/svelte";
  import { canvasContext, type CanvasContext } from "./canvas";
  import {
    CELL_WIDTH,
    CELL_HEIGHT,
    GRID_LEFT,
    GRID_TOP,
    MAX_DETAIL_CELLS,
    dimensions,
    mappingPages,
    visibleWindow,
    type Window,
  } from "./tensorGeometry";
  import {
    api,
    show,
    type Cell,
    type View,
    type Selection,
    type Mapping,
    type GridSettings,
  } from "./api";

  let {
    id,
    width,
    height,
    nodeX,
    nodeY,
    captureId = "",
    objectId,
    view,
    selection,
    onselect,
    onmapping,
    settings,
    onsettings,
  }: {
    id: string;
    width: number;
    height: number;
    nodeX: number;
    nodeY: number;
    captureId?: string;
    objectId: string;
    view: View;
    selection: Selection | null;
    onselect: (role: string, cell: Cell) => void;
    onmapping: (role: string, result: Mapping | null) => void;
    settings?: GridSettings;
    onsettings?: (settings: GridSettings) => void;
  } = $props();
  const context = getContext<CanvasContext>(canvasContext);
  const { setCenter } = useSvelteFlow();
  const dims = $derived(dimensions(view.shape));
  let row = $state(untrack(() => settings?.row ?? 0));
  let column = $state(
    untrack(() => settings?.column ?? (dims.length > 1 ? 1 : -1)),
  );
  let fixed = $state<Record<number, number>>(
    untrack(() => settings?.fixed ?? {}),
  );
  let color = $state(untrack(() => settings?.color ?? "thread"));
  let before = $state(untrack(() => settings?.before ?? false));
  let bankBase = $state(untrack(() => settings?.bankBase ?? 0));
  const rows = $derived(Math.max(1, dims[row]?.size ?? 1));
  const columns = $derived(
    column < 0 ? 1 : Math.max(1, dims[column]?.size ?? 1),
  );
  const svgWidth = $derived(columns * CELL_WIDTH + GRID_LEFT);
  const svgHeight = $derived(rows * CELL_HEIGHT + GRID_TOP);
  // Browser SVG coordinates clamp at large magnitudes. Normalize drawing
  // coordinates while preserving the complete logical shape and exact indices.
  const scaleX = $derived(Math.min(1, 1_000_000 / svgWidth));
  const scaleY = $derived(Math.min(1, 1_000_000 / svgHeight));
  let svg = $state<SVGSVGElement>();
  let detailSvg = $state<SVGSVGElement>();
  let windowKey = $state("");
  let drawing = $state<{
    left: number;
    top: number;
    width: number;
    height: number;
    originX: number;
    originY: number;
    cellWidth: number;
    cellHeight: number;
  } | null>(null);
  const window = $derived(windowKey ? (JSON.parse(windowKey) as Window) : null);
  const overview = $derived(
    !!window && window.extent[0] * window.extent[1] > MAX_DETAIL_CELLS,
  );
  let result = $state<Mapping | null>(null),
    error = $state(""),
    pending = $state(false);

  $effect(() => {
    const state = { row, column, fixed, color, before, bankBase };
    untrack(() => onsettings?.(state));
  });

  function measure() {
    if (!svg || !context.viewportElement) return;
    const grid = svg.getBoundingClientRect(),
      canvas = context.viewportElement.getBoundingClientRect();
    const visible = visibleWindow(grid, canvas, rows, columns);
    windowKey = visible ? JSON.stringify(visible) : "";
    if (visible) {
      const left = Math.max(grid.left, canvas.left),
        top = Math.max(grid.top, canvas.top);
      drawing = {
        left: (left - grid.left) / context.viewport.zoom,
        top: (top - grid.top) / context.viewport.zoom,
        width: Math.min(grid.right, canvas.right) - left,
        height: Math.min(grid.bottom, canvas.bottom) - top,
        originX: grid.left - left + (GRID_LEFT / svgWidth) * grid.width,
        originY: grid.top - top + (GRID_TOP / svgHeight) * grid.height,
        cellWidth: (CELL_WIDTH / svgWidth) * grid.width,
        cellHeight: (CELL_HEIGHT / svgHeight) * grid.height,
      };
    } else drawing = null;
  }
  $effect(() => {
    // Svelte Flow transforms never resize the SVG element. Track the viewport
    // and node position as well as the element's ResizeObserver below.
    [
      context.viewport.x,
      context.viewport.y,
      context.viewport.zoom,
      context.viewportWidth,
      context.viewportHeight,
      width,
      height,
      nodeX,
      nodeY,
      rows,
      columns,
    ];
    const frame = requestAnimationFrame(measure);
    return () => cancelAnimationFrame(frame);
  });
  onMount(() => {
    const observer = new ResizeObserver(measure);
    if (svg) observer.observe(svg);
    measure();
    return () => observer.disconnect();
  });
  $effect(() => {
    const key = id;
    context.grids[key] = {
      jump,
      reveal: (coordinate) => {
        fixed = Object.fromEntries(
          coordinate
            .map((value, axis) => [axis, value])
            .filter(([axis]) => axis !== row && axis !== column),
        );
        tick().then(() =>
          jump(coordinate[row], column < 0 ? 0 : coordinate[column]),
        );
      },
      hit,
      select: (cell) => onselect(view.role, cell),
    };
    return () => {
      delete context.grids[key];
    };
  });
  onDestroy(() => untrack(() => onmapping(view.role, null)));

  const requestKey = $derived(
    JSON.stringify({
      window: overview ? { start: [0, 0], extent: [1, 1] } : window,
      overview,
      objectId,
      role: view.role,
      axes: column < 0 ? [row] : [row, column],
      fixed,
      selection,
      bankBase,
      captureId,
      bindings: context.bindings,
      focus_object_id: context.focusObjectId,
    }),
  );
  async function query(body: object, signal?: AbortSignal) {
    const data = await api<Mapping>(
      `/api/mapping/query${captureId ? `?capture_id=${encodeURIComponent(captureId)}` : ""}`,
      body,
      signal,
    );
    if (view.storage.value === "smem") {
      const placed = data.cells.filter((c) => c.byte_offset.status === "known");
      if (placed.length) {
        const banks = await api<{ placement: { banks: number[] }[] }>(
          "/api/analysis/banks",
          {
            byte_offsets: placed.map((c) => c.byte_offset.value),
            base_byte_offset: bankBase,
            access_width_bytes: Math.max(1, view.element_bits.value / 8 || 4),
          },
          signal,
        );
        placed.forEach((c, i) => (c.bank = banks.placement[i].banks[0]));
      }
    }
    return data;
  }
  $effect(() => {
    const key = requestKey;
    const input = JSON.parse(key);
    const controller = new AbortController();
    error = "";
    if (!input.window) {
      result = null;
      pending = false;
      untrack(() => onmapping(view.role, null));
      return;
    }
    pending = true;
    // Debounce wheel/drag events; requests never exceed the API's 4096-cell cap.
    const timer = setTimeout(async () => {
      try {
        const filters = Object.fromEntries(
          Object.entries(input.fixed).filter(
            ([axis]) => !input.axes.includes(Number(axis)),
          ),
        );
        const pages = mappingPages(input.window);
        const maps = await Promise.all(
          pages.map((page) =>
            query(
              {
                object_id: objectId,
                role: view.role,
                axes: input.axes,
                fixed: filters,
                ...page,
                selection: input.selection,
                bindings: input.bindings,
                focus_object_id: input.focus_object_id,
              },
              controller.signal,
            ),
          ),
        );
        if (controller.signal.aborted) return;
        const merged: Mapping = {
          ...maps[0],
          start: [0, 0],
          extent: [rows, columns],
          cells: input.overview
            ? []
            : maps.flatMap((map) =>
                map.cells.map((cell) => ({
                  ...cell,
                  x: cell.x + map.start[1],
                  y: cell.y + map.start[0],
                })),
              ),
        };
        result = merged;
        onmapping(view.role, merged);
      } catch (e) {
        if (!controller.signal.aborted) {
          result = null;
          error = e instanceof Error ? e.message : String(e);
          onmapping(view.role, null);
        }
      } finally {
        if (!controller.signal.aborted) pending = false;
      }
    }, 65);
    return () => {
      clearTimeout(timer);
      controller.abort();
    };
  });

  const cellsByPosition = $derived(
    new Map(result?.cells.map((cell) => [`${cell.x},${cell.y}`, cell]) ?? []),
  );
  function hit(clientX: number, clientY: number): Cell | null {
    if (!detailSvg || !drawing || overview) return null;
    const rect = detailSvg.getBoundingClientRect();
    if (
      clientX < rect.left ||
      clientX >= rect.right ||
      clientY < rect.top ||
      clientY >= rect.bottom
    )
      return null;
    const x =
      (((clientX - rect.left) / rect.width) * drawing.width - drawing.originX) /
      drawing.cellWidth;
    const y =
      (((clientY - rect.top) / rect.height) * drawing.height -
        drawing.originY) /
      drawing.cellHeight;
    if (
      x - Math.floor(x) > 74 / CELL_WIDTH ||
      y - Math.floor(y) > 36 / CELL_HEIGHT
    )
      return null;
    return cellsByPosition.get(`${Math.floor(x)},${Math.floor(y)}`) ?? null;
  }

  function jump(y: number, x: number) {
    if (!svg || !context.viewportElement) return;
    const rect = svg.getBoundingClientRect(),
      canvas = context.viewportElement.getBoundingClientRect();
    const viewport = context.viewport;
    const flowWidth = rect.width / viewport.zoom,
      flowHeight = rect.height / viewport.zoom;
    const centerX =
      (rect.left - canvas.left - viewport.x) / viewport.zoom +
      ((GRID_LEFT + (x + 0.5) * CELL_WIDTH) / svgWidth) * flowWidth;
    const centerY =
      (rect.top - canvas.top - viewport.y) / viewport.zoom +
      ((GRID_TOP + (y + 0.5) * CELL_HEIGHT) / svgHeight) * flowHeight;
    const zoom = Math.min(
      100_000_000,
      Math.max(
        78 / ((flowWidth * CELL_WIDTH) / svgWidth),
        40 / ((flowHeight * CELL_HEIGHT) / svgHeight),
      ),
    );
    setCenter(centerX, centerY, { zoom });
  }
  function zoomAt(event: MouseEvent) {
    if (!svg) return;
    const rect = svg.getBoundingClientRect();
    const x = Math.max(
      0,
      Math.min(
        columns - 1,
        Math.floor(
          (((event.clientX - rect.left) / rect.width) * svgWidth - GRID_LEFT) /
            CELL_WIDTH,
        ),
      ),
    );
    const y = Math.max(
      0,
      Math.min(
        rows - 1,
        Math.floor(
          (((event.clientY - rect.top) / rect.height) * svgHeight - GRID_TOP) /
            CELL_HEIGHT,
        ),
      ),
    );
    jump(y, x);
  }
  const palette = [
    "#cfe8e4",
    "#d9e3fa",
    "#f7ddc6",
    "#e9d7f0",
    "#efe4b6",
    "#cfe4f3",
    "#e3e9bf",
    "#f3d9de",
  ];
  function fill(cell: Cell): string {
    const owner = cell.owners[0];
    const value =
      color === "bank"
        ? cell.bank
        : color === "warp"
          ? owner?.warp
          : color === "value"
            ? owner?.value
            : owner?.thread;
    if (cell.in_focus && value == null) return "#cce8f4";
    return value == null ? "#edf0f0" : palette[value % palette.length];
  }
  function text(cell: Cell) {
    if (before && cell.unswizzled_offset != null)
      return `${cell.unswizzled_offset}`;
    if (color === "bank" && cell.bank != null) return `B${cell.bank}`;
    if (cell.owners.length)
      return `T${cell.owners[0].thread}·V${cell.owners[0].value}${cell.owners.length > 1 ? "+" : ""}`;
    if (cell.tmem.status === "known")
      return `${cell.tmem.value.data_path}:${cell.tmem.value.column}`;
    if (cell.fragment_index?.status === "known")
      return `V${cell.fragment_index.value}`;
    return cell.element_offset.status === "known"
      ? String(cell.element_offset.value)
      : "·";
  }
  async function key(event: KeyboardEvent, cell: Cell) {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      onselect(view.role, cell);
      return;
    }
    const delta: Record<string, number[]> = {
      ArrowLeft: [-1, 0],
      ArrowRight: [1, 0],
      ArrowUp: [0, -1],
      ArrowDown: [0, 1],
    };
    if (!delta[event.key]) return;
    event.preventDefault();
    const [dx, dy] = delta[event.key];
    const x = cell.x + dx,
      y = cell.y + dy;
    if (x < 0 || x >= columns || y < 0 || y >= rows) return;
    let next = result?.cells.find((c) => c.x === x && c.y === y);
    if (!next) {
      try {
        const axes = column < 0 ? [row] : [row, column];
        const map = await query({
          object_id: objectId,
          role: view.role,
          axes,
          fixed: Object.fromEntries(
            Object.entries(fixed).filter(
              ([axis]) => !axes.includes(Number(axis)),
            ),
          ),
          start: [y, x],
          extent: [1, 1],
          bindings: context.bindings,
          focus_object_id: context.focusObjectId,
        });
        next = { ...map.cells[0], x, y };
        jump(y, x);
      } catch (e) {
        error = e instanceof Error ? e.message : String(e);
        return;
      }
    }
    onselect(view.role, next);
    await tick();
    document
      .getElementById(`cell-${objectId}-${view.role}-${x}-${y}`)
      ?.focus({ preventScroll: true });
  }
  const visibleRows = $derived(
    result ? [...new Set(result.cells.map((c) => c.y))] : [],
  );
  const visibleColumns = $derived(
    result ? [...new Set(result.cells.map((c) => c.x))] : [],
  );
</script>

<section class="grid-panel" aria-label={`Operand ${view.role}`}>
  <div class="grid-metadata nopan nowheel">
    <div class="panel-heading">
      <div>
        <span class="operand"
          >{view.role === "tensor" ? "Tensor" : view.role}</span
        ><span class="tag">{show(view.storage)}</span><span class="muted"
          >{view.dtype.status === "known" ? show(view.dtype) : ""}</span
        >
      </div>
      <span class="muted" aria-live="polite">{pending ? "Updating…" : ""}</span>
    </div>
    <div class="shape">
      <span class="eyebrow">Shape / nested modes</span><code
        >{JSON.stringify(view.shape)}</code
      >
    </div>
    <div class="grid-controls">
      <label
        >Rows <select
          aria-label={`${view.role} row axis`}
          bind:value={row}
          onchange={() => {
            if (row === column) column = -1;
          }}
        >
          {#each dims as d, i}<option value={i}
              >mode {d.path} ({d.size ?? "?"})</option
            >{/each}
        </select></label
      >
      <label
        >Columns <select
          aria-label={`${view.role} column axis`}
          bind:value={column}
        >
          <option value={-1}>None</option
          >{#each dims as d, i}{#if i !== row}<option value={i}
                >mode {d.path} ({d.size ?? "?"})</option
              >{/if}{/each}
        </select></label
      >
      <span class="page-window"
        >Full {rows.toLocaleString()} × {columns.toLocaleString()} grid</span
      >
      {#each dims as d, i}{#if i !== row && i !== column}<label
            >Fix mode {d.path}<input
              aria-label={`${view.role} fix ${d.path}`}
              type="number"
              min="0"
              max={(d.size ?? 1) - 1}
              value={fixed[i] ?? 0}
              oninput={(e) =>
                (fixed = { ...fixed, [i]: Number(e.currentTarget.value) })}
            /></label
          >{/if}{/each}
    </div>
    <div class="grid-controls subtle">
      <label
        >Color <select aria-label={`${view.role} color`} bind:value={color}
          ><option value="thread">Thread</option><option value="warp"
            >Warp</option
          ><option value="value">Value index</option
          >{#if view.storage.value === "smem"}<option value="bank"
              >Bank placement</option
            >{/if}</select
        ></label
      >
      {#if view.storage.value === "smem"}<label
          >Assumed base bytes<input
            type="number"
            min="0"
            bind:value={bankBase}
            aria-label={`${view.role} bank base`}
          /></label
        >{/if}
      {#if view.layout?.kind === "compose" && view.layout?.left?.kind === "swizzle"}<label
          class="check"
          ><input type="checkbox" bind:checked={before} /> Unswizzled offsets</label
        >{/if}
    </div>
    {#if error}<p class="notice" role="status">{error}</p>{/if}
  </div>
  <div
    class="tensor-grid"
    class:overview
    data-detail={overview ? "overview" : "cells"}
    role="grid"
    tabindex="0"
    aria-label={`${view.role} coordinate grid`}
    aria-rowcount={rows}
    aria-colcount={columns}
    ondblclick={zoomAt}
  >
    <svg
      class="tensor-extent"
      bind:this={svg}
      width="100%"
      height="100%"
      viewBox={`0 0 ${svgWidth * scaleX} ${svgHeight * scaleY}`}
      preserveAspectRatio="none"
      aria-hidden="true"
    >
      <rect
        x={GRID_LEFT * scaleX}
        y={GRID_TOP * scaleY}
        width={columns * CELL_WIDTH * scaleX}
        height={rows * CELL_HEIGHT * scaleY}
        fill={overview ? "#e3ede8" : "#fafcfb"}
      />
      {#each result?.footprints ?? [] as footprint}{#if footprint.exact && footprint.extent.every((n) => n > 0)}
          <rect
            class="tile-footprint"
            x={(GRID_LEFT + footprint.start[1] * CELL_WIDTH) * scaleX}
            y={(GRID_TOP + footprint.start[0] * CELL_HEIGHT) * scaleY}
            width={footprint.extent[1] * CELL_WIDTH * scaleX}
            height={footprint.extent[0] * CELL_HEIGHT * scaleY}
            fill="#57a9cc"
            fill-opacity="0.3"
            ><title>{footprint.label} footprint in backing tensor</title></rect
          >
        {/if}{/each}
    </svg>
    {#if !overview && drawing && result}
      <!-- Keep SVG coordinates and hit testing close to the camera, even when
           the containing card spans millions of logical rows or columns. -->
      <svg
        class="tensor-detail"
        bind:this={detailSvg}
        style:left={`${drawing.left}px`}
        style:top={`${drawing.top}px`}
        width={drawing.width / context.viewport.zoom}
        height={drawing.height / context.viewport.zoom}
        viewBox={`0 0 ${drawing.width} ${drawing.height}`}
        role="presentation"
      >
        {#each visibleColumns as x}<text
            class="axis"
            x={drawing.originX + (x + 0.5) * drawing.cellWidth}
            y={drawing.originY - drawing.cellHeight * 0.25}
            text-anchor="middle">{x}</text
          >{/each}
        {#each visibleRows as y}<text
            class="axis"
            x={drawing.originX - drawing.cellWidth * 0.2}
            y={drawing.originY + (y + 0.65) * drawing.cellHeight}
            text-anchor="end">{y}</text
          >{/each}
        {#each result.cells as cell (cell.coordinate.join(","))}
          {@const x = drawing.originX + cell.x * drawing.cellWidth}
          {@const y = drawing.originY + cell.y * drawing.cellHeight}
          <g
            id={`cell-${objectId}-${view.role}-${cell.x}-${cell.y}`}
            class="nopan"
            role="gridcell"
            tabindex={cell === result.cells[0] || cell.selected ? 0 : -1}
            aria-selected={cell.selected}
            data-in-tile={cell.in_focus ?? false}
            aria-rowindex={cell.y + 1}
            aria-colindex={cell.x + 1}
            aria-label={`${view.role} coordinate ${cell.coordinate.join(",")}, ${text(cell)}`}
            onclick={() => onselect(view.role, cell)}
            onkeydown={(event) => key(event, cell)}
          >
            <title>{cell.explanation}</title>
            <rect
              {x}
              {y}
              width={(drawing.cellWidth * 74) / CELL_WIDTH}
              height={(drawing.cellHeight * 36) / CELL_HEIGHT}
              rx={Math.min(4, drawing.cellWidth * 0.05)}
              fill={fill(cell)}
              stroke={cell.selected ? "#c75322" : "transparent"}
              stroke-width={cell.selected
                ? Math.min(3, drawing.cellWidth * 0.08)
                : 0}
            />
            <text
              x={x + drawing.cellWidth * 0.47}
              y={y + drawing.cellHeight * 0.55}
              style:font-size={`${Math.min(12, drawing.cellWidth / Math.max(8, text(cell).length * 0.65), drawing.cellHeight * 0.3)}px`}
              text-anchor="middle">{text(cell)}</text
            >
          </g>
        {/each}
      </svg>
    {/if}
    {#if overview}<div class="tensor-overview">
        <div
          class="overview-label"
          style:transform={`scale(${1 / context.viewport.zoom})`}
        >
          <strong>{rows.toLocaleString()} × {columns.toLocaleString()}</strong>
          <span
            >Full tensor · Double-click or go to a cell to inspect details</span
          >
        </div>
      </div>{/if}
  </div>
  <div class="grid-caption nopan">
    {#if result?.footprints?.length}<span class="footprint-legend"
        >Blue = {result.footprints.map((f) => f.label).join(", ")} footprint in this
        backing tensor</span
      >{/if}
    {#each result?.mapping_diagnostics ?? [] as diagnostic}<p
        class="field-note"
      >
        {diagnostic}
      </p>{/each}
    <span
      >{overview
        ? "Overview · zoom in for cell mappings"
        : `${result?.cells.length.toLocaleString() ?? "0"} cells detailed in this viewport`}
      · full {(rows * columns).toLocaleString()}-cell slice</span
    >
    <p class="legend">
      {color === "bank"
        ? "B = starting bank · 32 banks × 4 bytes · placement only"
        : "T = thread · V = value index · + = multiple owners"}<br
      />{view.address_unit === "tmem"
        ? "Numbers are data path : column in tensor memory."
        : "Unowned cells show element offsets when known; · means unresolved."} Orange
      outline = linked selection. Arrow keys move between cells.
    </p>
    {#if view.layout_status.status !== "known"}<p class="field-note">
        {view.layout_status.reason}
      </p>{/if}
  </div>
</section>
