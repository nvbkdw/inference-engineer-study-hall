<script lang="ts">
  import { setContext, untrack, tick, type Snippet } from "svelte";
  import {
    SvelteFlow,
    Background,
    MiniMap,
    useSvelteFlow,
    type Node,
    type Edge,
  } from "@xyflow/svelte";
  import "@xyflow/svelte/dist/style.css";
  import TensorNode from "./TensorNode.svelte";
  import OwnershipNode from "./OwnershipNode.svelte";
  import { dimensions, naturalCardSize } from "./tensorGeometry";
  import { canvasContext, type CanvasContext } from "./canvas";
  import type {
    CapturedObject,
    Selection,
    Cell,
    Mapping,
    GridSettings,
  } from "./api";

  let {
    objects,
    scopeLabel,
    focusObjectId,
    bindings,
    captureId,
    objectId,
    activeRole,
    selection,
    mappings,
    onactivate,
    onselect,
    onmapping,
    owners,
  }: {
    objects: CapturedObject[];
    scopeLabel: string;
    focusObjectId: string;
    bindings: Record<string, number>;
    captureId: string;
    objectId: string;
    activeRole: string;
    selection: Selection | null;
    mappings: Record<string, Mapping>;
    onactivate: (object: CapturedObject, role: string) => void;
    onselect: (object: CapturedObject, role: string, cell: Cell) => void;
    onmapping: (
      object: CapturedObject,
      role: string,
      mapping: Mapping | null,
    ) => void;
    owners: Snippet;
  } = $props();
  const { fitView, setZoom, getZoom, setViewport } = useSvelteFlow();
  const nodeTypes = { tensor: TensorNode, ownership: OwnershipNode };
  let settings = $state<Record<string, GridSettings>>({});
  let expanded = $state(false),
    showOwners = $state(true),
    showLinks = $state(false);
  let viewport = $state({ x: 0, y: 0, zoom: 0.7 });
  let viewportElement = $state<HTMLDivElement>();
  let viewportWidth = $state(0),
    viewportHeight = $state(0);
  let grids = $state<CanvasContext["grids"]>({});
  let jumpRow = $state(0),
    jumpColumn = $state(0);
  const activeId = $derived(`${objectId}:${activeRole}`);
  const activeView = $derived(
    objects
      .find((o) => o.id === objectId)
      ?.views.find((v) => v.role === activeRole),
  );
  const activeDims = $derived(activeView ? dimensions(activeView.shape) : []);
  const activeRows = $derived(
    activeDims[settings[activeId]?.row ?? 0]?.size ?? 1,
  );
  const activeColumns = $derived(
    activeDims[settings[activeId]?.column ?? 1]?.size ?? 1,
  );
  function jumpToCell() {
    grids[activeId]?.jump(
      Math.max(0, Math.min(activeRows - 1, Math.trunc(jumpRow || 0))),
      Math.max(0, Math.min(activeColumns - 1, Math.trunc(jumpColumn || 0))),
    );
  }
  function layout(previous: Node[] = []): Node[] {
    let column = 0,
      x = 0,
      y = 0,
      rowHeight = 0;
    return [
      {
        id: "owners",
        type: "ownership",
        position: { x: -340, y: 0 },
        data: {},
        width: 300,
        height: 610,
        dragHandle: ".ownership-title",
        deletable: false,
      },
      ...objects.flatMap((object) => {
        if (column && column + Math.min(object.views.length, 3) > 3) {
          y += rowHeight + 60;
          x = rowHeight = column = 0;
        }
        return object.views.map((view) => {
          const id = `${object.id}:${view.role}`;
          const saved = previous.find((n) => n.id === id);
          const size = saved
            ? { width: saved.width, height: saved.height }
            : naturalCardSize(view, settings[id]);
          const node: Node = {
            id,
            type: "tensor",
            position: { x, y },
            data: { object, view },
            ...size,
            dragHandle: ".node-title",
            deletable: false,
            ariaLabel: `${object.label} ${view.role}`,
          };
          x += (node.width ?? 640) + 60;
          rowHeight = Math.max(rowHeight, node.height ?? 460);
          if (++column === 3) {
            y += rowHeight + 60;
            x = rowHeight = column = 0;
          }
          return node;
        });
      }),
    ];
  }
  let nodes = $state.raw<Node[]>(untrack(layout));
  $effect(() => {
    const camera = viewport;
    if (Math.max(Math.abs(camera.x), Math.abs(camera.y)) < 4_000_000) return;
    // Rebase the world origin before browser hit testing loses precision at
    // large CSS translations. Relative positions, sizes and zoom stay intact.
    untrack(() => {
      nodes = nodes.map((node) => ({
        ...node,
        position: {
          x: node.position.x + camera.x / camera.zoom,
          y: node.position.y + camera.y / camera.zoom,
        },
      }));
      viewport = { x: 0, y: 0, zoom: camera.zoom };
      setViewport(viewport);
    });
  });
  setContext<CanvasContext>(canvasContext, {
    get captureId() {
      return captureId;
    },
    get focusObjectId() {
      return focusObjectId;
    },
    get bindings() {
      return bindings;
    },
    get objectId() {
      return objectId;
    },
    get activeRole() {
      return activeRole;
    },
    get selection() {
      return selection;
    },
    get settings() {
      return settings;
    },
    get viewport() {
      return viewport;
    },
    get viewportElement() {
      return viewportElement;
    },
    get viewportWidth() {
      return viewportWidth;
    },
    get viewportHeight() {
      return viewportHeight;
    },
    get grids() {
      return grids;
    },
    get owners() {
      return owners;
    },
    onactivate: (object, role) => onactivate(object, role),
    onselect: (object, role, cell) => onselect(object, role, cell),
    onmapping: (object, role, mapping) => onmapping(object, role, mapping),
  });
  const ownerFilter = $derived(
    !!selection &&
      [
        selection.thread,
        selection.value,
        selection.warp,
        selection.warpgroup,
      ].some((x) => x != null),
  );
  const edges = $derived.by(() => {
    if (!showLinks) return [];
    const result: Edge[] = [];
    const viewIds = new Set(
      objects.flatMap((object) =>
        object.views.map((view) => `${object.id}:${view.role}`),
      ),
    );
    const linked = new Set(
      Object.entries(mappings)
        .filter(
          ([key, m]) => viewIds.has(key) && m.cells.some((c) => c.selected),
        )
        .map(([key]) => key),
    );
    const source = ownerFilter
      ? "owners"
      : `${selection?.object_id ?? objectId}:${selection?.role ?? activeRole}`;
    if (selection && (ownerFilter ? showOwners : viewIds.has(source)))
      for (const target of linked) {
        if (target !== source)
          result.push({
            id: `selected:${source}:${target}`,
            source,
            target,
            type: "smoothstep",
            style: "stroke: #c75322; stroke-width: 3;",
            label: ownerFilter ? "thread / value" : "linked cells",
            zIndex: 2,
            selectable: false,
          });
      }
    for (const obj of objects)
      for (const relation of obj.relationships) {
        if (relation.kind !== "derived_parent") continue;
        const parent = objects.find((o) => o.id === relation.target);
        if (obj.views[0] && parent?.views[0])
          result.push({
            id: `backing:${obj.id}:${parent.id}`,
            source: `${obj.id}:${obj.views[0].role}`,
            target: `${parent.id}:${parent.views[0].role}`,
            style: "stroke: #4e8eaa; stroke-dasharray: 5 5;",
            label: "backing tensor",
            selectable: false,
          });
      }
    // These edges indicate an explicit observed-operand relationship, not inferred addresses.
    for (const obj of objects)
      for (const view of obj.views) {
        const observed = objects.find((o) => o.id === view.observed_object);
        if (!observed?.views.length) continue;
        const source = `${obj.id}:${view.role}`,
          target = `${observed.id}:${observed.views[0].role}`;
        if (!result.some((e) => e.source === source && e.target === target))
          result.push({
            id: `observed:${source}:${target}`,
            source,
            target,
            style: "stroke: #8daba3; stroke-dasharray: 5 5;",
            label: "observed operand",
            selectable: false,
          });
      }
    return result;
  });
  let pointerStart = { x: 0, y: 0 };
  function pickFallback(event: MouseEvent) {
    if (
      Math.hypot(
        event.clientX - pointerStart.x,
        event.clientY - pointerStart.y,
      ) > 5
    )
      return;
    const target = event.target as Element;
    if (
      target.closest(
        'button, input, select, [role="gridcell"], .tensor-resize-handle, .svelte-flow__minimap',
      )
    )
      return;
    // Native SVG hit testing can reject visible cells at extreme zoom. Pick
    // against the viewport geometry and the already captured mapping instead.
    for (const node of [...nodes]
      .reverse()
      .sort((a, b) => Number(!!b.selected) - Number(!!a.selected))) {
      const grid = grids[node.id];
      const cell = grid?.hit(event.clientX, event.clientY);
      if (cell) {
        grid.select(cell);
        break;
      }
    }
  }

  let pendingReveal = $state<{ id: string; coordinate: number[] } | null>(null);
  export function reveal(object: string, role: string, coordinate: number[]) {
    const id = `${object}:${role}`;
    if (grids[id]) grids[id].reveal(coordinate);
    else {
      pendingReveal = { id, coordinate };
      fitView({
        nodes: [{ id }],
        minZoom: 0.000001,
        maxZoom: 1,
        padding: 0.08,
      });
    }
  }
  $effect(() => {
    const target = pendingReveal;
    if (target && grids[target.id]) {
      grids[target.id].reveal(target.coordinate);
      pendingReveal = null;
    }
  });

  function focusSelected() {
    const focused = nodes.filter(
      (node) =>
        node.data.object &&
        (node.data.object as CapturedObject).id === objectId,
    );
    // Keep the ownership panel beside the focused group so its controls stay usable.
    if (showOwners && focused.length) {
      const left = Math.min(...focused.map((n) => n.position.x)),
        top = Math.min(...focused.map((n) => n.position.y));
      nodes = nodes.map((n) =>
        n.id === "owners" ? { ...n, position: { x: left - 340, y: top } } : n,
      );
    }
    fitView({
      nodes: [...focused, ...(showOwners ? [{ id: "owners" }] : [])],
      minZoom: 0.000001,
      maxZoom: 1,
      padding: 0.08,
      duration: 150,
    });
  }
  $effect(() => {
    const hidden = !showOwners;
    untrack(() => {
      nodes = nodes.map((n) => (n.id === "owners" ? { ...n, hidden } : n));
    });
  });
  function arrange() {
    nodes = layout(nodes);
    fitView({ minZoom: 0.000001, maxZoom: 1, padding: 0.08 });
  }
</script>

<svelte:window
  onkeydown={(event) => {
    if (event.key === "Escape") expanded = false;
  }}
/>
<div
  class="tensor-canvas"
  class:canvas-expanded={expanded}
  style:--flow-x={viewport.x}
  style:--flow-y={viewport.y}
  style:--flow-zoom={viewport.zoom}
>
  <div class="canvas-toolbar">
    <div>
      <strong>Tensor canvas</strong><span
        >{scopeLabel} and related objects · {nodes.length - 1} views</span
      >
    </div>
    <div class="canvas-actions">
      <button aria-label="Zoom out" onclick={() => setZoom(getZoom() / 1.2)}
        >−</button
      ><output aria-label="Canvas zoom"
        >{Number((viewport.zoom * 100).toPrecision(3))}%</output
      ><button aria-label="Zoom in" onclick={() => setZoom(getZoom() * 1.2)}
        >+</button
      >
      <button
        onclick={() =>
          fitView({ minZoom: 0.000001, maxZoom: 1, padding: 0.08 })}
        >Fit all</button
      >
      <button onclick={focusSelected}>Focus selected</button><button
        onclick={arrange}>Arrange</button
      >
      <button
        aria-pressed={showOwners}
        onclick={() => (showOwners = !showOwners)}>Thread / value</button
      >
      <button aria-pressed={showLinks} onclick={() => (showLinks = !showLinks)}
        >Links</button
      >
      <button onclick={() => (expanded = !expanded)}
        >{expanded ? "Exit expanded canvas" : "Expand canvas"}</button
      >
    </div>
  </div>
  <form
    class="canvas-navigation"
    onsubmit={(event) => {
      event.preventDefault();
      jumpToCell();
    }}
  >
    <span>Go to cell · {activeView?.role ?? "tensor"}</span>
    <label
      >Row <input
        aria-label="Go to row"
        type="number"
        min="0"
        max={activeRows - 1}
        bind:value={jumpRow}
      /></label
    >
    <label
      >Column <input
        aria-label="Go to column"
        type="number"
        min="0"
        max={activeColumns - 1}
        bind:value={jumpColumn}
      /></label
    >
    <button type="submit" disabled={!grids[activeId]}>Go to cell</button>
    <span
      >Full {activeRows.toLocaleString()} × {activeColumns.toLocaleString()} slice
      · Drag a card’s bottom-right corner to resize</span
    >
  </form>
  <div
    class="canvas-viewport"
    role="region"
    aria-label="Tensor canvas"
    onpointerdowncapture={(event) => {
      pointerStart = { x: event.clientX, y: event.clientY };
    }}
    onclickcapture={pickFallback}
    bind:this={viewportElement}
    bind:clientWidth={viewportWidth}
    bind:clientHeight={viewportHeight}
  >
    <SvelteFlow
      bind:nodes
      {edges}
      {nodeTypes}
      bind:viewport
      minZoom={0.000001}
      maxZoom={100_000_000}
      onlyRenderVisibleElements
      nodesConnectable={false}
      deleteKey={null}
      zoomOnDoubleClick={false}
      panOnScroll
      zoomOnScroll={false}
      oninit={() => {
        tick().then(() =>
          fitView({ minZoom: 0.000001, maxZoom: 1, padding: 0.08 }),
        );
      }}
    >
      <Background gap={24} />
      <MiniMap
        position="bottom-left"
        pannable
        zoomable
        nodeColor={(node) =>
          node.type === "ownership" ? "#e6af7b" : "#9cc7bc"}
      />
    </SvelteFlow>
  </div>
  <div class="canvas-footer">
    <span
      >Drag background to pan · Ctrl/⌘ + scroll to zoom · Drag titles to arrange</span
    ><span>Full tensor grids · Orange = linked cells · Links are optional</span>
  </div>
</div>
