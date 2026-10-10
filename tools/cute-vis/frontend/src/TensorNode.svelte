<script lang="ts">
  import { getContext } from "svelte";
  import {
    Handle,
    Position,
    NodeResizeControl,
    type NodeProps,
    type Node,
  } from "@xyflow/svelte";
  import Grid from "./Grid.svelte";
  import { canvasContext, type CanvasContext } from "./canvas";
  import type { CapturedObject, View } from "./api";
  let {
    id,
    data,
    width = 640,
    height = 460,
    positionAbsoluteX,
    positionAbsoluteY,
  }: NodeProps<Node<{ object: CapturedObject; view: View }>> = $props();
  const context = getContext<CanvasContext>(canvasContext);
</script>

<section
  class="tensor-node"
  style:--handle-scale={1 / context.viewport.zoom}
  style:border-color={context.viewport.zoom > 1 ? "transparent" : undefined}
  style:outline={context.viewport.zoom > 1
    ? `${2 / context.viewport.zoom}px solid #b5cfc1`
    : undefined}
  style:box-shadow={context.viewport.zoom > 1 ? "none" : undefined}
  class:active-node={data.object.id === context.objectId &&
    data.view.role === context.activeRole}
  data-object-id={data.object.id}
  data-role={data.view.role}
  aria-label={`${data.object.label} · ${data.view.role}`}
>
  <Handle type="target" position={Position.Left} isConnectable={false} />
  <NodeResizeControl
    position="bottom-right"
    autoScale={false}
    minWidth={440}
    minHeight={420}
    maxWidth={1_000_000}
    maxHeight={1_000_000}
    class="tensor-resize-handle"
    aria-label={`Resize ${data.object.label} ${data.view.role}`}
    title="Drag to resize this tensor"
    ><span aria-hidden="true">◢</span></NodeResizeControl
  >
  <button
    class="node-title"
    aria-label={`Focus ${data.object.label} ${data.view.role}; drag to move`}
    onclick={() => context.onactivate(data.object, data.view.role)}
  >
    <span class={`kind ${data.object.kind}`}>{data.object.kind}</span><strong
      >{data.object.label}{data.view.role !== "tensor"
        ? ` / ${data.view.role}`
        : ""}</strong
    >
    <span class="node-line"
      >{data.object.source ? `line ${data.object.source.line}` : "No source"} · ⠿</span
    >
  </button>
  <div class="nodrag tensor-content">
    <Grid
      {id}
      {width}
      {height}
      nodeX={positionAbsoluteX}
      nodeY={positionAbsoluteY}
      captureId={context.captureId}
      objectId={data.object.id}
      view={data.view}
      selection={context.selection}
      settings={context.settings[id]}
      onsettings={(value) => (context.settings[id] = value)}
      onselect={(role, cell) => context.onselect(data.object, role, cell)}
      onmapping={(role, mapping) =>
        context.onmapping(data.object, role, mapping)}
    />
  </div>
  <Handle type="source" position={Position.Right} isConnectable={false} />
</section>
