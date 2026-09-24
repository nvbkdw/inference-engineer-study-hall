<script lang="ts">
  import { getContext } from "svelte";
  import { Handle, Position, type NodeProps } from "@xyflow/svelte";
  let { id }: NodeProps = $props();
  import { canvasContext, type CanvasContext } from "./canvas";
  const context = getContext<CanvasContext>(canvasContext);
</script>

<div class="ownership-node" data-node-id={id}>
  <div class="node-title ownership-title">
    Thread / value selection <span>⠿</span>
  </div>
  <div class="nodrag nopan nowheel">{@render context.owners()}</div>
  <Handle type="source" position={Position.Right} isConnectable={false} />
</div>
