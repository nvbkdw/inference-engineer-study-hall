import type { Snippet } from "svelte";
import type {
  CapturedObject,
  Cell,
  GridSettings,
  Mapping,
  Selection,
} from "./api";
export const canvasContext = Symbol("cuteviz-canvas");

/** A selected probe's immediate relationships, including complete operand groups.
 * Keep the neighborhood local: following every ancestor's children would pull
 * unrelated branches of a larger kernel back onto the canvas.
 */
export function relatedObjects(
  objects: CapturedObject[],
  selectedId: string,
): CapturedObject[] {
  const selected = objects.find((object) => object.id === selectedId);
  if (!selected) return [];
  const references = (object: CapturedObject) =>
    new Set([
      ...object.relationships.map((relation) => relation.target),
      ...object.views.flatMap((view) =>
        view.observed_object ? [view.observed_object] : [],
      ),
    ]);
  const ids = new Set([selectedId, ...references(selected)]);
  for (const object of objects) {
    if (!references(object).has(selectedId)) continue;
    ids.add(object.id);
    // Selecting an operand also shows the operation and its sibling operands.
    if (object.kind === "copy" || object.kind === "mma")
      for (const target of references(object)) ids.add(target);
  }
  // Follow backing ancestry toward the original tensor, without pulling in
  // unrelated descendants or other operations that use that allocation.
  const queue = [...ids];
  for (let i = 0; i < queue.length; i++) {
    const object = objects.find((o) => o.id === queue[i]);
    for (const relation of object?.relationships ?? []) {
      if (relation.kind === "derived_parent" && !ids.has(relation.target)) {
        ids.add(relation.target);
        queue.push(relation.target);
      }
    }
  }
  return [
    selected,
    ...objects.filter(
      (object) => object.id !== selectedId && ids.has(object.id),
    ),
  ];
}

export interface CanvasContext {
  captureId: string;
  focusObjectId: string;
  bindings: Record<string, number>;
  objectId: string;
  activeRole: string;
  selection: Selection | null;
  settings: Record<string, GridSettings>;
  viewport: { x: number; y: number; zoom: number };
  viewportElement: HTMLDivElement | undefined;
  viewportWidth: number;
  viewportHeight: number;
  grids: Record<
    string,
    {
      jump: (row: number, column: number) => void;
      reveal: (coordinate: number[]) => void;
      hit: (clientX: number, clientY: number) => Cell | null;
      select: (cell: Cell) => void;
    }
  >;
  owners: Snippet;
  onactivate: (object: CapturedObject, role: string) => void;
  onselect: (object: CapturedObject, role: string, cell: Cell) => void;
  onmapping: (
    object: CapturedObject,
    role: string,
    mapping: Mapping | null,
  ) => void;
}
