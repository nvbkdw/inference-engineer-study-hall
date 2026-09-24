import type { GridSettings, View } from "./api";

export const CELL_WIDTH = 78;
export const CELL_HEIGHT = 40;
export const GRID_LEFT = 38;
export const GRID_TOP = 28;
export const MAX_DETAIL_CELLS = 16_384;
export const MAX_CARD_SIZE = 1_000_000;

export function dimensions(
  tree: unknown,
  path = "",
): { path: string; size: number | null }[] {
  return Array.isArray(tree)
    ? tree.flatMap((child, i) =>
        dimensions(child, path ? `${path}.${i}` : `${i}`),
      )
    : [{ path: path || "0", size: typeof tree === "number" ? tree : null }];
}

export function naturalCardSize(view: View, settings?: GridSettings) {
  const dims = dimensions(view.shape);
  const rows = dims[settings?.row ?? 0]?.size ?? 1;
  const column = settings?.column ?? (dims.length > 1 ? 1 : -1);
  const columns = column < 0 ? 1 : (dims[column]?.size ?? 1);
  // Keep CSS boxes within browser layout limits; the full logical extent is
  // preserved by the SVG viewBox even for billion-by-billion tensors.
  return {
    width: Math.min(MAX_CARD_SIZE, Math.max(640, columns * CELL_WIDTH + 70)),
    height: Math.min(MAX_CARD_SIZE, Math.max(460, rows * CELL_HEIGHT + 360)),
  };
}

export type Window = { start: number[]; extent: number[] };

/** Map the visible intersection of the SVG and canvas into logical indices. */
export function visibleWindow(
  grid: DOMRect,
  canvas: DOMRect,
  rows: number,
  columns: number,
): Window | null {
  const left = Math.max(grid.left, canvas.left),
    right = Math.min(grid.right, canvas.right);
  const top = Math.max(grid.top, canvas.top),
    bottom = Math.min(grid.bottom, canvas.bottom);
  if (right <= left || bottom <= top || !grid.width || !grid.height)
    return null;
  const toColumn = (x: number) =>
    (((x - grid.left) / grid.width) * (columns * CELL_WIDTH + GRID_LEFT) -
      GRID_LEFT) /
    CELL_WIDTH;
  const toRow = (y: number) =>
    (((y - grid.top) / grid.height) * (rows * CELL_HEIGHT + GRID_TOP) -
      GRID_TOP) /
    CELL_HEIGHT;
  const x = Math.max(0, Math.floor(toColumn(left)) - 1);
  const y = Math.max(0, Math.floor(toRow(top)) - 1);
  const endX = Math.min(columns, Math.ceil(toColumn(right)) + 1);
  const endY = Math.min(rows, Math.ceil(toRow(bottom)) + 1);
  if (x >= endX || y >= endY) return null;
  return { start: [y, x], extent: [endY - y, endX - x] };
}

/** Use the largest API pages that cover this window (at most 4096 cells each). */
export function mappingPages(window: Window): Window[] {
  const pages: Window[] = [];
  const height = Math.min(64, window.extent[0]);
  const width = Math.min(Math.floor(4096 / height), window.extent[1]);
  for (let y = 0; y < window.extent[0]; y += height)
    for (let x = 0; x < window.extent[1]; x += width)
      pages.push({
        start: [window.start[0] + y, window.start[1] + x],
        extent: [
          Math.min(height, window.extent[0] - y),
          Math.min(width, window.extent[1] - x),
        ],
      });
  return pages;
}
