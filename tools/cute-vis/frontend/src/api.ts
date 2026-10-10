export type Observation = {
  status: "known" | "symbolic" | "unsupported" | "unavailable";
  value: any;
  reason: string | null;
};
export type Selection = {
  object_id?: string;
  role?: string;
  coordinate?: number[];
  thread?: number;
  value?: number;
  warp?: number;
  warpgroup?: number;
};
export type View = {
  role: string;
  shape: any;
  storage: Observation;
  dtype: Observation;
  element_bits: Observation;
  layout_status: Observation;
  ownership: Observation;
  observed_object: string | null;
  layout: any;
  address_unit: string;
};
export type CapturedObject = {
  id: string;
  label: string;
  kind: string;
  source: { source_id: string; line: number; end_line: number } | null;
  fields: Record<string, Observation>;
  views: View[];
  relationships: {
    target: string;
    kind: string;
    transform: string | null;
    mapping?: {
      operation: string;
      origin: number[];
      matrix: number[][];
      parameters: Record<string, { label: string; coefficients: number[] }>;
    } | null;
    mapping_status?: Observation;
  }[];
  diagnostics: string[];
};
export type Capture = {
  schema_version: string;
  observation: string;
  target: Observation;
  compiler: Observation;
  parameters: Record<string, unknown>;
  objects: CapturedObject[];
  sources: { id: string; path: string; text: string }[];
  diagnostics: string[];
  outcome: string;
};
export type Owner = {
  thread: number;
  value: number;
  warp: number;
  warpgroup: number;
};
export type CoordinateMapping = {
  object_id: string;
  label: string;
  role: string;
  coordinate: number[];
  in_bounds: boolean;
  element_offset: Observation;
  byte_offset: Observation;
  operation: string;
  calculation: string[];
  path: string[];
};
export type Footprint = {
  label: string;
  source_id: string;
  start: number[];
  extent: number[];
  exact: boolean;
};
export type Cell = {
  coordinate: number[];
  x: number;
  y: number;
  owners: Owner[];
  selected: boolean;
  in_focus?: boolean;
  coordinate_mappings?: CoordinateMapping[];
  element_offset: Observation;
  byte_offset: Observation;
  tmem: Observation;
  fragment_index: Observation;
  explanation: string;
  links: {
    role: string;
    coordinate: number[];
    thread: number;
    value: number;
  }[];
  unswizzled_offset: number | null;
  bank?: number;
};
export type Mapping = {
  footprints?: Footprint[];
  mapping_diagnostics?: string[];
  role: string;
  cells: Cell[];
  axes: number[];
  start: number[];
  extent: number[];
  total_elements: number;
  dimensions: { path: string; size: number }[];
};

export const show = (field?: Observation) =>
  !field
    ? "Unavailable"
    : field.status === "known"
      ? typeof field.value === "object"
        ? JSON.stringify(field.value)
        : String(field.value)
      : `${field.status}: ${field.reason}`;
let token = "";
export function setToken(value: string) {
  token = value;
}
export type CompileInput = {
  source: string;
  entry: string;
  args_factory: string;
  target: string;
};
export type Job = {
  id: string;
  status: string;
  log: string;
  error: string;
  started: number;
  finished: number | null;
  entry: string;
  target: string;
  capture_available: boolean;
  outcome: string | null;
  object_count: number;
  capture_path: string | null;
};
export type WorkbenchInfo = {
  enabled: boolean;
  can_compile: boolean;
  compiler: string | null;
  reason: string;
  python: string;
  workdir: string;
  output_dir: string;
  timeout_seconds: number;
  token: string;
  has_capture: boolean;
  templates: { id: string; source: string }[];
};
export async function api<T>(
  path: string,
  body?: unknown,
  signal?: AbortSignal,
): Promise<T> {
  const result = await fetch(path, {
    method: body ? "POST" : "GET",
    headers: {
      "Content-Type": "application/json",
      ...(token ? { "X-CuteViz-Token": token } : {}),
    },
    body: body ? JSON.stringify(body) : undefined,
    signal,
  });
  if (!result.ok) {
    const error = await result.json();
    throw new Error(
      typeof error.detail === "string"
        ? error.detail
        : JSON.stringify(error.detail),
    );
  }
  return result.json();
}

export type GridSettings = {
  row: number;
  column: number;
  fixed: Record<number, number>;
  color: string;
  before: boolean;
  bankBase: number;
};
