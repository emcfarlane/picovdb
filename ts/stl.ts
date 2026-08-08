// STL -> PicoVDB import, backed by the Zig wasm module (src/c_api.zig).
//
//   import { importSTL } from './stl.ts';
//   const { file, bytes, stats } = await importSTL(stlBytes, {
//     voxelsPerUnit: 8,
//     maxVoxels: 100e6, // fail fast instead of ~1 GiB+ peak memory
//   });
//
// The wasm binary comes from `zig build wasm` (zig-out/wasm/picovdb.wasm);
// bundler users should call initSTL({ wasmURL }) or initSTL({ wasmBinary }).
// Voxelization is CPU-bound (roughly 1 s per ~7 M active voxels) — call from a
// Web Worker for large inputs. Each call instantiates a fresh wasm instance so
// linear memory (which never shrinks) is released afterwards.

import { PicoVDBFile } from './picovdb.ts';

export const LEVEL_SET_HALF_WIDTH = 3.0;

export interface STLInfo {
  triangleCount: number;
  bboxMin: [number, number, number];
  bboxMax: [number, number, number];
}

export interface STLImportOptions {
  /** Grid resolution in voxels per world unit. Required, > 0. */
  voxelsPerUnit: number;
  /** Narrow band half-width in voxels (default 3). */
  halfWidth?: number;
  valueType?: 'f32' | 'u8';
  /** Rotations in degrees, applied about x, then y, then z. */
  rotateDeg?: [number, number, number];
  /**
   * Fail if the voxel estimate (mesh bbox dilated by the narrow band) exceeds
   * this, instead of voxelizing; peak memory is ~8 bytes per estimated voxel.
   * Default: unlimited.
   */
  maxVoxels?: number;
}

export interface STLImportStats {
  activeVoxels: number;
  surfaceVoxels: number;
  leafCount: number;
  lowerCount: number;
  upperCount: number;
  /** Index-space bounds of active voxels. */
  bboxMin: [number, number, number];
  bboxMax: [number, number, number];
  /** Post-rotation mesh bounds in world units. */
  worldMin: [number, number, number];
  worldMax: [number, number, number];
}

export interface STLImportResult {
  file: PicoVDBFile;
  /** The encoded .pvdb, ready for WebGPU upload or saving. */
  bytes: Uint8Array<ArrayBuffer>;
  stats: STLImportStats;
}

interface StlWasmExports {
  memory: WebAssembly.Memory;
  picovdb_abi_version(): number;
  picovdb_alloc(len: number): number;
  picovdb_stl_info(stl: number, len: number, out: number): number;
  picovdb_stl_to_grid(stl: number, len: number, opts: number, out: number): number;
  picovdb_error_string(code: number): number;
}

let modulePromise: Promise<WebAssembly.Module> | null = null;

/**
 * Compile the STL importer wasm. Optional: the other entry points call this
 * lazily with the default URL. Bundlers that don't resolve
 * `new URL(..., import.meta.url)` should pass wasmURL or wasmBinary.
 */
export function initSTL(
  options: { wasmURL?: string | URL; wasmBinary?: BufferSource } = {}
): Promise<WebAssembly.Module> {
  if (options.wasmBinary) {
    modulePromise = WebAssembly.compile(options.wasmBinary);
  } else if (options.wasmURL) {
    modulePromise = compileFromURL(options.wasmURL);
  } else {
    modulePromise ??= compileFromURL(new URL('../zig-out/wasm/picovdb.wasm', import.meta.url));
  }
  return modulePromise;
}

async function compileFromURL(url: string | URL): Promise<WebAssembly.Module> {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to load STL importer wasm from ${url}: ${response.status}`);
  }
  // compileStreaming requires the application/wasm MIME type; fall back for
  // servers that don't send it.
  if ('compileStreaming' in WebAssembly && response.headers.get('Content-Type')?.includes('application/wasm')) {
    return WebAssembly.compileStreaming(response);
  }
  return WebAssembly.compile(await response.arrayBuffer());
}

async function instantiate(): Promise<StlWasmExports> {
  const module = await initSTL();
  const instance = await WebAssembly.instantiate(module, {});
  const ex = instance.exports as unknown as StlWasmExports;
  const abi = ex.picovdb_abi_version();
  if (abi !== 1) throw new Error(`STL importer ABI version ${abi}, expected 1`);
  return ex;
}

function stage(ex: StlWasmExports, bytes: Uint8Array): number {
  const ptr = ex.picovdb_alloc(bytes.length);
  if (!ptr) throw new Error('wasm allocation failed');
  new Uint8Array(ex.memory.buffer).set(bytes, ptr);
  return ptr;
}

function throwPicoVDBError(ex: StlWasmExports, rc: number): never {
  const mem = new Uint8Array(ex.memory.buffer, ex.picovdb_error_string(rc));
  let end = 0;
  while (mem[end] !== 0) end++;
  throw new Error(`STL import failed (${rc}): ${new TextDecoder().decode(mem.subarray(0, end))}`);
}

function readInfo(view: DataView, ptr: number): STLInfo {
  const f3 = (o: number): [number, number, number] => [
    view.getFloat32(o, true),
    view.getFloat32(o + 4, true),
    view.getFloat32(o + 8, true),
  ];
  return {
    triangleCount: view.getUint32(ptr, true),
    bboxMin: f3(ptr + 4),
    bboxMax: f3(ptr + 16),
  };
}

/** Triangle count and world bounds, without voxelizing. */
export async function stlInfo(stl: Uint8Array): Promise<STLInfo> {
  const ex = await instantiate();
  const infoPtr = ex.picovdb_alloc(28);
  const rc = ex.picovdb_stl_info(stage(ex, stl), stl.length, infoPtr);
  if (rc !== 0) throwPicoVDBError(ex, rc);
  return readInfo(new DataView(ex.memory.buffer), infoPtr);
}

/** Convert an STL (binary or ASCII) to a PicoVDB narrow-band SDF grid. */
export async function importSTL(stl: Uint8Array, options: STLImportOptions): Promise<STLImportResult> {
  const { voxelsPerUnit } = options;
  if (!(voxelsPerUnit > 0)) throw new Error(`voxelsPerUnit must be > 0, got ${voxelsPerUnit}`);
  const halfWidth = options.halfWidth ?? LEVEL_SET_HALF_WIDTH;
  const rotateDeg = options.rotateDeg ?? [0, 0, 0];

  const ex = await instantiate();
  const stlPtr = stage(ex, stl);

  // picovdb_mesh_to_grid_options { max_voxels u64, voxels_per_unit, half_width, value_type, rotate_deg[3] }
  const optsPtr = ex.picovdb_alloc(32);
  {
    const view = new DataView(ex.memory.buffer);
    const maxVoxels = options.maxVoxels ?? 0;
    view.setBigUint64(optsPtr + 0, BigInt(Number.isFinite(maxVoxels) ? Math.floor(maxVoxels) : 0), true);
    view.setFloat32(optsPtr + 8, voxelsPerUnit, true);
    view.setFloat32(optsPtr + 12, halfWidth, true);
    view.setUint32(optsPtr + 16, options.valueType === 'u8' ? 1 : 0, true);
    for (let axis = 0; axis < 3; axis++) view.setFloat32(optsPtr + 20 + axis * 4, rotateDeg[axis], true);
  }

  const outPtr = ex.picovdb_alloc(88); // picovdb_buffer, wasm32 layout
  const rc = ex.picovdb_stl_to_grid(stlPtr, stl.length, optsPtr, outPtr);
  if (rc !== 0) throwPicoVDBError(ex, rc);

  // Read picovdb_buffer { data u32@0, len u32@4, stats@8 } and copy the result out;
  // the instance (and its grown memory) is discarded, so no frees needed.
  const view = new DataView(ex.memory.buffer);
  const dataPtr = view.getUint32(outPtr + 0, true);
  const dataLen = view.getUint32(outPtr + 4, true);
  const i3 = (o: number): [number, number, number] => [
    view.getInt32(o, true),
    view.getInt32(o + 4, true),
    view.getInt32(o + 8, true),
  ];
  const f3 = (o: number): [number, number, number] => [
    view.getFloat32(o, true),
    view.getFloat32(o + 4, true),
    view.getFloat32(o + 8, true),
  ];
  const stats: STLImportStats = {
    activeVoxels: Number(view.getBigUint64(outPtr + 8, true)),
    surfaceVoxels: Number(view.getBigUint64(outPtr + 16, true)),
    leafCount: view.getUint32(outPtr + 24, true),
    lowerCount: view.getUint32(outPtr + 28, true),
    upperCount: view.getUint32(outPtr + 32, true),
    bboxMin: i3(outPtr + 36),
    bboxMax: i3(outPtr + 48),
    worldMin: f3(outPtr + 60),
    worldMax: f3(outPtr + 72),
  };

  const bytes = new Uint8Array(ex.memory.buffer, dataPtr, dataLen).slice();
  return { file: new PicoVDBFile(bytes.buffer), bytes, stats };
}
