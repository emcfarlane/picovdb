// PicoVDB file format

// "PicoVDB0" in hex little endian  
export const PICOVDB_MAGIC = [0x6f636950, 0x30424456];

// Grid type constants
export const GRID_TYPE_SDF_FLOAT = 1;
export const GRID_TYPE_SDF_UINT8 = 2;
export const GRID_TYPE_FOG_FLOAT = 3;

export const PICOVDB_FILE_HEADER_SIZE = 32;
export const PICOVDB_GRID_SIZE = 64;
export const PICOVDB_ROOT_SIZE = 8;
export const PICOVDB_NODE_MASK_SIZE = 12;
export const PICOVDB_LEAF_MASK_SIZE = 8;
export const PICOVDB_UPPER_SIZE = 12304;
export const PICOVDB_LOWER_SIZE = 1552;
export const PICOVDB_LEAF_SIZE = 208;
export const PICOVDB_DATA_SIZE = 16;

export interface PicoVDBFileHeader {
  magic: [number, number];
  version: number;
  gridCount: number;
  upperCount: number;
  lowerCount: number;
  leafCount: number;
  dataCount: number;
}

export interface PicoVDBGrid {
  gridIndex: number;
  upperStart: number;
  lowerStart: number;
  leafStart: number;
  dataStart: number;
  dataElemCount: number;
  gridType: number;
  indexBoundsMin: Int32Array, // 3 elements (min)
  indexBoundsMax: Int32Array, // 6 elements (max)
}

export interface PicoVDBRoot {
  key: [number, number]; // 64-bit coordinate key (8 bytes)
}

// Node element encoding (upper/lower levels):
//   state=0,value=0 -> outside implicit    state=0,value=1 -> stored value
//   state=1,value=0 -> inside implicit     state=1,value=1 -> child reference
// Leaf-level encoding:
//   state=0,value=0 -> outside implicit    state=0,value=1 -> narrow-band (non-surface)
//   state=1,value=0 -> inside implicit     state=1,value=1 -> surface/cross-over voxel
export interface PicoVDBNodeElement {
  stateMask: number;
  valueMask: number;
  packedLocalIndex: number; // (localState_u16 << 16) | localValue_u16
}

export interface PicoVDBUpper {
  baseInsideIndex: number;
  baseActiveIndex: number;
  elements: PicoVDBNodeElement[]; // 1024 elements
}

export interface PicoVDBLower {
  baseInsideIndex: number;
  baseActiveIndex: number;
  elements: PicoVDBNodeElement[]; // 128 elements
}

export interface PicoVDBLeaf {
  baseInsideIndex: number;
  baseActiveIndex: number;
  elements: PicoVDBNodeElement[]; // 16 elements
}

export class PicoVDBFile {
  private buffer: ArrayBuffer;
  private view: DataView;

  // Header
  header: PicoVDBFileHeader;

  // Slices (as Uint8Arrays for WebGPU - explicitly typed for ArrayBuffer, not SharedArrayBuffer)
  gridsBuffer: Uint8Array<ArrayBuffer>;
  rootsBuffer: Uint8Array<ArrayBuffer>;
  uppersBuffer: Uint8Array<ArrayBuffer>;
  lowersBuffer: Uint8Array<ArrayBuffer>;
  leavesBuffer: Uint8Array<ArrayBuffer>;
  dataBuffer: Uint8Array<ArrayBuffer>;

  constructor(buffer: ArrayBuffer) {
    this.buffer = buffer;
    this.view = new DataView(buffer);

    let offset = 0;

    // Parse header
    this.header = {
      magic: [this.view.getUint32(offset + 0, true), this.view.getUint32(offset + 4, true)],
      version: this.view.getUint32(offset + 8, true),
      gridCount: this.view.getUint32(offset + 12, true),
      upperCount: this.view.getUint32(offset + 16, true),
      lowerCount: this.view.getUint32(offset + 20, true),
      leafCount: this.view.getUint32(offset + 24, true),
      dataCount: this.view.getUint32(offset + 28, true),
    };
    offset += PICOVDB_FILE_HEADER_SIZE;

    // Validate magic
    if (this.header.magic[0] !== PICOVDB_MAGIC[0] || this.header.magic[1] !== PICOVDB_MAGIC[1]) {
      throw new Error(`Invalid PicoVDB magic: [0x${this.header.magic[0].toString(16)}, 0x${this.header.magic[1].toString(16)}]`);
    }

    // Create buffer slices for WebGPU
    this.gridsBuffer = new Uint8Array(buffer, offset, this.header.gridCount * PICOVDB_GRID_SIZE);
    offset += this.header.gridCount * PICOVDB_GRID_SIZE;

    const rootCount = this.getRootCountPadded()
    this.rootsBuffer = new Uint8Array(buffer, offset, rootCount * PICOVDB_ROOT_SIZE);
    offset += rootCount * PICOVDB_ROOT_SIZE;

    this.uppersBuffer = new Uint8Array(buffer, offset, this.header.upperCount * PICOVDB_UPPER_SIZE);
    offset += this.header.upperCount * PICOVDB_UPPER_SIZE;

    this.lowersBuffer = new Uint8Array(buffer, offset, this.header.lowerCount * PICOVDB_LOWER_SIZE);
    offset += this.header.lowerCount * PICOVDB_LOWER_SIZE;

    this.leavesBuffer = new Uint8Array(buffer, offset, this.header.leafCount * PICOVDB_LEAF_SIZE);
    offset += this.header.leafCount * PICOVDB_LEAF_SIZE;

    this.dataBuffer = new Uint8Array(buffer, offset, this.header.dataCount * PICOVDB_DATA_SIZE);
    offset += this.header.dataCount * PICOVDB_DATA_SIZE;
  }

  getSize(): number {
    return this.buffer.byteLength;
  }

  getGrid(index: number): PicoVDBGrid {
    if (index >= this.header.gridCount) {
      throw new Error(`Grid index ${index} out of bounds (max: ${this.header.gridCount - 1})`);
    }

    const baseOffset = PICOVDB_FILE_HEADER_SIZE + index * PICOVDB_GRID_SIZE;
    const offset = baseOffset;

    return {
      gridIndex: this.view.getUint32(offset + 0, true),
      upperStart: this.view.getUint32(offset + 4, true),
      lowerStart: this.view.getUint32(offset + 8, true),
      leafStart: this.view.getUint32(offset + 12, true),
      dataStart: this.view.getUint32(offset + 16, true),
      dataElemCount: this.view.getUint32(offset + 20, true),
      gridType: this.view.getUint32(offset + 24, true),
      indexBoundsMin: new Int32Array(this.buffer, offset + 32, 3),
      indexBoundsMax: new Int32Array(this.buffer, offset + 48, 3),
    };
  }

  /** Node and value ranges of one grid. Node indices inside a grid's nodes are relative to these starts. */
  getGridRange(index: number): { upperStart: number; upperCount: number; lowerStart: number; lowerCount: number; leafStart: number; leafCount: number; dataStart: number; dataElemCount: number } {
    const grid = this.getGrid(index);
    const next = index + 1 < this.header.gridCount ? this.getGrid(index + 1) : null;
    return {
      upperStart: grid.upperStart,
      upperCount: (next ? next.upperStart : this.header.upperCount) - grid.upperStart,
      lowerStart: grid.lowerStart,
      lowerCount: (next ? next.lowerStart : this.header.lowerCount) - grid.lowerStart,
      leafStart: grid.leafStart,
      leafCount: (next ? next.leafStart : this.header.leafCount) - grid.leafStart,
      dataStart: grid.dataStart,
      dataElemCount: grid.dataElemCount,
    };
  }

  getRootCountPadded(): number {
    return ((this.header.upperCount + 1) / 2 | 0) * 2 // Padding to even number
  }

  getRoot(index: number): PicoVDBRoot {
    if (index >= this.header.upperCount) {
      throw new Error(`Root index ${index} out of bounds (max: ${this.header.upperCount - 1})`);
    }
    const baseOffset = PICOVDB_FILE_HEADER_SIZE +
      this.header.gridCount * PICOVDB_GRID_SIZE +
      index * PICOVDB_ROOT_SIZE;
    return {
      key: [
        this.view.getUint32(baseOffset + 0, true),
        this.view.getUint32(baseOffset + 4, true),
      ],
    };
  }

  getUpper(index: number): PicoVDBUpper {
    if (index >= this.header.upperCount) {
      throw new Error(`Upper index ${index} out of bounds (max: ${this.header.upperCount - 1})`);
    }

    const baseOffset = PICOVDB_FILE_HEADER_SIZE +
      this.header.gridCount * PICOVDB_GRID_SIZE +
      this.getRootCountPadded() * PICOVDB_ROOT_SIZE +
      index * PICOVDB_UPPER_SIZE;

    let offset = baseOffset;
    const baseInsideIndex = this.view.getUint32(offset + 0, true);
    console.assert(this.view.getUint32(offset + 4, true) === 0, 'baseInsideIndex high u32 must be 0');
    const baseActiveIndex = this.view.getUint32(offset + 8, true);
    console.assert(this.view.getUint32(offset + 12, true) === 0, 'baseActiveIndex high u32 must be 0');
    offset += 16;

    const elements: PicoVDBNodeElement[] = [];
    for (let i = 0; i < 1024; i++) {
      const maskOffset = offset + i * PICOVDB_NODE_MASK_SIZE;
      elements.push({
        stateMask: this.view.getUint32(maskOffset + 0, true),
        valueMask: this.view.getUint32(maskOffset + 4, true),
        packedLocalIndex: this.view.getUint32(maskOffset + 8, true),
      });
    }
    return { baseInsideIndex, baseActiveIndex, elements };
  }

  getLower(index: number): PicoVDBLower {
    if (index >= this.header.lowerCount) {
      throw new Error(`Lower index ${index} out of bounds (max: ${this.header.lowerCount - 1})`);
    }

    const baseOffset = PICOVDB_FILE_HEADER_SIZE +
      this.header.gridCount * PICOVDB_GRID_SIZE +
      this.getRootCountPadded() * PICOVDB_ROOT_SIZE +
      this.header.upperCount * PICOVDB_UPPER_SIZE +
      index * PICOVDB_LOWER_SIZE;

    let offset = baseOffset;
    const baseInsideIndex = this.view.getUint32(offset + 0, true);
    console.assert(this.view.getUint32(offset + 4, true) === 0, 'baseInsideIndex high u32 must be 0');
    const baseActiveIndex = this.view.getUint32(offset + 8, true);
    console.assert(this.view.getUint32(offset + 12, true) === 0, 'baseActiveIndex high u32 must be 0');
    offset += 16;

    const elements: PicoVDBNodeElement[] = [];
    for (let i = 0; i < 128; i++) {
      const maskOffset = offset + i * PICOVDB_NODE_MASK_SIZE;
      elements.push({
        stateMask: this.view.getUint32(maskOffset + 0, true),
        valueMask: this.view.getUint32(maskOffset + 4, true),
        packedLocalIndex: this.view.getUint32(maskOffset + 8, true),
      });
    }
    return { baseInsideIndex, baseActiveIndex, elements };
  }

  getLeaf(index: number): PicoVDBLeaf {
    if (index >= this.header.leafCount) {
      throw new Error(`Leaf index ${index} out of bounds (max: ${this.header.leafCount - 1})`);
    }

    const baseOffset = PICOVDB_FILE_HEADER_SIZE +
      this.header.gridCount * PICOVDB_GRID_SIZE +
      this.getRootCountPadded() * PICOVDB_ROOT_SIZE +
      this.header.upperCount * PICOVDB_UPPER_SIZE +
      this.header.lowerCount * PICOVDB_LOWER_SIZE +
      index * PICOVDB_LEAF_SIZE;

    let offset = baseOffset;
    const baseInsideIndex = this.view.getUint32(offset + 0, true);
    console.assert(this.view.getUint32(offset + 4, true) === 0, 'baseInsideIndex high u32 must be 0');
    const baseActiveIndex = this.view.getUint32(offset + 8, true);
    console.assert(this.view.getUint32(offset + 12, true) === 0, 'baseActiveIndex high u32 must be 0');
    offset += 16;

    const elements: PicoVDBNodeElement[] = [];
    for (let i = 0; i < 16; i++) {
      const maskOffset = offset + i * PICOVDB_NODE_MASK_SIZE;
      elements.push({
        stateMask: this.view.getUint32(maskOffset + 0, true),
        valueMask: this.view.getUint32(maskOffset + 4, true),
        packedLocalIndex: this.view.getUint32(maskOffset + 8, true),
      });
    }
    return { baseInsideIndex, baseActiveIndex, elements };
  }

  getVoxelCount(): number {
    let count = 0
    for (let i = 0; i < this.header.gridCount; i++) {
      count += this.getGrid(i).dataElemCount - 2 // Minus background values
    }
    return count
  }

  // TODO: this needs to use the dataStart to first slice the dataBuffer in 16 byte chunks
  // then capture the value with the dataElemCount.
  //getGridFloat(grid: PicoVDBGrid, index: number): number {
  //  const dataPtr = new Float32Array(this.dataBuffer.buffer, this.dataBuffer.byteOffset);
  //  return dataPtr[grid.dataIndex / 4 + index]; // dataIndex is in bytes, convert to float index
  //}
}

export async function fetchPicoVDB(
  url: string,
  options: RequestInit = {}
): Promise<PicoVDBFile> {
  const response = await fetch(url, {
    ...options,
    headers: {
      'Accept': 'application/octet-stream',
      ...options.headers,
    },
  });

  if (!response.ok) {
    throw new Error(`Failed to load PicoVDB: ${response.status} ${response.statusText}`);
  }

  // We check the Content-Type, Content-Encoding, and the URL extension as a fallback
  const contentType = response.headers.get('Content-Type') ?? '';
  const contentEncoding = response.headers.get('Content-Encoding') ?? '';
  const isGzipped =
    url.endsWith('.gz') ||
    contentType.includes('gzip') ||
    contentEncoding.includes('gzip');

  let buffer: ArrayBuffer;

  // If the browser already handled decompression (via Content-Encoding),
  // we don't need to do it manually.
  if (isGzipped && contentEncoding !== 'gzip') {
    if (typeof DecompressionStream === 'undefined') {
      throw new Error('Gzip decompression requires DecompressionStream API support.');
    }

    const decompressionStream = new DecompressionStream('gzip');
    const stream = response.body!.pipeThrough(decompressionStream);
    buffer = await new Response(stream).arrayBuffer();
  } else {
    buffer = await response.arrayBuffer();
  }

  // Ensure 4-byte alignment.
  const remainder = buffer.byteLength % 4;
  const alignedBuffer = remainder === 0
    ? buffer
    : new ArrayBuffer(buffer.byteLength + (4 - remainder));

  if (remainder !== 0) {
    new Uint8Array(alignedBuffer).set(new Uint8Array(buffer));
  }
  return new PicoVDBFile(alignedBuffer);
}
