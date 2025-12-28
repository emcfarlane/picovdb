// PicoVDB file format

// "PicoVDB0" in hex little endian  
export const PICOVDB_MAGIC = [0x6f636950, 0x30424456];

// Grid type constants
export const GRID_TYPE_SDF_FLOAT = 1;
export const GRID_TYPE_SDF_UINT8 = 2;

export const PICOVDB_FILE_HEADER_SIZE = 48;
export const PICOVDB_GRID_SIZE = 160;
export const PICOVDB_ROOT_SIZE = 16;
export const PICOVDB_NODE_MASK_SIZE = 20;
export const PICOVDB_LEAF_MASK_SIZE = 12;
export const PICOVDB_UPPER_SIZE = 20512;
export const PICOVDB_LOWER_SIZE = 2592;
export const PICOVDB_LEAF_SIZE = 224;

export interface PicoVDBFileHeader {
  magic: [number, number];
  checksum: number;
  version: number;
  gridCount: number;
  rootCount: number;
  upperCount: number;
  lowerCount: number;
  leafCount: number;
  dataCount: number;
  voxelCount: number;
  nameCount: number;
}

export interface PicoVDBGrid {
  matf: Float32Array; // 9 elements
  matfi: Float32Array; // 9 elements
  worldBBox: Float32Array; // 6 elements
  indexBBox: Int32Array; // 6 elements
  gridIndex: number;
  rootIndex: number;
  upperIndex: number;
  lowerIndex: number;
  leafIndex: number;
  dataIndex: number;
  gridType: number;
  nameIndex: number;
  nameSizeBytes: number;
}

export interface PicoVDBRoot {
  key: [number, number];
  state: number;
}

export interface PicoVDBNodeMask {
  value: number;
  valueCount: number;
  child: number;
  childCount: number;
  inside: number;
}

export interface PicoVDBLeafMask {
  value: number;
  valueCount: number;
  inside: number;
}

export interface PicoVDBUpper {
  indexBBox: Int32Array; // 6 elements
  mask: PicoVDBNodeMask[]; // 1024 elements
}

export interface PicoVDBLower {
  indexBBox: Int32Array; // 6 elements
  mask: PicoVDBNodeMask[]; // 128 elements
}

export interface PicoVDBLeaf {
  indexBBox: Int32Array; // 6 elements
  mask: PicoVDBLeafMask[]; // 16 elements
}

export class PicoVDBFile {
  private buffer: ArrayBuffer;
  private view: DataView;

  // Header
  header: PicoVDBFileHeader;

  // Slices (as Uint8Arrays for WebGPU)
  gridsBuffer: Uint8Array;
  rootsBuffer: Uint8Array;
  uppersBuffer: Uint8Array;
  lowersBuffer: Uint8Array;
  leavesBuffer: Uint8Array;
  dataBuffer: Uint8Array;

  constructor(buffer: ArrayBuffer) {
    this.buffer = buffer;
    this.view = new DataView(buffer);

    let offset = 0;

    // Parse header
    this.header = {
      magic: [this.view.getUint32(offset + 0, true), this.view.getUint32(offset + 4, true)],
      checksum: this.view.getUint32(offset + 8, true),
      version: this.view.getUint32(offset + 12, true),
      gridCount: this.view.getUint32(offset + 16, true),
      rootCount: this.view.getUint32(offset + 20, true),
      upperCount: this.view.getUint32(offset + 24, true),
      lowerCount: this.view.getUint32(offset + 28, true),
      leafCount: this.view.getUint32(offset + 32, true),
      dataCount: this.view.getUint32(offset + 36, true),
      voxelCount: this.view.getUint32(offset + 40, true),
      nameCount: this.view.getUint32(offset + 44, true),
    };
    offset += PICOVDB_FILE_HEADER_SIZE;

    // Validate magic
    if (this.header.magic[0] !== PICOVDB_MAGIC[0] || this.header.magic[1] !== PICOVDB_MAGIC[1]) {
      throw new Error(`Invalid PicoVDB magic: [0x${this.header.magic[0].toString(16)}, 0x${this.header.magic[1].toString(16)}]`);
    }

    // Create buffer slices for WebGPU
    this.gridsBuffer = new Uint8Array(buffer, offset, this.header.gridCount * PICOVDB_GRID_SIZE);
    offset += this.header.gridCount * PICOVDB_GRID_SIZE;

    this.rootsBuffer = new Uint8Array(buffer, offset, this.header.rootCount * PICOVDB_ROOT_SIZE);
    offset += this.header.rootCount * PICOVDB_ROOT_SIZE;

    this.uppersBuffer = new Uint8Array(buffer, offset, this.header.upperCount * PICOVDB_UPPER_SIZE);
    offset += this.header.upperCount * PICOVDB_UPPER_SIZE;

    this.lowersBuffer = new Uint8Array(buffer, offset, this.header.lowerCount * PICOVDB_LOWER_SIZE);
    offset += this.header.lowerCount * PICOVDB_LOWER_SIZE;

    this.leavesBuffer = new Uint8Array(buffer, offset, this.header.leafCount * PICOVDB_LEAF_SIZE);
    offset += this.header.leafCount * PICOVDB_LEAF_SIZE;

    this.dataBuffer = new Uint8Array(buffer, offset, this.header.dataCount);
  }

  getSize(): number {
    return this.buffer.byteLength;
  }

  getGrid(index: number): PicoVDBGrid {
    if (index >= this.header.gridCount) {
      throw new Error(`Grid index ${index} out of bounds (max: ${this.header.gridCount - 1})`);
    }

    const baseOffset = PICOVDB_FILE_HEADER_SIZE + index * PICOVDB_GRID_SIZE;
    let offset = baseOffset;

    return {
      matf: new Float32Array(this.buffer, offset + 0, 9),
      matfi: new Float32Array(this.buffer, offset + 36, 9),
      worldBBox: new Float32Array(this.buffer, offset + 72, 6),
      indexBBox: new Int32Array(this.buffer, offset + 96, 6),
      gridIndex: this.view.getUint32(offset + 120, true),
      rootIndex: this.view.getUint32(offset + 124, true),
      upperIndex: this.view.getUint32(offset + 128, true),
      lowerIndex: this.view.getUint32(offset + 132, true),
      leafIndex: this.view.getUint32(offset + 136, true),
      dataIndex: this.view.getUint32(offset + 140, true),
      gridType: this.view.getUint32(offset + 144, true),
      nameIndex: this.view.getUint32(offset + 148, true),
      nameSizeBytes: this.view.getUint32(offset + 152, true),
    };
  }

  getRoot(index: number): PicoVDBRoot {
    if (index >= this.header.rootCount) {
      throw new Error(`Root index ${index} out of bounds (max: ${this.header.rootCount - 1})`);
    }

    const baseOffset = PICOVDB_FILE_HEADER_SIZE + this.header.gridCount * PICOVDB_GRID_SIZE + index * PICOVDB_ROOT_SIZE;

    return {
      key: [this.view.getUint32(baseOffset + 0, true), this.view.getUint32(baseOffset + 4, true)],
      state: this.view.getUint32(baseOffset + 8, true),
    };
  }

  getUpper(index: number): PicoVDBUpper {
    if (index >= this.header.upperCount) {
      throw new Error(`Upper index ${index} out of bounds (max: ${this.header.upperCount - 1})`);
    }

    const baseOffset = PICOVDB_FILE_HEADER_SIZE +
      this.header.gridCount * PICOVDB_GRID_SIZE +
      this.header.rootCount * PICOVDB_ROOT_SIZE +
      index * PICOVDB_UPPER_SIZE;

    const masks: PicoVDBNodeMask[] = [];
    for (let i = 0; i < 1024; i++) {
      const maskOffset = baseOffset + 24 + i * PICOVDB_NODE_MASK_SIZE;
      masks.push({
        value: this.view.getUint32(maskOffset + 0, true),
        valueCount: this.view.getUint32(maskOffset + 4, true),
        child: this.view.getUint32(maskOffset + 8, true),
        childCount: this.view.getUint32(maskOffset + 12, true),
        inside: this.view.getUint32(maskOffset + 16, true),
      });
    }

    return {
      indexBBox: new Int32Array(this.buffer, baseOffset + 0, 6),
      mask: masks,
    };
  }

  getLower(index: number): PicoVDBLower {
    if (index >= this.header.lowerCount) {
      throw new Error(`Lower index ${index} out of bounds (max: ${this.header.lowerCount - 1})`);
    }

    const baseOffset = PICOVDB_FILE_HEADER_SIZE +
      this.header.gridCount * PICOVDB_GRID_SIZE +
      this.header.rootCount * PICOVDB_ROOT_SIZE +
      this.header.upperCount * PICOVDB_UPPER_SIZE +
      index * PICOVDB_LOWER_SIZE;

    const masks: PicoVDBNodeMask[] = [];
    for (let i = 0; i < 128; i++) {
      const maskOffset = baseOffset + 24 + i * PICOVDB_NODE_MASK_SIZE;
      masks.push({
        value: this.view.getUint32(maskOffset + 0, true),
        valueCount: this.view.getUint32(maskOffset + 4, true),
        child: this.view.getUint32(maskOffset + 8, true),
        childCount: this.view.getUint32(maskOffset + 12, true),
        inside: this.view.getUint32(maskOffset + 16, true),
      });
    }

    return {
      indexBBox: new Int32Array(this.buffer, baseOffset + 0, 6),
      mask: masks,
    };
  }

  getLeaf(index: number): PicoVDBLeaf {
    if (index >= this.header.leafCount) {
      throw new Error(`Leaf index ${index} out of bounds (max: ${this.header.leafCount - 1})`);
    }

    const baseOffset = PICOVDB_FILE_HEADER_SIZE +
      this.header.gridCount * PICOVDB_GRID_SIZE +
      this.header.rootCount * PICOVDB_ROOT_SIZE +
      this.header.upperCount * PICOVDB_UPPER_SIZE +
      this.header.lowerCount * PICOVDB_LOWER_SIZE +
      index * PICOVDB_LEAF_SIZE;

    const masks: PicoVDBLeafMask[] = [];
    for (let i = 0; i < 16; i++) {
      const maskOffset = baseOffset + 24 + i * PICOVDB_LEAF_MASK_SIZE;
      masks.push({
        value: this.view.getUint32(maskOffset + 0, true),
        valueCount: this.view.getUint32(maskOffset + 4, true),
        inside: this.view.getUint32(maskOffset + 8, true),
      });
    }

    return {
      indexBBox: new Int32Array(this.buffer, baseOffset + 0, 6),
      mask: masks,
    };
  }

  getGridFloat(grid: PicoVDBGrid, index: number): number {
    const dataPtr = new Float32Array(this.dataBuffer.buffer, this.dataBuffer.byteOffset);
    return dataPtr[grid.dataIndex / 4 + index]; // dataIndex is in bytes, convert to float index
  }
}
