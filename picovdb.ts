// PicoVDB file format

// "PicoVDB0" in hex little endian  
export const PICOVDB_MAGIC = [0x6f636950, 0x30424456];

// Grid type constants
export const GRID_TYPE_SDF_FLOAT = 1;
export const GRID_TYPE_SDF_UINT8 = 2;

export const PICOVDB_FILE_HEADER_SIZE = 32;
export const PICOVDB_GRID_SIZE = 64;
export const PICOVDB_ROOT_SIZE = 8;
export const PICOVDB_NODE_MASK_SIZE = 16;
export const PICOVDB_LEAF_MASK_SIZE = 12;
export const PICOVDB_UPPER_SIZE = 16384;
export const PICOVDB_LOWER_SIZE = 2048;
export const PICOVDB_LEAF_SIZE = 192;
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

export interface PicoVDBNodeMask {
  inside: number;
  value: number;
  valueOffset: number;
  childOffset: number;
}

export interface PicoVDBLeafMask {
  inside: number;
  value: number;
  valueOffset: number;
}

export interface PicoVDBUpper {
  mask: PicoVDBNodeMask[]; // 1024 elements
}

export interface PicoVDBLower {
  mask: PicoVDBNodeMask[]; // 128 elements
}

export interface PicoVDBLeaf {
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
    console.log("GRIDS BUFFER", this.gridsBuffer.length)

    const rootCount = this.getRootCountPadded()
    this.rootsBuffer = new Uint8Array(buffer, offset, rootCount * PICOVDB_ROOT_SIZE);
    offset += rootCount * PICOVDB_ROOT_SIZE;
    console.log("ROOTS BUFFER", this.rootsBuffer.length)

    this.uppersBuffer = new Uint8Array(buffer, offset, this.header.upperCount * PICOVDB_UPPER_SIZE);
    offset += this.header.upperCount * PICOVDB_UPPER_SIZE;
    console.log("UPPERS BUFFER", this.uppersBuffer.length)

    this.lowersBuffer = new Uint8Array(buffer, offset, this.header.lowerCount * PICOVDB_LOWER_SIZE);
    offset += this.header.lowerCount * PICOVDB_LOWER_SIZE;
    console.log("LOWERS BUFFER", this.lowersBuffer.length)

    this.leavesBuffer = new Uint8Array(buffer, offset, this.header.leafCount * PICOVDB_LEAF_SIZE);
    offset += this.header.leafCount * PICOVDB_LEAF_SIZE;
    console.log("LEAVES BUFFER", this.leavesBuffer.length)

    this.dataBuffer = new Uint8Array(buffer, offset, this.header.dataCount * PICOVDB_DATA_SIZE);
    offset += this.header.dataCount * PICOVDB_DATA_SIZE;
    console.log("DATA BUFFER", this.dataBuffer.length)
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

    const masks: PicoVDBNodeMask[] = [];
    for (let i = 0; i < 1024; i++) {
      const maskOffset = baseOffset + i * PICOVDB_NODE_MASK_SIZE;
      masks.push({
        inside: this.view.getUint32(maskOffset + 0, true),
        value: this.view.getUint32(maskOffset + 4, true),
        valueOffset: this.view.getUint32(maskOffset + 8, true),
        childOffset: this.view.getUint32(maskOffset + 12, true),
      });
    }
    return {
      mask: masks,
    };
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

    const masks: PicoVDBNodeMask[] = [];
    for (let i = 0; i < 128; i++) {
      const maskOffset = baseOffset + i * PICOVDB_NODE_MASK_SIZE;
      masks.push({
        inside: this.view.getUint32(maskOffset + 0, true),
        value: this.view.getUint32(maskOffset + 4, true),
        valueOffset: this.view.getUint32(maskOffset + 8, true),
        childOffset: this.view.getUint32(maskOffset + 12, true),
      });
    }
    return {
      mask: masks,
    };
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

    const masks: PicoVDBLeafMask[] = [];
    for (let i = 0; i < 16; i++) {
      const maskOffset = baseOffset + i * PICOVDB_LEAF_MASK_SIZE;
      masks.push({
        inside: this.view.getUint32(maskOffset + 0, true),
        value: this.view.getUint32(maskOffset + 4, true),
        valueOffset: this.view.getUint32(maskOffset + 8, true),
      });
    }
    return {
      mask: masks,
    };
  }

  getVoxelCount(): number {
    var count = 0
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
