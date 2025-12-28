// Loader for PicoVDB files in a web-based 3D viewer using TypeScript and Fetch API.

import { PicoVDBFile } from '../../picovdb.js';

export async function loadPicoVDB(url: string): Promise<PicoVDBFile> {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to load PicoVDB file ${url}: ${response.statusText}`);
  }

  let data: ArrayBuffer;

  // Check if file is gzipped based on URL extension
  if (url.endsWith('.gz')) {
    console.log('Decompressing gzipped PicoVDB file...');
    const compressedData = await response.arrayBuffer();
    console.log(`Loaded compressed PicoVDB file: ${compressedData.byteLength} bytes`);

    // Check if DecompressionStream is supported
    if (typeof DecompressionStream === 'undefined') {
      throw new Error('Gzip decompression not supported in this browser. Please use a modern browser with Compression Streams API support.');
    }

    // Decompress using native Compression Streams API
    const stream = new Response(compressedData).body!
      .pipeThrough(new DecompressionStream('gzip'));
    data = await new Response(stream).arrayBuffer();
    console.log(`Decompressed PicoVDB file: ${data.byteLength} bytes`);
  } else {
    data = await response.arrayBuffer();
    console.log(`Loaded raw PicoVDB file: ${data.byteLength} bytes`);
  }

  // Ensure 4-byte alignment
  const paddedSize = Math.ceil(data.byteLength / 4) * 4;
  let alignedData: ArrayBuffer;

  if (paddedSize === data.byteLength) {
    alignedData = data;
  } else {
    // Create padded buffer
    alignedData = new ArrayBuffer(paddedSize);
    const paddedView = new Uint8Array(alignedData);
    paddedView.set(new Uint8Array(data));
    console.log(`PicoVDB file padded: ${data.byteLength} → ${paddedSize} bytes`);
  }

  // Create PicoVDB file view (validates magic number internally)
  const picoFile = new PicoVDBFile(alignedData);

  console.log('PicoVDB File loaded successfully:');
  console.log('PicoVDB File Header:');
  console.log(`  Magic: [0x${picoFile.header.magic[0].toString(16)}, 0x${picoFile.header.magic[1].toString(16)}]`);
  console.log(`  Version: ${picoFile.header.version}`);
  console.log(`  Grid Count: ${picoFile.header.gridCount}`);
  console.log(`  Root Count: ${picoFile.header.rootCount}`);
  console.log(`  Upper Count: ${picoFile.header.upperCount}`);
  console.log(`  Lower Count: ${picoFile.header.lowerCount}`);
  console.log(`  Leaf Count: ${picoFile.header.leafCount}`);
  console.log(`  Data Count: ${picoFile.header.dataCount} bytes`);
  console.log(`  Voxel Count: ${picoFile.header.voxelCount}`);
  if (picoFile.header.gridCount === 0) {
    throw new Error('PicoVDB file contains no grids');
  }
  return picoFile
}
