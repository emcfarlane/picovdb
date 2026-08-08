// WebGPU device and buffer helpers.

/** Whether a GPU adapter is available. */
export async function hasWebGPU(): Promise<boolean> {
  const gpu = (globalThis as { navigator?: Navigator }).navigator?.gpu;
  if (!gpu) return false;
  return (await gpu.requestAdapter()) !== null;
}

export async function requestDevice(): Promise<GPUDevice> {
  const gpu = (globalThis as { navigator?: Navigator }).navigator?.gpu;
  if (!gpu) throw new Error('WebGPU unavailable');
  const adapter = await gpu.requestAdapter();
  if (!adapter) throw new Error('WebGPU adapter unavailable');
  return adapter.requestDevice();
}

export function createU32Buffer(device: GPUDevice, data: Uint32Array<ArrayBuffer>, extraUsage: GPUBufferUsageFlags = 0): GPUBuffer {
  const buffer = device.createBuffer({
    size: Math.max(data.byteLength, 4),
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC | extraUsage,
  });
  device.queue.writeBuffer(buffer, 0, data);
  return buffer;
}

export async function readBackU32(device: GPUDevice, src: GPUBuffer, count: number): Promise<Uint32Array> {
  const staging = device.createBuffer({
    size: count * 4,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });
  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(src, 0, staging, 0, count * 4);
  device.queue.submit([encoder.finish()]);
  await staging.mapAsync(GPUMapMode.READ);
  const out = new Uint32Array(staging.getMappedRange().slice(0));
  staging.destroy();
  return out;
}
