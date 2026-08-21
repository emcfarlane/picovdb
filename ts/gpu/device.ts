// WebGPU device and buffer helpers.

/** Whether a GPU adapter is available. */
export async function hasWebGPU(): Promise<boolean> {
  const gpu = (globalThis as { navigator?: Navigator }).navigator?.gpu;
  if (!gpu) return false;
  return (await gpu.requestAdapter()) !== null;
}

/**
 * Device limits the grid kernels need above the WebGPU defaults, raised to
 * what the adapter supports: value slabs are 2 KB per leaf, workgroups are
 * 256 wide, and kernels bind up to ten storage buffers, which the defaults
 * of a compatibility mode adapter do not allow.
 */
export function gridLimits(adapter: GPUAdapter): Record<string, number> {
  const l = adapter.limits;
  return {
    maxStorageBufferBindingSize: l.maxStorageBufferBindingSize,
    maxBufferSize: l.maxBufferSize,
    maxStorageBuffersPerShaderStage: l.maxStorageBuffersPerShaderStage,
    maxComputeWorkgroupSizeX: l.maxComputeWorkgroupSizeX,
    maxComputeInvocationsPerWorkgroup: l.maxComputeInvocationsPerWorkgroup,
    maxComputeWorkgroupStorageSize: l.maxComputeWorkgroupStorageSize,
    maxComputeWorkgroupsPerDimension: l.maxComputeWorkgroupsPerDimension,
  };
}

export async function requestDevice(): Promise<GPUDevice> {
  const gpu = (globalThis as { navigator?: Navigator }).navigator?.gpu;
  if (!gpu) throw new Error('WebGPU unavailable');
  const adapter = await gpu.requestAdapter();
  if (!adapter) throw new Error('WebGPU adapter unavailable');
  return adapter.requestDevice({ requiredLimits: gridLimits(adapter) });
}

export function createU32Buffer(device: GPUDevice, data: Uint32Array<ArrayBuffer>, extraUsage: GPUBufferUsageFlags = 0): GPUBuffer {
  const buffer = device.createBuffer({
    size: Math.max(data.byteLength, 4),
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC | extraUsage,
  });
  device.queue.writeBuffer(buffer, 0, data);
  return buffer;
}

export const DISPATCH_STRIDE = 65535;

/**
 * Linearized 2D dispatch. The WGSL side derives the workgroup index from
 * the stride, so when groups spill into y the x count must equal the
 * stride.
 */
export function dispatch2D(pass: GPUComputePassEncoder, groups: number): void {
  if (groups <= DISPATCH_STRIDE) {
    pass.dispatchWorkgroups(groups, 1);
  } else {
    pass.dispatchWorkgroups(DISPATCH_STRIDE, Math.ceil(groups / DISPATCH_STRIDE));
  }
}

/** Reads one u32 per request in a single copy and map. */
export async function readBackTotals(
  device: GPUDevice,
  reads: Array<{ buffer: GPUBuffer; index: number }>
): Promise<number[]> {
  const staging = device.createBuffer({
    size: reads.length * 4,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });
  const encoder = device.createCommandEncoder();
  reads.forEach((r, i) => encoder.copyBufferToBuffer(r.buffer, r.index * 4, staging, i * 4, 4));
  device.queue.submit([encoder.finish()]);
  await staging.mapAsync(GPUMapMode.READ);
  const out = [...new Uint32Array(staging.getMappedRange().slice(0))];
  staging.destroy();
  return out;
}

/** Throws when a storage binding would exceed the device limit. */
export function checkBindingSize(device: GPUDevice, bytes: number, label: string): void {
  if (bytes > device.limits.maxStorageBufferBindingSize) {
    throw new Error(`${label} needs ${bytes} bytes, over the ${device.limits.maxStorageBufferBindingSize} byte storage binding limit`);
  }
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
