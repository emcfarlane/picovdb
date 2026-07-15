// Early error trap for headless debugging (before any async module work)
window.addEventListener('error', (e) =>
  ((window as any).__errs = ((window as any).__errs ?? [])).push(String(e.error?.stack ?? e.message)));
window.addEventListener('unhandledrejection', (e) =>
  ((window as any).__errs = ((window as any).__errs ?? [])).push('rejection: ' + String((e as PromiseRejectionEvent).reason?.stack ?? (e as PromiseRejectionEvent).reason)));

import { vec3, mat4 } from 'wgpu-matrix';
import DisplayShader from "./blit.wgsl";
import ComputeShader from "./compute.wgsl";
import SpectraShader from "./spectra.wgsl";
import DenoiseShader from "./denoise.wgsl";
import PicoVDBShader from "./../picovdb.wgsl";
import { fetchPicoVDB } from '../picovdb.ts';
import { createOrbitCamera } from './lib/camera';
import { createModelHand } from './lib/hand';
import { createInputHandler } from "./lib/input";
import { initGUI } from './lib/gui';
import type { ModelConfig } from './lib/gui';
import { ILLUMINANTS, buildIlluminantLut, packFloat16 } from './lib/illuminants';
import type { IlluminantLut } from './lib/illuminants';
import { parseFourierLut, srgbToFourierSrgb } from './lib/fourier_lut';
import { parseEnv, buildEnvCdfs } from './lib/env';
import type { Vec3 } from './lib/spectra';

const MODEL_BASE = './models/';
const models: ModelConfig[] = [
  { name: 'Bunny', url: `${MODEL_BASE}bunny.pvdb.gz`, translation: [-40, 240, 0], scale: 120 },
  { name: 'Bunny u8', url: `${MODEL_BASE}bunny.u8.pvdb.gz`, translation: [-40, 240, 0], scale: 120 },
  { name: 'Dragon u8', url: `${MODEL_BASE}dragon.u8.pvdb.gz`, translation: [0, 80, 0], scale: 240 },
  //{ name: 'Smoke', url: `${MODEL_BASE}smoke.pvdb.gz`, translation: [0, 0, 0], scale: 60 },
  { name: 'Sphere', url: `${MODEL_BASE}sphere.pvdb.gz`, translation: [0, 0, 0], scale: 30 },
];

const modelParam = new URLSearchParams(window.location.search).get('model');
const initialModel = models.find(m => m.url.endsWith('/' + modelParam)) ?? models[0];

const { controls, gui, modelController, pauseController, highDPIController, rotationController } = initGUI(models, ILLUMINANTS.map(i => i.name), initialModel.name);
import { createSkyState } from "./lib/hw_skymodel";
import { TimestampQueryManager } from './lib/TimestampQueryManager';
import { Stats } from './lib/Stats';

const canvas = document.getElementById("canvas") as HTMLCanvasElement;
const infoTextElement = document.getElementById("info-text")!;

if (!canvas) {
  throw new Error("No canvas found.");
}
if (!navigator.gpu) {
  const isInsecure = window.isSecureContext === false;
  throw new Error(
    isInsecure
      ? "WebGPU requires a secure context (HTTPS or localhost). Current origin is not secure."
      : "WebGPU not supported on this browser."
  );
}
console.log("WebGPU is supported!");

const adapter = await navigator.gpu.requestAdapter({
  featureLevel: 'compatibility',
});
if (!adapter) {
  throw new Error("No appropriate GPUAdapter found.");
}

let width = canvas.width;
let height = canvas.height;
// Path tracing runs at renderScale * canvas resolution; the blit pass
// upscales with linear filtering.
let renderWidth = width;
let renderHeight = height;
let raytracedTexture: GPUTexture;
let accumulationBuffer: GPUBuffer;
let gbufferAccumBuffer: GPUBuffer;
// Primary-hit G-buffer ping-pong (denoiser temporal input)
let gbufferTextures: GPUTexture[] = [];
// SVGF denoiser textures: raw radiance + history/moments/a-trous ping-pongs
let illumTexture: GPUTexture;
let specTexture: GPUTexture;
let denoisedTexture: GPUTexture;
let stabTextures: GPUTexture[] = [];
let histTextures: GPUTexture[] = [];
// ReBLUR fast history: ~6-frame luma EMA ping-pong (r32float)
let fastTextures: GPUTexture[] = [];
let momentsTextures: GPUTexture[] = [];
let atrousTextures: GPUTexture[] = [];
let denoiseTemporalGroups: GPUBindGroup[] = [];
let denoiseAtrousGroups: GPUBindGroup[][] = [];
let denoiseResolveGroups: GPUBindGroup[] = [];
let denoiseStabilizeGroups: GPUBindGroup[] = [];
let displayBindGroups: GPUBindGroup[] = [];
let perFrameBindGroup: GPUBindGroup;
let dataBindGroup: GPUBindGroup;
// Two pass bind groups with the G-buffer ping-pong swapped
let passBindGroups: GPUBindGroup[] = [];
let gdepthTexture: GPUTexture;
let hitdistTexture: GPUTexture;
let backdropTexture: GPUTexture;

// Set canvas to fullscreen size and recreate GPU resources
function resizeCanvas() {
  const pixelRatio = controls.highDPI ? window.devicePixelRatio : 1.0;
  canvas.width = window.innerWidth * pixelRatio;
  canvas.height = window.innerHeight * pixelRatio;
  width = canvas.width;
  height = canvas.height;

  // Will recreate GPU resources after they're initially created
  if (raytracedTexture) {
    createGPUResources();
    updatePixelRadius();
  }
}

resizeCanvas();
window.addEventListener('resize', resizeCanvas);

// Update canvas size when High DPI setting changes
highDPIController.onChange(() => {
  resizeCanvas();
  updatePixelRadius();
});

// The use of timestamps require a dedicated adapter feature:
// The adapter may or may not support timestamp queries. If not, we simply
// don't measure timestamps and deactivate the timer display.
const timestampQueryFeature = 'timestamp-query'
const supportsTimestampQueries = adapter?.features.has(timestampQueryFeature);

const requiredFeatures: GPUFeatureName[] = [];
if (supportsTimestampQueries) { requiredFeatures.push(timestampQueryFeature); }

// The path tracer binds 9 storage buffers (6 PicoVDB + objects + sky +
// accumulation); the default limit is 8. Request whatever headroom the
// adapter offers (a paint buffer will join later).
const requiredLimits: Record<string, number> = {};
if (adapter.limits.maxStorageBuffersPerShaderStage > 8) {
  requiredLimits.maxStorageBuffersPerShaderStage =
    Math.min(16, adapter.limits.maxStorageBuffersPerShaderStage);
}
// The path reservoirs are 128 B/pixel (64 B x 2 halves): at 100% render
// scale on a large window that exceeds the DEFAULT limits (128 MB binding,
// 256 MB buffer) — request what the adapter actually offers.
requiredLimits.maxStorageBufferBindingSize = adapter.limits.maxStorageBufferBindingSize;
requiredLimits.maxBufferSize = adapter.limits.maxBufferSize;
// computeMain writes 5 storage textures (output, spec, gbuffer, illum, depth)
if (adapter.limits.maxStorageTexturesPerShaderStage > 4) {
  requiredLimits.maxStorageTexturesPerShaderStage =
    Math.min(8, adapter.limits.maxStorageTexturesPerShaderStage);
}

const device = await adapter.requestDevice({ requiredFeatures, requiredLimits });
device.addEventListener('uncapturederror', event => {
  console.log(event.error);
  ((window as any).__errs = ((window as any).__errs ?? [])).push('gpu: ' + event.error.message);
});

const context = canvas.getContext("webgpu");
if (!context) {
  throw new Error("No context found.");
}

var stats = new Stats();
var gpuPanel = stats.addPanel(new Stats.Panel('GPU', '#ff8', '#221'));
document.body.appendChild(stats.dom);

// GPU-side timer and the CPU-side counter where we accumulate statistics:
// NB: Look for 'timestampQueryManager' in this file to locate parts of this
// snippets that are related to timestamps. Most of the logic is in
// TimestampQueryManager.ts.
let lastGpuMs = 0;
const timestampQueryManager = new TimestampQueryManager(device, (elapsedNs) => {
  // Convert from nanoseconds to milliseconds:
  const elapsedMs = Number(elapsedNs) * 1e-6;
  lastGpuMs = elapsedMs;
  gpuPanel.update(elapsedMs, 16); // 16ms = 60fps target
});
// Headless-test hook: EMA of measured GPU frame time (ms), for FPS A/Bs
let gpuMsEma = 0;
(window as any).__gpuMs = () => {
  gpuMsEma = gpuMsEma === 0 ? lastGpuMs : gpuMsEma * 0.8 + lastGpuMs * 0.2;
  return JSON.stringify({ last: +lastGpuMs.toFixed(2), ema: +gpuMsEma.toFixed(2) });
};

const canvasFormat = navigator.gpu.getPreferredCanvasFormat();
context.configure({
  device: device,
  format: canvasFormat,
});

// https://webgpufundamentals.org/webgpu/lessons/webgpu-large-triangle-to-cover-clip-space.html
const vertices = new Float32Array([
  // X,  Y,
  -1, 3, // Triangle 1
  3, -1,
  -1, -1,
]);

const vertexBuffer = device.createBuffer({
  label: "Display vertices",
  size: vertices.byteLength, // 4 bytes * 6 vertices = 24 bytes.
  usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
});
device.queue.writeBuffer(vertexBuffer, /* offset */ 0, vertices);

const vertexBufferLayout: GPUVertexBufferLayout = {
  // 2 floats for position.
  arrayStride: 8,
  attributes: [{
    format: "float32x2",
    offset: 0,
    shaderLocation: 0, // Position, see vertex shader
  }],
};

// GPU buffers for PicoVDB data (reassigned on model load)
let gridsBuffer: GPUBuffer;
let rootsBuffer: GPUBuffer;
let uppersBuffer: GPUBuffer;
let lowersBuffer: GPUBuffer;
let leavesBuffer: GPUBuffer;
let dataBuffer: GPUBuffer;
let currentModelConfig: ModelConfig = models[0];

// Create size-dependent GPU resources
function createGPUResources() {
  if (raytracedTexture) { raytracedTexture.destroy(); }
  if (accumulationBuffer) { accumulationBuffer.destroy(); }

  // Number() guards against option values arriving as strings
  let scale = Number(controls.renderScale) || 0.5;
  // Clamp so the reservoir buffer (128 B/pixel) fits the device's storage
  // binding limit rather than failing createBuffer and killing the frame
  const maxPixels = Math.floor(device.limits.maxStorageBufferBindingSize / 240);
  if (width * scale * height * scale > maxPixels) {
    const fit = Math.sqrt(maxPixels / (width * height));
    console.warn(`Render scale ${scale} exceeds reservoir budget; clamping to ${fit.toFixed(2)}`);
    scale = fit;
  }
  renderWidth = Math.max(1, Math.floor(width * scale));
  renderHeight = Math.max(1, Math.floor(height * scale));

  raytracedTexture = device.createTexture({
    size: [renderWidth, renderHeight],
    format: 'rgba8unorm',
    usage: GPUTextureUsage.STORAGE_BINDING |
      GPUTextureUsage.TEXTURE_BINDING |
      GPUTextureUsage.COPY_SRC
  });

  // Radiance sum per pixel (vec4f: rgb + count); restarted via frame_index
  accumulationBuffer = device.createBuffer({
    label: 'Accumulation',
    size: renderWidth * renderHeight * 16,
    // COPY_SRC: __readAccum debug readback
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
  // G-buffer accumulation (stride 2 vec4/pixel): stable anti-aliased normal/depth
  gbufferAccumBuffer?.destroy();
  gbufferAccumBuffer = device.createBuffer({
    label: 'G-buffer + illum accumulation',
    // Stride 5 vec4/pixel: nsum, dsum, isum, ssum, msum (miss/backdrop)
    size: renderWidth * renderHeight * 5 * 16,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
  resetAccumulation("gpu-resources");


  // Bind group 0: per-frame
  perFrameBindGroup = device.createBindGroup({
    label: 'Per-frame bind group',
    layout: perFrameBindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: inputBuffer } },
      { binding: 1, resource: { buffer: objectsBuffer } },
      { binding: 2, resource: { buffer: skyStateBuffer } },
      { binding: 3, resource: { buffer: materialsBuffer } },
      { binding: 4, resource: { buffer: lightsBuffer } },
      { binding: 5, resource: illuminantTexture.createView() },
      { binding: 6, resource: illuminantSampler },
      { binding: 7, resource: environmentTexture.createView() },
      { binding: 8, resource: envCdfConditionalTexture.createView() },
      { binding: 9, resource: envCdfMarginalTexture.createView() },
      { binding: 10, resource: fourierLutTexture.createView() },
    ]
  });

  // Bind group 1: data
  if (gridsBuffer) {
    dataBindGroup = device.createBindGroup({
      label: 'Data bind group',
      layout: dataBindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: gridsBuffer } },
        { binding: 1, resource: { buffer: rootsBuffer } },
        { binding: 2, resource: { buffer: uppersBuffer } },
        { binding: 3, resource: { buffer: lowersBuffer } },
        { binding: 4, resource: { buffer: leavesBuffer } },
        { binding: 5, resource: { buffer: dataBuffer } },
      ]
    });
  }

  // G-buffer ping-pong (normal + material index per primary hit)
  gbufferTextures.forEach(t => t.destroy());
  gbufferTextures = [0, 1].map(i => device.createTexture({
    label: `G-buffer ${i}`,
    size: [renderWidth, renderHeight],
    format: 'rgba32float',
    // COPY_SRC: __readGbuf debug readback
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC,
  }));

  const denoiseTex = (label: string) => device.createTexture({
    label,
    size: [renderWidth, renderHeight],
    format: 'rgba32float',
    // COPY_SRC: __readHist debug readback
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC,
  });
  [illumTexture, specTexture, denoisedTexture, ...stabTextures, ...histTextures, ...momentsTextures, ...atrousTextures]
    .forEach(t => t?.destroy());
  illumTexture = denoiseTex('Denoise illum');
  specTexture = denoiseTex('Denoise specular');
  denoisedTexture = denoiseTex('Denoise resolved (linear)');
  stabTextures = [denoiseTex('Stab hist 0'), denoiseTex('Stab hist 1')];
  histTextures = [denoiseTex('Denoise hist 0'), denoiseTex('Denoise hist 1')];
  fastTextures.forEach(t => t.destroy());
  fastTextures = [0, 1].map(i => device.createTexture({
    label: `Fast history ${i}`,
    size: [renderWidth, renderHeight],
    format: 'r32float',
    // COPY_SRC: __readTex debug readback
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC,
  }));
  momentsTextures = [denoiseTex('Denoise moments 0'), denoiseTex('Denoise moments 1')];
  atrousTextures = [denoiseTex('Denoise atrous 0'), denoiseTex('Denoise atrous 1')];
  // Accumulated first-bounce distance -> ReBLUR blur radius (read by the blur
  // pass, so it must exist before createDenoiseBindGroups())
  hitdistTexture?.destroy();
  hitdistTexture = device.createTexture({
    label: 'Hit distance',
    size: [renderWidth, renderHeight],
    format: 'r32float',
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING,
  });
  // Mean miss-frame radiance: the display composites
  // mix(backdrop, denoised, coverage) to keep the jitter's AA
  backdropTexture?.destroy();
  backdropTexture = device.createTexture({
    label: 'Backdrop (miss mean)',
    size: [renderWidth, renderHeight],
    format: 'rgba32float',
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING,
  });
  createDenoiseBindGroups();

  // Exact primary depth (r32float; the P4 duplication map becomes a buffer)
  gdepthTexture?.destroy();
  gdepthTexture = device.createTexture({
    label: 'Primary depth',
    size: [renderWidth, renderHeight],
    format: 'r32float',
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING,
  });

  displayBindGroups = [0, 1].map(P => device.createBindGroup({
    label: `Display bind group ${P}`,
    layout: displayPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: raytracedTexture.createView() },
      { binding: 1, resource: displaySampler },
      { binding: 2, resource: gbufferTextures[1 - P].createView() },
      { binding: 3, resource: gdepthTexture.createView() },
      { binding: 4, resource: illumTexture.createView() },
      { binding: 5, resource: { buffer: inputBuffer } },
      { binding: 6, resource: { buffer: objectsBuffer } },
    ]
  }));


  // Bind group 2 variants (parity = ping-pong; gbuffer_out = [1 - parity]).
  // passBindGroups: full-res — Reference megakernel + primaryMain (writes
  // full-res G-buffer 3 + depth 8; other bindings present but unused).
  // Bind group 2 for computeMain (PT): writes G-buffer 3 + depth 8 + illum 5
  // + output 0 + accumulation 1 (parity: gbuffer_out = [1 - parity])
  passBindGroups = [0, 1].map(parity => device.createBindGroup({
    label: `Pass bind group ${parity}`,
    layout: passBindGroupLayout,
    entries: [
      { binding: 0, resource: raytracedTexture.createView() },
      { binding: 1, resource: { buffer: accumulationBuffer } },
      { binding: 2, resource: specTexture.createView() },
      { binding: 6, resource: hitdistTexture.createView() },
      { binding: 7, resource: backdropTexture.createView() },
      { binding: 4, resource: { buffer: gbufferAccumBuffer } },
      { binding: 3, resource: gbufferTextures[1 - parity].createView() },
      { binding: 5, resource: illumTexture.createView() },
      { binding: 8, resource: gdepthTexture.createView() },
    ]
  }));
}

// Create the sampler
const displaySampler = device.createSampler({
  addressModeU: 'clamp-to-edge',
  addressModeV: 'clamp-to-edge',
  magFilter: 'linear',
  minFilter: 'linear',
});

const displayShaderModule = device.createShaderModule({
  label: "Display shader",
  code: DisplayShader,
});

const displayPipeline = device.createRenderPipeline({
  label: "Display pipeline",
  layout: "auto",
  vertex: {
    module: displayShaderModule,
    entryPoint: "vertexMain",
    buffers: [vertexBufferLayout]
  },
  fragment: {
    module: displayShaderModule,
    entryPoint: "fragmentMain",
    targets: [{
      format: canvasFormat,
    }],
  },
});

const inputHandler = createInputHandler(window, canvas);

// Load a PicoVDB model and create GPU buffers
async function loadModel(config: ModelConfig) {
  infoTextElement.textContent = `Loading ${config.name}...`;

  const picoVDBFile = await fetchPicoVDB(config.url);
  console.log('PicoVDB File loaded successfully:');
  console.log('PicoVDB File Header:');
  console.log(`  Magic: [0x${picoVDBFile.header.magic[0].toString(16)}, 0x${picoVDBFile.header.magic[1].toString(16)}]`);
  console.log(`  Version: ${picoVDBFile.header.version}`);
  console.log(`  Grid Count: ${picoVDBFile.header.gridCount}`);
  console.log(`  Upper Count: ${picoVDBFile.header.upperCount}`);
  console.log(`  Lower Count: ${picoVDBFile.header.lowerCount}`);
  console.log(`  Leaf Count: ${picoVDBFile.header.leafCount}`);
  console.log(`  Data Count: ${picoVDBFile.header.dataCount} bytes`);
  console.log(`  Voxel Count: ${picoVDBFile.getVoxelCount()}`);
  if (picoVDBFile.header.gridCount === 0) {
    throw new Error('PicoVDB file contains no grids');
  }

  // Destroy old GPU buffers
  if (gridsBuffer) {
    gridsBuffer.destroy();
    rootsBuffer.destroy();
    uppersBuffer.destroy();
    lowersBuffer.destroy();
    leavesBuffer.destroy();
    dataBuffer.destroy();
  }

  // Create new GPU buffers
  gridsBuffer = device.createBuffer({
    label: 'PicoVDB Grids',
    size: picoVDBFile.gridsBuffer.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(gridsBuffer, 0, picoVDBFile.gridsBuffer);

  rootsBuffer = device.createBuffer({
    label: 'PicoVDB Roots',
    size: picoVDBFile.rootsBuffer.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(rootsBuffer, 0, picoVDBFile.rootsBuffer);

  uppersBuffer = device.createBuffer({
    label: 'PicoVDB Uppers',
    size: picoVDBFile.uppersBuffer.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(uppersBuffer, 0, picoVDBFile.uppersBuffer);

  lowersBuffer = device.createBuffer({
    label: 'PicoVDB Lowers',
    size: picoVDBFile.lowersBuffer.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(lowersBuffer, 0, picoVDBFile.lowersBuffer);

  leavesBuffer = device.createBuffer({
    label: 'PicoVDB Leaves',
    size: picoVDBFile.leavesBuffer.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(leavesBuffer, 0, picoVDBFile.leavesBuffer);

  dataBuffer = device.createBuffer({
    label: 'PicoVDB Data',
    size: picoVDBFile.dataBuffer.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(dataBuffer, 0, picoVDBFile.dataBuffer);

  currentModelConfig = config;
  resetAccumulation("model-load");


  // Recreate data bind group with new buffers
  dataBindGroup = device.createBindGroup({
    label: 'Data bind group',
    layout: dataBindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: gridsBuffer } },
      { binding: 1, resource: { buffer: rootsBuffer } },
      { binding: 2, resource: { buffer: uppersBuffer } },
      { binding: 3, resource: { buffer: lowersBuffer } },
      { binding: 4, resource: { buffer: leavesBuffer } },
      { binding: 5, resource: { buffer: dataBuffer } },
    ]
  });

  // Update transform for new model
  updateObjects();

  // Update info display
  const sizeMB = (picoVDBFile.getSize() / 1024 / 1024).toFixed(1);
  const grid = picoVDBFile.getGrid(0);
  const bboxSize = [
    (grid.indexBoundsMax[0] - grid.indexBoundsMin[0]),
    (grid.indexBoundsMax[1] - grid.indexBoundsMin[1]),
    (grid.indexBoundsMax[2] - grid.indexBoundsMin[2])
  ];
  modelInfoText = `PicoVDB
${config.name} ${sizeMB}MB
Grid: ${bboxSize[0]} × ${bboxSize[1]} × ${bboxSize[2]} units
Voxels: ${picoVDBFile.getVoxelCount()}`;
  infoTextElement.textContent = modelInfoText;
}
let modelInfoText = '';

const fov = (2 * Math.PI) / 5; // 72 degrees
const fovScaled = Math.tan(fov / 2);
const initialCameraPosition = vec3.create(3, 2, 5);
const initialCameraTarget = vec3.create(0, 0, 0);
let camera = createOrbitCamera({
  position: initialCameraPosition,
  target: initialCameraTarget,
});

// Model-in-hand: drag rotates/moves the model; the camera only dollies
// (zoom), so the backdrop and lighting stay put.
const modelHand = createModelHand();

controls.resetCamera = () => {
  camera = createOrbitCamera({
    position: initialCameraPosition,
    target: initialCameraTarget,
  });
  modelHand.reset();
  updateObjects();
};


// Input uniform, must match struct Input in compute.wgsl (224 bytes)
const inputValues = new ArrayBuffer(224);
const inputViews = {
  camera_matrix: new Float32Array(inputValues, 0, 16),
  fov_scale: new Float32Array(inputValues, 64, 1),
  time_delta: new Float32Array(inputValues, 68, 1),
  pixel_radius: new Float32Array(inputValues, 72, 1),
  debug_iterations: new Uint32Array(inputValues, 76, 1),
  frame_index: new Uint32Array(inputValues, 80, 1),
  environment: new Uint32Array(inputValues, 84, 1),
  max_bounces: new Uint32Array(inputValues, 88, 1),
  emission_integral: new Float32Array(inputValues, 92, 1),
  dome_integral: new Float32Array(inputValues, 96, 1),
  exposure: new Float32Array(inputValues, 100, 1),
  light_count: new Uint32Array(inputValues, 104, 1),
  white_background: new Uint32Array(inputValues, 108, 1),
  rng_frame: new Uint32Array(inputValues, 112, 1),
  pass_mode: new Uint32Array(inputValues, 116, 1),
  prev_view: new Float32Array(inputValues, 128, 16),
  prev_camera_pos: new Float32Array(inputValues, 192, 3),
  wavelength_u: new Float32Array(inputValues, 204, 1),
  jitter: new Float32Array(inputValues, 208, 2),
  jitter_mean: new Float32Array(inputValues, 216, 2),
  // Previous frame's jitter_mean (the denoiser's history-fetch offset),
  // packed in the old _pad1/_pad2 slot
  jitter_mean_prev: new Float32Array(inputValues, 120, 2),
};
const inputBuffer = device.createBuffer({
  label: 'Input Uniforms',
  size: inputValues.byteLength,
  usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
});
inputViews.fov_scale[0] = fovScaled;

// --- Materials buffer (matches struct Material in compute.wgsl, 32 bytes) ---
// The palette is authored in linear sRGB; base colors are converted to
// Fourier sRGB at load through the bundled 33^3 LUT (demo/lib/fourier_lut.ts).
const MATERIALS = [
  { // 0: model — saturated blue, glossy
    rgb: [0.0, 0.1, 1.0],
    roughness: 0.4, diffuseAlbedo: [1.0, 0.0], fresnel0: [0.0, 0.04],
  },
  { // 1: ground — near-white seamless studio sweep, matte
    rgb: [0.82, 0.82, 0.82],
    roughness: 0.9, diffuseAlbedo: [1.0, 0.0], fresnel0: [0.0, 0.04],
  },
  { // 2: brush cursor — warm orange, slightly glossy
    rgb: [1.0, 0.45, 0.1],
    roughness: 0.35, diffuseAlbedo: [1.0, 0.0], fresnel0: [0.0, 0.04],
  },
];
const fourierLut = parseFourierLut(
  await (await fetch('./srgb_to_fourier_srgb.lut')).arrayBuffer());
// Uniform arrays with fixed WGSL-side capacity (MAX_MATERIALS / MAX_LIGHTS)
const MATERIAL_STRUCT_SIZE = 32;
const MAX_MATERIALS = 8;
const materialsData = new ArrayBuffer(MATERIAL_STRUCT_SIZE * MAX_MATERIALS);
MATERIALS.forEach((m, i) => {
  const f32 = new Float32Array(materialsData, i * MATERIAL_STRUCT_SIZE, 8);
  f32.set(srgbToFourierSrgb(fourierLut, m.rgb as Vec3), 0);
  f32[3] = m.roughness;
  f32.set(m.diffuseAlbedo, 4);
  f32.set(m.fresnel0, 6);
});
const materialsBuffer = device.createBuffer({
  label: 'Materials',
  size: materialsData.byteLength,
  usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
});
device.queue.writeBuffer(materialsBuffer, 0, materialsData);

// Per-material linear-RGB albedo for the denoiser's SVGF demodulation
const materialAlbedoData = new Float32Array(MAX_MATERIALS * 4);
MATERIALS.forEach((m, i) => materialAlbedoData.set([m.rgb[0], m.rgb[1], m.rgb[2], 0], i * 4));
const materialAlbedoBuffer = device.createBuffer({
  label: 'Material albedo', size: materialAlbedoData.byteLength,
  usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
});
device.queue.writeBuffer(materialAlbedoBuffer, 0, materialAlbedoData);

// Model roughness override (P3 glossy-reuse gate): rewrites material 0's
// roughness so the hybrid-shift reconnection criteria can be exercised
function updatePaintMaterial() {
  const f32 = new Float32Array(materialsData, 0, 8);
  f32[3] = Number((controls as any).modelRoughness) || 0.7; // roughness
  // Wet paint reads as a smooth clear layer over the pigment: tighter
  // (low-roughness) GGX highlight + a slightly stronger dielectric sheen.
  // Dry paint is a broad, matte finish.
  f32[7] = (controls as any).paintFinish === 'Wet' ? 0.06 : 0.04; // fresnel0.y
  device.queue.writeBuffer(materialsBuffer, 0, materialsData);
}

// --- Spherical lights (xyz = world center, w = radius) ---
// Removed from the scene (2026-07-04): the studio look is environment-only;
// the lamp machinery stays for a future artistic light rig.
const MAX_LIGHTS = 8;
const LIGHTS: number[][] = [];
const lightsData = new Float32Array(MAX_LIGHTS * 4);
lightsData.set(LIGHTS.flat());
const lightsBuffer = device.createBuffer({
  label: 'Spherical Lights',
  size: lightsData.byteLength,
  usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
});
device.queue.writeBuffer(lightsBuffer, 0, lightsData);
inputViews.light_count[0] = LIGHTS.length;

// --- Illuminant wavelength-sampling LUT texture ---
const ILLUMINANT_LUT_RESOLUTION = 1024;
const illuminantTexture = device.createTexture({
  label: 'Illuminant Spectrum LUT',
  size: [ILLUMINANT_LUT_RESOLUTION, 1],
  format: 'rgba16float',
  usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
});
// The same LUT as a 3D texture: lets the shader spectrally upsample
// arbitrary RGB radiance (environment light). rgba8unorm, 33^3, rows
// padded to 256 bytes for writeTexture.
const fourierLutTexture = device.createTexture({
  label: 'Fourier LUT 3D',
  size: [fourierLut.n, fourierLut.n, fourierLut.n],
  format: 'rgba8unorm',
  dimension: '3d',
  usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
});
{
  const n = fourierLut.n;
  const rowBytes = 256 * Math.ceil((n * 4) / 256);
  const padded = new Uint8Array(rowBytes * n * n);
  for (let b = 0; b < n; b++) {
    for (let g = 0; g < n; g++) {
      for (let r = 0; r < n; r++) {
        const si = ((b * n + g) * n + r) * 3;
        const di = (b * n + g) * rowBytes + r * 4;
        padded[di] = fourierLut.data[si];
        padded[di + 1] = fourierLut.data[si + 1];
        padded[di + 2] = fourierLut.data[si + 2];
        padded[di + 3] = 255;
      }
    }
  }
  device.queue.writeTexture(
    { texture: fourierLutTexture },
    padded,
    { bytesPerRow: rowBytes, rowsPerImage: n },
    [n, n, n],
  );
}

const illuminantSampler = device.createSampler({
  addressModeU: 'clamp-to-edge',
  magFilter: 'linear',
  minFilter: 'linear',
});
// --- HDRI environment (studio_small_03, CC0 from polyhaven.com) ---
const envMap = parseEnv(await (await fetch('./studio_small_03.env')).arrayBuffer());
const environmentTexture = device.createTexture({
  label: 'Environment HDRI',
  size: [envMap.width, envMap.height],
  format: 'rgba16float',
  usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
});
device.queue.writeTexture(
  { texture: environmentTexture },
  envMap.rgba16,
  { bytesPerRow: envMap.width * 8 },
  [envMap.width, envMap.height],
);
// Luminance CDFs for importance sampling (see demo/lib/env.ts)
const envCdfs = buildEnvCdfs(envMap);
const envCdfConditionalTexture = device.createTexture({
  label: 'Env CDF conditional',
  size: [envMap.width, envMap.height],
  format: 'r32float',
  usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
});
device.queue.writeTexture(
  { texture: envCdfConditionalTexture },
  envCdfs.conditional,
  { bytesPerRow: envMap.width * 4 },
  [envMap.width, envMap.height],
);
const envCdfMarginalTexture = device.createTexture({
  label: 'Env CDF marginal',
  size: [envMap.height, 1],
  format: 'r32float',
  usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
});
device.queue.writeTexture(
  { texture: envCdfMarginalTexture },
  envCdfs.marginal,
  { bytesPerRow: envMap.height * 4 },
  [envMap.height, 1],
);

const illuminantLutCache = new Map<string, IlluminantLut>();
function setIlluminant(name: string) {
  const entry = ILLUMINANTS.find(i => i.name === name) ?? ILLUMINANTS[0];
  let lut = illuminantLutCache.get(entry.name);
  if (!lut) {
    lut = buildIlluminantLut(entry.spd, ILLUMINANT_LUT_RESOLUTION);
    illuminantLutCache.set(entry.name, lut);
  }
  device.queue.writeTexture(
    { texture: illuminantTexture },
    packFloat16(lut.rgbAndPhase),
    { bytesPerRow: ILLUMINANT_LUT_RESOLUTION * 8 },
    [ILLUMINANT_LUT_RESOLUTION, 1],
  );
}
setIlluminant(controls.illuminant);

// --- Object buffer ---
// Object struct: header(16) + transform(64) + transform_inverse(64) + motion(64) = 208 bytes
const OBJECT_STRUCT_SIZE = 208;
const OBJECT_COUNT = 2;
const objectsData = new ArrayBuffer(OBJECT_STRUCT_SIZE * OBJECT_COUNT);
const objectsBuffer = device.createBuffer({
  label: 'Objects',
  size: objectsData.byteLength,
  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
});
const objectViews = Array.from({ length: OBJECT_COUNT }, (_, index) => {
  const offset = OBJECT_STRUCT_SIZE * index;
  const view = {
    object_type: new Uint32Array(objectsData, offset + 0, 1),
    type_index: new Uint32Array(objectsData, offset + 4, 1),
    material_index: new Uint32Array(objectsData, offset + 8, 1),
    _pad: new Uint32Array(objectsData, offset + 12, 1),
    transform: new Float32Array(objectsData, offset + 16, 16),
    transform_inverse: new Float32Array(objectsData, offset + 80, 16),
    motion: new Float32Array(objectsData, offset + 144, 16),
  };
  view.motion.set(mat4.identity());
  return view;
});
// Last frame's world->index transforms, for per-object motion matrices
const prevObjectTransforms = objectViews.map(() => new Float32Array(16));

// Per-frame: motion maps a current world position on the object to its
// world position last frame: M_prev_inverse * M_current (ReSTIR reprojection)
function updateObjectMotion() {
  objectViews.forEach((view, i) => {
    if (prevObjectTransforms[i].every(v => v === 0)) {
      prevObjectTransforms[i].set(view.transform); // first frame
    }
    const prevInverse = mat4.inverse(prevObjectTransforms[i]);
    view.motion.set(mat4.multiply(prevInverse, view.transform));
    prevObjectTransforms[i].set(view.transform);
  });
  device.queue.writeBuffer(objectsBuffer, 0, objectsData);
}
// VDB object
const vdbObjectView = objectViews[0];
vdbObjectView.object_type[0] = 1; // VDB
vdbObjectView.type_index[0] = 0; // first volume
vdbObjectView.material_index[0] = 0;
// Ground plane
const groundObjectView = objectViews[1];
groundObjectView.object_type[0] = 2; // SDF
groundObjectView.type_index[0] = 0; // first sdf
groundObjectView.material_index[0] = 1;
groundObjectView.transform.set(mat4.translation(vec3.create(0, 2, 0)));
groundObjectView.transform_inverse.set(mat4.translation(vec3.create(0, -2, 0)));

// --- Sky buffer ---
const sunZenith = 30.0 * Math.PI / 180;
const sunAzimuth = 0.0;
const sunDirection = vec3.create(
  Math.sin(sunZenith) * Math.cos(sunAzimuth),
  Math.cos(sunZenith),
  - Math.sin(sunZenith) * Math.sin(sunAzimuth),
);
const skyState = createSkyState({
  elevation: 0.5 * Math.PI - sunZenith,
  turbidity: 2.0,
  albedo: [0.3, 0.3, 0.3],
})
const skyStateData = new ArrayBuffer(144);
const skyStateBuffer = device.createBuffer({
  label: 'SkyState',
  size: skyStateData.byteLength,
  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
});
const skyStateView = {
  sunDirection: new Float32Array(skyStateData, 0, 3),
  params: new Float32Array(skyStateData, 12, 27),
  skyRadiances: new Float32Array(skyStateData, 120, 3),
  solarRadiances: new Float32Array(skyStateData, 132, 3),
};
skyStateView.sunDirection.set(sunDirection);
skyStateView.params.set(skyState.params);
skyStateView.skyRadiances.set(skyState.skyRadiances);
skyStateView.solarRadiances.set(skyState.solarRadiances);
console.log("SKY STATE", skyState);
device.queue.writeBuffer(skyStateBuffer, 0, skyStateData);

// Calculate pixel radius for cone tracing: how much ray spreads per unit distance
// pixel_radius = fov_scale / resolution_height (in normalized coordinates)
function computePixelRadius(fov_y_radians: number, resolution_height: number) {
  const fov_scale = Math.tan(fov_y_radians * 0.5);
  // This gives the angular size of one pixel
  return (2.0 * fov_scale) / resolution_height;
}

// Update pixel radius (call on resize / render-scale change)
function updatePixelRadius() {
  inputViews.pixel_radius[0] = computePixelRadius(fov, renderHeight);
}

// Initialize cone constants
updatePixelRadius();

// Update VDB object transform (object 0)
function updateObjects() {
  const transformMatrix = mat4.identity();
  const [tx, ty, tz] = currentModelConfig.translation;
  const s = currentModelConfig.scale;
  mat4.translation(vec3.create(tx, ty, tz), transformMatrix);
  mat4.scale(transformMatrix, vec3.create(s, s, s), transformMatrix);

  const rotationRadians = (controls.rotation * Math.PI) / 180;
  mat4.rotateY(transformMatrix, rotationRadians, transformMatrix);
  // In-hand rotation/pan (world-space pre-image, see lib/hand.ts)
  mat4.multiply(transformMatrix, modelHand.transform, transformMatrix);

  vdbObjectView.transform.set(transformMatrix);
  vdbObjectView.transform_inverse.set(mat4.inverse(transformMatrix));

  device.queue.writeBuffer(objectsBuffer, 0, objectsData);
  device.queue.writeBuffer(skyStateBuffer, 0, skyStateData);
}

// Initial object setup
updateObjects();

// Update objects when rotation changes
rotationController.onChange(() => {
  updateObjects();
});

// Progressive accumulation: number of frames accumulated so far. Zero makes
// the shader restart the running sum, so resetting is just a counter reset.
// Once converged (MAX_ACCUM_FRAMES) the compute pass stops dispatching
// entirely until something changes — an idle, converged image is free.
const MAX_ACCUM_FRAMES = 65536;
let accumFrameIndex = 0;
let rngFrame = 0;
// Sum of the sub-pixel jitters folded into gbuffer_accum since the last
// accumulation reset; its mean rides in the uniform so the denoiser's
// world_pos reconstruction looks through the same offset as the rays did
const jitterSum = [0, 0];

// Halton radical-inverse (low-discrepancy) for stable sub-pixel jitter:
// a white-noise per-frame offset makes silhouette coverage a random binary
// each frame -> high-frequency edge flicker; a Halton sequence distributes
// the offsets evenly so the temporal average converges smoothly with far
// less edge variance (the standard TAA jitter).
function halton(index: number, base: number): number {
  let f = 1;
  let r = 0;
  let i = index;
  while (i > 0) {
    f /= base;
    r += f * (i % base);
    i = Math.floor(i / base);
  }
  return r;
}
// Last frame's world->camera matrix + camera position for ReSTIR
const prevViewMatrix = new Float32Array(16);
mat4.identity(prevViewMatrix);
const prevCameraPos = new Float32Array(3);
let shouldDispatch = true;
const prevCameraMatrix = new Float32Array(16);
let lastResetCause = 'init';
function resetAccumulation(cause = 'unknown') {
  accumFrameIndex = 0;
  lastResetCause = cause;
}
// Debug introspection for headless testing
(window as any).__dbg = () => ({ accumFrameIndex, shouldDispatch, lastResetCause });
// Headless-test hook: set a GUI control by property name (bypasses DOM)
(window as any).__set = (prop: string, value: unknown) => {
  // Prefer setValue so the control's onChange side effects fire (e.g.
  // rotation -> updateObjects), matching real GUI interaction
  const ctrl = gui.controllersRecursive().find(c => (c as any).property === prop);
  if (ctrl) {
    ctrl.setValue(value);
  } else {
    (controls as any)[prop] = value;
  }
  gui.controllersRecursive().forEach(c => c.updateDisplay());
  resetAccumulation('test-set');
  return `${prop}=${value}`;
};
// Headless-test hook: raw texel of a named denoiser texture
(window as any).__readTex = async (name: string, x: number, y: number) => {
  const map: Record<string, GPUTexture> = {
    fast0: fastTextures[0], fast1: fastTextures[1],
    illum: illumTexture, denoised: denoisedTexture,
    stab0: stabTextures[0], stab1: stabTextures[1],
    moments0: momentsTextures[0], moments1: momentsTextures[1],
  };
  const buf = device.createBuffer({ size: 256, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const enc = device.createCommandEncoder();
  enc.copyTextureToBuffer({ texture: map[name], origin: [x, y] }, { buffer: buf, bytesPerRow: 256 }, [1, 1]);
  device.queue.submit([enc.finish()]);
  await buf.mapAsync(GPUMapMode.READ);
  const f = Array.from(new Float32Array(buf.getMappedRange().slice(0, 16)));
  buf.unmap(); buf.destroy();
  return JSON.stringify(f.map(v => +v.toPrecision(5)));
};
// Headless-test hook: both G-buffer ping-pong textures at a pixel
// (normal.xyz, w = material + depth/16384; w < 0 = miss)
(window as any).__readGbuf = async (x: number, y: number) => {
  const out: number[][] = [];
  for (const tex of gbufferTextures) {
    const buf = device.createBuffer({ size: 256, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const enc = device.createCommandEncoder();
    enc.copyTextureToBuffer({ texture: tex, origin: [x, y] }, { buffer: buf, bytesPerRow: 256 }, [1, 1]);
    device.queue.submit([enc.finish()]);
    await buf.mapAsync(GPUMapMode.READ);
    out.push(Array.from(new Float32Array(buf.getMappedRange().slice(0, 16))));
    buf.unmap(); buf.destroy();
  }
  return JSON.stringify({ tex0_w: out[0][3], tex1_w: out[1][3], tex0_n: out[0].slice(0, 3), tex1_n: out[1].slice(0, 3) });
};
// Headless-test hook: denoiser history (rgb, history length) at a pixel
(window as any).__readHist = async (x: number, y: number) => {
  const P = (rngFrame - 1) & 1; // last write went to histTextures[1 - P]
  const tex = histTextures[1 - P];
  const buf = device.createBuffer({ size: 256, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const enc = device.createCommandEncoder();
  enc.copyTextureToBuffer({ texture: tex, origin: [x, y] }, { buffer: buf, bytesPerRow: 256 }, [1, 1]);
  device.queue.submit([enc.finish()]);
  await buf.mapAsync(GPUMapMode.READ);
  const f = new Float32Array(buf.getMappedRange().slice(0, 16));
  buf.unmap(); buf.destroy();
  return JSON.stringify({ rgb: [f[0], f[1], f[2]], hist_len: f[3] });
};
// Headless-test hook: linear-space accumulated mean at a pixel (rgb, count)
(window as any).__readAccum = async (x: number, y: number) => {
  const idx = (y * renderWidth + x) * 16;
  const buf = device.createBuffer({ size: 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const enc = device.createCommandEncoder();
  enc.copyBufferToBuffer(accumulationBuffer, idx, buf, 0, 16);
  device.queue.submit([enc.finish()]);
  await buf.mapAsync(GPUMapMode.READ);
  const f = new Float32Array(buf.getMappedRange().slice(0));
  buf.unmap(); buf.destroy();
  return JSON.stringify({ mean: [f[0]/f[3], f[1]/f[3], f[2]/f[3]], count: f[3] });
};
// Debug: read the accumulation buffer's isum (illum sum + count) at center
(window as any).__accSamp = async () => {
  const idx = (Math.floor(renderHeight / 2) * renderWidth + Math.floor(renderWidth / 2)) * 5;
  const buf = device.createBuffer({ size: 48, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const enc = device.createCommandEncoder();
  enc.copyBufferToBuffer(gbufferAccumBuffer, idx * 16, buf, 0, 48);
  device.queue.submit([enc.finish()]);
  await buf.mapAsync(GPUMapMode.READ);
  const f = new Float32Array(buf.getMappedRange().slice(0));
  buf.unmap(); buf.destroy();
  return JSON.stringify({ nsum_hitCount: f[3], dsum_depthSum: f[4], isum_rgb: [f[8], f[9], f[10]], isum_count: f[11], frame: (window as any).__dbg().accumFrameIndex });
};
// Headless-test hook: GPU readback of the presented (tonemapped) texture at
// render resolution — measurement without daemon screenshots or UI overlay
(window as any).__readPixels = async () => {
  const w = renderWidth, h = renderHeight;
  const bytesPerRow = Math.ceil((w * 4) / 256) * 256;
  const buf = device.createBuffer({
    size: bytesPerRow * h,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const enc = device.createCommandEncoder();
  enc.copyTextureToBuffer({ texture: raytracedTexture }, { buffer: buf, bytesPerRow }, [w, h]);
  device.queue.submit([enc.finish()]);
  await buf.mapAsync(GPUMapMode.READ);
  const data = new Uint8Array(buf.getMappedRange());
  const out = new Uint8Array(w * h * 4);
  for (let y = 0; y < h; y++) {
    out.set(data.subarray(y * bytesPerRow, y * bytesPerRow + w * 4), y * w * 4);
  }
  buf.unmap();
  buf.destroy();
  let s = '';
  for (let i = 0; i < out.length; i += 0x8000) {
    s += String.fromCharCode(...out.subarray(i, i + 0x8000));
  }
  return JSON.stringify({ w, h, b64: btoa(s) });
};

function updateInput(deltaTime: number) {
  // Update time delta
  inputViews.time_delta[0] = deltaTime;

  // Update debug flag
  // Debug Iterations moved into the Pass dropdown
  inputViews.debug_iterations[0] = controls.passMode === 'Iterations' ? 1 : 0;

  // Drag rotates/pans the model in hand; the camera keeps only zoom (dolly
  // along its fixed view direction), so backdrop and lights stay put.
  const frameInput = inputHandler();
  if (modelHand.update(deltaTime, frameInput, camera.matrix)) {
    updateObjects();
    resetAccumulation("hand-motion");
  }
  camera.update(deltaTime, {
    digital: frameInput.digital,
    analog: { ...frameInput.analog, x: 0, y: 0, panning: false },
  });
  for (let i = 0; i < 16; ++i) {
    if (camera.matrix[i] !== prevCameraMatrix[i]) {
      resetAccumulation("camera");
      break;
    }
  }
  prevCameraMatrix.set(camera.matrix);
  inputViews.camera_matrix.set(camera.matrix);

  shouldDispatch = accumFrameIndex < MAX_ACCUM_FRAMES;
  inputViews.frame_index[0] = accumFrameIndex;
  // Halton(2,3) sub-pixel jitter (cycled over 64 frames, same key the RNG
  // seed uses) + the running mean of the jitters accumulated since reset.
  // The PREVIOUS frame's mean also rides along: history texels correspond
  // to surfaces seen through last frame's mean offset, so the denoiser
  // subtracts it when fetching history (static = exact identity fetch).
  if (shouldDispatch) {
    const h = (rngFrame & 63) + 1;
    const jx = halton(h, 2) - 0.5;
    const jy = halton(h, 3) - 0.5;
    if (accumFrameIndex === 0) { jitterSum[0] = 0; jitterSum[1] = 0; }
    inputViews.jitter_mean_prev.set(inputViews.jitter_mean);
    jitterSum[0] += jx;
    jitterSum[1] += jy;
    inputViews.jitter.set([jx, jy]);
    inputViews.jitter_mean.set([
      jitterSum[0] / (accumFrameIndex + 1),
      jitterSum[1] / (accumFrameIndex + 1),
    ]);
  }
  if (shouldDispatch) { accumFrameIndex++; }
  // Monotonic RNG frame: samples stay fresh across accumulation resets
  inputViews.rng_frame[0] = rngFrame;
  // Pass debug visualizer selection; read by blit.wgsl
  const PASS = { 'Final': 0, 'Denoised': 1, 'Raw': 2, 'GBuffer Normals': 3, 'Motion Vectors': 4, 'Depth': 5, 'Iterations': 6 } as const;
  inputViews.pass_mode[0] = PASS[controls.passMode as keyof typeof PASS] ?? 0;
  // ReSTIR reprojection state: last frame's view matrix + object motion
  inputViews.prev_view.set(prevViewMatrix);
  inputViews.prev_camera_pos.set(prevCameraPos);
  mat4.inverse(camera.matrix, prevViewMatrix);
  prevCameraPos.set(camera.position.subarray(0, 3));
  prevCameraPos.set([camera.matrix[12], camera.matrix[13], camera.matrix[14]]);
  updateObjectMotion();

  if (shouldDispatch) {
    rngFrame++;
    // Global hero-wavelength offset: one stratified set per frame so all
    // reservoirs share the same lambdas (spectral reservoirs plan)
    inputViews.wavelength_u[0] = Math.random();
  }
  const ENV_INDEX = { 'Studio': 0, 'Sky': 1, 'Studio HDRI': 2 } as const;
  inputViews.environment[0] = ENV_INDEX[controls.environment];
  inputViews.max_bounces[0] = controls.maxBounces;
  // Lights and dome emit the illuminant spectrum at these intensities
  inputViews.emission_integral[0] = controls.lightIntensity;
  inputViews.dome_integral[0] = controls.domeIntensity;
  inputViews.exposure[0] = controls.exposure;
  inputViews.white_background[0] = controls.whiteBackdrop ? 1 : 0;

  // Write entire input buffer at once
  device.queue.writeBuffer(inputBuffer, 0, inputValues);
}

// Combine PicoVDB shader library, spectral module and compute shader
const combinedShader = /* wgsl */ `// Hello GPU
${PicoVDBShader}
${SpectraShader}
${ComputeShader}`

const computeShaderModule = device.createShaderModule({
  label: 'Raytracing Compute Shader',
  code: combinedShader,
});

// Check for shader compilation errors
const shaderInfo = await computeShaderModule.getCompilationInfo();
if (shaderInfo.messages.length > 0) {
  console.error('Shader compilation messages:', shaderInfo.messages);
  for (const message of shaderInfo.messages) {
    console.log(`${message.type} at line ${message.lineNum}: ${message.message}`);
    if (message.type === 'error') {
      alert(`Shader error at line ${message.lineNum}: ${message.message}`);
    }
  }
}

// --- Bind group 0: per-frame ---
const perFrameBindGroupLayout = device.createBindGroupLayout({
  label: 'Per-frame Bind Group Layout',
  entries: [
    { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
    { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
    { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
    { binding: 5, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float', viewDimension: '2d' } },
    { binding: 6, visibility: GPUShaderStage.COMPUTE, sampler: { type: 'filtering' } },
    { binding: 7, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float', viewDimension: '2d' } },
    { binding: 8, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'unfilterable-float', viewDimension: '2d' } },
    { binding: 9, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'unfilterable-float', viewDimension: '2d' } },
    { binding: 10, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float', viewDimension: '3d' } },
  ]
});

// --- Bind group 1: data ---
const dataBindGroupLayout = device.createBindGroupLayout({
  label: 'Data Bind Group Layout',
  entries: [
    { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
  ]
});

// --- Bind group 2: pass (computeMain PT) ---
const passBindGroupLayout = device.createBindGroupLayout({
  label: 'Pass Bind Group Layout',
  entries: [
    { binding: 0, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: 'rgba8unorm', viewDimension: '2d' } },
    { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    { binding: 2, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: 'rgba32float', viewDimension: '2d' } },
    { binding: 6, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: 'r32float', viewDimension: '2d' } },
    { binding: 7, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: 'rgba32float', viewDimension: '2d' } },
    { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    { binding: 3, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: 'rgba32float', viewDimension: '2d' } },
    { binding: 5, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: 'rgba32float', viewDimension: '2d' } },
    { binding: 8, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: 'r32float', viewDimension: '2d' } },
  ]
});

const computePipelineLayout = device.createPipelineLayout({
  label: 'Compute Pipeline Layout',
  bindGroupLayouts: [perFrameBindGroupLayout, dataBindGroupLayout, passBindGroupLayout],
});

const computePipeline = await device.createComputePipelineAsync({
  label: 'Compute Pipeline',
  layout: computePipelineLayout,
  compute: { module: computeShaderModule, entryPoint: 'computeMain' },
}).catch((error) => {
  console.error('Pipeline creation failed:', error);
  alert(`Pipeline error: ${error.message}`);
  throw error;
});

// --- SVGF denoiser (demo/denoise.wgsl, auto layouts per entry point) ---
const denoiseModule = device.createShaderModule({ label: 'Denoise', code: DenoiseShader });
const denoiseTemporalPipeline = await device.createComputePipelineAsync({
  label: 'Denoise Temporal', layout: 'auto',
  compute: { module: denoiseModule, entryPoint: 'temporalAccumMain' },
});
const denoiseAtrousPipeline = await device.createComputePipelineAsync({
  label: 'Denoise A-Trous', layout: 'auto',
  compute: { module: denoiseModule, entryPoint: 'atrousMain' },
});
const denoiseResolvePipeline = await device.createComputePipelineAsync({
  label: 'Denoise Resolve', layout: 'auto',
  compute: { module: denoiseModule, entryPoint: 'resolveMain' },
});
const denoiseStabilizePipeline = await device.createComputePipelineAsync({
  label: 'Denoise Stabilize (anti-lag)', layout: 'auto',
  compute: { module: denoiseModule, entryPoint: 'stabilizeMain' },
});
// One params buffer per a-trous iteration: (step, write_history)
const ATROUS_ITERATIONS = 5;
const atrousParamBuffers = Array.from({ length: ATROUS_ITERATIONS }, (_, i) => {
  const buf = device.createBuffer({
    label: `Atrous params ${i}`, size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  // write_history stays 0: feeding the FILTERED first a-trous iteration
  // back as history (SVGF's default) rectifies our wavelength-cast chroma
  // noise through the luminance-guided weights — long-history pixels
  // drifted pink. Temporal-integration-only history is convex per channel.
  device.queue.writeBuffer(buf, 0, new Uint32Array([i, 0, 0, 0]));
  return buf;
});

// P = (rngFrame - 1) & 1: gbuffer/hist/moments cur = [1 - P], prev = [P];
// a-trous ping-pong: temporal seeds [0], iterations alternate 0->1
function createDenoiseBindGroups() {
  denoiseTemporalGroups = [0, 1].map(P => device.createBindGroup({
    label: `Denoise temporal ${P}`,
    layout: denoiseTemporalPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: inputBuffer } },
      { binding: 1, resource: { buffer: objectsBuffer } },
      { binding: 3, resource: illumTexture.createView() },
      { binding: 4, resource: gbufferTextures[1 - P].createView() },
      { binding: 5, resource: gbufferTextures[P].createView() },
      { binding: 6, resource: histTextures[P].createView() },
      { binding: 7, resource: momentsTextures[P].createView() },
      { binding: 8, resource: histTextures[1 - P].createView() },
      { binding: 9, resource: momentsTextures[1 - P].createView() },
      { binding: 11, resource: atrousTextures[0].createView() },
      // ReBLUR anti-lag: last frame's stabilization confidence (in .a)
      { binding: 17, resource: stabTextures[P].createView() },
      // ReBLUR fast history ping-pong (luma EMA, ~6-frame cap)
      { binding: 20, resource: fastTextures[P].createView() },
      { binding: 21, resource: fastTextures[1 - P].createView() },
    ]
  }));
  denoiseAtrousGroups = [0, 1].map(P =>
    Array.from({ length: ATROUS_ITERATIONS }, (_, i) => device.createBindGroup({
      label: `Denoise atrous ${P}.${i}`,
      layout: denoiseAtrousPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: inputBuffer } },
        { binding: 2, resource: { buffer: atrousParamBuffers[i] } },
        { binding: 4, resource: gbufferTextures[1 - P].createView() },
        { binding: 7, resource: momentsTextures[1 - P].createView() },
        { binding: 8, resource: histTextures[1 - P].createView() },
        { binding: 10, resource: atrousTextures[i % 2].createView() },
        { binding: 11, resource: atrousTextures[(i + 1) % 2].createView() },
        { binding: 19, resource: hitdistTexture.createView() },
      ]
    })));
  denoiseResolveGroups = [0, 1].map(P => device.createBindGroup({
    label: `Denoise resolve ${P}`,
    layout: denoiseResolvePipeline.getBindGroupLayout(0),
    entries: [
      { binding: 4, resource: gbufferTextures[1 - P].createView() },
      { binding: 10, resource: atrousTextures[ATROUS_ITERATIONS % 2].createView() },
      { binding: 13, resource: specTexture.createView() },
      { binding: 14, resource: { buffer: materialAlbedoBuffer } },
      { binding: 15, resource: denoisedTexture.createView() },
    ]
  }));
  // Anti-lag stabilization: reproject the stab history (ping-pong P),
  // neighborhood-clamp, blend, tonemap to the display.
  denoiseStabilizeGroups = [0, 1].map(P => device.createBindGroup({
    label: `Denoise stabilize ${P}`,
    layout: denoiseStabilizePipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: inputBuffer } },
      { binding: 1, resource: { buffer: objectsBuffer } },
      { binding: 4, resource: gbufferTextures[1 - P].createView() },
      // Current accumulated frame count A (moments written this frame)
      // scales the anti-lag confidence's sensitivity
      { binding: 7, resource: momentsTextures[1 - P].createView() },
      // Coverage (illum alpha) + miss-side backdrop for the display AA mix
      { binding: 3, resource: illumTexture.createView() },
      { binding: 22, resource: backdropTexture.createView() },
      { binding: 12, resource: raytracedTexture.createView() },
      { binding: 16, resource: denoisedTexture.createView() },
      { binding: 17, resource: stabTextures[P].createView() },
      { binding: 18, resource: stabTextures[1 - P].createView() },
    ]
  }));
}

console.log('Pipeline created.');


const computePassDescriptor: GPUComputePassDescriptor = {
  label: "Compute pass",
}
timestampQueryManager.addTimestampWrite(computePassDescriptor);

// Initial creation of GPU resources (after all dependencies are defined)
createGPUResources();

// Load initial model (from URL param or first in list)
await loadModel(initialModel);

// Wire up model switching — update URL and load
modelController.onChange(async (name: string) => {
  const config = models.find(m => m.name === name)!;
  const filename = config.url.split('/').pop()!;
  const url = new URL(window.location.href);
  url.searchParams.set('model', filename);
  history.replaceState(null, '', url);
  await loadModel(config);
});

// Any GUI change invalidates the accumulated image
gui.onChange((event) => {
  resetAccumulation("gui");
  if (event.property === 'illuminant') {
    setIlluminant(controls.illuminant);
  } else if (event.property === 'environment') {
    // Sensible exposure defaults per environment (sky radiances are huge)
    controls.exposure = controls.environment === 'Sky' ? 0.05 : 1.0;
    gui.controllersRecursive().forEach(c => c.updateDisplay());
  } else if (event.property === 'renderScale' || event.property === 'giScale') {
    createGPUResources();
    updatePixelRadius();
  } else if (event.property === 'modelRoughness') {
    updatePaintMaterial();
  } else if (event.property === 'paintFinish') {
    // Preset roughness for the finish (the slider stays an override)
    controls.modelRoughness = controls.paintFinish === 'Wet' ? 0.12 : 0.7;
    gui.controllersRecursive().forEach(c => c.updateDisplay());
    updatePaintMaterial();
  }
});

const colorAttachment: GPURenderPassColorAttachment = {
  view: context.getCurrentTexture().createView(), // Assigned on render
  clearValue: { r: 0, g: 0, b: 0, a: 1 },
  loadOp: 'clear',
  storeOp: 'store',
}

const renderPassDescriptor: GPURenderPassDescriptor = {
  label: "Display pass",
  colorAttachments: [colorAttachment],
}

let lastFrameMS = (performance || Date).now();
function requestFrame() {
  if (!context) {
    throw new Error("No context found.");
  }
  const beginTime = stats.begin();
  const deltaTime = (beginTime - lastFrameMS) / 1000;
  lastFrameMS = beginTime;

  // Update uniforms first: queue writes land before this frame's passes
  updateInput(deltaTime);

  // Show accumulation progress so convergence state is always visible
  if (accumFrameIndex % 15 === 0 && modelInfoText) {
    infoTextElement.textContent = `${modelInfoText}\nSamples: ${accumFrameIndex}`;
  }

  const encoder = device.createCommandEncoder({ label: "Command Encoder" });

  // Skip path tracing entirely once the image has converged
  if (shouldDispatch) {
    const wgX = Math.ceil(renderWidth / 8);
    const wgY = Math.ceil(renderHeight / 8);
    const computePass = encoder.beginComputePass(computePassDescriptor);
    computePass.setBindGroup(0, perFrameBindGroup);
    computePass.setBindGroup(1, dataBindGroup);
    // rngFrame was already advanced for this frame in updateInput
    const p = (rngFrame - 1) & 1;
    computePass.setBindGroup(2, passBindGroups[p]);
    // Plain path tracer: writes G-buffer + depth + raw (demodulated) illum
    // + accumulation; the denoiser + Pass visualizer consume those.
    computePass.setPipeline(computePipeline);
    computePass.dispatchWorkgroups(wgX, wgY, 1);
    // Denoiser runs for the Final/Denoised passes; other passes visualize
    // upstream buffers directly (and Iterations needs the raw heatmap)
    if (controls.denoise && (controls.passMode === 'Final' || controls.passMode === 'Denoised')) {
      const P = (rngFrame - 1) & 1;
      computePass.setPipeline(denoiseTemporalPipeline);
      computePass.setBindGroup(0, denoiseTemporalGroups[P]);
      computePass.dispatchWorkgroups(wgX, wgY, 1);
      computePass.setPipeline(denoiseAtrousPipeline);
      for (let i = 0; i < ATROUS_ITERATIONS; i++) {
        computePass.setBindGroup(0, denoiseAtrousGroups[P][i]);
        computePass.dispatchWorkgroups(wgX, wgY, 1);
      }
      computePass.setPipeline(denoiseResolvePipeline);
      computePass.setBindGroup(0, denoiseResolveGroups[P]);
      computePass.dispatchWorkgroups(wgX, wgY, 1);
      computePass.setPipeline(denoiseStabilizePipeline);
      computePass.setBindGroup(0, denoiseStabilizeGroups[P]);
      computePass.dispatchWorkgroups(wgX, wgY, 1);
    }
    computePass.end();
  }

  // Start a display pass.
  colorAttachment.view = context.getCurrentTexture().createView();
  const displayPass = encoder.beginRenderPass(renderPassDescriptor);
  displayPass.setPipeline(displayPipeline);
  displayPass.setVertexBuffer(0, vertexBuffer);
  displayPass.setBindGroup(0, displayBindGroups[(rngFrame - 1) & 1]);
  displayPass.draw(3, 1, 0, 0);
  displayPass.end();

  // Resolve timestamp queries, so that their result is available in
  // a GPU-side buffer.
  timestampQueryManager.resolve(encoder);

  // Finish the command buffer and immediately submit it.
  device.queue.submit([encoder.finish()]);

  // Try to download the time stamp.
  timestampQueryManager.tryInitiateTimestampDownload();
  stats.end();
}

// Pause/resume functionality. Use requestAnimationFrame for optimal frame timing.
let animationId: number | null = null;

// Headless-test hook: drive frames directly (the rAF loop occasionally
// wedges machine-wide; this keeps GPU testing possible regardless)
(window as any).__tick = (n: number) => {
  for (let i = 0; i < (n || 1); i++) { requestFrame(); }
  return accumFrameIndex;
};

function renderLoop() {
  (window as any).__loop = ((window as any).__loop ?? 0) + 1;
  if (animationId === null) return;
  requestFrame();
  animationId = requestAnimationFrame(renderLoop);
}

function startRenderLoop() {
  (window as any).__started = true;
  animationId = requestAnimationFrame(renderLoop);
}

function stopRenderLoop() {
  if (animationId !== null) {
    cancelAnimationFrame(animationId);
    animationId = null;
  }
}

pauseController.onChange((paused: boolean) => {
  if (paused) {
    stopRenderLoop();
  } else {
    startRenderLoop();
  }
})

// Start the render loop
startRenderLoop();
