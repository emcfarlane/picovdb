import { vec3, mat4 } from 'wgpu-matrix';
import DisplayShader from "./blit.wgsl" with { type: "text" };
import ComputeShader from "./compute.wgsl" with { type: "text" };
import { picovdbWgsl as PicoVDBShader } from "picovdb/ts/shaders.ts";
import { fetchPicoVDB, type PicoVDBFile, GRID_TYPE_SDF_FLOAT, PICOVDB_GRID_SIZE } from '../ts/picovdb.ts';
import { gridLimits } from '../ts/gpu/device.ts';
import { Space, Op, type Solid, type PicoVDBTree } from '../ts/model.ts';
import { createOrbitCamera } from './lib/camera.ts';
import { createInputHandler } from "./lib/input.ts";
import { initGUI } from './lib/gui.ts';
import type { ModelConfig } from './lib/gui.ts';

const MODEL_BASE = './models/';
const models: ModelConfig[] = [
  { name: 'Bunny', url: `${MODEL_BASE}bunny.pvdb.gz`, translation: [-40, 240, 0], scale: 120 },
  { name: 'Bunny u8', url: `${MODEL_BASE}bunny.u8.pvdb.gz`, translation: [-40, 240, 0], scale: 120 },
  { name: 'Dragon u8', url: `${MODEL_BASE}dragon.u8.pvdb.gz`, translation: [0, 80, 0], scale: 240 },
  //{ name: 'Smoke', url: `${MODEL_BASE}smoke.pvdb.gz`, translation: [0, 0, 0], scale: 60 },
  { name: 'Sphere', url: `${MODEL_BASE}sphere.pvdb.gz`, translation: [0, 0, 0], scale: 30 },
  //{ name: 'Skeleton', url: `${MODEL_BASE}skeleton.pvdb.gz`, translation: [0, 240, 0], scale: 120 },
];

const modelParam = new URLSearchParams(globalThis.location.search).get('model');
const initialModel = models.find(m => m.url.endsWith('/' + modelParam)) ?? models[0];

const { controls, modelController, pauseController, highDPIController, rotationController } = initGUI(models, initialModel.name);
import { createSkyState } from "./lib/hw_skymodel.ts";
import { computeSkyIrradianceSH } from "./lib/sky_irradiance.ts";
import { TimestampQueryManager } from './lib/TimestampQueryManager.ts';
import { Stats } from './lib/Stats.ts';

const canvas = document.getElementById("canvas") as HTMLCanvasElement;
const infoTextElement = document.getElementById("info-text")!;

if (!canvas) {
  throw new Error("No canvas found.");
}
if (!navigator.gpu) {
  const isInsecure = globalThis.isSecureContext === false;
  throw new Error(
    isInsecure
      ? "WebGPU requires a secure context (HTTPS or localhost). Current origin is not secure."
      : "WebGPU not supported on this browser."
  );
}
console.log("WebGPU is supported!");

// featureLevel is newer than Deno's bundled WebGPU types.
const adapter = await navigator.gpu.requestAdapter({
  featureLevel: 'compatibility',
} as unknown as GPURequestAdapterOptions);
if (!adapter) {
  throw new Error("No appropriate GPUAdapter found.");
}

let width = canvas.width;
let height = canvas.height;
let raytracedTexture: GPUTexture;
let displayBindGroup: GPUBindGroup;
let perFrameBindGroup: GPUBindGroup;
let dataBindGroup: GPUBindGroup;
let passBindGroup: GPUBindGroup;

// Set canvas to fullscreen size and recreate GPU resources
function resizeCanvas() {
  const pixelRatio = controls.highDPI ? globalThis.devicePixelRatio : 1.0;
  canvas.width = globalThis.innerWidth * pixelRatio;
  canvas.height = globalThis.innerHeight * pixelRatio;
  width = canvas.width;
  height = canvas.height;

  // Will recreate GPU resources after they're initially created
  if (raytracedTexture) {
    createGPUResources();
    updatePixelRadius();
  }
}

resizeCanvas();
globalThis.addEventListener('resize', resizeCanvas);

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

const device = await adapter.requestDevice({ requiredFeatures: requiredFeatures, requiredLimits: gridLimits(adapter) });
device.addEventListener('uncapturederror', event => {
  console.log(event.error);
});

// The DOM lib doesn't know the "webgpu" context id.
const context = canvas.getContext("webgpu") as unknown as GPUCanvasContext | null;
if (!context) {
  throw new Error("No context found.");
}

const stats = new Stats();
const gpuPanel = stats.addPanel(new Stats.Panel('GPU', '#ff8', '#221'));
document.body.appendChild(stats.dom);

// GPU-side timer and the CPU-side counter where we accumulate statistics:
// NB: Look for 'timestampQueryManager' in this file to locate parts of this
// snippets that are related to timestamps. Most of the logic is in
// TimestampQueryManager.ts.
const timestampQueryManager = new TimestampQueryManager(device, (elapsedNs) => {
  // Convert from nanoseconds to milliseconds:
  const elapsedMs = Number(elapsedNs) * 1e-6;
  gpuPanel.update(elapsedMs, 16); // 16ms = 60fps target
});

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
let currentFile: PicoVDBFile | null = null;

// Create size-dependent GPU resources
function createGPUResources() {
  if (raytracedTexture) { raytracedTexture.destroy(); }

  raytracedTexture = device.createTexture({
    size: [width, height],
    format: 'rgba8unorm',
    usage: GPUTextureUsage.STORAGE_BINDING |
      GPUTextureUsage.TEXTURE_BINDING |
      GPUTextureUsage.COPY_SRC
  });

  displayBindGroup = device.createBindGroup({
    label: "Display bind group",
    layout: displayPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: raytracedTexture.createView() },
      { binding: 1, resource: displaySampler },
    ]
  });

  // Bind group 0: per-frame
  perFrameBindGroup = device.createBindGroup({
    label: 'Per-frame bind group',
    layout: perFrameBindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: inputBuffer } },
      { binding: 1, resource: { buffer: objectsBuffer } },
      { binding: 2, resource: { buffer: skyStateBuffer } },
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

  // Bind group 2: pass
  passBindGroup = device.createBindGroup({
    label: 'Pass bind group',
    layout: passBindGroupLayout,
    entries: [
      { binding: 0, resource: raytracedTexture.createView() },
    ]
  });
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

function uploadBytes(label: string, bytes: Uint8Array<ArrayBuffer>): GPUBuffer {
  const buffer = device.createBuffer({ label, size: bytes.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(buffer, 0, bytes);
  return buffer;
}

// Swaps in a set of picovdb node buffers and rebuilds the data bind group.
function setGridBuffers(b: { grids: GPUBuffer; roots: GPUBuffer; uppers: GPUBuffer; lowers: GPUBuffer; leaves: GPUBuffer; data: GPUBuffer }) {
  if (gridsBuffer) {
    gridsBuffer.destroy();
    rootsBuffer.destroy();
    uppersBuffer.destroy();
    lowersBuffer.destroy();
    leavesBuffer.destroy();
    dataBuffer.destroy();
  }
  gridsBuffer = b.grids;
  rootsBuffer = b.roots;
  uppersBuffer = b.uppers;
  lowersBuffer = b.lowers;
  leavesBuffer = b.leaves;
  dataBuffer = b.data;
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

// Renders an emitted tree in place of the loaded model, keeping its transform.
function showTree(tree: PicoVDBTree, name: string) {
  const grid = new ArrayBuffer(PICOVDB_GRID_SIZE);
  new Uint32Array(grid, 0, 8).set([0, 0, 0, 0, 0, tree.dataElemCount, GRID_TYPE_SDF_FLOAT, 0]);
  new Int32Array(grid, 32, 3).set(tree.indexBoundsMin);
  new Int32Array(grid, 48, 3).set(tree.indexBoundsMax);
  setGridBuffers({
    grids: uploadBytes('PicoVDB Grids', new Uint8Array(grid)),
    roots: tree.roots,
    uppers: tree.uppers,
    lowers: tree.lowers,
    leaves: tree.leaves,
    data: tree.data,
  });
  updateObjects();
  const size = tree.indexBoundsMax.map((v, a) => v - tree.indexBoundsMin[a]);
  infoTextElement.textContent = `PicoVDB
${name}
Grid: ${size[0]} × ${size[1]} × ${size[2]} units
Voxels: ${tree.activeVoxels}`;
}

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

  setGridBuffers({
    grids: uploadBytes('PicoVDB Grids', picoVDBFile.gridsBuffer),
    roots: uploadBytes('PicoVDB Roots', picoVDBFile.rootsBuffer),
    uppers: uploadBytes('PicoVDB Uppers', picoVDBFile.uppersBuffer),
    lowers: uploadBytes('PicoVDB Lowers', picoVDBFile.lowersBuffer),
    leaves: uploadBytes('PicoVDB Leaves', picoVDBFile.leavesBuffer),
    data: uploadBytes('PicoVDB Data', picoVDBFile.dataBuffer),
  });
  currentModelConfig = config;
  currentFile = picoVDBFile;
  sceneSolid?.destroy();
  sceneSolid = null;

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
  infoTextElement.textContent = `PicoVDB
${config.name} ${sizeMB}MB
Grid: ${bboxSize[0]} × ${bboxSize[1]} × ${bboxSize[2]} units
Voxels: ${picoVDBFile.getVoxelCount()}`;
}

const fov = (2 * Math.PI) / 5; // 72 degrees
const fovScaled = Math.tan(fov / 2);
const initialCameraPosition = vec3.create(3, 2, 5);
const initialCameraTarget = vec3.create(0, 0, 0);
let camera = createOrbitCamera({
  position: initialCameraPosition,
  target: initialCameraTarget,
});

controls.resetCamera = () => {
  camera = createOrbitCamera({
    position: initialCameraPosition,
    target: initialCameraTarget,
  });
};


// Input uniform: camera_matrix(64) + fov_scale(4) + time_delta(4) + pixel_radius(4) + debug_iterations(4) = 80 bytes
const inputValues = new ArrayBuffer(80);
const inputViews = {
  camera_matrix: new Float32Array(inputValues, 0, 16),
  fov_scale: new Float32Array(inputValues, 64, 1),
  time_delta: new Float32Array(inputValues, 68, 1),
  pixel_radius: new Float32Array(inputValues, 72, 1),
  debug_iterations: new Uint32Array(inputValues, 76, 1),
};
const inputBuffer = device.createBuffer({
  label: 'Input Uniforms',
  size: inputValues.byteLength,
  usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
});
inputViews.fov_scale[0] = fovScaled;

// --- Object buffer ---
// Object struct: object_type(4) + type_id(4) + material_id(4) + _pad(4) + transform(64) + transform_inverse(64) = 144 bytes
const OBJECT_STRUCT_SIZE = 144;
const OBJECT_COUNT = 2;
const objectsData = new ArrayBuffer(OBJECT_STRUCT_SIZE * OBJECT_COUNT);
const objectsBuffer = device.createBuffer({
  label: 'Objects',
  size: objectsData.byteLength,
  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
});
new Array(27).slice()
const objectViews = [];
for (let index = 0; index < OBJECT_COUNT; index++) {
  const offset = OBJECT_STRUCT_SIZE * index;
  objectViews.push({
    object_type: new Uint32Array(objectsData, offset + 0, 1),
    type_index: new Uint32Array(objectsData, offset + 4, 1),
    material_index: new Uint32Array(objectsData, offset + 8, 1),
    _pad: new Uint32Array(objectsData, offset + 12, 1),
    transform: new Float32Array(objectsData, offset + 16, 16),
    transform_inverse: new Float32Array(objectsData, offset + 80, 16),
  });
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
const skyStateData = new ArrayBuffer(288);
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
  irradianceSH: new Float32Array(skyStateData, 144, 36),
};
skyStateView.sunDirection.set(sunDirection);
skyStateView.params.set(skyState.params);
skyStateView.skyRadiances.set(skyState.skyRadiances);
skyStateView.solarRadiances.set(skyState.solarRadiances);
skyStateView.irradianceSH.set(computeSkyIrradianceSH(skyState, sunDirection));
console.log("SKY STATE", skyState);
device.queue.writeBuffer(skyStateBuffer, 0, skyStateData);

// Calculate pixel radius for cone tracing: how much ray spreads per unit distance
// pixel_radius = fov_scale / resolution_height (in normalized coordinates)
function computePixelRadius(fov_y_radians: number, resolution_height: number) {
  const fov_scale = Math.tan(fov_y_radians * 0.5);
  // This gives the angular size of one pixel
  return (2.0 * fov_scale) / resolution_height;
}

// Update pixel radius (call on resize)
function updatePixelRadius() {
  inputViews.pixel_radius[0] = computePixelRadius(fov, height);
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

function updateInput(deltaTime: number) {
  // Update time delta
  inputViews.time_delta[0] = deltaTime;

  // Update debug flag
  inputViews.debug_iterations[0] = controls.debugIterations ? 1 : 0;

  // Update camera
  camera.update(deltaTime, inputHandler());
  inputViews.camera_matrix.set(camera.matrix);

  // Write entire input buffer at once
  device.queue.writeBuffer(inputBuffer, 0, inputValues);
}

// Combine PicoVDB shader library with compute shader
const combinedShader = /* wgsl */ `// Hello GPU
${PicoVDBShader}
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

// --- Bind group 2: pass ---
const passBindGroupLayout = device.createBindGroupLayout({
  label: 'Pass Bind Group Layout',
  entries: [
    {
      binding: 0, visibility: GPUShaderStage.COMPUTE,
      storageTexture: { access: 'write-only', format: 'rgba8unorm', viewDimension: '2d' },
    },
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

console.log('Pipeline created.');

const computePassDescriptor: GPUComputePassDescriptor = {
  label: "Compute pass",
}
timestampQueryManager.addTimestampWrite(computePassDescriptor);

// Initial creation of GPU resources (after all dependencies are defined)
createGPUResources();

// Load initial model (from URL param or first in list)
await loadModel(initialModel);

// Modelling console. Try in devtools:
//   scene.solid = scene.solid.offset(2).subtract(scene.solid);
globalThis.space = new Space(device);
let sceneSolid: Solid | null = null;
globalThis.scene = {
  /** The rendered model as a solid, loaded from the current file on first read. */
  get solid(): Solid {
    if (!currentFile) throw new Error('no model loaded');
    return sceneSolid ??= space.fromPvdb(currentFile);
  },
  /** Renders a solid, or the result of an op, in place of the loaded model. */
  set solid(value: Solid | Op) {
    (async () => {
      const t0 = performance.now();
      const solid = await value;
      const tree = await solid.toTree();
      showTree(tree, value instanceof Op ? 'op' : 'solid');
      if (sceneSolid && sceneSolid !== solid) sceneSolid.destroy();
      sceneSolid = solid;
      console.log(`scene: ${tree.leafCount} leaves, ${tree.activeVoxels} active, ${tree.surfaceVoxels} surface in ${(performance.now() - t0).toFixed(0)} ms`);
    })().catch(console.error);
  },
};
declare global {
  var space: Space;
  var scene: { solid: Solid | Op };
}

// Wire up model switching — update URL and load
modelController.onChange(async (name: string) => {
  const config = models.find(m => m.name === name)!;
  const filename = config.url.split('/').pop()!;
  const url = new URL(globalThis.location.href);
  url.searchParams.set('model', filename);
  history.replaceState(null, '', url);
  await loadModel(config);
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

  const encoder = device.createCommandEncoder({ label: "Command Encoder" });

  const computePass = encoder.beginComputePass(computePassDescriptor);
  computePass.setPipeline(computePipeline);
  computePass.setBindGroup(0, perFrameBindGroup);
  computePass.setBindGroup(1, dataBindGroup);
  computePass.setBindGroup(2, passBindGroup);
  computePass.dispatchWorkgroups(Math.ceil(width / 8), Math.ceil(height / 8), 1);
  computePass.end();

  // Start a display pass.
  colorAttachment.view = context.getCurrentTexture().createView();
  const displayPass = encoder.beginRenderPass(renderPassDescriptor);
  displayPass.setPipeline(displayPipeline);
  displayPass.setVertexBuffer(0, vertexBuffer);
  displayPass.setBindGroup(0, displayBindGroup);
  displayPass.draw(3, 1, 0, 0);
  displayPass.end();

  updateInput(deltaTime);

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

function renderLoop() {
  if (animationId === null) return;
  requestFrame();
  animationId = requestAnimationFrame(renderLoop);
}

function startRenderLoop() {
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
