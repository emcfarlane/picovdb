import { vec3, mat4 } from 'wgpu-matrix';
import DisplayShader from "./blit.wgsl";
import ComputeShader from "./compute.wgsl";
import PicoVDBShader from "./../picovdb.wgsl";
import { loadPicoVDB } from './lib/loader';
import { createOrbitCamera } from './lib/camera';
import { createInputHandler } from "./lib/input";
import { controls, pauseController, highDPIController, rotationController } from './lib/gui';
import { TimestampQueryManager } from './lib/TimestampQueryManager';
import { Stats } from './lib/Stats';

const canvas = document.getElementById("canvas") as HTMLCanvasElement;
const infoTextElement = document.getElementById("info-text")!;

if (!canvas) {
  throw new Error("No canvas found.");
}
if (!navigator.gpu) {
  throw new Error("WebGPU not supported on this browser.");
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
let raytracedTexture: GPUTexture;
let displayBindGroup: GPUBindGroup;
let perFrameBindGroup: GPUBindGroup;
let dataBindGroup: GPUBindGroup;
let passBindGroup: GPUBindGroup;

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

const device = await adapter.requestDevice({ requiredFeatures: requiredFeatures });
device.addEventListener('uncapturederror', event => {
  console.log(event.error);
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
    ]
  });

  // Bind group 1: data
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

// Load PicoVDB data
infoTextElement.textContent = "Loading bunny.pvdb.gz...";
const picoVDBFile = await loadPicoVDB('./bunny.pvdb.gz');

const gridsBuffer = device.createBuffer({
  label: 'PicoVDB Grids',
  size: picoVDBFile.gridsBuffer.byteLength,
  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
});
device.queue.writeBuffer(gridsBuffer, 0, picoVDBFile.gridsBuffer);

const rootsBuffer = device.createBuffer({
  label: 'PicoVDB Roots',
  size: picoVDBFile.rootsBuffer.byteLength,
  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
});
device.queue.writeBuffer(rootsBuffer, 0, picoVDBFile.rootsBuffer);

const uppersBuffer = device.createBuffer({
  label: 'PicoVDB Uppers',
  size: picoVDBFile.uppersBuffer.byteLength,
  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
});
device.queue.writeBuffer(uppersBuffer, 0, picoVDBFile.uppersBuffer);

const lowersBuffer = device.createBuffer({
  label: 'PicoVDB Lowers',
  size: picoVDBFile.lowersBuffer.byteLength,
  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
});
device.queue.writeBuffer(lowersBuffer, 0, picoVDBFile.lowersBuffer);

const leavesBuffer = device.createBuffer({
  label: 'PicoVDB Leaves',
  size: picoVDBFile.leavesBuffer.byteLength,
  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
});
device.queue.writeBuffer(leavesBuffer, 0, picoVDBFile.leavesBuffer);

const dataBuffer = device.createBuffer({
  label: 'PicoVDB Data',
  size: picoVDBFile.dataBuffer.byteLength,
  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
});
device.queue.writeBuffer(dataBuffer, 0, picoVDBFile.dataBuffer);

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
// Bunny
const bunnyObjectView = objectViews[0];
bunnyObjectView.object_type[0] = 1; // VDB
bunnyObjectView.type_index[0] = 0; // first volume
bunnyObjectView.material_index[0] = 0;
// Ground plane
const groundObjectView = objectViews[1];
groundObjectView.object_type[0] = 2; // SDF
groundObjectView.type_index[0] = 0; // first sdf
groundObjectView.material_index[0] = 1;
groundObjectView.transform.set(mat4.translation(vec3.create(0, 2, 0)));
groundObjectView.transform_inverse.set(mat4.translation(vec3.create(0, -2, 0)));


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
  mat4.translation(vec3.create(-40, 240, 0), transformMatrix);
  mat4.scale(transformMatrix, vec3.create(120, 120, 120), transformMatrix);

  const rotationRadians = (controls.bunnyRotation * Math.PI) / 180;
  mat4.rotateY(transformMatrix, rotationRadians, transformMatrix);

  bunnyObjectView.transform.set(transformMatrix);
  bunnyObjectView.transform_inverse.set(mat4.inverse(transformMatrix));

  device.queue.writeBuffer(objectsBuffer, 0, objectsData);
}

// Initial object setup
updateObjects();

// Update objects when rotation changes
rotationController.onChange(() => {
  updateObjects();
});

// Update info display
const sizeMB = (picoVDBFile.getSize() / 1024 / 1024).toFixed(1);
const grid = picoVDBFile.getGrid(0);
const bboxSize = [
  (grid.indexBoundsMax[0] - grid.indexBoundsMin[0]),
  (grid.indexBoundsMax[1] - grid.indexBoundsMin[1]),
  (grid.indexBoundsMax[2] - grid.indexBoundsMin[2])
]
infoTextElement.textContent = `PicoVDB
bunny.pvdb ${sizeMB}MB
Grid: ${bboxSize[0]} × ${bboxSize[1]} × ${bboxSize[2]} units
Voxels: ${picoVDBFile.getVoxelCount()}`;
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

// Pause/resume functionality. UserequestAnimationFrame for optimal frame timing.
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
