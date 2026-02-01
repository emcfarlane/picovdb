// node_modules/wgpu-matrix/dist/3.x/wgpu-matrix.module.js
function wrapConstructor(OriginalConstructor, modifier) {
  return class extends OriginalConstructor {
    constructor(...args) {
      super(...args);
      modifier(this);
    }
  };
}
var ZeroArray = wrapConstructor(Array, (a) => a.fill(0));
var EPSILON = 1e-6;
function getAPIImpl$5(Ctor) {
  function create(x = 0, y = 0) {
    const newDst = new Ctor(2);
    if (x !== void 0) {
      newDst[0] = x;
      if (y !== void 0) {
        newDst[1] = y;
      }
    }
    return newDst;
  }
  const fromValues = create;
  function set(x, y, dst) {
    const newDst = dst ?? new Ctor(2);
    newDst[0] = x;
    newDst[1] = y;
    return newDst;
  }
  function ceil(v, dst) {
    const newDst = dst ?? new Ctor(2);
    newDst[0] = Math.ceil(v[0]);
    newDst[1] = Math.ceil(v[1]);
    return newDst;
  }
  function floor(v, dst) {
    const newDst = dst ?? new Ctor(2);
    newDst[0] = Math.floor(v[0]);
    newDst[1] = Math.floor(v[1]);
    return newDst;
  }
  function round(v, dst) {
    const newDst = dst ?? new Ctor(2);
    newDst[0] = Math.round(v[0]);
    newDst[1] = Math.round(v[1]);
    return newDst;
  }
  function clamp2(v, min2 = 0, max2 = 1, dst) {
    const newDst = dst ?? new Ctor(2);
    newDst[0] = Math.min(max2, Math.max(min2, v[0]));
    newDst[1] = Math.min(max2, Math.max(min2, v[1]));
    return newDst;
  }
  function add(a, b, dst) {
    const newDst = dst ?? new Ctor(2);
    newDst[0] = a[0] + b[0];
    newDst[1] = a[1] + b[1];
    return newDst;
  }
  function addScaled(a, b, scale2, dst) {
    const newDst = dst ?? new Ctor(2);
    newDst[0] = a[0] + b[0] * scale2;
    newDst[1] = a[1] + b[1] * scale2;
    return newDst;
  }
  function angle(a, b) {
    const ax = a[0];
    const ay = a[1];
    const bx = b[0];
    const by = b[1];
    const mag1 = Math.sqrt(ax * ax + ay * ay);
    const mag2 = Math.sqrt(bx * bx + by * by);
    const mag = mag1 * mag2;
    const cosine = mag && dot(a, b) / mag;
    return Math.acos(cosine);
  }
  function subtract(a, b, dst) {
    const newDst = dst ?? new Ctor(2);
    newDst[0] = a[0] - b[0];
    newDst[1] = a[1] - b[1];
    return newDst;
  }
  const sub = subtract;
  function equalsApproximately(a, b) {
    return Math.abs(a[0] - b[0]) < EPSILON && Math.abs(a[1] - b[1]) < EPSILON;
  }
  function equals(a, b) {
    return a[0] === b[0] && a[1] === b[1];
  }
  function lerp2(a, b, t, dst) {
    const newDst = dst ?? new Ctor(2);
    newDst[0] = a[0] + t * (b[0] - a[0]);
    newDst[1] = a[1] + t * (b[1] - a[1]);
    return newDst;
  }
  function lerpV(a, b, t, dst) {
    const newDst = dst ?? new Ctor(2);
    newDst[0] = a[0] + t[0] * (b[0] - a[0]);
    newDst[1] = a[1] + t[1] * (b[1] - a[1]);
    return newDst;
  }
  function max(a, b, dst) {
    const newDst = dst ?? new Ctor(2);
    newDst[0] = Math.max(a[0], b[0]);
    newDst[1] = Math.max(a[1], b[1]);
    return newDst;
  }
  function min(a, b, dst) {
    const newDst = dst ?? new Ctor(2);
    newDst[0] = Math.min(a[0], b[0]);
    newDst[1] = Math.min(a[1], b[1]);
    return newDst;
  }
  function mulScalar(v, k, dst) {
    const newDst = dst ?? new Ctor(2);
    newDst[0] = v[0] * k;
    newDst[1] = v[1] * k;
    return newDst;
  }
  const scale = mulScalar;
  function divScalar(v, k, dst) {
    const newDst = dst ?? new Ctor(2);
    newDst[0] = v[0] / k;
    newDst[1] = v[1] / k;
    return newDst;
  }
  function inverse(v, dst) {
    const newDst = dst ?? new Ctor(2);
    newDst[0] = 1 / v[0];
    newDst[1] = 1 / v[1];
    return newDst;
  }
  const invert = inverse;
  function cross(a, b, dst) {
    const newDst = dst ?? new Ctor(3);
    const z = a[0] * b[1] - a[1] * b[0];
    newDst[0] = 0;
    newDst[1] = 0;
    newDst[2] = z;
    return newDst;
  }
  function dot(a, b) {
    return a[0] * b[0] + a[1] * b[1];
  }
  function length(v) {
    const v0 = v[0];
    const v1 = v[1];
    return Math.sqrt(v0 * v0 + v1 * v1);
  }
  const len = length;
  function lengthSq(v) {
    const v0 = v[0];
    const v1 = v[1];
    return v0 * v0 + v1 * v1;
  }
  const lenSq = lengthSq;
  function distance(a, b) {
    const dx = a[0] - b[0];
    const dy = a[1] - b[1];
    return Math.sqrt(dx * dx + dy * dy);
  }
  const dist = distance;
  function distanceSq(a, b) {
    const dx = a[0] - b[0];
    const dy = a[1] - b[1];
    return dx * dx + dy * dy;
  }
  const distSq = distanceSq;
  function normalize(v, dst) {
    const newDst = dst ?? new Ctor(2);
    const v0 = v[0];
    const v1 = v[1];
    const len2 = Math.sqrt(v0 * v0 + v1 * v1);
    if (len2 > 1e-5) {
      newDst[0] = v0 / len2;
      newDst[1] = v1 / len2;
    } else {
      newDst[0] = 0;
      newDst[1] = 0;
    }
    return newDst;
  }
  function negate(v, dst) {
    const newDst = dst ?? new Ctor(2);
    newDst[0] = -v[0];
    newDst[1] = -v[1];
    return newDst;
  }
  function copy(v, dst) {
    const newDst = dst ?? new Ctor(2);
    newDst[0] = v[0];
    newDst[1] = v[1];
    return newDst;
  }
  const clone = copy;
  function multiply(a, b, dst) {
    const newDst = dst ?? new Ctor(2);
    newDst[0] = a[0] * b[0];
    newDst[1] = a[1] * b[1];
    return newDst;
  }
  const mul = multiply;
  function divide(a, b, dst) {
    const newDst = dst ?? new Ctor(2);
    newDst[0] = a[0] / b[0];
    newDst[1] = a[1] / b[1];
    return newDst;
  }
  const div = divide;
  function random(scale2 = 1, dst) {
    const newDst = dst ?? new Ctor(2);
    const angle2 = Math.random() * 2 * Math.PI;
    newDst[0] = Math.cos(angle2) * scale2;
    newDst[1] = Math.sin(angle2) * scale2;
    return newDst;
  }
  function zero(dst) {
    const newDst = dst ?? new Ctor(2);
    newDst[0] = 0;
    newDst[1] = 0;
    return newDst;
  }
  function transformMat4(v, m, dst) {
    const newDst = dst ?? new Ctor(2);
    const x = v[0];
    const y = v[1];
    newDst[0] = x * m[0] + y * m[4] + m[12];
    newDst[1] = x * m[1] + y * m[5] + m[13];
    return newDst;
  }
  function transformMat3(v, m, dst) {
    const newDst = dst ?? new Ctor(2);
    const x = v[0];
    const y = v[1];
    newDst[0] = m[0] * x + m[4] * y + m[8];
    newDst[1] = m[1] * x + m[5] * y + m[9];
    return newDst;
  }
  function rotate2(a, b, rad, dst) {
    const newDst = dst ?? new Ctor(2);
    const p0 = a[0] - b[0];
    const p1 = a[1] - b[1];
    const sinC = Math.sin(rad);
    const cosC = Math.cos(rad);
    newDst[0] = p0 * cosC - p1 * sinC + b[0];
    newDst[1] = p0 * sinC + p1 * cosC + b[1];
    return newDst;
  }
  function setLength(a, len2, dst) {
    const newDst = dst ?? new Ctor(2);
    normalize(a, newDst);
    return mulScalar(newDst, len2, newDst);
  }
  function truncate(a, maxLen, dst) {
    const newDst = dst ?? new Ctor(2);
    if (length(a) > maxLen) {
      return setLength(a, maxLen, newDst);
    }
    return copy(a, newDst);
  }
  function midpoint(a, b, dst) {
    const newDst = dst ?? new Ctor(2);
    return lerp2(a, b, 0.5, newDst);
  }
  return {
    create,
    fromValues,
    set,
    ceil,
    floor,
    round,
    clamp: clamp2,
    add,
    addScaled,
    angle,
    subtract,
    sub,
    equalsApproximately,
    equals,
    lerp: lerp2,
    lerpV,
    max,
    min,
    mulScalar,
    scale,
    divScalar,
    inverse,
    invert,
    cross,
    dot,
    length,
    len,
    lengthSq,
    lenSq,
    distance,
    dist,
    distanceSq,
    distSq,
    normalize,
    negate,
    copy,
    clone,
    multiply,
    mul,
    divide,
    div,
    random,
    zero,
    transformMat4,
    transformMat3,
    rotate: rotate2,
    setLength,
    truncate,
    midpoint
  };
}
var cache$5 = /* @__PURE__ */ new Map();
function getAPI$5(Ctor) {
  let api = cache$5.get(Ctor);
  if (!api) {
    api = getAPIImpl$5(Ctor);
    cache$5.set(Ctor, api);
  }
  return api;
}
function getAPIImpl$4(Ctor) {
  function create(x, y, z) {
    const newDst = new Ctor(3);
    if (x !== void 0) {
      newDst[0] = x;
      if (y !== void 0) {
        newDst[1] = y;
        if (z !== void 0) {
          newDst[2] = z;
        }
      }
    }
    return newDst;
  }
  const fromValues = create;
  function set(x, y, z, dst) {
    const newDst = dst ?? new Ctor(3);
    newDst[0] = x;
    newDst[1] = y;
    newDst[2] = z;
    return newDst;
  }
  function ceil(v, dst) {
    const newDst = dst ?? new Ctor(3);
    newDst[0] = Math.ceil(v[0]);
    newDst[1] = Math.ceil(v[1]);
    newDst[2] = Math.ceil(v[2]);
    return newDst;
  }
  function floor(v, dst) {
    const newDst = dst ?? new Ctor(3);
    newDst[0] = Math.floor(v[0]);
    newDst[1] = Math.floor(v[1]);
    newDst[2] = Math.floor(v[2]);
    return newDst;
  }
  function round(v, dst) {
    const newDst = dst ?? new Ctor(3);
    newDst[0] = Math.round(v[0]);
    newDst[1] = Math.round(v[1]);
    newDst[2] = Math.round(v[2]);
    return newDst;
  }
  function clamp2(v, min2 = 0, max2 = 1, dst) {
    const newDst = dst ?? new Ctor(3);
    newDst[0] = Math.min(max2, Math.max(min2, v[0]));
    newDst[1] = Math.min(max2, Math.max(min2, v[1]));
    newDst[2] = Math.min(max2, Math.max(min2, v[2]));
    return newDst;
  }
  function add(a, b, dst) {
    const newDst = dst ?? new Ctor(3);
    newDst[0] = a[0] + b[0];
    newDst[1] = a[1] + b[1];
    newDst[2] = a[2] + b[2];
    return newDst;
  }
  function addScaled(a, b, scale2, dst) {
    const newDst = dst ?? new Ctor(3);
    newDst[0] = a[0] + b[0] * scale2;
    newDst[1] = a[1] + b[1] * scale2;
    newDst[2] = a[2] + b[2] * scale2;
    return newDst;
  }
  function angle(a, b) {
    const ax = a[0];
    const ay = a[1];
    const az = a[2];
    const bx = b[0];
    const by = b[1];
    const bz = b[2];
    const mag1 = Math.sqrt(ax * ax + ay * ay + az * az);
    const mag2 = Math.sqrt(bx * bx + by * by + bz * bz);
    const mag = mag1 * mag2;
    const cosine = mag && dot(a, b) / mag;
    return Math.acos(cosine);
  }
  function subtract(a, b, dst) {
    const newDst = dst ?? new Ctor(3);
    newDst[0] = a[0] - b[0];
    newDst[1] = a[1] - b[1];
    newDst[2] = a[2] - b[2];
    return newDst;
  }
  const sub = subtract;
  function equalsApproximately(a, b) {
    return Math.abs(a[0] - b[0]) < EPSILON && Math.abs(a[1] - b[1]) < EPSILON && Math.abs(a[2] - b[2]) < EPSILON;
  }
  function equals(a, b) {
    return a[0] === b[0] && a[1] === b[1] && a[2] === b[2];
  }
  function lerp2(a, b, t, dst) {
    const newDst = dst ?? new Ctor(3);
    newDst[0] = a[0] + t * (b[0] - a[0]);
    newDst[1] = a[1] + t * (b[1] - a[1]);
    newDst[2] = a[2] + t * (b[2] - a[2]);
    return newDst;
  }
  function lerpV(a, b, t, dst) {
    const newDst = dst ?? new Ctor(3);
    newDst[0] = a[0] + t[0] * (b[0] - a[0]);
    newDst[1] = a[1] + t[1] * (b[1] - a[1]);
    newDst[2] = a[2] + t[2] * (b[2] - a[2]);
    return newDst;
  }
  function max(a, b, dst) {
    const newDst = dst ?? new Ctor(3);
    newDst[0] = Math.max(a[0], b[0]);
    newDst[1] = Math.max(a[1], b[1]);
    newDst[2] = Math.max(a[2], b[2]);
    return newDst;
  }
  function min(a, b, dst) {
    const newDst = dst ?? new Ctor(3);
    newDst[0] = Math.min(a[0], b[0]);
    newDst[1] = Math.min(a[1], b[1]);
    newDst[2] = Math.min(a[2], b[2]);
    return newDst;
  }
  function mulScalar(v, k, dst) {
    const newDst = dst ?? new Ctor(3);
    newDst[0] = v[0] * k;
    newDst[1] = v[1] * k;
    newDst[2] = v[2] * k;
    return newDst;
  }
  const scale = mulScalar;
  function divScalar(v, k, dst) {
    const newDst = dst ?? new Ctor(3);
    newDst[0] = v[0] / k;
    newDst[1] = v[1] / k;
    newDst[2] = v[2] / k;
    return newDst;
  }
  function inverse(v, dst) {
    const newDst = dst ?? new Ctor(3);
    newDst[0] = 1 / v[0];
    newDst[1] = 1 / v[1];
    newDst[2] = 1 / v[2];
    return newDst;
  }
  const invert = inverse;
  function cross(a, b, dst) {
    const newDst = dst ?? new Ctor(3);
    const t1 = a[2] * b[0] - a[0] * b[2];
    const t2 = a[0] * b[1] - a[1] * b[0];
    newDst[0] = a[1] * b[2] - a[2] * b[1];
    newDst[1] = t1;
    newDst[2] = t2;
    return newDst;
  }
  function dot(a, b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  }
  function length(v) {
    const v0 = v[0];
    const v1 = v[1];
    const v2 = v[2];
    return Math.sqrt(v0 * v0 + v1 * v1 + v2 * v2);
  }
  const len = length;
  function lengthSq(v) {
    const v0 = v[0];
    const v1 = v[1];
    const v2 = v[2];
    return v0 * v0 + v1 * v1 + v2 * v2;
  }
  const lenSq = lengthSq;
  function distance(a, b) {
    const dx = a[0] - b[0];
    const dy = a[1] - b[1];
    const dz = a[2] - b[2];
    return Math.sqrt(dx * dx + dy * dy + dz * dz);
  }
  const dist = distance;
  function distanceSq(a, b) {
    const dx = a[0] - b[0];
    const dy = a[1] - b[1];
    const dz = a[2] - b[2];
    return dx * dx + dy * dy + dz * dz;
  }
  const distSq = distanceSq;
  function normalize(v, dst) {
    const newDst = dst ?? new Ctor(3);
    const v0 = v[0];
    const v1 = v[1];
    const v2 = v[2];
    const len2 = Math.sqrt(v0 * v0 + v1 * v1 + v2 * v2);
    if (len2 > 1e-5) {
      newDst[0] = v0 / len2;
      newDst[1] = v1 / len2;
      newDst[2] = v2 / len2;
    } else {
      newDst[0] = 0;
      newDst[1] = 0;
      newDst[2] = 0;
    }
    return newDst;
  }
  function negate(v, dst) {
    const newDst = dst ?? new Ctor(3);
    newDst[0] = -v[0];
    newDst[1] = -v[1];
    newDst[2] = -v[2];
    return newDst;
  }
  function copy(v, dst) {
    const newDst = dst ?? new Ctor(3);
    newDst[0] = v[0];
    newDst[1] = v[1];
    newDst[2] = v[2];
    return newDst;
  }
  const clone = copy;
  function multiply(a, b, dst) {
    const newDst = dst ?? new Ctor(3);
    newDst[0] = a[0] * b[0];
    newDst[1] = a[1] * b[1];
    newDst[2] = a[2] * b[2];
    return newDst;
  }
  const mul = multiply;
  function divide(a, b, dst) {
    const newDst = dst ?? new Ctor(3);
    newDst[0] = a[0] / b[0];
    newDst[1] = a[1] / b[1];
    newDst[2] = a[2] / b[2];
    return newDst;
  }
  const div = divide;
  function random(scale2 = 1, dst) {
    const newDst = dst ?? new Ctor(3);
    const angle2 = Math.random() * 2 * Math.PI;
    const z = Math.random() * 2 - 1;
    const zScale = Math.sqrt(1 - z * z) * scale2;
    newDst[0] = Math.cos(angle2) * zScale;
    newDst[1] = Math.sin(angle2) * zScale;
    newDst[2] = z * scale2;
    return newDst;
  }
  function zero(dst) {
    const newDst = dst ?? new Ctor(3);
    newDst[0] = 0;
    newDst[1] = 0;
    newDst[2] = 0;
    return newDst;
  }
  function transformMat4(v, m, dst) {
    const newDst = dst ?? new Ctor(3);
    const x = v[0];
    const y = v[1];
    const z = v[2];
    const w = m[3] * x + m[7] * y + m[11] * z + m[15] || 1;
    newDst[0] = (m[0] * x + m[4] * y + m[8] * z + m[12]) / w;
    newDst[1] = (m[1] * x + m[5] * y + m[9] * z + m[13]) / w;
    newDst[2] = (m[2] * x + m[6] * y + m[10] * z + m[14]) / w;
    return newDst;
  }
  function transformMat4Upper3x3(v, m, dst) {
    const newDst = dst ?? new Ctor(3);
    const v0 = v[0];
    const v1 = v[1];
    const v2 = v[2];
    newDst[0] = v0 * m[0 * 4 + 0] + v1 * m[1 * 4 + 0] + v2 * m[2 * 4 + 0];
    newDst[1] = v0 * m[0 * 4 + 1] + v1 * m[1 * 4 + 1] + v2 * m[2 * 4 + 1];
    newDst[2] = v0 * m[0 * 4 + 2] + v1 * m[1 * 4 + 2] + v2 * m[2 * 4 + 2];
    return newDst;
  }
  function transformMat3(v, m, dst) {
    const newDst = dst ?? new Ctor(3);
    const x = v[0];
    const y = v[1];
    const z = v[2];
    newDst[0] = x * m[0] + y * m[4] + z * m[8];
    newDst[1] = x * m[1] + y * m[5] + z * m[9];
    newDst[2] = x * m[2] + y * m[6] + z * m[10];
    return newDst;
  }
  function transformQuat(v, q, dst) {
    const newDst = dst ?? new Ctor(3);
    const qx = q[0];
    const qy = q[1];
    const qz = q[2];
    const w2 = q[3] * 2;
    const x = v[0];
    const y = v[1];
    const z = v[2];
    const uvX = qy * z - qz * y;
    const uvY = qz * x - qx * z;
    const uvZ = qx * y - qy * x;
    newDst[0] = x + uvX * w2 + (qy * uvZ - qz * uvY) * 2;
    newDst[1] = y + uvY * w2 + (qz * uvX - qx * uvZ) * 2;
    newDst[2] = z + uvZ * w2 + (qx * uvY - qy * uvX) * 2;
    return newDst;
  }
  function getTranslation(m, dst) {
    const newDst = dst ?? new Ctor(3);
    newDst[0] = m[12];
    newDst[1] = m[13];
    newDst[2] = m[14];
    return newDst;
  }
  function getAxis(m, axis, dst) {
    const newDst = dst ?? new Ctor(3);
    const off = axis * 4;
    newDst[0] = m[off + 0];
    newDst[1] = m[off + 1];
    newDst[2] = m[off + 2];
    return newDst;
  }
  function getScaling(m, dst) {
    const newDst = dst ?? new Ctor(3);
    const xx = m[0];
    const xy = m[1];
    const xz = m[2];
    const yx = m[4];
    const yy = m[5];
    const yz = m[6];
    const zx = m[8];
    const zy = m[9];
    const zz = m[10];
    newDst[0] = Math.sqrt(xx * xx + xy * xy + xz * xz);
    newDst[1] = Math.sqrt(yx * yx + yy * yy + yz * yz);
    newDst[2] = Math.sqrt(zx * zx + zy * zy + zz * zz);
    return newDst;
  }
  function rotateX(a, b, rad, dst) {
    const newDst = dst ?? new Ctor(3);
    const p = [];
    const r = [];
    p[0] = a[0] - b[0];
    p[1] = a[1] - b[1];
    p[2] = a[2] - b[2];
    r[0] = p[0];
    r[1] = p[1] * Math.cos(rad) - p[2] * Math.sin(rad);
    r[2] = p[1] * Math.sin(rad) + p[2] * Math.cos(rad);
    newDst[0] = r[0] + b[0];
    newDst[1] = r[1] + b[1];
    newDst[2] = r[2] + b[2];
    return newDst;
  }
  function rotateY(a, b, rad, dst) {
    const newDst = dst ?? new Ctor(3);
    const p = [];
    const r = [];
    p[0] = a[0] - b[0];
    p[1] = a[1] - b[1];
    p[2] = a[2] - b[2];
    r[0] = p[2] * Math.sin(rad) + p[0] * Math.cos(rad);
    r[1] = p[1];
    r[2] = p[2] * Math.cos(rad) - p[0] * Math.sin(rad);
    newDst[0] = r[0] + b[0];
    newDst[1] = r[1] + b[1];
    newDst[2] = r[2] + b[2];
    return newDst;
  }
  function rotateZ(a, b, rad, dst) {
    const newDst = dst ?? new Ctor(3);
    const p = [];
    const r = [];
    p[0] = a[0] - b[0];
    p[1] = a[1] - b[1];
    p[2] = a[2] - b[2];
    r[0] = p[0] * Math.cos(rad) - p[1] * Math.sin(rad);
    r[1] = p[0] * Math.sin(rad) + p[1] * Math.cos(rad);
    r[2] = p[2];
    newDst[0] = r[0] + b[0];
    newDst[1] = r[1] + b[1];
    newDst[2] = r[2] + b[2];
    return newDst;
  }
  function setLength(a, len2, dst) {
    const newDst = dst ?? new Ctor(3);
    normalize(a, newDst);
    return mulScalar(newDst, len2, newDst);
  }
  function truncate(a, maxLen, dst) {
    const newDst = dst ?? new Ctor(3);
    if (length(a) > maxLen) {
      return setLength(a, maxLen, newDst);
    }
    return copy(a, newDst);
  }
  function midpoint(a, b, dst) {
    const newDst = dst ?? new Ctor(3);
    return lerp2(a, b, 0.5, newDst);
  }
  return {
    create,
    fromValues,
    set,
    ceil,
    floor,
    round,
    clamp: clamp2,
    add,
    addScaled,
    angle,
    subtract,
    sub,
    equalsApproximately,
    equals,
    lerp: lerp2,
    lerpV,
    max,
    min,
    mulScalar,
    scale,
    divScalar,
    inverse,
    invert,
    cross,
    dot,
    length,
    len,
    lengthSq,
    lenSq,
    distance,
    dist,
    distanceSq,
    distSq,
    normalize,
    negate,
    copy,
    clone,
    multiply,
    mul,
    divide,
    div,
    random,
    zero,
    transformMat4,
    transformMat4Upper3x3,
    transformMat3,
    transformQuat,
    getTranslation,
    getAxis,
    getScaling,
    rotateX,
    rotateY,
    rotateZ,
    setLength,
    truncate,
    midpoint
  };
}
var cache$4 = /* @__PURE__ */ new Map();
function getAPI$4(Ctor) {
  let api = cache$4.get(Ctor);
  if (!api) {
    api = getAPIImpl$4(Ctor);
    cache$4.set(Ctor, api);
  }
  return api;
}
function getAPIImpl$3(Ctor) {
  const vec22 = getAPI$5(Ctor);
  const vec32 = getAPI$4(Ctor);
  function create(v0, v1, v2, v3, v4, v5, v6, v7, v8) {
    const newDst = new Ctor(12);
    newDst[3] = 0;
    newDst[7] = 0;
    newDst[11] = 0;
    if (v0 !== void 0) {
      newDst[0] = v0;
      if (v1 !== void 0) {
        newDst[1] = v1;
        if (v2 !== void 0) {
          newDst[2] = v2;
          if (v3 !== void 0) {
            newDst[4] = v3;
            if (v4 !== void 0) {
              newDst[5] = v4;
              if (v5 !== void 0) {
                newDst[6] = v5;
                if (v6 !== void 0) {
                  newDst[8] = v6;
                  if (v7 !== void 0) {
                    newDst[9] = v7;
                    if (v8 !== void 0) {
                      newDst[10] = v8;
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
    return newDst;
  }
  function set(v0, v1, v2, v3, v4, v5, v6, v7, v8, dst) {
    const newDst = dst ?? new Ctor(12);
    newDst[0] = v0;
    newDst[1] = v1;
    newDst[2] = v2;
    newDst[3] = 0;
    newDst[4] = v3;
    newDst[5] = v4;
    newDst[6] = v5;
    newDst[7] = 0;
    newDst[8] = v6;
    newDst[9] = v7;
    newDst[10] = v8;
    newDst[11] = 0;
    return newDst;
  }
  function fromMat4(m4, dst) {
    const newDst = dst ?? new Ctor(12);
    newDst[0] = m4[0];
    newDst[1] = m4[1];
    newDst[2] = m4[2];
    newDst[3] = 0;
    newDst[4] = m4[4];
    newDst[5] = m4[5];
    newDst[6] = m4[6];
    newDst[7] = 0;
    newDst[8] = m4[8];
    newDst[9] = m4[9];
    newDst[10] = m4[10];
    newDst[11] = 0;
    return newDst;
  }
  function fromQuat(q, dst) {
    const newDst = dst ?? new Ctor(12);
    const x = q[0];
    const y = q[1];
    const z = q[2];
    const w = q[3];
    const x2 = x + x;
    const y2 = y + y;
    const z2 = z + z;
    const xx = x * x2;
    const yx = y * x2;
    const yy = y * y2;
    const zx = z * x2;
    const zy = z * y2;
    const zz = z * z2;
    const wx = w * x2;
    const wy = w * y2;
    const wz = w * z2;
    newDst[0] = 1 - yy - zz;
    newDst[1] = yx + wz;
    newDst[2] = zx - wy;
    newDst[3] = 0;
    newDst[4] = yx - wz;
    newDst[5] = 1 - xx - zz;
    newDst[6] = zy + wx;
    newDst[7] = 0;
    newDst[8] = zx + wy;
    newDst[9] = zy - wx;
    newDst[10] = 1 - xx - yy;
    newDst[11] = 0;
    return newDst;
  }
  function negate(m, dst) {
    const newDst = dst ?? new Ctor(12);
    newDst[0] = -m[0];
    newDst[1] = -m[1];
    newDst[2] = -m[2];
    newDst[4] = -m[4];
    newDst[5] = -m[5];
    newDst[6] = -m[6];
    newDst[8] = -m[8];
    newDst[9] = -m[9];
    newDst[10] = -m[10];
    return newDst;
  }
  function multiplyScalar(m, s, dst) {
    const newDst = dst ?? new Ctor(12);
    newDst[0] = m[0] * s;
    newDst[1] = m[1] * s;
    newDst[2] = m[2] * s;
    newDst[4] = m[4] * s;
    newDst[5] = m[5] * s;
    newDst[6] = m[6] * s;
    newDst[8] = m[8] * s;
    newDst[9] = m[9] * s;
    newDst[10] = m[10] * s;
    return newDst;
  }
  const mulScalar = multiplyScalar;
  function add(a, b, dst) {
    const newDst = dst ?? new Ctor(12);
    newDst[0] = a[0] + b[0];
    newDst[1] = a[1] + b[1];
    newDst[2] = a[2] + b[2];
    newDst[4] = a[4] + b[4];
    newDst[5] = a[5] + b[5];
    newDst[6] = a[6] + b[6];
    newDst[8] = a[8] + b[8];
    newDst[9] = a[9] + b[9];
    newDst[10] = a[10] + b[10];
    return newDst;
  }
  function copy(m, dst) {
    const newDst = dst ?? new Ctor(12);
    newDst[0] = m[0];
    newDst[1] = m[1];
    newDst[2] = m[2];
    newDst[4] = m[4];
    newDst[5] = m[5];
    newDst[6] = m[6];
    newDst[8] = m[8];
    newDst[9] = m[9];
    newDst[10] = m[10];
    return newDst;
  }
  const clone = copy;
  function equalsApproximately(a, b) {
    return Math.abs(a[0] - b[0]) < EPSILON && Math.abs(a[1] - b[1]) < EPSILON && Math.abs(a[2] - b[2]) < EPSILON && Math.abs(a[4] - b[4]) < EPSILON && Math.abs(a[5] - b[5]) < EPSILON && Math.abs(a[6] - b[6]) < EPSILON && Math.abs(a[8] - b[8]) < EPSILON && Math.abs(a[9] - b[9]) < EPSILON && Math.abs(a[10] - b[10]) < EPSILON;
  }
  function equals(a, b) {
    return a[0] === b[0] && a[1] === b[1] && a[2] === b[2] && a[4] === b[4] && a[5] === b[5] && a[6] === b[6] && a[8] === b[8] && a[9] === b[9] && a[10] === b[10];
  }
  function identity(dst) {
    const newDst = dst ?? new Ctor(12);
    newDst[0] = 1;
    newDst[1] = 0;
    newDst[2] = 0;
    newDst[4] = 0;
    newDst[5] = 1;
    newDst[6] = 0;
    newDst[8] = 0;
    newDst[9] = 0;
    newDst[10] = 1;
    return newDst;
  }
  function transpose(m, dst) {
    const newDst = dst ?? new Ctor(12);
    if (newDst === m) {
      let t;
      t = m[1];
      m[1] = m[4];
      m[4] = t;
      t = m[2];
      m[2] = m[8];
      m[8] = t;
      t = m[6];
      m[6] = m[9];
      m[9] = t;
      return newDst;
    }
    const m00 = m[0 * 4 + 0];
    const m01 = m[0 * 4 + 1];
    const m02 = m[0 * 4 + 2];
    const m10 = m[1 * 4 + 0];
    const m11 = m[1 * 4 + 1];
    const m12 = m[1 * 4 + 2];
    const m20 = m[2 * 4 + 0];
    const m21 = m[2 * 4 + 1];
    const m22 = m[2 * 4 + 2];
    newDst[0] = m00;
    newDst[1] = m10;
    newDst[2] = m20;
    newDst[4] = m01;
    newDst[5] = m11;
    newDst[6] = m21;
    newDst[8] = m02;
    newDst[9] = m12;
    newDst[10] = m22;
    return newDst;
  }
  function inverse(m, dst) {
    const newDst = dst ?? new Ctor(12);
    const m00 = m[0 * 4 + 0];
    const m01 = m[0 * 4 + 1];
    const m02 = m[0 * 4 + 2];
    const m10 = m[1 * 4 + 0];
    const m11 = m[1 * 4 + 1];
    const m12 = m[1 * 4 + 2];
    const m20 = m[2 * 4 + 0];
    const m21 = m[2 * 4 + 1];
    const m22 = m[2 * 4 + 2];
    const b01 = m22 * m11 - m12 * m21;
    const b11 = -m22 * m10 + m12 * m20;
    const b21 = m21 * m10 - m11 * m20;
    const invDet = 1 / (m00 * b01 + m01 * b11 + m02 * b21);
    newDst[0] = b01 * invDet;
    newDst[1] = (-m22 * m01 + m02 * m21) * invDet;
    newDst[2] = (m12 * m01 - m02 * m11) * invDet;
    newDst[4] = b11 * invDet;
    newDst[5] = (m22 * m00 - m02 * m20) * invDet;
    newDst[6] = (-m12 * m00 + m02 * m10) * invDet;
    newDst[8] = b21 * invDet;
    newDst[9] = (-m21 * m00 + m01 * m20) * invDet;
    newDst[10] = (m11 * m00 - m01 * m10) * invDet;
    return newDst;
  }
  function determinant(m) {
    const m00 = m[0 * 4 + 0];
    const m01 = m[0 * 4 + 1];
    const m02 = m[0 * 4 + 2];
    const m10 = m[1 * 4 + 0];
    const m11 = m[1 * 4 + 1];
    const m12 = m[1 * 4 + 2];
    const m20 = m[2 * 4 + 0];
    const m21 = m[2 * 4 + 1];
    const m22 = m[2 * 4 + 2];
    return m00 * (m11 * m22 - m21 * m12) - m10 * (m01 * m22 - m21 * m02) + m20 * (m01 * m12 - m11 * m02);
  }
  const invert = inverse;
  function multiply(a, b, dst) {
    const newDst = dst ?? new Ctor(12);
    const a00 = a[0];
    const a01 = a[1];
    const a02 = a[2];
    const a10 = a[4 + 0];
    const a11 = a[4 + 1];
    const a12 = a[4 + 2];
    const a20 = a[8 + 0];
    const a21 = a[8 + 1];
    const a22 = a[8 + 2];
    const b00 = b[0];
    const b01 = b[1];
    const b02 = b[2];
    const b10 = b[4 + 0];
    const b11 = b[4 + 1];
    const b12 = b[4 + 2];
    const b20 = b[8 + 0];
    const b21 = b[8 + 1];
    const b22 = b[8 + 2];
    newDst[0] = a00 * b00 + a10 * b01 + a20 * b02;
    newDst[1] = a01 * b00 + a11 * b01 + a21 * b02;
    newDst[2] = a02 * b00 + a12 * b01 + a22 * b02;
    newDst[4] = a00 * b10 + a10 * b11 + a20 * b12;
    newDst[5] = a01 * b10 + a11 * b11 + a21 * b12;
    newDst[6] = a02 * b10 + a12 * b11 + a22 * b12;
    newDst[8] = a00 * b20 + a10 * b21 + a20 * b22;
    newDst[9] = a01 * b20 + a11 * b21 + a21 * b22;
    newDst[10] = a02 * b20 + a12 * b21 + a22 * b22;
    return newDst;
  }
  const mul = multiply;
  function setTranslation(a, v, dst) {
    const newDst = dst ?? identity();
    if (a !== newDst) {
      newDst[0] = a[0];
      newDst[1] = a[1];
      newDst[2] = a[2];
      newDst[4] = a[4];
      newDst[5] = a[5];
      newDst[6] = a[6];
    }
    newDst[8] = v[0];
    newDst[9] = v[1];
    newDst[10] = 1;
    return newDst;
  }
  function getTranslation(m, dst) {
    const newDst = dst ?? vec22.create();
    newDst[0] = m[8];
    newDst[1] = m[9];
    return newDst;
  }
  function getAxis(m, axis, dst) {
    const newDst = dst ?? vec22.create();
    const off = axis * 4;
    newDst[0] = m[off + 0];
    newDst[1] = m[off + 1];
    return newDst;
  }
  function setAxis(m, v, axis, dst) {
    const newDst = dst === m ? m : copy(m, dst);
    const off = axis * 4;
    newDst[off + 0] = v[0];
    newDst[off + 1] = v[1];
    return newDst;
  }
  function getScaling(m, dst) {
    const newDst = dst ?? vec22.create();
    const xx = m[0];
    const xy = m[1];
    const yx = m[4];
    const yy = m[5];
    newDst[0] = Math.sqrt(xx * xx + xy * xy);
    newDst[1] = Math.sqrt(yx * yx + yy * yy);
    return newDst;
  }
  function get3DScaling(m, dst) {
    const newDst = dst ?? vec32.create();
    const xx = m[0];
    const xy = m[1];
    const xz = m[2];
    const yx = m[4];
    const yy = m[5];
    const yz = m[6];
    const zx = m[8];
    const zy = m[9];
    const zz = m[10];
    newDst[0] = Math.sqrt(xx * xx + xy * xy + xz * xz);
    newDst[1] = Math.sqrt(yx * yx + yy * yy + yz * yz);
    newDst[2] = Math.sqrt(zx * zx + zy * zy + zz * zz);
    return newDst;
  }
  function translation(v, dst) {
    const newDst = dst ?? new Ctor(12);
    newDst[0] = 1;
    newDst[1] = 0;
    newDst[2] = 0;
    newDst[4] = 0;
    newDst[5] = 1;
    newDst[6] = 0;
    newDst[8] = v[0];
    newDst[9] = v[1];
    newDst[10] = 1;
    return newDst;
  }
  function translate(m, v, dst) {
    const newDst = dst ?? new Ctor(12);
    const v0 = v[0];
    const v1 = v[1];
    const m00 = m[0];
    const m01 = m[1];
    const m02 = m[2];
    const m10 = m[1 * 4 + 0];
    const m11 = m[1 * 4 + 1];
    const m12 = m[1 * 4 + 2];
    const m20 = m[2 * 4 + 0];
    const m21 = m[2 * 4 + 1];
    const m22 = m[2 * 4 + 2];
    if (m !== newDst) {
      newDst[0] = m00;
      newDst[1] = m01;
      newDst[2] = m02;
      newDst[4] = m10;
      newDst[5] = m11;
      newDst[6] = m12;
    }
    newDst[8] = m00 * v0 + m10 * v1 + m20;
    newDst[9] = m01 * v0 + m11 * v1 + m21;
    newDst[10] = m02 * v0 + m12 * v1 + m22;
    return newDst;
  }
  function rotation(angleInRadians, dst) {
    const newDst = dst ?? new Ctor(12);
    const c = Math.cos(angleInRadians);
    const s = Math.sin(angleInRadians);
    newDst[0] = c;
    newDst[1] = s;
    newDst[2] = 0;
    newDst[4] = -s;
    newDst[5] = c;
    newDst[6] = 0;
    newDst[8] = 0;
    newDst[9] = 0;
    newDst[10] = 1;
    return newDst;
  }
  function rotate2(m, angleInRadians, dst) {
    const newDst = dst ?? new Ctor(12);
    const m00 = m[0 * 4 + 0];
    const m01 = m[0 * 4 + 1];
    const m02 = m[0 * 4 + 2];
    const m10 = m[1 * 4 + 0];
    const m11 = m[1 * 4 + 1];
    const m12 = m[1 * 4 + 2];
    const c = Math.cos(angleInRadians);
    const s = Math.sin(angleInRadians);
    newDst[0] = c * m00 + s * m10;
    newDst[1] = c * m01 + s * m11;
    newDst[2] = c * m02 + s * m12;
    newDst[4] = c * m10 - s * m00;
    newDst[5] = c * m11 - s * m01;
    newDst[6] = c * m12 - s * m02;
    if (m !== newDst) {
      newDst[8] = m[8];
      newDst[9] = m[9];
      newDst[10] = m[10];
    }
    return newDst;
  }
  function rotationX(angleInRadians, dst) {
    const newDst = dst ?? new Ctor(12);
    const c = Math.cos(angleInRadians);
    const s = Math.sin(angleInRadians);
    newDst[0] = 1;
    newDst[1] = 0;
    newDst[2] = 0;
    newDst[4] = 0;
    newDst[5] = c;
    newDst[6] = s;
    newDst[8] = 0;
    newDst[9] = -s;
    newDst[10] = c;
    return newDst;
  }
  function rotateX(m, angleInRadians, dst) {
    const newDst = dst ?? new Ctor(12);
    const m10 = m[4];
    const m11 = m[5];
    const m12 = m[6];
    const m20 = m[8];
    const m21 = m[9];
    const m22 = m[10];
    const c = Math.cos(angleInRadians);
    const s = Math.sin(angleInRadians);
    newDst[4] = c * m10 + s * m20;
    newDst[5] = c * m11 + s * m21;
    newDst[6] = c * m12 + s * m22;
    newDst[8] = c * m20 - s * m10;
    newDst[9] = c * m21 - s * m11;
    newDst[10] = c * m22 - s * m12;
    if (m !== newDst) {
      newDst[0] = m[0];
      newDst[1] = m[1];
      newDst[2] = m[2];
    }
    return newDst;
  }
  function rotationY(angleInRadians, dst) {
    const newDst = dst ?? new Ctor(12);
    const c = Math.cos(angleInRadians);
    const s = Math.sin(angleInRadians);
    newDst[0] = c;
    newDst[1] = 0;
    newDst[2] = -s;
    newDst[4] = 0;
    newDst[5] = 1;
    newDst[6] = 0;
    newDst[8] = s;
    newDst[9] = 0;
    newDst[10] = c;
    return newDst;
  }
  function rotateY(m, angleInRadians, dst) {
    const newDst = dst ?? new Ctor(12);
    const m00 = m[0 * 4 + 0];
    const m01 = m[0 * 4 + 1];
    const m02 = m[0 * 4 + 2];
    const m20 = m[2 * 4 + 0];
    const m21 = m[2 * 4 + 1];
    const m22 = m[2 * 4 + 2];
    const c = Math.cos(angleInRadians);
    const s = Math.sin(angleInRadians);
    newDst[0] = c * m00 - s * m20;
    newDst[1] = c * m01 - s * m21;
    newDst[2] = c * m02 - s * m22;
    newDst[8] = c * m20 + s * m00;
    newDst[9] = c * m21 + s * m01;
    newDst[10] = c * m22 + s * m02;
    if (m !== newDst) {
      newDst[4] = m[4];
      newDst[5] = m[5];
      newDst[6] = m[6];
    }
    return newDst;
  }
  const rotationZ = rotation;
  const rotateZ = rotate2;
  function scaling(v, dst) {
    const newDst = dst ?? new Ctor(12);
    newDst[0] = v[0];
    newDst[1] = 0;
    newDst[2] = 0;
    newDst[4] = 0;
    newDst[5] = v[1];
    newDst[6] = 0;
    newDst[8] = 0;
    newDst[9] = 0;
    newDst[10] = 1;
    return newDst;
  }
  function scale(m, v, dst) {
    const newDst = dst ?? new Ctor(12);
    const v0 = v[0];
    const v1 = v[1];
    newDst[0] = v0 * m[0 * 4 + 0];
    newDst[1] = v0 * m[0 * 4 + 1];
    newDst[2] = v0 * m[0 * 4 + 2];
    newDst[4] = v1 * m[1 * 4 + 0];
    newDst[5] = v1 * m[1 * 4 + 1];
    newDst[6] = v1 * m[1 * 4 + 2];
    if (m !== newDst) {
      newDst[8] = m[8];
      newDst[9] = m[9];
      newDst[10] = m[10];
    }
    return newDst;
  }
  function scaling3D(v, dst) {
    const newDst = dst ?? new Ctor(12);
    newDst[0] = v[0];
    newDst[1] = 0;
    newDst[2] = 0;
    newDst[4] = 0;
    newDst[5] = v[1];
    newDst[6] = 0;
    newDst[8] = 0;
    newDst[9] = 0;
    newDst[10] = v[2];
    return newDst;
  }
  function scale3D(m, v, dst) {
    const newDst = dst ?? new Ctor(12);
    const v0 = v[0];
    const v1 = v[1];
    const v2 = v[2];
    newDst[0] = v0 * m[0 * 4 + 0];
    newDst[1] = v0 * m[0 * 4 + 1];
    newDst[2] = v0 * m[0 * 4 + 2];
    newDst[4] = v1 * m[1 * 4 + 0];
    newDst[5] = v1 * m[1 * 4 + 1];
    newDst[6] = v1 * m[1 * 4 + 2];
    newDst[8] = v2 * m[2 * 4 + 0];
    newDst[9] = v2 * m[2 * 4 + 1];
    newDst[10] = v2 * m[2 * 4 + 2];
    return newDst;
  }
  function uniformScaling(s, dst) {
    const newDst = dst ?? new Ctor(12);
    newDst[0] = s;
    newDst[1] = 0;
    newDst[2] = 0;
    newDst[4] = 0;
    newDst[5] = s;
    newDst[6] = 0;
    newDst[8] = 0;
    newDst[9] = 0;
    newDst[10] = 1;
    return newDst;
  }
  function uniformScale(m, s, dst) {
    const newDst = dst ?? new Ctor(12);
    newDst[0] = s * m[0 * 4 + 0];
    newDst[1] = s * m[0 * 4 + 1];
    newDst[2] = s * m[0 * 4 + 2];
    newDst[4] = s * m[1 * 4 + 0];
    newDst[5] = s * m[1 * 4 + 1];
    newDst[6] = s * m[1 * 4 + 2];
    if (m !== newDst) {
      newDst[8] = m[8];
      newDst[9] = m[9];
      newDst[10] = m[10];
    }
    return newDst;
  }
  function uniformScaling3D(s, dst) {
    const newDst = dst ?? new Ctor(12);
    newDst[0] = s;
    newDst[1] = 0;
    newDst[2] = 0;
    newDst[4] = 0;
    newDst[5] = s;
    newDst[6] = 0;
    newDst[8] = 0;
    newDst[9] = 0;
    newDst[10] = s;
    return newDst;
  }
  function uniformScale3D(m, s, dst) {
    const newDst = dst ?? new Ctor(12);
    newDst[0] = s * m[0 * 4 + 0];
    newDst[1] = s * m[0 * 4 + 1];
    newDst[2] = s * m[0 * 4 + 2];
    newDst[4] = s * m[1 * 4 + 0];
    newDst[5] = s * m[1 * 4 + 1];
    newDst[6] = s * m[1 * 4 + 2];
    newDst[8] = s * m[2 * 4 + 0];
    newDst[9] = s * m[2 * 4 + 1];
    newDst[10] = s * m[2 * 4 + 2];
    return newDst;
  }
  return {
    add,
    clone,
    copy,
    create,
    determinant,
    equals,
    equalsApproximately,
    fromMat4,
    fromQuat,
    get3DScaling,
    getAxis,
    getScaling,
    getTranslation,
    identity,
    inverse,
    invert,
    mul,
    mulScalar,
    multiply,
    multiplyScalar,
    negate,
    rotate: rotate2,
    rotateX,
    rotateY,
    rotateZ,
    rotation,
    rotationX,
    rotationY,
    rotationZ,
    scale,
    scale3D,
    scaling,
    scaling3D,
    set,
    setAxis,
    setTranslation,
    translate,
    translation,
    transpose,
    uniformScale,
    uniformScale3D,
    uniformScaling,
    uniformScaling3D
  };
}
var cache$3 = /* @__PURE__ */ new Map();
function getAPI$3(Ctor) {
  let api = cache$3.get(Ctor);
  if (!api) {
    api = getAPIImpl$3(Ctor);
    cache$3.set(Ctor, api);
  }
  return api;
}
function getAPIImpl$2(Ctor) {
  const vec32 = getAPI$4(Ctor);
  function create(v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15) {
    const newDst = new Ctor(16);
    if (v0 !== void 0) {
      newDst[0] = v0;
      if (v1 !== void 0) {
        newDst[1] = v1;
        if (v2 !== void 0) {
          newDst[2] = v2;
          if (v3 !== void 0) {
            newDst[3] = v3;
            if (v4 !== void 0) {
              newDst[4] = v4;
              if (v5 !== void 0) {
                newDst[5] = v5;
                if (v6 !== void 0) {
                  newDst[6] = v6;
                  if (v7 !== void 0) {
                    newDst[7] = v7;
                    if (v8 !== void 0) {
                      newDst[8] = v8;
                      if (v9 !== void 0) {
                        newDst[9] = v9;
                        if (v10 !== void 0) {
                          newDst[10] = v10;
                          if (v11 !== void 0) {
                            newDst[11] = v11;
                            if (v12 !== void 0) {
                              newDst[12] = v12;
                              if (v13 !== void 0) {
                                newDst[13] = v13;
                                if (v14 !== void 0) {
                                  newDst[14] = v14;
                                  if (v15 !== void 0) {
                                    newDst[15] = v15;
                                  }
                                }
                              }
                            }
                          }
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
    return newDst;
  }
  function set(v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, dst) {
    const newDst = dst ?? new Ctor(16);
    newDst[0] = v0;
    newDst[1] = v1;
    newDst[2] = v2;
    newDst[3] = v3;
    newDst[4] = v4;
    newDst[5] = v5;
    newDst[6] = v6;
    newDst[7] = v7;
    newDst[8] = v8;
    newDst[9] = v9;
    newDst[10] = v10;
    newDst[11] = v11;
    newDst[12] = v12;
    newDst[13] = v13;
    newDst[14] = v14;
    newDst[15] = v15;
    return newDst;
  }
  function fromMat3(m3, dst) {
    const newDst = dst ?? new Ctor(16);
    newDst[0] = m3[0];
    newDst[1] = m3[1];
    newDst[2] = m3[2];
    newDst[3] = 0;
    newDst[4] = m3[4];
    newDst[5] = m3[5];
    newDst[6] = m3[6];
    newDst[7] = 0;
    newDst[8] = m3[8];
    newDst[9] = m3[9];
    newDst[10] = m3[10];
    newDst[11] = 0;
    newDst[12] = 0;
    newDst[13] = 0;
    newDst[14] = 0;
    newDst[15] = 1;
    return newDst;
  }
  function fromQuat(q, dst) {
    const newDst = dst ?? new Ctor(16);
    const x = q[0];
    const y = q[1];
    const z = q[2];
    const w = q[3];
    const x2 = x + x;
    const y2 = y + y;
    const z2 = z + z;
    const xx = x * x2;
    const yx = y * x2;
    const yy = y * y2;
    const zx = z * x2;
    const zy = z * y2;
    const zz = z * z2;
    const wx = w * x2;
    const wy = w * y2;
    const wz = w * z2;
    newDst[0] = 1 - yy - zz;
    newDst[1] = yx + wz;
    newDst[2] = zx - wy;
    newDst[3] = 0;
    newDst[4] = yx - wz;
    newDst[5] = 1 - xx - zz;
    newDst[6] = zy + wx;
    newDst[7] = 0;
    newDst[8] = zx + wy;
    newDst[9] = zy - wx;
    newDst[10] = 1 - xx - yy;
    newDst[11] = 0;
    newDst[12] = 0;
    newDst[13] = 0;
    newDst[14] = 0;
    newDst[15] = 1;
    return newDst;
  }
  function negate(m, dst) {
    const newDst = dst ?? new Ctor(16);
    newDst[0] = -m[0];
    newDst[1] = -m[1];
    newDst[2] = -m[2];
    newDst[3] = -m[3];
    newDst[4] = -m[4];
    newDst[5] = -m[5];
    newDst[6] = -m[6];
    newDst[7] = -m[7];
    newDst[8] = -m[8];
    newDst[9] = -m[9];
    newDst[10] = -m[10];
    newDst[11] = -m[11];
    newDst[12] = -m[12];
    newDst[13] = -m[13];
    newDst[14] = -m[14];
    newDst[15] = -m[15];
    return newDst;
  }
  function add(a, b, dst) {
    const newDst = dst ?? new Ctor(16);
    newDst[0] = a[0] + b[0];
    newDst[1] = a[1] + b[1];
    newDst[2] = a[2] + b[2];
    newDst[3] = a[3] + b[3];
    newDst[4] = a[4] + b[4];
    newDst[5] = a[5] + b[5];
    newDst[6] = a[6] + b[6];
    newDst[7] = a[7] + b[7];
    newDst[8] = a[8] + b[8];
    newDst[9] = a[9] + b[9];
    newDst[10] = a[10] + b[10];
    newDst[11] = a[11] + b[11];
    newDst[12] = a[12] + b[12];
    newDst[13] = a[13] + b[13];
    newDst[14] = a[14] + b[14];
    newDst[15] = a[15] + b[15];
    return newDst;
  }
  function multiplyScalar(m, s, dst) {
    const newDst = dst ?? new Ctor(16);
    newDst[0] = m[0] * s;
    newDst[1] = m[1] * s;
    newDst[2] = m[2] * s;
    newDst[3] = m[3] * s;
    newDst[4] = m[4] * s;
    newDst[5] = m[5] * s;
    newDst[6] = m[6] * s;
    newDst[7] = m[7] * s;
    newDst[8] = m[8] * s;
    newDst[9] = m[9] * s;
    newDst[10] = m[10] * s;
    newDst[11] = m[11] * s;
    newDst[12] = m[12] * s;
    newDst[13] = m[13] * s;
    newDst[14] = m[14] * s;
    newDst[15] = m[15] * s;
    return newDst;
  }
  const mulScalar = multiplyScalar;
  function copy(m, dst) {
    const newDst = dst ?? new Ctor(16);
    newDst[0] = m[0];
    newDst[1] = m[1];
    newDst[2] = m[2];
    newDst[3] = m[3];
    newDst[4] = m[4];
    newDst[5] = m[5];
    newDst[6] = m[6];
    newDst[7] = m[7];
    newDst[8] = m[8];
    newDst[9] = m[9];
    newDst[10] = m[10];
    newDst[11] = m[11];
    newDst[12] = m[12];
    newDst[13] = m[13];
    newDst[14] = m[14];
    newDst[15] = m[15];
    return newDst;
  }
  const clone = copy;
  function equalsApproximately(a, b) {
    return Math.abs(a[0] - b[0]) < EPSILON && Math.abs(a[1] - b[1]) < EPSILON && Math.abs(a[2] - b[2]) < EPSILON && Math.abs(a[3] - b[3]) < EPSILON && Math.abs(a[4] - b[4]) < EPSILON && Math.abs(a[5] - b[5]) < EPSILON && Math.abs(a[6] - b[6]) < EPSILON && Math.abs(a[7] - b[7]) < EPSILON && Math.abs(a[8] - b[8]) < EPSILON && Math.abs(a[9] - b[9]) < EPSILON && Math.abs(a[10] - b[10]) < EPSILON && Math.abs(a[11] - b[11]) < EPSILON && Math.abs(a[12] - b[12]) < EPSILON && Math.abs(a[13] - b[13]) < EPSILON && Math.abs(a[14] - b[14]) < EPSILON && Math.abs(a[15] - b[15]) < EPSILON;
  }
  function equals(a, b) {
    return a[0] === b[0] && a[1] === b[1] && a[2] === b[2] && a[3] === b[3] && a[4] === b[4] && a[5] === b[5] && a[6] === b[6] && a[7] === b[7] && a[8] === b[8] && a[9] === b[9] && a[10] === b[10] && a[11] === b[11] && a[12] === b[12] && a[13] === b[13] && a[14] === b[14] && a[15] === b[15];
  }
  function identity(dst) {
    const newDst = dst ?? new Ctor(16);
    newDst[0] = 1;
    newDst[1] = 0;
    newDst[2] = 0;
    newDst[3] = 0;
    newDst[4] = 0;
    newDst[5] = 1;
    newDst[6] = 0;
    newDst[7] = 0;
    newDst[8] = 0;
    newDst[9] = 0;
    newDst[10] = 1;
    newDst[11] = 0;
    newDst[12] = 0;
    newDst[13] = 0;
    newDst[14] = 0;
    newDst[15] = 1;
    return newDst;
  }
  function transpose(m, dst) {
    const newDst = dst ?? new Ctor(16);
    if (newDst === m) {
      let t;
      t = m[1];
      m[1] = m[4];
      m[4] = t;
      t = m[2];
      m[2] = m[8];
      m[8] = t;
      t = m[3];
      m[3] = m[12];
      m[12] = t;
      t = m[6];
      m[6] = m[9];
      m[9] = t;
      t = m[7];
      m[7] = m[13];
      m[13] = t;
      t = m[11];
      m[11] = m[14];
      m[14] = t;
      return newDst;
    }
    const m00 = m[0 * 4 + 0];
    const m01 = m[0 * 4 + 1];
    const m02 = m[0 * 4 + 2];
    const m03 = m[0 * 4 + 3];
    const m10 = m[1 * 4 + 0];
    const m11 = m[1 * 4 + 1];
    const m12 = m[1 * 4 + 2];
    const m13 = m[1 * 4 + 3];
    const m20 = m[2 * 4 + 0];
    const m21 = m[2 * 4 + 1];
    const m22 = m[2 * 4 + 2];
    const m23 = m[2 * 4 + 3];
    const m30 = m[3 * 4 + 0];
    const m31 = m[3 * 4 + 1];
    const m32 = m[3 * 4 + 2];
    const m33 = m[3 * 4 + 3];
    newDst[0] = m00;
    newDst[1] = m10;
    newDst[2] = m20;
    newDst[3] = m30;
    newDst[4] = m01;
    newDst[5] = m11;
    newDst[6] = m21;
    newDst[7] = m31;
    newDst[8] = m02;
    newDst[9] = m12;
    newDst[10] = m22;
    newDst[11] = m32;
    newDst[12] = m03;
    newDst[13] = m13;
    newDst[14] = m23;
    newDst[15] = m33;
    return newDst;
  }
  function inverse(m, dst) {
    const newDst = dst ?? new Ctor(16);
    const m00 = m[0 * 4 + 0];
    const m01 = m[0 * 4 + 1];
    const m02 = m[0 * 4 + 2];
    const m03 = m[0 * 4 + 3];
    const m10 = m[1 * 4 + 0];
    const m11 = m[1 * 4 + 1];
    const m12 = m[1 * 4 + 2];
    const m13 = m[1 * 4 + 3];
    const m20 = m[2 * 4 + 0];
    const m21 = m[2 * 4 + 1];
    const m22 = m[2 * 4 + 2];
    const m23 = m[2 * 4 + 3];
    const m30 = m[3 * 4 + 0];
    const m31 = m[3 * 4 + 1];
    const m32 = m[3 * 4 + 2];
    const m33 = m[3 * 4 + 3];
    const tmp0 = m22 * m33;
    const tmp1 = m32 * m23;
    const tmp2 = m12 * m33;
    const tmp3 = m32 * m13;
    const tmp4 = m12 * m23;
    const tmp5 = m22 * m13;
    const tmp6 = m02 * m33;
    const tmp7 = m32 * m03;
    const tmp8 = m02 * m23;
    const tmp9 = m22 * m03;
    const tmp10 = m02 * m13;
    const tmp11 = m12 * m03;
    const tmp12 = m20 * m31;
    const tmp13 = m30 * m21;
    const tmp14 = m10 * m31;
    const tmp15 = m30 * m11;
    const tmp16 = m10 * m21;
    const tmp17 = m20 * m11;
    const tmp18 = m00 * m31;
    const tmp19 = m30 * m01;
    const tmp20 = m00 * m21;
    const tmp21 = m20 * m01;
    const tmp22 = m00 * m11;
    const tmp23 = m10 * m01;
    const t0 = tmp0 * m11 + tmp3 * m21 + tmp4 * m31 - (tmp1 * m11 + tmp2 * m21 + tmp5 * m31);
    const t1 = tmp1 * m01 + tmp6 * m21 + tmp9 * m31 - (tmp0 * m01 + tmp7 * m21 + tmp8 * m31);
    const t2 = tmp2 * m01 + tmp7 * m11 + tmp10 * m31 - (tmp3 * m01 + tmp6 * m11 + tmp11 * m31);
    const t3 = tmp5 * m01 + tmp8 * m11 + tmp11 * m21 - (tmp4 * m01 + tmp9 * m11 + tmp10 * m21);
    const d = 1 / (m00 * t0 + m10 * t1 + m20 * t2 + m30 * t3);
    newDst[0] = d * t0;
    newDst[1] = d * t1;
    newDst[2] = d * t2;
    newDst[3] = d * t3;
    newDst[4] = d * (tmp1 * m10 + tmp2 * m20 + tmp5 * m30 - (tmp0 * m10 + tmp3 * m20 + tmp4 * m30));
    newDst[5] = d * (tmp0 * m00 + tmp7 * m20 + tmp8 * m30 - (tmp1 * m00 + tmp6 * m20 + tmp9 * m30));
    newDst[6] = d * (tmp3 * m00 + tmp6 * m10 + tmp11 * m30 - (tmp2 * m00 + tmp7 * m10 + tmp10 * m30));
    newDst[7] = d * (tmp4 * m00 + tmp9 * m10 + tmp10 * m20 - (tmp5 * m00 + tmp8 * m10 + tmp11 * m20));
    newDst[8] = d * (tmp12 * m13 + tmp15 * m23 + tmp16 * m33 - (tmp13 * m13 + tmp14 * m23 + tmp17 * m33));
    newDst[9] = d * (tmp13 * m03 + tmp18 * m23 + tmp21 * m33 - (tmp12 * m03 + tmp19 * m23 + tmp20 * m33));
    newDst[10] = d * (tmp14 * m03 + tmp19 * m13 + tmp22 * m33 - (tmp15 * m03 + tmp18 * m13 + tmp23 * m33));
    newDst[11] = d * (tmp17 * m03 + tmp20 * m13 + tmp23 * m23 - (tmp16 * m03 + tmp21 * m13 + tmp22 * m23));
    newDst[12] = d * (tmp14 * m22 + tmp17 * m32 + tmp13 * m12 - (tmp16 * m32 + tmp12 * m12 + tmp15 * m22));
    newDst[13] = d * (tmp20 * m32 + tmp12 * m02 + tmp19 * m22 - (tmp18 * m22 + tmp21 * m32 + tmp13 * m02));
    newDst[14] = d * (tmp18 * m12 + tmp23 * m32 + tmp15 * m02 - (tmp22 * m32 + tmp14 * m02 + tmp19 * m12));
    newDst[15] = d * (tmp22 * m22 + tmp16 * m02 + tmp21 * m12 - (tmp20 * m12 + tmp23 * m22 + tmp17 * m02));
    return newDst;
  }
  function determinant(m) {
    const m00 = m[0 * 4 + 0];
    const m01 = m[0 * 4 + 1];
    const m02 = m[0 * 4 + 2];
    const m03 = m[0 * 4 + 3];
    const m10 = m[1 * 4 + 0];
    const m11 = m[1 * 4 + 1];
    const m12 = m[1 * 4 + 2];
    const m13 = m[1 * 4 + 3];
    const m20 = m[2 * 4 + 0];
    const m21 = m[2 * 4 + 1];
    const m22 = m[2 * 4 + 2];
    const m23 = m[2 * 4 + 3];
    const m30 = m[3 * 4 + 0];
    const m31 = m[3 * 4 + 1];
    const m32 = m[3 * 4 + 2];
    const m33 = m[3 * 4 + 3];
    const tmp0 = m22 * m33;
    const tmp1 = m32 * m23;
    const tmp2 = m12 * m33;
    const tmp3 = m32 * m13;
    const tmp4 = m12 * m23;
    const tmp5 = m22 * m13;
    const tmp6 = m02 * m33;
    const tmp7 = m32 * m03;
    const tmp8 = m02 * m23;
    const tmp9 = m22 * m03;
    const tmp10 = m02 * m13;
    const tmp11 = m12 * m03;
    const t0 = tmp0 * m11 + tmp3 * m21 + tmp4 * m31 - (tmp1 * m11 + tmp2 * m21 + tmp5 * m31);
    const t1 = tmp1 * m01 + tmp6 * m21 + tmp9 * m31 - (tmp0 * m01 + tmp7 * m21 + tmp8 * m31);
    const t2 = tmp2 * m01 + tmp7 * m11 + tmp10 * m31 - (tmp3 * m01 + tmp6 * m11 + tmp11 * m31);
    const t3 = tmp5 * m01 + tmp8 * m11 + tmp11 * m21 - (tmp4 * m01 + tmp9 * m11 + tmp10 * m21);
    return m00 * t0 + m10 * t1 + m20 * t2 + m30 * t3;
  }
  const invert = inverse;
  function multiply(a, b, dst) {
    const newDst = dst ?? new Ctor(16);
    const a00 = a[0];
    const a01 = a[1];
    const a02 = a[2];
    const a03 = a[3];
    const a10 = a[4 + 0];
    const a11 = a[4 + 1];
    const a12 = a[4 + 2];
    const a13 = a[4 + 3];
    const a20 = a[8 + 0];
    const a21 = a[8 + 1];
    const a22 = a[8 + 2];
    const a23 = a[8 + 3];
    const a30 = a[12 + 0];
    const a31 = a[12 + 1];
    const a32 = a[12 + 2];
    const a33 = a[12 + 3];
    const b00 = b[0];
    const b01 = b[1];
    const b02 = b[2];
    const b03 = b[3];
    const b10 = b[4 + 0];
    const b11 = b[4 + 1];
    const b12 = b[4 + 2];
    const b13 = b[4 + 3];
    const b20 = b[8 + 0];
    const b21 = b[8 + 1];
    const b22 = b[8 + 2];
    const b23 = b[8 + 3];
    const b30 = b[12 + 0];
    const b31 = b[12 + 1];
    const b32 = b[12 + 2];
    const b33 = b[12 + 3];
    newDst[0] = a00 * b00 + a10 * b01 + a20 * b02 + a30 * b03;
    newDst[1] = a01 * b00 + a11 * b01 + a21 * b02 + a31 * b03;
    newDst[2] = a02 * b00 + a12 * b01 + a22 * b02 + a32 * b03;
    newDst[3] = a03 * b00 + a13 * b01 + a23 * b02 + a33 * b03;
    newDst[4] = a00 * b10 + a10 * b11 + a20 * b12 + a30 * b13;
    newDst[5] = a01 * b10 + a11 * b11 + a21 * b12 + a31 * b13;
    newDst[6] = a02 * b10 + a12 * b11 + a22 * b12 + a32 * b13;
    newDst[7] = a03 * b10 + a13 * b11 + a23 * b12 + a33 * b13;
    newDst[8] = a00 * b20 + a10 * b21 + a20 * b22 + a30 * b23;
    newDst[9] = a01 * b20 + a11 * b21 + a21 * b22 + a31 * b23;
    newDst[10] = a02 * b20 + a12 * b21 + a22 * b22 + a32 * b23;
    newDst[11] = a03 * b20 + a13 * b21 + a23 * b22 + a33 * b23;
    newDst[12] = a00 * b30 + a10 * b31 + a20 * b32 + a30 * b33;
    newDst[13] = a01 * b30 + a11 * b31 + a21 * b32 + a31 * b33;
    newDst[14] = a02 * b30 + a12 * b31 + a22 * b32 + a32 * b33;
    newDst[15] = a03 * b30 + a13 * b31 + a23 * b32 + a33 * b33;
    return newDst;
  }
  const mul = multiply;
  function setTranslation(a, v, dst) {
    const newDst = dst ?? identity();
    if (a !== newDst) {
      newDst[0] = a[0];
      newDst[1] = a[1];
      newDst[2] = a[2];
      newDst[3] = a[3];
      newDst[4] = a[4];
      newDst[5] = a[5];
      newDst[6] = a[6];
      newDst[7] = a[7];
      newDst[8] = a[8];
      newDst[9] = a[9];
      newDst[10] = a[10];
      newDst[11] = a[11];
    }
    newDst[12] = v[0];
    newDst[13] = v[1];
    newDst[14] = v[2];
    newDst[15] = 1;
    return newDst;
  }
  function getTranslation(m, dst) {
    const newDst = dst ?? vec32.create();
    newDst[0] = m[12];
    newDst[1] = m[13];
    newDst[2] = m[14];
    return newDst;
  }
  function getAxis(m, axis, dst) {
    const newDst = dst ?? vec32.create();
    const off = axis * 4;
    newDst[0] = m[off + 0];
    newDst[1] = m[off + 1];
    newDst[2] = m[off + 2];
    return newDst;
  }
  function setAxis(m, v, axis, dst) {
    const newDst = dst === m ? dst : copy(m, dst);
    const off = axis * 4;
    newDst[off + 0] = v[0];
    newDst[off + 1] = v[1];
    newDst[off + 2] = v[2];
    return newDst;
  }
  function getScaling(m, dst) {
    const newDst = dst ?? vec32.create();
    const xx = m[0];
    const xy = m[1];
    const xz = m[2];
    const yx = m[4];
    const yy = m[5];
    const yz = m[6];
    const zx = m[8];
    const zy = m[9];
    const zz = m[10];
    newDst[0] = Math.sqrt(xx * xx + xy * xy + xz * xz);
    newDst[1] = Math.sqrt(yx * yx + yy * yy + yz * yz);
    newDst[2] = Math.sqrt(zx * zx + zy * zy + zz * zz);
    return newDst;
  }
  function perspective(fieldOfViewYInRadians, aspect, zNear, zFar, dst) {
    const newDst = dst ?? new Ctor(16);
    const f = Math.tan(Math.PI * 0.5 - 0.5 * fieldOfViewYInRadians);
    newDst[0] = f / aspect;
    newDst[1] = 0;
    newDst[2] = 0;
    newDst[3] = 0;
    newDst[4] = 0;
    newDst[5] = f;
    newDst[6] = 0;
    newDst[7] = 0;
    newDst[8] = 0;
    newDst[9] = 0;
    newDst[11] = -1;
    newDst[12] = 0;
    newDst[13] = 0;
    newDst[15] = 0;
    if (Number.isFinite(zFar)) {
      const rangeInv = 1 / (zNear - zFar);
      newDst[10] = zFar * rangeInv;
      newDst[14] = zFar * zNear * rangeInv;
    } else {
      newDst[10] = -1;
      newDst[14] = -zNear;
    }
    return newDst;
  }
  function perspectiveReverseZ(fieldOfViewYInRadians, aspect, zNear, zFar = Infinity, dst) {
    const newDst = dst ?? new Ctor(16);
    const f = 1 / Math.tan(fieldOfViewYInRadians * 0.5);
    newDst[0] = f / aspect;
    newDst[1] = 0;
    newDst[2] = 0;
    newDst[3] = 0;
    newDst[4] = 0;
    newDst[5] = f;
    newDst[6] = 0;
    newDst[7] = 0;
    newDst[8] = 0;
    newDst[9] = 0;
    newDst[11] = -1;
    newDst[12] = 0;
    newDst[13] = 0;
    newDst[15] = 0;
    if (zFar === Infinity) {
      newDst[10] = 0;
      newDst[14] = zNear;
    } else {
      const rangeInv = 1 / (zFar - zNear);
      newDst[10] = zNear * rangeInv;
      newDst[14] = zFar * zNear * rangeInv;
    }
    return newDst;
  }
  function ortho(left, right, bottom, top, near, far, dst) {
    const newDst = dst ?? new Ctor(16);
    newDst[0] = 2 / (right - left);
    newDst[1] = 0;
    newDst[2] = 0;
    newDst[3] = 0;
    newDst[4] = 0;
    newDst[5] = 2 / (top - bottom);
    newDst[6] = 0;
    newDst[7] = 0;
    newDst[8] = 0;
    newDst[9] = 0;
    newDst[10] = 1 / (near - far);
    newDst[11] = 0;
    newDst[12] = (right + left) / (left - right);
    newDst[13] = (top + bottom) / (bottom - top);
    newDst[14] = near / (near - far);
    newDst[15] = 1;
    return newDst;
  }
  function frustum(left, right, bottom, top, near, far, dst) {
    const newDst = dst ?? new Ctor(16);
    const dx = right - left;
    const dy = top - bottom;
    const dz = near - far;
    newDst[0] = 2 * near / dx;
    newDst[1] = 0;
    newDst[2] = 0;
    newDst[3] = 0;
    newDst[4] = 0;
    newDst[5] = 2 * near / dy;
    newDst[6] = 0;
    newDst[7] = 0;
    newDst[8] = (left + right) / dx;
    newDst[9] = (top + bottom) / dy;
    newDst[10] = far / dz;
    newDst[11] = -1;
    newDst[12] = 0;
    newDst[13] = 0;
    newDst[14] = near * far / dz;
    newDst[15] = 0;
    return newDst;
  }
  function frustumReverseZ(left, right, bottom, top, near, far = Infinity, dst) {
    const newDst = dst ?? new Ctor(16);
    const dx = right - left;
    const dy = top - bottom;
    newDst[0] = 2 * near / dx;
    newDst[1] = 0;
    newDst[2] = 0;
    newDst[3] = 0;
    newDst[4] = 0;
    newDst[5] = 2 * near / dy;
    newDst[6] = 0;
    newDst[7] = 0;
    newDst[8] = (left + right) / dx;
    newDst[9] = (top + bottom) / dy;
    newDst[11] = -1;
    newDst[12] = 0;
    newDst[13] = 0;
    newDst[15] = 0;
    if (far === Infinity) {
      newDst[10] = 0;
      newDst[14] = near;
    } else {
      const rangeInv = 1 / (far - near);
      newDst[10] = near * rangeInv;
      newDst[14] = far * near * rangeInv;
    }
    return newDst;
  }
  const xAxis = vec32.create();
  const yAxis = vec32.create();
  const zAxis = vec32.create();
  function aim(position, target, up, dst) {
    const newDst = dst ?? new Ctor(16);
    vec32.normalize(vec32.subtract(target, position, zAxis), zAxis);
    vec32.normalize(vec32.cross(up, zAxis, xAxis), xAxis);
    vec32.normalize(vec32.cross(zAxis, xAxis, yAxis), yAxis);
    newDst[0] = xAxis[0];
    newDst[1] = xAxis[1];
    newDst[2] = xAxis[2];
    newDst[3] = 0;
    newDst[4] = yAxis[0];
    newDst[5] = yAxis[1];
    newDst[6] = yAxis[2];
    newDst[7] = 0;
    newDst[8] = zAxis[0];
    newDst[9] = zAxis[1];
    newDst[10] = zAxis[2];
    newDst[11] = 0;
    newDst[12] = position[0];
    newDst[13] = position[1];
    newDst[14] = position[2];
    newDst[15] = 1;
    return newDst;
  }
  function cameraAim(eye, target, up, dst) {
    const newDst = dst ?? new Ctor(16);
    vec32.normalize(vec32.subtract(eye, target, zAxis), zAxis);
    vec32.normalize(vec32.cross(up, zAxis, xAxis), xAxis);
    vec32.normalize(vec32.cross(zAxis, xAxis, yAxis), yAxis);
    newDst[0] = xAxis[0];
    newDst[1] = xAxis[1];
    newDst[2] = xAxis[2];
    newDst[3] = 0;
    newDst[4] = yAxis[0];
    newDst[5] = yAxis[1];
    newDst[6] = yAxis[2];
    newDst[7] = 0;
    newDst[8] = zAxis[0];
    newDst[9] = zAxis[1];
    newDst[10] = zAxis[2];
    newDst[11] = 0;
    newDst[12] = eye[0];
    newDst[13] = eye[1];
    newDst[14] = eye[2];
    newDst[15] = 1;
    return newDst;
  }
  function lookAt(eye, target, up, dst) {
    const newDst = dst ?? new Ctor(16);
    vec32.normalize(vec32.subtract(eye, target, zAxis), zAxis);
    vec32.normalize(vec32.cross(up, zAxis, xAxis), xAxis);
    vec32.normalize(vec32.cross(zAxis, xAxis, yAxis), yAxis);
    newDst[0] = xAxis[0];
    newDst[1] = yAxis[0];
    newDst[2] = zAxis[0];
    newDst[3] = 0;
    newDst[4] = xAxis[1];
    newDst[5] = yAxis[1];
    newDst[6] = zAxis[1];
    newDst[7] = 0;
    newDst[8] = xAxis[2];
    newDst[9] = yAxis[2];
    newDst[10] = zAxis[2];
    newDst[11] = 0;
    newDst[12] = -(xAxis[0] * eye[0] + xAxis[1] * eye[1] + xAxis[2] * eye[2]);
    newDst[13] = -(yAxis[0] * eye[0] + yAxis[1] * eye[1] + yAxis[2] * eye[2]);
    newDst[14] = -(zAxis[0] * eye[0] + zAxis[1] * eye[1] + zAxis[2] * eye[2]);
    newDst[15] = 1;
    return newDst;
  }
  function translation(v, dst) {
    const newDst = dst ?? new Ctor(16);
    newDst[0] = 1;
    newDst[1] = 0;
    newDst[2] = 0;
    newDst[3] = 0;
    newDst[4] = 0;
    newDst[5] = 1;
    newDst[6] = 0;
    newDst[7] = 0;
    newDst[8] = 0;
    newDst[9] = 0;
    newDst[10] = 1;
    newDst[11] = 0;
    newDst[12] = v[0];
    newDst[13] = v[1];
    newDst[14] = v[2];
    newDst[15] = 1;
    return newDst;
  }
  function translate(m, v, dst) {
    const newDst = dst ?? new Ctor(16);
    const v0 = v[0];
    const v1 = v[1];
    const v2 = v[2];
    const m00 = m[0];
    const m01 = m[1];
    const m02 = m[2];
    const m03 = m[3];
    const m10 = m[1 * 4 + 0];
    const m11 = m[1 * 4 + 1];
    const m12 = m[1 * 4 + 2];
    const m13 = m[1 * 4 + 3];
    const m20 = m[2 * 4 + 0];
    const m21 = m[2 * 4 + 1];
    const m22 = m[2 * 4 + 2];
    const m23 = m[2 * 4 + 3];
    const m30 = m[3 * 4 + 0];
    const m31 = m[3 * 4 + 1];
    const m32 = m[3 * 4 + 2];
    const m33 = m[3 * 4 + 3];
    if (m !== newDst) {
      newDst[0] = m00;
      newDst[1] = m01;
      newDst[2] = m02;
      newDst[3] = m03;
      newDst[4] = m10;
      newDst[5] = m11;
      newDst[6] = m12;
      newDst[7] = m13;
      newDst[8] = m20;
      newDst[9] = m21;
      newDst[10] = m22;
      newDst[11] = m23;
    }
    newDst[12] = m00 * v0 + m10 * v1 + m20 * v2 + m30;
    newDst[13] = m01 * v0 + m11 * v1 + m21 * v2 + m31;
    newDst[14] = m02 * v0 + m12 * v1 + m22 * v2 + m32;
    newDst[15] = m03 * v0 + m13 * v1 + m23 * v2 + m33;
    return newDst;
  }
  function rotationX(angleInRadians, dst) {
    const newDst = dst ?? new Ctor(16);
    const c = Math.cos(angleInRadians);
    const s = Math.sin(angleInRadians);
    newDst[0] = 1;
    newDst[1] = 0;
    newDst[2] = 0;
    newDst[3] = 0;
    newDst[4] = 0;
    newDst[5] = c;
    newDst[6] = s;
    newDst[7] = 0;
    newDst[8] = 0;
    newDst[9] = -s;
    newDst[10] = c;
    newDst[11] = 0;
    newDst[12] = 0;
    newDst[13] = 0;
    newDst[14] = 0;
    newDst[15] = 1;
    return newDst;
  }
  function rotateX(m, angleInRadians, dst) {
    const newDst = dst ?? new Ctor(16);
    const m10 = m[4];
    const m11 = m[5];
    const m12 = m[6];
    const m13 = m[7];
    const m20 = m[8];
    const m21 = m[9];
    const m22 = m[10];
    const m23 = m[11];
    const c = Math.cos(angleInRadians);
    const s = Math.sin(angleInRadians);
    newDst[4] = c * m10 + s * m20;
    newDst[5] = c * m11 + s * m21;
    newDst[6] = c * m12 + s * m22;
    newDst[7] = c * m13 + s * m23;
    newDst[8] = c * m20 - s * m10;
    newDst[9] = c * m21 - s * m11;
    newDst[10] = c * m22 - s * m12;
    newDst[11] = c * m23 - s * m13;
    if (m !== newDst) {
      newDst[0] = m[0];
      newDst[1] = m[1];
      newDst[2] = m[2];
      newDst[3] = m[3];
      newDst[12] = m[12];
      newDst[13] = m[13];
      newDst[14] = m[14];
      newDst[15] = m[15];
    }
    return newDst;
  }
  function rotationY(angleInRadians, dst) {
    const newDst = dst ?? new Ctor(16);
    const c = Math.cos(angleInRadians);
    const s = Math.sin(angleInRadians);
    newDst[0] = c;
    newDst[1] = 0;
    newDst[2] = -s;
    newDst[3] = 0;
    newDst[4] = 0;
    newDst[5] = 1;
    newDst[6] = 0;
    newDst[7] = 0;
    newDst[8] = s;
    newDst[9] = 0;
    newDst[10] = c;
    newDst[11] = 0;
    newDst[12] = 0;
    newDst[13] = 0;
    newDst[14] = 0;
    newDst[15] = 1;
    return newDst;
  }
  function rotateY(m, angleInRadians, dst) {
    const newDst = dst ?? new Ctor(16);
    const m00 = m[0 * 4 + 0];
    const m01 = m[0 * 4 + 1];
    const m02 = m[0 * 4 + 2];
    const m03 = m[0 * 4 + 3];
    const m20 = m[2 * 4 + 0];
    const m21 = m[2 * 4 + 1];
    const m22 = m[2 * 4 + 2];
    const m23 = m[2 * 4 + 3];
    const c = Math.cos(angleInRadians);
    const s = Math.sin(angleInRadians);
    newDst[0] = c * m00 - s * m20;
    newDst[1] = c * m01 - s * m21;
    newDst[2] = c * m02 - s * m22;
    newDst[3] = c * m03 - s * m23;
    newDst[8] = c * m20 + s * m00;
    newDst[9] = c * m21 + s * m01;
    newDst[10] = c * m22 + s * m02;
    newDst[11] = c * m23 + s * m03;
    if (m !== newDst) {
      newDst[4] = m[4];
      newDst[5] = m[5];
      newDst[6] = m[6];
      newDst[7] = m[7];
      newDst[12] = m[12];
      newDst[13] = m[13];
      newDst[14] = m[14];
      newDst[15] = m[15];
    }
    return newDst;
  }
  function rotationZ(angleInRadians, dst) {
    const newDst = dst ?? new Ctor(16);
    const c = Math.cos(angleInRadians);
    const s = Math.sin(angleInRadians);
    newDst[0] = c;
    newDst[1] = s;
    newDst[2] = 0;
    newDst[3] = 0;
    newDst[4] = -s;
    newDst[5] = c;
    newDst[6] = 0;
    newDst[7] = 0;
    newDst[8] = 0;
    newDst[9] = 0;
    newDst[10] = 1;
    newDst[11] = 0;
    newDst[12] = 0;
    newDst[13] = 0;
    newDst[14] = 0;
    newDst[15] = 1;
    return newDst;
  }
  function rotateZ(m, angleInRadians, dst) {
    const newDst = dst ?? new Ctor(16);
    const m00 = m[0 * 4 + 0];
    const m01 = m[0 * 4 + 1];
    const m02 = m[0 * 4 + 2];
    const m03 = m[0 * 4 + 3];
    const m10 = m[1 * 4 + 0];
    const m11 = m[1 * 4 + 1];
    const m12 = m[1 * 4 + 2];
    const m13 = m[1 * 4 + 3];
    const c = Math.cos(angleInRadians);
    const s = Math.sin(angleInRadians);
    newDst[0] = c * m00 + s * m10;
    newDst[1] = c * m01 + s * m11;
    newDst[2] = c * m02 + s * m12;
    newDst[3] = c * m03 + s * m13;
    newDst[4] = c * m10 - s * m00;
    newDst[5] = c * m11 - s * m01;
    newDst[6] = c * m12 - s * m02;
    newDst[7] = c * m13 - s * m03;
    if (m !== newDst) {
      newDst[8] = m[8];
      newDst[9] = m[9];
      newDst[10] = m[10];
      newDst[11] = m[11];
      newDst[12] = m[12];
      newDst[13] = m[13];
      newDst[14] = m[14];
      newDst[15] = m[15];
    }
    return newDst;
  }
  function axisRotation(axis, angleInRadians, dst) {
    const newDst = dst ?? new Ctor(16);
    let x = axis[0];
    let y = axis[1];
    let z = axis[2];
    const n = Math.sqrt(x * x + y * y + z * z);
    x /= n;
    y /= n;
    z /= n;
    const xx = x * x;
    const yy = y * y;
    const zz = z * z;
    const c = Math.cos(angleInRadians);
    const s = Math.sin(angleInRadians);
    const oneMinusCosine = 1 - c;
    newDst[0] = xx + (1 - xx) * c;
    newDst[1] = x * y * oneMinusCosine + z * s;
    newDst[2] = x * z * oneMinusCosine - y * s;
    newDst[3] = 0;
    newDst[4] = x * y * oneMinusCosine - z * s;
    newDst[5] = yy + (1 - yy) * c;
    newDst[6] = y * z * oneMinusCosine + x * s;
    newDst[7] = 0;
    newDst[8] = x * z * oneMinusCosine + y * s;
    newDst[9] = y * z * oneMinusCosine - x * s;
    newDst[10] = zz + (1 - zz) * c;
    newDst[11] = 0;
    newDst[12] = 0;
    newDst[13] = 0;
    newDst[14] = 0;
    newDst[15] = 1;
    return newDst;
  }
  const rotation = axisRotation;
  function axisRotate(m, axis, angleInRadians, dst) {
    const newDst = dst ?? new Ctor(16);
    let x = axis[0];
    let y = axis[1];
    let z = axis[2];
    const n = Math.sqrt(x * x + y * y + z * z);
    x /= n;
    y /= n;
    z /= n;
    const xx = x * x;
    const yy = y * y;
    const zz = z * z;
    const c = Math.cos(angleInRadians);
    const s = Math.sin(angleInRadians);
    const oneMinusCosine = 1 - c;
    const r00 = xx + (1 - xx) * c;
    const r01 = x * y * oneMinusCosine + z * s;
    const r02 = x * z * oneMinusCosine - y * s;
    const r10 = x * y * oneMinusCosine - z * s;
    const r11 = yy + (1 - yy) * c;
    const r12 = y * z * oneMinusCosine + x * s;
    const r20 = x * z * oneMinusCosine + y * s;
    const r21 = y * z * oneMinusCosine - x * s;
    const r22 = zz + (1 - zz) * c;
    const m00 = m[0];
    const m01 = m[1];
    const m02 = m[2];
    const m03 = m[3];
    const m10 = m[4];
    const m11 = m[5];
    const m12 = m[6];
    const m13 = m[7];
    const m20 = m[8];
    const m21 = m[9];
    const m22 = m[10];
    const m23 = m[11];
    newDst[0] = r00 * m00 + r01 * m10 + r02 * m20;
    newDst[1] = r00 * m01 + r01 * m11 + r02 * m21;
    newDst[2] = r00 * m02 + r01 * m12 + r02 * m22;
    newDst[3] = r00 * m03 + r01 * m13 + r02 * m23;
    newDst[4] = r10 * m00 + r11 * m10 + r12 * m20;
    newDst[5] = r10 * m01 + r11 * m11 + r12 * m21;
    newDst[6] = r10 * m02 + r11 * m12 + r12 * m22;
    newDst[7] = r10 * m03 + r11 * m13 + r12 * m23;
    newDst[8] = r20 * m00 + r21 * m10 + r22 * m20;
    newDst[9] = r20 * m01 + r21 * m11 + r22 * m21;
    newDst[10] = r20 * m02 + r21 * m12 + r22 * m22;
    newDst[11] = r20 * m03 + r21 * m13 + r22 * m23;
    if (m !== newDst) {
      newDst[12] = m[12];
      newDst[13] = m[13];
      newDst[14] = m[14];
      newDst[15] = m[15];
    }
    return newDst;
  }
  const rotate2 = axisRotate;
  function scaling(v, dst) {
    const newDst = dst ?? new Ctor(16);
    newDst[0] = v[0];
    newDst[1] = 0;
    newDst[2] = 0;
    newDst[3] = 0;
    newDst[4] = 0;
    newDst[5] = v[1];
    newDst[6] = 0;
    newDst[7] = 0;
    newDst[8] = 0;
    newDst[9] = 0;
    newDst[10] = v[2];
    newDst[11] = 0;
    newDst[12] = 0;
    newDst[13] = 0;
    newDst[14] = 0;
    newDst[15] = 1;
    return newDst;
  }
  function scale(m, v, dst) {
    const newDst = dst ?? new Ctor(16);
    const v0 = v[0];
    const v1 = v[1];
    const v2 = v[2];
    newDst[0] = v0 * m[0 * 4 + 0];
    newDst[1] = v0 * m[0 * 4 + 1];
    newDst[2] = v0 * m[0 * 4 + 2];
    newDst[3] = v0 * m[0 * 4 + 3];
    newDst[4] = v1 * m[1 * 4 + 0];
    newDst[5] = v1 * m[1 * 4 + 1];
    newDst[6] = v1 * m[1 * 4 + 2];
    newDst[7] = v1 * m[1 * 4 + 3];
    newDst[8] = v2 * m[2 * 4 + 0];
    newDst[9] = v2 * m[2 * 4 + 1];
    newDst[10] = v2 * m[2 * 4 + 2];
    newDst[11] = v2 * m[2 * 4 + 3];
    if (m !== newDst) {
      newDst[12] = m[12];
      newDst[13] = m[13];
      newDst[14] = m[14];
      newDst[15] = m[15];
    }
    return newDst;
  }
  function uniformScaling(s, dst) {
    const newDst = dst ?? new Ctor(16);
    newDst[0] = s;
    newDst[1] = 0;
    newDst[2] = 0;
    newDst[3] = 0;
    newDst[4] = 0;
    newDst[5] = s;
    newDst[6] = 0;
    newDst[7] = 0;
    newDst[8] = 0;
    newDst[9] = 0;
    newDst[10] = s;
    newDst[11] = 0;
    newDst[12] = 0;
    newDst[13] = 0;
    newDst[14] = 0;
    newDst[15] = 1;
    return newDst;
  }
  function uniformScale(m, s, dst) {
    const newDst = dst ?? new Ctor(16);
    newDst[0] = s * m[0 * 4 + 0];
    newDst[1] = s * m[0 * 4 + 1];
    newDst[2] = s * m[0 * 4 + 2];
    newDst[3] = s * m[0 * 4 + 3];
    newDst[4] = s * m[1 * 4 + 0];
    newDst[5] = s * m[1 * 4 + 1];
    newDst[6] = s * m[1 * 4 + 2];
    newDst[7] = s * m[1 * 4 + 3];
    newDst[8] = s * m[2 * 4 + 0];
    newDst[9] = s * m[2 * 4 + 1];
    newDst[10] = s * m[2 * 4 + 2];
    newDst[11] = s * m[2 * 4 + 3];
    if (m !== newDst) {
      newDst[12] = m[12];
      newDst[13] = m[13];
      newDst[14] = m[14];
      newDst[15] = m[15];
    }
    return newDst;
  }
  return {
    add,
    aim,
    axisRotate,
    axisRotation,
    cameraAim,
    clone,
    copy,
    create,
    determinant,
    equals,
    equalsApproximately,
    fromMat3,
    fromQuat,
    frustum,
    frustumReverseZ,
    getAxis,
    getScaling,
    getTranslation,
    identity,
    inverse,
    invert,
    lookAt,
    mul,
    mulScalar,
    multiply,
    multiplyScalar,
    negate,
    ortho,
    perspective,
    perspectiveReverseZ,
    rotate: rotate2,
    rotateX,
    rotateY,
    rotateZ,
    rotation,
    rotationX,
    rotationY,
    rotationZ,
    scale,
    scaling,
    set,
    setAxis,
    setTranslation,
    translate,
    translation,
    transpose,
    uniformScale,
    uniformScaling
  };
}
var cache$2 = /* @__PURE__ */ new Map();
function getAPI$2(Ctor) {
  let api = cache$2.get(Ctor);
  if (!api) {
    api = getAPIImpl$2(Ctor);
    cache$2.set(Ctor, api);
  }
  return api;
}
function getAPIImpl$1(Ctor) {
  const vec32 = getAPI$4(Ctor);
  function create(x, y, z, w) {
    const newDst = new Ctor(4);
    if (x !== void 0) {
      newDst[0] = x;
      if (y !== void 0) {
        newDst[1] = y;
        if (z !== void 0) {
          newDst[2] = z;
          if (w !== void 0) {
            newDst[3] = w;
          }
        }
      }
    }
    return newDst;
  }
  const fromValues = create;
  function set(x, y, z, w, dst) {
    const newDst = dst ?? new Ctor(4);
    newDst[0] = x;
    newDst[1] = y;
    newDst[2] = z;
    newDst[3] = w;
    return newDst;
  }
  function fromAxisAngle(axis, angleInRadians, dst) {
    const newDst = dst ?? new Ctor(4);
    const halfAngle = angleInRadians * 0.5;
    const s = Math.sin(halfAngle);
    newDst[0] = s * axis[0];
    newDst[1] = s * axis[1];
    newDst[2] = s * axis[2];
    newDst[3] = Math.cos(halfAngle);
    return newDst;
  }
  function toAxisAngle(q, dst) {
    const newDst = dst ?? vec32.create(3);
    const angle2 = Math.acos(q[3]) * 2;
    const s = Math.sin(angle2 * 0.5);
    if (s > EPSILON) {
      newDst[0] = q[0] / s;
      newDst[1] = q[1] / s;
      newDst[2] = q[2] / s;
    } else {
      newDst[0] = 1;
      newDst[1] = 0;
      newDst[2] = 0;
    }
    return { angle: angle2, axis: newDst };
  }
  function angle(a, b) {
    const d = dot(a, b);
    return Math.acos(2 * d * d - 1);
  }
  function multiply(a, b, dst) {
    const newDst = dst ?? new Ctor(4);
    const ax = a[0];
    const ay = a[1];
    const az = a[2];
    const aw = a[3];
    const bx = b[0];
    const by = b[1];
    const bz = b[2];
    const bw = b[3];
    newDst[0] = ax * bw + aw * bx + ay * bz - az * by;
    newDst[1] = ay * bw + aw * by + az * bx - ax * bz;
    newDst[2] = az * bw + aw * bz + ax * by - ay * bx;
    newDst[3] = aw * bw - ax * bx - ay * by - az * bz;
    return newDst;
  }
  const mul = multiply;
  function rotateX(q, angleInRadians, dst) {
    const newDst = dst ?? new Ctor(4);
    const halfAngle = angleInRadians * 0.5;
    const qx = q[0];
    const qy = q[1];
    const qz = q[2];
    const qw = q[3];
    const bx = Math.sin(halfAngle);
    const bw = Math.cos(halfAngle);
    newDst[0] = qx * bw + qw * bx;
    newDst[1] = qy * bw + qz * bx;
    newDst[2] = qz * bw - qy * bx;
    newDst[3] = qw * bw - qx * bx;
    return newDst;
  }
  function rotateY(q, angleInRadians, dst) {
    const newDst = dst ?? new Ctor(4);
    const halfAngle = angleInRadians * 0.5;
    const qx = q[0];
    const qy = q[1];
    const qz = q[2];
    const qw = q[3];
    const by = Math.sin(halfAngle);
    const bw = Math.cos(halfAngle);
    newDst[0] = qx * bw - qz * by;
    newDst[1] = qy * bw + qw * by;
    newDst[2] = qz * bw + qx * by;
    newDst[3] = qw * bw - qy * by;
    return newDst;
  }
  function rotateZ(q, angleInRadians, dst) {
    const newDst = dst ?? new Ctor(4);
    const halfAngle = angleInRadians * 0.5;
    const qx = q[0];
    const qy = q[1];
    const qz = q[2];
    const qw = q[3];
    const bz = Math.sin(halfAngle);
    const bw = Math.cos(halfAngle);
    newDst[0] = qx * bw + qy * bz;
    newDst[1] = qy * bw - qx * bz;
    newDst[2] = qz * bw + qw * bz;
    newDst[3] = qw * bw - qz * bz;
    return newDst;
  }
  function slerp(a, b, t, dst) {
    const newDst = dst ?? new Ctor(4);
    const ax = a[0];
    const ay = a[1];
    const az = a[2];
    const aw = a[3];
    let bx = b[0];
    let by = b[1];
    let bz = b[2];
    let bw = b[3];
    let cosOmega = ax * bx + ay * by + az * bz + aw * bw;
    if (cosOmega < 0) {
      cosOmega = -cosOmega;
      bx = -bx;
      by = -by;
      bz = -bz;
      bw = -bw;
    }
    let scale0;
    let scale1;
    if (1 - cosOmega > EPSILON) {
      const omega = Math.acos(cosOmega);
      const sinOmega = Math.sin(omega);
      scale0 = Math.sin((1 - t) * omega) / sinOmega;
      scale1 = Math.sin(t * omega) / sinOmega;
    } else {
      scale0 = 1 - t;
      scale1 = t;
    }
    newDst[0] = scale0 * ax + scale1 * bx;
    newDst[1] = scale0 * ay + scale1 * by;
    newDst[2] = scale0 * az + scale1 * bz;
    newDst[3] = scale0 * aw + scale1 * bw;
    return newDst;
  }
  function inverse(q, dst) {
    const newDst = dst ?? new Ctor(4);
    const a0 = q[0];
    const a1 = q[1];
    const a2 = q[2];
    const a3 = q[3];
    const dot2 = a0 * a0 + a1 * a1 + a2 * a2 + a3 * a3;
    const invDot = dot2 ? 1 / dot2 : 0;
    newDst[0] = -a0 * invDot;
    newDst[1] = -a1 * invDot;
    newDst[2] = -a2 * invDot;
    newDst[3] = a3 * invDot;
    return newDst;
  }
  function conjugate(q, dst) {
    const newDst = dst ?? new Ctor(4);
    newDst[0] = -q[0];
    newDst[1] = -q[1];
    newDst[2] = -q[2];
    newDst[3] = q[3];
    return newDst;
  }
  function fromMat(m, dst) {
    const newDst = dst ?? new Ctor(4);
    const trace = m[0] + m[5] + m[10];
    if (trace > 0) {
      const root = Math.sqrt(trace + 1);
      newDst[3] = 0.5 * root;
      const invRoot = 0.5 / root;
      newDst[0] = (m[6] - m[9]) * invRoot;
      newDst[1] = (m[8] - m[2]) * invRoot;
      newDst[2] = (m[1] - m[4]) * invRoot;
    } else {
      let i = 0;
      if (m[5] > m[0]) {
        i = 1;
      }
      if (m[10] > m[i * 4 + i]) {
        i = 2;
      }
      const j = (i + 1) % 3;
      const k = (i + 2) % 3;
      const root = Math.sqrt(m[i * 4 + i] - m[j * 4 + j] - m[k * 4 + k] + 1);
      newDst[i] = 0.5 * root;
      const invRoot = 0.5 / root;
      newDst[3] = (m[j * 4 + k] - m[k * 4 + j]) * invRoot;
      newDst[j] = (m[j * 4 + i] + m[i * 4 + j]) * invRoot;
      newDst[k] = (m[k * 4 + i] + m[i * 4 + k]) * invRoot;
    }
    return newDst;
  }
  function fromEuler(xAngleInRadians, yAngleInRadians, zAngleInRadians, order, dst) {
    const newDst = dst ?? new Ctor(4);
    const xHalfAngle = xAngleInRadians * 0.5;
    const yHalfAngle = yAngleInRadians * 0.5;
    const zHalfAngle = zAngleInRadians * 0.5;
    const sx = Math.sin(xHalfAngle);
    const cx = Math.cos(xHalfAngle);
    const sy = Math.sin(yHalfAngle);
    const cy = Math.cos(yHalfAngle);
    const sz = Math.sin(zHalfAngle);
    const cz = Math.cos(zHalfAngle);
    switch (order) {
      case "xyz":
        newDst[0] = sx * cy * cz + cx * sy * sz;
        newDst[1] = cx * sy * cz - sx * cy * sz;
        newDst[2] = cx * cy * sz + sx * sy * cz;
        newDst[3] = cx * cy * cz - sx * sy * sz;
        break;
      case "xzy":
        newDst[0] = sx * cy * cz - cx * sy * sz;
        newDst[1] = cx * sy * cz - sx * cy * sz;
        newDst[2] = cx * cy * sz + sx * sy * cz;
        newDst[3] = cx * cy * cz + sx * sy * sz;
        break;
      case "yxz":
        newDst[0] = sx * cy * cz + cx * sy * sz;
        newDst[1] = cx * sy * cz - sx * cy * sz;
        newDst[2] = cx * cy * sz - sx * sy * cz;
        newDst[3] = cx * cy * cz + sx * sy * sz;
        break;
      case "yzx":
        newDst[0] = sx * cy * cz + cx * sy * sz;
        newDst[1] = cx * sy * cz + sx * cy * sz;
        newDst[2] = cx * cy * sz - sx * sy * cz;
        newDst[3] = cx * cy * cz - sx * sy * sz;
        break;
      case "zxy":
        newDst[0] = sx * cy * cz - cx * sy * sz;
        newDst[1] = cx * sy * cz + sx * cy * sz;
        newDst[2] = cx * cy * sz + sx * sy * cz;
        newDst[3] = cx * cy * cz - sx * sy * sz;
        break;
      case "zyx":
        newDst[0] = sx * cy * cz - cx * sy * sz;
        newDst[1] = cx * sy * cz + sx * cy * sz;
        newDst[2] = cx * cy * sz - sx * sy * cz;
        newDst[3] = cx * cy * cz + sx * sy * sz;
        break;
      default:
        throw new Error(`Unknown rotation order: ${order}`);
    }
    return newDst;
  }
  function copy(q, dst) {
    const newDst = dst ?? new Ctor(4);
    newDst[0] = q[0];
    newDst[1] = q[1];
    newDst[2] = q[2];
    newDst[3] = q[3];
    return newDst;
  }
  const clone = copy;
  function add(a, b, dst) {
    const newDst = dst ?? new Ctor(4);
    newDst[0] = a[0] + b[0];
    newDst[1] = a[1] + b[1];
    newDst[2] = a[2] + b[2];
    newDst[3] = a[3] + b[3];
    return newDst;
  }
  function subtract(a, b, dst) {
    const newDst = dst ?? new Ctor(4);
    newDst[0] = a[0] - b[0];
    newDst[1] = a[1] - b[1];
    newDst[2] = a[2] - b[2];
    newDst[3] = a[3] - b[3];
    return newDst;
  }
  const sub = subtract;
  function mulScalar(v, k, dst) {
    const newDst = dst ?? new Ctor(4);
    newDst[0] = v[0] * k;
    newDst[1] = v[1] * k;
    newDst[2] = v[2] * k;
    newDst[3] = v[3] * k;
    return newDst;
  }
  const scale = mulScalar;
  function divScalar(v, k, dst) {
    const newDst = dst ?? new Ctor(4);
    newDst[0] = v[0] / k;
    newDst[1] = v[1] / k;
    newDst[2] = v[2] / k;
    newDst[3] = v[3] / k;
    return newDst;
  }
  function dot(a, b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
  }
  function lerp2(a, b, t, dst) {
    const newDst = dst ?? new Ctor(4);
    newDst[0] = a[0] + t * (b[0] - a[0]);
    newDst[1] = a[1] + t * (b[1] - a[1]);
    newDst[2] = a[2] + t * (b[2] - a[2]);
    newDst[3] = a[3] + t * (b[3] - a[3]);
    return newDst;
  }
  function length(v) {
    const v0 = v[0];
    const v1 = v[1];
    const v2 = v[2];
    const v3 = v[3];
    return Math.sqrt(v0 * v0 + v1 * v1 + v2 * v2 + v3 * v3);
  }
  const len = length;
  function lengthSq(v) {
    const v0 = v[0];
    const v1 = v[1];
    const v2 = v[2];
    const v3 = v[3];
    return v0 * v0 + v1 * v1 + v2 * v2 + v3 * v3;
  }
  const lenSq = lengthSq;
  function normalize(v, dst) {
    const newDst = dst ?? new Ctor(4);
    const v0 = v[0];
    const v1 = v[1];
    const v2 = v[2];
    const v3 = v[3];
    const len2 = Math.sqrt(v0 * v0 + v1 * v1 + v2 * v2 + v3 * v3);
    if (len2 > 1e-5) {
      newDst[0] = v0 / len2;
      newDst[1] = v1 / len2;
      newDst[2] = v2 / len2;
      newDst[3] = v3 / len2;
    } else {
      newDst[0] = 0;
      newDst[1] = 0;
      newDst[2] = 0;
      newDst[3] = 1;
    }
    return newDst;
  }
  function equalsApproximately(a, b) {
    return Math.abs(a[0] - b[0]) < EPSILON && Math.abs(a[1] - b[1]) < EPSILON && Math.abs(a[2] - b[2]) < EPSILON && Math.abs(a[3] - b[3]) < EPSILON;
  }
  function equals(a, b) {
    return a[0] === b[0] && a[1] === b[1] && a[2] === b[2] && a[3] === b[3];
  }
  function identity(dst) {
    const newDst = dst ?? new Ctor(4);
    newDst[0] = 0;
    newDst[1] = 0;
    newDst[2] = 0;
    newDst[3] = 1;
    return newDst;
  }
  const tempVec3 = vec32.create();
  const xUnitVec3 = vec32.create();
  const yUnitVec3 = vec32.create();
  function rotationTo(aUnit, bUnit, dst) {
    const newDst = dst ?? new Ctor(4);
    const dot2 = vec32.dot(aUnit, bUnit);
    if (dot2 < -0.999999) {
      vec32.cross(xUnitVec3, aUnit, tempVec3);
      if (vec32.len(tempVec3) < 1e-6) {
        vec32.cross(yUnitVec3, aUnit, tempVec3);
      }
      vec32.normalize(tempVec3, tempVec3);
      fromAxisAngle(tempVec3, Math.PI, newDst);
      return newDst;
    } else if (dot2 > 0.999999) {
      newDst[0] = 0;
      newDst[1] = 0;
      newDst[2] = 0;
      newDst[3] = 1;
      return newDst;
    } else {
      vec32.cross(aUnit, bUnit, tempVec3);
      newDst[0] = tempVec3[0];
      newDst[1] = tempVec3[1];
      newDst[2] = tempVec3[2];
      newDst[3] = 1 + dot2;
      return normalize(newDst, newDst);
    }
  }
  const tempQuat1 = new Ctor(4);
  const tempQuat2 = new Ctor(4);
  function sqlerp(a, b, c, d, t, dst) {
    const newDst = dst ?? new Ctor(4);
    slerp(a, d, t, tempQuat1);
    slerp(b, c, t, tempQuat2);
    slerp(tempQuat1, tempQuat2, 2 * t * (1 - t), newDst);
    return newDst;
  }
  return {
    create,
    fromValues,
    set,
    fromAxisAngle,
    toAxisAngle,
    angle,
    multiply,
    mul,
    rotateX,
    rotateY,
    rotateZ,
    slerp,
    inverse,
    conjugate,
    fromMat,
    fromEuler,
    copy,
    clone,
    add,
    subtract,
    sub,
    mulScalar,
    scale,
    divScalar,
    dot,
    lerp: lerp2,
    length,
    len,
    lengthSq,
    lenSq,
    normalize,
    equalsApproximately,
    equals,
    identity,
    rotationTo,
    sqlerp
  };
}
var cache$1 = /* @__PURE__ */ new Map();
function getAPI$1(Ctor) {
  let api = cache$1.get(Ctor);
  if (!api) {
    api = getAPIImpl$1(Ctor);
    cache$1.set(Ctor, api);
  }
  return api;
}
function getAPIImpl(Ctor) {
  function create(x, y, z, w) {
    const newDst = new Ctor(4);
    if (x !== void 0) {
      newDst[0] = x;
      if (y !== void 0) {
        newDst[1] = y;
        if (z !== void 0) {
          newDst[2] = z;
          if (w !== void 0) {
            newDst[3] = w;
          }
        }
      }
    }
    return newDst;
  }
  const fromValues = create;
  function set(x, y, z, w, dst) {
    const newDst = dst ?? new Ctor(4);
    newDst[0] = x;
    newDst[1] = y;
    newDst[2] = z;
    newDst[3] = w;
    return newDst;
  }
  function ceil(v, dst) {
    const newDst = dst ?? new Ctor(4);
    newDst[0] = Math.ceil(v[0]);
    newDst[1] = Math.ceil(v[1]);
    newDst[2] = Math.ceil(v[2]);
    newDst[3] = Math.ceil(v[3]);
    return newDst;
  }
  function floor(v, dst) {
    const newDst = dst ?? new Ctor(4);
    newDst[0] = Math.floor(v[0]);
    newDst[1] = Math.floor(v[1]);
    newDst[2] = Math.floor(v[2]);
    newDst[3] = Math.floor(v[3]);
    return newDst;
  }
  function round(v, dst) {
    const newDst = dst ?? new Ctor(4);
    newDst[0] = Math.round(v[0]);
    newDst[1] = Math.round(v[1]);
    newDst[2] = Math.round(v[2]);
    newDst[3] = Math.round(v[3]);
    return newDst;
  }
  function clamp2(v, min2 = 0, max2 = 1, dst) {
    const newDst = dst ?? new Ctor(4);
    newDst[0] = Math.min(max2, Math.max(min2, v[0]));
    newDst[1] = Math.min(max2, Math.max(min2, v[1]));
    newDst[2] = Math.min(max2, Math.max(min2, v[2]));
    newDst[3] = Math.min(max2, Math.max(min2, v[3]));
    return newDst;
  }
  function add(a, b, dst) {
    const newDst = dst ?? new Ctor(4);
    newDst[0] = a[0] + b[0];
    newDst[1] = a[1] + b[1];
    newDst[2] = a[2] + b[2];
    newDst[3] = a[3] + b[3];
    return newDst;
  }
  function addScaled(a, b, scale2, dst) {
    const newDst = dst ?? new Ctor(4);
    newDst[0] = a[0] + b[0] * scale2;
    newDst[1] = a[1] + b[1] * scale2;
    newDst[2] = a[2] + b[2] * scale2;
    newDst[3] = a[3] + b[3] * scale2;
    return newDst;
  }
  function subtract(a, b, dst) {
    const newDst = dst ?? new Ctor(4);
    newDst[0] = a[0] - b[0];
    newDst[1] = a[1] - b[1];
    newDst[2] = a[2] - b[2];
    newDst[3] = a[3] - b[3];
    return newDst;
  }
  const sub = subtract;
  function equalsApproximately(a, b) {
    return Math.abs(a[0] - b[0]) < EPSILON && Math.abs(a[1] - b[1]) < EPSILON && Math.abs(a[2] - b[2]) < EPSILON && Math.abs(a[3] - b[3]) < EPSILON;
  }
  function equals(a, b) {
    return a[0] === b[0] && a[1] === b[1] && a[2] === b[2] && a[3] === b[3];
  }
  function lerp2(a, b, t, dst) {
    const newDst = dst ?? new Ctor(4);
    newDst[0] = a[0] + t * (b[0] - a[0]);
    newDst[1] = a[1] + t * (b[1] - a[1]);
    newDst[2] = a[2] + t * (b[2] - a[2]);
    newDst[3] = a[3] + t * (b[3] - a[3]);
    return newDst;
  }
  function lerpV(a, b, t, dst) {
    const newDst = dst ?? new Ctor(4);
    newDst[0] = a[0] + t[0] * (b[0] - a[0]);
    newDst[1] = a[1] + t[1] * (b[1] - a[1]);
    newDst[2] = a[2] + t[2] * (b[2] - a[2]);
    newDst[3] = a[3] + t[3] * (b[3] - a[3]);
    return newDst;
  }
  function max(a, b, dst) {
    const newDst = dst ?? new Ctor(4);
    newDst[0] = Math.max(a[0], b[0]);
    newDst[1] = Math.max(a[1], b[1]);
    newDst[2] = Math.max(a[2], b[2]);
    newDst[3] = Math.max(a[3], b[3]);
    return newDst;
  }
  function min(a, b, dst) {
    const newDst = dst ?? new Ctor(4);
    newDst[0] = Math.min(a[0], b[0]);
    newDst[1] = Math.min(a[1], b[1]);
    newDst[2] = Math.min(a[2], b[2]);
    newDst[3] = Math.min(a[3], b[3]);
    return newDst;
  }
  function mulScalar(v, k, dst) {
    const newDst = dst ?? new Ctor(4);
    newDst[0] = v[0] * k;
    newDst[1] = v[1] * k;
    newDst[2] = v[2] * k;
    newDst[3] = v[3] * k;
    return newDst;
  }
  const scale = mulScalar;
  function divScalar(v, k, dst) {
    const newDst = dst ?? new Ctor(4);
    newDst[0] = v[0] / k;
    newDst[1] = v[1] / k;
    newDst[2] = v[2] / k;
    newDst[3] = v[3] / k;
    return newDst;
  }
  function inverse(v, dst) {
    const newDst = dst ?? new Ctor(4);
    newDst[0] = 1 / v[0];
    newDst[1] = 1 / v[1];
    newDst[2] = 1 / v[2];
    newDst[3] = 1 / v[3];
    return newDst;
  }
  const invert = inverse;
  function dot(a, b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
  }
  function length(v) {
    const v0 = v[0];
    const v1 = v[1];
    const v2 = v[2];
    const v3 = v[3];
    return Math.sqrt(v0 * v0 + v1 * v1 + v2 * v2 + v3 * v3);
  }
  const len = length;
  function lengthSq(v) {
    const v0 = v[0];
    const v1 = v[1];
    const v2 = v[2];
    const v3 = v[3];
    return v0 * v0 + v1 * v1 + v2 * v2 + v3 * v3;
  }
  const lenSq = lengthSq;
  function distance(a, b) {
    const dx = a[0] - b[0];
    const dy = a[1] - b[1];
    const dz = a[2] - b[2];
    const dw = a[3] - b[3];
    return Math.sqrt(dx * dx + dy * dy + dz * dz + dw * dw);
  }
  const dist = distance;
  function distanceSq(a, b) {
    const dx = a[0] - b[0];
    const dy = a[1] - b[1];
    const dz = a[2] - b[2];
    const dw = a[3] - b[3];
    return dx * dx + dy * dy + dz * dz + dw * dw;
  }
  const distSq = distanceSq;
  function normalize(v, dst) {
    const newDst = dst ?? new Ctor(4);
    const v0 = v[0];
    const v1 = v[1];
    const v2 = v[2];
    const v3 = v[3];
    const len2 = Math.sqrt(v0 * v0 + v1 * v1 + v2 * v2 + v3 * v3);
    if (len2 > 1e-5) {
      newDst[0] = v0 / len2;
      newDst[1] = v1 / len2;
      newDst[2] = v2 / len2;
      newDst[3] = v3 / len2;
    } else {
      newDst[0] = 0;
      newDst[1] = 0;
      newDst[2] = 0;
      newDst[3] = 0;
    }
    return newDst;
  }
  function negate(v, dst) {
    const newDst = dst ?? new Ctor(4);
    newDst[0] = -v[0];
    newDst[1] = -v[1];
    newDst[2] = -v[2];
    newDst[3] = -v[3];
    return newDst;
  }
  function copy(v, dst) {
    const newDst = dst ?? new Ctor(4);
    newDst[0] = v[0];
    newDst[1] = v[1];
    newDst[2] = v[2];
    newDst[3] = v[3];
    return newDst;
  }
  const clone = copy;
  function multiply(a, b, dst) {
    const newDst = dst ?? new Ctor(4);
    newDst[0] = a[0] * b[0];
    newDst[1] = a[1] * b[1];
    newDst[2] = a[2] * b[2];
    newDst[3] = a[3] * b[3];
    return newDst;
  }
  const mul = multiply;
  function divide(a, b, dst) {
    const newDst = dst ?? new Ctor(4);
    newDst[0] = a[0] / b[0];
    newDst[1] = a[1] / b[1];
    newDst[2] = a[2] / b[2];
    newDst[3] = a[3] / b[3];
    return newDst;
  }
  const div = divide;
  function zero(dst) {
    const newDst = dst ?? new Ctor(4);
    newDst[0] = 0;
    newDst[1] = 0;
    newDst[2] = 0;
    newDst[3] = 0;
    return newDst;
  }
  function transformMat4(v, m, dst) {
    const newDst = dst ?? new Ctor(4);
    const x = v[0];
    const y = v[1];
    const z = v[2];
    const w = v[3];
    newDst[0] = m[0] * x + m[4] * y + m[8] * z + m[12] * w;
    newDst[1] = m[1] * x + m[5] * y + m[9] * z + m[13] * w;
    newDst[2] = m[2] * x + m[6] * y + m[10] * z + m[14] * w;
    newDst[3] = m[3] * x + m[7] * y + m[11] * z + m[15] * w;
    return newDst;
  }
  function setLength(a, len2, dst) {
    const newDst = dst ?? new Ctor(4);
    normalize(a, newDst);
    return mulScalar(newDst, len2, newDst);
  }
  function truncate(a, maxLen, dst) {
    const newDst = dst ?? new Ctor(4);
    if (length(a) > maxLen) {
      return setLength(a, maxLen, newDst);
    }
    return copy(a, newDst);
  }
  function midpoint(a, b, dst) {
    const newDst = dst ?? new Ctor(4);
    return lerp2(a, b, 0.5, newDst);
  }
  return {
    create,
    fromValues,
    set,
    ceil,
    floor,
    round,
    clamp: clamp2,
    add,
    addScaled,
    subtract,
    sub,
    equalsApproximately,
    equals,
    lerp: lerp2,
    lerpV,
    max,
    min,
    mulScalar,
    scale,
    divScalar,
    inverse,
    invert,
    dot,
    length,
    len,
    lengthSq,
    lenSq,
    distance,
    dist,
    distanceSq,
    distSq,
    normalize,
    negate,
    copy,
    clone,
    multiply,
    mul,
    divide,
    div,
    zero,
    transformMat4,
    setLength,
    truncate,
    midpoint
  };
}
var cache = /* @__PURE__ */ new Map();
function getAPI(Ctor) {
  let api = cache.get(Ctor);
  if (!api) {
    api = getAPIImpl(Ctor);
    cache.set(Ctor, api);
  }
  return api;
}
function wgpuMatrixAPI(Mat3Ctor, Mat4Ctor, QuatCtor, Vec2Ctor, Vec3Ctor, Vec4Ctor) {
  return {
    /** @namespace mat3 */
    mat3: getAPI$3(Mat3Ctor),
    /** @namespace mat4 */
    mat4: getAPI$2(Mat4Ctor),
    /** @namespace quat */
    quat: getAPI$1(QuatCtor),
    /** @namespace vec2 */
    vec2: getAPI$5(Vec2Ctor),
    /** @namespace vec3 */
    vec3: getAPI$4(Vec3Ctor),
    /** @namespace vec4 */
    vec4: getAPI(Vec4Ctor)
  };
}
var {
  /**
   * 3x3 Matrix functions that default to returning `Float32Array`
   * @namespace
   */
  mat3,
  /**
   * 4x4 Matrix functions that default to returning `Float32Array`
   * @namespace
   */
  mat4,
  /**
   * Quaternion functions that default to returning `Float32Array`
   * @namespace
   */
  quat,
  /**
   * Vec2 functions that default to returning `Float32Array`
   * @namespace
   */
  vec2,
  /**
   * Vec3 functions that default to returning `Float32Array`
   * @namespace
   */
  vec3,
  /**
   * Vec3 functions that default to returning `Float32Array`
   * @namespace
   */
  vec4
} = wgpuMatrixAPI(Float32Array, Float32Array, Float32Array, Float32Array, Float32Array, Float32Array);
var {
  /**
   * 3x3 Matrix functions that default to returning `Float64Array`
   * @namespace
   */
  mat3: mat3d,
  /**
   * 4x4 Matrix functions that default to returning `Float64Array`
   * @namespace
   */
  mat4: mat4d,
  /**
   * Quaternion functions that default to returning `Float64Array`
   * @namespace
   */
  quat: quatd,
  /**
   * Vec2 functions that default to returning `Float64Array`
   * @namespace
   */
  vec2: vec2d,
  /**
   * Vec3 functions that default to returning `Float64Array`
   * @namespace
   */
  vec3: vec3d,
  /**
   * Vec3 functions that default to returning `Float64Array`
   * @namespace
   */
  vec4: vec4d
} = wgpuMatrixAPI(Float64Array, Float64Array, Float64Array, Float64Array, Float64Array, Float64Array);
var {
  /**
   * 3x3 Matrix functions that default to returning `number[]`
   * @namespace
   */
  mat3: mat3n,
  /**
   * 4x4 Matrix functions that default to returning `number[]`
   * @namespace
   */
  mat4: mat4n,
  /**
   * Quaternion functions that default to returning `number[]`
   * @namespace
   */
  quat: quatn,
  /**
   * Vec2 functions that default to returning `number[]`
   * @namespace
   */
  vec2: vec2n,
  /**
   * Vec3 functions that default to returning `number[]`
   * @namespace
   */
  vec3: vec3n,
  /**
   * Vec3 functions that default to returning `number[]`
   * @namespace
   */
  vec4: vec4n
} = wgpuMatrixAPI(ZeroArray, Array, Array, Array, Array, Array);

// blit.wgsl
var blit_default = "struct VertexInput {\n    @location(0) position: vec2f,\n}\n\nstruct VertexOutput {\n    @builtin(position) pos: vec4f,\n    @location(0) uv: vec2f,\n}\n\nstruct FragmentInput {\n    @location(0) uv: vec2f,\n}\n\n@group(0) @binding(0) var raytracedTexture: texture_2d<f32>;\n@group(0) @binding(1) var textureSampler: sampler;\n\n@vertex\nfn vertexMain(input: VertexInput) -> VertexOutput {\n    // Convert from [-1,1] to [0,1] for UV coordinates\n    let uv = input.position * 0.5 + 0.5;\n    return VertexOutput(\n        vec4f(input.position, 0.0, 1.0),\n        uv\n    );\n}\n\n@fragment\nfn fragmentMain(\n    input: FragmentInput,\n) -> @location(0) vec4f {\n    let color = textureSample(raytracedTexture, textureSampler, input.uv);\n    // GPU hardware handles filtering and edge cases automatically\n    return color;\n}\n";

// compute.wgsl
var compute_default = "struct Input {\n    camera_matrix: mat4x4f,\n    fov_scale: f32, // tan(fov * 0.5)\n    time_delta: f32,\n    _pad: vec2u,\n    transform_matrix: mat4x4f,\n    transform_inverse_matrix: mat4x4f,\n}\n\n@group(0) @binding(0) var outputTexture: texture_storage_2d<rgba8unorm, write>;\n@group(0) @binding(1) var<uniform> input: Input;\n@group(0) @binding(2) var<storage> picovdb_grids: array<PicoVDBGrid>;\n@group(0) @binding(3) var<storage> picovdb_roots: array<PicoVDBRoot>;\n@group(0) @binding(4) var<storage> picovdb_uppers: array<PicoVDBUpper>;\n@group(0) @binding(5) var<storage> picovdb_lowers: array<PicoVDBLower>;\n@group(0) @binding(6) var<storage> picovdb_leaves: array<PicoVDBLeaf>;\n@group(0) @binding(7) var<storage> picovdb_buffer: array<u32>;\n\nstruct RayHit {\n    distance: f32,\n    normal: vec3f,\n}\n\nfn intersect_picovdb(\n    world_ray: Ray,\n    world_to_index: mat4x4f,\n    index_to_world: mat4x4f,\n) -> RayHit {\n    let grid = picovdb_grids[0];\n    let idx_origin = (world_to_index * vec4f(world_ray.origin, 1.0)).xyz;\n    let idx_dir_unnorm = (world_to_index * vec4f(world_ray.direction, 0.0)).xyz;\n    let idx_dir_len = length(idx_dir_unnorm);\n    let idx_direction = idx_dir_unnorm / idx_dir_len;\n\n    let index_ray = Ray(idx_origin, idx_direction);\n    let tmin = 0.0;\n    let tmax = 10000.0; // TODO: set max world clip distance.\n\n    var accessor: PicoVDBReadAccessor;\n    picovdbReadAccessorInit(&accessor, 0);\n\n    // Inside Check (Works even if camera is in background space)\n    let start_val = picovdbSampleTrilinear(&accessor, grid, idx_origin);\n    if start_val < 0.0 {\n        return RayHit(0.01, -world_ray.direction);\n    }\n\n    var hit_t_idx: f32;\n    var hit_v: f32;\n    let hit = picovdbHDDAZeroCrossing(\n        &accessor, grid, index_ray.origin, tmin, index_ray.direction, tmax, &hit_t_idx, &hit_v,\n    );\n    if !hit { return RayHit(-1.0, vec3f(0)); }\n\n    let index_hit_point = index_ray.origin + index_ray.direction * hit_t_idx;\n    let world_hit_point = (index_to_world * vec4f(index_hit_point, 1.0)).xyz;\n    let world_distance = length(world_hit_point - world_ray.origin);\n\n    let stencil = picovdbSampleStencil(&accessor, grid, vec3i(floor(index_hit_point)));\n    let index_gradient = picovdbTrilinearGradient(fract(index_hit_point), stencil);\n    let normal = normalize((index_to_world * vec4f(index_gradient, 0.0)).xyz);\n\n    return RayHit(world_distance, normal);\n}\n\nstruct Ray {\n    origin: vec3f,\n    direction: vec3f,\n}\n\n// Ground plane intersection (only visible from above)\nfn intersect_ground_plane(ray: Ray, plane_y: f32) -> f32 {\n    if ray.direction.y >= 0.0 || abs(ray.direction.y) < 0.001 {\n        return -1.0;\n    }\n    let t = (plane_y - ray.origin.y) / ray.direction.y;\n    return select(-1.0, t, t > 0.001);\n}\n\nfn raymarch_scene_graph(ray: Ray) -> vec3f {\n    let volume_hit = intersect_picovdb(ray, input.transform_matrix, input.transform_inverse_matrix);\n    let ground_t = intersect_ground_plane(ray, -2.0);\n    \n    let background = vec3f(0.95, 0.95, 1.0);\n    let light_pos = vec3f(20.0, 30.0, 10.0);\n\n    // Determine primary hit\n    var t = 1e6f;\n    var is_volume = false;\n    if volume_hit.distance > 0.0 { t = volume_hit.distance; is_volume = true; }\n    if ground_t > 0.0 && ground_t < t { t = ground_t; is_volume = false; }\n\n    if t > 1e5f { return background; }\n\n    let hit_point = ray.origin + ray.direction * t;\n    let light_vec = light_pos - hit_point;\n    let light_dir = normalize(light_vec);\n    let light_dist = length(light_vec);\n\n    if is_volume {\n        var diffuse = 0.5 + 0.5 * dot(volume_hit.normal, light_dir);\n        diffuse = diffuse * diffuse;\n        \n        // Use the optimized occlusion check for shadows\n        var acc: PicoVDBReadAccessor; picovdbReadAccessorInit(&acc, 0);\n        let in_shadow = picovdbHDDAIsOccluded(\n            &acc, picovdb_grids[0],\n            (input.transform_matrix * vec4f(hit_point + volume_hit.normal * 0.1, 1.0)).xyz,\n            0.0,\n            (input.transform_matrix * vec4f(light_dir, 0.0)).xyz,\n            light_dist\n        );\n        let shadow = select(1.0, 0.8, in_shadow);\n        return vec3f(0.2, 0.5, 1.0) * (diffuse * shadow);\n    } else {\n        // Simple ground shadow\n        var acc: PicoVDBReadAccessor; picovdbReadAccessorInit(&acc, 0);\n        let in_shadow = picovdbHDDAIsOccluded(\n            &acc, picovdb_grids[0],\n            (input.transform_matrix * vec4f(hit_point + vec3f(0, 0.1, 0), 1.0)).xyz,\n            0.0,\n            (input.transform_matrix * vec4f(light_dir, 0.0)).xyz,\n            light_dist\n        );\n        let fade = 1.0 - smoothstep(20.0, 30.0, length(hit_point.xz));\n        return mix(background, vec3f(0.3), select(0.0, fade * 0.7, in_shadow));\n    }\n}\n\nfn generate_camera_ray(screen_coord: vec2f, screen_size: vec2f) -> Ray {\n    // Convert to normalized coordinates [-1, 1]\n    let uv = (screen_coord / screen_size) * 2.0 - 1.0;\n    \n    // Calculate aspect ratio\n    let aspect_ratio = screen_size.x / screen_size.y;\n\n    // Extract camera basis vectors from view matrix\n    let right: vec3f = input.camera_matrix[0].xyz;\n    let up: vec3f = input.camera_matrix[1].xyz;\n    let forward: vec3f = -input.camera_matrix[2].xyz;\n\n    // Extract camera position\n    let camera_pos: vec3f = input.camera_matrix[3].xyz;\n    \n    // Calculate ray direction\n    let ray_direction = normalize(\n        forward + uv.x * right * aspect_ratio * input.fov_scale + uv.y * up * input.fov_scale\n    );\n    return Ray(camera_pos, ray_direction);\n}\n\n@compute @workgroup_size(16, 16)\nfn computeMain(@builtin(global_invocation_id) global_id: vec3u) {\n    let screen_size = textureDimensions(outputTexture);\n    \n    // Early exit for out-of-bounds threads\n    if global_id.x >= screen_size.x || global_id.y >= screen_size.y {\n        return;\n    }\n\n    // Generate ray\n    let ray = generate_camera_ray(vec2f(global_id.xy) + 0.5, vec2f(screen_size));\n    \n    // Raymarching using scene graph\n    let color = raymarch_scene_graph(ray);\n    \n    // Write result\n    textureStore(outputTexture, global_id.xy, vec4f(color, 1.0));\n}\n";

// ../picovdb.wgsl
var picovdb_default = `
//@group(0) @binding(0) var<storage> picovdb_grids: array<PicoVDBGrid>;
//@group(0) @binding(1) var<storage> picovdb_roots: array<PicoVDBRoot>;
//@group(0) @binding(2) var<storage> picovdb_uppers: array<PicoVDBUpper>;
//@group(0) @binding(3) var<storage> picovdb_lowers: array<PicoVDBLower>;
//@group(0) @binding(4) var<storage> picovdb_leaves: array<PicoVDBLeaf>;
//@group(0) @binding(5) var<storage> picovdb_buffer: array<u32>;

struct PicoVDBFileHeader {
  magic: vec2u,    // 'PicoVDB0' little endian (8 bytes)
  version: u32,    // Format version (4 bytes)
  gridCount: u32,  // Number of grids (4 bytes)
  upperCount: u32, // Total upper nodes (4 bytes)
  lowerCount: u32, // Total lower nodes (4 bytes)
  leafCount: u32,  // Total leaf nodes (4 bytes)
  dataCount: u32,  // Total data buffer size in 16-byte units (4 bytes)
}

struct PicoVDBGrid {
  gridIndex: u32,     // This grid's index (4 bytes)
  upperStart: u32,    // Index into uppers array (= root index) (4 bytes)
  lowerStart: u32,    // Index into lowers array (4 bytes)
  leafStart: u32,     // Index into leaves array (4 bytes)
  dataStart: u32,     // 16-byte index into data buffer (4 bytes)
  dataElemCount: u32, // Number of data elements for this grid (4 bytes)
  gridType: u32,      // GRID_TYPE_SDF_FLOAT=1, GRID_TYPE_SDF_UINT8=2 (4 bytes)
  _pad1: u32,
  indexBoundsMin: vec3i, // Index min (12 bytes)
  _pad2: u32,
  indexBoundsMax: vec3i, // Index min (12 bytes)
  _pad3: u32,
}

const GRID_TYPE_SDF_FLOAT = 1;
const GRID_TYPE_SDF_UINT8 = 2;

// https://webgpufundamentals.org/webgpu/lessons/resources/wgsl-offset-computer.html

// Root key for spatial lookup - maps coordinate to upper node index.
// Roots are 1:1 with uppers (root[i] -> upper[i]).
// Count derived from upperCount. Padded to 16-byte alignment.
struct PicoVDBRoot {
  key: vec2u,  // 64-bit coordinate key (8 bytes)
}

struct PicoVDBNodeMask {
  inside: u32,      // Bitmask of outside/inside (+/-) (4 bytes)
  value: u32,       // Bitmask of value, inside && value set this is a child (4 bytes)
  valueOffset: u32, // Prefix sum offset of value (4 bytes)
  childOffset: u32, // Prefix sum offset of child (4 bytes)
}

struct PicoVDBLeafMask{
  inside: u32,      // Bitmask of outside/inside (+/-) (4 bytes)
  value: u32,       // Bifmask of value, inside && value always is 0 (4 bytes)
  valueOffset: u32, // Prefix sum offset of value (4 bytes)
}

struct PicoVDBUpper {
  mask: array<PicoVDBNodeMask,1024>,
}

struct PicoVDBLower {
  mask: array<PicoVDBNodeMask,128>,
}

struct PicoVDBLeaf {
  mask: array<PicoVDBLeafMask,16>,
}

struct PicoVDBLevelCount {
    level: u32,  // Level of value found
    count: u32,  // Count offset (0 means no active values/background)
}

struct PicoVDBReadAccessor {
  key: vec3i,
  grid: u32,
  upper: u32,
  lower: u32,
  leaf: u32,
  _pad: u32,
}

const PICOVDB_INVALID_INDEX: u32 = 0xFFFFFFFFu;

fn picovdbReadAccessorInit(acc: ptr<function, PicoVDBReadAccessor>, grid: u32) {
    (*acc).key = vec3i(0x7FFFFFFF);
    (*acc).grid = grid;
    (*acc).upper = PICOVDB_INVALID_INDEX;
    (*acc).lower = PICOVDB_INVALID_INDEX;
    (*acc).leaf = PICOVDB_INVALID_INDEX;
    (*acc)._pad = 0u;
}

fn picovdbReadAccessorIsCachedLeaf(acc: ptr<function, PicoVDBReadAccessor>, dirty: i32) -> bool {
    let addr = (*acc).leaf;
    let is_cached = (addr != PICOVDB_INVALID_INDEX) && (dirty & ~0x7i) == 0; // Leaf is 8x8x8 (bits 0-2)
    (*acc).leaf = select(PICOVDB_INVALID_INDEX, addr, is_cached);
    return is_cached;
}

fn picovdbReadAccessorIsCachedLower(acc: ptr<function, PicoVDBReadAccessor>, dirty: i32) -> bool {
    let addr = (*acc).lower;
    let is_cached = (addr != PICOVDB_INVALID_INDEX) && (dirty & ~0x7Fi) == 0; // Lower is 128x128x128 (bits 0-6)
    (*acc).lower = select(PICOVDB_INVALID_INDEX, addr, is_cached);
    return is_cached;
}

fn picovdbReadAccessorIsCachedUpper(acc: ptr<function, PicoVDBReadAccessor>, dirty: i32) -> bool {
    let addr = (*acc).upper;
    let is_cached = (addr != PICOVDB_INVALID_INDEX) && (dirty & ~0xFFFi) == 0; // Upper is 4096x4096x4096 (bits 0-11)
    (*acc).upper = select(PICOVDB_INVALID_INDEX, addr, is_cached);
    return is_cached;
}

fn picovdbReadAccessorComputeDirty(acc: ptr<function, PicoVDBReadAccessor>, ijk: vec3i) -> i32 {
    return (ijk.x ^ (*acc).key.x) | (ijk.y ^ (*acc).key.y) | (ijk.z ^ (*acc).key.z);
}

fn picovdbCoordToKey(ijk: vec3i) -> vec2u {
    // Use the non-native 64-bit path since WGSL doesn't have native 64-bit
    let iu = u32(ijk.x) >> 12u;
    let ju = u32(ijk.y) >> 12u;
    let ku = u32(ijk.z) >> 12u;
    let key_x = ku | (ju << 21u);
    let key_y = (iu << 10u) | (ju >> 11u);
    return vec2u(key_x, key_y);
}

fn picovdbUpperCoordToOffset(ijk: vec3i) -> u32 {
    return (((u32(ijk.x) & 0xFFFu) >> 7u) << 10u) |
           (((u32(ijk.y) & 0xFFFu) >> 7u) << 5u)  |
            ((u32(ijk.z) & 0xFFFu) >> 7u);
}

fn picovdbLowerCoordToOffset(ijk: vec3i) -> u32 {
    return (((u32(ijk.x) & 0x7Fu) >> 3u) << 8u) |
           (((u32(ijk.y) & 0x7Fu) >> 3u) << 4u) |
            ((u32(ijk.z) & 0x7Fu) >> 3u);
}

fn picovdbLeafCoordToOffset(ijk: vec3i) -> u32 {
    return ((u32(ijk.x) & 0x7u) << 6u) |
           ((u32(ijk.y) & 0x7u) << 3u) |
            (u32(ijk.z) & 0x7u);
}


// Find root/upper index for coordinate within grid bounds.
// Roots are 1:1 with uppers, so the returned index works for both.
fn picovdbReadAccessorFindUpperIndex(
    ijk: vec3i,
    grid: PicoVDBGrid,
) -> i32 {
    let coordKey = picovdbCoordToKey(ijk);
    let startIndex = grid.upperStart;
    let endIndex = select(
      picovdb_grids[grid.gridIndex + 1u].upperStart, // false: use next grid's start
      arrayLength(&picovdb_roots),                   // true: use total roots count
      arrayLength(&picovdb_grids) - 1u == grid.gridIndex,
    );
    for (var i = startIndex; i < endIndex; i++) {
        let root = picovdb_roots[i];
        if (coordKey.x == root.key.x && coordKey.y == root.key.y) {
            return i32(i);
        }
    }
    return -1; // Not found
}

fn picovdbReadAccessorLeafGetLevelCountAndCache(
    acc: ptr<function, PicoVDBReadAccessor>,
    ijk: vec3i,
    grid: PicoVDBGrid,
) -> PicoVDBLevelCount {
    let n = picovdbLeafCoordToOffset(ijk);
    let word_index = n >> 5u; // Fast divide by 32
    let bit_index = n & 31u; // Fast modulo 32
    let mask = picovdb_leaves[grid.leafStart + (*acc).leaf].mask[word_index];

    let bit_at_pos = 1u << bit_index;
    let is_value = (mask.value & bit_at_pos) != 0u;
    let is_inside = (mask.inside & bit_at_pos) != 0u;
    let preceding_bits = extractBits(mask.value & ~mask.inside, 0u, bit_index);
    let count = select(
        u32(is_inside),
        mask.valueOffset + countOneBits(preceding_bits),
        is_value,
    );
    (*acc).key = ijk;
    return PicoVDBLevelCount(0u, count);
}

fn picovdbReadAccessorLowerGetLevelCountAndCache(
    acc: ptr<function, PicoVDBReadAccessor>,
    ijk: vec3i,
    grid: PicoVDBGrid,
) -> PicoVDBLevelCount {
    let n = picovdbLowerCoordToOffset(ijk);
    let word_index = n >> 5u; // Fast divide by 32
    let bit_index = n & 31u; // Fast modulo 32
    let mask = picovdb_lowers[grid.lowerStart + (*acc).lower].mask[word_index];

    let bit_at_pos = 1u << bit_index;
    let is_value = (mask.value & bit_at_pos) != 0u;
    let is_inside = (mask.inside & bit_at_pos) != 0u;

    if (is_value && is_inside) {
        let preceding_bits = extractBits(mask.value & mask.inside, 0u, bit_index);
        (*acc).leaf = mask.childOffset + countOneBits(preceding_bits);
        (*acc).key = ijk;
        return picovdbReadAccessorLeafGetLevelCountAndCache(acc, ijk, grid);
    }
    let preceding_bits = extractBits(mask.value & ~mask.inside, 0u, bit_index);
    let count = select(
        u32(is_inside),
        mask.valueOffset + countOneBits(preceding_bits),
        is_value,
    );
    return PicoVDBLevelCount(1u, count);
}

fn picovdbReadAccessorUpperGetLevelCountAndCache(
    acc: ptr<function, PicoVDBReadAccessor>,
    ijk: vec3i,
    grid: PicoVDBGrid,
) -> PicoVDBLevelCount {
    let n = picovdbUpperCoordToOffset(ijk);
    let word_index = n >> 5u; // Fast divide by 32
    let bit_index = n & 31u; // Fast modulo 32
    let mask = picovdb_uppers[grid.upperStart + (*acc).upper].mask[word_index];

    let bit_at_pos = 1u << bit_index;
    let is_value = (mask.value & bit_at_pos) != 0u;
    let is_inside = (mask.inside & bit_at_pos) != 0u;

    if (is_value && is_inside) {
        let preceding_bits = extractBits(mask.value & mask.inside, 0u, bit_index);
        (*acc).lower = mask.childOffset + countOneBits(preceding_bits);
        (*acc).key = ijk;
        return picovdbReadAccessorLowerGetLevelCountAndCache(acc, ijk, grid);
    }
    let preceding_bits = extractBits(mask.value & ~mask.inside, 0u, bit_index);
    let count = select(
        u32(is_inside),
        mask.valueOffset + countOneBits(preceding_bits),
        is_value,
    );
    return PicoVDBLevelCount(2u, count);
}

// Get level and count from root and update cache
fn picovdbReadAccessorRootGetLevelCountAndCache(
    acc: ptr<function, PicoVDBReadAccessor>,
    ijk: vec3i,
    grid: PicoVDBGrid,
) -> PicoVDBLevelCount {
    let rootIndex = picovdbReadAccessorFindUpperIndex(ijk, grid);
    if (rootIndex == -1) {
        // No matching root tile, return background
        return PicoVDBLevelCount(3u, 0u);
    }
    (*acc).upper = u32(rootIndex);
    (*acc).key = ijk;
    return picovdbReadAccessorUpperGetLevelCountAndCache(acc, ijk, grid);
}

fn picovdbReadAccessorGetLevelCount(
    acc: ptr<function, PicoVDBReadAccessor>,
    ijk: vec3i,
    grid: PicoVDBGrid,
) -> PicoVDBLevelCount {
    let dirty = picovdbReadAccessorComputeDirty(acc, ijk);
    
    if (picovdbReadAccessorIsCachedLeaf(acc, dirty)) {
        return picovdbReadAccessorLeafGetLevelCountAndCache(acc, ijk, grid);
    } else if (picovdbReadAccessorIsCachedLower(acc, dirty)) {
        return picovdbReadAccessorLowerGetLevelCountAndCache(acc, ijk, grid);
    } else if (picovdbReadAccessorIsCachedUpper(acc, dirty)) {
        return picovdbReadAccessorUpperGetLevelCountAndCache(acc, ijk, grid);
    } else {
        return picovdbReadAccessorRootGetLevelCountAndCache(acc, ijk, grid);
    }
}

// --- HDDA (Hierarchical Digital Differential Analyzer) ---
const PICOVDB_HDDA_FLOAT_MAX: f32 = 1e38;

struct PicoVDBHDDA {
    voxel: vec3i,
    dim: i32,
    step: vec3i,
    tmin: f32,
    delta: vec3f,
    tmax: f32,
    next: vec3f,
}

fn picovdbHDDAPosToVoxel(pos: vec3f, dim: i32) -> vec3i {
    let mask = ~(dim - 1);
    return vec3i(floor(pos)) & vec3i(mask);
}

fn picovdbHDDAInit(
    hdda: ptr<function, PicoVDBHDDA>,
    origin: vec3f,
    tmin: f32,
    direction: vec3f,
    tmax: f32,
    direction_inv: vec3f,
    dim: i32
) {
    let pos = origin + direction * tmin;
    let vox = picovdbHDDAPosToVoxel(pos, dim);

    (*hdda).dim = dim;
    (*hdda).tmin = tmin;
    (*hdda).tmax = tmax;
    (*hdda).voxel = vox;
    (*hdda).step = vec3i(sign(direction));
    (*hdda).delta = abs(f32(dim) * direction_inv); // Pre-multiply delta by dim

    let boundary = select(vec3f(vox), vec3f(vox + vec3i(dim)), direction > vec3f(0.0));

    // Safety: handle cases where direction is 0 to avoid NaNs
    (*hdda).next = select(
        tmin + (boundary - pos) * direction_inv,
        vec3f(PICOVDB_HDDA_FLOAT_MAX),
        direction == vec3f(0.0)
    );
}

// Update HDDA to switch hierarchical level
fn picovdbHDDAUpdate(
    hdda: ptr<function, PicoVDBHDDA>,
    origin: vec3f,
    dim: i32,
    direction: vec3f,
    direction_inv: vec3f,
) {
    (*hdda).dim = dim;
    (*hdda).delta = abs(f32(dim) * direction_inv);

    // Re-calculate position at the exact current tmin plus a safety nudge
    let eps = max(1e-4f, (*hdda).tmin * 1e-7f);
    let pos = origin + direction * ((*hdda).tmin + eps);

    // Crucial: Re-mask the voxel to the new dimension
    (*hdda).voxel = picovdbHDDAPosToVoxel(pos, dim);

    // Re-calculate next boundary for the new grid scale
    let boundary = select(vec3f((*hdda).voxel), vec3f((*hdda).voxel + vec3i(dim)), direction > vec3f(0.0));
    let t_to_boundary = (boundary - (origin + direction * (*hdda).tmin)) * direction_inv;

    // Ensure the next step is always forward
    (*hdda).next = (*hdda).tmin + max(t_to_boundary, vec3f(eps));
}

fn picovdbHDDAStep(hdda: ptr<function, PicoVDBHDDA>) -> bool {
    // Determine which axis has the nearest boundary
    let next = (*hdda).next;
    if (next.x < next.y && next.x < next.z) { // X is smallest
        (*hdda).tmin = (*hdda).next.x;
        (*hdda).next.x += (*hdda).delta.x;
        (*hdda).voxel.x += (*hdda).dim * (*hdda).step.x;
    } else if (next.y < next.z) { // Y is smallest
        (*hdda).tmin = (*hdda).next.y;
        (*hdda).next.y += (*hdda).delta.y;
        (*hdda).voxel.y += (*hdda).dim * (*hdda).step.y;
    } else { // Z is smallest
        (*hdda).tmin = (*hdda).next.z;
        (*hdda).next.z += (*hdda).delta.z;
        (*hdda).voxel.z += (*hdda).dim * (*hdda).step.z;
    }
    return (*hdda).tmin <= (*hdda).tmax;
}

// Clip ray to bounding box
fn picovdbHDDARayClip(
    bbox_min: vec3f,
    bbox_max: vec3f,
    origin: vec3f,
    tmin: ptr<function, f32>,
    dir_inv: vec3f,
    tmax: ptr<function, f32>
) -> bool {
    let t0 = (bbox_min - origin) * dir_inv;
    let t1 = (bbox_max - origin) * dir_inv;
    let tmin3 = min(t0, t1);
    let tmax3 = max(t0, t1);
    let tnear = max(tmin3.x, max(tmin3.y, tmin3.z));
    let tfar = min(tmax3.x, min(tmax3.y, tmax3.z));
    let hit = tnear <= tfar;
    *tmin = max(*tmin, tnear);
    *tmax = min(*tmax, tfar);
    return hit;
}

// Dimension based on level (for HDDA stepping)
// Level 0 (Leaf) -> 1
// Level 1 (Lower) -> 8 (2^3)
// Level 2 (Upper) -> 128 (2^7)
// Level 3 (Root) -> 4096 (2^12)
const picovdbDimForLevel = array(1, 8, 128, 4096);

// Check if voxel is active (count > 1 means has value)
fn picovdbIsActive(level_count: PicoVDBLevelCount) -> bool {
    return level_count.count > 1u;
}
// Get float value from data buffer using grid offset and value index
fn picovdbGetValue(grid: PicoVDBGrid, count: u32) -> f32 {
    // dataStart is in 16-byte units, multiply by 4 to get u32 index (16 bytes = 4 u32s)
    let u32Index = grid.dataStart * 4u + count;
    return bitcast<f32>(picovdb_buffer[u32Index]);
}

// Zero-crossing detection for level set raymarching
fn picovdbHDDAZeroCrossing(
    acc: ptr<function, PicoVDBReadAccessor>,
    grid: PicoVDBGrid,
    origin: vec3f,
    tmin: f32,
    direction: vec3f,
    tmax: f32,
    thit: ptr<function, f32>,
    v: ptr<function, f32>
) -> bool {
    let direction_inv = 1 / direction;
    var tmin_mut = tmin;
    var tmax_mut = tmax;
    if (!picovdbHDDARayClip(vec3f(grid.indexBoundsMin), vec3f(grid.indexBoundsMax + vec3i(1)), origin, &tmin_mut, direction_inv, &tmax_mut)) {
        return false;
    }

    // Get initial hierarchy level
    let start_pos = origin + direction * tmin_mut;
    let res0 = picovdbReadAccessorGetLevelCount(acc, vec3i(floor(start_pos)), grid);
    let v0 = picovdbGetValue(grid, res0.count);
    
    var hdda: PicoVDBHDDA;
    picovdbHDDAInit(&hdda, origin, tmin_mut, direction, tmax_mut, direction_inv, picovdbDimForLevel[res0.level]);

    for (var i = 0; i < 512; i++) { // Fixed loop limit for GPU safety
        let result = picovdbReadAccessorGetLevelCount(acc, hdda.voxel, grid);
        let target_dim = picovdbDimForLevel[result.level];

        // If hierarchy changed, update HDDA and re-read
        if (hdda.dim != target_dim) {
            picovdbHDDAUpdate(&hdda, origin, target_dim, direction, direction_inv);
            continue; // Re-evaluate with the new aligned voxel
        }

        // Leaf level logic (Surface detection)
        if (hdda.dim == 1 && picovdbIsActive(result)) {
            let val = picovdbGetValue(grid, result.count);
            if (val * v0 < 0.0) {
                *thit = hdda.tmin;
                *v = val;
                return true;
            }
            // Optional: v0 = val; // Update previous value for continuous crossing
        }

        // Step to next boundary
        if (!picovdbHDDAStep(&hdda)) { break; }
    }
    return false;
}

struct PicoVDBStencil {
    v000: f32, v001: f32, v010: f32, v011: f32,
    v100: f32, v101: f32, v110: f32, v111: f32,
}

// Sample 2x2x2 stencil of voxel values around a point
fn picovdbSampleStencil(
    acc: ptr<function, PicoVDBReadAccessor>,
    grid: PicoVDBGrid,
    ijk: vec3i
) -> PicoVDBStencil {
    var s: PicoVDBStencil;
    s.v000 = picovdbGetValue(grid, picovdbReadAccessorGetLevelCount(acc, ijk + vec3i(0, 0, 0), grid).count);
    s.v001 = picovdbGetValue(grid, picovdbReadAccessorGetLevelCount(acc, ijk + vec3i(0, 0, 1), grid).count);
    s.v010 = picovdbGetValue(grid, picovdbReadAccessorGetLevelCount(acc, ijk + vec3i(0, 1, 0), grid).count);
    s.v011 = picovdbGetValue(grid, picovdbReadAccessorGetLevelCount(acc, ijk + vec3i(0, 1, 1), grid).count);
    s.v100 = picovdbGetValue(grid, picovdbReadAccessorGetLevelCount(acc, ijk + vec3i(1, 0, 0), grid).count);
    s.v101 = picovdbGetValue(grid, picovdbReadAccessorGetLevelCount(acc, ijk + vec3i(1, 0, 1), grid).count);
    s.v110 = picovdbGetValue(grid, picovdbReadAccessorGetLevelCount(acc, ijk + vec3i(1, 1, 0), grid).count);
    s.v111 = picovdbGetValue(grid, picovdbReadAccessorGetLevelCount(acc, ijk + vec3i(1, 1, 1), grid).count);
    return s;
}

// Compute trilinear gradient from 2x2x2 stencil
fn picovdbTrilinearGradient(uvw: vec3f, s: PicoVDBStencil) -> vec3f {
    // Interpolate values along Z for the four XY columns
    let v00z = mix(s.v000, s.v001, uvw.z);
    let v01z = mix(s.v010, s.v011, uvw.z);
    let v10z = mix(s.v100, s.v101, uvw.z);
    let v11z = mix(s.v110, s.v111, uvw.z);

    // Interpolate values along Y for the two X slabs
    let v0yz = mix(v00z, v01z, uvw.y);
    let v1yz = mix(v10z, v11z, uvw.y);

    // X Gradient: Difference between the two YZ-interpolated slabs
    let grad_x = v1yz - v0yz;

    // Y Gradient: Interpolate the differences along X
    let grad_y = mix(v01z - v00z, v11z - v10z, uvw.x);

    // Z Gradient: Interpolate the differences along X and Y
    let dZ00 = s.v001 - s.v000;
    let dZ01 = s.v011 - s.v010;
    let dZ10 = s.v101 - s.v100;
    let dZ11 = s.v111 - s.v110;
    let grad_z = mix(mix(dZ00, dZ01, uvw.y), mix(dZ10, dZ11, uvw.y), uvw.x);

    return vec3f(grad_x, grad_y, grad_z);
}

// Trilinear interpolation of a value at position uvw within a voxel stencil
fn picovdbTrilinearInterpolation(uvw: vec3f, s: PicoVDBStencil) -> f32 {
    // Interpolate along Z
    let v00 = mix(s.v000, s.v001, uvw.z);
    let v01 = mix(s.v010, s.v011, uvw.z);
    let v10 = mix(s.v100, s.v101, uvw.z);
    let v11 = mix(s.v110, s.v111, uvw.z);
    
    // Interpolate along Y
    let v0 = mix(v00, v01, uvw.y);
    let v1 = mix(v10, v11, uvw.y);
    
    // Interpolate along X
    return mix(v0, v1, uvw.x);
}

fn picovdbHDDAIsOccluded(
    acc: ptr<function, PicoVDBReadAccessor>,
    grid: PicoVDBGrid,
    origin: vec3f,
    tmin: f32,
    direction: vec3f,
    tmax: f32
) -> bool {
    let direction_inv = 1 / direction;
    var tmin_mut = tmin;
    var tmax_mut = tmax;
    if (!picovdbHDDARayClip(vec3f(grid.indexBoundsMin), vec3f(grid.indexBoundsMax + vec3i(1)), origin, &tmin_mut, direction_inv, &tmax_mut)) {
        return false;
    }

    var hdda: PicoVDBHDDA;
    // Shadow rays can start at a coarser level (e.g., dim 8) to skip empty space faster
    picovdbHDDAInit(&hdda, origin, tmin_mut, direction, tmax_mut, direction_inv, 8);

    for (var i = 0; i < 256; i++) { // Shadow rays usually need fewer steps
        let result = picovdbReadAccessorGetLevelCount(acc, hdda.voxel, grid);
        let target_dim = picovdbDimForLevel[result.level];

        if (hdda.dim != target_dim) {
            picovdbHDDAUpdate(&hdda, origin, target_dim, direction, direction_inv);
            continue;
        }

        // For SDFs, active leaf usually means "near or inside the surface"
        if (hdda.dim == 1 && picovdbIsActive(result)) {
            let val = picovdbGetValue(grid, result.count);
            if (val <= 0.0) { return true; } // Inside or on the surface
        }

        if (!picovdbHDDAStep(&hdda)) { break; }
    }
    return false;
}

fn picovdbSampleTrilinear(
    acc: ptr<function, PicoVDBReadAccessor>,
    grid: PicoVDBGrid,
    pos: vec3f
) -> f32 {
    let ijk = vec3i(floor(pos));
    let uvw = fract(pos);
    let s = picovdbSampleStencil(acc, grid, ijk);
    return picovdbTrilinearInterpolation(uvw, s);
}

//fn picovdbHDDASmoothTraversal(
//    acc: ptr<function, PicoVDBReadAccessor>,
//    grid: PicoVDBGrid,
//    origin: vec3f,
//    tmin: f32,
//    direction: vec3f,
//    tmax: f32,
//    thit: ptr<function, f32>
//) -> bool {
//    let direction_inv = 1 / direction;
//    var tmin_mut = tmin;
//    var tmax_mut = tmax;
//    if (!picovdbHDDARayClip(vec3f(grid.indexBoundsMin), vec3f(grid.indexBoundsMax + vec3i(1)), origin, &tmin_mut, direction_inv, &tmax_mut)) {
//        return false;
//    }
//
//    var hdda: PicoVDBHDDA;
//    picovdbHDDAInit(&hdda, origin, tmin_mut, direction, tmax, direction_inv, 4096);
//
//    // Cache the trilinear value at the entry point of the current voxel
//    var v_entry: f32 = picovdbSampleTrilinear(acc, grid, origin + direction * hdda.tmin);
//
//    for (var i = 0; i < 256; i++) {
//        let result = picovdbReadAccessorGetLevelCount(acc, hdda.voxel, grid);
//        let target_dim = picovdbDimForLevel[result.level];
//
//        if (hdda.dim != target_dim) {
//            picovdbHDDAUpdate(&hdda, origin, target_dim, direction, direction_inv);
//            // Re-sync v_entry after a level jump
//            v_entry = picovdbSampleTrilinear(acc, grid, origin + direction * hdda.tmin);
//            continue;
//        }
//
//        if (hdda.dim == 1) {
//            let discrete_val = picovdbGetValue(grid, result.count);
//            
//            // OPTIMIZATION 1: Discrete Guard
//            // If the discrete value is far from the surface, don't sample the stencil.
//            // 1.732 (sqrt(3)) is the max distance from a voxel center to its corner.
//            if (discrete_val > 1.2) { 
//                if (!picovdbHDDAStep(&hdda)) { break; }
//                // Use discrete value as a cheap approximation for the next v_entry
//                v_entry = discrete_val; 
//                continue; 
//            }
//
//            let t_exit = min(min(hdda.next.x, hdda.next.y), hdda.next.z);
//            let v_exit = picovdbSampleTrilinear(acc, grid, origin + direction * t_exit);
//
//            // OPTIMIZATION 2: Early Exit for solid interior
//            if (v_entry <= 0.0) {
//                *thit = hdda.tmin;
//                return true;
//            }
//
//            // Check for zero-crossing
//            if (v_entry * v_exit <= 0.0) {
//                // Iterations reduced to 3 for performance; 
//                // Secant search is better than Binary for SDFs
//                *thit = picovdbRefineIntersection(acc, grid, origin, direction, hdda.tmin, t_exit, v_entry, v_exit);
//                return true;
//            }
//            
//            // OPTIMIZATION 3: Value Caching
//            // The exit of this voxel IS the entry of the next.
//            v_entry = v_exit;
//        }
//
//        if (!picovdbHDDAStep(&hdda)) { break; }
//    }
//    return false;
//}
//
// Optimize refinement using Secant Method
//fn picovdbRefineIntersection(
//    acc: ptr<function, PicoVDBReadAccessor>,
//    grid: PicoVDBGrid,
//    origin: vec3f,
//    direction: vec3f,
//    t_start: f32,
//    t_end: f32,
//    v_start: f32,
//    v_end: f32
//) -> f32 {
//    var ts = t_start;
//    var te = t_end;
//    var vs = v_start;
//    var ve = v_end;
//
//    for (var i = 0; i < 2; i++) {
//        // Linear interpolation to find where the line crosses zero
//        let t_guess = ts + (te - ts) * (vs / (vs - ve));
//        let v_guess = picovdbSampleTrilinear(acc, grid, origin + direction * t_guess);
//
//        if (v_guess > 0.0) {
//            ts = t_guess;
//            vs = v_guess;
//        } else {
//            te = t_guess;
//            ve = v_guess;
//        }
//    }
//    return te;
//}
`;

// ../picovdb.ts
var PICOVDB_MAGIC = [1868786e3, 809649238];
var PICOVDB_FILE_HEADER_SIZE = 32;
var PICOVDB_GRID_SIZE = 64;
var PICOVDB_ROOT_SIZE = 8;
var PICOVDB_NODE_MASK_SIZE = 16;
var PICOVDB_LEAF_MASK_SIZE = 12;
var PICOVDB_UPPER_SIZE = 16384;
var PICOVDB_LOWER_SIZE = 2048;
var PICOVDB_LEAF_SIZE = 192;
var PICOVDB_DATA_SIZE = 16;
var PicoVDBFile = class {
  buffer;
  view;
  // Header
  header;
  // Slices (as Uint8Arrays for WebGPU)
  gridsBuffer;
  rootsBuffer;
  uppersBuffer;
  lowersBuffer;
  leavesBuffer;
  dataBuffer;
  constructor(buffer) {
    this.buffer = buffer;
    this.view = new DataView(buffer);
    let offset = 0;
    this.header = {
      magic: [this.view.getUint32(offset + 0, true), this.view.getUint32(offset + 4, true)],
      version: this.view.getUint32(offset + 8, true),
      gridCount: this.view.getUint32(offset + 12, true),
      upperCount: this.view.getUint32(offset + 16, true),
      lowerCount: this.view.getUint32(offset + 20, true),
      leafCount: this.view.getUint32(offset + 24, true),
      dataCount: this.view.getUint32(offset + 28, true)
    };
    offset += PICOVDB_FILE_HEADER_SIZE;
    if (this.header.magic[0] !== PICOVDB_MAGIC[0] || this.header.magic[1] !== PICOVDB_MAGIC[1]) {
      throw new Error(`Invalid PicoVDB magic: [0x${this.header.magic[0].toString(16)}, 0x${this.header.magic[1].toString(16)}]`);
    }
    this.gridsBuffer = new Uint8Array(buffer, offset, this.header.gridCount * PICOVDB_GRID_SIZE);
    offset += this.header.gridCount * PICOVDB_GRID_SIZE;
    console.log("GRIDS BUFFER", this.gridsBuffer.length);
    const rootCount = this.getRootCountPadded();
    this.rootsBuffer = new Uint8Array(buffer, offset, rootCount * PICOVDB_ROOT_SIZE);
    offset += rootCount * PICOVDB_ROOT_SIZE;
    console.log("ROOTS BUFFER", this.rootsBuffer.length);
    this.uppersBuffer = new Uint8Array(buffer, offset, this.header.upperCount * PICOVDB_UPPER_SIZE);
    offset += this.header.upperCount * PICOVDB_UPPER_SIZE;
    console.log("UPPERS BUFFER", this.uppersBuffer.length);
    this.lowersBuffer = new Uint8Array(buffer, offset, this.header.lowerCount * PICOVDB_LOWER_SIZE);
    offset += this.header.lowerCount * PICOVDB_LOWER_SIZE;
    console.log("LOWERS BUFFER", this.lowersBuffer.length);
    this.leavesBuffer = new Uint8Array(buffer, offset, this.header.leafCount * PICOVDB_LEAF_SIZE);
    offset += this.header.leafCount * PICOVDB_LEAF_SIZE;
    console.log("LEAVES BUFFER", this.leavesBuffer.length);
    this.dataBuffer = new Uint8Array(buffer, offset, this.header.dataCount * PICOVDB_DATA_SIZE);
    offset += this.header.dataCount * PICOVDB_DATA_SIZE;
    console.log("DATA BUFFER", this.dataBuffer.length);
  }
  getSize() {
    return this.buffer.byteLength;
  }
  getGrid(index) {
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
      indexBoundsMax: new Int32Array(this.buffer, offset + 48, 3)
    };
  }
  getRootCountPadded() {
    return ((this.header.upperCount + 1) / 2 | 0) * 2;
  }
  getRoot(index) {
    if (index >= this.header.upperCount) {
      throw new Error(`Root index ${index} out of bounds (max: ${this.header.upperCount - 1})`);
    }
    const baseOffset = PICOVDB_FILE_HEADER_SIZE + this.header.gridCount * PICOVDB_GRID_SIZE + index * PICOVDB_ROOT_SIZE;
    return {
      key: [
        this.view.getUint32(baseOffset + 0, true),
        this.view.getUint32(baseOffset + 4, true)
      ]
    };
  }
  getUpper(index) {
    if (index >= this.header.upperCount) {
      throw new Error(`Upper index ${index} out of bounds (max: ${this.header.upperCount - 1})`);
    }
    const baseOffset = PICOVDB_FILE_HEADER_SIZE + this.header.gridCount * PICOVDB_GRID_SIZE + this.getRootCountPadded() * PICOVDB_ROOT_SIZE + index * PICOVDB_UPPER_SIZE;
    const masks = [];
    for (let i = 0; i < 1024; i++) {
      const maskOffset = baseOffset + i * PICOVDB_NODE_MASK_SIZE;
      masks.push({
        inside: this.view.getUint32(maskOffset + 0, true),
        value: this.view.getUint32(maskOffset + 4, true),
        valueOffset: this.view.getUint32(maskOffset + 8, true),
        childOffset: this.view.getUint32(maskOffset + 12, true)
      });
    }
    return {
      mask: masks
    };
  }
  getLower(index) {
    if (index >= this.header.lowerCount) {
      throw new Error(`Lower index ${index} out of bounds (max: ${this.header.lowerCount - 1})`);
    }
    const baseOffset = PICOVDB_FILE_HEADER_SIZE + this.header.gridCount * PICOVDB_GRID_SIZE + this.getRootCountPadded() * PICOVDB_ROOT_SIZE + this.header.upperCount * PICOVDB_UPPER_SIZE + index * PICOVDB_LOWER_SIZE;
    const masks = [];
    for (let i = 0; i < 128; i++) {
      const maskOffset = baseOffset + i * PICOVDB_NODE_MASK_SIZE;
      masks.push({
        inside: this.view.getUint32(maskOffset + 0, true),
        value: this.view.getUint32(maskOffset + 4, true),
        valueOffset: this.view.getUint32(maskOffset + 8, true),
        childOffset: this.view.getUint32(maskOffset + 12, true)
      });
    }
    return {
      mask: masks
    };
  }
  getLeaf(index) {
    if (index >= this.header.leafCount) {
      throw new Error(`Leaf index ${index} out of bounds (max: ${this.header.leafCount - 1})`);
    }
    const baseOffset = PICOVDB_FILE_HEADER_SIZE + this.header.gridCount * PICOVDB_GRID_SIZE + this.getRootCountPadded() * PICOVDB_ROOT_SIZE + this.header.upperCount * PICOVDB_UPPER_SIZE + this.header.lowerCount * PICOVDB_LOWER_SIZE + index * PICOVDB_LEAF_SIZE;
    const masks = [];
    for (let i = 0; i < 16; i++) {
      const maskOffset = baseOffset + i * PICOVDB_LEAF_MASK_SIZE;
      masks.push({
        inside: this.view.getUint32(maskOffset + 0, true),
        value: this.view.getUint32(maskOffset + 4, true),
        valueOffset: this.view.getUint32(maskOffset + 8, true)
      });
    }
    return {
      mask: masks
    };
  }
  getVoxelCount() {
    var count = 0;
    for (let i = 0; i < this.header.gridCount; i++) {
      count += this.getGrid(i).dataElemCount - 2;
    }
    return count;
  }
  // TODO: this needs to use the dataStart to first slice the dataBuffer in 16 byte chunks
  // then capture the value with the dataElemCount.
  //getGridFloat(grid: PicoVDBGrid, index: number): number {
  //  const dataPtr = new Float32Array(this.dataBuffer.buffer, this.dataBuffer.byteOffset);
  //  return dataPtr[grid.dataIndex / 4 + index]; // dataIndex is in bytes, convert to float index
  //}
};

// lib/loader.ts
async function loadPicoVDB(url) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to load PicoVDB file ${url}: ${response.statusText}`);
  }
  let data;
  if (url.endsWith(".gz")) {
    console.log("Decompressing gzipped PicoVDB file...");
    const compressedData = await response.arrayBuffer();
    console.log(`Loaded compressed PicoVDB file: ${compressedData.byteLength} bytes`);
    if (typeof DecompressionStream === "undefined") {
      throw new Error("Gzip decompression not supported in this browser. Please use a modern browser with Compression Streams API support.");
    }
    const stream = new Response(compressedData).body.pipeThrough(new DecompressionStream("gzip"));
    data = await new Response(stream).arrayBuffer();
    console.log(`Decompressed PicoVDB file: ${data.byteLength} bytes`);
  } else {
    data = await response.arrayBuffer();
    console.log(`Loaded raw PicoVDB file: ${data.byteLength} bytes`);
  }
  const paddedSize = Math.ceil(data.byteLength / 4) * 4;
  let alignedData;
  if (paddedSize === data.byteLength) {
    alignedData = data;
  } else {
    alignedData = new ArrayBuffer(paddedSize);
    const paddedView = new Uint8Array(alignedData);
    paddedView.set(new Uint8Array(data));
    console.log(`PicoVDB file padded: ${data.byteLength} \u2192 ${paddedSize} bytes`);
  }
  const picoFile = new PicoVDBFile(alignedData);
  console.log("PicoVDB File loaded successfully:");
  console.log("PicoVDB File Header:");
  console.log(`  Magic: [0x${picoFile.header.magic[0].toString(16)}, 0x${picoFile.header.magic[1].toString(16)}]`);
  console.log(`  Version: ${picoFile.header.version}`);
  console.log(`  Grid Count: ${picoFile.header.gridCount}`);
  console.log(`  Upper Count: ${picoFile.header.upperCount}`);
  console.log(`  Lower Count: ${picoFile.header.lowerCount}`);
  console.log(`  Leaf Count: ${picoFile.header.leafCount}`);
  console.log(`  Data Count: ${picoFile.header.dataCount} bytes`);
  console.log(`  Voxel Count: ${picoFile.getVoxelCount()}`);
  if (picoFile.header.gridCount === 0) {
    throw new Error("PicoVDB file contains no grids");
  }
  return picoFile;
}

// lib/camera.ts
var CameraBase = class {
  // The camera matrix
  matrix_ = new Float32Array([
    1,
    0,
    0,
    0,
    0,
    1,
    0,
    0,
    0,
    0,
    1,
    0,
    0,
    0,
    0,
    1
  ]);
  // The calculated view matrix
  view_ = mat4.create();
  // Aliases to column vectors of the matrix
  right_ = new Float32Array(this.matrix_.buffer, 4 * 0, 4);
  up_ = new Float32Array(this.matrix_.buffer, 4 * 4, 4);
  back_ = new Float32Array(this.matrix_.buffer, 4 * 8, 4);
  position_ = new Float32Array(this.matrix_.buffer, 4 * 12, 4);
  // Returns the camera matrix
  get matrix() {
    return this.matrix_;
  }
  // Assigns `mat` to the camera matrix
  set matrix(mat) {
    mat4.copy(mat, this.matrix_);
  }
  // Returns the camera view matrix
  get view() {
    return this.view_;
  }
  // Assigns `mat` to the camera view
  set view(mat) {
    mat4.copy(mat, this.view_);
  }
  // Returns column vector 0 of the camera matrix
  get right() {
    return this.right_;
  }
  // Assigns `vec` to the first 3 elements of column vector 0 of the camera matrix
  set right(vec) {
    vec3.copy(vec, this.right_);
  }
  // Returns column vector 1 of the camera matrix
  get up() {
    return this.up_;
  }
  // Assigns `vec` to the first 3 elements of column vector 1 of the camera matrix
  set up(vec) {
    vec3.copy(vec, this.up_);
  }
  // Returns column vector 2 of the camera matrix
  get back() {
    return this.back_;
  }
  // Assigns `vec` to the first 3 elements of column vector 2 of the camera matrix
  set back(vec) {
    vec3.copy(vec, this.back_);
  }
  // Returns column vector 3 of the camera matrix
  get position() {
    return this.position_;
  }
  // Assigns `vec` to the first 3 elements of column vector 3 of the camera matrix
  set position(vec) {
    vec3.copy(vec, this.position_);
  }
};
var WASDCamera = class extends CameraBase {
  // The camera absolute pitch angle
  pitch = 0;
  // The camera absolute yaw angle
  yaw = 0;
  // The movement veloicty
  velocity_ = vec3.create();
  // Speed multiplier for camera movement
  movementSpeed = 10;
  // Speed multiplier for camera rotation
  rotationSpeed = 1;
  // Movement velocity drag coeffient [0 .. 1]
  // 0: Continues forever
  // 1: Instantly stops moving
  frictionCoefficient = 0.99;
  // Returns velocity vector
  get velocity() {
    return this.velocity_;
  }
  // Assigns `vec` to the velocity vector
  set velocity(vec) {
    vec3.copy(vec, this.velocity_);
  }
  // Construtor
  constructor(options) {
    super();
    if (options && (options.position || options.target)) {
      const position = options.position ?? vec3.create(0, 0, -5);
      const target = options.target ?? vec3.create(0, 0, 0);
      const back = vec3.normalize(vec3.sub(position, target));
      this.recalculateAngles(back);
      this.position = position;
    }
  }
  // Returns the camera matrix
  get matrix() {
    return super.matrix;
  }
  // Assigns `mat` to the camera matrix, and recalcuates the camera angles
  set matrix(mat) {
    super.matrix = mat;
    this.recalculateAngles(this.back);
  }
  update(deltaTime, input) {
    const sign = (positive, negative) => (positive ? 1 : 0) - (negative ? 1 : 0);
    this.yaw -= input.analog.x * deltaTime * this.rotationSpeed;
    this.pitch -= input.analog.y * deltaTime * this.rotationSpeed;
    this.yaw = mod(this.yaw, Math.PI * 2);
    this.pitch = clamp(this.pitch, -Math.PI / 2, Math.PI / 2);
    const position = vec3.copy(this.position);
    super.matrix = mat4.rotateX(mat4.rotationY(this.yaw), this.pitch);
    const digital = input.digital;
    const deltaRight = sign(digital.right, digital.left);
    const deltaUp = sign(digital.up, digital.down);
    const targetVelocity = vec3.create();
    const deltaBack = sign(digital.backward, digital.forward);
    vec3.addScaled(targetVelocity, this.right, deltaRight, targetVelocity);
    vec3.addScaled(targetVelocity, this.up, deltaUp, targetVelocity);
    vec3.addScaled(targetVelocity, this.back, deltaBack, targetVelocity);
    vec3.normalize(targetVelocity, targetVelocity);
    vec3.mulScalar(targetVelocity, this.movementSpeed, targetVelocity);
    this.velocity = lerp(
      targetVelocity,
      this.velocity,
      Math.pow(1 - this.frictionCoefficient, deltaTime)
    );
    this.position = vec3.addScaled(position, this.velocity, deltaTime);
    this.view = mat4.invert(this.matrix);
    return this.view;
  }
  // Recalculates the yaw and pitch values from a directional vector
  recalculateAngles(dir) {
    this.yaw = Math.atan2(dir[0], dir[2]);
    this.pitch = -Math.asin(dir[1]);
  }
};
var ArcballCamera = class extends CameraBase {
  // The camera distance from the target
  distance = 0;
  // The current angular velocity
  angularVelocity = 0;
  // The current rotation axis
  axis_ = vec3.create();
  // Returns the rotation axis
  get axis() {
    return this.axis_;
  }
  // Assigns `vec` to the rotation axis
  set axis(vec) {
    vec3.copy(vec, this.axis_);
  }
  // Speed multiplier for camera rotation
  rotationSpeed = 1;
  // Speed multiplier for camera zoom
  zoomSpeed = 0.1;
  // Rotation velocity drag coeffient [0 .. 1]
  // 0: Spins forever
  // 1: Instantly stops spinning
  frictionCoefficient = 0.999;
  // Construtor
  constructor(options) {
    super();
    if (options && options.position) {
      this.position = options.position;
      this.distance = vec3.len(this.position);
      this.back = vec3.normalize(this.position);
      this.recalcuateRight();
      this.recalcuateUp();
    }
  }
  // Returns the camera matrix
  get matrix() {
    return super.matrix;
  }
  // Assigns `mat` to the camera matrix, and recalcuates the distance
  set matrix(mat) {
    super.matrix = mat;
    this.distance = vec3.len(this.position);
  }
  update(deltaTime, input) {
    const epsilon = 1e-7;
    if (input.analog.touching) {
      this.angularVelocity = 0;
    } else {
      this.angularVelocity *= Math.pow(1 - this.frictionCoefficient, deltaTime);
    }
    const movement = vec3.create();
    vec3.addScaled(movement, this.right, input.analog.x, movement);
    vec3.addScaled(movement, this.up, -input.analog.y, movement);
    const crossProduct = vec3.cross(movement, this.back);
    const magnitude = vec3.len(crossProduct);
    if (magnitude > epsilon) {
      this.axis = vec3.scale(crossProduct, 1 / magnitude);
      this.angularVelocity = magnitude * this.rotationSpeed;
    }
    const rotationAngle = this.angularVelocity * deltaTime;
    if (rotationAngle > epsilon) {
      this.back = vec3.normalize(rotate(this.back, this.axis, rotationAngle));
      this.recalcuateRight();
      this.recalcuateUp();
    }
    if (input.analog.zoom !== 0) {
      this.distance *= 1 + input.analog.zoom * this.zoomSpeed;
    }
    this.position = vec3.scale(this.back, this.distance);
    this.view = mat4.invert(this.matrix);
    return this.view;
  }
  // Assigns `this.right` with the cross product of `this.up` and `this.back`
  recalcuateRight() {
    this.right = vec3.normalize(vec3.cross(this.up, this.back));
  }
  // Assigns `this.up` with the cross product of `this.back` and `this.right`
  recalcuateUp() {
    this.up = vec3.normalize(vec3.cross(this.back, this.right));
  }
};
function clamp(x, min, max) {
  return Math.min(Math.max(x, min), max);
}
function mod(x, div) {
  return x - Math.floor(Math.abs(x) / div) * div * Math.sign(x);
}
function rotate(vec, axis, angle) {
  return vec3.transformMat4Upper3x3(vec, mat4.rotation(axis, angle));
}
function lerp(a, b, s) {
  return vec3.addScaled(a, vec3.sub(b, a), s);
}

// lib/input.ts
function createInputHandler(window2, canvas2) {
  const digital = {
    forward: false,
    backward: false,
    left: false,
    right: false,
    up: false,
    down: false
  };
  const analog = {
    x: 0,
    y: 0,
    zoom: 0
  };
  let mouseDown = false;
  const setDigital = (e, value) => {
    switch (e.code) {
      case "KeyW":
        digital.forward = value;
        e.preventDefault();
        e.stopPropagation();
        break;
      case "KeyS":
        digital.backward = value;
        e.preventDefault();
        e.stopPropagation();
        break;
      case "KeyA":
        digital.left = value;
        e.preventDefault();
        e.stopPropagation();
        break;
      case "KeyD":
        digital.right = value;
        e.preventDefault();
        e.stopPropagation();
        break;
      case "Space":
        digital.up = value;
        e.preventDefault();
        e.stopPropagation();
        break;
      case "ShiftLeft":
      case "ControlLeft":
      case "KeyC":
        digital.down = value;
        e.preventDefault();
        e.stopPropagation();
        break;
    }
  };
  window2.addEventListener("keydown", (e) => setDigital(e, true));
  window2.addEventListener("keyup", (e) => setDigital(e, false));
  canvas2.style.touchAction = "pinch-zoom";
  canvas2.addEventListener("pointerdown", () => {
    mouseDown = true;
  });
  canvas2.addEventListener("pointerup", () => {
    mouseDown = false;
  });
  canvas2.addEventListener("pointermove", (e) => {
    mouseDown = e.pointerType == "mouse" ? (e.buttons & 1) !== 0 : true;
    if (mouseDown) {
      analog.x += e.movementX;
      analog.y += e.movementY;
    }
  });
  canvas2.addEventListener(
    "wheel",
    (e) => {
      mouseDown = (e.buttons & 1) !== 0;
      if (mouseDown) {
        analog.zoom += Math.sign(e.deltaY);
        e.preventDefault();
        e.stopPropagation();
      }
    },
    { passive: false }
  );
  return () => {
    const out = {
      digital,
      analog: {
        x: analog.x,
        y: analog.y,
        zoom: analog.zoom,
        touching: mouseDown
      }
    };
    analog.x = 0;
    analog.y = 0;
    analog.zoom = 0;
    return out;
  };
}

// node_modules/lil-gui/dist/lil-gui.esm.js
var Controller = class _Controller {
  constructor(parent, object, property, className, elementType = "div") {
    this.parent = parent;
    this.object = object;
    this.property = property;
    this._disabled = false;
    this._hidden = false;
    this.initialValue = this.getValue();
    this.domElement = document.createElement(elementType);
    this.domElement.classList.add("lil-controller");
    this.domElement.classList.add(className);
    this.$name = document.createElement("div");
    this.$name.classList.add("lil-name");
    _Controller.nextNameID = _Controller.nextNameID || 0;
    this.$name.id = `lil-gui-name-${++_Controller.nextNameID}`;
    this.$widget = document.createElement("div");
    this.$widget.classList.add("lil-widget");
    this.$disable = this.$widget;
    this.domElement.appendChild(this.$name);
    this.domElement.appendChild(this.$widget);
    this.domElement.addEventListener("keydown", (e) => e.stopPropagation());
    this.domElement.addEventListener("keyup", (e) => e.stopPropagation());
    this.parent.children.push(this);
    this.parent.controllers.push(this);
    this.parent.$children.appendChild(this.domElement);
    this._listenCallback = this._listenCallback.bind(this);
    this.name(property);
  }
  /**
   * Sets the name of the controller and its label in the GUI.
   * @param {string} name
   * @returns {this}
   */
  name(name) {
    this._name = name;
    this.$name.textContent = name;
    return this;
  }
  /**
   * Pass a function to be called whenever the value is modified by this controller.
   * The function receives the new value as its first parameter. The value of `this` will be the
   * controller.
   *
   * For function controllers, the `onChange` callback will be fired on click, after the function
   * executes.
   * @param {Function} callback
   * @returns {this}
   * @example
   * const controller = gui.add( object, 'property' );
   *
   * controller.onChange( function( v ) {
   * 	console.log( 'The value is now ' + v );
   * 	console.assert( this === controller );
   * } );
   */
  onChange(callback) {
    this._onChange = callback;
    return this;
  }
  /**
   * Calls the onChange methods of this controller and its parent GUI.
   * @protected
   */
  _callOnChange() {
    this.parent._callOnChange(this);
    if (this._onChange !== void 0) {
      this._onChange.call(this, this.getValue());
    }
    this._changed = true;
  }
  /**
   * Pass a function to be called after this controller has been modified and loses focus.
   * @param {Function} callback
   * @returns {this}
   * @example
   * const controller = gui.add( object, 'property' );
   *
   * controller.onFinishChange( function( v ) {
   * 	console.log( 'Changes complete: ' + v );
   * 	console.assert( this === controller );
   * } );
   */
  onFinishChange(callback) {
    this._onFinishChange = callback;
    return this;
  }
  /**
   * Should be called by Controller when its widgets lose focus.
   * @protected
   */
  _callOnFinishChange() {
    if (this._changed) {
      this.parent._callOnFinishChange(this);
      if (this._onFinishChange !== void 0) {
        this._onFinishChange.call(this, this.getValue());
      }
    }
    this._changed = false;
  }
  /**
   * Sets the controller back to its initial value.
   * @returns {this}
   */
  reset() {
    this.setValue(this.initialValue);
    this._callOnFinishChange();
    return this;
  }
  /**
   * Enables this controller.
   * @param {boolean} enabled
   * @returns {this}
   * @example
   * controller.enable();
   * controller.enable( false ); // disable
   * controller.enable( controller._disabled ); // toggle
   */
  enable(enabled = true) {
    return this.disable(!enabled);
  }
  /**
   * Disables this controller.
   * @param {boolean} disabled
   * @returns {this}
   * @example
   * controller.disable();
   * controller.disable( false ); // enable
   * controller.disable( !controller._disabled ); // toggle
   */
  disable(disabled = true) {
    if (disabled === this._disabled) return this;
    this._disabled = disabled;
    this.domElement.classList.toggle("lil-disabled", disabled);
    this.$disable.toggleAttribute("disabled", disabled);
    return this;
  }
  /**
   * Shows the Controller after it's been hidden.
   * @param {boolean} show
   * @returns {this}
   * @example
   * controller.show();
   * controller.show( false ); // hide
   * controller.show( controller._hidden ); // toggle
   */
  show(show = true) {
    this._hidden = !show;
    this.domElement.style.display = this._hidden ? "none" : "";
    return this;
  }
  /**
   * Hides the Controller.
   * @returns {this}
   */
  hide() {
    return this.show(false);
  }
  /**
   * Changes this controller into a dropdown of options.
   *
   * Calling this method on an option controller will simply update the options. However, if this
   * controller was not already an option controller, old references to this controller are
   * destroyed, and a new controller is added to the end of the GUI.
   * @example
   * // safe usage
   *
   * gui.add( obj, 'prop1' ).options( [ 'a', 'b', 'c' ] );
   * gui.add( obj, 'prop2' ).options( { Big: 10, Small: 1 } );
   * gui.add( obj, 'prop3' );
   *
   * // danger
   *
   * const ctrl1 = gui.add( obj, 'prop1' );
   * gui.add( obj, 'prop2' );
   *
   * // calling options out of order adds a new controller to the end...
   * const ctrl2 = ctrl1.options( [ 'a', 'b', 'c' ] );
   *
   * // ...and ctrl1 now references a controller that doesn't exist
   * assert( ctrl2 !== ctrl1 )
   * @param {object|Array} options
   * @returns {Controller}
   */
  options(options) {
    const controller = this.parent.add(this.object, this.property, options);
    controller.name(this._name);
    this.destroy();
    return controller;
  }
  /**
   * Sets the minimum value. Only works on number controllers.
   * @param {number} min
   * @returns {this}
   */
  min(min) {
    return this;
  }
  /**
   * Sets the maximum value. Only works on number controllers.
   * @param {number} max
   * @returns {this}
   */
  max(max) {
    return this;
  }
  /**
   * Values set by this controller will be rounded to multiples of `step`. Only works on number
   * controllers.
   * @param {number} step
   * @returns {this}
   */
  step(step) {
    return this;
  }
  /**
   * Rounds the displayed value to a fixed number of decimals, without affecting the actual value
   * like `step()`. Only works on number controllers.
   * @example
   * gui.add( object, 'property' ).listen().decimals( 4 );
   * @param {number} decimals
   * @returns {this}
   */
  decimals(decimals) {
    return this;
  }
  /**
   * Calls `updateDisplay()` every animation frame. Pass `false` to stop listening.
   * @param {boolean} listen
   * @returns {this}
   */
  listen(listen = true) {
    this._listening = listen;
    if (this._listenCallbackID !== void 0) {
      cancelAnimationFrame(this._listenCallbackID);
      this._listenCallbackID = void 0;
    }
    if (this._listening) {
      this._listenCallback();
    }
    return this;
  }
  _listenCallback() {
    this._listenCallbackID = requestAnimationFrame(this._listenCallback);
    const curValue = this.save();
    if (curValue !== this._listenPrevValue) {
      this.updateDisplay();
    }
    this._listenPrevValue = curValue;
  }
  /**
   * Returns `object[ property ]`.
   * @returns {any}
   */
  getValue() {
    return this.object[this.property];
  }
  /**
   * Sets the value of `object[ property ]`, invokes any `onChange` handlers and updates the display.
   * @param {any} value
   * @returns {this}
   */
  setValue(value) {
    if (this.getValue() !== value) {
      this.object[this.property] = value;
      this._callOnChange();
      this.updateDisplay();
    }
    return this;
  }
  /**
   * Updates the display to keep it in sync with the current value. Useful for updating your
   * controllers when their values have been modified outside of the GUI.
   * @returns {this}
   */
  updateDisplay() {
    return this;
  }
  load(value) {
    this.setValue(value);
    this._callOnFinishChange();
    return this;
  }
  save() {
    return this.getValue();
  }
  /**
   * Destroys this controller and removes it from the parent GUI.
   */
  destroy() {
    this.listen(false);
    this.parent.children.splice(this.parent.children.indexOf(this), 1);
    this.parent.controllers.splice(this.parent.controllers.indexOf(this), 1);
    this.parent.$children.removeChild(this.domElement);
  }
};
var BooleanController = class extends Controller {
  constructor(parent, object, property) {
    super(parent, object, property, "lil-boolean", "label");
    this.$input = document.createElement("input");
    this.$input.setAttribute("type", "checkbox");
    this.$input.setAttribute("aria-labelledby", this.$name.id);
    this.$widget.appendChild(this.$input);
    this.$input.addEventListener("change", () => {
      this.setValue(this.$input.checked);
      this._callOnFinishChange();
    });
    this.$disable = this.$input;
    this.updateDisplay();
  }
  updateDisplay() {
    this.$input.checked = this.getValue();
    return this;
  }
};
function normalizeColorString(string) {
  let match, result;
  if (match = string.match(/(#|0x)?([a-f0-9]{6})/i)) {
    result = match[2];
  } else if (match = string.match(/rgb\(\s*(\d*)\s*,\s*(\d*)\s*,\s*(\d*)\s*\)/)) {
    result = parseInt(match[1]).toString(16).padStart(2, 0) + parseInt(match[2]).toString(16).padStart(2, 0) + parseInt(match[3]).toString(16).padStart(2, 0);
  } else if (match = string.match(/^#?([a-f0-9])([a-f0-9])([a-f0-9])$/i)) {
    result = match[1] + match[1] + match[2] + match[2] + match[3] + match[3];
  }
  if (result) {
    return "#" + result;
  }
  return false;
}
var STRING = {
  isPrimitive: true,
  match: (v) => typeof v === "string",
  fromHexString: normalizeColorString,
  toHexString: normalizeColorString
};
var INT = {
  isPrimitive: true,
  match: (v) => typeof v === "number",
  fromHexString: (string) => parseInt(string.substring(1), 16),
  toHexString: (value) => "#" + value.toString(16).padStart(6, 0)
};
var ARRAY = {
  isPrimitive: false,
  match: (v) => Array.isArray(v) || ArrayBuffer.isView(v),
  fromHexString(string, target, rgbScale = 1) {
    const int = INT.fromHexString(string);
    target[0] = (int >> 16 & 255) / 255 * rgbScale;
    target[1] = (int >> 8 & 255) / 255 * rgbScale;
    target[2] = (int & 255) / 255 * rgbScale;
  },
  toHexString([r, g, b], rgbScale = 1) {
    rgbScale = 255 / rgbScale;
    const int = r * rgbScale << 16 ^ g * rgbScale << 8 ^ b * rgbScale << 0;
    return INT.toHexString(int);
  }
};
var OBJECT = {
  isPrimitive: false,
  match: (v) => Object(v) === v,
  fromHexString(string, target, rgbScale = 1) {
    const int = INT.fromHexString(string);
    target.r = (int >> 16 & 255) / 255 * rgbScale;
    target.g = (int >> 8 & 255) / 255 * rgbScale;
    target.b = (int & 255) / 255 * rgbScale;
  },
  toHexString({ r, g, b }, rgbScale = 1) {
    rgbScale = 255 / rgbScale;
    const int = r * rgbScale << 16 ^ g * rgbScale << 8 ^ b * rgbScale << 0;
    return INT.toHexString(int);
  }
};
var FORMATS = [STRING, INT, ARRAY, OBJECT];
function getColorFormat(value) {
  return FORMATS.find((format) => format.match(value));
}
var ColorController = class extends Controller {
  constructor(parent, object, property, rgbScale) {
    super(parent, object, property, "lil-color");
    this.$input = document.createElement("input");
    this.$input.setAttribute("type", "color");
    this.$input.setAttribute("tabindex", -1);
    this.$input.setAttribute("aria-labelledby", this.$name.id);
    this.$text = document.createElement("input");
    this.$text.setAttribute("type", "text");
    this.$text.setAttribute("spellcheck", "false");
    this.$text.setAttribute("aria-labelledby", this.$name.id);
    this.$display = document.createElement("div");
    this.$display.classList.add("lil-display");
    this.$display.appendChild(this.$input);
    this.$widget.appendChild(this.$display);
    this.$widget.appendChild(this.$text);
    this._format = getColorFormat(this.initialValue);
    this._rgbScale = rgbScale;
    this._initialValueHexString = this.save();
    this._textFocused = false;
    this.$input.addEventListener("input", () => {
      this._setValueFromHexString(this.$input.value);
    });
    this.$input.addEventListener("blur", () => {
      this._callOnFinishChange();
    });
    this.$text.addEventListener("input", () => {
      const tryParse = normalizeColorString(this.$text.value);
      if (tryParse) {
        this._setValueFromHexString(tryParse);
      }
    });
    this.$text.addEventListener("focus", () => {
      this._textFocused = true;
      this.$text.select();
    });
    this.$text.addEventListener("blur", () => {
      this._textFocused = false;
      this.updateDisplay();
      this._callOnFinishChange();
    });
    this.$disable = this.$text;
    this.updateDisplay();
  }
  reset() {
    this._setValueFromHexString(this._initialValueHexString);
    return this;
  }
  _setValueFromHexString(value) {
    if (this._format.isPrimitive) {
      const newValue = this._format.fromHexString(value);
      this.setValue(newValue);
    } else {
      this._format.fromHexString(value, this.getValue(), this._rgbScale);
      this._callOnChange();
      this.updateDisplay();
    }
  }
  save() {
    return this._format.toHexString(this.getValue(), this._rgbScale);
  }
  load(value) {
    this._setValueFromHexString(value);
    this._callOnFinishChange();
    return this;
  }
  updateDisplay() {
    this.$input.value = this._format.toHexString(this.getValue(), this._rgbScale);
    if (!this._textFocused) {
      this.$text.value = this.$input.value.substring(1);
    }
    this.$display.style.backgroundColor = this.$input.value;
    return this;
  }
};
var FunctionController = class extends Controller {
  constructor(parent, object, property) {
    super(parent, object, property, "lil-function");
    this.$button = document.createElement("button");
    this.$button.appendChild(this.$name);
    this.$widget.appendChild(this.$button);
    this.$button.addEventListener("click", (e) => {
      e.preventDefault();
      this.getValue().call(this.object);
      this._callOnChange();
    });
    this.$button.addEventListener("touchstart", () => {
    }, { passive: true });
    this.$disable = this.$button;
  }
};
var NumberController = class extends Controller {
  constructor(parent, object, property, min, max, step) {
    super(parent, object, property, "lil-number");
    this._initInput();
    this.min(min);
    this.max(max);
    const stepExplicit = step !== void 0;
    this.step(stepExplicit ? step : this._getImplicitStep(), stepExplicit);
    this.updateDisplay();
  }
  decimals(decimals) {
    this._decimals = decimals;
    this.updateDisplay();
    return this;
  }
  min(min) {
    this._min = min;
    this._onUpdateMinMax();
    return this;
  }
  max(max) {
    this._max = max;
    this._onUpdateMinMax();
    return this;
  }
  step(step, explicit = true) {
    this._step = step;
    this._stepExplicit = explicit;
    return this;
  }
  updateDisplay() {
    const value = this.getValue();
    if (this._hasSlider) {
      let percent = (value - this._min) / (this._max - this._min);
      percent = Math.max(0, Math.min(percent, 1));
      this.$fill.style.width = percent * 100 + "%";
    }
    if (!this._inputFocused) {
      this.$input.value = this._decimals === void 0 ? value : value.toFixed(this._decimals);
    }
    return this;
  }
  _initInput() {
    this.$input = document.createElement("input");
    this.$input.setAttribute("type", "text");
    this.$input.setAttribute("aria-labelledby", this.$name.id);
    const isTouch = window.matchMedia("(pointer: coarse)").matches;
    if (isTouch) {
      this.$input.setAttribute("type", "number");
      this.$input.setAttribute("step", "any");
    }
    this.$widget.appendChild(this.$input);
    this.$disable = this.$input;
    const onInput = () => {
      let value = parseFloat(this.$input.value);
      if (isNaN(value)) return;
      if (this._stepExplicit) {
        value = this._snap(value);
      }
      this.setValue(this._clamp(value));
    };
    const increment = (delta) => {
      const value = parseFloat(this.$input.value);
      if (isNaN(value)) return;
      this._snapClampSetValue(value + delta);
      this.$input.value = this.getValue();
    };
    const onKeyDown = (e) => {
      if (e.key === "Enter") {
        this.$input.blur();
      }
      if (e.code === "ArrowUp") {
        e.preventDefault();
        increment(this._step * this._arrowKeyMultiplier(e));
      }
      if (e.code === "ArrowDown") {
        e.preventDefault();
        increment(this._step * this._arrowKeyMultiplier(e) * -1);
      }
    };
    const onWheel = (e) => {
      if (this._inputFocused) {
        e.preventDefault();
        increment(this._step * this._normalizeMouseWheel(e));
      }
    };
    let testingForVerticalDrag = false, initClientX, initClientY, prevClientY, initValue, dragDelta;
    const DRAG_THRESH = 5;
    const onMouseDown = (e) => {
      initClientX = e.clientX;
      initClientY = prevClientY = e.clientY;
      testingForVerticalDrag = true;
      initValue = this.getValue();
      dragDelta = 0;
      window.addEventListener("mousemove", onMouseMove);
      window.addEventListener("mouseup", onMouseUp);
    };
    const onMouseMove = (e) => {
      if (testingForVerticalDrag) {
        const dx = e.clientX - initClientX;
        const dy = e.clientY - initClientY;
        if (Math.abs(dy) > DRAG_THRESH) {
          e.preventDefault();
          this.$input.blur();
          testingForVerticalDrag = false;
          this._setDraggingStyle(true, "vertical");
        } else if (Math.abs(dx) > DRAG_THRESH) {
          onMouseUp();
        }
      }
      if (!testingForVerticalDrag) {
        const dy = e.clientY - prevClientY;
        dragDelta -= dy * this._step * this._arrowKeyMultiplier(e);
        if (initValue + dragDelta > this._max) {
          dragDelta = this._max - initValue;
        } else if (initValue + dragDelta < this._min) {
          dragDelta = this._min - initValue;
        }
        this._snapClampSetValue(initValue + dragDelta);
      }
      prevClientY = e.clientY;
    };
    const onMouseUp = () => {
      this._setDraggingStyle(false, "vertical");
      this._callOnFinishChange();
      window.removeEventListener("mousemove", onMouseMove);
      window.removeEventListener("mouseup", onMouseUp);
    };
    const onFocus = () => {
      this._inputFocused = true;
    };
    const onBlur = () => {
      this._inputFocused = false;
      this.updateDisplay();
      this._callOnFinishChange();
    };
    this.$input.addEventListener("input", onInput);
    this.$input.addEventListener("keydown", onKeyDown);
    this.$input.addEventListener("wheel", onWheel, { passive: false });
    this.$input.addEventListener("mousedown", onMouseDown);
    this.$input.addEventListener("focus", onFocus);
    this.$input.addEventListener("blur", onBlur);
  }
  _initSlider() {
    this._hasSlider = true;
    this.$slider = document.createElement("div");
    this.$slider.classList.add("lil-slider");
    this.$fill = document.createElement("div");
    this.$fill.classList.add("lil-fill");
    this.$slider.appendChild(this.$fill);
    this.$widget.insertBefore(this.$slider, this.$input);
    this.domElement.classList.add("lil-has-slider");
    const map = (v, a, b, c, d) => {
      return (v - a) / (b - a) * (d - c) + c;
    };
    const setValueFromX = (clientX) => {
      const rect = this.$slider.getBoundingClientRect();
      let value = map(clientX, rect.left, rect.right, this._min, this._max);
      this._snapClampSetValue(value);
    };
    const mouseDown = (e) => {
      this._setDraggingStyle(true);
      setValueFromX(e.clientX);
      window.addEventListener("mousemove", mouseMove);
      window.addEventListener("mouseup", mouseUp);
    };
    const mouseMove = (e) => {
      setValueFromX(e.clientX);
    };
    const mouseUp = () => {
      this._callOnFinishChange();
      this._setDraggingStyle(false);
      window.removeEventListener("mousemove", mouseMove);
      window.removeEventListener("mouseup", mouseUp);
    };
    let testingForScroll = false, prevClientX, prevClientY;
    const beginTouchDrag = (e) => {
      e.preventDefault();
      this._setDraggingStyle(true);
      setValueFromX(e.touches[0].clientX);
      testingForScroll = false;
    };
    const onTouchStart = (e) => {
      if (e.touches.length > 1) return;
      if (this._hasScrollBar) {
        prevClientX = e.touches[0].clientX;
        prevClientY = e.touches[0].clientY;
        testingForScroll = true;
      } else {
        beginTouchDrag(e);
      }
      window.addEventListener("touchmove", onTouchMove, { passive: false });
      window.addEventListener("touchend", onTouchEnd);
    };
    const onTouchMove = (e) => {
      if (testingForScroll) {
        const dx = e.touches[0].clientX - prevClientX;
        const dy = e.touches[0].clientY - prevClientY;
        if (Math.abs(dx) > Math.abs(dy)) {
          beginTouchDrag(e);
        } else {
          window.removeEventListener("touchmove", onTouchMove);
          window.removeEventListener("touchend", onTouchEnd);
        }
      } else {
        e.preventDefault();
        setValueFromX(e.touches[0].clientX);
      }
    };
    const onTouchEnd = () => {
      this._callOnFinishChange();
      this._setDraggingStyle(false);
      window.removeEventListener("touchmove", onTouchMove);
      window.removeEventListener("touchend", onTouchEnd);
    };
    const callOnFinishChange = this._callOnFinishChange.bind(this);
    const WHEEL_DEBOUNCE_TIME = 400;
    let wheelFinishChangeTimeout;
    const onWheel = (e) => {
      const isVertical = Math.abs(e.deltaX) < Math.abs(e.deltaY);
      if (isVertical && this._hasScrollBar) return;
      e.preventDefault();
      const delta = this._normalizeMouseWheel(e) * this._step;
      this._snapClampSetValue(this.getValue() + delta);
      this.$input.value = this.getValue();
      clearTimeout(wheelFinishChangeTimeout);
      wheelFinishChangeTimeout = setTimeout(callOnFinishChange, WHEEL_DEBOUNCE_TIME);
    };
    this.$slider.addEventListener("mousedown", mouseDown);
    this.$slider.addEventListener("touchstart", onTouchStart, { passive: false });
    this.$slider.addEventListener("wheel", onWheel, { passive: false });
  }
  _setDraggingStyle(active, axis = "horizontal") {
    if (this.$slider) {
      this.$slider.classList.toggle("lil-active", active);
    }
    document.body.classList.toggle("lil-dragging", active);
    document.body.classList.toggle(`lil-${axis}`, active);
  }
  _getImplicitStep() {
    if (this._hasMin && this._hasMax) {
      return (this._max - this._min) / 1e3;
    }
    return 0.1;
  }
  _onUpdateMinMax() {
    if (!this._hasSlider && this._hasMin && this._hasMax) {
      if (!this._stepExplicit) {
        this.step(this._getImplicitStep(), false);
      }
      this._initSlider();
      this.updateDisplay();
    }
  }
  _normalizeMouseWheel(e) {
    let { deltaX, deltaY } = e;
    if (Math.floor(e.deltaY) !== e.deltaY && e.wheelDelta) {
      deltaX = 0;
      deltaY = -e.wheelDelta / 120;
      deltaY *= this._stepExplicit ? 1 : 10;
    }
    const wheel = deltaX + -deltaY;
    return wheel;
  }
  _arrowKeyMultiplier(e) {
    let mult = this._stepExplicit ? 1 : 10;
    if (e.shiftKey) {
      mult *= 10;
    } else if (e.altKey) {
      mult /= 10;
    }
    return mult;
  }
  _snap(value) {
    let offset = 0;
    if (this._hasMin) {
      offset = this._min;
    } else if (this._hasMax) {
      offset = this._max;
    }
    value -= offset;
    value = Math.round(value / this._step) * this._step;
    value += offset;
    value = parseFloat(value.toPrecision(15));
    return value;
  }
  _clamp(value) {
    if (value < this._min) value = this._min;
    if (value > this._max) value = this._max;
    return value;
  }
  _snapClampSetValue(value) {
    this.setValue(this._clamp(this._snap(value)));
  }
  get _hasScrollBar() {
    const root = this.parent.root.$children;
    return root.scrollHeight > root.clientHeight;
  }
  get _hasMin() {
    return this._min !== void 0;
  }
  get _hasMax() {
    return this._max !== void 0;
  }
};
var OptionController = class extends Controller {
  constructor(parent, object, property, options) {
    super(parent, object, property, "lil-option");
    this.$select = document.createElement("select");
    this.$select.setAttribute("aria-labelledby", this.$name.id);
    this.$display = document.createElement("div");
    this.$display.classList.add("lil-display");
    this.$select.addEventListener("change", () => {
      this.setValue(this._values[this.$select.selectedIndex]);
      this._callOnFinishChange();
    });
    this.$select.addEventListener("focus", () => {
      this.$display.classList.add("lil-focus");
    });
    this.$select.addEventListener("blur", () => {
      this.$display.classList.remove("lil-focus");
    });
    this.$widget.appendChild(this.$select);
    this.$widget.appendChild(this.$display);
    this.$disable = this.$select;
    this.options(options);
  }
  options(options) {
    this._values = Array.isArray(options) ? options : Object.values(options);
    this._names = Array.isArray(options) ? options : Object.keys(options);
    this.$select.replaceChildren();
    this._names.forEach((name) => {
      const $option = document.createElement("option");
      $option.textContent = name;
      this.$select.appendChild($option);
    });
    this.updateDisplay();
    return this;
  }
  updateDisplay() {
    const value = this.getValue();
    const index = this._values.indexOf(value);
    this.$select.selectedIndex = index;
    this.$display.textContent = index === -1 ? value : this._names[index];
    return this;
  }
};
var StringController = class extends Controller {
  constructor(parent, object, property) {
    super(parent, object, property, "lil-string");
    this.$input = document.createElement("input");
    this.$input.setAttribute("type", "text");
    this.$input.setAttribute("spellcheck", "false");
    this.$input.setAttribute("aria-labelledby", this.$name.id);
    this.$input.addEventListener("input", () => {
      this.setValue(this.$input.value);
    });
    this.$input.addEventListener("keydown", (e) => {
      if (e.code === "Enter") {
        this.$input.blur();
      }
    });
    this.$input.addEventListener("blur", () => {
      this._callOnFinishChange();
    });
    this.$widget.appendChild(this.$input);
    this.$disable = this.$input;
    this.updateDisplay();
  }
  updateDisplay() {
    this.$input.value = this.getValue();
    return this;
  }
};
var stylesheet = `.lil-gui {
  font-family: var(--font-family);
  font-size: var(--font-size);
  line-height: 1;
  font-weight: normal;
  font-style: normal;
  text-align: left;
  color: var(--text-color);
  user-select: none;
  -webkit-user-select: none;
  touch-action: manipulation;
  --background-color: #1f1f1f;
  --text-color: #ebebeb;
  --title-background-color: #111111;
  --title-text-color: #ebebeb;
  --widget-color: #424242;
  --hover-color: #4f4f4f;
  --focus-color: #595959;
  --number-color: #2cc9ff;
  --string-color: #a2db3c;
  --font-size: 11px;
  --input-font-size: 11px;
  --font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif;
  --font-family-mono: Menlo, Monaco, Consolas, "Droid Sans Mono", monospace;
  --padding: 4px;
  --spacing: 4px;
  --widget-height: 20px;
  --title-height: calc(var(--widget-height) + var(--spacing) * 1.25);
  --name-width: 45%;
  --slider-knob-width: 2px;
  --slider-input-width: 27%;
  --color-input-width: 27%;
  --slider-input-min-width: 45px;
  --color-input-min-width: 45px;
  --folder-indent: 7px;
  --widget-padding: 0 0 0 3px;
  --widget-border-radius: 2px;
  --checkbox-size: calc(0.75 * var(--widget-height));
  --scrollbar-width: 5px;
}
.lil-gui, .lil-gui * {
  box-sizing: border-box;
  margin: 0;
  padding: 0;
}
.lil-gui.lil-root {
  width: var(--width, 245px);
  display: flex;
  flex-direction: column;
  background: var(--background-color);
}
.lil-gui.lil-root > .lil-title {
  background: var(--title-background-color);
  color: var(--title-text-color);
}
.lil-gui.lil-root > .lil-children {
  overflow-x: hidden;
  overflow-y: auto;
}
.lil-gui.lil-root > .lil-children::-webkit-scrollbar {
  width: var(--scrollbar-width);
  height: var(--scrollbar-width);
  background: var(--background-color);
}
.lil-gui.lil-root > .lil-children::-webkit-scrollbar-thumb {
  border-radius: var(--scrollbar-width);
  background: var(--focus-color);
}
@media (pointer: coarse) {
  .lil-gui.lil-allow-touch-styles, .lil-gui.lil-allow-touch-styles .lil-gui {
    --widget-height: 28px;
    --padding: 6px;
    --spacing: 6px;
    --font-size: 13px;
    --input-font-size: 16px;
    --folder-indent: 10px;
    --scrollbar-width: 7px;
    --slider-input-min-width: 50px;
    --color-input-min-width: 65px;
  }
}
.lil-gui.lil-force-touch-styles, .lil-gui.lil-force-touch-styles .lil-gui {
  --widget-height: 28px;
  --padding: 6px;
  --spacing: 6px;
  --font-size: 13px;
  --input-font-size: 16px;
  --folder-indent: 10px;
  --scrollbar-width: 7px;
  --slider-input-min-width: 50px;
  --color-input-min-width: 65px;
}
.lil-gui.lil-auto-place, .lil-gui.autoPlace {
  max-height: 100%;
  position: fixed;
  top: 0;
  right: 15px;
  z-index: 1001;
}

.lil-controller {
  display: flex;
  align-items: center;
  padding: 0 var(--padding);
  margin: var(--spacing) 0;
}
.lil-controller.lil-disabled {
  opacity: 0.5;
}
.lil-controller.lil-disabled, .lil-controller.lil-disabled * {
  pointer-events: none !important;
}
.lil-controller > .lil-name {
  min-width: var(--name-width);
  flex-shrink: 0;
  white-space: pre;
  padding-right: var(--spacing);
  line-height: var(--widget-height);
}
.lil-controller .lil-widget {
  position: relative;
  display: flex;
  align-items: center;
  width: 100%;
  min-height: var(--widget-height);
}
.lil-controller.lil-string input {
  color: var(--string-color);
}
.lil-controller.lil-boolean {
  cursor: pointer;
}
.lil-controller.lil-color .lil-display {
  width: 100%;
  height: var(--widget-height);
  border-radius: var(--widget-border-radius);
  position: relative;
}
@media (hover: hover) {
  .lil-controller.lil-color .lil-display:hover:before {
    content: " ";
    display: block;
    position: absolute;
    border-radius: var(--widget-border-radius);
    border: 1px solid #fff9;
    top: 0;
    right: 0;
    bottom: 0;
    left: 0;
  }
}
.lil-controller.lil-color input[type=color] {
  opacity: 0;
  width: 100%;
  height: 100%;
  cursor: pointer;
}
.lil-controller.lil-color input[type=text] {
  margin-left: var(--spacing);
  font-family: var(--font-family-mono);
  min-width: var(--color-input-min-width);
  width: var(--color-input-width);
  flex-shrink: 0;
}
.lil-controller.lil-option select {
  opacity: 0;
  position: absolute;
  width: 100%;
  max-width: 100%;
}
.lil-controller.lil-option .lil-display {
  position: relative;
  pointer-events: none;
  border-radius: var(--widget-border-radius);
  height: var(--widget-height);
  line-height: var(--widget-height);
  max-width: 100%;
  overflow: hidden;
  word-break: break-all;
  padding-left: 0.55em;
  padding-right: 1.75em;
  background: var(--widget-color);
}
@media (hover: hover) {
  .lil-controller.lil-option .lil-display.lil-focus {
    background: var(--focus-color);
  }
}
.lil-controller.lil-option .lil-display.lil-active {
  background: var(--focus-color);
}
.lil-controller.lil-option .lil-display:after {
  font-family: "lil-gui";
  content: "\u2195";
  position: absolute;
  top: 0;
  right: 0;
  bottom: 0;
  padding-right: 0.375em;
}
.lil-controller.lil-option .lil-widget,
.lil-controller.lil-option select {
  cursor: pointer;
}
@media (hover: hover) {
  .lil-controller.lil-option .lil-widget:hover .lil-display {
    background: var(--hover-color);
  }
}
.lil-controller.lil-number input {
  color: var(--number-color);
}
.lil-controller.lil-number.lil-has-slider input {
  margin-left: var(--spacing);
  width: var(--slider-input-width);
  min-width: var(--slider-input-min-width);
  flex-shrink: 0;
}
.lil-controller.lil-number .lil-slider {
  width: 100%;
  height: var(--widget-height);
  background: var(--widget-color);
  border-radius: var(--widget-border-radius);
  padding-right: var(--slider-knob-width);
  overflow: hidden;
  cursor: ew-resize;
  touch-action: pan-y;
}
@media (hover: hover) {
  .lil-controller.lil-number .lil-slider:hover {
    background: var(--hover-color);
  }
}
.lil-controller.lil-number .lil-slider.lil-active {
  background: var(--focus-color);
}
.lil-controller.lil-number .lil-slider.lil-active .lil-fill {
  opacity: 0.95;
}
.lil-controller.lil-number .lil-fill {
  height: 100%;
  border-right: var(--slider-knob-width) solid var(--number-color);
  box-sizing: content-box;
}

.lil-dragging .lil-gui {
  --hover-color: var(--widget-color);
}
.lil-dragging * {
  cursor: ew-resize !important;
}
.lil-dragging.lil-vertical * {
  cursor: ns-resize !important;
}

.lil-gui .lil-title {
  height: var(--title-height);
  font-weight: 600;
  padding: 0 var(--padding);
  width: 100%;
  text-align: left;
  background: none;
  text-decoration-skip: objects;
}
.lil-gui .lil-title:before {
  font-family: "lil-gui";
  content: "\u25BE";
  padding-right: 2px;
  display: inline-block;
}
.lil-gui .lil-title:active {
  background: var(--title-background-color);
  opacity: 0.75;
}
@media (hover: hover) {
  body:not(.lil-dragging) .lil-gui .lil-title:hover {
    background: var(--title-background-color);
    opacity: 0.85;
  }
  .lil-gui .lil-title:focus {
    text-decoration: underline var(--focus-color);
  }
}
.lil-gui.lil-root > .lil-title:focus {
  text-decoration: none !important;
}
.lil-gui.lil-closed > .lil-title:before {
  content: "\u25B8";
}
.lil-gui.lil-closed > .lil-children {
  transform: translateY(-7px);
  opacity: 0;
}
.lil-gui.lil-closed:not(.lil-transition) > .lil-children {
  display: none;
}
.lil-gui.lil-transition > .lil-children {
  transition-duration: 300ms;
  transition-property: height, opacity, transform;
  transition-timing-function: cubic-bezier(0.2, 0.6, 0.35, 1);
  overflow: hidden;
  pointer-events: none;
}
.lil-gui .lil-children:empty:before {
  content: "Empty";
  padding: 0 var(--padding);
  margin: var(--spacing) 0;
  display: block;
  height: var(--widget-height);
  font-style: italic;
  line-height: var(--widget-height);
  opacity: 0.5;
}
.lil-gui.lil-root > .lil-children > .lil-gui > .lil-title {
  border: 0 solid var(--widget-color);
  border-width: 1px 0;
  transition: border-color 300ms;
}
.lil-gui.lil-root > .lil-children > .lil-gui.lil-closed > .lil-title {
  border-bottom-color: transparent;
}
.lil-gui + .lil-controller {
  border-top: 1px solid var(--widget-color);
  margin-top: 0;
  padding-top: var(--spacing);
}
.lil-gui .lil-gui .lil-gui > .lil-title {
  border: none;
}
.lil-gui .lil-gui .lil-gui > .lil-children {
  border: none;
  margin-left: var(--folder-indent);
  border-left: 2px solid var(--widget-color);
}
.lil-gui .lil-gui .lil-controller {
  border: none;
}

.lil-gui label, .lil-gui input, .lil-gui button {
  -webkit-tap-highlight-color: transparent;
}
.lil-gui input {
  border: 0;
  outline: none;
  font-family: var(--font-family);
  font-size: var(--input-font-size);
  border-radius: var(--widget-border-radius);
  height: var(--widget-height);
  background: var(--widget-color);
  color: var(--text-color);
  width: 100%;
}
@media (hover: hover) {
  .lil-gui input:hover {
    background: var(--hover-color);
  }
  .lil-gui input:active {
    background: var(--focus-color);
  }
}
.lil-gui input:disabled {
  opacity: 1;
}
.lil-gui input[type=text],
.lil-gui input[type=number] {
  padding: var(--widget-padding);
  -moz-appearance: textfield;
}
.lil-gui input[type=text]:focus,
.lil-gui input[type=number]:focus {
  background: var(--focus-color);
}
.lil-gui input[type=checkbox] {
  appearance: none;
  width: var(--checkbox-size);
  height: var(--checkbox-size);
  border-radius: var(--widget-border-radius);
  text-align: center;
  cursor: pointer;
}
.lil-gui input[type=checkbox]:checked:before {
  font-family: "lil-gui";
  content: "\u2713";
  font-size: var(--checkbox-size);
  line-height: var(--checkbox-size);
}
@media (hover: hover) {
  .lil-gui input[type=checkbox]:focus {
    box-shadow: inset 0 0 0 1px var(--focus-color);
  }
}
.lil-gui button {
  outline: none;
  cursor: pointer;
  font-family: var(--font-family);
  font-size: var(--font-size);
  color: var(--text-color);
  width: 100%;
  border: none;
}
.lil-gui .lil-controller button {
  height: var(--widget-height);
  text-transform: none;
  background: var(--widget-color);
  border-radius: var(--widget-border-radius);
}
@media (hover: hover) {
  .lil-gui .lil-controller button:hover {
    background: var(--hover-color);
  }
  .lil-gui .lil-controller button:focus {
    box-shadow: inset 0 0 0 1px var(--focus-color);
  }
}
.lil-gui .lil-controller button:active {
  background: var(--focus-color);
}

@font-face {
  font-family: "lil-gui";
  src: url("data:application/font-woff2;charset=utf-8;base64,d09GMgABAAAAAALkAAsAAAAABtQAAAKVAAEAAAAAAAAAAAAAAAAAAAAAAAAAAAAAHFQGYACDMgqBBIEbATYCJAMUCwwABCAFhAoHgQQbHAbIDiUFEYVARAAAYQTVWNmz9MxhEgodq49wYRUFKE8GWNiUBxI2LBRaVnc51U83Gmhs0Q7JXWMiz5eteLwrKwuxHO8VFxUX9UpZBs6pa5ABRwHA+t3UxUnH20EvVknRerzQgX6xC/GH6ZUvTcAjAv122dF28OTqCXrPuyaDER30YBA1xnkVutDDo4oCi71Ca7rrV9xS8dZHbPHefsuwIyCpmT7j+MnjAH5X3984UZoFFuJ0yiZ4XEJFxjagEBeqs+e1iyK8Xf/nOuwF+vVK0ur765+vf7txotUi0m3N0m/84RGSrBCNrh8Ee5GjODjF4gnWP+dJrH/Lk9k4oT6d+gr6g/wssA2j64JJGP6cmx554vUZnpZfn6ZfX2bMwPPrlANsB86/DiHjhl0OP+c87+gaJo/gY084s3HoYL/ZkWHTRfBXvvoHnnkHvngKun4KBE/ede7tvq3/vQOxDXB1/fdNz6XbPdcr0Vhpojj9dG+owuSKFsslCi1tgEjirjXdwMiov2EioadxmqTHUCIwo8NgQaeIasAi0fTYSPTbSmwbMOFduyh9wvBrESGY0MtgRjtgQR8Q1bRPohn2UoCRZf9wyYANMXFeJTysqAe0I4mrherOekFdKMrYvJjLvOIUM9SuwYB5DVZUwwVjJJOaUnZCmcEkIZZrKqNvRGRMvmFZsmhP4VMKCSXBhSqUBxgMS7h0cZvEd71AWkEhGWaeMFcNnpqyJkyXgYL7PQ1MoSq0wDAkRtJIijkZSmqYTiSImfLiSWXIZwhRh3Rug2X0kk1Dgj+Iu43u5p98ghopcpSo0Uyc8SnjlYX59WUeaMoDqmVD2TOWD9a4pCRAzf2ECgwGcrHjPOWY9bNxq/OL3I/QjwEAAAA=") format("woff2");
}`;
function _injectStyles(cssContent) {
  const injected = document.createElement("style");
  injected.innerHTML = cssContent;
  const before = document.querySelector("head link[rel=stylesheet], head style");
  if (before) {
    document.head.insertBefore(injected, before);
  } else {
    document.head.appendChild(injected);
  }
}
var stylesInjected = false;
var GUI = class _GUI {
  /**
   * Creates a panel that holds controllers.
   * @example
   * new GUI();
   * new GUI( { container: document.getElementById( 'custom' ) } );
   *
   * @param {object} [options]
   * @param {boolean} [options.autoPlace=true]
   * Adds the GUI to `document.body` and fixes it to the top right of the page.
   *
   * @param {Node} [options.container]
   * Adds the GUI to this DOM element. Overrides `autoPlace`.
   *
   * @param {number} [options.width=245]
   * Width of the GUI in pixels, usually set when name labels become too long. Note that you can make
   * name labels wider in CSS with `.lil‑gui { ‑‑name‑width: 55% }`.
   *
   * @param {string} [options.title=Controls]
   * Name to display in the title bar.
   *
   * @param {boolean} [options.closeFolders=false]
   * Pass `true` to close all folders in this GUI by default.
   *
   * @param {boolean} [options.injectStyles=true]
   * Injects the default stylesheet into the page if this is the first GUI.
   * Pass `false` to use your own stylesheet.
   *
   * @param {number} [options.touchStyles=true]
   * Makes controllers larger on touch devices. Pass `false` to disable touch styles.
   *
   * @param {GUI} [options.parent]
   * Adds this GUI as a child in another GUI. Usually this is done for you by `addFolder()`.
   */
  constructor({
    parent,
    autoPlace = parent === void 0,
    container,
    width: width2,
    title = "Controls",
    closeFolders = false,
    injectStyles = true,
    touchStyles = true
  } = {}) {
    this.parent = parent;
    this.root = parent ? parent.root : this;
    this.children = [];
    this.controllers = [];
    this.folders = [];
    this._closed = false;
    this._hidden = false;
    this.domElement = document.createElement("div");
    this.domElement.classList.add("lil-gui");
    this.$title = document.createElement("button");
    this.$title.classList.add("lil-title");
    this.$title.setAttribute("aria-expanded", true);
    this.$title.addEventListener("click", () => this.openAnimated(this._closed));
    this.$title.addEventListener("touchstart", () => {
    }, { passive: true });
    this.$children = document.createElement("div");
    this.$children.classList.add("lil-children");
    this.domElement.appendChild(this.$title);
    this.domElement.appendChild(this.$children);
    this.title(title);
    if (this.parent) {
      this.parent.children.push(this);
      this.parent.folders.push(this);
      this.parent.$children.appendChild(this.domElement);
      return;
    }
    this.domElement.classList.add("lil-root");
    if (touchStyles) {
      this.domElement.classList.add("lil-allow-touch-styles");
    }
    if (!stylesInjected && injectStyles) {
      _injectStyles(stylesheet);
      stylesInjected = true;
    }
    if (container) {
      container.appendChild(this.domElement);
    } else if (autoPlace) {
      this.domElement.classList.add("lil-auto-place", "autoPlace");
      document.body.appendChild(this.domElement);
    }
    if (width2) {
      this.domElement.style.setProperty("--width", width2 + "px");
    }
    this._closeFolders = closeFolders;
  }
  /**
   * Adds a controller to the GUI, inferring controller type using the `typeof` operator.
   * @example
   * gui.add( object, 'property' );
   * gui.add( object, 'number', 0, 100, 1 );
   * gui.add( object, 'options', [ 1, 2, 3 ] );
   *
   * @param {object} object The object the controller will modify.
   * @param {string} property Name of the property to control.
   * @param {number|object|Array} [$1] Minimum value for number controllers, or the set of
   * selectable values for a dropdown.
   * @param {number} [max] Maximum value for number controllers.
   * @param {number} [step] Step value for number controllers.
   * @returns {Controller}
   */
  add(object, property, $1, max, step) {
    if (Object($1) === $1) {
      return new OptionController(this, object, property, $1);
    }
    const initialValue = object[property];
    switch (typeof initialValue) {
      case "number":
        return new NumberController(this, object, property, $1, max, step);
      case "boolean":
        return new BooleanController(this, object, property);
      case "string":
        return new StringController(this, object, property);
      case "function":
        return new FunctionController(this, object, property);
    }
    console.error(`gui.add failed
	property:`, property, `
	object:`, object, `
	value:`, initialValue);
  }
  /**
   * Adds a color controller to the GUI.
   * @example
   * params = {
   * 	cssColor: '#ff00ff',
   * 	rgbColor: { r: 0, g: 0.2, b: 0.4 },
   * 	customRange: [ 0, 127, 255 ],
   * };
   *
   * gui.addColor( params, 'cssColor' );
   * gui.addColor( params, 'rgbColor' );
   * gui.addColor( params, 'customRange', 255 );
   *
   * @param {object} object The object the controller will modify.
   * @param {string} property Name of the property to control.
   * @param {number} rgbScale Maximum value for a color channel when using an RGB color. You may
   * need to set this to 255 if your colors are too bright.
   * @returns {Controller}
   */
  addColor(object, property, rgbScale = 1) {
    return new ColorController(this, object, property, rgbScale);
  }
  /**
   * Adds a folder to the GUI, which is just another GUI. This method returns
   * the nested GUI so you can add controllers to it.
   * @example
   * const folder = gui.addFolder( 'Position' );
   * folder.add( position, 'x' );
   * folder.add( position, 'y' );
   * folder.add( position, 'z' );
   *
   * @param {string} title Name to display in the folder's title bar.
   * @returns {GUI}
   */
  addFolder(title) {
    const folder = new _GUI({ parent: this, title });
    if (this.root._closeFolders) folder.close();
    return folder;
  }
  /**
   * Recalls values that were saved with `gui.save()`.
   * @param {object} obj
   * @param {boolean} recursive Pass false to exclude folders descending from this GUI.
   * @returns {this}
   */
  load(obj, recursive = true) {
    if (obj.controllers) {
      this.controllers.forEach((c) => {
        if (c instanceof FunctionController) return;
        if (c._name in obj.controllers) {
          c.load(obj.controllers[c._name]);
        }
      });
    }
    if (recursive && obj.folders) {
      this.folders.forEach((f) => {
        if (f._title in obj.folders) {
          f.load(obj.folders[f._title]);
        }
      });
    }
    return this;
  }
  /**
   * Returns an object mapping controller names to values. The object can be passed to `gui.load()` to
   * recall these values.
   * @example
   * {
   * 	controllers: {
   * 		prop1: 1,
   * 		prop2: 'value',
   * 		...
   * 	},
   * 	folders: {
   * 		folderName1: { controllers, folders },
   * 		folderName2: { controllers, folders }
   * 		...
   * 	}
   * }
   *
   * @param {boolean} recursive Pass false to exclude folders descending from this GUI.
   * @returns {object}
   */
  save(recursive = true) {
    const obj = {
      controllers: {},
      folders: {}
    };
    this.controllers.forEach((c) => {
      if (c instanceof FunctionController) return;
      if (c._name in obj.controllers) {
        throw new Error(`Cannot save GUI with duplicate property "${c._name}"`);
      }
      obj.controllers[c._name] = c.save();
    });
    if (recursive) {
      this.folders.forEach((f) => {
        if (f._title in obj.folders) {
          throw new Error(`Cannot save GUI with duplicate folder "${f._title}"`);
        }
        obj.folders[f._title] = f.save();
      });
    }
    return obj;
  }
  /**
   * Opens a GUI or folder. GUI and folders are open by default.
   * @param {boolean} open Pass false to close.
   * @returns {this}
   * @example
   * gui.open(); // open
   * gui.open( false ); // close
   * gui.open( gui._closed ); // toggle
   */
  open(open = true) {
    this._setClosed(!open);
    this.$title.setAttribute("aria-expanded", !this._closed);
    this.domElement.classList.toggle("lil-closed", this._closed);
    return this;
  }
  /**
   * Closes the GUI.
   * @returns {this}
   */
  close() {
    return this.open(false);
  }
  _setClosed(closed) {
    if (this._closed === closed) return;
    this._closed = closed;
    this._callOnOpenClose(this);
  }
  /**
   * Shows the GUI after it's been hidden.
   * @param {boolean} show
   * @returns {this}
   * @example
   * gui.show();
   * gui.show( false ); // hide
   * gui.show( gui._hidden ); // toggle
   */
  show(show = true) {
    this._hidden = !show;
    this.domElement.style.display = this._hidden ? "none" : "";
    return this;
  }
  /**
   * Hides the GUI.
   * @returns {this}
   */
  hide() {
    return this.show(false);
  }
  openAnimated(open = true) {
    this._setClosed(!open);
    this.$title.setAttribute("aria-expanded", !this._closed);
    requestAnimationFrame(() => {
      const initialHeight = this.$children.clientHeight;
      this.$children.style.height = initialHeight + "px";
      this.domElement.classList.add("lil-transition");
      const onTransitionEnd = (e) => {
        if (e.target !== this.$children) return;
        this.$children.style.height = "";
        this.domElement.classList.remove("lil-transition");
        this.$children.removeEventListener("transitionend", onTransitionEnd);
      };
      this.$children.addEventListener("transitionend", onTransitionEnd);
      const targetHeight = !open ? 0 : this.$children.scrollHeight;
      this.domElement.classList.toggle("lil-closed", !open);
      requestAnimationFrame(() => {
        this.$children.style.height = targetHeight + "px";
      });
    });
    return this;
  }
  /**
   * Change the title of this GUI.
   * @param {string} title
   * @returns {this}
   */
  title(title) {
    this._title = title;
    this.$title.textContent = title;
    return this;
  }
  /**
   * Resets all controllers to their initial values.
   * @param {boolean} recursive Pass false to exclude folders descending from this GUI.
   * @returns {this}
   */
  reset(recursive = true) {
    const controllers = recursive ? this.controllersRecursive() : this.controllers;
    controllers.forEach((c) => c.reset());
    return this;
  }
  /**
   * Pass a function to be called whenever a controller in this GUI changes.
   * @param {function({object:object, property:string, value:any, controller:Controller})} callback
   * @returns {this}
   * @example
   * gui.onChange( event => {
   * 	event.object     // object that was modified
   * 	event.property   // string, name of property
   * 	event.value      // new value of controller
   * 	event.controller // controller that was modified
   * } );
   */
  onChange(callback) {
    this._onChange = callback;
    return this;
  }
  _callOnChange(controller) {
    if (this.parent) {
      this.parent._callOnChange(controller);
    }
    if (this._onChange !== void 0) {
      this._onChange.call(this, {
        object: controller.object,
        property: controller.property,
        value: controller.getValue(),
        controller
      });
    }
  }
  /**
   * Pass a function to be called whenever a controller in this GUI has finished changing.
   * @param {function({object:object, property:string, value:any, controller:Controller})} callback
   * @returns {this}
   * @example
   * gui.onFinishChange( event => {
   * 	event.object     // object that was modified
   * 	event.property   // string, name of property
   * 	event.value      // new value of controller
   * 	event.controller // controller that was modified
   * } );
   */
  onFinishChange(callback) {
    this._onFinishChange = callback;
    return this;
  }
  _callOnFinishChange(controller) {
    if (this.parent) {
      this.parent._callOnFinishChange(controller);
    }
    if (this._onFinishChange !== void 0) {
      this._onFinishChange.call(this, {
        object: controller.object,
        property: controller.property,
        value: controller.getValue(),
        controller
      });
    }
  }
  /**
   * Pass a function to be called when this GUI or its descendants are opened or closed.
   * @param {function(GUI)} callback
   * @returns {this}
   * @example
   * gui.onOpenClose( changedGUI => {
   * 	console.log( changedGUI._closed );
   * } );
   */
  onOpenClose(callback) {
    this._onOpenClose = callback;
    return this;
  }
  _callOnOpenClose(changedGUI) {
    if (this.parent) {
      this.parent._callOnOpenClose(changedGUI);
    }
    if (this._onOpenClose !== void 0) {
      this._onOpenClose.call(this, changedGUI);
    }
  }
  /**
   * Destroys all DOM elements and event listeners associated with this GUI.
   */
  destroy() {
    if (this.parent) {
      this.parent.children.splice(this.parent.children.indexOf(this), 1);
      this.parent.folders.splice(this.parent.folders.indexOf(this), 1);
    }
    if (this.domElement.parentElement) {
      this.domElement.parentElement.removeChild(this.domElement);
    }
    Array.from(this.children).forEach((c) => c.destroy());
  }
  /**
   * Returns an array of controllers contained by this GUI and its descendents.
   * @returns {Controller[]}
   */
  controllersRecursive() {
    let controllers = Array.from(this.controllers);
    this.folders.forEach((f) => {
      controllers = controllers.concat(f.controllersRecursive());
    });
    return controllers;
  }
  /**
   * Returns an array of folders contained by this GUI and its descendents.
   * @returns {GUI[]}
   */
  foldersRecursive() {
    let folders = Array.from(this.folders);
    this.folders.forEach((f) => {
      folders = folders.concat(f.foldersRecursive());
    });
    return folders;
  }
};

// lib/gui.ts
var gui = new GUI();
var controls = {
  pause: false,
  cameraType: "arcball",
  highDPI: false,
  bunnyRotation: 0
};
var pauseController = gui.add(controls, "pause").name("Pause");
var cameraController = gui.add(controls, "cameraType", ["arcball", "WASD"]).name("Camera Type");
var highDPIController = gui.add(controls, "highDPI").name("High DPI");
var rotationController = gui.add(controls, "bunnyRotation", 0, 360, 1).name("Bunny Rotation");

// lib/TimestampQueryManager.ts
var TimestampQueryManager = class {
  // The device may not support timestamp queries, on which case this whole
  // class does nothing.
  timestampSupported;
  // The query objects. This is meant to be used in a ComputePassDescriptor's
  // or RenderPassDescriptor's 'timestampWrites' field.
  #timestampQuerySet;
  // A buffer where to store query results
  #timestampBuffer;
  // A buffer to map this result back to CPU
  #timestampMapBuffer;
  // Callback to call when results are available.
  #callback;
  // Device must have the "timestamp-query" feature
  constructor(device2, callback) {
    this.timestampSupported = device2.features.has("timestamp-query");
    if (!this.timestampSupported) return;
    this.#callback = callback;
    this.#timestampQuerySet = device2.createQuerySet({
      type: "timestamp",
      count: 2
      // begin and end
    });
    const timestampByteSize = 8;
    this.#timestampBuffer = device2.createBuffer({
      size: this.#timestampQuerySet.count * timestampByteSize,
      usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.QUERY_RESOLVE
    });
    this.#timestampMapBuffer = device2.createBuffer({
      size: this.#timestampBuffer.size,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
    });
  }
  // Add both a start and end timestamp.
  addTimestampWrite(passDescriptor) {
    if (this.timestampSupported) {
      passDescriptor.timestampWrites = {
        querySet: this.#timestampQuerySet,
        beginningOfPassWriteIndex: 0,
        endOfPassWriteIndex: 1
      };
    }
    return passDescriptor;
  }
  // Resolve the timestamp queries and copy the result into the mappable buffer if possible.
  resolve(commandEncoder) {
    if (!this.timestampSupported) return;
    commandEncoder.resolveQuerySet(
      this.#timestampQuerySet,
      0,
      this.#timestampQuerySet.count,
      this.#timestampBuffer,
      0
      /* destinationOffset */
    );
    if (this.#timestampMapBuffer.mapState === "unmapped") {
      commandEncoder.copyBufferToBuffer(
        this.#timestampBuffer,
        0,
        // source offset
        this.#timestampMapBuffer,
        0,
        // destination offset
        this.#timestampBuffer.size
        // size
      );
    }
  }
  // Read the values of timestamps.
  tryInitiateTimestampDownload() {
    if (!this.timestampSupported) return;
    if (this.#timestampMapBuffer.mapState !== "unmapped") return;
    const buffer = this.#timestampMapBuffer;
    void buffer.mapAsync(GPUMapMode.READ).then(() => {
      const rawData = buffer.getMappedRange();
      const timestamps = new BigUint64Array(rawData);
      const elapsedNs = Number(timestamps[1] - timestamps[0]);
      if (elapsedNs >= 0) {
        this.#callback(elapsedNs);
      }
      buffer.unmap();
    });
  }
};

// lib/Stats.ts
var Panel = class {
  canvas;
  context;
  name;
  fg;
  bg;
  min = Infinity;
  max = 0;
  round = Math.round;
  PR = Math.round(window.devicePixelRatio || 1);
  WIDTH = 80 * this.PR;
  HEIGHT = 48 * this.PR;
  TEXT_X = 3 * this.PR;
  TEXT_Y = 2 * this.PR;
  GRAPH_X = 3 * this.PR;
  GRAPH_Y = 15 * this.PR;
  GRAPH_WIDTH = 74 * this.PR;
  GRAPH_HEIGHT = 30 * this.PR;
  constructor(name, fg, bg) {
    this.name = name;
    this.fg = fg;
    this.bg = bg;
    this.canvas = document.createElement("canvas");
    this.canvas.width = this.WIDTH;
    this.canvas.height = this.HEIGHT;
    this.canvas.style.cssText = "width:80px;height:48px";
    this.context = this.canvas.getContext("2d");
    this.context.font = "bold " + 9 * this.PR + "px Helvetica,Arial,sans-serif";
    this.context.textBaseline = "top";
    this.context.fillStyle = bg;
    this.context.fillRect(0, 0, this.WIDTH, this.HEIGHT);
    this.context.fillStyle = fg;
    this.context.fillText(name, this.TEXT_X, this.TEXT_Y);
    this.context.fillRect(this.GRAPH_X, this.GRAPH_Y, this.GRAPH_WIDTH, this.GRAPH_HEIGHT);
    this.context.fillStyle = bg;
    this.context.globalAlpha = 0.9;
    this.context.fillRect(this.GRAPH_X, this.GRAPH_Y, this.GRAPH_WIDTH, this.GRAPH_HEIGHT);
  }
  update(value, maxValue) {
    this.min = Math.min(this.min, value);
    this.max = Math.max(this.max, value);
    this.context.fillStyle = this.bg;
    this.context.globalAlpha = 1;
    this.context.fillRect(0, 0, this.WIDTH, this.GRAPH_Y);
    this.context.fillStyle = this.fg;
    this.context.fillText(
      this.round(value) + " " + this.name + " (" + this.round(this.min) + "-" + this.round(this.max) + ")",
      this.TEXT_X,
      this.TEXT_Y
    );
    this.context.drawImage(
      this.canvas,
      this.GRAPH_X + this.PR,
      this.GRAPH_Y,
      this.GRAPH_WIDTH - this.PR,
      this.GRAPH_HEIGHT,
      this.GRAPH_X,
      this.GRAPH_Y,
      this.GRAPH_WIDTH - this.PR,
      this.GRAPH_HEIGHT
    );
    this.context.fillRect(this.GRAPH_X + this.GRAPH_WIDTH - this.PR, this.GRAPH_Y, this.PR, this.GRAPH_HEIGHT);
    this.context.fillStyle = this.bg;
    this.context.globalAlpha = 0.9;
    this.context.fillRect(
      this.GRAPH_X + this.GRAPH_WIDTH - this.PR,
      this.GRAPH_Y,
      this.PR,
      this.round((1 - value / maxValue) * this.GRAPH_HEIGHT)
    );
  }
  get dom() {
    return this.canvas;
  }
};
var Stats = class {
  container;
  fpsPanel;
  //private msPanel: Panel;
  memPanel = null;
  beginTime = (performance || Date).now();
  prevTime = this.beginTime;
  frames = 0;
  constructor() {
    this.container = document.createElement("div");
    this.container.style.cssText = "position:fixed;top:0;left:0;opacity:0.9;z-index:10000";
    this.fpsPanel = this.addPanel(new Panel("FPS", "#0ff", "#002"));
    if (performance.memory) {
      this.memPanel = this.addPanel(new Panel("MB", "#f08", "#201"));
    }
  }
  addPanel(panel) {
    this.container.appendChild(panel.dom);
    return panel;
  }
  begin() {
    this.beginTime = (performance || Date).now();
    return this.beginTime;
  }
  end() {
    this.frames++;
    const time = (performance || Date).now();
    const frameTime = time - this.beginTime;
    if (time >= this.prevTime + 1e3) {
      console.log(frameTime);
      this.fpsPanel.update(this.frames * 1e3 / (time - this.prevTime), 100);
      this.prevTime = time;
      this.frames = 0;
      if (this.memPanel && performance.memory) {
        const memory = performance.memory;
        this.memPanel.update(memory.usedJSHeapSize / 1048576, memory.jsHeapSizeLimit / 1048576);
      }
    }
    return time;
  }
  get dom() {
    return this.container;
  }
  static Panel = Panel;
};

// index.ts
var canvas = document.getElementById("canvas");
var infoTextElement = document.getElementById("info-text");
if (!canvas) {
  throw new Error("No canvas found.");
}
if (!navigator.gpu) {
  throw new Error("WebGPU not supported on this browser.");
}
console.log("WebGPU is supported!");
var adapter = await navigator.gpu.requestAdapter({
  featureLevel: "compatibility"
});
if (!adapter) {
  throw new Error("No appropriate GPUAdapter found.");
}
var width = canvas.width;
var height = canvas.height;
var raytracedTexture;
var displayBindGroup;
var computeBindGroup;
function resizeCanvas() {
  const pixelRatio = controls.highDPI ? window.devicePixelRatio : 1;
  canvas.width = window.innerWidth * pixelRatio;
  canvas.height = window.innerHeight * pixelRatio;
  width = canvas.width;
  height = canvas.height;
  if (raytracedTexture) {
    createGPUResources();
  }
}
resizeCanvas();
window.addEventListener("resize", resizeCanvas);
highDPIController.onChange(() => {
  resizeCanvas();
});
var supportsTimestampQueries = adapter?.features.has("timestamp-query");
var device = await adapter.requestDevice({
  // We request a device that has support for timestamp queries
  requiredFeatures: supportsTimestampQueries ? ["timestamp-query"] : []
});
device.addEventListener("uncapturederror", (event) => {
  console.log(event.error);
});
var context = canvas.getContext("webgpu");
if (!context) {
  throw new Error("No context found.");
}
var stats = new Stats();
var gpuPanel = stats.addPanel(new Stats.Panel("GPU", "#ff8", "#221"));
document.body.appendChild(stats.dom);
var timestampQueryManager = new TimestampQueryManager(device, (elapsedNs) => {
  const elapsedMs = Number(elapsedNs) * 1e-6;
  gpuPanel.update(elapsedMs, 16);
});
var canvasFormat = navigator.gpu.getPreferredCanvasFormat();
context.configure({
  device,
  format: canvasFormat
});
var vertices = new Float32Array([
  // X,  Y,
  -1,
  3,
  // Triangle 1
  3,
  -1,
  -1,
  -1
]);
var vertexBuffer = device.createBuffer({
  // Labels are useful for debugging.
  label: "Display vertices",
  // 4 bytes * 6 vertices = 24 bytes.
  size: vertices.byteLength,
  // The buffer will be used as the source of vertex data.
  usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST
});
device.queue.writeBuffer(
  vertexBuffer,
  /* offset */
  0,
  vertices
);
var vertexBufferLayout = {
  // 2 floats for position.
  arrayStride: 8,
  attributes: [{
    format: "float32x2",
    offset: 0,
    shaderLocation: 0
    // Position, see vertex shader
  }]
};
function createGPUResources() {
  if (raytracedTexture) {
    raytracedTexture.destroy();
  }
  raytracedTexture = device.createTexture({
    size: [width, height],
    format: "rgba8unorm",
    usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC
  });
  displayBindGroup = device.createBindGroup({
    label: "Display bind group",
    layout: displayPipeline.getBindGroupLayout(0),
    entries: [
      {
        binding: 0,
        // Corresponds to @binding(0) in the shader.
        resource: raytracedTexture.createView()
      },
      {
        binding: 1,
        // Corresponds to @binding(1) in the shader.
        resource: displaySampler
      }
    ]
  });
  computeBindGroup = device.createBindGroup({
    label: "Raytracing Bind Group",
    layout: computeBindGroupLayout,
    entries: [
      {
        binding: 0,
        resource: raytracedTexture.createView()
      },
      {
        binding: 1,
        resource: {
          buffer: inputBuffer
        }
      },
      {
        binding: 2,
        resource: {
          buffer: gridsBuffer
        }
      },
      {
        binding: 3,
        resource: {
          buffer: rootsBuffer
        }
      },
      {
        binding: 4,
        resource: {
          buffer: uppersBuffer
        }
      },
      {
        binding: 5,
        resource: {
          buffer: lowersBuffer
        }
      },
      {
        binding: 6,
        resource: {
          buffer: leavesBuffer
        }
      },
      {
        binding: 7,
        resource: {
          buffer: dataBuffer
        }
      }
    ]
  });
}
var displaySampler = device.createSampler({
  addressModeU: "clamp-to-edge",
  addressModeV: "clamp-to-edge",
  magFilter: "linear",
  minFilter: "linear"
});
var displayShaderModule = device.createShaderModule({
  label: "Display shader",
  code: blit_default
});
var displayPipeline = device.createRenderPipeline({
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
      format: canvasFormat
    }]
  }
});
var inputHandler = createInputHandler(window, canvas);
infoTextElement.textContent = "Loading bunny.pvdb.gz...";
var picoVDBFile = await loadPicoVDB("./bunny.pvdb.gz");
var gridsBuffer = device.createBuffer({
  label: "PicoVDB Grids",
  size: picoVDBFile.gridsBuffer.byteLength,
  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
});
device.queue.writeBuffer(gridsBuffer, 0, picoVDBFile.gridsBuffer);
var rootsBuffer = device.createBuffer({
  label: "PicoVDB Roots",
  size: picoVDBFile.rootsBuffer.byteLength,
  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
});
device.queue.writeBuffer(rootsBuffer, 0, picoVDBFile.rootsBuffer);
var uppersBuffer = device.createBuffer({
  label: "PicoVDB Uppers",
  size: picoVDBFile.uppersBuffer.byteLength,
  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
});
device.queue.writeBuffer(uppersBuffer, 0, picoVDBFile.uppersBuffer);
var lowersBuffer = device.createBuffer({
  label: "PicoVDB Lowers",
  size: picoVDBFile.lowersBuffer.byteLength,
  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
});
device.queue.writeBuffer(lowersBuffer, 0, picoVDBFile.lowersBuffer);
var leavesBuffer = device.createBuffer({
  label: "PicoVDB Leaves",
  size: picoVDBFile.leavesBuffer.byteLength,
  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
});
device.queue.writeBuffer(leavesBuffer, 0, picoVDBFile.leavesBuffer);
var dataBuffer = device.createBuffer({
  label: "PicoVDB Data",
  size: picoVDBFile.dataBuffer.byteLength,
  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
});
device.queue.writeBuffer(dataBuffer, 0, picoVDBFile.dataBuffer);
var fov = 2 * Math.PI / 5;
var fovScaled = Math.tan(fov / 2);
var initialCameraPosition = vec3.create(3, 2, 5);
var cameras = {
  arcball: new ArcballCamera({ position: initialCameraPosition }),
  WASD: new WASDCamera({ position: initialCameraPosition })
};
var currentCamera = cameras[controls.cameraType];
cameraController.onChange((newCameraType) => {
  const oldCamera = currentCamera;
  const newCamera = cameras[newCameraType];
  newCamera.matrix = oldCamera.matrix;
  currentCamera = newCamera;
  controls.cameraType = newCameraType;
});
var InputValues = new ArrayBuffer(256);
var InputViews = {
  camera_matrix: new Float32Array(InputValues, 0, 16),
  fov_scale: new Float32Array(InputValues, 64, 1),
  time_delta: new Float32Array(InputValues, 68, 1),
  _pad: new Uint32Array(InputValues, 72, 2),
  transform_matrix: new Float32Array(InputValues, 80, 16),
  transform_inverse_matrix: new Float32Array(InputValues, 144, 16)
};
var inputBuffer = device.createBuffer({
  label: "Input Uniforms",
  size: InputValues.byteLength,
  usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
});
InputViews.fov_scale[0] = fovScaled;
function updateTransformMatrices() {
  const transformMatrix = mat4.identity();
  mat4.translation(vec3.create(-40, 240, 0), transformMatrix);
  mat4.scale(transformMatrix, vec3.create(120, 120, 120), transformMatrix);
  const rotationRadians = controls.bunnyRotation * Math.PI / 180;
  mat4.rotateY(transformMatrix, rotationRadians, transformMatrix);
  const transformInverseMatrix = mat4.inverse(transformMatrix);
  InputViews.transform_matrix.set(transformMatrix);
  InputViews.transform_inverse_matrix.set(transformInverseMatrix);
}
updateTransformMatrices();
rotationController.onChange(() => {
  updateTransformMatrices();
});
var sizeMB = (picoVDBFile.getSize() / 1024 / 1024).toFixed(1);
var grid = picoVDBFile.getGrid(0);
var bboxSize = [
  grid.indexBoundsMax[0] - grid.indexBoundsMin[0],
  grid.indexBoundsMax[1] - grid.indexBoundsMin[1],
  grid.indexBoundsMax[2] - grid.indexBoundsMin[2]
];
infoTextElement.textContent = `PicoVDB
bunny.pvdb ${sizeMB}MB
Grid: ${bboxSize[0]} \xD7 ${bboxSize[1]} \xD7 ${bboxSize[2]} units
Voxels: ${picoVDBFile.getVoxelCount()}`;
function updateInput(deltaTime) {
  InputViews.time_delta[0] = deltaTime;
  currentCamera.update(deltaTime, inputHandler());
  InputViews.camera_matrix.set(currentCamera.matrix);
  device.queue.writeBuffer(inputBuffer, 0, InputValues);
}
var combinedShader = picovdb_default + "\n" + compute_default;
var computeShaderModule = device.createShaderModule({
  label: "Raytracing Compute Shader",
  code: combinedShader
});
var shaderInfo = await computeShaderModule.getCompilationInfo();
if (shaderInfo.messages.length > 0) {
  console.error("Shader compilation messages:", shaderInfo.messages);
  for (const message of shaderInfo.messages) {
    console.log(`${message.type} at line ${message.lineNum}: ${message.message}`);
    if (message.type === "error") {
      alert(`Shader error at line ${message.lineNum}: ${message.message}`);
    }
  }
}
var computeBindGroupLayout = device.createBindGroupLayout({
  label: "Compute Bind Group Layout",
  entries: [
    {
      binding: 0,
      visibility: GPUShaderStage.COMPUTE,
      storageTexture: {
        access: "write-only",
        format: "rgba8unorm",
        viewDimension: "2d"
      }
    },
    {
      binding: 1,
      visibility: GPUShaderStage.COMPUTE,
      buffer: {
        type: "uniform"
      }
    },
    {
      binding: 2,
      visibility: GPUShaderStage.COMPUTE,
      buffer: {
        type: "read-only-storage"
      }
    },
    {
      binding: 3,
      visibility: GPUShaderStage.COMPUTE,
      buffer: {
        type: "read-only-storage"
      }
    },
    {
      binding: 4,
      visibility: GPUShaderStage.COMPUTE,
      buffer: {
        type: "read-only-storage"
      }
    },
    {
      binding: 5,
      visibility: GPUShaderStage.COMPUTE,
      buffer: {
        type: "read-only-storage"
      }
    },
    {
      binding: 6,
      visibility: GPUShaderStage.COMPUTE,
      buffer: {
        type: "read-only-storage"
      }
    },
    {
      binding: 7,
      visibility: GPUShaderStage.COMPUTE,
      buffer: {
        type: "read-only-storage"
      }
    }
  ]
});
var computePipelineLayout = device.createPipelineLayout({
  label: "Compute Pipeline Layout",
  bindGroupLayouts: [computeBindGroupLayout]
});
var computePipeline = await device.createComputePipelineAsync({
  label: "Raytracing Compute Pipeline",
  layout: computePipelineLayout,
  compute: {
    module: computeShaderModule,
    entryPoint: "computeMain"
  }
}).catch((error) => {
  console.error("Pipeline creation failed:", error);
  alert(`Pipeline error: ${error.message}`);
  throw error;
});
console.log("Pipeline created, getting bind group layout...");
var computePassDescriptor = {
  label: "Compute pass"
};
timestampQueryManager.addTimestampWrite(computePassDescriptor);
createGPUResources();
var colorAttachment = {
  view: context.getCurrentTexture().createView(),
  // Assigned on render 
  clearValue: { r: 0, g: 0, b: 0, a: 1 },
  loadOp: "clear",
  storeOp: "store"
};
var renderPassDescriptor = {
  label: "Display pass",
  colorAttachments: [colorAttachment]
};
var lastFrameMS = (performance || Date).now();
function requestFrame() {
  if (!context) {
    throw new Error("No context found.");
  }
  const beginTime = stats.begin();
  const deltaTime = (beginTime - lastFrameMS) / 1e3;
  lastFrameMS = beginTime;
  const encoder = device.createCommandEncoder({ label: "Command Encoder" });
  const computePass = encoder.beginComputePass(computePassDescriptor);
  computePass.setPipeline(computePipeline);
  computePass.setBindGroup(0, computeBindGroup);
  const workgroups_x = Math.ceil(width / 16);
  const workgroups_y = Math.ceil(height / 16);
  computePass.dispatchWorkgroups(workgroups_x, workgroups_y, 1);
  computePass.end();
  colorAttachment.view = context.getCurrentTexture().createView();
  const displayPass = encoder.beginRenderPass(renderPassDescriptor);
  displayPass.setPipeline(displayPipeline);
  displayPass.setVertexBuffer(0, vertexBuffer);
  displayPass.setBindGroup(0, displayBindGroup);
  displayPass.draw(3, 1, 0, 0);
  displayPass.end();
  updateInput(deltaTime);
  timestampQueryManager.resolve(encoder);
  device.queue.submit([encoder.finish()]);
  timestampQueryManager.tryInitiateTimestampDownload();
  stats.end();
}
var animationId = null;
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
pauseController.onChange((paused) => {
  if (paused) {
    stopRenderLoop();
  } else {
    startRenderLoop();
  }
});
startRenderLoop();
/*! Bundled license information:

lil-gui/dist/lil-gui.esm.js:
  (**
   * lil-gui
   * https://lil-gui.georgealways.com
   * @version 0.21.0
   * @author George Michael Brower
   * @license MIT
   *)
*/
