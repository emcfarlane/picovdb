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
  function clamp(v, min2 = 0, max2 = 1, dst) {
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
  function lerp(a, b, t, dst) {
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
  function rotate(a, b, rad, dst) {
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
    return lerp(a, b, 0.5, newDst);
  }
  return {
    create,
    fromValues,
    set,
    ceil,
    floor,
    round,
    clamp,
    add,
    addScaled,
    angle,
    subtract,
    sub,
    equalsApproximately,
    equals,
    lerp,
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
    rotate,
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
  function clamp(v, min2 = 0, max2 = 1, dst) {
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
  function lerp(a, b, t, dst) {
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
    return lerp(a, b, 0.5, newDst);
  }
  return {
    create,
    fromValues,
    set,
    ceil,
    floor,
    round,
    clamp,
    add,
    addScaled,
    angle,
    subtract,
    sub,
    equalsApproximately,
    equals,
    lerp,
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
  function rotate(m, angleInRadians, dst) {
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
  const rotateZ = rotate;
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
    rotate,
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
  const rotate = axisRotate;
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
    rotate,
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
  function lerp(a, b, t, dst) {
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
    lerp,
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
  function clamp(v, min2 = 0, max2 = 1, dst) {
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
  function lerp(a, b, t, dst) {
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
    return lerp(a, b, 0.5, newDst);
  }
  return {
    create,
    fromValues,
    set,
    ceil,
    floor,
    round,
    clamp,
    add,
    addScaled,
    subtract,
    sub,
    equalsApproximately,
    equals,
    lerp,
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
var blit_default = "struct VertexInput {\n    @location(0) position: vec2f,\n}\n\nstruct VertexOutput {\n    @builtin(position) pos: vec4f,\n    @location(0) uv: vec2f,\n}\n\nstruct FragmentInput {\n    @location(0) uv: vec2f,\n}\n\n@group(0) @binding(0) var raytracedTexture: texture_2d<f32>;\n@group(0) @binding(1) var textureSampler: sampler;\n\n@vertex\nfn vertexMain(input: VertexInput) -> VertexOutput {\n    // Convert from [-1,1] to [0,1] for UV coordinates\n    let uv = input.position * 0.5 + 0.5;\n    return VertexOutput(vec4f(input.position, 0.0, 1.0), uv);\n}\n\n@fragment\nfn fragmentMain(\n    input: FragmentInput,\n) -> @location(0) vec4f {\n    let color = textureSample(raytracedTexture, textureSampler, input.uv);\n    return color;\n}\n";

// compute.wgsl
var compute_default = "struct Input {\n    camera_matrix: mat4x4f,\n    fov_scale: f32, // tan(fov * 0.5)\n    time_delta: f32,\n    pixel_radius: f32, // Cone spread per unit distance: 1 / (resolution.y * focal_length)\n    debug_iterations: u32, // 0 = normal rendering, 1 = debug iteration heatmap\n}\n\n// --- Object types ---\nconst OBJECT_TYPE_UNKNOWN: u32 = 0u;\nconst OBJECT_TYPE_VDB: u32 = 1u;\nconst OBJECT_TYPE_SDF: u32 = 2u;\n\nstruct Object { // 144\n    object_type: u32,\n    type_index: u32,\n    material_index: u32,\n    _pad: u32,\n    transform: mat4x4f,\n    transform_inverse: mat4x4f,\n}\n\nstruct Material { // 32\n    color: vec3f,\n    albedo: f32,\n    metallic: f32,\n    roughness: f32,\n    _pad: array<f32, 2>,\n}\n\n// --- Bind group 0: per-frame ---\n@group(0) @binding(0) var<uniform> input: Input;\n@group(0) @binding(1) var<storage> objects: array<Object>;\n@group(0) @binding(2) var<storage, read> skyState: SkyState;\n\n// -- Bind group 1: data ---\n@group(1) @binding(0) var<storage> picovdb_grids: array<PicoVDBGrid>;\n@group(1) @binding(1) var<storage> picovdb_roots: array<PicoVDBRoot>;\n@group(1) @binding(2) var<storage> picovdb_uppers: array<PicoVDBUpper>;\n@group(1) @binding(3) var<storage> picovdb_lowers: array<PicoVDBLower>;\n@group(1) @binding(4) var<storage> picovdb_leaves: array<PicoVDBLeaf>;\n@group(1) @binding(5) var<storage> picovdb_buffer: array<u32>;\n\n// --- Bind group 2: pass ---\n@group(2) @binding(0) var output_texture: texture_storage_2d<rgba8unorm, write>;\n\nconst MAX_DIST: f32 = 1e7;\n\nstruct Intersection {\n    distance: f32,\n    object_index: i32,\n    iterations: u32,\n    normal: vec3f,\n}\n\nfn no_intersection() -> Intersection {\n    return Intersection(MAX_DIST, -1, 0, vec3f(0));\n}\n\nstruct Ray {\n    origin: vec3f,\n    direction: vec3f,\n}\n\nfn intersect_picovdb(\n    ray: Ray,\n    grid_index: u32,\n    hit_distance: ptr<function, f32>,\n    hit_normal: ptr<function, vec3f>,\n    hit_iterations: ptr<function, u32>,\n) -> bool {\n    let tmin = 0.0;\n    let tmax = 10000.0;\n\n    let grid = picovdb_grids[grid_index];\n    var accessor: PicoVDBReadAccessor;\n    picovdbReadAccessorInit(&accessor, grid_index);\n\n    // Inside Check (Works even if camera is in background space)\n    let start_val = picovdbSampleTrilinear(&accessor, grid, ray.origin);\n    if start_val < 0.0 {\n        *hit_distance = tmin;\n        *hit_normal = -ray.direction;\n        return true;\n    }\n\n    return picovdbHDDAZeroCrossing(\n        &accessor, grid, ray.origin, tmin, ray.direction, tmax, input.pixel_radius, hit_distance, hit_normal, hit_iterations,\n    );\n}\n\nfn intersect_sdf(\n    ray: Ray,\n    index: u32,\n    hit_distance: ptr<function, f32>,\n    hit_normal: ptr<function, vec3f>,\n    iterations: ptr<function, u32>,\n) -> bool {\n    switch index {\n        case 0u: { // ground plane at y=0 in index space\n            if ray.direction.y >= 0.0 || abs(ray.direction.y) < 0.001 {\n                return false;\n            }\n            let t = -ray.origin.y / ray.direction.y;\n            if t < 0.001 {\n                return false;\n            }\n            *hit_distance = t;\n            *hit_normal = vec3f(0, 1, 0);\n            return true;\n        }\n        case default: { return false; }\n    }\n}\n\nfn intersect_scene(world_ray: Ray, iterations: ptr<function, u32>) -> Intersection {\n    var min_hit = no_intersection();\n    for (var i = 0i; i < i32(arrayLength(&objects)); i++) {\n        let obj = objects[i];\n        let idx_origin = (obj.transform * vec4f(world_ray.origin, 1.0)).xyz;\n        let idx_dir_unnorm = (obj.transform * vec4f(world_ray.direction, 0.0)).xyz;\n        let idx_direction = normalize(idx_dir_unnorm);\n        let index_ray = Ray(idx_origin, idx_direction);\n\n        var hit = false;\n        var hit_distance = MAX_DIST;\n        var hit_normal = vec3f(0);\n        var hit_iterations = 0u;\n        switch obj.object_type {\n            case OBJECT_TYPE_VDB: {\n                hit = intersect_picovdb(index_ray, obj.type_index, &hit_distance, &hit_normal, &hit_iterations);\n            }\n            case OBJECT_TYPE_SDF: {\n                hit = intersect_sdf(index_ray, obj.type_index, &hit_distance, &hit_normal, &hit_iterations);\n            }\n            case default: { \n                hit = false;\n            }\n        }\n        *iterations += hit_iterations;\n        if !hit {\n            continue;\n        }\n        let index_hit_point = index_ray.origin + index_ray.direction * hit_distance;\n        let world_hit_point = (obj.transform_inverse * vec4f(index_hit_point, 1.0)).xyz;\n        let world_distance = length(world_hit_point - world_ray.origin);\n        if world_distance >= min_hit.distance {\n            continue;\n        }\n\n        min_hit.distance = world_distance;\n        min_hit.object_index = i;\n        min_hit.normal = (obj.transform_inverse * vec4f(hit_normal, 0.0)).xyz;\n    }\n    min_hit.normal = normalize(min_hit.normal);\n    return min_hit;\n}\n\nfn generate_camera_ray(screen_coord: vec2f, screen_size: vec2f) -> Ray {\n    // Convert to normalized coordinates [-1, 1k\n    let uv = (screen_coord / screen_size) * 2.0 - 1.0;\n\n    // Calculate aspect ratio\n    let aspect_ratio = screen_size.x / screen_size.y;\n\n    // Extract camera basis vectors from view matrix\n    let right: vec3f = input.camera_matrix[0].xyz;\n    let up: vec3f = input.camera_matrix[1].xyz;\n    let forward: vec3f = -input.camera_matrix[2].xyz;\n\n    // Extract camera position\n    let camera_pos: vec3f = input.camera_matrix[3].xyz;\n\n    // Calculate ray direction\n    let ray_direction = normalize(\n        forward + uv.x * right * aspect_ratio * input.fov_scale + uv.y * up * input.fov_scale\n    );\n    return Ray(camera_pos, ray_direction);\n}\n\nfn get_material(hit: Intersection, obj: Object) -> Material {\n    switch obj.material_index {\n        case 0u: {\n            return Material(vec3f(0.0, 0.1, 1.0), 0.0, 0.0, 0.1, array(0,0));\n        }\n        case 1u: {\n            return Material(vec3f(0.2, 0.2, 0.2), 1.0, 1.0, 1.0, array(0,0));\n        }\n        default: {\n            return Material(vec3f(0.0, 0.0, 0.0), 0, 0, 0, array(0,0));\n        }\n    }\n}\n\nfn traceShadowRay(origin: vec3f, normal: vec3f) -> f32 {\n    // Offset origin slightly along normal to avoid self-intersection\n    let shadowOrigin = origin + normal * 0.01;\n    let shadowRay = Ray(shadowOrigin, skyState.sunDirection);\n    var iterations: u32;\n    let hit = intersect_scene(shadowRay, &iterations);\n    if hit.object_index >= 0 {\n        return 0.0;  // Fully shadowed\n    }\n    return 1.0;  // Fully lit\n}\n\nfn applyFog(color: vec3f, distance: f32, rayDir: vec3f, fogDensity: f32) -> vec3f {\n    let fogAmount = 1.0 - exp(-distance * fogDensity);\n    // Blend between sky color and slight blue haze\n    let skyFog = skyRadianceRGB(rayDir, false);\n    let hazeTint = vec3f(0.7, 0.8, 1.0);  // Subtle blue\n    let fogColor = mix(skyFog, skyFog * hazeTint, 0.3);\n    return mix(color, fogColor, fogAmount);\n}\n\nfn computeColor(ray: Ray, hit: Intersection) -> vec3f {\n    if hit.object_index < 0 {\n        return skyRadianceRGB(ray.direction, true);\n    }\n\n    let obj = objects[hit.object_index];\n    let material = get_material(hit, obj);\n    let hitPoint = ray.origin + ray.direction * hit.distance;\n\n    let albedo = material.color;\n    let metallic = material.metallic;\n    let roughness = max(material.roughness, 0.04);  // Clamp to avoid division issues\n    let ao = 1.0;  // Could come from material or SSAO later\n\n    let n = normalize(hit.normal);\n    let v = normalize(-ray.direction);\n    let l = skyState.sunDirection;\n    let h = normalize(v + l);\n    let r = reflect(-v, n);\n\n    let f0 = mix(vec3f(0.04), albedo, metallic);\n\n    // Direct sun lighting\n    let nDotL = max(dot(n, l), 0.0);\n    var lo = vec3f(0.0);\n\n    if nDotL > 0.0 {\n        let shadow = traceShadowRay(hitPoint, n);\n        if shadow > 0.0 {\n            let sunRadiance = sunIrradiance();\n\n            let d = distributionGGX(n, h, roughness);\n            let g = geometrySmith(n, v, l, roughness);\n            let f = fresnelSchlick(max(dot(h, v), 0.0), f0);\n\n            let numerator = d * g * f;\n            let denominator = 4.0 * max(dot(n, v), 0.0) * nDotL + 0.0001;\n            let specular = numerator / denominator;\n\n            let kS = f;\n            var kD = vec3f(1.0) - kS;\n            kD *= 1.0 - metallic;\n\n            lo = (kD * albedo / PI + specular) * sunRadiance * nDotL;\n        }\n    }\n\n    // Ambient / environment lighting\n    let f = fresnelSchlickRoughness(max(dot(n, v), 0.0001), f0, roughness);\n    let kS = f;\n    var kD = vec3f(1.0) - kS;\n    kD *= 1.0 - metallic;\n\n    // Diffuse irradiance from sky hemisphere\n    let irradiance = skyIrradiance(n);\n    let diffuse = irradiance * albedo / PI;\n\n    // Specular reflection, sample sky in reflection direction.\n    // For rough surfaces, we'd ideally blur/filter, but single sample works ok\n    let prefilteredColor = skyRadianceRGB(r, true);\n    \n    // Approximate the BRDF integration (simplified - no LUT)\n    // This is a rough approximation of the split-sum BRDF\n    let nDotV = max(dot(n, v), 0.0);\n    let envBRDF = vec2f(\n        1.0 - roughness * 0.5,  // Approximate F scale\n        roughness * 0.5         // Approximate F bias\n    );\n    let specular = prefilteredColor * (f * envBRDF.x + envBRDF.y);\n\n    let ambient = (kD * diffuse + specular) * ao;\n\n    var color = ambient + lo;\n    if hit.distance > 10 {\n        color = applyFog(color, hit.distance-10, ray.direction, 0.01);\n    }\n    return color;\n}\n\n// toneMapping implements ACES\nfn toneMapping(color: vec3f) -> vec3f {\n    let exposure = 0.05; // Tuneable\n    let exposed = color * exposure;\n    let a = 2.51;\n    let b = 0.03;\n    let c = 2.43;\n    let d = 0.59;\n    let e = 0.14;\n    return (exposed * (a * exposed + b)) / (exposed * (c * exposed + d) + e);\n}\n\n@compute @workgroup_size(8, 8)\nfn computeMain(@builtin(global_invocation_id) global_id: vec3u) {\n    let dims = textureDimensions(output_texture);\n    if global_id.x >= dims.x || global_id.y >= dims.y { return; }\n\n    let ray = generate_camera_ray(vec2f(global_id.xy) + 0.5, vec2f(dims));\n    var iterations: u32;\n    let hit = intersect_scene(ray, &iterations);\n\n    var color = computeColor(ray, hit);\n    color = toneMapping(color);\n    color = pow(color, vec3f(1.0 / 2.2));  // Gamma correction\n\n    if input.debug_iterations == 1u {\n        let heat = clamp(f32(iterations) / 128.0, 0.0, 1.0);\n        color = vec3f(0.0, heat, 0.0);\n    }\n    textureStore(output_texture, global_id.xy, vec4f(color, 1.0));\n}\n\n// ============================================================================\n// PBR\n// ============================================================================\n\nconst PI = 3.14159265359;\n\nfn distributionGGX(n: vec3f, h: vec3f, roughness: f32) -> f32 {\n    let a = roughness * roughness;\n    let a2 = a * a;\n    let nDotH = max(dot(n, h), 0.0);\n    let nDotH2 = nDotH * nDotH;\n    let num = a2;\n    let denom = nDotH2 * (a2 - 1.0) + 1.0;\n    return a2 / (PI * denom * denom);\n}\n\nfn geometrySchlickGGX(nDotV: f32, roughness: f32) -> f32 {\n    let r = roughness + 1.0;\n    let k = (r * r) / 8.0;\n    return nDotV / (nDotV * (1.0 - k) + k);\n}\n\nfn geometrySmith(n: vec3f, v: vec3f, l: vec3f, roughness: f32) -> f32 {\n    let nDotV = max(dot(n, v), 0.0);\n    let nDotL = max(dot(n, l), 0.0);\n    let ggx2 = geometrySchlickGGX(nDotV, roughness);\n    let ggx1 = geometrySchlickGGX(nDotL, roughness);\n    return ggx1 * ggx2;\n}\n\nfn fresnelSchlick(cosTheta: f32, f0: vec3f) -> vec3f {\n  return f0 + (1.0 - f0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);\n}\n\nfn fresnelSchlickRoughness(cosTheta: f32, f0: vec3f, roughness: f32) -> vec3f {\n  return f0 + (max(vec3(1.0 - roughness), f0) - f0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);\n}\n\n// http://holger.dammertz.org/stuff/notes_HammersleyOnHemisphere.html\n// efficient VanDerCorpus calculation.\nfn radicalInverseVdC(bits: u32) -> f32 {\n  var result = bits;\n  result = (bits << 16u) | (bits >> 16u);\n  result = ((result & 0x55555555u) << 1u) | ((result & 0xAAAAAAAAu) >> 1u);\n  result = ((result & 0x33333333u) << 2u) | ((result & 0xCCCCCCCCu) >> 2u);\n  result = ((result & 0x0F0F0F0Fu) << 4u) | ((result & 0xF0F0F0F0u) >> 4u);\n  result = ((result & 0x00FF00FFu) << 8u) | ((result & 0xFF00FF00u) >> 8u);\n  return f32(result) * 2.3283064365386963e-10;\n}\n\nfn hammersley(i: u32, n: u32) -> vec2f {\n  return vec2f(f32(i) / f32(n), radicalInverseVdC(i));\n}\n\nfn importanceSampleGGX(xi: vec2f, n: vec3f, roughness: f32) -> vec3f {\n  let a = roughness * roughness;\n\n  let phi = 2.0 * PI * xi.x;\n  let cosTheta = sqrt((1.0 - xi.y) / (1.0 + (a * a - 1.0) * xi.y));\n  let sinTheta = sqrt(1.0 - cosTheta * cosTheta);\n\n  // from spherical coordinates to cartesian coordinates - halfway vector\n  let h = vec3f(cos(phi) * sinTheta, sin(phi) * sinTheta, cosTheta);\n\n  // from tangent-space H vector to world-space sample vector\n  let up: vec3f = select(vec3f(1.0, 0.0, 0.0), vec3f(0.0, 0.0, 1.0), abs(n.z) < 0.999);\n  let tangent = normalize(cross(up, n));\n  let bitangent = cross(n, tangent);\n\n  let sampleVec = tangent * h.x + bitangent * h.y + n * h.z;\n  return normalize(sampleVec);\n}\n\n\n// ============================================================================\n// Sky Model\n// ============================================================================\nconst CHANNEL_R = 0u;\nconst CHANNEL_G = 1u;\nconst CHANNEL_B = 2u;\nconst SOLAR_RADIUS_RADIANS = 0.004450589; // 0.255 degrees\n\nstruct SkyState {\n    sunDirection: vec3<f32>,\n    params: array<f32, 27>,\n    skyRadiances: array<f32, 3>,\n    solarRadiances: array<f32, 3>,\n}\n\nfn radiance(theta: f32, gamma: f32, channel: u32, includeSun: bool) -> f32 {\n    let r = skyState.skyRadiances[channel];\n    let idx = 9u * channel;\n    let p0 = skyState.params[idx + 0u];\n    let p1 = skyState.params[idx + 1u];\n    let p2 = skyState.params[idx + 2u];\n    let p3 = skyState.params[idx + 3u];\n    let p4 = skyState.params[idx + 4u];\n    let p5 = skyState.params[idx + 5u];\n    let p6 = skyState.params[idx + 6u];\n    let p7 = skyState.params[idx + 7u];\n    let p8 = skyState.params[idx + 8u];\n\n    let cosGamma = cos(gamma);\n    let cosGamma2 = cosGamma * cosGamma;\n    let cosTheta = abs(cos(theta));\n\n    let expM = exp(p4 * gamma);\n    let rayM = cosGamma2;\n    let mieMLhs = 1.0 + cosGamma2;\n    let mieMRhs = pow(1.0 + p8 * p8 - 2.0 * p8 * cosGamma, 1.5f);\n    let mieM = mieMLhs / mieMRhs;\n    let zenith = sqrt(cosTheta);\n    let radianceLhs = 1.0 + p0 * exp(p1 / (cosTheta + 0.01));\n    let radianceRhs = p2 + p3 * expM + p5 * rayM + p6 * mieM + p7 * zenith;\n    let radianceDist = radianceLhs * radianceRhs;\n\n    let solarDiskRadius = gamma / SOLAR_RADIUS_RADIANS;\n    let solarRadiance = select(0.0, skyState.solarRadiances[channel], includeSun && solarDiskRadius <= 1.0);\n\n    return r * radianceDist + solarRadiance;\n}\n\nfn skyRadianceRGB(direction: vec3f, includeSun: bool) -> vec3f {\n    let v = normalize(direction);\n    let s = skyState.sunDirection;\n    let theta = acos(clamp(v.y, -1.0, 1.0));\n    let gamma = acos(clamp(dot(v, s), -1.0, 1.0));\n    return vec3f(\n        radiance(theta, gamma, CHANNEL_R, includeSun),\n        radiance(theta, gamma, CHANNEL_G, includeSun),\n        radiance(theta, gamma, CHANNEL_B, includeSun)\n    );\n}\n\nfn sunIrradiance() -> vec3f {\n    // Solar radiance * solid angle of sun disk\n    let sunSolidAngle = PI * SOLAR_RADIUS_RADIANS * SOLAR_RADIUS_RADIANS;\n    return vec3f(\n        skyState.solarRadiances[0],\n        skyState.solarRadiances[1],\n        skyState.solarRadiances[2]\n    ) * sunSolidAngle;\n}\n\nfn skyIrradiance(n: vec3f) -> vec3f {\n    var irradiance = vec3f(0.0);\n    let SAMPLE_COUNT = 16u;\n    \n    for (var i = 0u; i < SAMPLE_COUNT; i++) {\n        let xi = hammersley(i, SAMPLE_COUNT);\n        \n        // Cosine-weighted hemisphere sampling\n        let phi = 2.0 * PI * xi.x;\n        let cosTheta = sqrt(1.0 - xi.y);  // Cosine-weighted\n        let sinTheta = sqrt(xi.y);\n\n        // To world space\n        let up = select(vec3f(1.0, 0.0, 0.0), vec3f(0.0, 0.0, 1.0), abs(n.z) < 0.999);\n        let tangent = normalize(cross(up, n));\n        let bitangent = cross(n, tangent);\n\n        let sampleDir = normalize(\n            tangent * cos(phi) * sinTheta +\n            bitangent * sin(phi) * sinTheta +\n            n * cosTheta\n        );\n        irradiance += skyRadianceRGB(sampleDir, false);\n    }\n    return irradiance * PI / f32(SAMPLE_COUNT);\n}\n";

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
    let count = select(u32(is_inside), mask.valueOffset + countOneBits(preceding_bits), is_value);
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
    if (!is_value) {
        return PicoVDBLevelCount(1u, u32(is_inside)); // fast path
    }
    if (is_inside) {
        let preceding_bits = extractBits(mask.value & mask.inside, 0u, bit_index);
        (*acc).leaf = mask.childOffset + countOneBits(preceding_bits);
        (*acc).key = ijk;
        return picovdbReadAccessorLeafGetLevelCountAndCache(acc, ijk, grid);
    }
    let preceding_bits = extractBits(mask.value & ~mask.inside, 0u, bit_index);
    let count = mask.valueOffset + countOneBits(preceding_bits);
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

    if (!is_value) {
        return PicoVDBLevelCount(2u, u32(is_inside)); // fast path
    }
    if (is_inside) {
        let preceding_bits = extractBits(mask.value & mask.inside, 0u, bit_index);
        (*acc).lower = mask.childOffset + countOneBits(preceding_bits);
        (*acc).key = ijk;
        return picovdbReadAccessorLowerGetLevelCountAndCache(acc, ijk, grid);
    }
    let preceding_bits = extractBits(mask.value & ~mask.inside, 0u, bit_index);
    let count = mask.valueOffset + countOneBits(preceding_bits);
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
    let mask = vec3i(~(dim - 1));
    let vox = vec3i(floor(pos)) & mask;

    (*hdda).dim = dim;
    (*hdda).tmin = tmin;
    (*hdda).tmax = tmax;
    (*hdda).voxel = vox;
    (*hdda).step = vec3i(sign(direction));
    (*hdda).delta = abs(f32(dim) * direction_inv); // Pre-multiply delta by dim

    let base = (*hdda).tmin + (vec3f(vox) - pos) * direction_inv;
    let pos_offset = base + (*hdda).delta;
    (*hdda).next = select(
        select(base, pos_offset, (*hdda).step > vec3i(0)),
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
    let mask = vec3i(~(dim - 1));
    let voxel_min = (*hdda).voxel & mask;
    let voxel_max = ((*hdda).voxel + vec3i((*hdda).dim - 1)) & mask;

    (*hdda).dim = dim;
    (*hdda).delta = abs(f32(dim) * direction_inv);

    let pos = origin + direction * (*hdda).tmin;
    let vox = clamp(vec3i(floor(pos)) & mask, voxel_min, voxel_max);
    (*hdda).voxel = vox;

    let base = (*hdda).tmin + (vec3f(vox) - pos) * direction_inv;
    let pos_offset = base + (*hdda).delta;
    (*hdda).next = select(
        select(base, pos_offset, (*hdda).step > vec3i(0)),
        vec3f(PICOVDB_HDDA_FLOAT_MAX),
        direction == vec3f(0.0)
    );
}

fn picovdbHDDAStep(hdda: ptr<function, PicoVDBHDDA>) -> bool {
    // Determine which axis has the nearest boundary
    let next = (*hdda).next;
    if (next.x <= next.y && next.x <= next.z) { // X is smallest
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

// Zero-crossing detection for level set raymarching.
fn picovdbHDDAZeroCrossing(
    acc: ptr<function, PicoVDBReadAccessor>,
    grid: PicoVDBGrid,
    origin: vec3f,
    tmin: f32,
    direction: vec3f,
    tmax: f32,
    pixel_radius: f32,
    out_distance: ptr<function, f32>,
    out_normal: ptr<function, vec3f>,
    out_iterations: ptr<function, u32>,
) -> bool {
    let direction_inv = 1 / direction;
    var tmin_mut = tmin;
    var tmax_mut = tmax;
    if (!picovdbHDDARayClip(vec3f(grid.indexBoundsMin), vec3f(grid.indexBoundsMax + vec3i(1)), origin, &tmin_mut, direction_inv, &tmax_mut)) {
        *out_iterations = 0u;
        return false;
    }

    // Get initial hierarchy level
    let start_pos = origin + direction * tmin_mut;
    let res0 = picovdbReadAccessorGetLevelCount(acc, vec3i(floor(start_pos)), grid);
    let v0 = picovdbGetValue(grid, res0.count);

    var hdda: PicoVDBHDDA;
    picovdbHDDAInit(&hdda, origin, tmin_mut, direction, tmax_mut, direction_inv, picovdbDimForLevel[res0.level]);

    var step_count = 0u;
    for (var i = 0; i < 512; i++) { // Fixed loop limit for GPU safety
        step_count += 1u;
        let result = picovdbReadAccessorGetLevelCount(acc, hdda.voxel, grid);
        let target_dim = picovdbDimForLevel[result.level];

        // If hierarchy changed, update HDDA and re-read
        if (hdda.dim != target_dim) {
            picovdbHDDAUpdate(&hdda, origin, target_dim, direction, direction_inv);
            continue; // Re-evaluate with the new aligned voxel
        }

        if (hdda.dim == 1 && picovdbIsActive(result)) {
            let val = picovdbGetValue(grid, result.count);
            if ((val <= 0.0) != (v0 <= 0.0)) {
                let cone_radius = hdda.tmin * pixel_radius;
                if (cone_radius < 0.5) {
                    // Voxel projects larger than a pixel \u2014 use analytical cubic solver
                    // for smooth, sub-voxel accurate intersection.
                    let stencil = picovdbSampleStencil(acc, grid, hdda.voxel);
                    let o_local = origin + direction * hdda.tmin - vec3f(hdda.voxel);
                    let t_exit = min(min(hdda.next.x, hdda.next.y), hdda.next.z) - hdda.tmin;
                    let hit = picovdbVoxelIntersect(o_local, direction, t_exit, stencil);
                    if (hit.hit) {
                        *out_distance = hdda.tmin + hit.t;
                        *out_normal = hit.normal;
                        *out_iterations = step_count;
                        return true;
                    }
                } else {
                    let stencil = picovdbSampleStencil(acc, grid, hdda.voxel);
                    let p_local = fract(origin + direction * hdda.tmin);
                    *out_distance = hdda.tmin;
                    *out_normal = picovdbTrilinearGradient(p_local, stencil);
                    *out_iterations = step_count;
                    return true;
                }
            }
        }
        // Step to next boundary
        if (!picovdbHDDAStep(&hdda)) {
            break;
        }
    }
    *out_iterations = step_count;
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
    s.v100 = picovdbGetValue(grid, picovdbReadAccessorGetLevelCount(acc, ijk + vec3i(1, 0, 0), grid).count);
    s.v010 = picovdbGetValue(grid, picovdbReadAccessorGetLevelCount(acc, ijk + vec3i(0, 1, 0), grid).count);
    s.v110 = picovdbGetValue(grid, picovdbReadAccessorGetLevelCount(acc, ijk + vec3i(1, 1, 0), grid).count);
    s.v001 = picovdbGetValue(grid, picovdbReadAccessorGetLevelCount(acc, ijk + vec3i(0, 0, 1), grid).count);
    s.v101 = picovdbGetValue(grid, picovdbReadAccessorGetLevelCount(acc, ijk + vec3i(1, 0, 1), grid).count);
    s.v011 = picovdbGetValue(grid, picovdbReadAccessorGetLevelCount(acc, ijk + vec3i(0, 1, 1), grid).count);
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

// ============================================================================
// Analytical Ray\u2013Voxel Intersection for Trilinearly Interpolated SDF Grids
//
// Based on: Hansson-S\xF6derlund, Evans, Akenine-M\xF6ller,
//   "Ray Tracing of Signed Distance Function Grids", JCGT 2022
//   https://jcgt.org/published/0011/03/06/
//
// Given the 2x2x2 SDF stencil at a voxel's corners, trilinear interpolation
// defines a cubic implicit surface f(x,y,z) = 0. Substituting the ray
// parametrically yields a cubic polynomial in t: c3*t^3 + c2*t^2 + c1*t + c0 = 0.
// We use Marmitt's interval splitting (via the derivative roots) to isolate
// monotonic sub-intervals, then Newton-Raphson to refine the root.
// ============================================================================

struct PicoVDBVoxelHit {
    hit:    bool,
    t:      f32,     // parametric distance in voxel-local space
    uvw:    vec3f,   // hit position in voxel-local [0,1]^3
    normal: vec3f,   // analytic surface normal
}

// Newton-Raphson refinement within a monotonic interval.
// 3 fixed iterations with regula falsi seed \u2014 no convergence branch.
fn picovdbSolveNewton(
    c: vec4f,
    t_start: f32, t_end: f32,
    g_start: f32, g_end: f32,
    o: vec3f, d: vec3f,
    stencil: PicoVDBStencil,
) -> PicoVDBVoxelHit {
    // Regula falsi initial guess
    var t = (g_end * t_start - g_start * t_end) / (g_end - g_start);

    // 3 NR iterations \u2014 quadratic convergence from a good initial guess
    // means ~12 digits of precision, well beyond f32's ~7.
    for (var i = 0; i < 3; i++) {
        let gt  = ((c.w * t + c.z) * t + c.y) * t + c.x;
        let gdt = (3.0 * c.w * t + 2.0 * c.z) * t + c.y;
        // Guard: if derivative is near zero (tangential graze), stop.
        // Without this, t can fly to infinity and corrupt the result.
        if (abs(gdt) < 1e-10) { break; }
        t -= gt / gdt;
    }

    t = clamp(t, t_start, t_end);
    let uvw = o + t * d;
    return PicoVDBVoxelHit(
        true,
        t,
        uvw,
        picovdbTrilinearGradient(uvw, stencil),  // unnormalized
    );
}

fn picovdbVoxelIntersect(
    o:       vec3f,
    d:       vec3f,
    t_far:   f32,
    stencil: PicoVDBStencil,
) -> PicoVDBVoxelHit {
    var result: PicoVDBVoxelHit;
    result.hit = false;

    // --- k-coefficients (Equation 3) ---
    let k0 = stencil.v000;
    let k1 = stencil.v100 - stencil.v000;
    let k2 = stencil.v010 - stencil.v000;
    let a  = stencil.v101 - stencil.v001;
    let k3 = stencil.v110 - stencil.v010 - k1;
    let k4 = k0 - stencil.v001;
    let k5 = k1 - a;
    let k6 = k2 - (stencil.v011 - stencil.v001);
    let k7 = k3 - (stencil.v111 - stencil.v011 - a);

    // --- m-intermediates (Equation 7) ---
    let m0 = o.x * o.y;
    let m1 = d.x * d.y;
    let m2 = o.x * d.y + o.y * d.x;
    let m3 = k5 * o.z - k1;
    let m4 = k6 * o.z - k2;
    let m5 = k7 * o.z - k3;

    // --- Cubic coefficients c3*t^3 + c2*t^2 + c1*t + c0 = 0 (Equation 6) ---
    // Packed as vec4f(c0, c1, c2, c3).
    // c.x == trilinear value at ray origin (t=0), proven algebraically.
    let c = vec4f(
        (k4 * o.z - k0) + o.x * m3 + o.y * m4 + m0 * m5,
        d.x * m3 + d.y * m4 + m2 * m5 + d.z * (k4 + k5 * o.x + k6 * o.y + k7 * m0),
        m1 * m5 + d.z * (k5 * d.x + k6 * d.y + k7 * m2),
        k7 * m1 * d.z,
    );

    // --- Solid voxel test (Section 2) ---
    // NOTE: c.x = -f(o) due to Equation 2's sign convention:
    //   f = z*(k4+...) - (k0+...), so c0 = -k0 - k1*ox - ... + oz*(k4+...)
    // which equals -f(ox,oy,oz). Therefore c.x > 0 means f(o) < 0 (inside).
    if (c.x > 0.0) {
        return PicoVDBVoxelHit(
            true, 0.0, o,
            picovdbTrilinearGradient(o, stencil),
        );
    }

    // --- Derivative roots for Marmitt interval splitting ---
    // g'(t) = 3*c3*t^2 + 2*c2*t + c1. Roots split [0, t_far] into
    // monotonic sub-intervals. Solved inline, no function call overhead.
    let qA = 3.0 * c.w;
    let qB = 2.0 * c.z;
    let qC = c.y;

    // Default: roots outside range (effectively ignored in interval checks)
    var r0 = -1.0;
    var r1 = -1.0;

    if (abs(qA) > 1e-8) {
        let disc = qB * qB - 4.0 * qA * qC;
        if (disc >= 0.0) {
            let inv2A = 0.5 / qA;
            let sqrtDisc = sqrt(disc);
            r0 = (-qB - sqrtDisc) * inv2A;
            r1 = (-qB + sqrtDisc) * inv2A;
        }
    } else if (abs(qB) > 1e-8) {
        r0 = -qC / qB;
    }

    // --- Unrolled interval checking ---
    // Up to 3 intervals: [0, r0], [r0, r1], [last_boundary, t_far]
    // Walk front-to-back, return at first sign change.
    var t_start = 0.0;
    var g_start = c.x;  // Already computed, reuse

    // Interval 1: [0, r0]
    if (r0 > 0.0 && r0 < t_far) {
        let g_r0 = ((c.w * r0 + c.z) * r0 + c.y) * r0 + c.x;
        if (g_start * g_r0 <= 0.0) {
            return picovdbSolveNewton(c, t_start, r0, g_start, g_r0, o, d, stencil);
        }
        t_start = r0;
        g_start = g_r0;
    }

    // Interval 2: [r0, r1]
    if (r1 > t_start && r1 < t_far) {
        let g_r1 = ((c.w * r1 + c.z) * r1 + c.y) * r1 + c.x;
        if (g_start * g_r1 <= 0.0) {
            return picovdbSolveNewton(c, t_start, r1, g_start, g_r1, o, d, stencil);
        }
        t_start = r1;
        g_start = g_r1;
    }

    // Interval 3: [last_boundary, t_far]
    let g_far = ((c.w * t_far + c.z) * t_far + c.y) * t_far + c.x;
    if ((g_start <= 0.0) != (g_far <= 0.0)) {
        return picovdbSolveNewton(c, t_start, t_far, g_start, g_far, o, d, stencil);
    }

    return result;
}
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
  // Slices (as Uint8Arrays for WebGPU - explicitly typed for ArrayBuffer, not SharedArrayBuffer)
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
function createOrbitCamera(options) {
  const matrix_ = new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);
  const view_ = mat4.create();
  const right_ = new Float32Array(matrix_.buffer, 0, 4);
  const up_ = new Float32Array(matrix_.buffer, 16, 4);
  const position_ = new Float32Array(matrix_.buffer, 48, 4);
  const pivot = options?.target ? vec3.clone(options.target) : vec3.create();
  let theta = 0, phi = 0, radius = 5;
  const targetPivot = vec3.clone(pivot);
  let targetTheta = 0, targetPhi = 0, targetRadius = 5;
  const smoothing = 0.15;
  const temp = vec3.create();
  const upWorld = vec3.create(0, 1, 0);
  if (options?.position) {
    vec3.sub(options.position, pivot, temp);
    radius = vec3.len(temp);
    if (radius > 1e-4) {
      theta = Math.atan2(temp[0], temp[2]);
      phi = Math.asin(temp[1] / radius);
    }
    targetTheta = theta;
    targetPhi = phi;
    targetRadius = radius;
  }
  recalc();
  function recalc() {
    const cy = Math.cos(phi);
    vec3.set(
      radius * cy * Math.sin(theta),
      radius * Math.sin(phi),
      radius * cy * Math.cos(theta),
      temp
    );
    vec3.add(pivot, temp, position_);
    mat4.lookAt(position_, pivot, upWorld, view_);
    mat4.invert(view_, matrix_);
  }
  return {
    get matrix() {
      return matrix_;
    },
    get view() {
      return view_;
    },
    get position() {
      return position_;
    },
    get pivot() {
      return pivot;
    },
    update(dt, input) {
      const { x: dx, y: dy, zoom: dz, panning } = input.analog;
      if (panning && (dx || dy)) {
        const speed = targetRadius * 2e-3;
        vec3.addScaled(targetPivot, right_, -dx * speed, targetPivot);
        vec3.addScaled(targetPivot, up_, dy * speed, targetPivot);
      } else if (dx || dy) {
        const orbitSpeed = 5e-3;
        targetTheta -= dx * orbitSpeed;
        targetPhi = Math.max(-1.5, Math.min(1.5, targetPhi + dy * orbitSpeed));
      }
      if (dz) {
        targetRadius *= Math.pow(1.1, dz * 0.5);
        targetRadius = Math.max(0.1, targetRadius);
      }
      const t = 1 - Math.pow(smoothing, dt * 60);
      const epsilon = 1e-6;
      let dirty = false;
      if (Math.abs(targetTheta - theta) > epsilon) {
        theta += (targetTheta - theta) * t;
        dirty = true;
      }
      if (Math.abs(targetPhi - phi) > epsilon) {
        phi += (targetPhi - phi) * t;
        dirty = true;
      }
      if (Math.abs(targetRadius - radius) > epsilon) {
        radius += (targetRadius - radius) * t;
        dirty = true;
      }
      const pivotDiff = vec3.sub(targetPivot, pivot, temp);
      if (vec3.lenSq(pivotDiff) > epsilon * epsilon) {
        vec3.addScaled(pivot, pivotDiff, t, pivot);
        dirty = true;
      }
      if (dirty) recalc();
      return view_;
    }
  };
}

// lib/input.ts
function createInputHandler(window2, canvas2) {
  const digital = { forward: false, backward: false, left: false, right: false, up: false, down: false };
  const analog = { x: 0, y: 0, zoom: 0, touching: false, panning: false };
  const pointers = /* @__PURE__ */ new Map();
  let prevDist = 0;
  let prevMidX = 0;
  let prevMidY = 0;
  let isAlt = false;
  const setKey = (e, v) => {
    if (e.key === "Alt") isAlt = v;
    switch (e.code) {
      case "KeyW":
        digital.forward = v;
        break;
      case "KeyS":
        digital.backward = v;
        break;
      case "KeyA":
        digital.left = v;
        break;
      case "KeyD":
        digital.right = v;
        break;
      case "Space":
        digital.up = v;
        break;
      case "ShiftLeft":
        digital.down = v;
        break;
    }
  };
  window2.addEventListener("keydown", (e) => setKey(e, true));
  window2.addEventListener("keyup", (e) => setKey(e, false));
  canvas2.style.touchAction = "none";
  canvas2.addEventListener("pointerdown", (e) => {
    canvas2.setPointerCapture(e.pointerId);
    pointers.set(e.pointerId, e);
    if (pointers.size === 2) {
      const p = [...pointers.values()];
      prevDist = Math.hypot(p[0].clientX - p[1].clientX, p[0].clientY - p[1].clientY);
      prevMidX = (p[0].clientX + p[1].clientX) / 2;
      prevMidY = (p[0].clientY + p[1].clientY) / 2;
    }
  });
  const removePointer = (e) => {
    canvas2.releasePointerCapture(e.pointerId);
    pointers.delete(e.pointerId);
  };
  canvas2.addEventListener("pointerup", removePointer);
  canvas2.addEventListener("pointercancel", removePointer);
  canvas2.addEventListener("pointermove", (e) => {
    const prev = pointers.get(e.pointerId);
    if (!prev) return;
    pointers.set(e.pointerId, e);
    const mdx = e.clientX - prev.clientX;
    const mdy = e.clientY - prev.clientY;
    if (pointers.size === 2) {
      const p = [...pointers.values()];
      const dist = Math.hypot(p[0].clientX - p[1].clientX, p[0].clientY - p[1].clientY);
      const midX = (p[0].clientX + p[1].clientX) / 2;
      const midY = (p[0].clientY + p[1].clientY) / 2;
      analog.zoom -= (dist - prevDist) * 0.05;
      analog.x += midX - prevMidX;
      analog.y += midY - prevMidY;
      analog.panning = true;
      prevDist = dist;
      prevMidX = midX;
      prevMidY = midY;
    } else if (pointers.size === 1) {
      analog.x += mdx;
      analog.y += mdy;
      analog.panning = (e.buttons & 4) !== 0 || isAlt;
    }
  });
  canvas2.addEventListener("wheel", (e) => {
    e.preventDefault();
    analog.zoom -= Math.sign(e.deltaY);
  }, { passive: false });
  return () => {
    const out = {
      digital,
      analog: { ...analog, touching: pointers.size > 0 }
    };
    analog.x = 0;
    analog.y = 0;
    analog.zoom = 0;
    analog.panning = false;
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
  highDPI: false,
  bunnyRotation: 0,
  resetCamera: () => {
  },
  debugIterations: false
};
var pauseController = gui.add(controls, "pause").name("Pause");
var cameraController = gui.add(controls, "resetCamera").name("Reset Camera");
var highDPIController = gui.add(controls, "highDPI").name("High DPI");
var rotationController = gui.add(controls, "bunnyRotation", 0, 360, 1).name("Bunny Rotation");
var debugController = gui.add(controls, "debugIterations").name("Debug Iterations");

// lib/hw_skymodel.ts
function quintic9(data, offset, t) {
  const t2 = t * t;
  const t3 = t2 * t;
  const t4 = t2 * t2;
  const t5 = t4 * t;
  const invT = 1 - t;
  const invT2 = invT * invT;
  const invT3 = invT2 * invT;
  const invT4 = invT2 * invT2;
  const invT5 = invT4 * invT;
  return data[offset + 0 * 9] * invT5 + data[offset + 1 * 9] * 5 * invT4 * t + data[offset + 2 * 9] * 10 * invT3 * t2 + data[offset + 3 * 9] * 10 * invT2 * t3 + data[offset + 4 * 9] * 5 * invT * t4 + data[offset + 5 * 9] * t5;
}
function quintic1(data, offset, t) {
  const t2 = t * t;
  const t3 = t2 * t;
  const t4 = t2 * t2;
  const t5 = t4 * t;
  const invT = 1 - t;
  const invT2 = invT * invT;
  const invT3 = invT2 * invT;
  const invT4 = invT2 * invT2;
  const invT5 = invT4 * invT;
  return data[offset + 0] * invT5 + data[offset + 1] * 5 * invT4 * t + data[offset + 2] * 10 * invT3 * t2 + data[offset + 3] * 10 * invT2 * t3 + data[offset + 4] * 5 * invT * t4 + data[offset + 5] * t5;
}
function initParams(outParams, outOffset, data, turbidity, albedo, t) {
  const turbidityInt = Math.floor(turbidity);
  const turbidityRem = turbidity % 1;
  if (turbidityInt <= 0) {
    throw new RangeError("Turbidity must be greater than 0");
  }
  const turbidityMin = turbidityInt - 1;
  const turbidityMax = Math.min(turbidityInt, 9);
  const p0Offset = 9 * 6 * turbidityMin;
  const p1Offset = 9 * 6 * turbidityMax;
  const p2Offset = 9 * 6 * 10 + 9 * 6 * turbidityMin;
  const p3Offset = 9 * 6 * 10 + 9 * 6 * turbidityMax;
  const s0 = (1 - albedo) * (1 - turbidityRem);
  const s1 = (1 - albedo) * turbidityRem;
  const s2 = albedo * (1 - turbidityRem);
  const s3 = albedo * turbidityRem;
  for (let i = 0; i < 9; i++) {
    outParams[outOffset + i] = s0 * quintic9(data, p0Offset + i, t) + s1 * quintic9(data, p1Offset + i, t) + s2 * quintic9(data, p2Offset + i, t) + s3 * quintic9(data, p3Offset + i, t);
  }
}
function initSkyRadiance(outRadiance, outIndex, data, turbidity, albedo, t) {
  const turbidityInt = Math.floor(turbidity);
  const turbidityRem = turbidity % 1;
  if (turbidityInt <= 0) {
    throw new RangeError("Turbidity must be greater than 0");
  }
  const turbidityMin = turbidityInt - 1;
  const turbidityMax = Math.min(turbidityInt, 9);
  const p0Offset = 6 * turbidityMin;
  const p1Offset = 6 * turbidityMax;
  const p2Offset = 6 * 10 + 6 * turbidityMin;
  const p3Offset = 6 * 10 + 6 * turbidityMax;
  const s0 = (1 - albedo) * (1 - turbidityRem);
  const s1 = (1 - albedo) * turbidityRem;
  const s2 = albedo * (1 - turbidityRem);
  const s3 = albedo * turbidityRem;
  outRadiance[outIndex] = s0 * quintic1(data, p0Offset, t) + s1 * quintic1(data, p1Offset, t) + s2 * quintic1(data, p2Offset, t) + s3 * quintic1(data, p3Offset, t);
}
function initSolarRadiance(outRadiance, outIndex, data, turbidity) {
  const turbidityInt = Math.floor(turbidity);
  if (turbidityInt <= 0) {
    throw new RangeError("Turbidity must be greater than 0");
  }
  const turbidityRem = turbidity % 1;
  const turbidityMin = turbidityInt - 1;
  const turbidityMax = Math.min(turbidityInt, 9);
  outRadiance[outIndex] = data[turbidityMin] * (1 - turbidityRem) + data[turbidityMax] * turbidityRem;
}
function createSkyState(params) {
  const { elevation, turbidity, albedo } = params;
  if (elevation < 0 || elevation > Math.PI) {
    throw new RangeError("Elevation must be in range [0, \u03C0]");
  }
  if (turbidity < 1 || turbidity > 10) {
    throw new RangeError("Turbidity must be in range [1, 10]");
  }
  if (albedo.some((a) => a < 0 || a > 1)) {
    throw new RangeError("Albedo components must be in range [0, 1]");
  }
  const t = Math.pow(elevation / (0.5 * Math.PI), 1 / 3);
  const state = {
    params: new Float32Array(27),
    skyRadiances: new Float32Array(3),
    solarRadiances: new Float32Array(3)
  };
  initParams(state.params, 0, params_r, turbidity, albedo[0], t);
  initParams(state.params, 9, params_g, turbidity, albedo[1], t);
  initParams(state.params, 18, params_b, turbidity, albedo[2], t);
  initSkyRadiance(state.skyRadiances, 0, radiances_r, turbidity, albedo[0], t);
  initSkyRadiance(state.skyRadiances, 1, radiances_g, turbidity, albedo[1], t);
  initSkyRadiance(state.skyRadiances, 2, radiances_b, turbidity, albedo[2], t);
  initSolarRadiance(state.solarRadiances, 0, solar_radiances_r, turbidity);
  initSolarRadiance(state.solarRadiances, 1, solar_radiances_g, turbidity);
  initSolarRadiance(state.solarRadiances, 2, solar_radiances_b, turbidity);
  return state;
}
var params_b = new Float32Array([
  -1.372629,
  -0.490559,
  -41.007889,
  41.221691,
  -7389e-6,
  0.483936,
  6475e-6,
  3.471755,
  0.509294,
  -1.523025,
  -0.649708,
  6.249857,
  -5.662543,
  -0.019084,
  0.551281,
  -22e-6,
  2.507663,
  0.43396,
  -1.035567,
  -0.074787,
  0.922103,
  -2.140047,
  -0.023741,
  0.379552,
  -0.017691,
  7.479831,
  0.77293,
  -1.271086,
  -0.558819,
  0.690802,
  2.096832,
  -0.245397,
  1.410648,
  0.04475,
  -4.719115,
  0.574119,
  -0.97126,
  -0.070339,
  0.916727,
  -0.95021,
  0.300468,
  0.454705,
  -0.05929,
  5.266196,
  0.720414,
  -1.087457,
  -0.18889,
  0.815669,
  0.310171,
  -2.155419,
  1.422205,
  0.096923,
  3.122404,
  0.499943,
  -1.42528,
  -0.541351,
  -34.548828,
  34.81142,
  -8687e-6,
  0.491427,
  -2e-6,
  3.239879,
  0.60942,
  -1.688557,
  -0.807087,
  7.018459,
  -6.244574,
  -0.021493,
  0.399397,
  0.012525,
  1.630662,
  0.109786,
  -0.866415,
  0.078691,
  -0.523654,
  -1.21896,
  -0.020591,
  0.66849,
  -0.055841,
  8.602299,
  1.410496,
  -1.319763,
  -0.598532,
  1.253918,
  1.914706,
  -0.321674,
  0.901121,
  0.132484,
  -5.252749,
  0.062313,
  -0.970601,
  -0.059141,
  0.569315,
  -1.175362,
  0.522164,
  0.751821,
  -0.082477,
  5.875635,
  0.985086,
  -1.08533,
  -0.19561,
  0.801961,
  0.53381,
  -3.423464,
  1.110444,
  0.150792,
  2.864942,
  0.499948,
  -1.431967,
  -0.547894,
  -32.862881,
  33.052879,
  -8381e-6,
  0.477205,
  -3e-6,
  3.289973,
  0.59763,
  -1.801361,
  -0.931589,
  5.391756,
  -4.588592,
  -0.020401,
  0.414468,
  0.018145,
  1.051795,
  0.114565,
  -0.790536,
  0.145133,
  -0.160566,
  -1.592174,
  456e-6,
  0.338032,
  -0.077703,
  8.775384,
  1.489512,
  -1.308575,
  -0.553923,
  0.918413,
  2.011479,
  -0.384247,
  1.432274,
  0.163715,
  -4.408856,
  0.05273,
  -0.982987,
  -0.08183,
  0.446456,
  -1.442716,
  1.029641,
  -0.069916,
  8702e-6,
  5.706417,
  0.911645,
  -1.08713,
  -0.203801,
  0.72608,
  0.916438,
  -5.006183,
  1.511271,
  0.125713,
  2.715439,
  0.620165,
  -1.448662,
  -0.579907,
  -28.33268,
  28.580231,
  -9134e-6,
  0.440478,
  -3e-6,
  3.029357,
  0.554007,
  -2.061772,
  -1.14519,
  7.918478,
  -7.212525,
  -0.020208,
  0.296272,
  0.046897,
  0.851721,
  0.233459,
  -0.641375,
  0.178043,
  -2.412919,
  1.064484,
  -0.0195,
  0.676974,
  -0.175276,
  7.262714,
  1.325869,
  -1.304871,
  -0.397558,
  1.219002,
  0.728518,
  -0.27101,
  0.777973,
  0.324714,
  -0.881817,
  0.183952,
  -1.001104,
  -0.19948,
  0.367674,
  -1.409737,
  0.290156,
  0.250694,
  2469e-6,
  3.398923,
  0.858464,
  -1.111552,
  -0.24872,
  0.741084,
  1.703749,
  -5.007855,
  1.057763,
  0.135451,
  2.088715,
  0.660001,
  -1.547227,
  -0.667947,
  -18.614651,
  18.84045,
  -0.012422,
  0.415734,
  -2e-6,
  2.812423,
  0.544696,
  -2.04389,
  -1.149081,
  2.304118,
  -1.715757,
  -0.024336,
  0.281684,
  0.071855,
  1.06486,
  0.270679,
  -0.904072,
  -0.082745,
  -0.255568,
  -0.632622,
  -0.027709,
  0.667602,
  -0.251353,
  5.903839,
  1.241452,
  -1.000013,
  -0.101077,
  0.369917,
  0.877453,
  -0.304201,
  0.695105,
  0.436181,
  0.679342,
  0.257389,
  -1.171332,
  -0.376819,
  0.370138,
  -1.470757,
  0.552594,
  0.029915,
  0.015818,
  2.365233,
  0.821451,
  -1.068667,
  -0.232633,
  0.672506,
  2.243733,
  -4.61437,
  1.033677,
  0.137629,
  2.013334,
  0.68653,
  -1.592991,
  -0.724695,
  -25.98204,
  26.219601,
  -8365e-6,
  0.420757,
  -3e-6,
  2.623735,
  0.587319,
  -2.271349,
  -1.280884,
  6.308739,
  -5.75835,
  -0.01977,
  0.367184,
  0.06698,
  1.150597,
  0.175922,
  -0.636862,
  -7436e-6,
  -2.230026,
  1.640997,
  -0.015485,
  0.314533,
  -0.249264,
  5.083843,
  1.260215,
  -1.177925,
  -0.096281,
  0.305115,
  -0.037495,
  -0.271321,
  1.164226,
  0.455997,
  2.175429,
  0.287428,
  -1.0785,
  -0.380178,
  0.478891,
  -0.479597,
  0.597762,
  -0.448853,
  0.033869,
  1.538143,
  0.806205,
  -1.108028,
  -0.259689,
  0.51622,
  1.557081,
  -4.265039,
  1.182535,
  0.156376,
  2.095084,
  0.688338,
  -1.668427,
  -0.790851,
  -27.7969,
  27.997459,
  -7187e-6,
  0.375777,
  -3e-6,
  2.563421,
  0.543969,
  -2.156175,
  -1.220004,
  3.585732,
  -3.235988,
  -0.010862,
  0.184614,
  0.104602,
  1.234427,
  0.284219,
  -1.117051,
  -0.410163,
  -0.846373,
  0.767147,
  -0.022266,
  0.857494,
  -0.343412,
  4.475715,
  1.154824,
  -0.744484,
  0.231208,
  -0.539372,
  0.157421,
  -0.176391,
  0.275169,
  0.55642,
  2.217672,
  0.348393,
  -1.273036,
  -0.527556,
  0.490251,
  -0.044984,
  0.433937,
  0.238668,
  0.023809,
  1.413444,
  0.785592,
  -1.084192,
  -0.293675,
  0.471943,
  1.384436,
  -3.257789,
  0.611954,
  0.168188,
  1.650441,
  0.693663,
  -1.84849,
  -0.951267,
  -30.052509,
  30.243151,
  -5635e-6,
  0.344778,
  -3e-6,
  2.309422,
  0.564356,
  -2.300008,
  -1.252335,
  -1.218876,
  1.49373,
  -6107e-6,
  0.079749,
  0.102345,
  1.505934,
  0.236095,
  -1.483705,
  -0.854757,
  -0.779715,
  0.644797,
  -0.026781,
  1.091263,
  -0.334489,
  3.830416,
  1.189425,
  -0.534801,
  0.398273,
  -0.407157,
  0.326557,
  -0.086588,
  -0.237089,
  0.53691,
  1.478279,
  0.31433,
  -1.320401,
  -0.604325,
  0.30192,
  -0.077329,
  0.476838,
  0.674576,
  0.036941,
  1.158234,
  0.816906,
  -1.10104,
  -0.342002,
  0.377566,
  1.769338,
  -2.990515,
  0.164953,
  0.197012,
  1.453355,
  0.675976,
  -2.251946,
  -1.229349,
  -32.718079,
  32.831139,
  -4252e-6,
  0.337229,
  -3e-6,
  2.154046,
  0.584267,
  -1.867834,
  -0.953125,
  -12.29365,
  12.69149,
  -6845e-6,
  0.118511,
  0.075396,
  1.846381,
  0.189941,
  -3.398629,
  -2.180862,
  2.335213,
  -3.382823,
  -8614e-6,
  0.84316,
  -0.239357,
  3.11246,
  1.218556,
  0.570838,
  0.940603,
  -0.689011,
  2.746233,
  -0.057721,
  0.1096,
  0.349198,
  0.728145,
  0.321205,
  -1.705909,
  -0.851722,
  0.113116,
  -2.141434,
  0.427404,
  0.33976,
  0.178649,
  0.90261,
  0.78828,
  -1.012865,
  -0.349555,
  0.336904,
  3.724205,
  -3.089586,
  0.126696,
  0.146179,
  1.170199,
  0.693105,
  -2.890318,
  -1.665573,
  -34.937561,
  35.003689,
  -2984e-6,
  0.262242,
  -4e-6,
  1.947681,
  0.690575,
  -1.956022,
  -1.0629,
  -19.19714,
  19.75164,
  -8865e-6,
  0.216554,
  0.054756,
  1.761134,
  3164e-6,
  -5.612198,
  -3.101371,
  4.098034,
  -6.144001,
  9945e-6,
  0.290547,
  -0.170711,
  3.199107,
  1.33766,
  0.835376,
  0.485594,
  -1.243589,
  5.147385,
  -0.07014,
  0.938041,
  0.233571,
  0.172774,
  0.28027,
  -1.524329,
  -0.738855,
  0.325902,
  -4.050634,
  0.405855,
  -0.259138,
  0.18983,
  0.355607,
  0.788413,
  -1.070371,
  -0.420786,
  0.173986,
  5.29341,
  -3.136757,
  0.232386,
  0.167371,
  1.007227,
  0.684429,
  -1.34172,
  -0.483489,
  -46.334469,
  46.82148,
  -6137e-6,
  0.459922,
  7047e-6,
  2.895798,
  0.49994,
  -1.529104,
  -0.649863,
  15.34103,
  -14.50675,
  -0.015314,
  0.328008,
  0.016829,
  1.901587,
  0.501323,
  -1.014776,
  -0.14545,
  -4.071085,
  2.954982,
  -0.026303,
  0.568153,
  -0.030165,
  6.773854,
  0.50035,
  -1.172413,
  -0.402632,
  2.960428,
  0.202071,
  -0.200495,
  0.937557,
  0.059982,
  -4.945934,
  0.45029,
  -0.989816,
  -0.057728,
  0.447002,
  -0.578666,
  0.115817,
  0.346804,
  -0.050434,
  6.867947,
  0.801236,
  -1.085111,
  -0.188267,
  1.223748,
  0.35655,
  -3.688357,
  0.565372,
  0.067276,
  2.69013,
  0.49994,
  -1.389119,
  -0.529025,
  -40.557739,
  41.059719,
  -7063e-6,
  0.456006,
  -2e-6,
  2.775512,
  0.667145,
  -1.584641,
  -0.720062,
  12.48067,
  -11.56028,
  -0.016596,
  0.305003,
  0.010999,
  1.438927,
  -0.02138,
  -0.982607,
  -0.088873,
  -2.960031,
  1.808816,
  -0.024782,
  0.603573,
  -0.048684,
  7.347705,
  1.584739,
  -1.150423,
  -0.407379,
  2.412991,
  0.487084,
  -0.23379,
  0.829511,
  0.112991,
  -5.150045,
  -0.090166,
  -1.016933,
  -0.063115,
  0.521894,
  -0.571643,
  0.125099,
  0.360152,
  -0.054976,
  7.060139,
  1.018333,
  -1.073151,
  -0.184544,
  1.155394,
  0.300449,
  -3.431711,
  0.465703,
  0.094012,
  2.68862,
  0.499954,
  -1.391257,
  -0.536582,
  -42.558811,
  42.991322,
  -5838e-6,
  0.422913,
  -3e-6,
  2.775531,
  0.62346,
  -1.780062,
  -0.922888,
  13.76172,
  -12.60946,
  -0.015075,
  0.311743,
  0.02205,
  0.609373,
  0.034634,
  -0.738817,
  0.127567,
  -3.999528,
  2.223993,
  -0.018569,
  0.543931,
  -0.088341,
  8.037139,
  1.645951,
  -1.322387,
  -0.532014,
  2.659359,
  1.086712,
  -0.212971,
  0.870465,
  0.180031,
  -4.967241,
  -0.138372,
  -0.937829,
  -0.015999,
  0.360756,
  -1.980561,
  0.379146,
  0.121227,
  -0.02846,
  6.825542,
  1.059139,
  -1.100832,
  -0.217231,
  1.211561,
  2.002721,
  -5.010011,
  0.571758,
  0.067777,
  2.160006,
  0.567639,
  -1.409373,
  -0.570875,
  -30.349739,
  30.79809,
  -7281e-6,
  0.37233,
  -2e-6,
  2.577348,
  0.591338,
  -1.954312,
  -1.11651,
  5.399148,
  -4.299553,
  -0.017247,
  0.374282,
  0.041871,
  0.104488,
  0.123273,
  -0.677221,
  0.20014,
  -0.367052,
  -1.014628,
  -3497e-6,
  0.409986,
  -0.158463,
  7.7504,
  1.514559,
  -1.2916,
  -0.497744,
  0.964191,
  1.56242,
  -0.322778,
  0.905543,
  0.304644,
  -3.385619,
  9546e-6,
  -0.975086,
  -0.087706,
  0.905426,
  -1.429236,
  0.897478,
  -0.121796,
  -0.051946,
  4.909409,
  0.958915,
  -1.088007,
  -0.19593,
  0.97458,
  1.260761,
  -5.008864,
  0.727125,
  0.109666,
  2.717295,
  0.634073,
  -1.45605,
  -0.622307,
  -22.28088,
  22.696039,
  -9341e-6,
  0.411831,
  -2e-6,
  2.442117,
  0.558964,
  -2.176449,
  -1.302416,
  2.222836,
  -1.22273,
  -0.017281,
  0.132351,
  0.070277,
  0.048357,
  0.209335,
  -0.578964,
  0.221541,
  0.214229,
  -1.201725,
  -0.011857,
  0.812298,
  -0.238042,
  6.706841,
  1.404146,
  -1.307463,
  -0.451517,
  0.644783,
  1.223841,
  -0.290239,
  0.498659,
  0.407365,
  -1.706696,
  0.106088,
  -0.969868,
  -0.130709,
  0.938935,
  -1.522852,
  0.77688,
  -0.13686,
  -0.038574,
  3.676935,
  0.898097,
  -1.104349,
  -0.238032,
  1.047043,
  1.865421,
  -5.011664,
  0.701495,
  0.096227,
  1.89136,
  0.668735,
  -1.502249,
  -0.672452,
  -28.88092,
  29.3036,
  -6686e-6,
  0.368546,
  -2e-6,
  2.310797,
  0.556675,
  -2.217125,
  -1.364924,
  4.048243,
  -3.111333,
  -0.013177,
  0.192195,
  0.086277,
  1982e-6,
  0.221369,
  -0.621576,
  0.1688,
  -0.594913,
  -0.155129,
  336e-6,
  0.689766,
  -0.285505,
  6.271042,
  1.363084,
  -1.216317,
  -0.348943,
  0.756623,
  0.540981,
  -0.283084,
  0.619183,
  0.475516,
  -0.913139,
  0.138391,
  -1.030437,
  -0.203406,
  0.8336,
  -1.050947,
  0.868909,
  -0.367231,
  -0.040562,
  3.111269,
  0.885684,
  -1.078984,
  -0.207055,
  0.968315,
  1.497022,
  -5.007653,
  0.770254,
  0.128582,
  2.225188,
  0.658791,
  -1.559291,
  -0.737404,
  -35.963112,
  36.3447,
  -4667e-6,
  0.327796,
  -2e-6,
  2.215652,
  0.576468,
  -2.356929,
  -1.444755,
  6.244526,
  -5.540162,
  -8795e-6,
  0.17921,
  0.095785,
  0.373768,
  0.192219,
  -0.658975,
  -0.029269,
  -1.831779,
  1.869962,
  -203e-5,
  0.755209,
  -0.316816,
  4.632196,
  1.294054,
  -1.161046,
  -0.147251,
  0.649414,
  -0.832717,
  -0.232072,
  0.339121,
  0.526964,
  0.937634,
  0.245857,
  -1.034427,
  -0.30625,
  0.897563,
  0.320353,
  0.856514,
  -0.125016,
  -0.04094,
  1.861304,
  0.822347,
  -1.109954,
  -0.274028,
  1.063811,
  0.70774,
  -4.695734,
  0.56217,
  0.124896,
  1.297723,
  0.678972,
  -1.788293,
  -0.936875,
  -43.8298,
  44.24963,
  -3653e-6,
  0.309433,
  -3e-6,
  1.904402,
  0.58616,
  -2.268206,
  -1.312676,
  2.863082,
  -2.373727,
  -5145e-6,
  0.171107,
  0.09316,
  0.93096,
  0.179168,
  -1.376966,
  -0.741858,
  -1.349589,
  1.563419,
  -3124e-6,
  0.696714,
  -0.306189,
  3.602731,
  1.255669,
  -0.601754,
  0.281593,
  0.542405,
  -0.688545,
  -0.162,
  0.298005,
  0.499557,
  0.73712,
  0.281247,
  -1.278853,
  -0.524533,
  0.787052,
  0.312507,
  0.77481,
  -0.077886,
  3491e-6,
  1.283748,
  0.813019,
  -1.05093,
  -0.278633,
  1.056344,
  1.053002,
  -4.047789,
  0.443217,
  0.116908,
  0.953262,
  0.680676,
  -2.084927,
  -1.203954,
  -48.81638,
  49.201599,
  -2896e-6,
  0.288298,
  -3e-6,
  1.702211,
  0.637418,
  -2.328567,
  -1.238023,
  -1.891019,
  2.45152,
  -5848e-6,
  0.20847,
  0.078481,
  1.211048,
  0.08095,
  -2.634632,
  -1.78946,
  -0.137056,
  -0.332644,
  2784e-6,
  0.523945,
  -0.254888,
  2.896327,
  1.324116,
  0.068826,
  0.599782,
  0.15354,
  1.375209,
  -0.126729,
  0.423974,
  0.401312,
  0.179467,
  0.239538,
  -1.430918,
  -0.643904,
  0.832598,
  -1.705612,
  0.723643,
  -0.055676,
  0.064087,
  0.683652,
  0.838889,
  -1.037956,
  -0.32154,
  0.945735,
  3.178114,
  -4.152156,
  0.223099,
  0.11562,
  0.760622,
  0.665692,
  -2.967314,
  -1.728778,
  -37.309879,
  37.555779,
  -2589e-6,
  0.292797,
  -4e-6,
  1.592161,
  0.686869,
  -2.123311,
  -1.175148,
  -13.14988,
  13.86882,
  -7829e-6,
  0.185203,
  0.05481,
  1.294309,
  0.024282,
  -5.443597,
  -3.156344,
  2.110838,
  -3.421556,
  0.011819,
  0.119695,
  -0.17429,
  2.404353,
  1.272805,
  1.029898,
  0.591252,
  -0.398353,
  3.286069,
  -0.092521,
  1.331381,
  0.256064,
  0.800175,
  0.362418,
  -1.547574,
  -0.78816,
  1.020902,
  -2.897069,
  0.521347,
  -0.924232,
  0.118559,
  -1.150721,
  0.731721,
  -0.962104,
  -0.199141,
  0.653129,
  3.925839,
  -3.596904,
  0.631733,
  0.153133,
  1.457846,
  0.696629
]);
var params_g = new Float32Array([
  -1.14053,
  -0.198275,
  -7.51273,
  8.403899,
  -0.05699,
  0.901591,
  0.033922,
  4.772522,
  0.511118,
  -1.165117,
  -0.185296,
  2.963684,
  -2.262274,
  -0.157168,
  0.633997,
  0.049779,
  7.243307,
  0.422005,
  -1.169936,
  -0.335743,
  1.911291,
  -0.239107,
  -0.479164,
  1.446113,
  -0.091781,
  -4.700239,
  0.809622,
  -1.060246,
  -0.105163,
  0.501383,
  2.832309,
  -0.370786,
  1.523131,
  0.091637,
  5.604183,
  0.720857,
  -1.089753,
  -0.238217,
  2.360312,
  -5.902562,
  -8.799894,
  1.377692,
  -0.061316,
  -1.415472,
  0.612406,
  -1.075481,
  -0.124239,
  1.425781,
  8.810319,
  -2.922646,
  1.48652,
  0.032706,
  3.889783,
  0.499948,
  -1.149342,
  -0.207634,
  -7.446587,
  8.014559,
  -0.048662,
  0.820304,
  0.063865,
  4.894198,
  0.545205,
  -1.120531,
  -0.151331,
  2.735504,
  -2.417591,
  -0.136111,
  0.429634,
  0.094275,
  8.171403,
  0.410245,
  -1.226964,
  -0.351638,
  1.308298,
  -0.050975,
  -0.484678,
  1.654619,
  -0.113494,
  -3.347854,
  1.131147,
  -0.966438,
  0.027676,
  0.165824,
  2.407439,
  -0.13003,
  0.917096,
  0.274289,
  6.642633,
  0.255006,
  -1.153358,
  -0.312622,
  2.078934,
  -5.857733,
  -8.659848,
  1.758505,
  -0.096161,
  -1.230863,
  0.966383,
  -1.05385,
  -0.133074,
  1.481738,
  10.49485,
  -3.528854,
  0.914236,
  0.124488,
  2.644615,
  0.500105,
  -1.173687,
  -0.236036,
  -3.741454,
  4.088507,
  -0.075282,
  0.664524,
  0.077183,
  4.65122,
  0.558632,
  -1.213757,
  -0.258956,
  0.713255,
  -0.425933,
  -0.198082,
  0.362781,
  0.046666,
  5.807984,
  0.584738,
  -1.108794,
  -0.225987,
  1.574179,
  -0.375373,
  -0.598474,
  1.659414,
  -0.01681,
  0.678522,
  0.864733,
  -1.060896,
  -0.013467,
  -0.752966,
  1.711319,
  -0.979244,
  0.202243,
  0.382649,
  5.725157,
  0.529071,
  -1.085145,
  -0.284072,
  2.088029,
  -4.935097,
  -9.056542,
  1.976149,
  -0.039125,
  -0.863606,
  0.745212,
  -1.077983,
  -0.141663,
  1.100848,
  10.15875,
  -2.943712,
  0.525514,
  0.216422,
  2.941143,
  0.669994,
  -1.223293,
  -0.286744,
  -1.624136,
  1.668299,
  -0.095376,
  0.501595,
  0.113074,
  4.244812,
  0.508215,
  -1.325342,
  -0.428099,
  0.470549,
  0.069266,
  -0.457259,
  0.534414,
  -0.025542,
  3.093939,
  0.66394,
  -1.113581,
  -0.119213,
  0.401154,
  0.701189,
  0.205284,
  0.988072,
  0.018075,
  4.69016,
  0.857624,
  -1.016063,
  -0.103814,
  -0.228039,
  0.789892,
  -11.27333,
  0.207455,
  0.538818,
  1.364263,
  0.466045,
  -1.099582,
  -0.222861,
  1.332648,
  5.135188,
  1.653152,
  1.41702,
  -0.108753,
  1.809275,
  0.808087,
  -1.064357,
  -0.152077,
  0.820737,
  -1324e-6,
  -5.009523,
  0.39463,
  0.43379,
  2.593198,
  0.671917,
  -1.278702,
  -0.351287,
  -0.451106,
  0.389576,
  -0.242967,
  0.427058,
  0.113535,
  3.71913,
  0.499887,
  -1.580069,
  -0.709548,
  -0.31989,
  1.715748,
  -1.185915,
  0.452316,
  -0.010262,
  0.792719,
  0.553835,
  -0.947402,
  0.11737,
  0.488138,
  -2.618684,
  3.251661,
  1.213931,
  -0.017363,
  8.000768,
  1.025998,
  -1.129091,
  -0.328769,
  -0.352408,
  3.352892,
  -14.16073,
  -0.848562,
  0.656077,
  -2.820937,
  0.31113,
  -1.030884,
  -0.113758,
  1.109855,
  8.082276,
  1.519214,
  2.112433,
  -0.15923,
  3.675905,
  0.870337,
  -1.075192,
  -0.162717,
  0.351491,
  1.168164,
  -4.255822,
  -0.601535,
  0.626578,
  2.884818,
  0.654838,
  -1.316017,
  -0.388965,
  -0.503085,
  0.44887,
  -0.31868,
  0.457076,
  0.089092,
  3.659274,
  0.501175,
  -1.731876,
  -0.849381,
  0.119487,
  2.002781,
  -2.006547,
  0.487223,
  -0.028546,
  0.266214,
  0.461163,
  -0.927368,
  0.138095,
  -0.330218,
  -3.553265,
  4.633345,
  0.969673,
  0.087998,
  8.291129,
  1.094451,
  -1.099377,
  -0.332539,
  0.250106,
  2.613712,
  -13.28142,
  -0.557953,
  0.499208,
  -3.504402,
  0.302292,
  -1.04842,
  -0.122777,
  0.584537,
  11.05869,
  0.038132,
  1.330409,
  0.019781,
  3.95943,
  0.839644,
  -1.063233,
  -0.156064,
  0.284003,
  0.875157,
  -3.41182,
  -0.143656,
  0.584658,
  2.899292,
  0.67991,
  -1.376715,
  -0.454157,
  -1.445491,
  1.569898,
  -0.139063,
  0.555827,
  0.041099,
  3.349451,
  0.551612,
  -1.953391,
  -1.035869,
  1.690563,
  -0.196469,
  -0.77871,
  0.579961,
  0.029456,
  0.042179,
  0.245137,
  -1.012422,
  0.071365,
  -1.862534,
  -0.722865,
  0.1948,
  0.209181,
  0.063992,
  7.928994,
  1.290733,
  -0.970671,
  -0.288095,
  1.107797,
  -2.731734,
  -8.445995,
  0.429677,
  0.511765,
  -3.824277,
  0.176121,
  -1.110611,
  -0.178941,
  0.210849,
  20.7143,
  -1.763174,
  0.095547,
  -0.029431,
  3.422079,
  0.88155,
  -1.048334,
  -0.161409,
  0.247518,
  0.021469,
  -2.983901,
  0.253822,
  0.560137,
  2.461925,
  0.677739,
  -1.393719,
  -0.500272,
  -2.40894,
  2.680983,
  -0.136283,
  0.739507,
  -3e-6,
  3.260889,
  0.813206,
  -2.128663,
  -1.151182,
  2.923026,
  -1.931838,
  -0.442617,
  0.230998,
  -5486e-6,
  0.327953,
  -0.222947,
  -1.618022,
  -0.376649,
  -3.163544,
  1.611608,
  -0.396748,
  0.393368,
  0.300674,
  6.835177,
  1.613765,
  -0.566906,
  -0.148175,
  2.071817,
  -8.157422,
  -5.988088,
  0.23872,
  0.144719,
  -4.296385,
  0.050113,
  -1.241724,
  -0.251935,
  -0.190861,
  29.52235,
  -3.33366,
  -0.018377,
  0.102225,
  2.92932,
  0.886726,
  -1.02167,
  -0.166733,
  0.178977,
  -2178e-6,
  -2.641572,
  -0.056415,
  0.530376,
  2.138196,
  0.678035,
  -1.669332,
  -0.758871,
  -2.993557,
  3.17876,
  -0.080664,
  0.654467,
  -8e-6,
  2.628924,
  0.900127,
  -1.755806,
  -0.873535,
  3.258881,
  -2.504785,
  -0.330079,
  0.118056,
  -9316e-6,
  1.785154,
  -0.320582,
  -3.720277,
  -1.73335,
  -3.332272,
  1.515869,
  0.173422,
  0.801196,
  0.199544,
  3.817666,
  1.638502,
  0.472464,
  0.320983,
  2.051443,
  -5.105574,
  -6.509139,
  -0.423204,
  0.259893,
  -2.151756,
  -3494e-6,
  -1.5256,
  -0.489761,
  -0.098911,
  23.46818,
  -2.278152,
  0.168122,
  -0.044694,
  1.051,
  0.929467,
  -0.990865,
  -0.200818,
  0.160514,
  -2463e-6,
  -2.477349,
  -0.121865,
  0.475012,
  1.460813,
  0.666136,
  -2.122119,
  -1.125475,
  -3.066599,
  3.145078,
  -0.054116,
  0.513363,
  -8e-6,
  2.268448,
  0.900142,
  -1.528158,
  -0.937025,
  2.567559,
  -1.591439,
  -0.363446,
  0.176326,
  112e-5,
  1.811848,
  -0.263793,
  -6.524387,
  -2.673507,
  -2.940472,
  -0.602561,
  0.785207,
  1.073499,
  -0.035404,
  3.517416,
  1.490466,
  0.888603,
  -0.096818,
  1.430554,
  4.993717,
  -6.071355,
  -0.605399,
  0.5093,
  -1.27301,
  0.074913,
  -1.481997,
  -0.589728,
  0.265926,
  1.267239,
  -0.574129,
  0.05983,
  -0.221731,
  -0.301645,
  0.926083,
  -1.010943,
  -0.207513,
  0.050667,
  14.70708,
  -3.780501,
  0.072532,
  0.404546,
  1.320164,
  0.655993,
  -1.129907,
  -0.188401,
  -8.04767,
  9.035776,
  -0.055394,
  0.882335,
  0.031971,
  4.839388,
  0.504282,
  -1.133821,
  -0.151078,
  3.362822,
  -2.453381,
  -0.146392,
  0.472871,
  0.059581,
  7.6363,
  0.480516,
  -1.176518,
  -0.35499,
  1.729044,
  -0.216097,
  -0.507586,
  1.675584,
  -0.089069,
  -5.386842,
  0.545222,
  -1.043563,
  -0.07521,
  0.875064,
  2.510518,
  7585e-6,
  0.936125,
  0.078891,
  6.066644,
  0.581311,
  -1.081304,
  -0.222225,
  2.517638,
  -4.45382,
  -8.663691,
  0.866256,
  -0.048027,
  -0.896545,
  0.488666,
  -1.083774,
  -0.137547,
  1.685818,
  5.63112,
  -3.100752,
  0.404594,
  0.023469,
  3.390321,
  0.500831,
  -1.143158,
  -0.205833,
  -9.660198,
  10.62394,
  -0.044341,
  0.860762,
  0.031773,
  4.416481,
  0.591816,
  -1.146773,
  -0.172739,
  4.626048,
  -4.684602,
  -0.083071,
  0.161962,
  0.148487,
  7.572868,
  0.268113,
  -1.151324,
  -0.30993,
  0.41256,
  2.340752,
  -0.421444,
  1.987375,
  -0.191341,
  -3.845978,
  1.337311,
  -1.034258,
  -7779e-6,
  0.705009,
  -0.803637,
  0.313857,
  0.246945,
  0.355997,
  7.485917,
  0.047903,
  -1.096568,
  -0.267317,
  2.575654,
  -0.805712,
  -8.884928,
  1.41617,
  -0.209131,
  -1.543494,
  1.065445,
  -1.083304,
  -0.152827,
  1.697727,
  2.503702,
  -2.885296,
  -0.12985,
  0.154887,
  2.479652,
  0.50665,
  -1.165736,
  -0.232994,
  -5.967964,
  6.705959,
  -0.059314,
  0.748564,
  0.039139,
  4.221591,
  0.618393,
  -1.212422,
  -0.254591,
  2.418626,
  -2.266104,
  -0.110201,
  0.013639,
  0.105541,
  5.648062,
  0.455741,
  -1.070436,
  -0.216334,
  0.709872,
  0.784307,
  -0.432393,
  2.109823,
  -0.095897,
  -0.198519,
  1.060428,
  -1.104879,
  -0.030136,
  0.029763,
  1.069707,
  0.141,
  -0.488002,
  0.445229,
  6.41859,
  0.319599,
  -1.048969,
  -0.265532,
  2.689426,
  -3.941038,
  -9.506461,
  1.837119,
  -0.189212,
  -1.562146,
  0.904341,
  -1.106145,
  -0.160164,
  1.544544,
  7.388492,
  -2.9246,
  -0.432845,
  0.176316,
  2.523111,
  0.58519,
  -1.203666,
  -0.277659,
  -2.084286,
  2.45084,
  -0.087466,
  0.525851,
  0.079833,
  3.860055,
  0.548617,
  -1.340448,
  -0.423059,
  0.346285,
  0.470761,
  -0.251263,
  0.153075,
  0.027242,
  3.035216,
  0.587613,
  -1.014554,
  -0.116879,
  0.947779,
  -1.061218,
  -0.419673,
  2.058832,
  -0.059896,
  3.058168,
  0.976386,
  -1.137388,
  -0.09854,
  -0.298489,
  3.64782,
  -0.658557,
  -1.47918,
  0.610293,
  3.265914,
  0.348033,
  -1.021816,
  -0.234496,
  2.463671,
  -7.240685,
  -8.862697,
  2.514058,
  -0.212277,
  -0.03314,
  0.902814,
  -1.126581,
  -0.187435,
  1.454154,
  10.34398,
  -3.237393,
  -0.865493,
  0.245725,
  1.845769,
  0.600248,
  -1.263727,
  -0.343935,
  -0.178639,
  0.398017,
  -0.334952,
  0.382517,
  0.102922,
  3.331096,
  0.499896,
  -1.53001,
  -0.68797,
  0.238042,
  1.608216,
  -1.682679,
  0.354636,
  -3915e-6,
  0.451766,
  0.51286,
  -0.968566,
  0.094804,
  0.060768,
  -3.217561,
  4.568074,
  1.069299,
  0.020836,
  7.301088,
  1.072165,
  -1.113925,
  -0.311238,
  0.395413,
  5.105907,
  -14.56866,
  -0.491738,
  0.528991,
  -2.678374,
  0.301471,
  -1.046864,
  -0.121575,
  1.778308,
  4.661489,
  0.256558,
  1.35368,
  -0.117577,
  3.415972,
  0.845775,
  -1.10448,
  -0.194091,
  1.343668,
  -1759e-6,
  -5.009204,
  -0.418695,
  0.312571,
  1.628183,
  0.672041,
  -1.286902,
  -0.378124,
  -0.089773,
  0.354539,
  -0.486652,
  0.384366,
  0.082817,
  3.122231,
  0.504699,
  -1.712597,
  -0.854911,
  0.480929,
  1.515398,
  -2.212211,
  0.253903,
  0.02336,
  -0.060895,
  0.426844,
  -0.880728,
  0.16461,
  -0.44379,
  -3.188247,
  5.984417,
  1.334779,
  -0.04027,
  7.546431,
  1.175751,
  -1.147253,
  -0.35382,
  0.610184,
  4.43778,
  -15.59813,
  -1.103222,
  0.624204,
  -3.091472,
  0.217429,
  -1.03823,
  -0.121348,
  1.547505,
  5.893176,
  1.368738,
  1.663127,
  -0.137713,
  3.185279,
  0.873645,
  -1.101026,
  -0.187491,
  1.272667,
  3.596524,
  -5.007243,
  -0.635248,
  0.304899,
  1.931613,
  0.678884,
  -1.342753,
  -0.438497,
  -1.213491,
  1.621399,
  -0.155144,
  0.561422,
  0.025917,
  2.958967,
  0.578213,
  -1.937684,
  -1.066019,
  1.913336,
  -0.734772,
  -0.591617,
  0.158759,
  0.109257,
  -0.6275,
  0.159907,
  -0.930239,
  0.148619,
  -1.603835,
  0.178371,
  1.100461,
  1.174181,
  -0.160236,
  7.868331,
  1.468971,
  -1.053631,
  -0.372705,
  1.114117,
  -0.960329,
  -10.62469,
  -1.16214,
  0.79528,
  -4.478765,
  -0.044409,
  -1.083629,
  -0.126141,
  1.229344,
  11.27825,
  0.131901,
  1.624729,
  -0.28259,
  3.661082,
  1.036911,
  -1.09395,
  -0.206746,
  1.258035,
  7.548645,
  -4.598387,
  -0.894493,
  0.329263,
  1.311304,
  0.629187,
  -1.385867,
  -0.506814,
  -1.48649,
  1.969049,
  -0.169803,
  0.662917,
  -5e-6,
  2.760315,
  0.864437,
  -2.107367,
  -1.175639,
  2.313241,
  -1.001653,
  -0.484314,
  0.112448,
  39e-6,
  -0.350247,
  -0.320478,
  -1.475244,
  -0.283305,
  -2.085824,
  1.192563,
  -0.76452,
  0.838008,
  0.220358,
  7.157885,
  1.753702,
  -0.664437,
  -0.254974,
  1.600273,
  -8.589034,
  -6.144718,
  -0.759973,
  0.289837,
  -5.770923,
  -0.096562,
  -1.211687,
  -0.165349,
  0.83934,
  27.92988,
  -3.395461,
  0.993375,
  -0.039769,
  3.776659,
  0.954653,
  -1.063757,
  -0.203756,
  1.117207,
  -1253e-6,
  -3.33233,
  -0.697141,
  0.338872,
  1.311398,
  0.663517,
  -1.678889,
  -0.79923,
  -2.421687,
  2.871029,
  -0.076628,
  0.604621,
  -8e-6,
  2.002314,
  0.900131,
  -1.692144,
  -0.880425,
  3.060895,
  -2.000009,
  -0.318356,
  0.083859,
  -6327e-6,
  1.206639,
  -0.336997,
  -3.676795,
  -1.719207,
  -2.534697,
  1.005285,
  0.155041,
  1.07291,
  0.131809,
  3.717018,
  1.689191,
  0.542454,
  0.326353,
  1.551055,
  -3.841058,
  -6.598996,
  -1.201779,
  0.353067,
  -2.542945,
  -0.064825,
  -1.553849,
  -0.457686,
  0.932468,
  19.509821,
  -2.344516,
  1.12102,
  -0.122154,
  0.72855,
  0.958282,
  -1.02065,
  -0.22158,
  1.009774,
  -2057e-6,
  -2.740338,
  -0.812235,
  0.332897,
  0.898277,
  0.659468,
  -2.24736,
  -1.221267,
  -3.072346,
  3.385139,
  -0.043876,
  0.508489,
  -7e-6,
  1.750107,
  0.90014,
  -1.248499,
  -0.844272,
  3.062611,
  -2.020314,
  -0.281534,
  0.052547,
  3345e-6,
  1.433225,
  -0.283591,
  -7.004119,
  -2.927978,
  -2.649852,
  0.797189,
  0.546689,
  1.442667,
  -0.060639,
  2.806194,
  1.547429,
  1.434882,
  0.091146,
  1.170089,
  0.035128,
  -5.861915,
  -1.411843,
  0.540049,
  -0.774652,
  0.02387,
  -1.559053,
  -0.55023,
  1.200396,
  13.47741,
  -2.344397,
  0.886891,
  -0.329266,
  -1.362105,
  0.921783,
  -1.044436,
  -0.236072,
  0.705447,
  -2905e-6,
  -2.092829,
  -0.511967,
  0.417486,
  0.968744,
  0.658843
]);
var params_r = new Float32Array([
  -1.099459,
  -0.133515,
  -4.083223,
  5.919603,
  -0.110417,
  1.600158,
  -1e-6,
  4.917807,
  0.512772,
  -1.169858,
  -0.183279,
  0.969474,
  0.094958,
  -0.047389,
  0.219417,
  0.109575,
  3.603604,
  0.381512,
  -0.966523,
  -0.140389,
  5.194457,
  -1.107607,
  -0.813518,
  4.969661,
  -0.230051,
  -2.48935,
  1.279158,
  -1.292508,
  -0.129955,
  -2.071404,
  -0.047525,
  1.215598,
  -1.904179,
  0.302799,
  8.707768,
  0.063324,
  -0.926467,
  -0.169678,
  4.57407,
  -0.423294,
  -7.575833,
  5.079755,
  -0.257634,
  -4.506805,
  0.690813,
  -1.139072,
  -0.179606,
  1.923311,
  6.788529,
  -2.364389,
  -1.064041,
  0.171701,
  1.534681,
  0.501581,
  -1.107257,
  -0.138441,
  -4.285744,
  5.713157,
  -0.101599,
  1.372638,
  0.065559,
  5.127514,
  0.655047,
  -1.187337,
  -0.196901,
  0.855105,
  0.052897,
  -0.076264,
  0.017332,
  0.177945,
  3.801038,
  0.474271,
  -0.968532,
  -0.155331,
  4.732492,
  -1.178935,
  -0.785279,
  4.604492,
  -0.266652,
  -2.367663,
  1.177527,
  -1.252817,
  -0.051299,
  -2.800433,
  -0.01296,
  1.308964,
  -2.204331,
  0.727601,
  8.699265,
  0.118839,
  -0.945951,
  -0.232213,
  4.375041,
  -0.171202,
  -7.451681,
  5.078019,
  -0.422354,
  -4.595561,
  1.074719,
  -1.125092,
  -0.179675,
  1.626399,
  6.989743,
  -2.406382,
  -0.906038,
  0.296161,
  1.337715,
  0.543814,
  -1.135338,
  -0.171616,
  -1.499253,
  2.373491,
  -0.165402,
  0.95664,
  0.111345,
  4.528473,
  0.657944,
  -1.13278,
  -0.145621,
  -1.736672,
  1.756589,
  -0.1087,
  0.375793,
  0.252507,
  7.178513,
  0.500381,
  -1.167176,
  -0.292722,
  5.727667,
  -3.139244,
  -0.64252,
  2.822634,
  -0.145781,
  -6.78708,
  1.017072,
  -1.042529,
  0.041108,
  -4.000629,
  4.362364,
  1.09054,
  -1.338674,
  0.824696,
  10.95249,
  0.291221,
  -1.061598,
  -0.209614,
  3.803155,
  -7.977069,
  -3.63788,
  3.707671,
  -0.190313,
  -3.397953,
  0.99715,
  -1.07356,
  -0.207796,
  1.492052,
  16.26322,
  -5.015304,
  -0.405989,
  0.265978,
  0.639538,
  0.563444,
  -1.172794,
  -0.211119,
  -1.360013,
  1.60408,
  -0.084737,
  0.721731,
  0.154803,
  4.25701,
  0.632897,
  -1.238374,
  -0.267083,
  0.324768,
  0.546631,
  -0.742595,
  0.527644,
  0.02678,
  5.484169,
  0.681473,
  -1.176923,
  -0.257459,
  2.304045,
  -2.797678,
  1.464405,
  1.998552,
  0.255056,
  -4.199772,
  0.754489,
  -1.003284,
  0.01944,
  -2.145066,
  10.30924,
  -15.25413,
  -2.02301,
  0.54487,
  8.159497,
  0.553915,
  -1.060017,
  -0.203721,
  2.483018,
  -4.595459,
  6.526991,
  4.031804,
  0.120651,
  -2.586527,
  0.787575,
  -1.081141,
  -0.21233,
  1.092275,
  2.683841,
  -4.166938,
  -1.396582,
  0.43712,
  1.030233,
  0.666486,
  -1.222392,
  -0.265192,
  -0.462504,
  0.352196,
  0.021489,
  0.507849,
  0.179159,
  3.852516,
  0.599822,
  -1.42461,
  -0.471016,
  -0.182682,
  1.786277,
  -1.952442,
  0.527761,
  -0.017736,
  2.415874,
  0.670127,
  -1.130655,
  -0.135861,
  0.91712,
  -4.660394,
  6.251162,
  1.904529,
  0.263967,
  1.85613,
  0.822844,
  -0.973902,
  -0.066747,
  -0.47689,
  12.48589,
  -19.94688,
  -2.353043,
  0.588557,
  1.287251,
  0.483014,
  -1.082178,
  -0.19745,
  1.050245,
  -4.792855,
  8.663406,
  3.246969,
  0.155673,
  0.811744,
  0.805038,
  -1.063354,
  -0.172711,
  0.968159,
  2.736077,
  -4.969269,
  -0.836057,
  0.599461,
  1.024039,
  0.678693,
  -1.261936,
  -0.305368,
  -0.426222,
  0.40002,
  -0.020594,
  0.47218,
  0.148003,
  3.505343,
  0.612134,
  -1.681088,
  -0.697192,
  -0.110565,
  0.743743,
  -0.65944,
  0.225422,
  0.087102,
  1.263913,
  0.568187,
  -0.9453,
  0.034604,
  0.606704,
  -1.985128,
  3.457236,
  2.655483,
  -0.011624,
  3.304716,
  1.00195,
  -1.086609,
  -0.202901,
  -0.639917,
  6.926885,
  -15.12189,
  -3.793051,
  0.945612,
  0.222222,
  0.289373,
  -1.041259,
  -0.138879,
  1.147331,
  6.282086,
  3.679836,
  4.398314,
  -0.135523,
  1.031134,
  0.927351,
  -1.063473,
  -0.191605,
  0.655698,
  -3372e-6,
  -3.699664,
  -1.926783,
  0.737115,
  1.179975,
  0.636707,
  -1.33639,
  -0.377893,
  -0.725948,
  0.227025,
  0.462751,
  0.136646,
  0.263735,
  3.292059,
  0.499821,
  -2.119878,
  -1.055472,
  0.542205,
  0.782665,
  -1.286065,
  0.951791,
  -0.143236,
  -0.237982,
  0.591051,
  -0.776143,
  0.212434,
  -0.684518,
  -0.981234,
  4.347257,
  0.967198,
  0.377315,
  5.789529,
  0.96466,
  -1.118734,
  -0.351382,
  0.550092,
  0.944963,
  -12.6207,
  -1.82528,
  0.473126,
  -3.326892,
  0.356877,
  -1.026437,
  -0.082579,
  0.32217,
  11.98372,
  1.55513,
  2.560304,
  0.140647,
  2.912858,
  0.864318,
  -1.069949,
  -0.202961,
  0.582504,
  -2399e-6,
  -3.278335,
  -1.349882,
  0.720843,
  0.850516,
  0.662539,
  -1.392309,
  -0.445495,
  -0.5664,
  0.628339,
  -0.376173,
  0.69498,
  0.077482,
  3.192797,
  0.596866,
  -2.713405,
  -1.395112,
  0.202923,
  0.187727,
  -0.371586,
  -0.165293,
  0.238586,
  -0.415077,
  0.137547,
  -0.958864,
  0.024339,
  -1.527493,
  -0.963287,
  5.496269,
  1.094931,
  0.200404,
  6.084554,
  1.369604,
  -0.802855,
  -0.247356,
  1.617898,
  2.073591,
  -11.49446,
  -0.839413,
  0.272685,
  -4.634538,
  0.136729,
  -1.198326,
  -0.180487,
  -0.356541,
  4.0732,
  1.662086,
  1.23977,
  0.336798,
  2.997402,
  0.936038,
  -1.013531,
  -0.185906,
  0.579986,
  13.31883,
  -4.346873,
  -1.11382,
  0.527571,
  0.804518,
  0.649637,
  -1.530103,
  -0.610747,
  -0.384177,
  1.881508,
  -1.464807,
  0.665469,
  -6e-6,
  2.738912,
  0.810101,
  -2.415469,
  -1.057499,
  -0.416197,
  -2.357548,
  0.63003,
  0.622491,
  0.01545,
  2.038561,
  -0.133942,
  -3.096796,
  -1.465688,
  -1.199232,
  4.567061,
  3.26098,
  -0.979491,
  0.895049,
  2.049235,
  1.331015,
  0.27139,
  0.285285,
  1.20209,
  -8.206784,
  -5.805762,
  1.804431,
  -0.609065,
  -1.990902,
  0.328886,
  -1.45658,
  -0.345596,
  -0.064093,
  16.67697,
  -2.311094,
  -0.97711,
  0.675986,
  1.245136,
  0.791193,
  -0.986039,
  -0.209956,
  0.294665,
  -3548e-6,
  -2.268313,
  -0.062056,
  0.470518,
  0.865799,
  0.685628,
  -1.971736,
  -0.941405,
  -0.340056,
  1.468763,
  -1.474284,
  0.550106,
  -11e-6,
  2.35637,
  0.90017,
  -1.589845,
  -0.779708,
  -0.558224,
  -0.813738,
  0.584662,
  0.112946,
  -0.02658,
  2.707248,
  -0.211249,
  -6.940173,
  -2.823963,
  -1.620848,
  1.090696,
  2.39173,
  1.370047,
  0.589046,
  1.7284,
  1.331253,
  1.293144,
  -192e-5,
  1.644206,
  -0.866697,
  -7.161953,
  -1.385018,
  -0.150537,
  -1.388643,
  0.253012,
  -1.48888,
  -0.24955,
  -0.237714,
  11.67714,
  -0.861712,
  1.053828,
  0.199274,
  0.363356,
  0.85533,
  -1.060891,
  -0.403583,
  0.282321,
  -237e-5,
  -1.876577,
  -0.595026,
  0.424102,
  0.31408,
  0.663167,
  -1.101204,
  -0.135135,
  -4.030882,
  6.096353,
  -0.11486,
  1.606507,
  -2e-6,
  4.436084,
  0.597372,
  -1.154597,
  -0.192338,
  0.851213,
  0.293489,
  -0.065228,
  0.138908,
  0.090915,
  3.133307,
  0.210854,
  -1.031588,
  -0.15468,
  5.266214,
  -0.949139,
  -0.718487,
  4.875626,
  -0.191191,
  -2.865642,
  1.087895,
  -1.159454,
  -0.095467,
  -1.508146,
  -0.020314,
  1.040653,
  -2.333508,
  0.254059,
  8.594981,
  0.093168,
  -1.03594,
  -0.202115,
  4.719343,
  -0.901932,
  -7.858046,
  3.901234,
  -0.223314,
  -4.344739,
  0.655073,
  -1.096669,
  -0.15582,
  2.057553,
  6.274495,
  -2.678352,
  -1.814927,
  0.155068,
  1.903276,
  0.499899,
  -1.114209,
  -0.147353,
  -7.602914,
  8.973685,
  -0.049801,
  1.289198,
  0.083669,
  4.557987,
  0.611876,
  -1.149397,
  -0.198163,
  4.914096,
  -3.498986,
  -0.062571,
  0.16674,
  0.104898,
  2.284689,
  0.593597,
  -1.056121,
  -0.145617,
  0.427266,
  2.912649,
  -0.550174,
  4.406542,
  -0.138768,
  1.245555,
  0.973301,
  -1.125047,
  -0.040037,
  1.058457,
  -3.462236,
  0.439528,
  -2.395805,
  0.517759,
  4.866247,
  0.425319,
  -1.051444,
  -0.280454,
  3.364668,
  3.293787,
  -10.15741,
  3.807407,
  -0.359238,
  -3.367415,
  0.790083,
  -1.093847,
  -0.143697,
  2.38478,
  5.78707,
  -2.445987,
  -1.311171,
  0.232656,
  1.158439,
  0.555542,
  -1.134824,
  -0.168047,
  -3.32562,
  4.458596,
  -0.113506,
  1.1045,
  0.077945,
  4.609952,
  0.685485,
  -1.143017,
  -0.156593,
  0.301469,
  -0.176303,
  -0.035579,
  -0.234241,
  0.252871,
  5.884085,
  0.47506,
  -1.136801,
  -0.29075,
  3.682423,
  -0.40612,
  -0.872816,
  4.00151,
  -0.15222,
  -5.528713,
  1.044847,
  -1.063652,
  0.078081,
  -1.983678,
  0.364808,
  2.102276,
  -3.06505,
  0.843195,
  10.3883,
  0.266283,
  -1.061015,
  -0.285981,
  4.223615,
  -2.290138,
  -8.31401,
  4.405718,
  -0.461363,
  -4.50291,
  1.008383,
  -1.106302,
  -0.169712,
  2.087196,
  8.238929,
  -2.992416,
  -1.821776,
  0.343486,
  0.775518,
  0.534119,
  -1.17111,
  -0.21063,
  -1.614361,
  2.378103,
  -0.162597,
  0.850448,
  0.105931,
  4.046256,
  0.661823,
  -1.20048,
  -0.223573,
  1.01439,
  -1.174074,
  -0.444018,
  0.226241,
  0.166587,
  5.461829,
  0.567631,
  -1.223587,
  -0.350262,
  1.699106,
  0.672427,
  1.268567,
  2.135102,
  804e-6,
  -5.221111,
  0.944569,
  -0.945267,
  0.146846,
  -1.335034,
  4.346628,
  -12.85652,
  -1.807046,
  0.817524,
  9.301065,
  0.36568,
  -1.134681,
  -0.331095,
  3.571244,
  -2.208948,
  6.04158,
  3.107577,
  -0.311213,
  -4.186351,
  0.918833,
  -1.083237,
  -0.183139,
  2.062654,
  1.385424,
  -5.00495,
  -1.332669,
  0.362735,
  0.332315,
  0.619118,
  -1.211527,
  -0.259062,
  -0.166087,
  0.36279,
  -0.103926,
  0.469792,
  0.167165,
  3.507497,
  0.602251,
  -1.433017,
  -0.473359,
  0.172444,
  0.995324,
  -1.874457,
  0.44321,
  0.017158,
  2.339272,
  0.644147,
  -1.08492,
  -0.15879,
  0.899958,
  -2.537516,
  5.877859,
  2.014554,
  0.096891,
  0.317724,
  0.90304,
  -1.008242,
  2793e-6,
  -0.350747,
  10.283,
  -20.804541,
  -2.781026,
  0.899509,
  3.366951,
  0.347387,
  -1.103151,
  -0.27996,
  2.525791,
  -4.255704,
  9.903388,
  3.722668,
  -0.360394,
  -1.303292,
  0.936945,
  -1.102235,
  -0.202506,
  2.08566,
  1.686787,
  -5.010957,
  -1.656458,
  0.458403,
  -0.275176,
  0.618416,
  -1.25613,
  -0.31049,
  0.163935,
  0.13155,
  -0.729758,
  0.477848,
  0.125926,
  3.012108,
  0.620273,
  -1.620114,
  -0.655267,
  -0.287716,
  1.094371,
  0.281891,
  0.369683,
  0.094285,
  1.450951,
  0.568131,
  -0.96862,
  -0.037556,
  1.46998,
  -3.103414,
  2.856583,
  1.883209,
  -0.057461,
  1.286383,
  1.001751,
  -1.089377,
  -0.102306,
  -1.498891,
  10.66455,
  -17.201839,
  -2.759314,
  1.061258,
  2.910211,
  0.26247,
  -1.044681,
  -0.215686,
  3.230136,
  -0.586386,
  6.09664,
  3.550019,
  -0.425577,
  -1.500033,
  0.96877,
  -1.133658,
  -0.25051,
  1.71784,
  848e-5,
  -5.011789,
  -1.740989,
  0.498343,
  -0.208183,
  0.608864,
  -1.335366,
  -0.386332,
  -0.527997,
  0.363832,
  0.32307,
  0.083397,
  0.248329,
  2.678646,
  0.499835,
  -2.004511,
  -0.995712,
  1.250807,
  0.01625,
  -0.341075,
  0.785824,
  -0.095068,
  0.026519,
  0.578864,
  -0.871416,
  0.119205,
  -0.848688,
  -0.37025,
  1.818277,
  1.103427,
  0.245487,
  3.841575,
  0.984735,
  -1.042618,
  -0.228579,
  0.362018,
  2.983368,
  -9.776844,
  -1.971587,
  0.669167,
  -0.790195,
  0.32132,
  -1.099112,
  -0.186987,
  2.044065,
  2.062964,
  1.265668,
  2.71013,
  -0.109944,
  0.217935,
  0.902411,
  -1.106985,
  -0.239688,
  1.809807,
  8.523319,
  -5.011788,
  -1.590086,
  0.324845,
  -0.100319,
  0.655061,
  -1.421285,
  -0.476702,
  -0.3885,
  0.827459,
  -0.364423,
  0.699951,
  0.051967,
  2.578431,
  0.624631,
  -2.611217,
  -1.398846,
  0.452742,
  -0.593214,
  0.222462,
  -0.559358,
  0.338963,
  -0.776711,
  0.06536,
  -0.988154,
  0.046848,
  -0.861661,
  0.879981,
  4.00313,
  1.739543,
  -0.080984,
  5.524802,
  1.499673,
  -0.754476,
  -0.231481,
  0.812577,
  -0.772413,
  -9.577645,
  -1.629433,
  0.679083,
  -4.193895,
  -0.025266,
  -1.273719,
  -0.218703,
  1.401798,
  5.231832,
  0.740509,
  1.775166,
  -0.072695,
  1.996087,
  1.05745,
  -1.046864,
  -0.224756,
  1.679449,
  11.40057,
  -4.948829,
  -1.182664,
  0.324104,
  -0.247001,
  0.61159,
  -1.514607,
  -0.598543,
  -0.187761,
  1.75693,
  -1.314206,
  0.611581,
  -6e-6,
  2.412975,
  0.81243,
  -2.308414,
  -1.083797,
  -0.117996,
  -1.728246,
  0.778474,
  0.549451,
  6203e-6,
  0.932625,
  -0.141952,
  -3.230837,
  -1.43867,
  -0.986829,
  2.974393,
  1.949339,
  -0.633786,
  0.816027,
  3.278606,
  1.354373,
  0.514938,
  0.275479,
  1.040965,
  -4.501186,
  -3.399057,
  0.966186,
  -0.473617,
  -4.037574,
  0.279485,
  -1.62187,
  -0.319276,
  0.878624,
  9.785565,
  -2.727652,
  0.019037,
  0.552126,
  2.138764,
  0.841987,
  -0.99517,
  -0.255061,
  1.498952,
  -2737e-6,
  -3.101832,
  -0.592133,
  0.286442,
  -0.440522,
  0.663141,
  -1.902954,
  -0.905692,
  -0.206957,
  1.191499,
  -1.092577,
  0.584956,
  -1e-5,
  2.048407,
  0.900153,
  -1.271627,
  -0.719392,
  -0.011366,
  -0.116795,
  3286e-6,
  -0.052628,
  -0.024739,
  1.716125,
  -0.218713,
  -7.647175,
  -3.114129,
  -1.490128,
  -0.526649,
  3.06309,
  1.474262,
  0.548146,
  2.052174,
  1.353089,
  2.191403,
  0.342112,
  1.44651,
  2.170943,
  -7.768187,
  -1.471207,
  -0.145671,
  -1.753574,
  0.231058,
  -1.932296,
  -0.381474,
  0.624542,
  6.748294,
  -0.306017,
  1.067747,
  0.250067,
  -0.12526,
  0.861461,
  -0.94711,
  -0.405264,
  1.300174,
  -3952e-6,
  -1.908284,
  -0.538572,
  0.213358,
  -0.625029,
  0.665801
]);
var radiances_b = new Float32Array([
  0.992652,
  1.999494,
  -4.136109,
  18.5627,
  13.51028,
  13.90238,
  0.963437,
  2.119694,
  -4.614523,
  19.19701,
  13.76644,
  14.18731,
  0.944654,
  2.17161,
  -4.915556,
  19.1824,
  15.37135,
  14.0053,
  0.907307,
  2.330536,
  -5.577596,
  19.61615,
  16.88365,
  14.46955,
  0.873912,
  2.388682,
  -5.842995,
  19.232651,
  18.87735,
  14.85698,
  0.856369,
  2.391534,
  -5.769133,
  18.28709,
  20.97209,
  14.69587,
  0.827053,
  2.34279,
  -5.558071,
  16.84993,
  23.56498,
  15.05975,
  0.790834,
  2.190341,
  -4.852571,
  13.74862,
  28.06846,
  15.48444,
  0.740362,
  1.783998,
  -2.983854,
  7.622563,
  35.076099,
  16.158051,
  0.684011,
  1.154457,
  -0.239383,
  -0.789689,
  42.827648,
  17.794689,
  1.1683,
  1.860993,
  -2.129074,
  12.51952,
  30.324989,
  29.387159,
  1.150338,
  1.918813,
  -2.413527,
  12.74862,
  30.87134,
  29.51432,
  1.114719,
  1.964689,
  -2.625423,
  12.47837,
  32.37949,
  29.435961,
  1.077948,
  2.006292,
  -2.846934,
  11.90195,
  34.59293,
  29.37492,
  1.035143,
  1.986681,
  -2.752584,
  10.60972,
  37.221851,
  29.18594,
  1.015992,
  1.992054,
  -2.812626,
  10.01416,
  38.473,
  29.246241,
  0.975689,
  1.939897,
  -2.533281,
  8.319176,
  40.839069,
  29.255859,
  0.926416,
  1.716454,
  -1.597044,
  4.739725,
  45.076832,
  28.78915,
  0.859519,
  1.346034,
  -0.028019,
  -0.658291,
  50.175228,
  28.52953,
  0.775412,
  0.770925,
  2.200201,
  -7.487661,
  54.366219,
  28.93432
]);
var solar_radiances_b = new Float32Array([
  234451.921875,
  230917.4375,
  225244.546875,
  216234.015625,
  206988.046875,
  200632.328125,
  190644.5625,
  175407.375,
  153216.40625,
  123259.976562
]);
var radiances_g = new Float32Array([
  1.59033,
  1.355401,
  1.151412,
  13.59116,
  5.857714,
  8.090833,
  1.55254,
  1.51004,
  0.127641,
  16.046431,
  5.912162,
  8.350009,
  1.470871,
  1.880464,
  -1.865398,
  20.308081,
  5.471461,
  9.109834,
  1.356563,
  2.373866,
  -4.653245,
  25.709221,
  5.686009,
  10.0948,
  1.244232,
  2.851519,
  -7.130942,
  29.93449,
  6.38212,
  11.14578,
  1.173693,
  3.120604,
  -8.491886,
  31.87393,
  7.290615,
  11.80066,
  1.091845,
  3.368888,
  -9.722083,
  32.685081,
  10.32424,
  12.36508,
  0.985898,
  3.500541,
  -10.26328,
  30.92956,
  16.10881,
  13.31222,
  0.886499,
  3.172888,
  -8.68755,
  23.621611,
  26.21851,
  14.74967,
  0.794697,
  2.189355,
  -4.207953,
  9.399091,
  40.62849,
  16.81753,
  1.711696,
  1.657311,
  0.932802,
  13.1788,
  15.06751,
  18.635559,
  1.666968,
  1.849993,
  -0.20886,
  15.86653,
  14.8688,
  19.40719,
  1.584846,
  2.170022,
  -2.019597,
  19.70826,
  14.90684,
  20.45055,
  1.469412,
  2.524017,
  -4.197267,
  23.652491,
  16.64588,
  21.34477,
  1.369714,
  2.843548,
  -6.059031,
  26.34993,
  18.81361,
  22.321859,
  1.310477,
  2.984444,
  -6.831686,
  26.8234,
  21.23267,
  22.597549,
  1.222552,
  3.176523,
  -7.731496,
  26.7176,
  24.84358,
  23.368629,
  1.115781,
  3.130635,
  -7.581744,
  23.365311,
  31.71048,
  24.13859,
  1.013181,
  2.699342,
  -5.602709,
  15.00158,
  42.176128,
  25.159571,
  0.897632,
  1.726948,
  -1.29612,
  1.183675,
  55.03215,
  26.43066
]);
var solar_radiances_g = new Float32Array([
  503392.1875,
  494736.96875,
  480889.4375,
  459145.1875,
  437110.40625,
  422110.4375,
  398753.53125,
  363654.46875,
  313553.90625,
  247504.40625
]);
var radiances_r = new Float32Array([
  1.962684,
  1.159831,
  4.450588,
  5.079633,
  4.437388,
  4.324573,
  1.946487,
  1.287515,
  3.703696,
  8.782833,
  3.440437,
  5.160333,
  1.88217,
  1.335878,
  2.648641,
  13.58368,
  3.105473,
  5.907387,
  1.738159,
  1.624289,
  -8787e-6,
  21.182529,
  2.770255,
  7.055672,
  1.571896,
  2.301786,
  -4.028545,
  29.66806,
  1.630876,
  8.711031,
  1.475048,
  2.679086,
  -6.311315,
  33.778961,
  2.140975,
  9.385283,
  1.326174,
  3.378759,
  -9.831444,
  39.420609,
  2.852702,
  10.82542,
  1.153344,
  3.967771,
  -12.65181,
  41.950161,
  7.468239,
  12.2135,
  0.974608,
  4.051626,
  -12.98454,
  37.549641,
  17.492319,
  14.20619,
  0.844802,
  3.181809,
  -8.757338,
  21.97962,
  35.24033,
  16.395491,
  2.029623,
  1.364434,
  4.201529,
  5.415099,
  9.825839,
  10.63328,
  2.023126,
  1.494728,
  3.420413,
  9.072178,
  9.205157,
  11.86639,
  1.956307,
  1.648665,
  2.039712,
  14.30239,
  9.039526,
  13.30453,
  1.825053,
  1.985022,
  -0.803631,
  22.024929,
  9.415361,
  15.17659,
  1.650367,
  2.593201,
  -4.469328,
  29.69817,
  9.410977,
  17.4485,
  1.555202,
  2.962925,
  -6.60817,
  33.29887,
  10.64559,
  18.50816,
  1.412478,
  3.439403,
  -9.196616,
  36.850769,
  13.45341,
  20.031281,
  1.25299,
  3.820805,
  -11.15338,
  37.215931,
  20.14916,
  21.8232,
  1.091952,
  3.663027,
  -10.3133,
  29.78985,
  32.968349,
  23.754499,
  0.950169,
  2.664579,
  -5.545167,
  12.81159,
  51.54768,
  25.74284
]);
var solar_radiances_r = new Float32Array([
  796325.9375,
  783219.8125,
  762242.5625,
  729388.375,
  696105.375,
  673463.0625,
  638143.3125,
  585026.4375,
  508931.3125,
  407766.5
]);

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
  const isInsecure = window.isSecureContext === false;
  throw new Error(
    isInsecure ? "WebGPU requires a secure context (HTTPS or localhost). Current origin is not secure." : "WebGPU not supported on this browser."
  );
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
var perFrameBindGroup;
var dataBindGroup;
var passBindGroup;
function resizeCanvas() {
  const pixelRatio = controls.highDPI ? window.devicePixelRatio : 1;
  canvas.width = window.innerWidth * pixelRatio;
  canvas.height = window.innerHeight * pixelRatio;
  width = canvas.width;
  height = canvas.height;
  if (raytracedTexture) {
    createGPUResources();
    updatePixelRadius();
  }
}
resizeCanvas();
window.addEventListener("resize", resizeCanvas);
highDPIController.onChange(() => {
  resizeCanvas();
  updatePixelRadius();
});
var timestampQueryFeature = "timestamp-query";
var supportsTimestampQueries = adapter?.features.has(timestampQueryFeature);
var requiredFeatures = [];
if (supportsTimestampQueries) {
  requiredFeatures.push(timestampQueryFeature);
}
var device = await adapter.requestDevice({ requiredFeatures });
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
  label: "Display vertices",
  size: vertices.byteLength,
  // 4 bytes * 6 vertices = 24 bytes.
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
      { binding: 0, resource: raytracedTexture.createView() },
      { binding: 1, resource: displaySampler }
    ]
  });
  perFrameBindGroup = device.createBindGroup({
    label: "Per-frame bind group",
    layout: perFrameBindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: inputBuffer } },
      { binding: 1, resource: { buffer: objectsBuffer } },
      { binding: 2, resource: { buffer: skyStateBuffer } }
    ]
  });
  dataBindGroup = device.createBindGroup({
    label: "Data bind group",
    layout: dataBindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: gridsBuffer } },
      { binding: 1, resource: { buffer: rootsBuffer } },
      { binding: 2, resource: { buffer: uppersBuffer } },
      { binding: 3, resource: { buffer: lowersBuffer } },
      { binding: 4, resource: { buffer: leavesBuffer } },
      { binding: 5, resource: { buffer: dataBuffer } }
    ]
  });
  passBindGroup = device.createBindGroup({
    label: "Pass bind group",
    layout: passBindGroupLayout,
    entries: [
      { binding: 0, resource: raytracedTexture.createView() }
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
var initialCameraTarget = vec3.create(0, 0, 0);
var camera = createOrbitCamera({
  position: initialCameraPosition,
  target: initialCameraTarget
});
controls.resetCamera = () => {
  camera = createOrbitCamera({
    position: initialCameraPosition,
    target: initialCameraTarget
  });
};
var inputValues = new ArrayBuffer(80);
var inputViews = {
  camera_matrix: new Float32Array(inputValues, 0, 16),
  fov_scale: new Float32Array(inputValues, 64, 1),
  time_delta: new Float32Array(inputValues, 68, 1),
  pixel_radius: new Float32Array(inputValues, 72, 1),
  debug_iterations: new Uint32Array(inputValues, 76, 1)
};
var inputBuffer = device.createBuffer({
  label: "Input Uniforms",
  size: inputValues.byteLength,
  usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
});
inputViews.fov_scale[0] = fovScaled;
var OBJECT_STRUCT_SIZE = 144;
var OBJECT_COUNT = 2;
var objectsData = new ArrayBuffer(OBJECT_STRUCT_SIZE * OBJECT_COUNT);
var objectsBuffer = device.createBuffer({
  label: "Objects",
  size: objectsData.byteLength,
  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
});
new Array(27).slice();
var objectViews = [];
for (let index = 0; index < OBJECT_COUNT; index++) {
  const offset = OBJECT_STRUCT_SIZE * index;
  objectViews.push({
    object_type: new Uint32Array(objectsData, offset + 0, 1),
    type_index: new Uint32Array(objectsData, offset + 4, 1),
    material_index: new Uint32Array(objectsData, offset + 8, 1),
    _pad: new Uint32Array(objectsData, offset + 12, 1),
    transform: new Float32Array(objectsData, offset + 16, 16),
    transform_inverse: new Float32Array(objectsData, offset + 80, 16)
  });
}
var bunnyObjectView = objectViews[0];
bunnyObjectView.object_type[0] = 1;
bunnyObjectView.type_index[0] = 0;
bunnyObjectView.material_index[0] = 0;
var groundObjectView = objectViews[1];
groundObjectView.object_type[0] = 2;
groundObjectView.type_index[0] = 0;
groundObjectView.material_index[0] = 1;
groundObjectView.transform.set(mat4.translation(vec3.create(0, 2, 0)));
groundObjectView.transform_inverse.set(mat4.translation(vec3.create(0, -2, 0)));
var sunZenith = 30 * Math.PI / 180;
var sunAzimuth = 0;
var sunDirection = vec3.create(
  Math.sin(sunZenith) * Math.cos(sunAzimuth),
  Math.cos(sunZenith),
  -Math.sin(sunZenith) * Math.sin(sunAzimuth)
);
var skyState = createSkyState({
  elevation: 0.5 * Math.PI - sunZenith,
  turbidity: 2,
  albedo: [0.3, 0.3, 0.3]
});
var skyStateData = new ArrayBuffer(144);
var skyStateBuffer = device.createBuffer({
  label: "SkyState",
  size: skyStateData.byteLength,
  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
});
var skyStateView = {
  sunDirection: new Float32Array(skyStateData, 0, 3),
  params: new Float32Array(skyStateData, 12, 27),
  skyRadiances: new Float32Array(skyStateData, 120, 3),
  solarRadiances: new Float32Array(skyStateData, 132, 3)
};
skyStateView.sunDirection.set(sunDirection);
skyStateView.params.set(skyState.params);
skyStateView.skyRadiances.set(skyState.skyRadiances);
skyStateView.solarRadiances.set(skyState.solarRadiances);
console.log("SKY STATE", skyState);
device.queue.writeBuffer(skyStateBuffer, 0, skyStateData);
function computePixelRadius(fov_y_radians, resolution_height) {
  const fov_scale = Math.tan(fov_y_radians * 0.5);
  return 2 * fov_scale / resolution_height;
}
function updatePixelRadius() {
  inputViews.pixel_radius[0] = computePixelRadius(fov, height);
}
updatePixelRadius();
function updateObjects() {
  const transformMatrix = mat4.identity();
  mat4.translation(vec3.create(-40, 240, 0), transformMatrix);
  mat4.scale(transformMatrix, vec3.create(120, 120, 120), transformMatrix);
  const rotationRadians = controls.bunnyRotation * Math.PI / 180;
  mat4.rotateY(transformMatrix, rotationRadians, transformMatrix);
  bunnyObjectView.transform.set(transformMatrix);
  bunnyObjectView.transform_inverse.set(mat4.inverse(transformMatrix));
  device.queue.writeBuffer(objectsBuffer, 0, objectsData);
  device.queue.writeBuffer(skyStateBuffer, 0, skyStateData);
}
updateObjects();
rotationController.onChange(() => {
  updateObjects();
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
  inputViews.time_delta[0] = deltaTime;
  inputViews.debug_iterations[0] = controls.debugIterations ? 1 : 0;
  camera.update(deltaTime, inputHandler());
  inputViews.camera_matrix.set(camera.matrix);
  device.queue.writeBuffer(inputBuffer, 0, inputValues);
}
var combinedShader = (
  /* wgsl */
  `// Hello GPU
${picovdb_default}
${compute_default}`
);
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
var perFrameBindGroupLayout = device.createBindGroupLayout({
  label: "Per-frame Bind Group Layout",
  entries: [
    { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
    { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
    { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } }
  ]
});
var dataBindGroupLayout = device.createBindGroupLayout({
  label: "Data Bind Group Layout",
  entries: [
    { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
    { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
    { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
    { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
    { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
    { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } }
  ]
});
var passBindGroupLayout = device.createBindGroupLayout({
  label: "Pass Bind Group Layout",
  entries: [
    {
      binding: 0,
      visibility: GPUShaderStage.COMPUTE,
      storageTexture: { access: "write-only", format: "rgba8unorm", viewDimension: "2d" }
    }
  ]
});
var computePipelineLayout = device.createPipelineLayout({
  label: "Compute Pipeline Layout",
  bindGroupLayouts: [perFrameBindGroupLayout, dataBindGroupLayout, passBindGroupLayout]
});
var computePipeline = await device.createComputePipelineAsync({
  label: "Compute Pipeline",
  layout: computePipelineLayout,
  compute: { module: computeShaderModule, entryPoint: "computeMain" }
}).catch((error) => {
  console.error("Pipeline creation failed:", error);
  alert(`Pipeline error: ${error.message}`);
  throw error;
});
console.log("Pipeline created.");
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
  computePass.setBindGroup(0, perFrameBindGroup);
  computePass.setBindGroup(1, dataBindGroup);
  computePass.setBindGroup(2, passBindGroup);
  computePass.dispatchWorkgroups(Math.ceil(width / 8), Math.ceil(height / 8), 1);
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
