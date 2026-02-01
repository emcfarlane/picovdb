

export default interface Input {
  readonly digital: {
    readonly forward: boolean;
    readonly backward: boolean;
    readonly left: boolean;
    readonly right: boolean;
    readonly up: boolean;
    readonly down: boolean;
  };
  readonly analog: {
    readonly x: number;    // Delta X
    readonly y: number;    // Delta Y
    readonly zoom: number; // Delta Zoom
    readonly touching: boolean;
    readonly panning: boolean; // True if gesture/key implies panning
  };
}

export type InputHandler = () => Input;

export function createInputHandler(window: Window, canvas: HTMLCanvasElement): InputHandler {
  const digital = { forward: false, backward: false, left: false, right: false, up: false, down: false };
  const analog = { x: 0, y: 0, zoom: 0, touching: false, panning: false };

  // State
  const pointers = new Map<number, PointerEvent>();
  let prevDist = 0;
  let isAlt = false;

  // Keyboard
  const setKey = (e: KeyboardEvent, v: boolean) => {
    if (e.key === 'Alt') isAlt = v;
    switch (e.code) {
      case 'KeyW': digital.forward = v; break;
      case 'KeyS': digital.backward = v; break;
      case 'KeyA': digital.left = v; break;
      case 'KeyD': digital.right = v; break;
      case 'Space': digital.up = v; break;
      case 'ShiftLeft': digital.down = v; break;
    }
  };
  window.addEventListener('keydown', e => setKey(e, true));
  window.addEventListener('keyup', e => setKey(e, false));

  // Pointer (Touch/Mouse)
  canvas.style.touchAction = 'none'; // Prevent browser scroll

  canvas.addEventListener('pointerdown', e => {
    canvas.setPointerCapture(e.pointerId);
    pointers.set(e.pointerId, e);
    if (pointers.size === 2) {
      // Init pinch
      const p = [...pointers.values()];
      prevDist = Math.hypot(p[0].clientX - p[1].clientX, p[0].clientY - p[1].clientY);
    }
  });

  canvas.addEventListener('pointerup', e => {
    canvas.releasePointerCapture(e.pointerId);
    pointers.delete(e.pointerId);
  });

  canvas.addEventListener('pointermove', e => {
    if (!pointers.has(e.pointerId)) return;
    pointers.set(e.pointerId, e); // Update cache

    // Multi-touch (Pinch + Pan)
    if (pointers.size === 2) {
      const p = [...pointers.values()];
      const dist = Math.hypot(p[0].clientX - p[1].clientX, p[0].clientY - p[1].clientY);

      analog.zoom += (dist - prevDist) * 0.01; // Pinch Zoom
      analog.x += e.movementX;                 // 2-finger Pan
      analog.y += e.movementY;
      analog.panning = true;
      prevDist = dist;
    }
    // Single Touch / Mouse
    else if (pointers.size === 1) {
      analog.x += e.movementX;
      analog.y += e.movementY;
      // Pan if Middle Mouse (4) or Alt held
      analog.panning = (e.buttons & 4) !== 0 || isAlt;
    }
  });

  // Wheel Zoom
  canvas.addEventListener('wheel', e => {
    e.preventDefault();
    analog.zoom -= Math.sign(e.deltaY);
  }, { passive: false });

  return () => {
    const out = {
      digital,
      analog: { ...analog, touching: pointers.size > 0 }
    };
    // Reset deltas
    analog.x = 0; analog.y = 0; analog.zoom = 0; analog.panning = false;
    return out;
  };
}
