import type { Mat4, Vec3 } from 'wgpu-matrix';
import { mat4, vec3 } from 'wgpu-matrix';
import type Input from './input.js';

export interface OrbitCamera {
	update(dt: number, input: Input): Mat4;
	matrix: Mat4;      // Camera-to-World
	view: Mat4;        // World-to-Camera
	position: Vec3;
	pivot: Vec3;       // The target point we orbit
}

/**
 * Creates an Orbit camera (Turntable style) with smooth interpolation.
 * Controls:
 * - Drag: Orbit (Rotate world)
 * - Alt+Drag / Middle Mouse / 2-Finger: Pan
 * - Scroll / Pinch: Zoom
 */
export function createOrbitCamera(options?: {
	position?: Vec3;
	target?: Vec3;
}): OrbitCamera {
	const matrix_ = new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);
	const view_ = mat4.create();

	// Aliases for direct matrix access
	const right_ = new Float32Array(matrix_.buffer, 0, 4);
	const up_ = new Float32Array(matrix_.buffer, 16, 4);
	const position_ = new Float32Array(matrix_.buffer, 48, 4);

	// Current state (smoothly interpolates toward target)
	const pivot = options?.target ? vec3.clone(options.target) : vec3.create();
	let theta = 0, phi = 0, radius = 5;

	// Target state (input updates this directly)
	const targetPivot = vec3.clone(pivot);
	let targetTheta = 0, targetPhi = 0, targetRadius = 5;

	// Tuning: 0 = instant, 1 = never moves. ~0.1-0.3 feels smooth.
	const smoothing = 0.15;

	const temp = vec3.create();
	const upWorld = vec3.create(0, 1, 0);

	// Init from position
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
		get matrix() { return matrix_; },
		get view() { return view_; },
		get position() { return position_; },
		get pivot() { return pivot; },

		update(dt: number, input: Input): Mat4 {
			const { x: dx, y: dy, zoom: dz, panning } = input.analog;

			// Update targets from input (1-1 mapping)
			if (panning && (dx || dy)) {
				const speed = targetRadius * 0.002;
				vec3.addScaled(targetPivot, right_, -dx * speed, targetPivot);
				vec3.addScaled(targetPivot, up_, dy * speed, targetPivot);
			} else if (dx || dy) {
				const orbitSpeed = 0.005;
				targetTheta -= dx * orbitSpeed;
				targetPhi = Math.max(-1.5, Math.min(1.5, targetPhi - dy * orbitSpeed));
			}

			if (dz) {
				targetRadius *= Math.pow(1.1, dz * 0.5);
				targetRadius = Math.max(0.1, targetRadius);
			}

			// Smooth interpolation toward targets
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

			// Interpolate pivot
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
