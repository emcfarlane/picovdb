import type { Mat4 } from 'wgpu-matrix';
import { mat4, quat, vec3 } from 'wgpu-matrix';
import type Input from './input.js';

export interface ModelHand {
	/** Consumes drag/pan input; returns true if the model moved this frame. */
	update(dt: number, input: Input, camera: Mat4): boolean;
	/** World->model-world transform to compose into the object's
	 * world->index transform: rotate about the model pivot, then un-pan. */
	readonly transform: Mat4;
	reset(): void;
}

/**
 * "Model in hand" controller: the camera and backdrop stay fixed while drag
 * rotates the model (trackball about the camera's up/right axes) and
 * alt/middle/two-finger drag moves it in the camera plane. Zoom is left to
 * the camera (a dolly along its fixed view direction).
 */
export function createModelHand(): ModelHand {
	// Accumulated INVERSE model rotation (the world->index transform needs
	// the pre-image of a world point, so drags append inverse rotations).
	let targetQuat = quat.identity();
	const currentQuat = quat.identity();
	let targetOffset = vec3.create();
	const currentOffset = vec3.create();

	const transform_ = mat4.identity();
	const rotationSpeed = 0.008; // rad / pixel
	const smoothing = 0.15;      // 0 = instant, matches the camera feel

	const tmpQuat = quat.identity();
	const tmpMat = mat4.identity();

	function recalc() {
		// M_hand = R_inv * T(-offset): pre-image of "translate then rotate"
		mat4.fromQuat(currentQuat, transform_);
		mat4.translation(vec3.negate(currentOffset, vec3.create()), tmpMat);
		mat4.multiply(transform_, tmpMat, transform_);
	}
	recalc();

	return {
		get transform() { return transform_; },

		reset() {
			targetQuat = quat.identity();
			quat.identity(currentQuat);
			targetOffset = vec3.create();
			vec3.set(0, 0, 0, currentOffset);
			recalc();
		},

		update(dt: number, input: Input, camera: Mat4): boolean {
			const { x: dx, y: dy, panning, touching } = input.analog;
			const right = vec3.create(camera[0], camera[1], camera[2]);
			const up = vec3.create(camera[4], camera[5], camera[6]);

			if (panning && (dx || dy)) {
				// Move the model in the camera plane, scaled by distance
				const dist = Math.max(0.5, Math.hypot(camera[12], camera[13], camera[14]));
				const speed = dist * 0.002;
				vec3.addScaled(targetOffset, right, dx * speed, targetOffset);
				vec3.addScaled(targetOffset, up, -dy * speed, targetOffset);
			} else if (touching && (dx || dy)) {
				// Trackball: drag right spins about camera-up, drag down tips
				// about camera-right. Append inverse rotations on the right.
				quat.fromAxisAngle(right, -dy * rotationSpeed, tmpQuat);
				targetQuat = quat.mul(targetQuat, tmpQuat, targetQuat);
				quat.fromAxisAngle(up, -dx * rotationSpeed, tmpQuat);
				targetQuat = quat.mul(targetQuat, tmpQuat, targetQuat);
				// Keep the target unit-length: norm drift from accumulated
				// multiplications would push |dot| permanently away from 1
				// and the settle snap below could never fire
				quat.normalize(targetQuat, targetQuat);
			}

			// Smooth toward targets (same feel as the camera). The settled
			// test must be exact component equality with a snap-to-target:
			// slerp/lerp asymptote in f32 and would otherwise stay "dirty"
			// forever, resetting accumulation every frame.
			const t = 1 - Math.pow(smoothing, dt * 60);
			let dirty = false;
			if (currentQuat[0] !== targetQuat[0] || currentQuat[1] !== targetQuat[1] ||
				currentQuat[2] !== targetQuat[2] || currentQuat[3] !== targetQuat[3]) {
				if (Math.abs(1 - Math.abs(quat.dot(currentQuat, targetQuat))) < 1e-6) {
					quat.copy(targetQuat, currentQuat);
				} else {
					quat.slerp(currentQuat, targetQuat, t, currentQuat);
					quat.normalize(currentQuat, currentQuat);
				}
				dirty = true;
			}
			if (currentOffset[0] !== targetOffset[0] || currentOffset[1] !== targetOffset[1] ||
				currentOffset[2] !== targetOffset[2]) {
				const offsetDiff = vec3.sub(targetOffset, currentOffset, vec3.create());
				// Snap threshold must exceed the f32 rounding residue at scene
				// scale (~1.5e-5/component at |offset| ~ 240): at large frame
				// times the lerp lands within rounding of the target and a
				// 1e-10 threshold never fires -> dirty forever -> accumulation
				// resets every frame (dt-dependent; seen with the denoiser on
				// slow machines). 1e-6 lenSq = 1e-3 world units, sub-pixel.
				if (vec3.lenSq(offsetDiff) < 1e-6) {
					vec3.copy(targetOffset, currentOffset);
				} else {
					vec3.addScaled(currentOffset, offsetDiff, t, currentOffset);
				}
				dirty = true;
			}
			if (dirty) recalc();
			return dirty;
		},
	};
}
