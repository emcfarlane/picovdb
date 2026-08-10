// Precomputed diffuse sky irradiance.
//
// Projects the Hosek-Wilkie sky (sun disk excluded) into second-order
// spherical harmonics and applies the Lambertian cosine convolution
// (Ramamoorthi & Hanrahan, "An Efficient Representation for Irradiance
// Environment Maps"). The shader then evaluates per-pixel irradiance with
// 9 fused multiply-adds instead of sampling the sky model per pixel.

import { skyStateRadiance, type Channel } from './hw_skymodel.ts';

const SOLAR_RADIUS_RADIANS = 0.004450589; // must match hw_skymodel.ts

interface SkyStateLike {
	params: Float32Array;
	skyRadiances: Float32Array;
	solarRadiances: Float32Array;
}

/**
 * Computes SH-2 sky irradiance coefficients in polynomial form:
 *
 *   E(n) = A0 + A1*y + A2*z + A3*x + A4*xy + A5*yz
 *        + A6*(3z^2 - 1) + A7*xz + A8*(x^2 - y^2)
 *
 * Returns 9 vec4s (xyz = RGB, w = 0), matching SkyState.irradianceSH in
 * demo/compute.wgsl. The sun disk is excluded — direct sun light is handled
 * by shadow-traced direct lighting, and its radiance spike would swamp SH-2.
 */
export function computeSkyIrradianceSH(
	state: SkyStateLike,
	sunDirection: ArrayLike<number>,
	sampleCount = 2048
): Float32Array {
	const L = new Float64Array(27); // 9 SH coefficients x 3 channels
	const sx = sunDirection[0];
	const sy = sunDirection[1];
	const sz = sunDirection[2];

	// Fibonacci sphere sampling over the full sphere. The sky model mirrors
	// below the horizon via abs(cos(theta)), matching the shader's radiance().
	const golden = Math.PI * (3.0 - Math.sqrt(5.0));
	for (let i = 0; i < sampleCount; i++) {
		const y = 1.0 - (2.0 * i + 1.0) / sampleCount;
		const r = Math.sqrt(Math.max(0.0, 1.0 - y * y));
		const phi = golden * i;
		const x = r * Math.cos(phi);
		const z = r * Math.sin(phi);

		const theta = Math.acos(Math.min(1.0, Math.max(-1.0, y)));
		const cosGamma = Math.min(1.0, Math.max(-1.0, x * sx + y * sy + z * sz));
		const gamma = Math.acos(cosGamma);
		const inSunDisk = gamma <= SOLAR_RADIUS_RADIANS;

		// Real SH basis evaluated at the sample direction
		const sh = [
			0.282095,
			0.488603 * y,
			0.488603 * z,
			0.488603 * x,
			1.092548 * x * y,
			1.092548 * y * z,
			0.315392 * (3.0 * z * z - 1.0),
			1.092548 * x * z,
			0.546274 * (x * x - y * y),
		];

		for (let ch = 0; ch < 3; ch++) {
			// skyStateRadiance includes the solar disk; subtract it back out.
			let radiance = skyStateRadiance(state, theta, gamma, ch as Channel);
			if (inSunDisk) {
				radiance -= state.solarRadiances[ch];
			}
			for (let c = 0; c < 9; c++) {
				L[c * 3 + ch] += radiance * sh[c];
			}
		}
	}
	// Monte-Carlo estimator weight: uniform sphere pdf = 1 / (4*pi)
	const w = (4.0 * Math.PI) / sampleCount;

	// Lambertian cosine convolution weights per SH band, with the SH basis
	// constants folded in for the polynomial form.
	const A0 = Math.PI;
	const A1 = (2.0 * Math.PI) / 3.0;
	const A2 = Math.PI / 4.0;
	const poly = [
		A0 * 0.282095,
		A1 * 0.488603,
		A1 * 0.488603,
		A1 * 0.488603,
		A2 * 1.092548,
		A2 * 1.092548,
		A2 * 0.315392,
		A2 * 1.092548,
		A2 * 0.546274,
	];

	const out = new Float32Array(36); // 9 x vec4 (w unused)
	for (let c = 0; c < 9; c++) {
		for (let ch = 0; ch < 3; ch++) {
			out[c * 4 + ch] = L[c * 3 + ch] * w * poly[c];
		}
	}
	return out;
}
