// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Intuition Labs LLC
//
// sim.js — the Kuramoto phase-field simulation core for the org site hero.
//
// ONE source of truth for the physics: index.html imports it to render, and
// selftest.mjs imports it to prove the same numbers under node (no DOM, no
// GL, no import). N = 24x24 oscillators on a torus, nearest-neighbour
// coupling, K ramping 0 -> 2.2 over 14s then holding. Seeded PRNG, seed 11.
//
// The initial phase field is a SMOOTH low-wavenumber field (a few seeded
// Fourier modes), not white noise. Two reasons, both load-bearing:
//   1. A smooth field carries no topological defects (no vortices), so pure
//      nearest-neighbour coupling flattens it monotonically toward R -> 1 —
//      no metastable vortex ever pins R below the 0.9 sync gate.
//   2. A smooth field renders as the "faint moving hairline contours" the
//      hero calls for; white noise would render as grid hash.
// Averaged nearest-neighbour coupling ((K/4) * sum of 4 sin) is the discrete
// heat equation near sync, so R climbs monotonically once K is up.

export const GRID = 24;          // 24 x 24 = 576 oscillators
export const K_MAX = 2.2;
export const K_RAMP_S = 14;      // K ramps 0 -> K_MAX over the first 14 seconds
export const SYNC_GATE = 0.9;    // KURAMOTO_SYNC — turns the HUD white, reveals content

// mulberry32: a small, fast, fully deterministic PRNG.
export function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// createSim(seed) -> { G, N, omega, theta, t }
export function createSim(seed = 11) {
  const G = GRID;
  const N = G * G;
  const rand = mulberry32(seed);

  // natural frequencies: a small seeded spread, omega_i in [-0.09, 0.09]
  const omega = new Float32Array(N);
  for (let i = 0; i < N; i++) omega[i] = (rand() - 0.5) * 0.18;

  // four low-wavenumber Fourier modes -> a smooth, defect-free initial field
  const modes = [];
  for (let m = 0; m < 4; m++) {
    modes.push({
      kx: 1 + Math.floor(rand() * 2),   // 1 or 2 cycles across the width
      ky: 1 + Math.floor(rand() * 2),
      amp: 0.5 + rand() * 0.45,         // modest amplitude -> no winding
      ph: rand() * 2 * Math.PI,
      sx: rand() < 0.5 ? -1 : 1,
    });
  }
  const theta = new Float32Array(N);
  for (let y = 0; y < G; y++) {
    for (let x = 0; x < G; x++) {
      let v = 0;
      for (const md of modes) {
        v += md.amp * Math.sin(2 * Math.PI * (md.sx * md.kx * x / G + md.ky * y / G) + md.ph);
      }
      theta[y * G + x] = v;
    }
  }
  return { G, N, omega, theta, t: 0 };
}

// couplingAt(t) -> the ramped coupling K at simulation time t (seconds).
export function couplingAt(t) {
  return K_MAX * Math.min(t / K_RAMP_S, 1);
}

// stepSim(sim, dt) advances the field by dt seconds (forward Euler) and
// returns the coupling K used for the step.
export function stepSim(sim, dt) {
  const { G, N, omega, theta } = sim;
  const K = couplingAt(sim.t);
  const next = new Float32Array(N);
  for (let y = 0; y < G; y++) {
    const yc = y * G;
    const yu = ((y + G - 1) % G) * G;
    const yd = ((y + 1) % G) * G;
    for (let x = 0; x < G; x++) {
      const i = yc + x;
      const xl = (x + G - 1) % G;
      const xr = (x + 1) % G;
      const th = theta[i];
      const coup =
        Math.sin(theta[yc + xl] - th) +
        Math.sin(theta[yc + xr] - th) +
        Math.sin(theta[yu + x] - th) +
        Math.sin(theta[yd + x] - th);
      next[i] = th + dt * (omega[i] + (K / 4) * coup);
    }
  }
  theta.set(next);
  sim.t += dt;
  return K;
}

// orderR(sim) -> the Kuramoto order parameter R = |mean(e^{i theta})|, in [0,1].
// This is kuramotoR (phase coherence) — never routed into any other gate.
export function orderR(sim) {
  const { N, theta } = sim;
  let sx = 0, sy = 0;
  for (let i = 0; i < N; i++) {
    sx += Math.cos(theta[i]);
    sy += Math.sin(theta[i]);
  }
  return Math.hypot(sx / N, sy / N);
}
