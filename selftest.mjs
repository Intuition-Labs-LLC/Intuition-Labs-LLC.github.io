// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Intuition Labs LLC
//
// selftest.mjs — headless proof that the site hero's physics is real.
//
// It imports the SAME sim core the browser renders (sim.js — no re-implement,
// no copy) and asserts the load-bearing behaviour the HUD depends on:
//   * R crosses the 0.9 sync gate by j-time tau = 20s, and
//   * R stays above 0.8 for the rest of the run.
// Deterministic (seed 11); no network, no GPU, no wall-clock. Run:
//   node selftest.mjs

import assert from 'node:assert/strict';
import { createSim, stepSim, orderR, couplingAt, SYNC_GATE } from './sim.js';

const DT = 1 / 30;
const T_END = 24;                      // simulate a little past the 20s gate
const STEPS = Math.round(T_END / DT);

const sim = createSim(11);
const r0 = orderR(sim);
const series = [{ t: 0, r: r0 }];
let crossT = -1;

for (let s = 0; s < STEPS; s++) {
  stepSim(sim, DT);
  const r = orderR(sim);
  series.push({ t: sim.t, r });
  if (crossT < 0 && r >= SYNC_GATE) crossT = sim.t;
}

// 1) R must genuinely cross the 0.9 gate, and do it by tau = 20s.
assert.ok(crossT >= 0, 'R never reached the 0.9 sync gate');
assert.ok(crossT <= 20, `R must cross 0.9 by tau=20s (crossed at ${crossT.toFixed(3)}s)`);

// 2) once synced, R must hold above 0.8 for the remainder.
const after = series.filter((p) => p.t >= crossT);
const minAfter = Math.min(...after.map((p) => p.r));
assert.ok(minAfter > 0.8, `R must stay >0.8 after crossing (min was ${minAfter.toFixed(4)})`);

// 3) sanity: no NaN, R stays in [0,1], K ramp reaches K_MAX and holds.
for (const p of series) {
  assert.ok(Number.isFinite(p.r) && p.r >= 0 && p.r <= 1.0000001, `R out of range at t=${p.t}`);
}
assert.equal(couplingAt(0), 0, 'K must start at 0');
assert.ok(couplingAt(20) >= 2.2 - 1e-9, 'K must reach K_MAX by t=20s');

const rEnd = series[series.length - 1].r;
console.log(
  `selftest OK  |  r0=${r0.toFixed(4)}  cross@tau=${crossT.toFixed(3)}s  ` +
  `minAfter=${minAfter.toFixed(4)}  rEnd=${rEnd.toFixed(4)}`
);
