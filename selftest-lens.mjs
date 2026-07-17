// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Intuition Labs LLC
//
// selftest-lens — headless proof of the lens-drop pipeline in lens-core.js.
//   node selftest-lens.mjs
//
// Three checks: (1) the fp16 decoder against hand-computed values; (2) exact
// cone recovery by pursuit; (3) parity with the jspace monorepo's pinned
// lens-gate numbers, when the sibling checkout is present (skipped with a
// note when it is not — degrade closed, L7).

import { readFileSync, existsSync } from 'node:fs';
import { halfToNumber, pursuit, deltaMu, gateProbe, PURSUIT_K, EXPLAIN_TOL } from './lens-core.js';

let failed = 0;
const check = (name, ok, detail = '') => {
  console.log(`${ok ? '✔' : '✖'} ${name}${detail ? ` — ${detail}` : ''}`);
  if (!ok) failed++;
};

// 1 — fp16 decoder: hand-computed values
check('fp16: 0x3c00 = 1', halfToNumber(0x3c00) === 1);
check('fp16: 0xc000 = -2', halfToNumber(0xc000) === -2);
check('fp16: 0x0001 = 2^-24 (subnormal)', halfToNumber(0x0001) === 2 ** -24);
check('fp16: 0x7c00 = +Inf', halfToNumber(0x7c00) === Infinity);
check('fp16: 0x3555 ≈ 0.333252', Math.abs(halfToNumber(0x3555) - 0.333251953125) < 1e-12);

// 2 — pursuit: exact recovery inside the cone
{
  const b0 = [1, 0, 0, 0];
  const b1 = [0, 1, 0, 0];
  const res = pursuit([2, 3, 0, 0], [b0, b1], 2);
  const c = new Map(res.indices.map((i, at) => [i, res.coeffs[at]]));
  check('pursuit: x = 2·b0 + 3·b1 recovered', res.residual < 1e-9 && Math.abs(c.get(0) - 2) < 1e-9 && Math.abs(c.get(1) - 3) < 1e-9);
  const neg = pursuit([-1, 0, 0, 0], [b0, b1], 2);
  check('pursuit: nonnegativity — nothing fits -b0', Math.abs(neg.residual - 1) < 1e-9);
  check('gateProbe: zero vector skips', gateProbe([0, 0, 0, 0], [b0]).verdict === 'skip');
}

// 3 — parity with the jspace monorepo's pinned run (sibling checkout only)
{
  const fx = (n) => `../../jspace/examples/lens-gate/fixtures/${n}`;
  const PINNED = 0.9492090372027219;
  if (existsSync(fx('gemma-3-270m.layer6.rows0-48.lens.json'))) {
    const load = (n) => JSON.parse(readFileSync(fx(n), 'utf8'));
    const F = load('gemma-3-270m.layer6.rows0-48.lens.json').vectors;
    const G = load('gemma-3-270m.layer12.rows0-48.lens.json').vectors;
    const sample = [
      ...load('gemma-3-270m.layer6.rows48-80.probes.json').vectors,
      ...load('gemma-3-270m.layer12.rows48-80.probes.json').vectors,
    ];
    const dmu = deltaMu(F, G, sample, PURSUIT_K);
    check('parity: page math = jspace math on the real fixtures', dmu === PINNED, `Δ_μ ${dmu}`);
    const commits = sample.map((x) => gateProbe(x, F)).filter((p) => p.verdict === 'commit').length;
    check(`parity: ${commits}/64 commits at tol ${EXPLAIN_TOL}`, commits === 9);
  } else {
    console.log('· parity check skipped — jspace sibling checkout not found (degrade closed)');
  }
}

console.log(failed === 0 ? 'SELFTEST-LENS PASSED' : `SELFTEST-LENS FAILED (${failed})`);
process.exit(failed === 0 ? 0 : 1);
