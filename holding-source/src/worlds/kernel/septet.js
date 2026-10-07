// ─────────────────────────────────────────────────────────────────────────────
// THE SEPTET  ·  COLLAPSE
// Seven operators. Every legal change to a world is exactly one of them.
//
//   0 EMIT     cardinality ↑      pays measure out of the budget
//   1 ADVECT   configuration      conserves measure, count, topology
//   2 BIND     topology ↑ (join)  conserves measure & momentum, count ↓
//   3 CLEAVE   topology ↓ (split) conserves measure & momentum, count ↑
//   4 RELAX    energy ↓           conserves measure, count, topology
//   5 COUPLE   the LAW itself     conserves state; bends Λ by the residual
//   6 OBSERVE  nothing            pure projection; the only lossy step
//
// Mutually exclusive: each touches a disjoint slot of (N, x, T⁺, T⁻, E, Λ, ∅).
// Collectively exhaustive: there is no other kind of change a world can make.
//
// The seven sit on the Fano plane. Any two operators lie on exactly one line;
// the third point of that line is the move that closes them. Composition is
// therefore not taste — it is incidence.
// ─────────────────────────────────────────────────────────────────────────────

export const SEPTET = ['EMIT', 'ADVECT', 'BIND', 'CLEAVE', 'RELAX', 'COUPLE', 'OBSERVE'];

export const FANO = [
  { pts: [0, 1, 2], name: 'SEDIMENT',    reads: 'birth → drift → gather' },
  { pts: [0, 3, 4], name: 'CRYSTALLISE', reads: 'birth → fracture → settle' },
  { pts: [0, 5, 6], name: 'ANNUNCIATE',  reads: 'birth → law bends → the world is seen' },
  { pts: [1, 3, 5], name: 'ERODE',       reads: 'drift → fracture → law bends' },
  { pts: [1, 4, 6], name: 'DRIFT',       reads: 'drift → settle → the world is seen' },
  { pts: [2, 3, 6], name: 'TESSELLATE',  reads: 'gather → fracture → the world is seen' },
  { pts: [2, 4, 5], name: 'ANNEAL',      reads: 'gather → settle → law bends' },
];

// the unique line through two operators — the closure rule
export function closes(a, b) {
  return FANO.find((L) => L.pts.includes(a) && L.pts.includes(b));
}

const REF = 60;           // reference sampling rate: rates are expressed per second
const KAPPA_MAX = 0.25;   // COUPLE contraction ceiling: |ΔΛ| ≤ κ|r|, κ < 1
const CFL = 0.35;         // no body may cross more than this fraction of a reach

// ── state ────────────────────────────────────────────────────────────────────
export function Body(d, cap) {
  return {
    d, cap, n: 0,
    x: new Float32Array(cap * d),
    p: new Float32Array(cap * d),      // previous position — the trace
    v: new Float32Array(cap * d),
    m: new Float32Array(cap),
    r: new Float32Array(cap),          // core radius
    phi: new Float32Array(cap),        // phase
    om: new Float32Array(cap),         // natural frequency
    age: new Float32Array(cap),
    str: new Float32Array(cap),        // accumulated strain
    lnk: new Int32Array(cap).fill(-1), // bond partner
    lrest: new Float32Array(cap),
    budget: 1, spent: 0,               // closed measure account
    E: 0, R: 0, resid: 0, deposits: 0,
    line: null, beats: 0, held: 0,
    // event debts. births, fusions and fractures are RATES, so how finely the
    // world is sampled cannot change how often they happen per unit time.
    emitDebt: 0, bindDebt: 0, cleaveDebt: 0, bindCursor: 0, cleaveCursor: 0,
    acc: { emit: 0, bind: 0, cleave: 0 },   // event accrual, so rates are per second
  };
}

// the world's own yardstick: mean spacing of what currently exists.
// every interaction scale is measured in this, so nothing is ever "5 pixels".
export const spacing = (S) => Math.pow(1 / Math.max(S.n, 8), 1 / S.d);

// Events are things that happen at a rate, not things that happen once per call.
// Sampling a world twice as finely must not make it emit, bind and cleave twice
// as often — so each event operator accrues against world-time and carries the
// remainder. 60 Hz is the reference rate: at Δt = 1/60 this returns 1 every beat.
export function cadence(S, key, ctx, per = 1) {
  const a = (S.acc[key] || 0) + per * 60 * ctx.dt;
  const n = Math.floor(a);
  S.acc[key] = a - n;
  return n;
}
export const quantum = (S) => S.budget / S.cap;   // the medium's grain of measure
function dist2(S, i, j) {
  let s = 0;
  for (let k = 0; k < S.d; k++) { const q = S.x[i * S.d + k] - S.x[j * S.d + k]; s += q * q; }
  return s;
}

// uniform spatial hash, any dimension
function grid(S, cell) {
  const g = new Map(); const key = new Array(S.d);
  for (let i = 0; i < S.n; i++) {
    for (let k = 0; k < S.d; k++) key[k] = Math.floor(S.x[i * S.d + k] / cell);
    const kk = key.join(',');
    let b = g.get(kk); if (!b) g.set(kk, (b = []));
    b.push(i);
  }
  return { g, cell };
}
function near(S, G, i, fn) {
  const base = new Array(S.d);
  for (let k = 0; k < S.d; k++) base[k] = Math.floor(S.x[i * S.d + k] / G.cell);
  const span = Math.pow(3, S.d);
  const key = new Array(S.d);
  for (let m = 0; m < span; m++) {
    let q = m;
    for (let k = 0; k < S.d; k++) { key[k] = base[k] + (q % 3) - 1; q = (q / 3) | 0; }
    const b = G.g.get(key.join(','));
    if (b) for (const j of b) if (j !== i) fn(j);
  }
}

// ── 0 · EMIT ─────────────────────────────────────────────────────────────────
export function EMIT(S, Λ, ctx) {
  const d = S.d;
  if (S.budget - S.spent <= 0 || S.n >= S.cap) return;
  const want = Λ.n('emit.rate', ctx, Math.max(1, (S.cap - S.n) >> 4));
  const k = Math.min(cadence(S, 'emit', ctx, want), S.cap - S.n);
  if (k <= 0) return;
  const σ = spacing(S), Q = quantum(S);
  const womb = ctx.womb || (ctx.womb = Λ.field('emit.womb', d, ctx));
  const gate = Λ.frac('emit.gate', ctx) - 0.5;
  const flow = ctx.flow || (ctx.flow = Λ.curl('advect.weather', d, ctx));
  for (let e = 0; e < k; e++) {
    let px = null;
    for (let tries = 0; tries < 6 && !px; tries++) {
      const c = Λ.vec('emit.site.' + (S.n + e) + '.' + tries, d, ctx);
      const rad = Λ.len('emit.spread', ctx) * Math.cbrt(Λ.frac('emit.rad.' + tries, ctx));
      const q = new Array(d);
      for (let j = 0; j < d; j++) q[j] = 0.5 + c[j] * rad;
      if (womb(q) > gate) px = q;
    }
    if (!px) continue;
    const mass = Q * (0.5 + Λ.frac('emit.mass', ctx) * 5.5);
    if (S.spent + mass > S.budget) return;
    const i = S.n++;
    S.spent += mass;
    const u = flow(px);
    const kick = σ * Λ.rate('emit.kick', ctx);
    for (let j = 0; j < d; j++) {
      S.x[i * d + j] = px[j]; S.p[i * d + j] = px[j];
      S.v[i * d + j] = u[j] * kick;
    }
    S.m[i] = mass;
    S.r[i] = σ * (0.12 + Λ.frac('emit.core', ctx) * 0.5) * Math.pow(mass / Q, 1 / d);
    S.phi[i] = Λ.ang('emit.phase.' + i, ctx);
    S.om[i] = Λ.rate('emit.omega', ctx) * (1 + Λ.bal('emit.omega.spread.' + i, ctx) * 0.3);
    S.age[i] = 0; S.str[i] = 0; S.lnk[i] = -1;
  }
}

// ── 1 · ADVECT ───────────────────────────────────────────────────────────────
export function ADVECT(S, Λ, ctx) {
  const d = S.d, dt = ctx.dt;
  const flow = ctx.flow || (ctx.flow = Λ.curl('advect.weather', d, ctx));
  const σ = spacing(S);
  const speed = σ * Λ.rate('advect.gain', ctx);       // how fast the weather moves things
  const drag = Λ.rate('advect.drag', ctx);            // how eagerly bodies obey it
  const swirl = Λ.bal('advect.chirality', ctx);
  const reach = σ;
  const mode = Λ.pick('world.boundary', ['wrap', 'reflect', 'absorb'], ctx);
  const q = new Array(d);
  for (let i = 0; i < S.n; i++) {
    for (let k = 0; k < d; k++) q[k] = S.x[i * d + k];
    const u = flow(q);
    const obey = Math.min(1, drag * dt);
    let sp = 0;
    for (let k = 0; k < d; k++) {
      const tgt = (u[k] + (k % 2 ? swirl : -swirl) * u[(k + 1) % d]) * speed;
      S.v[i * d + k] += (tgt - S.v[i * d + k]) * obey;
      sp += S.v[i * d + k] * S.v[i * d + k];
    }
    sp = Math.sqrt(sp);
    const lim = (CFL * reach) / Math.max(dt, 1e-6);
    const sc = sp > lim ? lim / sp : 1;                 // CFL: no tunnelling
    for (let k = 0; k < d; k++) {
      S.p[i * d + k] = S.x[i * d + k];
      S.v[i * d + k] *= sc;
      S.x[i * d + k] += S.v[i * d + k] * dt;
    }
    S.age[i] += dt;
    // the manifold answers back
    let lost = false;
    for (let k = 0; k < d; k++) {
      let c = S.x[i * d + k];
      if (c < 0 || c > 1) {
        if (mode === 'wrap') { c = ((c % 1) + 1) % 1; S.p[i * d + k] = c; }
        else if (mode === 'reflect') { c = c < 0 ? -c : 2 - c; S.v[i * d + k] *= -1; }
        else lost = true;
        S.x[i * d + k] = c;
      }
    }
    if (lost) {                                          // measure returns to the account
      S.spent -= S.m[i];
      const last = --S.n;
      if (i !== last) copy(S, last, i);
      i--;
    }
  }
}

function copy(S, from, to) {
  const d = S.d;
  for (let k = 0; k < d; k++) {
    S.x[to * d + k] = S.x[from * d + k];
    S.p[to * d + k] = S.p[from * d + k];
    S.v[to * d + k] = S.v[from * d + k];
  }
  S.m[to] = S.m[from]; S.r[to] = S.r[from]; S.phi[to] = S.phi[from];
  S.om[to] = S.om[from]; S.age[to] = S.age[from]; S.str[to] = S.str[from];
  S.lnk[to] = -1; S.lrest[to] = S.lrest[from];
}

// ── 2 · BIND ─────────────────────────────────────────────────────────────────
export function BIND(S, Λ, ctx) {
  const rounds = cadence(S, 'bind', ctx);
  for (let r = 0; r < rounds; r++) bindOnce(S, Λ, ctx);
}
function bindOnce(S, Λ, ctx) {
  const d = S.d;
  const σ = spacing(S), Q = quantum(S);
  const reach = σ * (0.6 + Λ.frac('bind.reach', ctx) * 2.4);
  const affinity = 1 - 2 * Λ.frac('bind.affinity', ctx);      // phase agreement needed
  const ceiling = Q * Λ.n('bind.ceiling', ctx, 96);           // no runaway blob
  const G = grid(S, Math.max(reach, 1e-3));
  S.bindDebt += REF * ctx.dt * S.n;                    // bodies considered per second
  let quota = Math.min(S.n, Math.floor(S.bindDebt));
  S.bindDebt -= quota;
  for (let q = 0; q < quota; q++) {
    const i = (S.bindCursor = (S.bindCursor + 1) % Math.max(1, S.n));
    if (i >= S.n || S.lnk[i] >= 0) continue;
    let best = -1, bd = reach * reach;
    near(S, G, i, (j) => {
      if (j >= S.n || j === i) return;
      const q = dist2(S, i, j);
      if (q < bd) { bd = q; best = j; }
    });
    if (best < 0) continue;
    const dd = Math.sqrt(bd);
    const contact = dd < (S.r[i] + S.r[best]) * (0.5 + Λ.frac('bind.fusion', ctx));
    const agree = Math.cos(S.phi[i] - S.phi[best]) > affinity;
    if (contact && agree && S.m[i] + S.m[best] <= ceiling && S.n > 1) {
      // fusion: measure and momentum conserved exactly, count falls
      const mi = S.m[i], mj = S.m[best], M = mi + mj;
      for (let k = 0; k < d; k++) {
        S.x[i * d + k] = (S.x[i * d + k] * mi + S.x[best * d + k] * mj) / M;
        S.v[i * d + k] = (S.v[i * d + k] * mi + S.v[best * d + k] * mj) / M;
      }
      S.phi[i] = Math.atan2(
        Math.sin(S.phi[i]) * mi + Math.sin(S.phi[best]) * mj,
        Math.cos(S.phi[i]) * mi + Math.cos(S.phi[best]) * mj);
      S.m[i] = M;
      S.r[i] = Math.pow(Math.pow(S.r[i], d) + Math.pow(S.r[best], d), 1 / d);
      S.str[i] += Λ.frac('bind.scar', ctx);
      const last = --S.n;
      if (best !== last) copy(S, last, best);
      if (i === last) break;
    } else if (S.lnk[best] < 0) {
      S.lnk[i] = best; S.lnk[best] = i;
      S.lrest[i] = S.lrest[best] = dd;
    }
  }
}

// ── 3 · CLEAVE ───────────────────────────────────────────────────────────────
export function CLEAVE(S, Λ, ctx) {
  const rounds = cadence(S, 'cleave', ctx);
  for (let r = 0; r < rounds; r++) cleaveOnce(S, Λ, ctx);
}
function cleaveOnce(S, Λ, ctx) {
  const d = S.d;
  const thr = Λ.frac('cleave.threshold', ctx);
  const snap = S.d && (spacing(S) * (1.6 + Λ.frac('bind.stretch', ctx) * 3));
  for (let i = 0; i < S.n; i++) {                       // bonds the world can no longer keep
    const j = S.lnk[i];
    if (j < 0 || j >= S.n) continue;
    if (Math.sqrt(dist2(S, i, j)) > S.lrest[i] + snap) {
      S.lnk[i] = -1; S.lnk[j] = -1;
      S.str[i] += Λ.frac('bind.scar', ctx); S.str[j] += Λ.frac('bind.scar', ctx);
    }
  }
  const throwv = spacing(S) * Λ.rate('cleave.throw', ctx);
  S.cleaveDebt += REF * ctx.dt * S.n;
  let quota = Math.min(S.n, Math.floor(S.cleaveDebt));
  S.cleaveDebt -= quota;
  for (let q = 0; q < quota && S.n < S.cap; q++) {
    const i = (S.cleaveCursor = (S.cleaveCursor + 1) % Math.max(1, S.n));
    if (i >= S.n || S.str[i] < thr) continue;
    const j = S.lnk[i];
    if (j >= 0) { S.lnk[j] = -1; S.lnk[i] = -1; S.str[i] *= 0.5; continue; }  // bond breaks first
    const c = S.n++;
    const axis = Λ.vec('cleave.axis.' + i, d, ctx);
    const half = S.m[i] * 0.5;
    S.m[i] = S.m[c] = half;
    const rr = S.r[i] * Math.pow(0.5, 1 / d);
    S.r[i] = S.r[c] = rr;
    for (let k = 0; k < d; k++) {                       // momentum conserved
      S.x[c * d + k] = S.x[i * d + k] - axis[k] * rr;
      S.x[i * d + k] += axis[k] * rr;
      S.p[c * d + k] = S.x[c * d + k];
      S.v[c * d + k] = S.v[i * d + k] - axis[k] * throwv;
      S.v[i * d + k] += axis[k] * throwv;
    }
    S.phi[c] = S.phi[i] + Math.PI * Λ.frac('cleave.phase.slip', ctx);
    S.om[c] = S.om[i]; S.age[c] = S.age[i];
    S.str[i] = S.str[c] = 0; S.lnk[c] = -1;
  }
}

// ── 4 · RELAX ────────────────────────────────────────────────────────────────
export function RELAX(S, Λ, ctx) {
  const d = S.d, dt = ctx.dt;
  const gamma = Λ.rate('relax.gamma', ctx);
  // coupling is measured against the same clock that sets natural frequencies,
  // so synchrony is reachable rather than accidental
  const K = Λ.rate('emit.omega', ctx) * (0.5 + 4 * Λ.frac('relax.sync', ctx));
  const push = Λ.rate('relax.push', ctx);
  const stiff = Λ.rate('relax.stiffness', ctx);
  const σ = spacing(S);
  const reach = σ * (0.6 + Λ.frac('bind.reach', ctx) * 2.4);
  const G = grid(S, Math.max(reach, 1e-3));
  const damp = Math.exp(-gamma * dt);
  const pullC = new Float32Array(S.n), pullS = new Float32Array(S.n), pullA = new Float32Array(S.n);
  for (let i = 0; i < S.n; i++) {
    let sc = 0, ss = 0, cnt = 0;
    near(S, G, i, (j) => {
      if (j >= S.n) return;
      const q2 = dist2(S, i, j);
      if (q2 > reach * reach) return;
      const q = Math.sqrt(q2) || 1e-6;
      sc += Math.cos(S.phi[j]); ss += Math.sin(S.phi[j]); cnt++;
      const overlap = S.r[i] + S.r[j] - q;
      if (overlap > 0) {                                  // soft core, non-expansive
        for (let k = 0; k < d; k++) {
          const nrm = (S.x[i * d + k] - S.x[j * d + k]) / q;
          S.v[i * d + k] += nrm * overlap * push * dt;
        }
        S.str[i] += overlap * dt * push;
      }
    });
    // the local mean field: a direction ψ and a strength r
    pullC[i] = cnt ? sc / cnt : 0; pullS[i] = cnt ? ss / cnt : 0;
    pullA[i] = cnt ? K * Math.hypot(pullC[i], pullS[i]) : 0;
    const j = S.lnk[i];
    if (j >= 0 && j < S.n) {                              // bonds pull to rest length
      let q = Math.sqrt(dist2(S, i, j)) || 1e-6;
      const over = q / (S.lrest[i] + reach);
      if (over > 1) S.str[i] += (over - 1) * dt * stiff;   // strain, not a cut
      const f = Math.min((q - S.lrest[i]) * stiff, stiff * reach * 4);
      for (let k = 0; k < d; k++) {
        const nrm = (S.x[i * d + k] - S.x[j * d + k]) / q;
        S.v[i * d + k] -= nrm * f * dt;
      }
      S.str[i] += Math.abs(q - S.lrest[i]) * dt * stiff;
    }
  }
  for (let i = 0; i < S.n; i++) {
    S.phi[i] += S.om[i] * dt;                                  // exact rotation
    if (pullA[i] > 0) {
      // φ' = A·sin(ψ−φ) has the closed form tan(θ/2) ← tan(θ/2)·e^(−A·dt).
      // Exact, unconditionally stable, and independent of the step size — which
      // explicit Euler is not, at any coupling strength that matters.
      const psi = Math.atan2(pullS[i], pullC[i]);
      let th = Math.atan2(Math.sin(psi - S.phi[i]), Math.cos(psi - S.phi[i]));
      const t2 = Math.tan(Math.max(-1.5707, Math.min(1.5707, th / 2))) * Math.exp(-pullA[i] * dt);
      S.phi[i] = psi - 2 * Math.atan(t2);
    }
    for (let k = 0; k < d; k++) S.v[i * d + k] *= damp;         // dissipative by construction
  }
}

// pure reads. measurement is the observer's privilege, not the world's.
export function measure(S) {
  const d = S.d, cell = spacing(S) * 3;
  const bins = new Map(); const key = new Array(d);
  let E = 0, gr = 0, giy = 0;
  for (let i = 0; i < S.n; i++) {
    let sp = 0;
    for (let k = 0; k < d; k++) sp += S.v[i * d + k] ** 2;
    E += 0.5 * S.m[i] * sp;
    const c = Math.cos(S.phi[i]), si = Math.sin(S.phi[i]);
    gr += c; giy += si;
    for (let k = 0; k < d; k++) key[k] = Math.floor(S.x[i * d + k] / cell);
    const kk = key.join(',');
    let b = bins.get(kk); if (!b) bins.set(kk, (b = [0, 0, 0]));
    b[0] += c; b[1] += si; b[2]++;
  }
  let coh = 0;
  for (const b of bins.values()) coh += Math.hypot(b[0], b[1]);
  S.E = E;
  S.Rg = S.n ? Math.hypot(gr, giy) / S.n : 0;               // global Kuramoto order
  S.R = S.n ? coh / S.n : 0;                                // coarse-grained local order
}

// bonds must stay reciprocal after any structural move
export function mend(S) {
  for (let i = 0; i < S.n; i++) {
    const j = S.lnk[i];
    if (j < 0) continue;
    if (j >= S.n || S.lnk[j] !== i) S.lnk[i] = -1;
  }
}

// ── 5 · COUPLE ───────────────────────────────────────────────────────────────
// the residual — how far the world is from the world its own law implies
export function residual(S, Λ, ctx) {
  const Rstar = Λ.frac('world.coherence.target', ctx);
  const Estar = Λ.frac('world.energy.target', ctx) * Λ.len('world.energy.scale', ctx);
  const μstar = Λ.frac('world.fill.target', ctx);
  const rR = Rstar - S.R;
  const rE = (S.E - Estar) / (Estar + 1e-9);
  const rμ = μstar - S.spent / (S.budget + 1e-9);
  // Ω — what the world has actually been seen to be. zero in a blind world.
  const Ωstar = Λ.frac('world.record.target', ctx);
  const rΩ = ctx.record === undefined ? 0 : Ωstar - ctx.record;
  return { rR, rE: Math.max(-4, Math.min(4, rE)), rμ, rΩ,
    mag: Math.abs(rR) + Math.abs(rμ) + Math.abs(rΩ) + Math.min(1, Math.abs(rE)) };
}

export function COUPLE(S, Λ, ctx) {
  const r = residual(S, Λ, ctx);
  const κ = Λ.frac('couple.gain', ctx) * KAPPA_MAX;
  const dt = ctx.dt;
  const set = ctx.couples || DEFAULT_COUPLES;
  for (const [name, chan, sense] of set) {
    const drive = chan === 'R' ? r.rR : chan === 'E' ? r.rE : chan === 'Ω' ? r.rΩ : r.rμ;
    Λ.bend(name, κ * drive * sense * dt);
  }
  S.resid = r.mag;
  return r;
}

// [ constant , residual channel (R coherence · E energy · μ measure · Ω record) , sign ]
export const DEFAULT_COUPLES = [
  ['emit.rate|n', 'μ', +1],
  ['advect.gain|r', 'E', -1],
  ['bind.reach|f', 'R', +1],
  ['cleave.threshold|f', 'R', +1],
  ['relax.gamma|r', 'E', +1],
  ['relax.sync|f', 'R', +1],
  ['ink.hue|a', 'R', +1],
  ['ink.weight|f', 'μ', +1],
];

// ── 6 · OBSERVE ──────────────────────────────────────────────────────────────
// pure. returns a read-only view. the only place a world becomes visible.
export function OBSERVE(S, Λ, ctx) {
  measure(S);
  return {
    d: S.d, n: S.n, x: S.x, p: S.p, v: S.v, m: S.m, r: S.r,
    phi: S.phi, age: S.age, str: S.str, lnk: S.lnk, mQ: S.budget / S.cap, budget: S.budget,
    witness: { R: S.R, tau: 1 - S.R, E: S.E, fill: S.spent / (S.budget + 1e-9), resid: S.resid },
    depositing: !!(S.line && S.line.pts.includes(6)),
  };
}

// ── the director ─────────────────────────────────────────────────────────────
// τ = 1 − R is the crystallisation witness. hot worlds explore; cold worlds
// exploit. the piece schedules itself; no frame counter ever chooses for it.
export function chooseLine(S, Λ, ctx) {
  const tau = 1 - S.R;
  const heat = Λ.frac('director.temperament', ctx);
  // τ steers the odds; it does not decide. Selection is resolved from the law
  // itself, so the schedule stays deterministic in the seed while remaining
  // ergodic over all seven gestures. Taking the argmax here — as this once did —
  // lets a world lock onto a single line forever, and a world locked onto a line
  // without OBSERVE never lays a mark at all. No world may be permanently blind.
  const FLOOR = 0.06;
  const w = FANO.map((L, li) => {
    const explore = (L.pts.includes(0) ? 1 : 0) + (L.pts.includes(3) ? 1 : 0);
    const exploit = (L.pts.includes(4) ? 1 : 0) + (L.pts.includes(2) ? 1 : 0);
    const bias = FLOOR + Λ.frac('director.line.' + li, ctx);
    return bias * (1 + explore * tau * heat + exploit * (1 - tau) * (1 - heat) * 2);
  });
  const total = w.reduce((a, b) => a + b, 0);
  let roll = Λ.frac('director.roll.' + S.beats, ctx) * total;
  for (let i = 0; i < w.length; i++) { roll -= w[i]; if (roll <= 0) return FANO[i]; }
  return FANO[FANO.length - 1];
}

export const OPS = [EMIT, ADVECT, BIND, CLEAVE, RELAX, COUPLE, OBSERVE];

// ── one beat of the world ────────────────────────────────────────────────────
export function step(S, Λ, ctx) {
  ctx.womb = null; ctx.flow = null;                 // fields are re-read each beat:
  ctx.womb = Λ.field('emit.womb', S.d, ctx);        // the law has moved since last
  ctx.flow = Λ.curl('advect.weather', S.d, ctx);
  // a gesture is held for a span of world-time; sampling the world more finely
  // must not make it change its mind more often
  const hold = Λ.n('director.hold', ctx, 12) / 60;
  S.held += ctx.dt;
  if (!S.line || S.held >= hold) { S.line = chooseLine(S, Λ, ctx); S.held = 0; }
  S.beats++;
  for (const p of S.line.pts) if (p !== 6) OPS[p](S, Λ, ctx);
  mend(S);
  const view = OBSERVE(S, Λ, ctx);
  if (view.depositing) S.deposits++;
  return view;
}
