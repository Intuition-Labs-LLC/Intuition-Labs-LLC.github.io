// ─────────────────────────────────────────────────────────────────────────────
// Λ  ·  LAW  ·  POTENTIAL
// Nothing in a world is a number until the world is asked for it.
// Outside kernel/, a numeric literal in artwork code is a defect.
// The literals below are DIMENSIONAL PRIORS: they define what a "length" or a
// "count" can mean at all. They are the physics of the medium, not choices
// about a piece.
// ─────────────────────────────────────────────────────────────────────────────

const FNV_OFF = 2166136261;
const FNV_PRM = 16777619;

// exact 32-bit hash — mirrored bit-for-bit in kernel/py/law.py and in GLSL,
// so one seed names the same world in every language this skill speaks.
export function h32(s) {
  let h = FNV_OFF >>> 0;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i) & 255;
    h = Math.imul(h, FNV_PRM) >>> 0;
  }
  h ^= h >>> 16; h = Math.imul(h, 2246822507) >>> 0;
  h ^= h >>> 13; h = Math.imul(h, 3266489909) >>> 0;
  h ^= h >>> 16;
  return h >>> 0;
}
export const u1 = (s) => h32(s) / 4294967296;

// recurrence ladder. every constant oscillates on a rational period, so the
// whole law returns to itself exactly at τ = LCM = 120. worlds are loops.
const LADDER = [1, 2, 3, 4, 5, 6, 8, 12];
const LCM = 120;
const OCT = 4;          // harmonics per constant
const FALL = 0.5;       // spectral falloff
const LEN_FLOOR = 0.0015, LEN_CEIL = 0.85;   // lengths live in this decade band
const EPS = 1e-3;       // finite-difference span, in world units

const clamp01 = (x) => (x < 0 ? 0 : x > 1 ? 1 : x);
const fade = (t) => t * t * t * (t * (t * 6 - 15) + 10);
const mix = (a, b, t) => a + (b - a) * t;

export function Law(seedText, opts = {}) {
  const seed = String(seedText);
  const drift = new Map();          // ← COUPLE writes here. the law is mutable.
  const memo = new Map();
  const MEMO_LIMIT = 2048;

  // the world's own tempo: one full recurrence in this many seconds
  const periodSec = mix(24, 96, u1(seed + '/tempo'));
  const w0 = LCM / periodSec;

  function osc(name, t) {
    const key = seed + '§' + name;
    let sum = 0, norm = 0;
    for (let k = 0; k < OCT; k++) {
      const a = Math.pow(FALL, k);
      const den = LADDER[h32(key + '/f' + k) % LADDER.length];
      const ph = u1(key + '/p' + k);
      sum += a * Math.sin(2 * Math.PI * ((t * w0) / den + ph));
      norm += a;
    }
    return 0.5 + 0.5 * (sum / norm);
  }

  // raw resolved value of a named constant ∈ [0,1], law-drift included
  function value(name, ctx) {
    const t = ctx && ctx.t !== undefined ? ctx.t : 0;
    return clamp01(osc(name, t) + (drift.get(name) || 0));
  }

  function uu(name) {                // static (τ-free) draw for a name
    let v = memo.get(name);
    if (v === undefined) {
      if (memo.size >= MEMO_LIMIT) {
        const oldest = memo.keys().next().value;
        memo.delete(oldest);
      }
      v = u1(seed + '§' + name);
      memo.set(name, v);
    }
    return v;
  }

  // ── gradient noise, generalised to d dimensions ───────────────────────────
  function gradAt(name, cell, d, i) {
    // component i of the unit gradient stored at an integer lattice site
    let s = seed + '§' + name + '#';
    for (let k = 0; k < d; k++) s += cell[k] + ',';
    return u1(s + 'g' + i) * 2 - 1;
  }

  function noiseN(name, p, d) {
    const c0 = new Array(d), f = new Array(d), w = new Array(d);
    for (let k = 0; k < d; k++) {
      c0[k] = Math.floor(p[k]);
      f[k] = p[k] - c0[k];
      w[k] = fade(f[k]);
    }
    const corners = 1 << d;
    let acc = 0;
    const cell = new Array(d);
    for (let m = 0; m < corners; m++) {
      let wt = 1, dot = 0;
      for (let k = 0; k < d; k++) {
        const bit = (m >> k) & 1;
        cell[k] = c0[k] + bit;
        wt *= bit ? w[k] : 1 - w[k];
        dot += gradAt(name, cell, d, k) * (f[k] - bit);
      }
      acc += wt * dot;
    }
    return acc * Math.sqrt(d);       // ≈ [-1,1] for any d
  }

  const Λ = {
    seed, drift, periodSec, w0, eps: EPS,
    // world extent is unity by construction; the observer decides pixels.
    extent: opts.extent !== undefined ? opts.extent : 1,

    raw: value,
    stat: uu,

    // ── typed resolution ────────────────────────────────────────────────────
    // fraction ∈ [0,1], with a law-chosen bias so worlds are not uniform
    frac(name, ctx) {
      const v = value(name + '|f', ctx);
      const g = Math.pow(2, (uu(name + '|γ') - 0.5) * 2);
      return Math.pow(v, g);
    },
    // signed ∈ [-1,1]
    bal(name, ctx) { return this.frac(name, ctx) * 2 - 1; },

    // length in world units, log-uniform inside a law-chosen band
    len(name, ctx, span) {
      const s = span === undefined ? this.extent : span;
      const a = mix(Math.log(LEN_FLOOR), Math.log(LEN_CEIL), uu(name + '|a'));
      const b = mix(Math.log(LEN_FLOOR), Math.log(LEN_CEIL), uu(name + '|b'));
      const lo = Math.min(a, b), hi = Math.max(a, b);
      return s * Math.exp(mix(lo, hi, value(name + '|l', ctx)));
    },
    // count ≥ 1, log-uniform up to a capacity the medium can actually hold
    n(name, ctx, cap) {
      const c = Math.max(1, cap === undefined ? 64 : cap);
      const v = value(name + '|n', ctx);
      const bias = mix(0.35, 1, uu(name + '|nb'));
      return Math.max(1, Math.round(Math.exp(mix(0, Math.log(c), Math.pow(v, 1 / bias)))));
    },
    // per-second rate, log-uniform around the world's own tempo
    rate(name, ctx) {
      const v = value(name + '|r', ctx);
      return (1 / this.periodSec) * Math.exp(mix(-2, 4, v));
    },
    ang(name, ctx) { return value(name + '|a', ctx) * Math.PI * 2; },
    sgn(name) { return uu(name + '|s') < 0.5 ? -1 : 1; },
    // choose from a set — the only legal way to branch on identity
    pick(name, arr, ctx) {
      return arr[Math.min(arr.length - 1, Math.floor(value(name + '|k', ctx) * arr.length))];
    },
    // unit vector in ANY dimension. this is what makes a piece dimension-free.
    vec(name, d, ctx) {
      const out = new Array(d);
      let s = 0;
      for (let k = 0; k < d; k++) {
        const g = value(name + '|v' + k, ctx) * 2 - 1;
        out[k] = g; s += g * g;
      }
      s = Math.sqrt(s) || 1;
      for (let k = 0; k < d; k++) out[k] /= s;
      return out;
    },

    // ── fields: laws that vary over space as well as time ──────────────────
    // scalar field f(p ∈ ℝ^d) → [-1,1], fractal, all parameters law-resolved
    field(name, d, ctx) {
      const oct = Math.max(1, Math.min(5, this.n(name + '|oct', ctx, 5)));
      const lac = mix(1.7, 2.6, uu(name + '|lac'));
      const gain = mix(0.35, 0.62, uu(name + '|gain'));
      const scale = 1 / this.len(name + '|scale', ctx);
      const flow = this.rate(name + '|flow', ctx) * 0.15;
      const t = ctx && ctx.t !== undefined ? ctx.t : 0;
      const off = new Array(d);
      for (let k = 0; k < d; k++) off[k] = uu(name + '|o' + k) * 64 + t * flow * (k + 1);
      return (p) => {
        let amp = 1, norm = 0, acc = 0, sc = scale;
        const q = new Array(d);
        for (let o = 0; o < oct; o++) {
          for (let k = 0; k < d; k++) q[k] = p[k] * sc + off[k];
          acc += amp * noiseN(name + '/' + o, q, d);
          norm += amp; amp *= gain; sc *= lac;
        }
        return acc / norm;
      };
    },

    // divergence-free flow in any dimension: area (2D) / volume (3D) preserving.
    // this is why advection looks like weather instead of like a screensaver.
    curl(name, d, ctx) {
      if (d === 2) {
        const psi = this.field(name + '|ψ', 2, ctx);
        return (p) => {
          const dx = (psi([p[0] + EPS, p[1]]) - psi([p[0] - EPS, p[1]])) / (2 * EPS);
          const dy = (psi([p[0], p[1] + EPS]) - psi([p[0], p[1] - EPS])) / (2 * EPS);
          return [dy, -dx];
        };
      }
      const A = [0, 1, 2].map((i) => this.field(name + '|A' + i, 3, ctx));
      return (p) => {
        const d_ = (f, ax) => {
          const a = p.slice(), b = p.slice();
          a[ax] += EPS; b[ax] -= EPS;
          return (f(a) - f(b)) / (2 * EPS);
        };
        return [
          d_(A[2], 1) - d_(A[1], 2),
          d_(A[0], 2) - d_(A[2], 0),
          d_(A[1], 0) - d_(A[0], 1),
        ];
      };
    },

    // ── COUPLE writes here. residual folds back into the constants. ─────────
    bend(name, delta, ceiling) {
      const cap = ceiling === undefined ? 0.4 : ceiling;
      const cur = drift.get(name) || 0;
      const nxt = Math.max(-cap, Math.min(cap, cur + delta));
      drift.set(name, nxt);
      return nxt;
    },
    bentBy() { let s = 0; for (const v of drift.values()) s += Math.abs(v); return s; },
  };

  return Λ;
}
