// Limen: a threshold with a band. Beats are wells, crossings are ridges, and the index flips only past the band.
import { DURATION_MS } from './universe.js';

const Q = (v, unit, basis) => Object.freeze({ v, unit, basis });
export const PRIORS = Object.freeze({
  glue: Q(0.7, 'ratio', 'rate drop at a beat: the glide slope is 1 - glue there and 1 + glue mid-ridge'),
  band: Q(0.12, 'rung', 'distance past the ridge before the held index flips'),
  ridge: Q(0.5, 'rung', 'the midpoint between two beats'),
  capture: Q(0.2, 'rail', 'at rest within this of a beat, the scroll settles into it'),
  rest: Q(0.002, 'rail', 'closer than this to a beat is already in it'),
  idle: Q(DURATION_MS.tick, 'ms', 'no scroll and no touch for this long is rest'),
  omega: Q(12, 'rad/s', 'the settling pull, a critical spring on the scroll offset'),
  turn: Q(2 * Math.PI, 'rad', 'one period of the glide'),
  arc: Q(4, 'ratio', 'the transient 4p(1-p) peaks at 1 mid-ridge'),
  stretch: Q(0.012, 'ratio', 'the fabric stretches this much along the travel axis mid-ridge'),
  settled: Q(0.5, 'px', 'the settling pull ends inside this distance'),
  settledSpeed: Q(1, 'px/s', 'and below this speed'),
  stepMax: Q(0.05, 's', 'the longest frame the settling spring integrates in one step'),
  fling: Q(2.5, 'rail/s', 'drive at which the fusee lets the beat go entirely'),
  driveOmega: Q(8, 'rad/s', 'the drive unwinds toward the measured scroll speed at this rate'),
  driveRest: Q(0.01, 'rail/s', 'drive below this has run down'),
});

const clamp = (value, low, high) => Math.max(low, Math.min(high, value));

// s(u) = u - g sin(2 pi u) / 2 pi: fixed ends, slope 1 - g at each beat, 1 + g on the ridge.
export function glide(u, glue = PRIORS.glue.v) {
  return u - glue * Math.sin(PRIORS.turn.v * u) / PRIORS.turn.v;
}

// The fusee: the ratio follows the remaining drive, so a fling runs through the beats and the well pulls as it unwinds.
export function fuseeGlue(drive, glue = PRIORS.glue.v, fling = PRIORS.fling.v) {
  const spent = Number.isFinite(drive) && fling > 0 ? Math.min(1, Math.abs(drive) / fling) : 0;
  return glue * (1 - spent);
}

// The drive after dt seconds: it unwinds toward the measured speed and never overshoots it.
export function unwind(drive, speed, dt, omega = PRIORS.driveOmega.v) {
  if (!Number.isFinite(speed) || !Number.isFinite(dt) || dt <= 0) return Number.isFinite(drive) ? drive : 0;
  const keep = Math.exp(-omega * dt);
  return speed + ((Number.isFinite(drive) ? drive : 0) - speed) * keep;
}

// The glide applied rung by rung to a float rung position.
export function glideFloat(float, glue = PRIORS.glue.v) {
  if (!Number.isFinite(float)) return 0;
  const k = Math.floor(float);
  return k + glide(float - k, glue);
}

// The held index moves to the nearest beat only once z is past the ridge by the band.
export function holdIndex(held, z, count, band = PRIORS.band.v) {
  const last = Math.max(0, count - 1);
  const nearest = clamp(Math.round(Number.isFinite(z) ? z : 0), 0, last);
  if (!Number.isInteger(held) || held < 0 || held > last) return nearest;
  if (nearest === held) return held;
  return Math.abs(z - held) >= PRIORS.ridge.v + band ? nearest : held;
}

// The transient: 0 at a beat, 1 mid-ridge.
export function transient(z) {
  if (!Number.isFinite(z)) return 0;
  const p = z - Math.floor(z);
  return PRIORS.arc.v * p * (1 - p);
}

// The beat offset a resting scroll settles into, or null when it rests on a ridge or already in a well.
export function captureOffset(offset, rail, count, capture = PRIORS.capture.v) {
  if (!Number.isFinite(offset) || !Number.isFinite(rail) || rail <= 0 || count < 1) return null;
  const u = offset / rail;
  const beat = clamp(Math.round(u), 0, count - 1);
  const distance = Math.abs(u - beat);
  if (distance <= PRIORS.rest.v || distance >= capture) return null;
  return beat * rail;
}

// The reading limen of maxwell/docs/data-context/agi-fabric/LIMINAL.md, ported line for line from maxwell/ops/fabric/limen.mjs:
// a setpoint theta with half-width h; inside the band is the liminal hold, outside is a signed subliminal depth.
export function trit(x, theta, h, opts = {}) {
  const { enter = h, exit = h, prev = 0 } = opts;
  const deviation = x - theta;
  if (prev !== 0 && prev * deviation >= exit) return prev;
  if (deviation >= enter) return 1;
  if (deviation <= -enter) return -1;
  return 0;
}

// A liminal (trit 0) reading is a hold, never a pass or a fail.
export function readingKind(t) {
  if (t > 0) return 'above';
  if (t < 0) return 'below';
  return 'hold';
}

// Signed subliminal depth: 0 inside the band, else excess past the edge in half-widths.
export function depth(x, theta, h) {
  const deviation = x - theta;
  const excess = Math.abs(deviation) - h;
  if (excess <= 0) return 0;
  return deviation < 0 ? -(excess / h) : excess / h;
}

// Bounded phase, the engine's setpoint_phase.
export function phase(x, theta, h) {
  const deviation = x - theta;
  const excess = Math.max(Math.abs(deviation) - h, 0);
  const magnitude = Math.atan2(excess, h);
  return deviation < 0 ? -magnitude : magnitude;
}

// The members of vector that project to the hold trit: the liminal subspace of this reading.
export function liminalSet(vector, theta, h, opts = {}) {
  const out = [];
  for (let i = 0; i < vector.length; i++) {
    const x = vector[i];
    if (trit(x, theta, h, opts) === 0) out.push({ index: i, x });
  }
  return out;
}

// Position inside the band: 0 at either edge, 1 at the setpoint; the transient of a crossing.
export function nearness(x, theta, h) {
  if (!(h > 0) || !Number.isFinite(x)) return 0;
  const u = Math.abs(x - theta) / h;
  return u >= 1 ? 0 : 1 - u;
}
