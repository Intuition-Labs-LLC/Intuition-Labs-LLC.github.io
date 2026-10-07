import { Law, h32 } from '../worlds/kernel/law.js';
import { FRAME_MS, DURATION_MS, CAGE_PERIOD_MS } from '../ds/universe.js';
import { PRIORS as LIMEN } from '../ds/limen.js';
import { PIECE } from './world.js';

export const VOCABULARY = Object.freeze([
  'intelligence', 'network', 'community', 'platform', 'internet', 'protocol',
  'society', 'technology', 'agent', 'environment', 'world', 'universe',
]);
export const THEME = Object.freeze({ paper: '#faf9f5', ink: '#14211f', muted: '#526059', sage: '#cbd9c9', violet: '#ded0df' });
export const MEDIUM = Object.freeze({
  secondMs: 1000, dayMs: 86400000, sampleSide: 72, contourSide: 64,
  particleCapacity: 96, maxDpr: 1.5, maxPixels: 2097152, drawEveryFrames: 3,
  modeCapacity: 7, maxHarmonic: 4, edgeWidth: 1.3,
  cycleMs: CAGE_PERIOD_MS, transitionMs: DURATION_MS.settle,
  frameMs: FRAME_MS, maxStepSeconds: LIMEN.stepMax.v,
});

export function earthContext(epochMs) {
  if (!Number.isFinite(epochMs)) throw new TypeError('finite clock required');
  const utcDay = Math.floor(epochMs / MEDIUM.dayMs);
  return Object.freeze({ utcDay, rotation: (epochMs - utcDay * MEDIUM.dayMs) / MEDIUM.dayMs });
}

export function vocabularyFor(seed) {
  return VOCABULARY.map((word, index) => ({ word, index, rank: h32(`${seed}/word/${word}`) }))
    .sort((a, b) => a.rank - b.rank || a.index - b.index).map(({ word }) => word);
}

export function declaration(context) {
  if (!Number.isSafeInteger(context?.utcDay) || !Number.isFinite(context.rotation) || context.rotation < 0 || context.rotation >= 1) throw new TypeError('invalid earth context');
  const seed = `${PIECE.seed}/earth-day/${context.utcDay}`;
  const law = Law(seed), zero = { t: 0 };
  const count = Math.max(3, law.n('holding.modes', zero, MEDIUM.modeCapacity));
  const basis = Array.from({ length: MEDIUM.maxHarmonic ** 2 }, (_, i) => ({ p: i % MEDIUM.maxHarmonic + 1, q: Math.floor(i / MEDIUM.maxHarmonic) + 1, index: i }))
    .filter(({ p, q }) => p !== q)
    .sort((a, b) => h32(`${seed}/basis/${a.index}`) - h32(`${seed}/basis/${b.index}`) || a.index - b.index);
  const modes = basis.slice(0, count).map(({ p, q }, i) => ({ p, q, name: `holding.mode.${i}` }));
  return Object.freeze({ seed, context: { ...context }, modes, vocabulary: vocabularyFor(seed), peak: count, periodSec: law.periodSec });
}

export function wordState(declared, elapsedMs) {
  const dwell = MEDIUM.cycleMs / declared.vocabulary.length;
  const elapsed = Math.max(0, elapsedMs);
  const index = Math.floor(elapsed / dwell) % declared.vocabulary.length;
  const into = elapsed % dwell;
  const fade = Math.min(MEDIUM.transitionMs, dwell / 3);
  const opacity = elapsed < fade ? 1 : into < fade ? into / fade : into > dwell - fade ? (dwell - into) / fade : 1;
  return { index, word: declared.vocabulary[index], opacity: elapsed === 0 ? 1 : opacity };
}

export function termsAt(declared, elapsedSeconds, law = Law(declared.seed)) {
  const context = { t: elapsedSeconds / declared.modes.length + declared.context.rotation * declared.periodSec };
  return declared.modes.map(({ p, q, name }) => ({ p, q, coef: law.bal(name, context) }));
}
