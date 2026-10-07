import { Law } from '../worlds/kernel/law.js';
import { Body, step, OBSERVE } from '../worlds/kernel/septet.js';
import { createFieldRenderer, renderField, sampleField, hexToRgb } from '../gl/field.js';
import { channels, GLASSES, fresnel, AIR } from '../ds/optics.js';
import { spectralScales, SPECTRAL } from '../ds/fold/refractMap.js';
import { stepCritical } from '../ds/spring.js';
import { PRIORS as LIMEN, trit, phase } from '../ds/limen.js';
import { declaration, earthContext, MEDIUM, THEME, termsAt, wordState } from './seed.mjs';
import { PIECE } from './world.js';

const canvas = document.querySelector('canvas');
const noun = document.querySelector('[data-noun]');
const pause = document.querySelector('[data-pause]');
const motion = matchMedia('(prefers-reduced-motion: reduce)');
const declared = declaration(earthContext(Date.now()));
const law = Law(declared.seed);
for (const [key, bend] of Object.entries(PIECE.hand)) law.bend(key, bend);
const body = Body(PIECE.dim, MEDIUM.particleCapacity);
body.budget = PIECE.budget(law, { t: 0 });
const field = createFieldRenderer(canvas, { prefer: 'canvas2d', capabilities: { canvas2d: true } });
const colors = [hexToRgb(THEME.sage), hexToRgb(THEME.violet), [193, 209, 218]];
const optics = channels(GLASSES.flint);
const split = spectralScales(GLASSES.flint, SPECTRAL.v);
const palette = { node: hexToRgb(THEME.paper), plus: [241, 245, 236], minus: [245, 237, 245], lift: [0, 0, 0] };
const thick = { field: new Float32Array(MEDIUM.sampleSide ** 2), lo: 0, hi: 1 };
let elapsed = 0, previous = null, lastDraw = -Infinity, raf = null, paused = false, disposed = false;
let pointer = { x: 0.5, y: 0.5 }, target = 0, exposure = { x: 0, v: 0 }, held = 0;
let width = innerWidth, height = innerHeight, simulationTime = 0;
const fixedStep = MEDIUM.frameMs / MEDIUM.secondMs;
const openingSteps = Math.floor(law.periodSec / fixedStep);
for (let i = 0; i < openingSteps; i++) step(body, law, { t: i * fixedStep, dt: fixedStep, d: PIECE.dim, couples: PIECE.couples });

function resize() {
  width = canvas.clientWidth; height = canvas.clientHeight;
  field.resize(Math.min(devicePixelRatio || 1, MEDIUM.maxDpr));
  draw();
}

function contours(terms, ink, alpha) {
  const n = MEDIUM.contourSide, values = new Float32Array((n + 1) ** 2);
  for (let y = 0; y <= n; y++) for (let x = 0; x <= n; x++) values[y * (n + 1) + x] = sampleField(terms, x / n, y / n);
  const lines = [];
  for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) {
    const corners = [[x, y], [x + 1, y], [x + 1, y + 1], [x, y + 1]];
    const crossings = [];
    for (let edge = 0; edge < corners.length; edge++) {
      const a = corners[edge], b = corners[(edge + 1) % corners.length];
      const va = values[a[1] * (n + 1) + a[0]], vb = values[b[1] * (n + 1) + b[0]];
      if ((va < 0) === (vb < 0) || va === vb) continue;
      const f = va / (va - vb);
      crossings.push([(a[0] + (b[0] - a[0]) * f) / n * width, (a[1] + (b[1] - a[1]) * f) / n * height]);
    }
    if (crossings.length === 4 && sampleField(terms, (x + 0.5) / n, (y + 0.5) / n) < 0) crossings.push(crossings.shift());
    for (let i = 0; i + 1 < crossings.length; i += 2) lines.push([crossings[i], crossings[i + 1]]);
  }
  for (let channel = 0; channel < colors.length; channel++) {
    const offset = Object.values(split)[channel] - split.g;
    ink.beginPath();
    for (const [a, b] of lines) { ink.moveTo(a[0] + offset, a[1] - offset); ink.lineTo(b[0] + offset, b[1] - offset); }
    ink.strokeStyle = `rgba(${colors[channel].join(',')},${alpha})`;
    ink.lineWidth = MEDIUM.edgeWidth + Math.abs(offset);
    ink.stroke();
  }
}

function draw() {
  const seconds = elapsed / MEDIUM.secondMs;
  const terms = termsAt(declared, seconds, law);
  const dpr = Math.min(devicePixelRatio || 1, MEDIUM.maxDpr);
  if (field.mode !== 'none' && width && height) {
    renderField(field, { terms, thick, n: MEDIUM.sampleSide, peak: declared.peak, flow: 0,
      palette, rect: { ox: 0, oy: 0, sw: width, sh: height }, viewport: { w: width, h: height }, dpr }, body.R);
    const ink = field.ink;
    const reflection = fresnel(Math.cos(phase(exposure.x, LIMEN.ridge.v, LIMEN.band.v)), AIR, optics.g).r;
    contours(terms, ink, Math.min(0.38, LIMEN.capture.v + reflection));
    const view = OBSERVE(body, law, { t: seconds, dt: 0, d: PIECE.dim });
    ink.fillStyle = `rgba(68,100,74,${LIMEN.capture.v + exposure.x * LIMEN.band.v})`;
    for (let i = 0; i < view.n; i++) {
      const u = view.x[i * view.d], v = view.x[i * view.d + 1];
      const radius = Math.max(0.5, Math.min(1.1, view.r[i] * Math.min(width, height)));
      for (const side of [-1, 1]) {
        const x = (side < 0 ? u * LIMEN.ridge.v - LIMEN.band.v : 1 + LIMEN.band.v - u * LIMEN.ridge.v) * width;
        const y = (side < 0 ? v : 1 - v) * height;
        ink.beginPath(); ink.arc(x, y, radius, 0, LIMEN.turn.v); ink.fill();
      }
    }
    canvas.style.opacity = String(0.52 + exposure.x * LIMEN.band.v);
  }
  const words = wordState(declared, elapsed);
  noun.textContent = `${words.word}.`;
  noun.style.opacity = String(motion.matches ? 1 : words.opacity);
  document.documentElement.dataset.fieldState = held < 0 ? 'below' : held > 0 ? 'above' : 'within';
}

function animate(now) {
  raf = null;
  if (disposed || paused || motion.matches || document.hidden) { previous = null; return; }
  const dt = previous === null ? 0 : Math.min(MEDIUM.maxStepSeconds, (now - previous) / MEDIUM.secondMs);
  previous = now; elapsed += dt * MEDIUM.secondMs;
  exposure = stepCritical(exposure, target, LIMEN.omega.v, dt);
  target *= Math.exp(-dt * LIMEN.driveOmega.v);
  held = trit(exposure.x, LIMEN.ridge.v, LIMEN.band.v, { prev: held });
  const fixed = MEDIUM.frameMs / MEDIUM.secondMs;
  while (simulationTime + fixed <= elapsed / MEDIUM.secondMs) {
    simulationTime += fixed;
    step(body, law, { t: simulationTime + openingSteps * fixedStep, dt: fixed, d: PIECE.dim, couples: PIECE.couples });
  }
  if (now - lastDraw >= MEDIUM.frameMs * MEDIUM.drawEveryFrames) { draw(); lastDraw = now; }
  raf = requestAnimationFrame(animate);
}

function start() { previous = null; if (!disposed && !paused && !motion.matches && !document.hidden && raf === null) raf = requestAnimationFrame(animate); }
function suspend() { if (raf !== null) cancelAnimationFrame(raf); raf = null; previous = null; }
function react(event) {
  const next = { x: event.clientX / Math.max(width, 1), y: event.clientY / Math.max(height, 1) };
  target = Math.min(1, target + Math.hypot(next.x - pointer.x, next.y - pointer.y)); pointer = next;
}
function pauseMotion() { paused = !paused; pause.textContent = paused ? 'Resume motion' : 'Pause motion'; pause.setAttribute('aria-pressed', String(paused)); paused ? suspend() : start(); }
function visibility() { document.hidden ? suspend() : start(); }
function preference() { motion.matches ? suspend() : start(); pause.hidden = motion.matches; draw(); }
function dispose() {
  disposed = true; suspend(); observer.disconnect(); field.destroy();
  removeEventListener('pointermove', react); removeEventListener('pointerdown', touch);
  document.removeEventListener('visibilitychange', visibility); motion.removeEventListener('change', preference);
}
function touch() { target = 1; }
const observer = new ResizeObserver(resize); observer.observe(canvas);
addEventListener('pointermove', react, { passive: true }); addEventListener('pointerdown', touch, { passive: true });
document.addEventListener('visibilitychange', visibility); motion.addEventListener('change', preference);
pause.addEventListener('click', pauseMotion);
addEventListener('pagehide', (event) => { if (event.persisted) suspend(); else dispose(); });
addEventListener('pageshow', (event) => { if (event.persisted) start(); });
preference(); resize(); start();
