/**
 * field.js — the plate's displacement, drawn by whatever this machine has.
 *
 * The 2D path in `app.js` is correct and it is the floor of this chain, not a
 * competitor to it. What it cannot do is resolve the node line: it evaluates
 * w(u,v) at 96×96 cell centres and the browser interpolates between them, so
 * the zero set — the one feature the picture exists to show, the line the sand
 * collects on — is smeared over half a cell, which at plate size is several
 * pixels of "somewhere around here".
 *
 * So this renderer does not upload a picture of the field. It uploads the field
 * itself: the (p, q, coefficient) terms `flattenModes` already produces, and the
 * fragment stage evaluates
 *
 *     w(u,v) = Σ coef · sin(p·π·u) · sin(q·π·v)
 *
 * per PIXEL. Sharp node lines at any size, from the same numbers the solver
 * produced, with no grid in the chain at all. `sampleField` here is the same
 * expression in JavaScript, and the test suite holds it against
 * `displacementField` cell by cell — that is what makes "the two paths are the
 * same physics" a checked statement rather than an intention.
 *
 * The thickness goes the other way: it is smooth, it changes only when the body
 * changes, and it is drawn as a dark wash. A grid and a texture are right for
 * it, and re-uploading it every frame would be work for nothing.
 *
 * Four ways to draw, in order, each one a real renderer:
 *
 *   webgpu    the newer path, same arithmetic, in WGSL
 *   webgl2    the same shader in GLSL ES 3.00
 *   webgl1    the same shader in GLSL ES 1.00, term count sized to the device
 *   canvas2d  the 96×96 sampling the page already had, with this palette
 *
 * Choosing is a pure function (`chooseBackend`) so it can be tested against a
 * made-up machine; building is separate, and a backend that is present but
 * fails to build falls through to the next one with the reason kept.
 *
 * No motion is added anywhere in this file. The only thing that moves is the
 * plate, and the plate moves because you hit it.
 */

import {
  MAX_TERMS, MAG_EXPONENT, ORDER_LIFT_FRACTION, capForDevice,
  vertexSource, fragmentSource, wgslSource
} from './shaders.js';

export { MAX_TERMS, MAG_EXPONENT, ORDER_LIFT_FRACTION, capForDevice, vertexSource, fragmentSource, wgslSource };

// ── the palette, which is the whole colour vocabulary ────────────────────────
//
// These are app.css's tokens, not a second opinion about them. Colour means the
// sign of the displacement and nothing else, so there are exactly four numbers
// here: the colour of zero, how far the thickness may lift it, and the two
// antinodes it may travel to.

/** `#rrggbb` to three numbers in 0..255. */
export function hexToRgb(hex) {
  const clean = String(hex).replace('#', '').trim();
  const full = clean.length === 3 ? clean.split('').map(c => c + c).join('') : clean;
  const n = Number.parseInt(full, 16);
  if (!Number.isFinite(n) || full.length !== 6) throw new Error(`not a colour: ${hex}`);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

export const PALETTE = {
  node: hexToRgb('#0b0e10'),   // zero displacement, and the thinnest sheet
  plus: hexToRgb('#e0714a'),   // full positive excursion
  minus: hexToRgb('#4b6cb7'),  // full negative excursion
  // How far the thickest part of the sheet lifts off `node` while at rest. The
  // top of this range is where the 2D path already sat, so a body looks like
  // itself whichever renderer draws it.
  lift: [55, 54.6, 57.9]
};

/** Physical optical & resonant palettes per material */
export const MATERIAL_PALETTES = {
  glass: {
    node: hexToRgb('#090d10'),
    plus: hexToRgb('#67e8f9'),   // quartz/cyan antinode
    minus: hexToRgb('#3b82f6'),  // refractive cobalt
    lift: [45, 65, 78]
  },
  bronze: {
    node: hexToRgb('#0b0e10'),
    plus: hexToRgb('#e0714a'),   // warm resonant copper
    minus: hexToRgb('#4b6cb7'),  // patina cyan
    lift: [55, 54.6, 57.9]
  },
  steel: {
    node: hexToRgb('#080a0c'),
    plus: hexToRgb('#f59e0b'),   // tempered amber
    minus: hexToRgb('#60a5fa'),  // cold steel blue
    lift: [52, 56, 62]
  },
  maple: {
    node: hexToRgb('#0c0c0e'),
    plus: hexToRgb('#d97706'),   // warm amber grain
    minus: hexToRgb('#059669'),  // deep forest walnut
    lift: [62, 50, 38]
  },
  rosewood: {
    node: hexToRgb('#0c0a0c'),
    plus: hexToRgb('#e11d48'),   // rosewood crimson
    minus: hexToRgb('#6366f1'),  // deep resonant violet
    lift: [58, 38, 42]
  },
  skin: {
    node: hexToRgb('#0e0c0b'),
    plus: hexToRgb('#ea580c'),   // warm vellum parchment
    minus: hexToRgb('#0284c7'),  // earth resonance
    lift: [48, 42, 38]
  },
  glassfibre: {
    node: hexToRgb('#06090a'),
    plus: hexToRgb('#10b981'),   // composite emerald
    minus: hexToRgb('#8b5cf6'),  // carbon violet
    lift: [38, 52, 58]
  },
  aluminium: {
    node: hexToRgb('#080a0d'),
    plus: hexToRgb('#38bdf8'),   // acoustic silver-cyan
    minus: hexToRgb('#f43f5e'),  // anodic rose
    lift: [60, 68, 74]
  }
};

export function paletteFor(material) {
  return MATERIAL_PALETTES[material] || PALETTE;
}

const unit = rgb => [rgb[0] / 255, rgb[1] / 255, rgb[2] / 255];
const orderOf = value => Math.max(0, Math.min(1, Number(value) || 0));

export function renderField(field, frame, order = 0) {
  field.render({ ...frame, order: orderOf(order) });
}

/**
 * The colour of one point of the plate: displacement in −1..1, thickness in 0..1.
 *
 * Zero displacement on the thinnest sheet is exactly `--node`. Full excursion is
 * exactly `--plus` or `--minus`. Everything between is a straight mix, curved
 * only in magnitude so a quiet ring still shows and a loud one does not clip.
 * The sign is load-bearing: it is the only thing colour is allowed to mean.
 */
export function colourAt(d, t, palette = PALETTE, order = 0) {
  const clamped = Math.max(-1, Math.min(1, Number(d) || 0));
  const wash = Math.max(0, Math.min(1, Number(t) || 0));
  const mag = Math.pow(Math.abs(clamped), MAG_EXPONENT);
  const tint = clamped >= 0 ? palette.plus : palette.minus;
  const orderMix = orderOf(order) * ORDER_LIFT_FRACTION;
  const out = [0, 0, 0];
  for (let i = 0; i < 3; i++) {
    const rest = palette.node[i] + wash * palette.lift[i];
    const orderedTint = orderMix > 0 ? tint[i] + orderMix * (palette.lift[i] - tint[i]) : tint[i];
    out[i] = rest + mag * (orderedTint - rest);
  }
  return out;
}

// ── the shader's arithmetic, in JavaScript ───────────────────────────────────

/**
 * w(u,v) from the sine terms — the exact expression the fragment stage runs.
 *
 * This is not a reimplementation to be kept in step by hand; it is the mirror
 * the tests hold against `displacementField`, and it is what the 2D backend
 * evaluates when the caller has not already sampled the field. If it and the
 * shader ever disagree, one of them is drawing a different instrument.
 */
export function sampleField(terms, u, v) {
  let w = 0;
  for (let i = 0; i < terms.length; i++) {
    const t = terms[i];
    w += t.coef * Math.sin(t.p * Math.PI * u) * Math.sin(t.q * Math.PI * v);
  }
  return w;
}

/**
 * The velocity of the air over the plate at this point, as a perpendicular
 * gradient of the displacement.
 *
 * Taking w as a stream function is what makes this exact rather than decorative:
 * a field built as the perpendicular gradient of a scalar has zero divergence
 * identically, so it is an incompressible flow with no pressure projection to
 * run. The contours of w are its streamlines, which puts the flow along the node
 * lines and a vortex at every antinode.
 *
 * Kept beside `sampleField` and derived from the same basis, because the four
 * backends have to draw one instrument and the JS path is the one that can be
 * read in a test.
 */
export function flowField(terms, u, v) {
  let du = 0;
  let dv = 0;
  for (let i = 0; i < terms.length; i++) {
    const t = terms[i];
    const a = t.p * Math.PI;
    const b = t.q * Math.PI;
    du += t.coef * a * Math.cos(a * u) * Math.sin(b * v);
    dv += t.coef * b * Math.sin(a * u) * Math.cos(b * v);
  }
  return { u: dv === 0 ? 0 : -dv, v: du === 0 ? 0 : du };
}

/**
 * The field as it is actually drawn: the modal sum, read at a point the flow has
 * already carried the sample to.
 *
 *   flow = 0   the linear layer. Superposition, nothing else. A plate.
 *   flow > 0   the nonlinear layer. The field advects itself, which is the
 *              (u·∇)u term and the only nonlinearity in Navier–Stokes.
 *
 * Two backward steps: semi-Lagrangian, because tracing back along the flow is
 * unconditionally stable at any step size, and two because one only shears where
 * two begin to fold.
 */
export function advectedField(terms, u, v, flow = 0) {
  if (!(flow > 0)) return sampleField(terms, u, v);
  let p = { u, v };
  for (let step = 0; step < 2; step++) {
    const q = flowField(terms, p.u, p.v);
    p = {
      u: Math.min(1, Math.max(0, p.u - flow * q.u)),
      v: Math.min(1, Math.max(0, p.v - flow * q.v))
    };
  }
  return sampleField(terms, p.u, p.v);
}

/** The same sum straight out of a packed buffer, so the packing is testable too. */
export function unpackTerms(data, count, stride = 3) {
  const out = [];
  for (let i = 0; i < count; i++) {
    out.push({ p: data[i * stride], q: data[i * stride + 1], coef: data[i * stride + 2] });
  }
  return out;
}

/**
 * Terms into the buffer the shader reads.
 *
 * The cap is a real limit — a WebGL1 device may only promise sixteen uniform
 * vectors — so going over it has to be a reported fact rather than a silent
 * shortening. What survives is the loudest terms, because dropping the smallest
 * coefficients is the least wrong reduction of a sum; what is lost is counted
 * and weighed, and the renderer hands both back from `render`.
 */
export function packTerms(terms, { cap = MAX_TERMS, stride = 3 } = {}) {
  const clean = [];
  for (const t of terms || []) {
    const p = Number(t.p), q = Number(t.q), coef = Number(t.coef);
    // A term that is not a number is not a shape; it must contribute nothing
    // rather than turn the whole field into NaN.
    if (!Number.isFinite(p) || !Number.isFinite(q) || !Number.isFinite(coef)) continue;
    if (coef === 0) continue;
    clean.push({ p, q, coef });
  }

  let kept = clean;
  let dropped = 0;
  let droppedWeight = 0;
  if (clean.length > cap) {
    const byLoudness = [...clean].sort((a, b) => Math.abs(b.coef) - Math.abs(a.coef));
    kept = byLoudness.slice(0, cap);
    const lost = byLoudness.slice(cap);
    dropped = lost.length;
    for (const t of lost) droppedWeight += Math.abs(t.coef);
  }

  const data = new Float32Array(cap * stride);
  for (let i = 0; i < kept.length; i++) {
    data[i * stride] = kept[i].p;
    data[i * stride + 1] = kept[i].q;
    data[i * stride + 2] = kept[i].coef;
  }
  return { data, count: kept.length, kept, dropped, droppedWeight, cap, stride };
}

/**
 * The thickness field as one byte per cell, ready to be a texture.
 *
 * `thicknessField` writes u fastest — `field[a*n + b]` with a the u index — and
 * a texture wants rows of constant v, so this transposes. Getting that backwards
 * draws a body that is the transpose of the one you are playing, which looks
 * plausible and is wrong.
 */
export function packThickness(thick, n, out = null) {
  const bytes = out && out.length === n * n ? out : new Uint8Array(n * n);
  const span = Math.max(1e-12, thick.hi - thick.lo);
  for (let b = 0; b < n; b++) {
    for (let a = 0; a < n; a++) {
      const t = (thick.field[a * n + b] - thick.lo) / span;
      bytes[b * n + a] = Math.max(0, Math.min(255, Math.round(t * 255)));
    }
  }
  return bytes;
}

// ── the newer path's uniform block ───────────────────────────────────────────
//
// One buffer, laid out by hand because the layout rules are the layout rules.
// Offsets are in floats; every vector starts on a four-float boundary, which is
// why the colours are carried as vec4 and the terms are padded to four.

// `uFlow` and `uOrder` sit after `uTermCount`. A vec4 in this
// block has to start on a four-float boundary, so the two floats after them are
// padding rather than a mistake — scale, termCount, flow and order fill 6..9, and
// the next vec4 cannot begin until 12.
export const UNIFORM_FLOATS = {
  rect: 0, viewport: 4, scale: 6, termCount: 7, flow: 8, order: 9,
  node: 12, lift: 16, plus: 20, minus: 24, terms: 28
};

export function uniformFloatCount(cap = MAX_TERMS) {
  return UNIFORM_FLOATS.terms + cap * 4;
}

export function packUniforms({ rect, viewport, scale, termCount, flow = 0, order = 0, terms, palette = PALETTE, cap = MAX_TERMS }) {
  const out = new Float32Array(uniformFloatCount(cap));
  out.set([rect.ox, rect.oy, rect.sw, rect.sh], UNIFORM_FLOATS.rect);
  out.set([viewport.w, viewport.h], UNIFORM_FLOATS.viewport);
  out[UNIFORM_FLOATS.scale] = scale;
  out[UNIFORM_FLOATS.termCount] = termCount;
  out[UNIFORM_FLOATS.flow] = flow;
  out[UNIFORM_FLOATS.order] = orderOf(order);
  out.set(unit(palette.node), UNIFORM_FLOATS.node);
  out.set(unit(palette.lift), UNIFORM_FLOATS.lift);
  out.set(unit(palette.plus), UNIFORM_FLOATS.plus);
  out.set(unit(palette.minus), UNIFORM_FLOATS.minus);
  if (terms) out.set(terms.subarray(0, cap * 4), UNIFORM_FLOATS.terms);
  return out;
}

// ── which renderer this machine gets ─────────────────────────────────────────

/** The chain, best first. Every step draws the real field; none is a stand-in. */
export const BACKENDS = ['webgpu', 'webgl2', 'webgl1', 'canvas2d'];

/** What each one is, in words an artist would use if they ever saw them. */
export const BACKEND_SAYS = {
  webgpu: 'drawn on the graphics chip, the newer way',
  webgl2: 'drawn on the graphics chip',
  webgl1: 'drawn on the graphics chip, the older way',
  canvas2d: 'drawn on the processor',
  none: 'nothing on this machine can draw the plate'
};

const ABSENT = 'this machine does not offer it';

/**
 * Pick a backend from what is available. Pure, so it can be asked about a
 * machine that does not exist.
 *
 * `prefer` forces one when it is available, which is how you look at a
 * particular path on a machine that would have chosen a better one. Asking for
 * something absent is not an error — you get the best that is actually there,
 * and the reason says so.
 */
export function chooseBackend(caps = {}) {
  const available = new Set(BACKENDS.filter(name => Boolean(caps[name])));
  const rejected = [];

  if (caps.prefer && available.has(caps.prefer)) {
    for (const name of BACKENDS) {
      if (name === caps.prefer) break;
      rejected.push({ name, why: available.has(name) ? 'a different one was asked for' : ABSENT });
    }
    return { mode: caps.prefer, why: BACKEND_SAYS[caps.prefer], asked: true, chain: BACKENDS.filter(n => available.has(n)), rejected };
  }

  for (const name of BACKENDS) {
    if (available.has(name)) {
      return {
        mode: name,
        why: BACKEND_SAYS[name],
        asked: false,
        chain: BACKENDS.filter(n => available.has(n)),
        rejected
      };
    }
    rejected.push({ name, why: ABSENT });
  }
  return { mode: 'none', why: BACKEND_SAYS.none, asked: false, chain: [], rejected };
}

/** What this machine actually has. Probed once, on a canvas nobody sees. */
export function detectCapabilities(win = globalThis) {
  const caps = { webgpu: false, webgl2: false, webgl1: false, canvas2d: false, reducedMotion: false };
  const doc = win.document;
  if (!doc) return caps;
  try {
    caps.webgpu = Boolean(win.navigator && win.navigator.gpu);
  } catch { caps.webgpu = false; }
  // A FRESH canvas per probe, and this is not tidiness. A canvas keeps the first
  // context kind it is ever handed, so probing one canvas for 'webgl2' and then
  // '2d' asks the second question of an element that can no longer answer it:
  // '2d' comes back null on every machine that has WebGL, and reports available
  // only on a machine that has nothing else. Exactly backwards, and it would have
  // left the last rung of the chain missing on precisely the machines that need
  // it. Found live 2026-08-18 — the detector said canvas2d: false in Chrome.
  const has = (kind) => {
    try {
      const probe = doc.createElement('canvas');
      probe.width = 1; probe.height = 1;
      return Boolean(probe.getContext(kind));
    } catch { return false; }
  };
  caps.webgl2 = has('webgl2');
  caps.webgl1 = caps.webgl2 || has('webgl') || has('experimental-webgl');
  caps.canvas2d = has('2d');
  try {
    caps.reducedMotion = Boolean(win.matchMedia && win.matchMedia('(prefers-reduced-motion: reduce)').matches);
  } catch { caps.reducedMotion = false; }
  return caps;
}

// ── the shared shape of a frame ──────────────────────────────────────────────

/** Uniform names the GL backends look up. The test holds the shaders to this list. */
export const VERTEX_UNIFORMS = ['uRect', 'uViewport'];
// uOrder comes from frame.order, the caller's local Kuramoto R in 0..1.
export const FRAGMENT_UNIFORMS = ['uTerms', 'uTermCount', 'uScale', 'uFlow', 'uOrder', 'uThick', 'uNode', 'uLift', 'uPlus', 'uMinus'];

const DEFAULT_N = 96;

/** How much to divide the field by so the picture uses its full range without clipping. */
function scaleFor(peak) {
  return peak > 1e-7 ? 1 / peak : 0;
}

/** The peak of the field, when the caller has not already measured one. */
function peakOfTerms(terms, n = 32) {
  let peak = 0;
  for (let a = 0; a < n; a++) {
    const u = (a + 0.5) / n;
    for (let b = 0; b < n; b++) {
      const w = Math.abs(sampleField(terms, u, (b + 0.5) / n));
      if (w > peak) peak = w;
    }
  }
  return peak;
}

// ── canvas2d: the floor of the chain, and still the whole picture ────────────

function build2d(canvas) {
  const ctx = canvas.getContext('2d');
  if (!ctx) throw new Error('this page cannot get a drawing surface');

  let n = 0;
  let buf = null, bufCtx = null, pixels = null;
  const ensure = (size) => {
    if (n === size) return;
    n = size;
    buf = canvas.ownerDocument.createElement('canvas');
    buf.width = n; buf.height = n;
    bufCtx = buf.getContext('2d');
    pixels = bufCtx.createImageData(n, n);
  };

  return {
    mode: 'canvas2d',
    ink: ctx,
    overlay: canvas,
    cap: MAX_TERMS,
    resize(w, h, dpr) {
      if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
        canvas.width = Math.round(w * dpr);
        canvas.height = Math.round(h * dpr);
      }
    },
    render(frame, packed) {
      const size = frame.n || DEFAULT_N;
      ensure(size);
      const { rect, thick, dpr } = frame;
      // Same palette the GPU paths pack into uniforms; without it this backend
      // silently painted every material in the default one.
      const pal = frame.palette || PALETTE;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, frame.viewport.w, frame.viewport.h);

      const span = Math.max(1e-12, thick.hi - thick.lo);
      const scale = scaleFor(frame.peak);
      const terms = packed.kept;
      const px = pixels.data;
      for (let b = 0; b < size; b++) {
        const v = (b + 0.5) / size;
        for (let a = 0; a < size; a++) {
          const t = (thick.field[a * size + b] - thick.lo) / span;
          // `disp` is a field the caller already sampled, so it is taken as it
          // stands; otherwise the same advected sum the shaders evaluate.
          const w = frame.disp
            ? frame.disp[a * size + b]
            : advectedField(terms, (a + 0.5) / size, v, frame.flow || 0);
          const [r, g, bl] = colourAt(w * scale, t, pal, frame.order);
          const i = (b * size + a) * 4;
          px[i] = r; px[i + 1] = g; px[i + 2] = bl; px[i + 3] = 255;
        }
      }
      bufCtx.putImageData(pixels, 0, 0);
      // The field is continuous, so it is drawn smooth. Showing the cells would
      // be a claim about the physics that is not true.
      ctx.imageSmoothingEnabled = true;
      ctx.imageSmoothingQuality = 'high';
      ctx.drawImage(buf, 0, 0, size, size, rect.ox, rect.oy, rect.sw, rect.sh);
    },
    destroy() { /* the page owns this canvas */ }
  };
}

// ── webgl1 / webgl2: one shader family, two spellings ────────────────────────

function compile(gl, kind, src) {
  const shader = gl.createShader(kind);
  gl.shaderSource(shader, src);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    const log = gl.getShaderInfoLog(shader);
    gl.deleteShader(shader);
    throw new Error(`the field shader did not build: ${log}`);
  }
  return shader;
}

function buildGl(canvas, { glsl3 }) {
  const attrs = { alpha: true, antialias: false, depth: false, stencil: false, premultipliedAlpha: true };
  const gl = glsl3
    ? canvas.getContext('webgl2', attrs)
    : (canvas.getContext('webgl', attrs) || canvas.getContext('experimental-webgl', attrs));
  if (!gl) throw new Error('this machine did not hand over the graphics chip');

  const cap = capForDevice(gl.getParameter(gl.MAX_FRAGMENT_UNIFORM_VECTORS));
  const vs = vertexSource({ glsl3 });
  const fs = fragmentSource({ glsl3, maxTerms: cap });

  let program = null, quad = null, texture = null, locations = null;
  let uploaded = null;   // which thickness is on the chip right now
  let lost = false;

  const make = () => {
    program = gl.createProgram();
    gl.attachShader(program, compile(gl, gl.VERTEX_SHADER, vs));
    gl.attachShader(program, compile(gl, gl.FRAGMENT_SHADER, fs));
    gl.bindAttribLocation(program, 0, 'aPos');
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      throw new Error(`the field shader did not link: ${gl.getProgramInfoLog(program)}`);
    }
    locations = {};
    for (const name of [...VERTEX_UNIFORMS, ...FRAGMENT_UNIFORMS]) {
      locations[name] = gl.getUniformLocation(program, name === 'uTerms' ? 'uTerms[0]' : name);
    }
    quad = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, quad);
    // Two triangles as one strip, in plate space: (0,0) (1,0) (0,1) (1,1).
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([0, 0, 1, 0, 0, 1, 1, 1]), gl.STATIC_DRAW);

    texture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    uploaded = null;
  };
  make();

  const onLost = (e) => { e.preventDefault(); lost = true; };
  const onRestored = () => { lost = false; make(); };
  canvas.addEventListener('webglcontextlost', onLost);
  canvas.addEventListener('webglcontextrestored', onRestored);

  const format = glsl3 ? gl.R8 : gl.LUMINANCE;
  const source = glsl3 ? gl.RED : gl.LUMINANCE;

  return {
    mode: glsl3 ? 'webgl2' : 'webgl1',
    ink: null,
    overlay: null,
    cap,
    get lost() { return lost; },
    resize(w, h, dpr) {
      const pw = Math.round(w * dpr), ph = Math.round(h * dpr);
      if (canvas.width !== pw || canvas.height !== ph) {
        canvas.width = pw;
        canvas.height = ph;
      }
      gl.viewport(0, 0, canvas.width, canvas.height);
    },
    render(frame, packed) {
      if (lost) return;
      const size = frame.n || DEFAULT_N;
      gl.viewport(0, 0, canvas.width, canvas.height);
      const pal = frame.palette || PALETTE;
      const [nr, ng, nb] = unit(pal.node);
      gl.clearColor(nr, ng, nb, 1);
      gl.clear(gl.COLOR_BUFFER_BIT);

      gl.useProgram(program);
      gl.bindBuffer(gl.ARRAY_BUFFER, quad);
      gl.enableVertexAttribArray(0);
      gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);

      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, texture);
      if (uploaded !== frame.thick) {
        const bytes = packThickness(frame.thick, size);
        gl.texImage2D(gl.TEXTURE_2D, 0, format, size, size, 0, source, gl.UNSIGNED_BYTE, bytes);
        uploaded = frame.thick;
      }

      gl.uniform4f(locations.uRect, frame.rect.ox, frame.rect.oy, frame.rect.sw, frame.rect.sh);
      gl.uniform2f(locations.uViewport, frame.viewport.w, frame.viewport.h);
      gl.uniform3fv(locations.uTerms, packed.data);
      gl.uniform1i(locations.uTermCount, packed.count);
      gl.uniform1f(locations.uScale, scaleFor(frame.peak));
      gl.uniform1f(locations.uFlow, frame.flow || 0);
      gl.uniform1f(locations.uOrder, orderOf(frame.order));
      gl.uniform1i(locations.uThick, 0);
      gl.uniform3fv(locations.uNode, unit(pal.node));
      gl.uniform3fv(locations.uLift, unit(pal.lift));
      gl.uniform3fv(locations.uPlus, unit(pal.plus));
      gl.uniform3fv(locations.uMinus, unit(pal.minus));

      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    },
    destroy() {
      canvas.removeEventListener('webglcontextlost', onLost);
      canvas.removeEventListener('webglcontextrestored', onRestored);
      gl.deleteBuffer(quad);
      gl.deleteTexture(texture);
      gl.deleteProgram(program);
    }
  };
}

// ── webgpu: the same arithmetic, the newer way ───────────────────────────────

function buildGpu(canvas, { device, win }) {
  const context = canvas.getContext('webgpu');
  if (!context) throw new Error('this machine did not hand over the newer graphics path');
  const gpu = (win.navigator && win.navigator.gpu) || null;
  const format = gpu && gpu.getPreferredCanvasFormat ? gpu.getPreferredCanvasFormat() : 'bgra8unorm';
  context.configure({ device, format, alphaMode: 'premultiplied' });

  const cap = MAX_TERMS;
  const module = device.createShaderModule({ code: wgslSource({ maxTerms: cap }) });
  const pipeline = device.createRenderPipeline({
    layout: 'auto',
    vertex: { module, entryPoint: 'vertexMain' },
    fragment: { module, entryPoint: 'fragmentMain', targets: [{ format }] },
    primitive: { topology: 'triangle-strip' }
  });

  const uniforms = device.createBuffer({
    size: uniformFloatCount(cap) * 4,
    usage: 0x40 | 0x8   // UNIFORM | COPY_DST, spelled out so this file needs no globals
  });
  const sampler = device.createSampler({ magFilter: 'linear', minFilter: 'linear' });

  let texture = null, bindGroup = null, uploaded = null, textureSize = 0;

  const ensureTexture = (size) => {
    if (texture && textureSize === size) return;
    if (texture) texture.destroy();
    textureSize = size;
    texture = device.createTexture({
      size: [size, size, 1],
      format: 'r8unorm',
      usage: 0x4 | 0x2   // TEXTURE_BINDING | COPY_DST
    });
    bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: uniforms } },
        { binding: 1, resource: sampler },
        { binding: 2, resource: texture.createView() }
      ]
    });
    uploaded = null;
  };

  return {
    mode: 'webgpu',
    ink: null,
    overlay: null,
    cap,
    resize(w, h, dpr) {
      const pw = Math.max(1, Math.round(w * dpr)), ph = Math.max(1, Math.round(h * dpr));
      if (canvas.width !== pw || canvas.height !== ph) {
        canvas.width = pw;
        canvas.height = ph;
      }
    },
    render(frame, packed) {
      const size = frame.n || DEFAULT_N;
      ensureTexture(size);

      if (uploaded !== frame.thick) {
        // Rows have to start on a 256-byte boundary, so the field is copied into
        // a padded block rather than handed over as it sits in memory.
        const bytes = packThickness(frame.thick, size);
        const stride = Math.ceil(size / 256) * 256;
        const padded = new Uint8Array(stride * size);
        for (let row = 0; row < size; row++) padded.set(bytes.subarray(row * size, row * size + size), row * stride);
        device.queue.writeTexture(
          { texture },
          padded,
          { bytesPerRow: stride, rowsPerImage: size },
          { width: size, height: size, depthOrArrayLayers: 1 }
        );
        uploaded = frame.thick;
      }

      device.queue.writeBuffer(uniforms, 0, packUniforms({
        rect: frame.rect,
        viewport: frame.viewport,
        scale: scaleFor(frame.peak),
        flow: frame.flow || 0,
        order: frame.order,
        termCount: packed.count,
        terms: packed.data,
        cap
      }));

      const [nr, ng, nb] = unit(PALETTE.node);
      const encoder = device.createCommandEncoder();
      const pass = encoder.beginRenderPass({
        colorAttachments: [{
          view: context.getCurrentTexture().createView(),
          clearValue: { r: nr, g: ng, b: nb, a: 1 },
          loadOp: 'clear',
          storeOp: 'store'
        }]
      });
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bindGroup);
      pass.draw(4);
      pass.end();
      device.queue.submit([encoder.finish()]);
    },
    destroy() {
      if (texture) texture.destroy();
      uniforms.destroy();
      try { context.unconfigure(); } catch { /* already gone */ }
    }
  };
}

// ── the factory ──────────────────────────────────────────────────────────────

/**
 * One call, then one call per frame.
 *
 *   const field = createFieldRenderer(canvas);
 *   field.render({ terms, thick, n, peak, rect, dpr });
 *   // then draw the plate outline and the partials on field.ink
 *
 * `terms` is `flattenModes(modes, weights).terms` — the same call the 2D path
 * already makes inside `displacementField`. `thick` is the object
 * `thicknessField` returns, unchanged, and it is re-uploaded only when the
 * object identity changes, which is exactly when the body changed. `rect` is
 * `plateRect()`. If the caller already sampled the field it can pass `disp` as
 * well and the 2D backend will use it instead of sampling again; the graphics
 * paths ignore it, because they do not sample.
 *
 * The partials ladder and the plate outline are deliberately NOT in here. They
 * are ink — one hue, `--sand`, drawn as thin rectangles and lines — and putting
 * text-weight 2D drawing through a shader would be a second way to draw the
 * same thing for no gain. `ink` is a 2D context sized and cleared for you: on
 * the graphics paths it is a transparent layer above the field, and on the 2D
 * path it is the field's own context, so the caller's code is the same either
 * way.
 */
export function createFieldRenderer(canvas, options = {}) {
  const win = options.window || (canvas && canvas.ownerDocument && canvas.ownerDocument.defaultView) || globalThis;
  const caps = options.capabilities || detectCapabilities(win);
  const device = options.device || null;
  const wanted = chooseBackend({ ...caps, webgpu: Boolean(caps.webgpu && device), prefer: options.prefer });

  const tried = [];
  let backend = null;
  const order = wanted.mode === 'none' ? [] : [wanted.mode, ...wanted.chain.filter(m => m !== wanted.mode)];

  for (const mode of order) {
    try {
      if (mode === 'webgpu') backend = buildGpu(canvas, { device, win });
      else if (mode === 'webgl2') backend = buildGl(canvas, { glsl3: true });
      else if (mode === 'webgl1') backend = buildGl(canvas, { glsl3: false });
      else if (mode === 'canvas2d') backend = build2d(canvas);
      if (backend) break;
    } catch (err) {
      // A canvas keeps the first context kind it was given, so a backend that
      // half-built cannot be retried on this element — but the reason is worth
      // more than the attempt, and the next one down still gets its turn.
      tried.push({ mode, why: err.message });
    }
  }

  if (!backend) {
    return {
      mode: 'none',
      why: BACKEND_SAYS.none,
      cap: 0,
      ink: null,
      tried,
      resize() {},
      render() { return { mode: 'none', terms: 0, dropped: 0 }; },
      destroy() {}
    };
  }

  // The ink layer. On the graphics paths the field canvas is spoken for, so the
  // outline and the partials get their own transparent sheet directly above it.
  let madeOverlay = null;
  let ink = backend.ink;
  if (!ink) {
    const overlay = options.overlay || (() => {
      const made = canvas.ownerDocument.createElement('canvas');
      made.setAttribute('aria-hidden', 'true');
      const s = made.style;
      s.position = 'fixed';
      s.left = '0'; s.top = '0'; s.right = '0'; s.bottom = '0';
      s.width = '100%'; s.height = '100%';
      s.display = 'block';
      s.pointerEvents = 'none';
      s.zIndex = '1';
      canvas.insertAdjacentElement('afterend', made);
      madeOverlay = made;
      return made;
    })();
    ink = overlay.getContext('2d');
    backend.overlay = overlay;
  }

  const api = {
    mode: backend.mode,
    why: BACKEND_SAYS[backend.mode],
    cap: backend.cap,
    ink,
    overlay: backend.overlay,
    tried,
    chain: wanted.chain,

    /** Size the drawing surfaces to the element. Cheap, and safe every frame. */
    resize(dpr) {
      const w = canvas.clientWidth, h = canvas.clientHeight;
      const ratio = dpr || Math.min(2, win.devicePixelRatio || 1);
      backend.resize(w, h, ratio);
      if (backend.overlay && backend.overlay !== canvas) {
        const pw = Math.round(w * ratio), ph = Math.round(h * ratio);
        if (backend.overlay.width !== pw || backend.overlay.height !== ph) {
          backend.overlay.width = pw;
          backend.overlay.height = ph;
        }
      }
      return { w, h, dpr: ratio };
    },

    render(frame) {
      const dpr = frame.dpr || Math.min(2, win.devicePixelRatio || 1);
      const viewport = api.resize(dpr);
      const terms = frame.terms || [];
      const packed = packTerms(terms, { cap: backend.cap, stride: backend.mode === 'webgpu' ? 4 : 3 });
      const peak = Number.isFinite(frame.peak) ? frame.peak : peakOfTerms(packed.kept);

      backend.render({
        ...frame,
        viewport: { w: viewport.w, h: viewport.h },
        dpr,
        peak,
        n: frame.n || DEFAULT_N
      }, packed);

      // The ink layer is handed back clear and in CSS pixels, so the caller draws
      // the outline and the partials in the same coordinates it computed the rect in.
      if (ink && backend.overlay !== canvas) {
        ink.setTransform(dpr, 0, 0, dpr, 0, 0);
        ink.clearRect(0, 0, viewport.w, viewport.h);
      }

      return {
        mode: backend.mode,
        terms: packed.count,
        dropped: packed.dropped,
        droppedWeight: packed.droppedWeight,
        cap: backend.cap
      };
    },

    destroy() {
      backend.destroy();
      if (madeOverlay && madeOverlay.parentNode) madeOverlay.parentNode.removeChild(madeOverlay);
    }
  };
  return api;
}

/**
 * The same factory, having first asked for the newer graphics path.
 *
 * That request is asynchronous and a canvas only ever gets one kind of context,
 * so it cannot be done after the fact — you either wait for the answer or you
 * are on WebGL for the life of the page. Waiting costs one await at startup and
 * nothing afterwards.
 */
export async function createFieldRendererAsync(canvas, options = {}) {
  const win = options.window || (canvas && canvas.ownerDocument && canvas.ownerDocument.defaultView) || globalThis;
  const caps = options.capabilities || detectCapabilities(win);
  let device = options.device || null;
  if (!device && caps.webgpu) {
    try {
      const adapter = await win.navigator.gpu.requestAdapter();
      device = adapter ? await adapter.requestDevice() : null;
    } catch {
      device = null;   // no adapter is not a failure; it is the next step down
    }
  }
  return createFieldRenderer(canvas, { ...options, capabilities: caps, device });
}
