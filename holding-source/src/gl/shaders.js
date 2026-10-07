/**
 * shaders.js — the field, as an expression a graphics chip can evaluate.
 *
 * The Canvas2D path samples w(u,v) on a 96×96 grid and lets the browser smooth
 * between the cells. That is honest about the physics but dishonest about the
 * node lines: the zero set is where the picture carries its information, and a
 * smoothed cell boundary puts it wherever the interpolation happens to land,
 * ±half a cell. Blown up to a 1400 px plate that is a 7 px smear on the one
 * feature that is supposed to be a line.
 *
 * So this path does not upload a sampled field at all. It uploads the sine
 * basis — the same (p, q, coefficient) triples `flattenModes` produces — and
 * evaluates
 *
 *     w(u,v) = Σ coef · sin(p·π·u) · sin(q·π·v)
 *
 * once per PIXEL. The node lines are then exactly where the sum vanishes at
 * that pixel, at any zoom, with no grid anywhere in the chain. It is also
 * cheaper than it sounds: the number of terms is bounded by the basis the
 * solver used, not by how many partials are ringing, which is the property
 * `kernel.resonance-shape.test.ts` pins.
 *
 * The thickness is the other way round. It changes only when the body changes,
 * it is smooth, and it is drawn as a dark grey wash, so a grid plus bilinear
 * sampling is the right answer for it and a texture is the right upload.
 *
 * Naming is a contract, not a style: every uniform is `uName`, every attribute
 * `aName`, every varying `vName`. `checkShaderSource` enforces it in both
 * directions, so a uniform the renderer sets but the shader never reads — and a
 * name the shader reads but nobody declared — are both test failures rather
 * than a silently black plate.
 */

/** The most sine terms any build of the shader will carry. */
export const MAX_TERMS = 64;

/**
 * How many terms this device can actually hold.
 *
 * WebGL1 only guarantees 16 fragment uniform vectors. Each term is one vec3 =
 * one vector, and the rest of the uniforms cost eleven more, so a fixed 64
 * would fail to link on exactly the small machines this chain exists for. Ask
 * the device and build the shader to fit.
 */
export function capForDevice(maxFragmentVectors) {
  const room = Math.floor(Number(maxFragmentVectors) || 0) - 12;
  if (!Number.isFinite(room)) return 8;
  return Math.max(8, Math.min(MAX_TERMS, room));
}

const PI_LITERAL = '3.141592653589793';

/** The shape of the excursion curve. Small motion still shows; loud does not clip. */
export const MAG_EXPONENT = 0.62;

/** The largest share of sign colour local order may pull toward lift. */
export const ORDER_LIFT_FRACTION = 0.35;

/**
 * Vertex stage: place the plate.
 *
 * The field owns a rectangle inside the canvas, not the whole canvas — the name
 * and the dock need room — so the quad is given in plate space (0..1 across the
 * plate) and mapped through the same rect the pointer uses. u runs left to
 * right, v runs top to bottom, which is the orientation `displacementField`
 * writes and the 2D path draws.
 */
export function vertexSource({ glsl3 = false } = {}) {
  const head = glsl3 ? '#version 300 es\n' : '';
  const IN = glsl3 ? 'in' : 'attribute';
  const OUT = glsl3 ? 'out' : 'varying';
  return `${head}${IN} vec2 aPos;
uniform vec4 uRect;
uniform vec2 uViewport;
${OUT} vec2 vUv;

void main() {
  vUv = aPos;
  vec2 px = uRect.xy + aPos * uRect.zw;
  gl_Position = vec4(px.x / uViewport.x * 2.0 - 1.0, 1.0 - px.y / uViewport.y * 2.0, 0.0, 1.0);
}
`;
}

/**
 * Fragment stage: the displacement at this pixel, coloured by its sign.
 *
 * Colour means one thing. Zero displacement is the body's own colour — the
 * thickness wash lifted off `uNode` — and the excursion mixes from there toward
 * `uPlus` one way and `uMinus` the other, reaching the palette colour exactly
 * at full excursion. A plate that is not ringing is `uNode` where it is
 * thinnest and stays dark everywhere else, which is the point: silence looks
 * like silence.
 */
export function fragmentSource({ glsl3 = false, maxTerms = MAX_TERMS } = {}) {
  const cap = Math.max(1, Math.floor(maxTerms));
  const head = glsl3 ? '#version 300 es\n' : '';
  const IN = glsl3 ? 'in' : 'varying';
  const sample = glsl3 ? 'texture' : 'texture2D';
  const outDecl = glsl3 ? 'out vec4 oColour;\n' : '';
  const write = glsl3 ? 'oColour' : 'gl_FragColor';
  return `${head}#ifdef GL_FRAGMENT_PRECISION_HIGH
precision highp float;
#else
precision mediump float;
#endif

const int TERM_CAP = ${cap};
const float PI = ${PI_LITERAL};

uniform vec3 uTerms[TERM_CAP];
uniform int uTermCount;
uniform float uScale;
uniform float uFlow;
uniform float uOrder;
uniform sampler2D uThick;
uniform vec3 uNode;
uniform vec3 uLift;
uniform vec3 uPlus;
uniform vec3 uMinus;

${IN} vec2 vUv;
${outDecl}
float wAt(vec2 p) {
  float w = 0.0;
  for (int i = 0; i < TERM_CAP; i++) {
    if (i >= uTermCount) break;
    vec3 term = uTerms[i];
    w += term.z * sin(term.x * PI * p.x) * sin(term.y * PI * p.y);
  }
  return w;
}

// The velocity of the air over the plate, as a PERPENDICULAR gradient.
//
// Taking the displacement as a stream function is what makes this exact rather
// than decorative: a field built as the perpendicular gradient of a scalar has
// zero divergence identically, so this is an incompressible flow with no
// pressure projection to run and no solver step to get wrong. The contours of w
// ARE the streamlines, which means the node lines — already the informative part
// of this picture — become the lines the flow follows, and every antinode
// becomes a vortex core with real circulation around it.
vec2 flowAt(vec2 p) {
  float dwdu = 0.0;
  float dwdv = 0.0;
  for (int i = 0; i < TERM_CAP; i++) {
    if (i >= uTermCount) break;
    vec3 term = uTerms[i];
    float a = term.x * PI;
    float b = term.y * PI;
    dwdu += term.z * a * cos(a * p.x) * sin(b * p.y);
    dwdv += term.z * b * sin(a * p.x) * cos(b * p.y);
  }
  return vec2(-dwdv, dwdu);
}

void main() {
  // The two layers, as one expression.
  //
  //   LINEAR      uFlow = 0. The modal sum, evaluated where the pixel is.
  //               Superposition, and nothing else. This is a plate.
  //   NONLINEAR   uFlow > 0. The sample point is carried along the flow the
  //               field itself induces before the field is read there — the
  //               field advecting itself, which is the (u·grad)u term and the
  //               only nonlinearity in Navier–Stokes. It is why turbulence
  //               exists, and it is why the bands shear here.
  //
  // Two semi-Lagrangian steps rather than one, because a single step only
  // shears and it takes a second to fold; and semi-Lagrangian rather than
  // forward differencing because tracing backwards along the flow is
  // unconditionally stable at any step size, which is the whole reason that
  // scheme is used.
  //
  // uFlow is the regime. At rest every term is zero, so the gradient is zero,
  // so the flow is zero — silence still looks like silence, without a special
  // case anywhere.
  vec2 p = vUv;
  if (uFlow > 0.0) {
    p -= uFlow * flowAt(p);
    p -= uFlow * flowAt(p);
    p = clamp(p, vec2(0.0), vec2(1.0));
  }

  float d = clamp(wAt(p) * uScale, -1.0, 1.0);
  float t = ${sample}(uThick, vUv).r;
  vec3 rest = uNode + t * uLift;
  float mag = pow(abs(d), ${MAG_EXPONENT});
  vec3 tint = d >= 0.0 ? uPlus : uMinus;
  float orderMix = clamp(uOrder, 0.0, 1.0) * ${ORDER_LIFT_FRACTION};
  if (orderMix > 0.0) tint = mix(tint, uLift, orderMix);
  ${write} = vec4(mix(rest, tint, mag), 1.0);
}
`;
}

/**
 * The same field, for the newer graphics path.
 *
 * One module, both stages, one uniform block. The arithmetic is transliterated
 * line for line from the fragment stage above rather than rewritten, because
 * two backends drawing two slightly different pictures is the failure this
 * whole lane exists to avoid.
 */
export function wgslSource({ maxTerms = MAX_TERMS } = {}) {
  const cap = Math.max(1, Math.floor(maxTerms));
  return `struct Field {
  uRect : vec4<f32>,
  uViewport : vec2<f32>,
  uScale : f32,
  uTermCount : f32,
  uFlow : f32,
  uOrder : f32,
  uNode : vec4<f32>,
  uLift : vec4<f32>,
  uPlus : vec4<f32>,
  uMinus : vec4<f32>,
  uTerms : array<vec4<f32>, ${cap}>,
};

@group(0) @binding(0) var<uniform> uField : Field;
@group(0) @binding(1) var uThickSampler : sampler;
@group(0) @binding(2) var uThickTexture : texture_2d<f32>;

const PI : f32 = ${PI_LITERAL};

struct Stage {
  @builtin(position) place : vec4<f32>,
  @location(0) uv : vec2<f32>,
};

@vertex
fn vertexMain(@builtin(vertex_index) index : u32) -> Stage {
  var corners = array<vec2<f32>, 4>(
    vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 0.0), vec2<f32>(0.0, 1.0), vec2<f32>(1.0, 1.0)
  );
  let corner = corners[index];
  let px = uField.uRect.xy + corner * uField.uRect.zw;
  var out : Stage;
  out.uv = corner;
  out.place = vec4<f32>(
    px.x / uField.uViewport.x * 2.0 - 1.0,
    1.0 - px.y / uField.uViewport.y * 2.0,
    0.0, 1.0
  );
  return out;
}

fn wAt(p : vec2<f32>) -> f32 {
  var w : f32 = 0.0;
  let count = i32(uField.uTermCount);
  for (var i : i32 = 0; i < ${cap}; i = i + 1) {
    if (i >= count) { break; }
    let term = uField.uTerms[i];
    w = w + term.z * sin(term.x * PI * p.x) * sin(term.y * PI * p.y);
  }
  return w;
}

// The perpendicular gradient — see the note on the other path. Divergence-free
// by construction, so the flow is incompressible with nothing to project.
fn flowAt(p : vec2<f32>) -> vec2<f32> {
  var dwdu : f32 = 0.0;
  var dwdv : f32 = 0.0;
  let count = i32(uField.uTermCount);
  for (var i : i32 = 0; i < ${cap}; i = i + 1) {
    if (i >= count) { break; }
    let term = uField.uTerms[i];
    let a = term.x * PI;
    let b = term.y * PI;
    dwdu = dwdu + term.z * a * cos(a * p.x) * sin(b * p.y);
    dwdv = dwdv + term.z * b * sin(a * p.x) * cos(b * p.y);
  }
  return vec2<f32>(-dwdv, dwdu);
}

@fragment
fn fragmentMain(stage : Stage) -> @location(0) vec4<f32> {
  // Transliterated line for line from the other path, including the two
  // semi-Lagrangian steps. Two backends drawing two slightly different pictures
  // is the failure this whole lane exists to avoid.
  var p = stage.uv;
  if (uField.uFlow > 0.0) {
    p = p - uField.uFlow * flowAt(p);
    p = p - uField.uFlow * flowAt(p);
    p = clamp(p, vec2<f32>(0.0), vec2<f32>(1.0));
  }

  let d = clamp(wAt(p) * uField.uScale, -1.0, 1.0);
  let t = textureSample(uThickTexture, uThickSampler, stage.uv).r;
  let rest = uField.uNode.rgb + t * uField.uLift.rgb;
  let mag = pow(abs(d), ${MAG_EXPONENT});
  var tint = uField.uMinus.rgb;
  if (d >= 0.0) { tint = uField.uPlus.rgb; }
  let orderMix = clamp(uField.uOrder, 0.0, 1.0) * ${ORDER_LIFT_FRACTION};
  if (orderMix > 0.0) { tint = mix(tint, uField.uLift.rgb, orderMix); }
  return vec4<f32>(mix(rest, tint, mag), 1.0);
}
`;
}

// ── reading a shader back ────────────────────────────────────────────────────
//
// A shader that fails to compile tells you loudly. A shader that compiles and
// quietly ignores the uniform you thought you were setting tells you nothing —
// you get a black plate and no message. So the source is parsed, not eyeballed.

/** Comments removed, so a name mentioned in prose does not count as a use. */
export function stripComments(src) {
  return String(src).replace(/\/\*[\s\S]*?\*\//g, ' ').replace(/\/\/[^\n]*/g, ' ');
}

const DECL = /\b(uniform|attribute|varying|in|out)\s+(?:(?:lowp|mediump|highp)\s+)?([A-Za-z_]\w*)\s+([A-Za-z_]\w*)\s*(?:\[\s*([A-Za-z_0-9]+)\s*\])?\s*;/g;

/**
 * Every declaration in a GLSL source, plus every name that follows the house
 * convention and how many times it appears.
 *
 * `in`/`out` are read as the newer spelling of `attribute`/`varying`, which is
 * exactly what they are here — this parser only has to understand the shaders
 * in this file, and pretending otherwise would be a parser project.
 */
export function parseGlsl(src) {
  const text = stripComments(src);
  const glsl3 = /^\s*#version\s+300\s+es/.test(text);
  const declared = new Map();
  for (const m of text.matchAll(DECL)) {
    const [, qualifier, type, name, size] = m;
    declared.set(name, { qualifier, type, name, size: size || null });
  }
  // Every identifier, counted. A declaration contributes exactly one of these,
  // so "declared and never read" is a count of one and nothing else.
  const used = new Map();
  for (const m of text.matchAll(/[A-Za-z_]\w*/g)) {
    used.set(m[0], (used.get(m[0]) || 0) + 1);
  }
  return { glsl3, declared, used, text };
}

/**
 * Both directions of the contract, as a list of plain problems.
 *
 * `expectUniforms` is the renderer's own list of names it will look up. If the
 * two sets ever disagree, the shader and the code driving it have drifted —
 * which is the bug that costs an afternoon, because nothing throws.
 */
export function checkShaderSource(src, { expectUniforms = null, allow = [] } = {}) {
  const { declared, used } = parseGlsl(src);
  const problems = [];
  const permitted = new Set(allow);

  for (const [name, decl] of declared) {
    const count = used.get(name) || 0;
    if (count <= 1) {
      problems.push(`${decl.qualifier} ${name} is declared but never read — setting it from the renderer would do nothing`);
    }
  }
  // The other direction only holds for names that follow the house convention —
  // `uThing`, `aThing`, `vThing`. Everything else in a shader is a keyword, a
  // type or a built-in, and this parser is not in the business of knowing them.
  for (const name of used.keys()) {
    if (!/^[uav][A-Z]/.test(name)) continue;
    if (!declared.has(name) && !permitted.has(name)) {
      problems.push(`${name} is used but never declared`);
    }
  }
  if (expectUniforms) {
    const shaderUniforms = new Set(
      [...declared.values()].filter(d => d.qualifier === 'uniform').map(d => d.name)
    );
    for (const name of expectUniforms) {
      if (!shaderUniforms.has(name)) problems.push(`the renderer sets ${name}, which this shader does not declare`);
    }
    for (const name of shaderUniforms) {
      if (!expectUniforms.includes(name)) problems.push(`${name} is declared but the renderer never sets it`);
    }
  }
  return problems;
}

/** The same two directions for the newer path, whose bindings are numbered. */
export function checkWgslSource(src) {
  const text = stripComments(src);
  const problems = [];
  const seen = new Map();
  // Count every identifier once, rather than building a pattern per name: the
  // names come out of the source itself, and a pattern built from text you just
  // read is a pattern you did not write.
  const counts = new Map();
  for (const m of text.matchAll(/[A-Za-z_]\w*/g)) counts.set(m[0], (counts.get(m[0]) || 0) + 1);
  const countOf = name => counts.get(name) || 0;

  for (const m of text.matchAll(/@group\((\d+)\)\s*@binding\((\d+)\)\s*var(?:<[^>]*>)?\s*([A-Za-z_]\w*)/g)) {
    const [, group, binding, name] = m;
    const key = `${group}:${binding}`;
    if (seen.has(key)) problems.push(`group ${group} binding ${binding} is claimed by both ${seen.get(key)} and ${name}`);
    seen.set(key, name);
    if (countOf(name) <= 1) problems.push(`${name} is bound but never read`);
  }
  for (const m of text.matchAll(/\b(u[A-Z]\w*)\s*:/g)) {
    if (countOf(m[1]) <= 1) problems.push(`${m[1]} is in the uniform block but never read`);
  }
  if (!/@vertex/.test(text)) problems.push('no vertex stage');
  if (!/@fragment/.test(text)) problems.push('no fragment stage');
  return problems;
}
