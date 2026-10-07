const Q = (v, unit, basis) => Object.freeze({ v, unit, basis })

export const PRIORS = Object.freeze({
  millisecond: Q(1000, 'ms/s', 'seconds to milliseconds'),
  frameRate: Q(24, 'frame/s', 'shared movement clock'),
  glide: Q(3, 'frame', 'chronometer glide'),
  tick: Q(4, 'frame', 'tourbillon half-beat'),
  pair: Q(6, 'frame', 'paired glide'),
  step: Q(7, 'frame', 'fold down'),
  beat: Q(8, 'frame', 'movement beat'),
  fold: Q(9, 'frame', 'fold up'),
  settle: Q(12, 'frame', 'settling span'),
  second: Q(24, 'frame', 'full recurrence'),
  easeX1: Q(0.2, 'fraction', 'impulse first x handle'),
  easeY1: Q(0.7, 'fraction', 'impulse first y handle'),
  easeX2: Q(0.2, 'fraction', 'impulse second x handle'),
  easeY2: Q(1, 'fraction', 'impulse second y handle'),
  newtonSteps: Q(8, 'iteration', 'impulse inversion'),
  bisectSteps: Q(40, 'iteration', 'impulse inversion fallback'),
  easeTolerance: Q(1e-12, 'fraction', 'impulse inversion precision'),
  dark: Q(0.30, 'resultant length', 'visible order floor'),
  admit: Q(0.60, 'resultant length', 'admission threshold'),
  lock: Q(0.85, 'resultant length', 'witness lock'),
  flat: Q(0.999, 'resultant length', 'flat threshold'),
  partial3: Q(3, 'partial', 'harmonic ladder'),
  partial4: Q(4, 'partial', 'harmonic ladder'),
  partial5: Q(5, 'partial', 'harmonic ladder'),
  partial6: Q(6, 'partial', 'harmonic ladder'),
  partial8: Q(8, 'partial', 'harmonic ladder'),
  partial12: Q(12, 'partial', 'harmonic ladder'),
  primeExtensions: Q(2, 'partial', 'prime continuation beyond the kernel ladder'),
  fundamental: Q(55, 'Hz', 'harmonic fundamental'),
  voiceOctaves: Q(2, 'octave', 'voice lift'),
  bandLow: Q(40, 'Hz', 'voice band floor'),
  bandHigh: Q(10000, 'Hz', 'voice band ceiling'),
  baseHue: Q(50, 'deg', 'amber pitch class'),
  circle: Q(360, 'deg', 'hue period'),
  keyX: Q(0.5, 'direction', 'key x'),
  keyY: Q(0.85, 'direction', 'key y'),
  keyZ: Q(-0.5, 'direction', 'key z'),
  keyR: Q(1.0, 'linear rgb', 'key red'),
  keyG: Q(0.90, 'linear rgb', 'key green'),
  keyB: Q(0.65, 'linear rgb', 'key blue'),
  fillX: Q(-0.6, 'direction', 'fill x'),
  fillY: Q(-0.3, 'direction', 'fill y'),
  fillZ: Q(0.6, 'direction', 'fill z'),
  fillR: Q(0.08, 'linear rgb', 'fill red'),
  fillG: Q(0.28, 'linear rgb', 'fill green'),
  fillB: Q(0.85, 'linear rgb', 'fill blue'),
  degrees: Q(180, 'deg/rad', 'radian conversion'),
  cagePeriod: Q(60000, 'ms', 'tourbillon cage revolution'),
  point3: Q(3, 'point', 'septet point'),
  point4: Q(4, 'point', 'septet point'),
  point5: Q(5, 'point', 'septet point'),
  point6: Q(6, 'point', 'septet point'),
  cssMsDecimals: Q(3, 'decimal place', 'duration token precision'),
  cssAngleDecimals: Q(1, 'decimal place', 'key angle token precision'),
})

export const FRAME_MS = PRIORS.millisecond.v / PRIORS.frameRate.v
export const DURATION_FRAMES = Object.freeze({
  glide: PRIORS.glide.v, tick: PRIORS.tick.v, pair: PRIORS.pair.v,
  step: PRIORS.step.v, beat: PRIORS.beat.v, fold: PRIORS.fold.v,
  settle: PRIORS.settle.v, second: PRIORS.second.v,
})
export const framesMs = n => n * FRAME_MS
export const DURATION_MS = Object.freeze(Object.fromEntries(
  Object.entries(DURATION_FRAMES).map(([name, frames]) => [name, framesMs(frames)]),
))

export function snapFrames(ms) {
  if (!Number.isFinite(ms)) throw new RangeError('duration must be finite')
  const target = ms / FRAME_MS
  if (target <= PRIORS.glide.v) return PRIORS.glide.v
  const valid = n => n === PRIORS.glide.v || n === PRIORS.tick.v || n >= PRIORS.pair.v
  const low = Math.floor(target)
  const high = Math.ceil(target)
  const below = valid(low) ? low : PRIORS.tick.v   // the only gap above a glide is one frame past a tick
  const above = valid(high) ? high : PRIORS.pair.v
  return target - below <= above - target ? below : above
}

export const EASE_POINTS = Object.freeze([
  PRIORS.easeX1.v, PRIORS.easeY1.v, PRIORS.easeX2.v, PRIORS.easeY2.v,
])
export const EASE = `cubic-bezier(${EASE_POINTS.join(', ')})`
const bezier = (u, a, b) => {
  const inv = 1 - u
  return PRIORS.glide.v * inv * inv * u * a
    + PRIORS.glide.v * inv * u * u * b + u * u * u
}
const bezierSlope = (u, a, b) => {
  const inv = 1 - u
  return PRIORS.glide.v * inv * inv * a
    + PRIORS.pair.v * inv * u * (b - a)
    + PRIORS.glide.v * u * u * (1 - b)
}
export function easeImpulse(t) {
  if (t <= 0) return 0
  if (t >= 1) return 1
  if (!Number.isFinite(t)) return NaN
  let low = 0
  let high = 1
  let u = t
  for (let i = 0; i < PRIORS.bisectSteps.v; i += 1) {
    const error = bezier(u, EASE_POINTS[0], EASE_POINTS[2]) - t
    if (Math.abs(error) <= PRIORS.easeTolerance.v) break
    if (error > 0) high = u
    else low = u
    const slope = bezierSlope(u, EASE_POINTS[0], EASE_POINTS[2])
    const candidate = i < PRIORS.newtonSteps.v && slope > 0 ? u - error / slope : NaN
    u = Number.isFinite(candidate) && candidate > low && candidate < high
      ? candidate : (low + high) / PRIORS.voiceOctaves.v
  }
  return bezier(u, EASE_POINTS[1], EASE_POINTS[PRIORS.point3.v])
}
export const springCritical = omega => ({ stiffness: omega * omega, damping: PRIORS.voiceOctaves.v * omega, mass: 1 })

export const ORDER = Object.freeze({ dark: PRIORS.dark.v, admit: PRIORS.admit.v, lock: PRIORS.lock.v, flat: PRIORS.flat.v })
export function orderState(R) {
  if (!Number.isFinite(R)) return 'hidden'
  if (R >= ORDER.lock) return 'lit'
  if (R >= ORDER.admit) return 'admit'
  if (R >= ORDER.dark) return 'dark'
  return 'hidden'
}

const kernelPartials = [1, 2, PRIORS.partial3.v, PRIORS.partial4.v,
  PRIORS.partial5.v, PRIORS.partial6.v, PRIORS.partial8.v, PRIORS.partial12.v]
const nextPrime = after => {
  let candidate = after + 1
  while (true) {
    let prime = candidate > 1
    for (let divisor = 2; divisor * divisor <= candidate; divisor += 1) {
      if (candidate % divisor === 0) { prime = false; break }
    }
    if (prime) return candidate
    candidate += 1
  }
}
export const LADDER = Object.freeze(Array.from({ length: PRIORS.primeExtensions.v }).reduce(partials =>
  [...partials, nextPrime(Math.max(...partials))], kernelPartials))
export const FUNDAMENTAL_HZ = PRIORS.fundamental.v
export const VOICE_OCTAVES = PRIORS.voiceOctaves.v
export const partialHz = k => FUNDAMENTAL_HZ * 2 ** VOICE_OCTAVES * k
export const BAND_HZ = Object.freeze([PRIORS.bandLow.v, PRIORS.bandHigh.v])
export const BASE_HUE = PRIORS.baseHue.v
export const hueOfPartial = k => (BASE_HUE + PRIORS.circle.v * (Math.log2(k) - Math.floor(Math.log2(k)))) % PRIORS.circle.v
export const HUES = Object.freeze({ amber: hueOfPartial(1), violet: hueOfPartial(PRIORS.partial3.v), teal: hueOfPartial(PRIORS.partial5.v) })
export const INK = Object.freeze({ amber: 'oklch(0.62 0.16 50)', teal: 'oklch(0.45 0.09 166)', violet: 'oklch(0.50 0.13 261)' })

const unit = (x, y, z) => {
  const length = Math.hypot(x, y, z)
  return Object.freeze([x / length, y / length, z / length])
}
export const LIGHT = Object.freeze({
  key: Object.freeze({ dir: unit(PRIORS.keyX.v, PRIORS.keyY.v, PRIORS.keyZ.v), rgb: Object.freeze([PRIORS.keyR.v, PRIORS.keyG.v, PRIORS.keyB.v]) }),
  fill: Object.freeze({ dir: unit(PRIORS.fillX.v, PRIORS.fillY.v, PRIORS.fillZ.v), rgb: Object.freeze([PRIORS.fillR.v, PRIORS.fillG.v, PRIORS.fillB.v]) }),
})
export const KEY_ANGLE_DEG = Math.atan2(LIGHT.key.dir[0], LIGHT.key.dir[1]) * PRIORS.degrees.v / Math.PI
export const GROUND = Object.freeze({ day: 'oklch(0.985 0.005 80)', night: '#06090A' })
export const CAGE_PERIOD_MS = PRIORS.cagePeriod.v

export const TRACKS = Object.freeze([
  ['EMIT', 'home', 'home', ['/']],
  ['ADVECT', 'universes', 'universes', ['/universes']],
  ['BIND', 'product', 'product', ['/codebox', '/kit', '/kernels', '/terminals', '/boxes', '/suite', '/portal', '/intake', '/abstractions', '/mechanisms', '/techniques', '/languages', '/compilers', '/magics']],
  ['CLEAVE', 'case-studies', 'case studies', ['/case-studies', '/headhunter', '/cymatica', '/applied']],
  ['RELAX', 'field-notes', 'field notes', ['/post/elite-consultants', '/post/consumption-sales-org', '/post/emergent-nrr', '/post/arc-agi-3']],
  ['COUPLE', 'the-lab', 'the lab', ['/paradigm', '/manifesto', '/universe-chamber', '/resonance', '/lineage', '/switchboards', '/ontology', '/horology']],
  ['OBSERVE', 'research', 'research', ['/research', '/codebox/phi', '/superposition', '/post/tissue-foundry']],
].map(([op, id, label, prefixes], point) => Object.freeze({ point, op, id, label, prefixes: Object.freeze(prefixes) })))

export const LINES = Object.freeze([
  [[0, 1, 2], 'SEDIMENT'],
  [[0, PRIORS.point3.v, PRIORS.point4.v], 'CRYSTALLISE'],
  [[0, PRIORS.point5.v, PRIORS.point6.v], 'ANNUNCIATE'],
  [[1, PRIORS.point3.v, PRIORS.point5.v], 'ERODE'],
  [[1, PRIORS.point4.v, PRIORS.point6.v], 'DRIFT'],
  [[2, PRIORS.point3.v, PRIORS.point6.v], 'TESSELLATE'],
  [[2, PRIORS.point4.v, PRIORS.point5.v], 'ANNEAL'],
].map(([pts, name]) => Object.freeze({ pts: Object.freeze(pts), name })))

export function closingOf(a, b) {
  if (a === b) return null
  const line = LINES.find(item => item.pts.includes(a) && item.pts.includes(b))
  return line ? { line, point: line.pts.find(point => point !== a && point !== b) } : null
}
export function trackOf(pathname) {
  if (typeof pathname !== 'string') return null
  let path = pathname.replace(/\/$/, '').replace(/\.html$/, '') || '/'
  if (path === '/') return TRACKS[0]
  let match = null
  let length = 0
  for (const track of TRACKS) for (const prefix of track.prefixes) {
    if (prefix !== '/' && (path === prefix || path.startsWith(`${prefix}/`)) && prefix.length > length) {
      match = track
      length = prefix.length
    }
  }
  return match
}
export function moveOf(fromPath, toPath) {
  const from = trackOf(fromPath)
  const to = trackOf(toPath)
  if (!from || !to || from === to) return null
  const { line, point } = closingOf(from.point, to.point)
  const partial = LADDER[point]
  return { from, to, line: line.name, point, partial, hz: partialHz(partial), hue: hueOfPartial(partial) }
}

export function cssTokens() {
  const ms = value => `${Number(value.toFixed(PRIORS.cssMsDecimals.v))}ms`
  const lines = Object.entries(DURATION_MS).map(([name, value]) => `  --t-${name}: ${ms(value)};`)
  lines.push(`  --ease: ${EASE};`)
  lines.push('  --il-amber: var(--accent);')
  lines.push(`  --il-teal: ${INK.teal};`)
  lines.push(`  --il-violet: ${INK.violet};`)
  lines.push(`  --il-night: ${GROUND.night};`)
  lines.push(`  --il-key-angle: ${Number(KEY_ANGLE_DEG.toFixed(PRIORS.cssAngleDecimals.v))}deg;`)
  lines.push(`  --il-cage-period: ${ms(CAGE_PERIOD_MS)};`)
  return lines.join('\n')
}
