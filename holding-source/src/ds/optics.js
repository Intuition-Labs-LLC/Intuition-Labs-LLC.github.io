// Optics: dispersion, refraction, reflection and absorption of light through glass, each law checked.
const Q = (v, unit, basis) => Object.freeze({ v, unit, basis });
export const PRIORS = Object.freeze({
  cauchyA: Q(1.5046, 'index', 'N-BK7 crown glass, Cauchy constant term'),
  cauchyB: Q(0.0042, 'um^2', 'N-BK7 crown glass, Cauchy dispersion term'),
  flintA: Q(1.73897, 'index', 'N-SF11 dense flint glass, Cauchy constant term'),
  flintB: Q(0.015944, 'um^2', 'N-SF11 dense flint glass, Cauchy dispersion term'),
  lineC: Q(0.6563, 'um', 'hydrogen C line, the red channel'),
  lineD: Q(0.5876, 'um', 'helium d line, the green channel'),
  lineF: Q(0.4861, 'um', 'hydrogen F line, the blue channel'),
  air: Q(1, 'index', 'the medium outside the glass'),
});

export const GLASS = Object.freeze({ a: PRIORS.cauchyA.v, b: PRIORS.cauchyB.v });
// crown barely fringes; dense flint splits the colours about two and a half times wider
export const GLASSES = Object.freeze({ crown: GLASS, flint: Object.freeze({ a: PRIORS.flintA.v, b: PRIORS.flintB.v }) });
export const LINES = Object.freeze({ r: PRIORS.lineC.v, g: PRIORS.lineD.v, b: PRIORS.lineF.v });
export const AIR = PRIORS.air.v;

// n(lambda) = A + B / lambda^2, lambda in micrometres
export const cauchy = (lambda, glass = GLASS) => glass.a + glass.b / (lambda * lambda);
export const channels = (glass = GLASS) => ({ r: cauchy(LINES.r, glass), g: cauchy(LINES.g, glass), b: cauchy(LINES.b, glass) });
// the Abbe number: mean refractivity over the red-to-blue spread; high means little colour fringing
export function abbe(glass = GLASS) {
  const n = channels(glass);
  return (n.g - 1) / (n.b - n.r);
}

// Snell in vector form (unit d travelling into the surface, unit normal n facing it); null past total internal reflection
export function refract(d, n, n1, n2) {
  const eta = n1 / n2;
  const cosI = -(d[0] * n[0] + d[1] * n[1] + d[2] * n[2]);
  const k = 1 - eta * eta * (1 - cosI * cosI);
  if (k < 0) return null;
  const s = eta * cosI - Math.sqrt(k);
  return [eta * d[0] + s * n[0], eta * d[1] + s * n[1], eta * d[2] + s * n[2]];
}
export const critical = (n1, n2) => (n1 > n2 ? Math.asin(n2 / n1) : null);
export const brewster = (n1, n2) => Math.atan(n2 / n1);

// exact reflectance of unpolarized light at incidence cosine cosI, both polarizations and their mean
export function fresnel(cosI, n1, n2) {
  const sinT = (n1 / n2) * Math.sqrt(Math.max(0, 1 - cosI * cosI));
  if (sinT >= 1) return { s: 1, p: 1, r: 1 };
  const cosT = Math.sqrt(1 - sinT * sinT);
  const s = ((n1 * cosI - n2 * cosT) / (n1 * cosI + n2 * cosT)) ** 2;
  const p = ((n1 * cosT - n2 * cosI) / (n1 * cosT + n2 * cosI)) ** 2;
  return { s, p, r: (s + p) / 2 };
}
// Schlick's polynomial; leaving the denser medium it runs on the transmitted angle
export function schlick(cosI, n1, n2) {
  const r0 = ((n1 - n2) / (n1 + n2)) ** 2;
  let c = cosI;
  if (n1 > n2) {
    const sinT = (n1 / n2) * Math.sqrt(Math.max(0, 1 - cosI * cosI));
    if (sinT >= 1) return 1;
    c = Math.sqrt(1 - sinT * sinT);
  }
  return r0 + (1 - r0) * (1 - c) ** 5;
}

// transmitted share after a path d through a medium with attenuation alpha per unit length
export const beerLambert = (alpha, d) => Math.exp(-alpha * d);

// a ray through a parallel slab leaves parallel to itself, shifted sideways by this much
export function slabShift(theta, thickness, n, outside = AIR) {
  const thetaT = Math.asin((outside / n) * Math.sin(theta));
  return (thickness * Math.sin(theta - thetaT)) / Math.cos(thetaT);
}

// light leaving glass through a face tilted alpha from the view axis turns by this much; null past total internal reflection
export function exitDeviation(alpha, n, outside = AIR) {
  const s = (n / outside) * Math.sin(alpha);
  return Math.abs(s) >= 1 ? null : Math.asin(s) - alpha;
}
// the per-channel turn through the same face: the colour split a glass edge shows
export function spectralDeviation(alpha, glass = GLASS) {
  const n = channels(glass);
  return { r: exitDeviation(alpha, n.r), g: exitDeviation(alpha, n.g), b: exitDeviation(alpha, n.b) };
}
