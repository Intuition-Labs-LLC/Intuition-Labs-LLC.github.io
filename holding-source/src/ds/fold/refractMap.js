import { AIR, GLASSES, abbe, channels } from '../optics.js';

export const LENS = Object.freeze({
  band: { v: 28, unit: 'px', basis: 'crease height' },
  strength: { v: 22, unit: 'px', basis: 'peak displacement' },
  curve: { v: 2, unit: 'power', basis: 'thickness falloff' },
});

// N-SF11 dense flint, strength set so the red-to-blue spread is a whole 2 px (spread x Abbe(flint))
export const SPECTRAL = Object.freeze({
  v: 2 * abbe(GLASSES.flint),
  unit: 'px',
  basis: 'N-SF11 dense flint, strength for a 2 px red-to-blue spread (spread x Abbe(flint))',
});

// green carries the base strength; red and blue scale by (n - 1), the same ratio that sets exitDeviation
export function spectralScales(glass, strength) {
  const n = channels(glass);
  const denom = n.g - AIR;
  return {
    r: (strength * (n.r - AIR)) / denom,
    g: strength,
    b: (strength * (n.b - AIR)) / denom,
  };
}

export function lensSlope(u) {
  const x = 2 * u - 1;
  if (x === 0) return 0;
  return -2 * LENS.curve.v * Math.sign(x) * Math.abs(x) ** (LENS.curve.v - 1);
}

export function displacementPixels(width, height) {
  const pixels = new Uint8ClampedArray(width * height * 4);
  const maxSlope = 2 * LENS.curve.v;
  for (let y = 0; y < height; y += 1) {
    const slope = height > 1 ? lensSlope(y / (height - 1)) : 0;
    const green = 128 + Math.round(127 * slope / maxSlope);
    for (let x = 0; x < width; x += 1) {
      const i = (y * width + x) * 4;
      pixels[i] = 128;
      pixels[i + 1] = green;
      pixels[i + 2] = 128;
      pixels[i + 3] = 255;
    }
  }
  return pixels;
}

export function refractMapDataUrl(doc, width, height) {
  if (!doc?.createElement) return null;
  const canvas = doc.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  if (typeof canvas.getContext !== 'function') return null;
  let context;
  try {
    context = canvas.getContext('2d');
  } catch {
    return null;
  }
  if (!context?.createImageData || !context.putImageData || !canvas.toDataURL) return null;
  try {
    const image = context.createImageData(width, height);
    image.data.set(displacementPixels(width, height));
    context.putImageData(image, 0, 0);
    return canvas.toDataURL('image/png');
  } catch {
    return null;
  }
}
