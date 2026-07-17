# Intuition-Labs-LLC.github.io

The Intuition Labs organization site — one self-contained page, zero build step,
zero external requests. Served at
[intuition-labs-llc.github.io](https://intuition-labs-llc.github.io/).

- `index.html` — the page (inline CSS + JS).
- `sim.js` — the Kuramoto phase-field simulation core, shared by the page and the test.
- `selftest.mjs` — headless proof of the physics: `node selftest.mjs`.
- `lens.html` — the lens-gate: drop a Neuronpedia pre-fitted Jacobian-lens `.pt` (or any
  lens-shaped JSON), get cone fits, Δ_μ, a verdict, and a downloadable receipt — computed
  entirely in the tab, zero requests.
- `lens-core.js` — the whole lens pipeline (zip walk, pickle subset, fp16, pursuit/Δ_μ, gate),
  shared by the page and the test.
- `selftest-lens.mjs` — headless proof of the lens pipeline, incl. bit-exact parity with the
  jspace monorepo's pinned `examples/lens-gate` numbers when the sibling checkout is present.
- `404.html` · `.nojekyll` · `LICENSE`.

## How the hero shader works, in five lines

1. `sim.js` steps 24×24 phase oscillators on a torus (nearest-neighbour coupling, K ramping 0→2.2), seed 11.
2. Each frame the 24×24 phases are quantized to a `LUMINANCE`/`UNSIGNED_BYTE` texture and uploaded.
3. A fragment shader samples it with bilinear filtering, so the coarse grid reads as a smooth phase field.
4. Brightness is `0.04 + 0.10·(0.5+0.5·cos θ)` — near-black — with thin `fwidth`-based iso-phase contour lines in white at ~12%.
5. JS computes the true order parameter `R = |⟨e^{iθ}⟩|` each frame for the HUD; at `R ≥ 0.9` the field is synced.

Grayscale only. `prefers-reduced-motion` pauses at a mid-sync frame with a step button; no WebGL falls back to a
CSS gradient with the live readout; all content is real, scrollable HTML.

`AGPL-3.0-or-later`.
