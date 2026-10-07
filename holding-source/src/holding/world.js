export const PIECE = Object.freeze({
  title: 'A beginning',
  seed: 'intuition-labs:uploading:one',
  dim: 2,
  hand: { 'emit.rate|n': 0.24, 'relax.sync|f': -0.16, 'bind.reach|f': -0.12 },
  couples: [['emit.rate|n', 'μ', 1], ['advect.chirality|f', 'E', 1], ['relax.sync|f', 'R', -1]],
  budget: (Λ, ctx) => Λ.frac('world.budget', ctx),
});
