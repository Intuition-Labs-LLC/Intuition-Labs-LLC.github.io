// The critical spring: the one exact step every eased follower on the site takes.
export function stepCritical(state, target, omega, dt) {
  const displacement = state.x - target;
  const decay = Math.exp(-omega * dt);
  const slope = state.v + omega * displacement;
  return {
    x: target + (displacement + slope * dt) * decay,
    v: (state.v - omega * slope * dt) * decay,
  };
}
