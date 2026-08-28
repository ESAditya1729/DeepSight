// Pure computation engine. No React/D3/Three imports allowed here —
// that constraint is what keeps this package testable and reusable
// outside the browser (e.g. a future VS Code extension host).

export * from "./vector.js";
export * from "./matrix.js";
export * from "./calculus.js";
export * from "./losses.js";
export * from "./nn/activations.js";
export * from "./nn/layers.js";
export * from "./nn/network.js";
export * from "./nn/trainer.js";
