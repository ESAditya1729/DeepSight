// Shared, framework-light visualization primitives (hooks, D3/Three
// helpers, control components) reused across every visualization.

export { useAnimationFrame } from "./useAnimationFrame.js";
export { useKeyDown } from "./useKeyDown.js";
export { PlaybackControls } from "./PlaybackControls.js";
export { Slider } from "./Slider.js";
export { EquationBlock } from "./EquationBlock.js";

// Consuming apps should import the stylesheet once:
//   import "@ml-visual-lab/viz-kit/viz-kit.css";
