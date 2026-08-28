import type { Vector } from "./vector.js";

export function mse(predicted: Vector, actual: Vector): number {
  let sum = 0;
  for (let i = 0; i < predicted.length; i++) {
    const diff = predicted[i] - actual[i];
    sum += diff * diff;
  }
  return sum / predicted.length;
}

export function mseGradient(predicted: Vector, actual: Vector): Vector {
  const n = predicted.length;
  return predicted.map((p, i) => (2 * (p - actual[i])) / n);
}

export function binaryCrossEntropy(predicted: Vector, actual: Vector): number {
  const eps = 1e-15;
  let sum = 0;
  for (let i = 0; i < predicted.length; i++) {
    const p = Math.max(eps, Math.min(1 - eps, predicted[i]));
    sum += -(actual[i] * Math.log(p) + (1 - actual[i]) * Math.log(1 - p));
  }
  return sum / predicted.length;
}

export function crossEntropy(predicted: Vector, actual: Vector): number {
  const eps = 1e-15;
  let sum = 0;
  for (let i = 0; i < predicted.length; i++) {
    const p = Math.max(eps, predicted[i]);
    sum += -actual[i] * Math.log(p);
  }
  return sum;
}

/** Simple 1D loss functions for gradient descent visualization. */
export const lossFunctions = {
  quadratic: (x: number) => x * x,
  rosenbrock: (x: number, y: number) =>
    (1 - x) ** 2 + 100 * (y - x * x) ** 2,
  beale: (x: number, y: number) =>
    (1.5 - x + x * y) ** 2 + (2.25 - x + x * y * y) ** 2 + (2.625 - x + x * y ** 3) ** 2,
  himmelblau: (x: number, y: number) =>
    (x * x + y - 11) ** 2 + (x + y * y - 7) ** 2,
} as const;

export type LossFunctionName = keyof typeof lossFunctions;

export function evalLoss(name: LossFunctionName, ...args: number[]): number {
  return lossFunctions[name](...(args as [number, number]));
}
