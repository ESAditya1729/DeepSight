import { cols, matScale, multiply, transpose, type Matrix } from "../matrix.js";

export function softmax(logits: readonly number[], temperature = 1): number[] {
  const scaled = logits.map((x) => x / temperature);
  const max = Math.max(...scaled);
  const exps = scaled.map((x) => Math.exp(x - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((x) => x / sum);
}

export interface AttentionResult {
  scores: Matrix;
  weights: Matrix;
  output: Matrix;
}

export function scaledDotProductAttention(
  Q: Matrix,
  K: Matrix,
  V: Matrix,
  temperature = 1,
): AttentionResult {
  const dK = cols(Q);
  const scores = matScale(multiply(Q, transpose(K)), 1 / Math.sqrt(dK));
  const weights = scores.map((row) => softmax(row, temperature));
  const output = multiply(weights, V);
  return { scores, weights, output };
}
