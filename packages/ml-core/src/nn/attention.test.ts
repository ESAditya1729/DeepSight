import { describe, expect, it } from "vitest";
import { softmax, scaledDotProductAttention } from "./attention.js";

describe("softmax", () => {
  it("sums to 1", () => {
    const weights = softmax([1, 2, 3]);
    const sum = weights.reduce((a, b) => a + b, 0);
    expect(sum).toBeCloseTo(1);
  });

  it("is uniform for equal logits", () => {
    const weights = softmax([5, 5, 5, 5]);
    weights.forEach((w) => expect(w).toBeCloseTo(0.25));
  });

  it("assigns the most weight to the largest logit", () => {
    const weights = softmax([1, 10, 2]);
    expect(weights[1]).toBeGreaterThan(weights[0]);
    expect(weights[1]).toBeGreaterThan(weights[2]);
  });

  it("is stable for large logits", () => {
    const weights = softmax([1000, 1001, 1002]);
    const sum = weights.reduce((a, b) => a + b, 0);
    expect(sum).toBeCloseTo(1);
    expect(Number.isFinite(weights[0])).toBe(true);
  });

  it("flattens the distribution as temperature increases", () => {
    const sharp = softmax([1, 5], 0.5);
    const flat = softmax([1, 5], 5);
    expect(sharp[1] - sharp[0]).toBeGreaterThan(flat[1] - flat[0]);
  });
});

describe("scaledDotProductAttention", () => {
  it("produces row-stochastic attention weights", () => {
    const Q = [[1, 0], [0, 1]];
    const K = [[1, 0], [0, 1]];
    const V = [[10, 0], [0, 10]];

    const { weights } = scaledDotProductAttention(Q, K, V);
    weights.forEach((row) => {
      const sum = row.reduce((a, b) => a + b, 0);
      expect(sum).toBeCloseTo(1);
    });
  });

  it("matches a hand-computed example", () => {
    const Q = [[1, 0], [0, 1]];
    const K = [[1, 0], [0, 1]];
    const V = [[10, 0], [0, 10]];

    const { weights, output } = scaledDotProductAttention(Q, K, V);

    expect(weights[0][0]).toBeCloseTo(0.6698, 3);
    expect(weights[0][1]).toBeCloseTo(0.3302, 3);
    expect(output[0][0]).toBeCloseTo(6.698, 2);
    expect(output[0][1]).toBeCloseTo(3.302, 2);
  });

  it("attends fully to the only other token when identical", () => {
    const Q = [[1, 1], [1, 1]];
    const K = [[1, 1], [1, 1]];
    const V = [[3, 4], [5, 6]];

    const { weights } = scaledDotProductAttention(Q, K, V);
    expect(weights[0][0]).toBeCloseTo(0.5);
    expect(weights[0][1]).toBeCloseTo(0.5);
  });

  it("sharpens weights as temperature decreases", () => {
    const Q = [[1, 0]];
    const K = [[1, 0], [0, 1]];
    const V = [[1, 0], [0, 1]];

    const sharp = scaledDotProductAttention(Q, K, V, 0.1);
    const flat = scaledDotProductAttention(Q, K, V, 5);

    expect(sharp.weights[0][0]).toBeGreaterThan(flat.weights[0][0]);
  });

  it("returns output with the shape of V's columns and Q's rows", () => {
    const Q = [[1, 2, 3], [4, 5, 6]];
    const K = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
    const V = [[1, 1], [2, 2], [3, 3]];

    const { output, weights } = scaledDotProductAttention(Q, K, V);
    expect(output.length).toBe(2);
    expect(output[0].length).toBe(2);
    expect(weights.length).toBe(2);
    expect(weights[0].length).toBe(3);
  });
});
