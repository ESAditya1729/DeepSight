import { describe, expect, it } from "vitest";
import { mse, mseGradient, binaryCrossEntropy, crossEntropy } from "./losses.js";

describe("mse", () => {
  it("returns 0 for identical vectors", () => {
    expect(mse([1, 2, 3], [1, 2, 3])).toBe(0);
  });

  it("computes mean squared error", () => {
    expect(mse([1, 2], [3, 4])).toBe(4);
  });

  it("is symmetric", () => {
    expect(mse([1, 2], [3, 4])).toBe(mse([3, 4], [1, 2]));
  });
});

describe("mseGradient", () => {
  it("returns zero gradient for identical vectors", () => {
    expect(mseGradient([1, 2], [1, 2])).toEqual([0, 0]);
  });

  it("gradient points toward lower predicted values when predicted > actual", () => {
    const grad = mseGradient([4, 4], [2, 2]);
    expect(grad[0]).toBeGreaterThan(0);
    expect(grad[1]).toBeGreaterThan(0);
  });
});

describe("binaryCrossEntropy", () => {
  it("returns 0 for perfect predictions", () => {
    expect(binaryCrossEntropy([0], [0])).toBeCloseTo(0, 4);
    expect(binaryCrossEntropy([1], [1])).toBeCloseTo(0, 4);
  });

  it("returns higher loss for worse predictions", () => {
    const good = binaryCrossEntropy([0.9], [1]);
    const bad = binaryCrossEntropy([0.1], [1]);
    expect(good).toBeLessThan(bad);
  });
});

describe("crossEntropy", () => {
  it("returns 0 for perfect one-hot prediction", () => {
    expect(crossEntropy([1], [1])).toBeCloseTo(0, 4);
  });

  it("returns large loss for very low probability", () => {
    const loss = crossEntropy([0.0001], [1]);
    expect(loss).toBeGreaterThan(5);
  });
});
