import { describe, expect, it } from "vitest";
import { sigmoid, relu, tanh_ } from "./activations.js";

describe("sigmoid", () => {
  it("returns ~0.5 at x=0", () => {
    expect(sigmoid.fn(0)).toBeCloseTo(0.5);
  });

  it("returns ~1 for large positive x", () => {
    expect(sigmoid.fn(10)).toBeCloseTo(1, 3);
  });

  it("returns ~0 for large negative x", () => {
    expect(sigmoid.fn(-10)).toBeCloseTo(0, 3);
  });

  it("derivative is correct at x=0", () => {
    expect(sigmoid.prime(0)).toBeCloseTo(0.25);
  });
});

describe("relu", () => {
  it("returns x for positive x", () => {
    expect(relu.fn(5)).toBe(5);
  });

  it("returns 0 for negative x", () => {
    expect(relu.fn(-3)).toBe(0);
  });

  it("derivative is 1 for positive x", () => {
    expect(relu.prime(5)).toBe(1);
  });

  it("derivative is 0 for negative x", () => {
    expect(relu.prime(-5)).toBe(0);
  });
});

describe("tanh_", () => {
  it("returns 0 at x=0", () => {
    expect(tanh_.fn(0)).toBeCloseTo(0);
  });

  it("returns ~1 for large positive x", () => {
    expect(tanh_.fn(10)).toBeCloseTo(1, 3);
  });

  it("derivative is 1 at x=0", () => {
    expect(tanh_.prime(0)).toBeCloseTo(1);
  });
});
