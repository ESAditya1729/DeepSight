import { describe, expect, it } from "vitest";
import { numericalGradient, gradientDescentStep, partialDerivative } from "./calculus.js";

describe("numericalGradient", () => {
  it("computes gradient of x^2 at x=3 as ~6", () => {
    const grad = numericalGradient((x) => x * x, 3);
    expect(grad[0]).toBeCloseTo(6, 4);
  });

  it("computes gradient of x^2 + y^2 at (1, 2)", () => {
    const grad = numericalGradient((x: number, y: number) => x * x + y * y, 1, 2);
    expect(grad[0]).toBeCloseTo(2, 4);
    expect(grad[1]).toBeCloseTo(4, 4);
  });

  it("gradient of a constant is zero", () => {
    const grad = numericalGradient(() => 42, 5, 3);
    expect(grad[0]).toBeCloseTo(0, 6);
    expect(grad[1]).toBeCloseTo(0, 6);
  });
});

describe("gradientDescentStep", () => {
  it("moves toward the minimum of x^2", () => {
    const fn = (x: number) => x * x;
    const next = gradientDescentStep(fn, [5], 0.1);
    expect(next[0]).toBeLessThan(5);
    expect(next[0]).toBeGreaterThan(0);
  });

  it("moves toward the minimum of x^2 + y^2", () => {
    const fn = (x: number, y: number) => x * x + y * y;
    const next = gradientDescentStep(fn, [3, 4], 0.1);
    expect(next[0]).toBeLessThan(3);
    expect(next[1]).toBeLessThan(4);
  });
});

describe("partialDerivative", () => {
  it("computes partial derivative of x^2*y w.r.t. x at (2, 3) as ~12", () => {
    const fn = (x: number, y: number) => x * x * y;
    const pd = partialDerivative(fn, [2, 3], 0);
    expect(pd).toBeCloseTo(12, 3);
  });

  it("computes partial derivative of x^2*y w.r.t. y at (2, 3) as ~4", () => {
    const fn = (x: number, y: number) => x * x * y;
    const pd = partialDerivative(fn, [2, 3], 1);
    expect(pd).toBeCloseTo(4, 3);
  });
});
