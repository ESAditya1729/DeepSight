import { describe, expect, it } from "vitest";
import {
  add, subtract, scale, dot, magnitude, normalize,
  cosineSimilarity, euclideanDistance, randomUnit,
} from "./vector.js";

describe("add", () => {
  it("adds component-wise", () => {
    expect(add([1, 2], [3, 4])).toEqual([4, 6]);
  });
});

describe("subtract", () => {
  it("subtracts component-wise", () => {
    expect(subtract([5, 1], [2, 4])).toEqual([3, -3]);
  });
});

describe("scale", () => {
  it("multiplies every component by the scalar", () => {
    expect(scale([1, -2, 3], 2)).toEqual([2, -4, 6]);
  });
});

describe("dot", () => {
  it("computes the sum of pairwise products", () => {
    expect(dot([1, 2, 3], [4, 5, 6])).toBe(32);
  });

  it("is zero for orthogonal vectors", () => {
    expect(dot([1, 0], [0, 1])).toBe(0);
  });

  it("is negative for opposing vectors", () => {
    expect(dot([1, 0], [-1, 0])).toBe(-1);
  });
});

describe("magnitude", () => {
  it("computes Euclidean length", () => {
    expect(magnitude([3, 4])).toBe(5);
  });

  it("is zero for the zero vector", () => {
    expect(magnitude([0, 0])).toBe(0);
  });
});

describe("normalize", () => {
  it("produces a unit vector in the same direction", () => {
    const [x, y] = normalize([3, 4]);
    expect(x).toBeCloseTo(0.6);
    expect(y).toBeCloseTo(0.8);
  });

  it("returns the zero vector unchanged instead of dividing by zero", () => {
    expect(normalize([0, 0])).toEqual([0, 0]);
  });
});

describe("cosineSimilarity", () => {
  it("returns 1 for identical vectors", () => {
    expect(cosineSimilarity([1, 2], [1, 2])).toBeCloseTo(1);
  });

  it("returns 0 for orthogonal vectors", () => {
    expect(cosineSimilarity([1, 0], [0, 1])).toBeCloseTo(0);
  });

  it("returns -1 for opposite vectors", () => {
    expect(cosineSimilarity([1, 0], [-1, 0])).toBeCloseTo(-1);
  });

  it("returns 0 for zero vector", () => {
    expect(cosineSimilarity([0, 0], [1, 2])).toBe(0);
  });
});

describe("euclideanDistance", () => {
  it("computes distance between two points", () => {
    expect(euclideanDistance([0, 0], [3, 4])).toBe(5);
  });

  it("returns 0 for same point", () => {
    expect(euclideanDistance([1, 2], [1, 2])).toBe(0);
  });
});

describe("randomUnit", () => {
  it("returns a unit vector", () => {
    const v = randomUnit(3);
    expect(v.length).toBe(3);
    expect(magnitude(v)).toBeCloseTo(1);
  });
});
