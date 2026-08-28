import { describe, expect, it } from "vitest";
import {
  zeros, random, transpose, multiply, multiplyVector,
  matAdd, matScale, map, rows, cols, get,
} from "./matrix.js";

describe("zeros", () => {
  it("creates a zero matrix of the given size", () => {
    const m = zeros(2, 3);
    expect(rows(m)).toBe(2);
    expect(cols(m)).toBe(3);
    expect(m).toEqual([[0, 0, 0], [0, 0, 0]]);
  });
});

describe("random", () => {
  it("creates a matrix of the given size", () => {
    const m = random(3, 2);
    expect(rows(m)).toBe(3);
    expect(cols(m)).toBe(2);
  });
});

describe("transpose", () => {
  it("transposes a matrix", () => {
    const m = [[1, 2, 3], [4, 5, 6]];
    expect(transpose(m)).toEqual([[1, 4], [2, 5], [3, 6]]);
  });

  it("handles square matrices", () => {
    const m = [[1, 2], [3, 4]];
    expect(transpose(m)).toEqual([[1, 3], [2, 4]]);
  });
});

describe("multiply", () => {
  it("multiplies two compatible matrices", () => {
    const a = [[1, 2], [3, 4]];
    const b = [[5, 6], [7, 8]];
    expect(multiply(a, b)).toEqual([[19, 22], [43, 50]]);
  });

  it("multiplies matrix by column vector", () => {
    const a = [[1, 2], [3, 4]];
    const b = [[5], [6]];
    expect(multiply(a, b)).toEqual([[17], [39]]);
  });

  it("throws for incompatible dimensions", () => {
    expect(() => multiply([[1, 2]], [[1, 2]])).toThrow("Incompatible");
  });
});

describe("multiplyVector", () => {
  it("multiplies matrix by vector", () => {
    const m = [[1, 2], [3, 4]];
    const v = [5, 6];
    expect(multiplyVector(m, v)).toEqual([17, 39]);
  });

  it("throws for incompatible dimensions", () => {
    expect(() => multiplyVector([[1, 2]], [1, 2, 3])).toThrow("Incompatible");
  });
});

describe("matAdd", () => {
  it("adds two matrices element-wise", () => {
    const a = [[1, 2], [3, 4]];
    const b = [[5, 6], [7, 8]];
    expect(matAdd(a, b)).toEqual([[6, 8], [10, 12]]);
  });
});

describe("matScale", () => {
  it("scales a matrix by a scalar", () => {
    const m = [[1, 2], [3, 4]];
    expect(matScale(m, 2)).toEqual([[2, 4], [6, 8]]);
  });
});

describe("map", () => {
  it("applies a function to each element", () => {
    const m = [[1, 2], [3, 4]];
    expect(map(m, (v) => v * v)).toEqual([[1, 4], [9, 16]]);
  });
});

describe("get", () => {
  it("returns element at position", () => {
    const m = [[1, 2], [3, 4]];
    expect(get(m, 1, 0)).toBe(3);
  });
});
