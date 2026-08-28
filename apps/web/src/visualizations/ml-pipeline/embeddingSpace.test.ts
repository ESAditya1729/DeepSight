import { describe, it, expect } from "vitest";
import {
  normalizeWord,
  categoryFor,
  getBasePosition,
  getWordPosition,
  mapSimilarity,
  categoryColor,
} from "./embeddingSpace";

describe("normalizeWord", () => {
  it("lowercases and strips punctuation", () => {
    expect(normalizeWord("The!")).toBe("the");
    expect(normalizeWord("Cat.")).toBe("cat");
    expect(normalizeWord("SHE'S")).toBe("shes");
  });

  it("keeps apostrophes inside words", () => {
    expect(normalizeWord("It's")).toBe("its");
  });
});

describe("categoryFor", () => {
  it("finds the category for a known word", () => {
    expect(categoryFor("cat")?.name).toBe("Animal");
    expect(categoryFor("mat")?.name).toBe("Surface");
    expect(categoryFor("king")?.name).toBe("Person");
  });

  it("returns undefined for unknown words", () => {
    expect(categoryFor("xyzzy")).toBeUndefined();
  });
});

describe("getBasePosition", () => {
  it("is deterministic for the same seed", () => {
    const a = getBasePosition("cat", 42);
    const b = getBasePosition("cat", 42);
    expect(a).toEqual(b);
  });

  it("varies with the seed", () => {
    const a = getBasePosition("cat", 42);
    const b = getBasePosition("cat", 7);
    expect(a).not.toEqual(b);
  });

  it("places words in their category cluster", () => {
    const mat = getBasePosition("mat", 42);
    const bed = getBasePosition("bed", 42);
    // Both should be near the Surface cluster at (2, 1.5)
    expect(Math.abs(mat.x - 2)).toBeLessThan(0.8);
    expect(Math.abs(mat.y - 1.5)).toBeLessThan(0.8);
    expect(mapSimilarity([mat.x, mat.y], [bed.x, bed.y])).toBeGreaterThan(0.6);
  });
});

describe("getWordPosition", () => {
  it("occurrence 0 equals the base position", () => {
    expect(getWordPosition("cat", 42, 0)).toEqual(getBasePosition("cat", 42));
  });

  it("later occurrences get jitter near the base", () => {
    const base = getBasePosition("cat", 42);
    const occ = getWordPosition("cat", 42, 1);
    expect(mapSimilarity([base.x, base.y], [occ.x, occ.y])).toBeGreaterThan(0.95);
  });
});

describe("mapSimilarity", () => {
  it("returns 1 for identical points", () => {
    expect(mapSimilarity([0, 0], [0, 0])).toBe(1);
  });

  it("monotonically decreases with distance", () => {
    const near = mapSimilarity([0, 0], [0.5, 0]);
    const far = mapSimilarity([0, 0], [10, 0]);
    expect(near).toBeGreaterThan(far);
  });

  it("is symmetric", () => {
    expect(mapSimilarity([1, 2], [3, 4])).toBeCloseTo(mapSimilarity([3, 4], [1, 2]), 10);
  });

  it("is bounded in [0, 1]", () => {
    for (const [a, b] of [[[0, 0], [100, 100]], [[0, 0], [0, 0]], [[-5, -3], [2, 1]]]) {
      const v = mapSimilarity(a as [number, number], b as [number, number]);
      expect(v).toBeGreaterThanOrEqual(0);
      expect(v).toBeLessThanOrEqual(1);
    }
  });
});

describe("categoryColor", () => {
  it("returns the category color for known words", () => {
    expect(categoryColor("cat")).toBe("#059669");
    expect(categoryColor("mat")).toBe("#7c3aed");
  });

  it("returns a fallback for unknown words", () => {
    expect(categoryColor("xyzzy")).toBe("#94a3b8");
    expect(categoryColor("xyzzy", "#000000")).toBe("#000000");
  });
});
