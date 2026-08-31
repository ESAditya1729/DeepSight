import { describe, it, expect } from "vitest";
import {
  POLYSEMY_EXAMPLES,
  analyzePolysemy,
  euclideanDistance,
} from "./contextualCore";

describe("analyzePolysemy", () => {
  it("produces one analysis per example with two sentences", () => {
    for (const ex of POLYSEMY_EXAMPLES) {
      const a = analyzePolysemy(ex, 1);
      expect(a.a.tokens.length).toBe(ex.sentences[0].tokens.length);
      expect(a.b.tokens.length).toBe(ex.sentences[1].tokens.length);
      expect(a.a.senseKey).not.toBe(a.b.senseKey);
    }
  });

  it("finds the target word in each sentence", () => {
    for (const ex of POLYSEMY_EXAMPLES) {
      const a = analyzePolysemy(ex, 1);
      expect(a.a.targetIndex).toBeGreaterThanOrEqual(0);
      expect(a.b.targetIndex).toBeGreaterThanOrEqual(0);
      expect(a.a.tokens[a.a.targetIndex].word.toLowerCase()).toBe(ex.target);
      expect(a.b.tokens[a.b.targetIndex].word.toLowerCase()).toBe(ex.target);
    }
  });

  it("keeps the target's base embedding identical in both sentences", () => {
    for (const ex of POLYSEMY_EXAMPLES) {
      const a = analyzePolysemy(ex, 1);
      const baseA = a.a.tokens[a.a.targetIndex].base;
      const baseB = a.b.tokens[a.b.targetIndex].base;
      expect(baseA).toEqual(baseB);
      expect(baseA).toEqual(ex.base);
    }
  });

  it("attendance weights each sum to 1 and are non-negative", () => {
    for (const ex of POLYSEMY_EXAMPLES) {
      const a = analyzePolysemy(ex, 1);
      for (const row of [...a.a.weights, ...a.b.weights]) {
        const sum = row.reduce((s, v) => s + v, 0);
        expect(sum).toBeCloseTo(1, 8);
        for (const v of row) expect(v).toBeGreaterThanOrEqual(0);
      }
    }
  });

  it("runs the target into the matching sense region: context is closer to the correct region than the other", () => {
    for (const ex of POLYSEMY_EXAMPLES) {
      const a = analyzePolysemy(ex, 1);
      const regionCenters = Object.fromEntries(
        Object.values(ex.regions).map((r) => [r.key, r.nextWords[0].near]),
      );
      for (const s of [a.a, a.b]) {
        const distToOwn = euclideanDistance(s.targetContext, regionCenters[s.senseKey]);
        const otherKey = s.senseKey === a.a.senseKey ? a.b.senseKey : a.a.senseKey;
        const distToOther = euclideanDistance(s.targetContext, regionCenters[otherKey]);
        expect(distToOwn).toBeLessThan(distToOther);
      }
    }
  });

  it("top next-word prediction belongs to the sentence's own sense", () => {
    for (const ex of POLYSEMY_EXAMPLES) {
      const a = analyzePolysemy(ex, 1);
      expect(a.a.predictions[0].senseKey).toBe(a.a.senseKey);
      expect(a.b.predictions[0].senseKey).toBe(a.b.senseKey);
    }
  });

  it("contextual embeddings differ between the two senses of the same word", () => {
    for (const ex of POLYSEMY_EXAMPLES) {
      const a = analyzePolysemy(ex, 1);
      expect(a.contextualDistance).toBeGreaterThan(0.5);
    }
  });

  it("the target moves away from its base (context != base) once attention is applied", () => {
    for (const ex of POLYSEMY_EXAMPLES) {
      const a = analyzePolysemy(ex, 1);
      for (const s of [a.a, a.b]) {
        const tc = s.tokens[s.targetIndex];
        expect(tc.shift).toBeGreaterThan(0.5);
      }
    }
  });
});
