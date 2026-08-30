import { describe, it, expect } from "vitest";
import {
  generateEmbeddings,
  computeSimilarityMatrix,
  softmax,
  computeAttentionBreakdown,
  attentionContext,
  predictNextWord,
  positionVocab,
  VOCAB_WORDS,
  GROUND_TRUTH,
} from "./pipelineCore";

describe("generateEmbeddings", () => {
  it("returns one embedding per token, in order", () => {
    const tokens = ["The", "cat", "sat"];
    const embs = generateEmbeddings(tokens, 42);
    expect(embs.map((e) => e.word)).toEqual(tokens);
  });

  it("repeated words get jitter but stay near their base", () => {
    const embs = generateEmbeddings(["the", "the", "the"], 42);
    const [a, b, c] = embs;
    expect(a.x).toBeCloseTo(b.x, 0);
    expect(b.x).toBeCloseTo(c.x, 0);
  });

  it("is deterministic", () => {
    expect(generateEmbeddings(["cat", "mat"], 42)).toEqual(generateEmbeddings(["cat", "mat"], 42));
  });
});

describe("computeSimilarityMatrix", () => {
  it("is symmetric with a diagonal of 1", () => {
    const embs = generateEmbeddings(["cat", "mat", "sat"], 42);
    const m = computeSimilarityMatrix(embs);
    expect(m.length).toBe(3);
    for (let i = 0; i < 3; i++) {
      expect(m[i][i]).toBeCloseTo(1, 10);
      for (let j = 0; j < 3; j++) {
        expect(m[i][j]).toBeCloseTo(m[j][i], 10);
        expect(m[i][j]).toBeGreaterThanOrEqual(0);
        expect(m[i][j]).toBeLessThanOrEqual(1);
      }
    }
  });

  it("related words score higher than unrelated ones", () => {
    const embs = generateEmbeddings(["cat", "mat", "king"], 42);
    const m = computeSimilarityMatrix(embs);
    // cat↔mat (Animal vs Surface) vs cat↔king (Animal vs Person)
    expect(m[0][1]).toBeGreaterThan(m[0][2]);
  });
});

describe("softmax", () => {
  it("scores sum to 1 and are positive", () => {
    const s = softmax([2, 1, 0.1], 1);
    const sum = s.reduce((a, b) => a + b, 0);
    expect(sum).toBeCloseTo(1, 10);
    for (const v of s) expect(v).toBeGreaterThan(0);
  });

  it("lower temperature sharpens the distribution", () => {
    const s = softmax([3, 3, 0], 1);
    const sharp = softmax([3, 3, 0], 0.1);
    expect(sharp[0]).toBeGreaterThan(s[0]);
  });
});

describe("computeAttentionBreakdown", () => {
  it("weights rows sum to 1 and are positive", () => {
    const embs = generateEmbeddings(["cat", "mat", "sat"], 42);
    const b = computeAttentionBreakdown(embs, 1);
    expect(b.scores.length).toBe(3);
    for (const row of b.weights) {
      const sum = row.reduce((a, b) => a + b, 0);
      expect(sum).toBeCloseTo(1, 8);
      for (const v of row) expect(v).toBeGreaterThan(0);
    }
  });

  it("uses identity projections: scores are dot products of the embeddings", () => {
    const embs = generateEmbeddings(["cat", "mat", "sat"], 42);
    const b = computeAttentionBreakdown(embs, 1);
    for (let i = 0; i < 3; i++) {
      for (let j = 0; j < 3; j++) {
        const expected = embs[i].x * embs[j].x + embs[i].y * embs[j].y;
        expect(b.scores[i][j]).toBeCloseTo(expected, 10);
      }
    }
  });

  it("scaled = score / √d_k with d_k = 2", () => {
    const embs = generateEmbeddings(["cat"], 42);
    const b = computeAttentionBreakdown(embs, 1);
    expect(b.dK).toBe(2);
    expect(b.scale).toBeCloseTo(Math.sqrt(2), 10);
    const self = embs[0].x * embs[0].x + embs[0].y * embs[0].y;
    expect(b.scaled[0][0]).toBeCloseTo(self / Math.sqrt(2), 10);
  });

  it("output is a convex combination of the value vectors", () => {
    const embs = generateEmbeddings(["cat", "mat", "sat"], 42);
    const b = computeAttentionBreakdown(embs, 1);
    b.output.forEach((row, i) => {
      const x = embs.reduce((sum, e, j) => sum + b.weights[i][j] * e.x, 0);
      const y = embs.reduce((sum, e, j) => sum + b.weights[i][j] * e.y, 0);
      expect(row[0]).toBeCloseTo(x, 8);
      expect(row[1]).toBeCloseTo(y, 8);
    });
  });

  it("lower temperature concentrates the weights", () => {
    const embs = generateEmbeddings(["cat", "mat", "sat"], 42);
    const sharp = computeAttentionBreakdown(embs, 0.1).weights;
    const flat = computeAttentionBreakdown(embs, 5).weights;
    for (let i = 0; i < 3; i++) {
      expect(Math.max(...sharp[i])).toBeGreaterThanOrEqual(Math.max(...flat[i]));
    }
  });
});

describe("attentionContext", () => {
  it("single token: context is the token's own embedding", () => {
    const embs = generateEmbeddings(["cat"], 42);
    const b = computeAttentionBreakdown(embs, 1);
    const [cx, cy] = attentionContext(b);
    expect(cx).toBeCloseTo(embs[0].x, 10);
    expect(cy).toBeCloseTo(embs[0].y, 10);
  });

  it("multi token: context is the last token's attended output", () => {
    const embs = generateEmbeddings(["cat", "mat", "sat"], 42);
    const b = computeAttentionBreakdown(embs, 1);
    const [cx, cy] = attentionContext(b);
    const last = b.output[b.output.length - 1];
    expect(cx).toBeCloseTo(last[0], 10);
    expect(cy).toBeCloseTo(last[1], 10);
  });

  it("is a blend lying between the sentence's own embeddings", () => {
    const embs = generateEmbeddings(["cat", "mat", "sat"], 42);
    const b = computeAttentionBreakdown(embs, 1);
    const [cx, cy] = attentionContext(b);
    const xs = embs.map((e) => e.x);
    const ys = embs.map((e) => e.y);
    expect(cx).toBeGreaterThanOrEqual(Math.min(...xs) - 1e-9);
    expect(cx).toBeLessThanOrEqual(Math.max(...xs) + 1e-9);
    expect(cy).toBeGreaterThanOrEqual(Math.min(...ys) - 1e-9);
    expect(cy).toBeLessThanOrEqual(Math.max(...ys) + 1e-9);
  });
});

describe("positionVocab", () => {
  it("returns every vocab word with coordinates", () => {
    const vocab = positionVocab(42);
    expect(vocab.length).toBe(VOCAB_WORDS.length);
    for (const v of vocab) {
      expect(typeof v.x).toBe("number");
      expect(typeof v.y).toBe("number");
    }
  });
});

describe("predictNextWord", () => {
  it("never returns an input token", () => {
    const vocab = positionVocab(42);
    const preds = predictNextWord([2, 1.5], vocab, ["mat"]);
    for (const p of preds) expect(p.word.toLowerCase()).not.toBe("mat");
  });

  it("returns at most topN results", () => {
    const vocab = positionVocab(42);
    expect(predictNextWord([2, 1.5], vocab, []).length).toBeLessThanOrEqual(8);
  });

  it("predicts 'mat' for 'The cat sat on the ___' using the attention context", () => {
    const tokens = ["The", "cat", "sat", "on", "the"];
    const embs = generateEmbeddings(tokens, 42);
    const vocab = positionVocab(42);
    const b = computeAttentionBreakdown(embs, 1);
    const preds = predictNextWord(attentionContext(b), vocab, tokens);
    const expected = GROUND_TRUTH["The cat sat on the"];
    expect(preds[0].word).toBe(expected);
  });

  it("predicts 'deposit' (a Money word) for 'She went to the bank to ___' using the attention context", () => {
    const tokens = ["She", "went", "to", "the", "bank", "to"];
    const embs = generateEmbeddings(tokens, 42);
    const vocab = positionVocab(42);
    const b = computeAttentionBreakdown(embs, 1);
    const preds = predictNextWord(attentionContext(b), vocab, tokens);
    const expected = GROUND_TRUTH["She went to the bank to"];
    expect(preds[0].word).toBe(expected);
  });
});