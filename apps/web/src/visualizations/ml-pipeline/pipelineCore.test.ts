import { describe, it, expect } from "vitest";
import {
  generateEmbeddings,
  computeSimilarityMatrix,
  softmax,
  computeAttention,
  contextVector,
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

describe("computeAttention", () => {
  it("rows sum to 1", () => {
    const embs = generateEmbeddings(["cat", "mat", "sat"], 42);
    const m = computeSimilarityMatrix(embs);
    const attn = computeAttention(m, 1);
    for (const row of attn) {
      const sum = row.reduce((a, b) => a + b, 0);
      expect(sum).toBeCloseTo(1, 8);
    }
  });
});

describe("contextVector", () => {
  it("favors the last content word over trailing stopwords", () => {
    // "mat" is a Surface word; the stopword "the" sits near the origin.
    // Context should be pulled toward the content words (mat is near (2,1.5)).
    const embs = generateEmbeddings(["the", "mat"], 42);
    const [cx] = contextVector(embs);
    // Should be closer to mat (2,1.5) than to "the" (0,0)
    const dxMat = Math.abs(cx - 2);
    const dxThe = Math.abs(cx - 0);
    expect(dxMat).toBeLessThan(dxThe);
  });

  it("handles a single token", () => {
    const embs = generateEmbeddings(["cat"], 42);
    const [cx, cy] = contextVector(embs);
    const cat = embs[0];
    expect(cx).toBeCloseTo(cat.x, 10);
    expect(cy).toBeCloseTo(cat.y, 10);
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

  it("predicts 'mat' for 'The cat sat on the ___'", () => {
    const tokens = ["The", "cat", "sat", "on", "the"];
    const embs = generateEmbeddings(tokens, 42);
    const vocab = positionVocab(42);
    const preds = predictNextWord(contextVector(embs), vocab, tokens);
    const expected = GROUND_TRUTH["The cat sat on the"];
    expect(preds[0].word).toBe(expected);
  });

  it("predicts 'deposit' (a Money word) for 'She went to the bank to ___'", () => {
    const tokens = ["She", "went", "to", "the", "bank", "to"];
    const embs = generateEmbeddings(tokens, 42);
    const vocab = positionVocab(42);
    const preds = predictNextWord(contextVector(embs), vocab, tokens);
    const expected = GROUND_TRUTH["She went to the bank to"];
    expect(preds[0].word).toBe(expected);
  });
});
