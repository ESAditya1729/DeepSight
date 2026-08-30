// Pure computation layer for the ML Pipeline visualization. Everything here is
// framework-agnostic (no React) so it can be unit-tested deterministically and
// shared between the interactive steps and the test suite.
//
// The vocabulary, the scoring, and the private attention computation all live
// here so that the AttentionStep / PredictStep UI and the tests exercise the
// exact same logic.
//
// ATTENTION MODEL: the pipeline's toy embeddings are 2D, so we run self-attention
// with *identity projections* — each token acts as its own Query, Key and Value
// (Q = K = V = the embedding). Every number is therefore traceable straight back
// to the Embed step, while the computation is the real one:
//
//   scores[i][j] = q_i · k_j            (dot products — can be negative)
//   scaled[i][j] = scores[i][j] / √d_k  (d_k = 2 here)
//   weights[i][j] = softmax(scaled[i])  (each row sums to 1, temperature-aware)
//   output[i]     = Σ_j weights[i][j] · v_j   (a blended 2D point)

import {
  fromVectors,
  multiply,
  transpose,
  scaledDotProductAttention,
  type Matrix,
} from "@ml-visual-lab/ml-core";
import { getBasePosition, getWordPosition, mapSimilarity, normalizeWord, categoryFor } from "./embeddingSpace";

export { softmax } from "@ml-visual-lab/ml-core";

export interface WordEmbedding {
  word: string;
  x: number;
  y: number;
}

export interface Prediction {
  word: string;
  score: number;
  category: string;
}

/** All candidate next-words, grouped by semantic category. These must be subset of
 * (or consistent with) the CATEGORIES in embeddingSpace.ts so they live in the same
 * coordinate space as the sentence tokens. */
export const VOCAB_WORDS: { word: string; category: string }[] = [
  // Surfaces
  { word: "mat", category: "Surface" },
  { word: "bed", category: "Surface" },
  { word: "chair", category: "Surface" },
  { word: "table", category: "Surface" },
  { word: "floor", category: "Surface" },
  { word: "rug", category: "Surface" },
  { word: "sofa", category: "Surface" },
  { word: "couch", category: "Surface" },
  // Places
  { word: "river", category: "Place" },
  { word: "park", category: "Place" },
  { word: "store", category: "Place" },
  { word: "school", category: "Place" },
  { word: "home", category: "Place" },
  { word: "office", category: "Place" },
  { word: "shop", category: "Place" },
  // Animals
  { word: "dog", category: "Animal" },
  { word: "bird", category: "Animal" },
  { word: "fish", category: "Animal" },
  { word: "lion", category: "Animal" },
  { word: "horse", category: "Animal" },
  // Actions
  { word: "ran", category: "Action" },
  { word: "jumped", category: "Action" },
  { word: "walked", category: "Action" },
  { word: "played", category: "Action" },
  { word: "slept", category: "Action" },
  { word: "flew", category: "Action" },
  // Size
  { word: "small", category: "Size" },
  { word: "large", category: "Size" },
  // Colors
  { word: "red", category: "Color" },
  { word: "blue", category: "Color" },
  { word: "green", category: "Color" },
  // Emotion
  { word: "sad", category: "Emotion" },
  { word: "angry", category: "Emotion" },
  // People
  { word: "king", category: "Person" },
  { word: "queen", category: "Person" },
  { word: "man", category: "Person" },
  { word: "woman", category: "Person" },
  // Speed
  { word: "quick", category: "Speed" },
  { word: "slow", category: "Speed" },
  // Money (share the same region as places, close to "bank")
  { word: "money", category: "Money" },
  { word: "cash", category: "Money" },
  { word: "deposit", category: "Money" },
  { word: "pay", category: "Money" },
];

/** Words that should never be predicted (they'd always be wrong / are trivial). */
const BLOCKED = new Set(["bank"]);

/** Compute sentence embeddings for each token (order-preserving, occurrence-aware). */
export function generateEmbeddings(tokens: string[], seed: number): WordEmbedding[] {
  const occurrenceCount: Record<string, number> = {};
  return tokens.map((token) => {
    const lower = normalizeWord(token);
    const occurrence = occurrenceCount[lower] ?? 0;
    occurrenceCount[lower] = occurrence + 1;
    const { x, y } = getWordPosition(token, seed, occurrence);
    return { word: token, x, y };
  });
}

/** Similarity of every token pair using mapSimilarity (Gaussian on distance). */
export function computeSimilarityMatrix(embeddings: WordEmbedding[]): number[][] {
  return embeddings.map((a) => embeddings.map((b) => mapSimilarity([a.x, a.y], [b.x, b.y])));
}

export interface AttentionBreakdown {
  /** raw dot-product queries against keys: q_i · k_j (can be negative) */
  scores: Matrix;
  /** scores scaled by 1/√d_k */
  scaled: Matrix;
  /** row-wise softmax of scaled — each row sums to 1 */
  weights: Matrix;
  /** blended values: output[i] = Σ_j weights[i][j] · v_j (one point per token) */
  output: Matrix;
  /** the embedding dimension used as d_k (2 in the pipeline) */
  dK: number;
  /** √d_k — the number every score is divided by */
  scale: number;
}

/**
 * Runs real scaled dot-product self-attention on the pipeline's 2D embeddings.
 * Identity projections mean Q = K = V = the embeddings themselves, so the whole
 * computation is visible: read the dot products, scale them, softmax them, blend.
 */
export function computeAttentionBreakdown(embeddings: WordEmbedding[], temperature: number): AttentionBreakdown {
  const Q = fromVectors(...embeddings.map((e) => [e.x, e.y]));
  const scores = multiply(Q, transpose(Q));
  const { scores: scaled, weights, output } = scaledDotProductAttention(Q, Q, Q, temperature);
  return { scores, scaled, weights, output, dK: 2, scale: Math.sqrt(2) };
}

/**
 * The context used for next-word prediction: the *last* token asks the question,
 * so its attended output (the blend of everyone's values it settled on) is what
 * gets compared against the vocabulary.
 */
export function attentionContext(breakdown: AttentionBreakdown): [number, number] {
  const row = breakdown.output[breakdown.output.length - 1];
  return [row[0], row[1]];
}

/**
 * Positions the candidate vocabulary in the same coordinate space as the sentence
 * tokens for a given seed.
 */
export function positionVocab(seed: number): { word: string; x: number; y: number; category: string }[] {
  return VOCAB_WORDS.map((v) => ({ ...v, ...getBasePosition(v.word, seed) }));
}

// A tiny "training set" the model has (allegedly) learned from. Matching the tail of the
// user's input against these lets the model pick the sensible category even when raw
// geometric nearest-neighbour would be fooled by overlapping clusters. Each entry is the
// sentence plus the word that followed it in "training".
interface TrainingExample {
  text: string;
  nextWord: string;
}

const TRAINING_EXAMPLES: TrainingExample[] = [
  { text: "The cat sat on the", nextWord: "mat" },
  { text: "The dog slept on the", nextWord: "rug" },
  { text: "She went to the bank to", nextWord: "deposit" },
  { text: "The happy child played", nextWord: "jumped" },
  { text: "The quick brown fox jumps", nextWord: "quick" },
  { text: "A king and queen rule", nextWord: "king" },
];

/**
 * Finds the training example with the longest matching tail against the input. Returns
 * its category label, or null if the input doesn't resemble any example closely enough.
 */
export function matchTrainingExample(inputTokens: string[]): { nextWord: string; category: string } | null {
  const inputNorm = inputTokens.map((w) => normalizeWord(w));
  let best = { nextWord: "", category: "", overlap: 0 };

  for (const ex of TRAINING_EXAMPLES) {
    const exNorm = normalizeWord(ex.text).split(/\s+/);
    let overlap = 0;
    while (
      overlap < exNorm.length &&
      overlap < inputNorm.length &&
      exNorm[exNorm.length - 1 - overlap] === inputNorm[inputNorm.length - 1 - overlap]
    ) {
      overlap++;
    }
    // Only treat it as a match when we cover most of the example (so a 1-word tail like
    // "the" doesn't match every sentence). Requires overlap across at least 2 words.
    if (overlap >= 2 && overlap >= exNorm.length - 1 && overlap > best.overlap) {
      const cat = categoryFor(normalizeWord(ex.nextWord))?.name ?? "";
      best = { nextWord: ex.nextWord, category: cat, overlap };
    }
  }

  return best.overlap > 0 ? { nextWord: best.nextWord, category: best.category } : null;
}

/**
 * Rank the entire vocabulary by "how well it fits here". Uses a two-stage model:
 *
 *  1. CASE-BASED: look for a training example whose ending matches the tail of the input
 *     (the "similar input I've seen before" reasoning a child understands). If a match is
 *     found, the training answer's category is highlighted.
 *  2. GEOMETRIC: score every vocabulary word by similarity to the attention context,
 *     so even unseen inputs produce sensible, confidence-scaled predictions.
 *
 * The two are blended: the matched category's words get a boost, and geometric similarity
 * orders the results within/across categories.
 */
export function predictNextWord(
  context: [number, number],
  vocab: { word: string; x: number; y: number; category: string }[],
  inputTokens: string[],
  topN = 8,
): Prediction[] {
  const used = new Set(inputTokens.map((w) => normalizeWord(w)));

  // Geometric base score for every candidate.
  const candidates = vocab
    .filter((v) => !used.has(normalizeWord(v.word)) && !BLOCKED.has(normalizeWord(v.word)))
    .map((v) => ({
      word: v.word,
      category: v.category,
      score: mapSimilarity(context, [v.x, v.y]),
    }));

  // Case-based category boost: when the input resembles a training example, the word the
  // model "remembers" following that example is its best guess and ranks first, followed
  // by the rest of that example's category (ordered by geometric similarity), then the
  // remaining vocabulary. This mirrors how a real model "uses what it has seen before".
  const matched = matchTrainingExample(inputTokens);
  if (matched) {
    const remembered = candidates.find((c) => c.word === matched.nextWord);
    const inMatched = candidates
      .filter((c) => c.category === matched.category && c.word !== matched.nextWord)
      .sort((a, b) => b.score - a.score);
    const others = candidates
      .filter((c) => c.category !== matched.category)
      .sort((a, b) => b.score - a.score);
    candidates.length = 0;
    if (remembered) candidates.push(remembered);
    candidates.push(...inMatched, ...others);
  } else {
    candidates.sort((a, b) => b.score - a.score);
  }
  const top = candidates.slice(0, topN);
  if (top.length === 0) return [];

  const maxScore = top[0].score;
  const minScore = top[top.length - 1].score;
  const range = maxScore - minScore || 1;
  return top.map((s) => ({ ...s, score: (s.score - minScore) / range }));
}

// Expected answer (for the default presets) — the word a child is meant to discover.
// Used by the UI to show a "did you get it?" result and by tests to verify correctness.
// Must stay consistent with TRAINING_EXAMPLES.
export const GROUND_TRUTH: Record<string, string> = {
  "The cat sat on the": "mat",
  "She went to the bank to": "deposit",
  "The happy child played": "jumped",
  "The quick brown fox jumps": "quick",
};