// Shared toy "embedding space" for the pipeline demo. Both the Embed step (positions the
// sentence's own tokens) and the Predict step (positions the fixed candidate vocabulary) must
// place a given word at the same spot for a given seed — otherwise next-word predictions are
// scored against a vocabulary that doesn't actually live in the same space as the sentence.
//
// getWordPosition is a pure function of (word, seed, occurrence): unlike a sequential seeded
// RNG, it doesn't matter what order words are processed in or what list they came from.

import { euclideanDistance } from "@ml-visual-lab/ml-core";

export interface Category {
  name: string;
  prefix: string[];
  cx: number;
  cy: number;
  color: string;
}

export const CATEGORIES: Category[] = [
  { name: "Function", prefix: ["the", "a", "an", "this", "that", "to", "on", "in", "of", "for", "is", "are", "was", "were", "at", "by", "with", "from"], cx: 0, cy: 0, color: "#64748b" },
  { name: "Animal", prefix: ["cat", "dog", "bird", "fish", "lion", "tiger", "horse", "puppy", "kitten", "fox", "bear", "wolf", "deer", "rabbit"], cx: 4, cy: 0.5, color: "#059669" },
  { name: "Action", prefix: ["sat", "ran", "jumped", "walked", "flew", "swam", "slept", "ate", "played", "went", "came", "saw", "heard"], cx: -1, cy: 3, color: "#d97706" },
  { name: "Size", prefix: ["big", "small", "large", "tiny", "huge", "little", "tall", "short", "wide", "narrow"], cx: 0.5, cy: -2, color: "#ec4899" },
  { name: "Color", prefix: ["red", "blue", "green", "yellow", "black", "white", "orange", "purple", "pink", "brown"], cx: -4, cy: -0.5, color: "#dc2626" },
  { name: "Emotion", prefix: ["happy", "sad", "angry", "joyful", "fearful", "brave", "calm", "kind", "wise", "strong"], cx: 1.5, cy: -3, color: "#f43f5e" },
  { name: "Person", prefix: ["man", "woman", "boy", "girl", "father", "mother", "brother", "sister", "husband", "wife", "king", "queen", "prince", "princess"], cx: 3, cy: -2.5, color: "#f97316" },
  { name: "Money", prefix: ["money", "bank", "deposit", "account", "cash", "payment", "buy", "sell", "price", "cost", "pay", "worth"], cx: 5, cy: 2, color: "#8b5cf6" },
  { name: "Surface", prefix: ["mat", "bed", "chair", "table", "floor", "room", "house", "home", "door", "window", "wall"], cx: 2, cy: 1.5, color: "#7c3aed" },
  { name: "Speed", prefix: ["quick", "slow", "fast", "rapid", "speed", "hurry", "rush", "wait", "stop", "go"], cx: -2, cy: 2, color: "#06b6d4" },
];

/** Color for a word based on its category, or a default neutral color for unknown words. */
export function categoryColor(word: string, fallback = "#94a3b8"): string {
  const cat = categoryFor(normalizeWord(word));
  return cat ? cat.color : fallback;
}

export function normalizeWord(word: string): string {
  return word.toLowerCase().replace(/[.,!?;:'"]/g, "");
}

export function categoryFor(lower: string): Category | undefined {
  return CATEGORIES.find((c) => c.prefix.includes(lower));
}

function hashString(s: string): number {
  let h = 2166136261;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}

function hashUnit(word: string, seed: number, salt: string): number {
  return (hashString(`${word}|${seed}|${salt}`) % 1_000_000) / 1_000_000;
}

/** Canonical position for a word — same for every caller given the same (word, seed). */
export function getBasePosition(word: string, seed: number): { x: number; y: number } {
  const lower = normalizeWord(word);
  const u1 = hashUnit(lower, seed, "base-x") - 0.5;
  const u2 = hashUnit(lower, seed, "base-y") - 0.5;
  const cat = categoryFor(lower);
  if (cat) return { x: cat.cx + u1 * 1.5, y: cat.cy + u2 * 1.5 };
  return { x: u1 * 6, y: u2 * 4 };
}

/**
 * Position for the Nth occurrence of a word in a sentence. Occurrence 0 is the canonical base
 * position; later occurrences get a small jitter around it so repeated words don't render as
 * exactly overlapping points.
 */
export function getWordPosition(word: string, seed: number, occurrence = 0): { x: number; y: number } {
  const base = getBasePosition(word, seed);
  if (occurrence === 0) return base;
  const lower = normalizeWord(word);
  const u1 = hashUnit(lower, seed, `jitter${occurrence}-x`) - 0.5;
  const u2 = hashUnit(lower, seed, `jitter${occurrence}-y`) - 0.5;
  return { x: base.x + u1 * 0.3, y: base.y + u2 * 0.3 };
}

const SIMILARITY_SIGMA = 2;

/**
 * Similarity for this "map"-style embedding space. Meaning here is encoded as position
 * (category clusters scattered across the plane), not as direction from the origin — so a
 * Gaussian kernel on distance is the right metric. Cosine similarity ignores distance
 * entirely and compares angle-from-origin instead, which puts unrelated but similarly-angled
 * categories (e.g. Money at (5,2) and the Surface category at (2,1.5)) at ~0.97 "similarity"
 * despite being well-separated clusters on the map.
 */
export function mapSimilarity(a: readonly [number, number], b: readonly [number, number]): number {
  const d = euclideanDistance(a, b);
  return Math.exp(-(d * d) / (2 * SIMILARITY_SIGMA * SIMILARITY_SIGMA));
}
