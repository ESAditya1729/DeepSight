// Pure computation layer for the "Word Senses & Contextual Embeddings" visualization.
// Framework-agnostic (no React) so it can be unit-tested deterministically.
//
// THE IDEA
// --------
// A word like "apple" has several meanings (a fruit, or a tech company). A static
// embedding — the dictionary lookup a model starts with — has ONE vector for "apple"
// no matter the sentence. Only once the sentence is run through self-attention does
// the word's meaning sharpen: attention *blends* the surrounding words into "apple"'s
// vector, so its contextual embedding drifts toward whichever meaning the sentence
// supports. The SAME base vector becomes two different contextual vectors.
//
// GEOMETRY
// --------
// Every polysemy example lays out a small 2D "sense map". The ambiguous word sits at a
// shared neutral base position, and its two senses occupy two regions on either side.
// Content words of each sentence are placed inside the relevant region. With identity
// projections (Q = K = V = the embedding), self-attention turns each token into a blend
// of the sentence's tokens weighted by score, so the target word's contextual embedding
// lands in the region its sentence points at.
//
//   Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) · V

import {
  transpose,
  multiply,
  scaledDotProductAttention,
  euclideanDistance,
  type Matrix,
} from "@ml-visual-lab/ml-core";

export { softmax, euclideanDistance } from "@ml-visual-lab/ml-core";

export type Vec2 = readonly [number, number];

export interface SenseRegion {
  key: string;
  label: string;
  emoji: string;
  nextWords: { word: string; near: Vec2 }[];
}

export interface SentenceDef {
  label: string;
  senseKey: string;
  tokens: { word: string; pos: Vec2 }[];
}

export interface PolysemyExample {
  slug: string;
  /** the ambiguous word shared by both sentences */
  target: string;
  /** one shared vector for the target before context is applied */
  base: Vec2;
  sentence: string;
  regions: Record<string, SenseRegion>;
  sentences: SentenceDef[];
  blurb: string;
}

// ---------------------------------------------------------------------------
// The showcased examples. Coordinates are hand-tuned so identity self-attention
// reliably pulls the target into the right region. Function words live at the
// origin so they add no directional pull.
// ---------------------------------------------------------------------------
const ORIGIN: Vec2 = [0, 0];

export const POLYSEMY_EXAMPLES: PolysemyExample[] = [
  {
    slug: "apple",
    target: "apple",
    base: [2, 0],
    sentence: "Apple the fruit vs. Apple the company",
    blurb:
      "“Apple” can be a piece of fruit or a computer company. It starts at a single neutral vector, then attention pulls it toward whichever neighbours the sentence supplies.",
    regions: {
      fruit: {
        key: "fruit",
        label: "Fruit",
        emoji: "🍎",
        nextWords: [
          { word: "orange", near: [4.2, 3] },
          { word: "banana", near: [4, 2] },
          { word: "pear", near: [3, 3] },
          { word: "juice", near: [5, 3] },
        ],
      },
      tech: {
        key: "tech",
        label: "Tech company",
        emoji: "📱",
        nextWords: [
          { word: "phone", near: [4.2, -3] },
          { word: "watch", near: [4, -2] },
          { word: "computer", near: [3, -3] },
          { word: "unveiled", near: [5, -4] },
        ],
      },
    },
    sentences: [
      {
        label: "Please buy an apple and an orange",
        senseKey: "fruit",
        tokens: [
          { word: "Please", pos: ORIGIN },
          { word: "buy", pos: ORIGIN },
          { word: "an", pos: ORIGIN },
          { word: "apple", pos: [2, 0] },
          { word: "and", pos: ORIGIN },
          { word: "an", pos: ORIGIN },
          { word: "orange", pos: [4.2, 3] },
        ],
      },
      {
        label: "Apple unveiled a new phone",
        senseKey: "tech",
        tokens: [
          { word: "Apple", pos: [2, 0] },
          { word: "unveiled", pos: [5, -4] },
          { word: "a", pos: ORIGIN },
          { word: "new", pos: ORIGIN },
          { word: "phone", pos: [4.2, -3] },
        ],
      },
    ],
  },
  {
    slug: "bank",
    target: "bank",
    base: [2, 0],
    sentence: "Bank the riverbank vs. bank the institution",
    blurb:
      "“Bank” can be the side of a river or a place that holds money. This 2D map treats both as real places—up is rivers, down is finance—so the context decides which “bank” you mean.",
    regions: {
      river: {
        key: "river",
        label: "Riverbank",
        emoji: "🏞️",
        nextWords: [
          { word: "river", near: [4.2, 3] },
          { word: "shore", near: [4, 2] },
          { word: "water", near: [5, 3] },
          { word: "boat", near: [3, 3] },
        ],
      },
      money: {
        key: "money",
        label: "Finance",
        emoji: "💰",
        nextWords: [
          { word: "money", near: [4.2, -3] },
          { word: "deposit", near: [4, -2] },
          { word: "cash", near: [5, -3] },
          { word: "loan", near: [3, -3] },
        ],
      },
    },
    sentences: [
      {
        label: "The fisherman stood on the bank of the river",
        senseKey: "river",
        tokens: [
          { word: "The", pos: ORIGIN },
          { word: "fisherman", pos: [4, 3] },
          { word: "stood", pos: [3, 3] },
          { word: "on", pos: ORIGIN },
          { word: "the", pos: ORIGIN },
          { word: "bank", pos: [2, 0] },
          { word: "of", pos: ORIGIN },
          { word: "the", pos: ORIGIN },
          { word: "river", pos: [4.2, 3] },
        ],
      },
      {
        label: "He withdrew cash from the bank",
        senseKey: "money",
        tokens: [
          { word: "He", pos: ORIGIN },
          { word: "withdrew", pos: [4, -2] },
          { word: "cash", pos: [5, -3] },
          { word: "from", pos: ORIGIN },
          { word: "the", pos: ORIGIN },
          { word: "bank", pos: [2, 0] },
        ],
      },
    ],
  },
  {
    slug: "bat",
    target: "bat",
    base: [2, 0],
    sentence: "Bat the animal vs. bat the sports equipment",
    blurb:
      "“Bat” can be a flying mammal or a piece of sports gear. On this map up is animals and down is sports, so a cave pulls “bat” upward while a baseball game pulls it downward.",
    regions: {
      animal: {
        key: "animal",
        label: "Animal",
        emoji: "🦇",
        nextWords: [
          { word: "cave", near: [4, 3] },
          { word: "wing", near: [3, 3] },
          { word: "night", near: [5, 3] },
          { word: "fly", near: [3, 4] },
        ],
      },
      sport: {
        key: "sport",
        label: "Sports",
        emoji: "⚾",
        nextWords: [
          { word: "ball", near: [4.2, -3] },
          { word: "pitch", near: [3, -3] },
          { word: "glove", near: [5, -3] },
          { word: "baseball", near: [4, -4] },
        ],
      },
    },
    sentences: [
      {
        label: "The bat flew out of the cave",
        senseKey: "animal",
        tokens: [
          { word: "The", pos: ORIGIN },
          { word: "bat", pos: [2, 0] },
          { word: "flew", pos: [3, 4] },
          { word: "out", pos: ORIGIN },
          { word: "of", pos: ORIGIN },
          { word: "the", pos: ORIGIN },
          { word: "cave", pos: [4, 3] },
        ],
      },
      {
        label: "The player hit the ball with a bat",
        senseKey: "sport",
        tokens: [
          { word: "The", pos: ORIGIN },
          { word: "player", pos: [4, -3] },
          { word: "hit", pos: [3, -3] },
          { word: "the", pos: ORIGIN },
          { word: "ball", pos: [4.2, -3] },
          { word: "with", pos: ORIGIN },
          { word: "a", pos: ORIGIN },
          { word: "bat", pos: [2, 0] },
        ],
      },
    ],
  },
  {
    slug: "spring",
    target: "spring",
    base: [2, 0],
    sentence: "Spring the season vs. spring the coil",
    blurb:
      "“Spring” can be a season or a coiled piece of metal. Up on this map is nature, down is hardware—the words around it decide which “spring” the sentence means.",
    regions: {
      season: {
        key: "season",
        label: "Season",
        emoji: "🌸",
        nextWords: [
          { word: "flower", near: [4, 3] },
          { word: "bloom", near: [3, 3] },
          { word: "warm", near: [5, 3] },
          { word: "summer", near: [4, 4] },
        ],
      },
      coil: {
        key: "coil",
        label: "Coil / hardware",
        emoji: "🛠️",
        nextWords: [
          { word: "coil", near: [4.2, -3] },
          { word: "metal", near: [4, -2] },
          { word: "wound", near: [3, -3] },
          { word: "tension", near: [5, -3] },
        ],
      },
    },
    sentences: [
      {
        label: "The flowers bloom in spring",
        senseKey: "season",
        tokens: [
          { word: "The", pos: ORIGIN },
          { word: "flowers", pos: [4, 3] },
          { word: "bloom", pos: [3, 3] },
          { word: "in", pos: ORIGIN },
          { word: "spring", pos: [2, 0] },
        ],
      },
      {
        label: "The coil is a metal spring",
        senseKey: "coil",
        tokens: [
          { word: "The", pos: ORIGIN },
          { word: "coil", pos: [4.2, -3] },
          { word: "is", pos: ORIGIN },
          { word: "a", pos: ORIGIN },
          { word: "metal", pos: [4, -2] },
          { word: "spring", pos: [2, 0] },
        ],
      },
    ],
  },
];

// ---------------------------------------------------------------------------
// The shared attention engine
// ---------------------------------------------------------------------------

export interface TokenContext {
  word: string;
  isTarget: boolean;
  /** the shared (context-free) base embedding */
  base: Vec2;
  /** the contextual embedding after attention — the "moving" value */
  contextual: Vec2;
  /** how far the token travelled from its base */
  shift: number;
}

export interface SentenceAnalysis {
  label: string;
  senseKey: string;
  senseLabel: string;
  senseEmoji: string;
  tokens: TokenContext[];
  /** n×n matrix row i = token i's attention over all tokens */
  scores: Matrix;
  scaled: Matrix;
  weights: Matrix;
  outputs: Matrix;
  dK: number;
  targetIndex: number;
  targetContext: Vec2;
  predictions: { word: string; score: number; senseKey: string }[];
}

export interface PolysemyAnalysis {
  example: PolysemyExample;
  a: SentenceAnalysis;
  b: SentenceAnalysis;
  /** distance between the target's contextual vectors in each sentence */
  contextualDistance: number;
}

/** Wraps region next-words into candidate vectors for scoring. */
function candidatesFor(region: SenseRegion): { word: string; pos: Vec2; senseKey: string }[] {
  return region.nextWords.map((n) => ({ word: n.word, pos: n.near, senseKey: region.key }));
}

/**
 * Ranks the sense region's candidate words by how close they are to a context
 * vector, using a Gaussian kernel on Euclidean distance (same metric the whole
 * "map-style" demo uses). Score is normalised so the best candidate is 1.
 */
function scoreCandidates(context: Vec2, candidates: { word: string; pos: Vec2; senseKey: string }[]): {
  word: string;
  score: number;
  senseKey: string;
}[] {
  const SIM_SIGMA_SQ = 8;
  const raw = candidates.map((c) => {
    const dx = context[0] - c.pos[0];
    const dy = context[1] - c.pos[1];
    const d2 = dx * dx + dy * dy;
    return {
      word: c.word,
      senseKey: c.senseKey,
      sim: Math.exp(-d2 / (2 * SIM_SIGMA_SQ)),
    };
  });
  raw.sort((p, q) => q.sim - p.sim);
  const top = raw[0]?.sim ?? 1;
  const bottom = raw[raw.length - 1]?.sim ?? 0;
  const range = top - bottom || 1;
  return raw.map((c) => ({ word: c.word, senseKey: c.senseKey, score: (c.sim - bottom) / range }));
}

/** Returns the raw, unscaled QK^T scores (before /√d_k) for display. */
function rawScores(Q: Matrix): Matrix {
  return multiply(Q, transpose(Q));
}

/**
 * Full analysis of a polysemy example: both sentences run through the same
 * attentional engine, plus the standout metric — how far the target word's
 * contextual embedding moved between the two senses.
 */
export function analyzePolysemy(example: PolysemyExample, temperature: number): PolysemyAnalysis {
  const [saDef, sbDef] = example.sentences;
  const targetLower = example.target.toLowerCase();

  const makeSentence = (def: SentenceDef): SentenceAnalysis => {
    const region = example.regions[def.senseKey];
    const Q = def.tokens.map((t) => [...t.pos]) as Matrix;
    const scores = rawScores(Q);
    const { scores: scaled, weights, output } = scaledDotProductAttention(Q, Q, Q, temperature);

    let targetIndex = -1;
    const tokens: TokenContext[] = def.tokens.map((t, i) => {
      const isTarget = t.word.toLowerCase() === targetLower;
      if (isTarget) targetIndex = i;
      const contextual: Vec2 = [output[i][0], output[i][1]];
      const shift = euclideanDistance(t.pos, contextual);
      return { word: t.word, isTarget, base: t.pos, contextual, shift };
    });

    const targetContext: Vec2 =
      targetIndex >= 0 ? ([output[targetIndex][0], output[targetIndex][1]] as Vec2) : example.base;

    const predictions = scoreCandidates(targetContext, candidatesFor(region));

    return {
      label: def.label,
      senseKey: def.senseKey,
      senseLabel: region.label,
      senseEmoji: region.emoji,
      tokens,
      scores,
      scaled,
      weights,
      outputs: output,
      dK: 2,
      targetIndex,
      targetContext,
      predictions,
    };
  };

  const a = makeSentence(saDef);
  const b = makeSentence(sbDef);
  const contextualDistance = euclideanDistance(a.targetContext, b.targetContext);

  return { example, a, b, contextualDistance };
}
