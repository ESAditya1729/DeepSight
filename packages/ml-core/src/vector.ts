export type Vector = readonly number[];

export function add(a: Vector, b: Vector): Vector {
  return a.map((ai, i) => ai + b[i]);
}

export function subtract(a: Vector, b: Vector): Vector {
  return a.map((ai, i) => ai - b[i]);
}

export function scale(v: Vector, scalar: number): Vector {
  return v.map((vi) => vi * scalar);
}

export function dot(a: Vector, b: Vector): number {
  return a.reduce((sum, ai, i) => sum + ai * b[i], 0);
}

export function magnitude(v: Vector): number {
  return Math.sqrt(dot(v, v));
}

export function normalize(v: Vector): Vector {
  const m = magnitude(v);
  return m === 0 ? v.map(() => 0) : scale(v, 1 / m);
}

export function cosineSimilarity(a: Vector, b: Vector): number {
  const magA = magnitude(a);
  const magB = magnitude(b);
  if (magA === 0 || magB === 0) return 0;
  return dot(a, b) / (magA * magB);
}

export function euclideanDistance(a: Vector, b: Vector): number {
  return magnitude(subtract(a, b));
}

export function randomUnit(dimension: number): Vector {
  const v = Array.from({ length: dimension }, () => Math.random() * 2 - 1);
  return normalize(v);
}
