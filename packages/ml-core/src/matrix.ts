import type { Vector } from "./vector.js";

export type Matrix = readonly (readonly number[])[];

export function rows(m: Matrix): number {
  return m.length;
}

export function cols(m: Matrix): number {
  return m[0]?.length ?? 0;
}

export function zeros(r: number, c: number): Matrix {
  return Array.from({ length: r }, () => Array(c).fill(0));
}

export function random(r: number, c: number, scale = 1): Matrix {
  return Array.from({ length: r }, () =>
    Array.from({ length: c }, () => (Math.random() * 2 - 1) * scale),
  );
}

export function fromVectors(...vectors: Vector[]): Matrix {
  return vectors.map((v) => [...v]);
}

export function get(m: Matrix, r: number, c: number): number {
  return m[r][c];
}

export function set(m: Matrix, r: number, c: number, value: number): Matrix {
  return m.map((row, ri) => row.map((v, ci) => (ri === r && ci === c ? value : v)));
}

export function transpose(m: Matrix): Matrix {
  const rc = rows(m);
  const cc = cols(m);
  return Array.from({ length: cc }, (_, c) =>
    Array.from({ length: rc }, (_, r) => m[r][c]),
  );
}

export function multiply(a: Matrix, b: Matrix): Matrix {
  const aR = rows(a);
  const aC = cols(a);
  const bC = cols(b);
  if (aC !== rows(b)) {
    throw new Error(`Incompatible dimensions: ${aR}x${aC} * ${rows(b)}x${bC}`);
  }
  return Array.from({ length: aR }, (_, r) =>
    Array.from({ length: bC }, (_, c) => {
      let sum = 0;
      for (let k = 0; k < aC; k++) sum += a[r][k] * b[k][c];
      return sum;
    }),
  );
}

export function multiplyVector(m: Matrix, v: Vector): Vector {
  if (cols(m) !== v.length) {
    throw new Error(`Incompatible: ${rows(m)}x${cols(m)} * vec(${v.length})`);
  }
  return Array.from({ length: rows(m) }, (_, r) => {
    let sum = 0;
    for (let k = 0; k < v.length; k++) sum += m[r][k] * v[k];
    return sum;
  });
}

export function elementWise(a: Matrix, b: Matrix, fn: (a: number, b: number) => number): Matrix {
  return a.map((row, r) => row.map((v, c) => fn(v, b[r][c])));
}

export function map(m: Matrix, fn: (value: number, r: number, c: number) => number): Matrix {
  return m.map((row, r) => row.map((v, c) => fn(v, r, c)));
}

export function matAdd(a: Matrix, b: Matrix): Matrix {
  return elementWise(a, b, (x, y) => x + y);
}

export function matScale(m: Matrix, s: number): Matrix {
  return map(m, (v) => v * s);
}

export function toVectorArray(m: Matrix): Vector[] {
  return m.map((row) => [...row]);
}
