import { random } from "../matrix.js";
import { getActivation, type Activation } from "./activations.js";

export interface LayerWeights {
  weights: number[][];
  biases: number[];
  activation: Activation;
  inputSize: number;
  outputSize: number;
}

export interface LayerCache {
  input: number[];
  z: number[];
  a: number[];
  dw: number[][];
  db: number[];
}

export function createLayer(inputSize: number, outputSize: number, activationName: string): LayerWeights {
  const scale = Math.sqrt(2 / inputSize);
  const weights = random(outputSize, inputSize, scale);
  const biases = new Array(outputSize).fill(0);
  return {
    weights: weights.map((r) => [...r]),
    biases: [...biases],
    activation: getActivation(activationName),
    inputSize,
    outputSize,
  };
}

export function forwardLayer(layer: LayerWeights, input: number[]): { output: number[]; cache: LayerCache } {
  const { weights, biases, activation } = layer;
  const z: number[] = [];
  for (let i = 0; i < weights.length; i++) {
    let sum = biases[i];
    for (let j = 0; j < input.length; j++) {
      sum += weights[i][j] * input[j];
    }
    z.push(sum);
  }
  const a = z.map((v) => activation.fn(v));
  return {
    output: a,
    cache: { input: [...input], z, a, dw: [], db: [] },
  };
}

export function backwardLayer(
  layer: LayerWeights,
  cache: LayerCache,
  dOutput: number[],
): { dInput: number[]; dw: number[][]; db: number[] } {
  const { weights, activation } = layer;
  const { input, z } = cache;
  const m = input.length;

  // dActivation
  const dZ = dOutput.map((d, i) => d * activation.prime(z[i]));

  const dw: number[][] = [];
  const db: number[] = [];
  const dInput = new Array(m).fill(0);

  for (let i = 0; i < weights.length; i++) {
    const row: number[] = [];
    for (let j = 0; j < m; j++) {
      row.push(dZ[i] * input[j]);
      dInput[j] += weights[i][j] * dZ[i];
    }
    dw.push(row);
    db.push(dZ[i]);
  }

  return { dInput, dw, db };
}

export function updateLayer(layer: LayerWeights, dw: number[][], db: number[], lr: number): void {
  for (let i = 0; i < layer.weights.length; i++) {
    for (let j = 0; j < layer.weights[i].length; j++) {
      layer.weights[i][j] -= lr * dw[i][j];
    }
    layer.biases[i] -= lr * db[i];
  }
}
