import { createLayer, forwardLayer, backwardLayer, updateLayer, type LayerWeights, type LayerCache } from "./layers.js";

export interface Network {
  layers: LayerWeights[];
  caches: LayerCache[];
}

export interface NetworkSnapshot {
  activations: number[][];
  weights: number[][][];
}

export function createNetwork(layerSizes: number[], activations: string[]): Network {
  if (layerSizes.length < 2) throw new Error("Need at least input and output layers");
  if (activations.length !== layerSizes.length - 1) {
    throw new Error("activations.length must equal layerSizes.length - 1");
  }
  const layers: LayerWeights[] = [];
  for (let i = 0; i < layerSizes.length - 1; i++) {
    layers.push(createLayer(layerSizes[i], layerSizes[i + 1], activations[i]));
  }
  return { layers, caches: [] };
}

export function forward(net: Network, input: number[]): number[] {
  net.caches = [];
  let current = [...input];
  for (const layer of net.layers) {
    const { output, cache } = forwardLayer(layer, current);
    net.caches.push(cache);
    current = output;
  }
  return current;
}

export function backward(net: Network, dOutput: number[], lr: number): void {
  let dCurrent = [...dOutput];
  const allGradients: { dw: number[][]; db: number[] }[] = [];

  for (let i = net.layers.length - 1; i >= 0; i--) {
    const { dInput, dw, db } = backwardLayer(net.layers[i], net.caches[i], dCurrent);
    allGradients.unshift({ dw, db });
    dCurrent = dInput;
  }

  for (let i = 0; i < net.layers.length; i++) {
    updateLayer(net.layers[i], allGradients[i].dw, allGradients[i].db, lr);
  }
}

export function getSnapshot(net: Network, input: number[]): NetworkSnapshot {
  const savedCaches = net.caches.map((c) => ({ ...c }));
  const savedWeights = net.layers.map((l) => ({
    weights: l.weights.map((r) => [...r]),
    biases: [...l.biases],
  }));

  forward(net, input);

  const activations: number[][] = [input];
  for (const cache of net.caches) {
    activations.push([...cache.a]);
  }

  // Restore
  for (let i = 0; i < net.layers.length; i++) {
    net.layers[i].weights = savedWeights[i].weights;
    net.layers[i].biases = savedWeights[i].biases;
  }
  net.caches = savedCaches;

  return {
    activations,
    weights: net.layers.map((l) => l.weights.map((r) => [...r])),
  };
}
