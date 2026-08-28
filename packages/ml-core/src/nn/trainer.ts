import { type Network, forward } from "./network.js";

export interface TrainResult {
  loss: number;
  predictions: number[][];
}

export function trainEpoch(
  net: Network,
  inputs: number[][],
  targets: number[][],
  lr: number,
): { loss: number; predictions: number[][] } {
  const predictions: number[][] = [];
  let totalLoss = 0;

  for (let s = 0; s < inputs.length; s++) {
    const output = forward(net, inputs[s]);
    predictions.push(output);

    // MSE loss
    let loss = 0;
    for (let j = 0; j < output.length; j++) {
      loss += (output[j] - targets[s][j]) ** 2;
    }
    totalLoss += loss / output.length;

    // dLoss/dOutput for MSE: 2*(pred - target)/n
    const dOutput = output.map((o, j) => (2 * (o - targets[s][j])) / output.length);

    // Backward pass — compute gradients and apply
    let dCurrent = [...dOutput];

    for (let i = net.layers.length - 1; i >= 0; i--) {
      const layer = net.layers[i];
      const cache = net.caches[i];
      const { weights, activation } = layer;
      const { input, z } = cache;

      // dZ = dOutput * activation'(z)
      const dZ = dCurrent.map((d, j) => d * activation.prime(z[j]));

      const dInput = new Array(input.length).fill(0);

      for (let oi = 0; oi < weights.length; oi++) {
        for (let ii = 0; ii < input.length; ii++) {
          // Accumulate dInput BEFORE updating weights
          dInput[ii] += weights[oi][ii] * dZ[oi];
          // Then update weight
          layer.weights[oi][ii] -= lr * dZ[oi] * input[ii];
        }
        layer.biases[oi] -= lr * dZ[oi];
      }

      dCurrent = dInput;
    }
  }

  return {
    loss: totalLoss / inputs.length,
    predictions,
  };
}
