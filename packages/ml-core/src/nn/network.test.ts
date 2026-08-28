import { describe, expect, it } from "vitest";
import { createNetwork, forward, type Network } from "./network.js";
import { trainEpoch } from "./trainer.js";

function evaluate(net: Network, inputs: number[][], targets: number[][]): number {
  let totalLoss = 0;
  for (let i = 0; i < inputs.length; i++) {
    const output = forward(net, inputs[i]);
    for (let j = 0; j < output.length; j++) {
      totalLoss += (output[j] - targets[i][j]) ** 2;
    }
  }
  return totalLoss / inputs.length;
}

describe("createNetwork", () => {
  it("creates a network with correct number of layers", () => {
    const net = createNetwork([2, 3, 1], ["relu", "sigmoid"]);
    expect(net.layers.length).toBe(2);
  });

  it("throws with mismatched activations", () => {
    expect(() => createNetwork([2, 3, 1], ["relu"])).toThrow("activations.length");
  });
});

describe("forward", () => {
  it("returns output with correct length", () => {
    const net = createNetwork([2, 3, 1], ["relu", "sigmoid"]);
    const output = forward(net, [1, 2]);
    expect(output.length).toBe(1);
  });

  it("sigmoid output is between 0 and 1", () => {
    const net = createNetwork([2, 3, 1], ["relu", "sigmoid"]);
    const output = forward(net, [1, 2]);
    expect(output[0]).toBeGreaterThan(0);
    expect(output[0]).toBeLessThan(1);
  });
});

describe("trainEpoch", () => {
  it("reduces loss after training", () => {
    const net = createNetwork([2, 4, 1], ["sigmoid", "sigmoid"]);
    const inputs = [[0, 0], [0, 1], [1, 0], [1, 1]];
    const targets = [[0], [1], [1], [0]];

    const beforeLoss = evaluate(net, inputs, targets);

    for (let i = 0; i < 500; i++) {
      trainEpoch(net, inputs, targets, 2);
    }

    const afterLoss = evaluate(net, inputs, targets);
    expect(afterLoss).toBeLessThan(beforeLoss);
  });
});
