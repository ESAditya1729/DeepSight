import { describe, expect, it } from "vitest";
import { createLayer, forwardLayer } from "./layers.js";

describe("createLayer", () => {
  it("creates a layer with correct dimensions", () => {
    const layer = createLayer(3, 2, "sigmoid");
    expect(layer.inputSize).toBe(3);
    expect(layer.outputSize).toBe(2);
    expect(layer.weights.length).toBe(2);
    expect(layer.weights[0].length).toBe(3);
    expect(layer.biases.length).toBe(2);
  });
});

describe("forwardLayer", () => {
  it("produces output with correct length", () => {
    const layer = createLayer(2, 3, "relu");
    const { output } = forwardLayer(layer, [1, 2]);
    expect(output.length).toBe(3);
  });

  it("output values are numbers", () => {
    const layer = createLayer(2, 3, "sigmoid");
    const { output } = forwardLayer(layer, [0.5, -0.5]);
    for (const v of output) {
      expect(typeof v).toBe("number");
      expect(isNaN(v)).toBe(false);
    }
  });

  it("sigmoid output is between 0 and 1", () => {
    const layer = createLayer(2, 3, "sigmoid");
    const { output } = forwardLayer(layer, [1, 2]);
    for (const v of output) {
      expect(v).toBeGreaterThan(0);
      expect(v).toBeLessThan(1);
    }
  });
});
