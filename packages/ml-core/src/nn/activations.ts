export type ActivationFn = (x: number) => number;
export type ActivationDerivativeFn = (x: number) => number;

export interface Activation {
  fn: ActivationFn;
  prime: ActivationDerivativeFn;
  name: string;
}

export const sigmoid: Activation = {
  name: "sigmoid",
  fn: (x) => 1 / (1 + Math.exp(-x)),
  prime: (x) => {
    const s = 1 / (1 + Math.exp(-x));
    return s * (1 - s);
  },
};

export const relu: Activation = {
  name: "relu",
  fn: (x) => Math.max(0, x),
  prime: (x) => (x > 0 ? 1 : 0),
};

export const tanh_: Activation = {
  name: "tanh",
  fn: (x) => Math.tanh(x),
  prime: (x) => {
    const t = Math.tanh(x);
    return 1 - t * t;
  },
};

export const linear: Activation = {
  name: "linear",
  fn: (x) => x,
  prime: () => 1,
};

export function getActivation(name: string): Activation {
  switch (name) {
    case "sigmoid": return sigmoid;
    case "relu": return relu;
    case "tanh": return tanh_;
    case "linear": return linear;
    default: return sigmoid;
  }
}
