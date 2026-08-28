export type ScalarFn = (...args: number[]) => number;

export function numericalGradient(fn: ScalarFn, ...args: number[]): number[] {
  const eps = 1e-7;
  return args.map((_, i) => {
    const params = [...args];
    params[i] += eps;
    const fPlus = fn(...params);
    params[i] = args[i] - eps;
    const fMinus = fn(...params);
    return (fPlus - fMinus) / (2 * eps);
  });
}

export function gradientDescentStep(
  fn: ScalarFn,
  params: number[],
  learningRate: number,
): number[] {
  const grad = numericalGradient(fn, ...params);
  return params.map((p, i) => p - learningRate * grad[i]);
}

export function partialDerivative(
  fn: ScalarFn,
  vars: number[],
  varIndex: number,
): number {
  const eps = 1e-7;
  const plus = [...vars];
  const minus = [...vars];
  plus[varIndex] += eps;
  minus[varIndex] -= eps;
  return (fn(...plus) - fn(...minus)) / (2 * eps);
}
