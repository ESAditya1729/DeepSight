# DeepSight

**An interactive visual laboratory for learning machine learning concepts through hands-on browser simulations.**

Instead of static diagrams and equations, DeepSight lets you drag vectors, scrub learning rates, and watch optimizers race across loss surfaces in real time — all running client-side, no backend required.

## Visualizations

| Category | Visualization | What it shows |
| --- | --- | --- |
| Pipeline | **ML Pipeline** | Tokenize → embed → similarity → attention → predict, stepped through interactively on raw text. |
| Linear Algebra | **Dot Product** | Drag two 2D vectors and watch their dot product update live. |
| Linear Algebra | **Cosine Similarity** | How direction-only similarity differs from magnitude-sensitive dot product. |
| Optimization | **Gradient Descent** | An optimizer descending a loss surface — set the learning rate and starting point, watch it converge or diverge. |
| Optimization | **Optimizer Benchmark** | SGD, Momentum, RMSProp, Adam, and Adagrad racing on the same loss surface side by side. |
| Embeddings | **Word Embeddings** | A 2D word-embedding space with nearest-neighbor lookup and vector arithmetic (`king - man + woman`). |
| Neural Networks | **Neural Network** | A small dense network learning XOR — weights, activations, and loss evolving during training. |
| Neural Networks | **Attention Explorer** | Transformer self-attention — click tokens to see attention weights, adjust temperature, inspect multiple heads. |
| Neural Networks | **Backprop Flow** | Gradients flowing backward through a network — healthy ReLU flow vs. vanishing/exploding gradients. |
| Neural Networks | **Decision Boundary Lab** | Train a network on 2D toy datasets and watch its decision boundary reshape in real time. |

## Tech stack

- **React 19 + TypeScript**, bundled with **Vite**, routed with **React Router**
- **KaTeX** for synchronous equation rendering (no layout "jump" as live numbers update)
- **Vitest** + **React Testing Library** for unit tests
- npm workspaces monorepo — no backend, everything runs in the browser

## Architecture

```
apps/web/            React app: routing, layout, and one folder per visualization
packages/ml-core/    Pure TypeScript math/ML engine — no React/DOM imports
packages/viz-kit/    Shared, framework-light UI primitives (Slider, PlaybackControls, EquationBlock, useAnimationFrame)
```

**`packages/ml-core`** is the computational core and has zero UI dependencies, so it's unit-testable in isolation and reusable outside the browser (e.g. a future dev-tool extension). It currently covers:
- Vector and matrix operations
- Numerical (finite-difference) calculus utilities, useful for gradient-checking analytic derivatives
- Loss functions (MSE, binary cross-entropy) and their gradients
- A small dense-layer neural network: forward pass, backward pass, and a trainer

**`packages/viz-kit`** holds the small set of UI primitives shared across every visualization — a slider, play/pause/step/reset controls, and a KaTeX equation block — kept deliberately framework-light so they don't accumulate per-visualization special cases.

**`apps/web/src/visualizations/`** holds one self-contained folder per visualization (component + simulation logic + scoped CSS). `apps/web/src/registry.ts` is the single file every new visualization needs to touch to appear in the sidebar and routing.

## Getting started

```bash
npm install
npm run dev    # starts the web app
npm run test   # runs all workspace test suites
npm run lint   # eslint across the repo
npm run build  # builds all workspaces
```

## License

MIT — see [LICENSE](LICENSE).
