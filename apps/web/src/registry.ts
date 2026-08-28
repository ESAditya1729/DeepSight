/**
 * The one file every new visualization touches from outside its own
 * folder. Adding a visualization means: build it under
 * src/visualizations/<name>/, then add one entry here.
 */
export type VizCategory =
  | "pipeline"
  | "linear-algebra"
  | "embeddings"
  | "optimization"
  | "neural-networks";

export interface VisualizationEntry {
  slug: string;
  title: string;
  description: string;
  category: VizCategory;
  isNew?: boolean;
}

export const CATEGORY_LABELS: Record<VizCategory, string> = {
  "pipeline": "ML Pipeline",
  "linear-algebra": "Linear Algebra",
  "embeddings": "Embeddings",
  "optimization": "Optimization",
  "neural-networks": "Neural Networks",
};

export const CATEGORY_ORDER: VizCategory[] = [
  "pipeline",
  "linear-algebra",
  "optimization",
  "embeddings",
  "neural-networks",
];

export const visualizationRegistry: VisualizationEntry[] = [
  {
    slug: "ml-pipeline",
    title: "ML Pipeline",
    description: "Follow the complete journey from raw text to prediction — tokenize, embed, compute similarity, apply attention, and classify. Step through each stage of an ML pipeline interactively.",
    category: "pipeline",
    isNew: true,
  },
  {
    slug: "dot-product",
    title: "Dot Product",
    description: "Drag two vectors and watch their dot product update in real time.",
    category: "linear-algebra",
  },
  {
    slug: "cosine-similarity",
    title: "Cosine Similarity",
    description: "Explore how cosine similarity measures directional similarity between vectors, independent of magnitude.",
    category: "linear-algebra",
  },
  {
    slug: "gradient-descent",
    title: "Gradient Descent",
    description: "Watch an optimizer find the minimum of a loss surface. Drag the starting point, adjust the learning rate, and observe convergence or divergence.",
    category: "optimization",
  },
  {
    slug: "embedding-playground",
    title: "Word Embeddings",
    description: "Explore a 2D word embedding space. Hover to see nearest neighbors, or try vector arithmetic like king - man + woman.",
    category: "embeddings",
  },
  {
    slug: "neural-network",
    title: "Neural Network",
    description: "Watch a small neural network learn XOR in real time. See weights, activations, and loss evolve as training progresses.",
    category: "neural-networks",
  },
  {
    slug: "attention-explorer",
    title: "Attention Explorer",
    description: "Visualize transformer self-attention — click tokens to see attention weights, adjust temperature, and explore multi-head patterns.",
    category: "neural-networks",
    isNew: true,
  },
  {
    slug: "backprop-flow",
    title: "Backprop Flow",
    description: "See gradients flow backward through a network. Compare healthy ReLU flow vs vanishing sigmoid and exploding gradients.",
    category: "neural-networks",
    isNew: true,
  },
  {
    slug: "optimizer-benchmark",
    title: "Optimizer Benchmark",
    description: "Race 6 optimizers on different loss surfaces — compare convergence, speed, and final loss side by side.",
    category: "optimization",
    isNew: true,
  },
  {
    slug: "decision-boundary-lab",
    title: "Decision Boundary Lab",
    description: "Train a neural network on 2D datasets and watch the decision boundary evolve in real time.",
    category: "neural-networks",
    isNew: true,
  },
];
