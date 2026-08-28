import { Suspense } from "react";
import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import { visualizationRegistry } from "./registry";
import Navbar from "./components/Navbar";
import Sidebar from "./components/Sidebar";
import "./App.css";

import DotProductViz from "./visualizations/dot-product/DotProductViz";
import CosineSimilarityViz from "./visualizations/cosine-similarity/CosineSimilarityViz";
import GradientDescentViz from "./visualizations/gradient-descent/GradientDescentViz";
import EmbeddingPlaygroundViz from "./visualizations/embedding-playground/EmbeddingPlaygroundViz";
import NeuralNetworkViz from "./visualizations/neural-network/NeuralNetworkViz";
import AttentionExplorerViz from "./visualizations/attention-explorer/AttentionExplorerViz";
import BackpropFlowViz from "./visualizations/backprop-flow/BackpropFlowViz";
import OptimizerBenchmarkViz from "./visualizations/optimizer-benchmark/OptimizerBenchmarkViz";
import DecisionBoundaryLabViz from "./visualizations/decision-boundary-lab/DecisionBoundaryLabViz";
import MLPipelineViz from "./visualizations/ml-pipeline/MLPipelineViz";

const vizComponents: Record<string, React.ComponentType> = {
  "ml-pipeline": MLPipelineViz,
  "dot-product": DotProductViz,
  "cosine-similarity": CosineSimilarityViz,
  "gradient-descent": GradientDescentViz,
  "embedding-playground": EmbeddingPlaygroundViz,
  "neural-network": NeuralNetworkViz,
  "attention-explorer": AttentionExplorerViz,
  "backprop-flow": BackpropFlowViz,
  "optimizer-benchmark": OptimizerBenchmarkViz,
  "decision-boundary-lab": DecisionBoundaryLabViz,
};

const vizRoutes = visualizationRegistry.map((viz) => ({
  path: `/${viz.slug}`,
  slug: viz.slug,
  title: viz.title,
  description: viz.description,
}));

function VizRoute({ slug, title, description }: { slug: string; title: string; description: string }) {
  const Component = vizComponents[slug];
  return (
    <div className="viz-page">
      <div className="viz-header">
        <h1>{title}</h1>
        <p className="lede">{description}</p>
      </div>
      <Suspense fallback={<div className="viz-loading">Loading visualization…</div>}>
        <Component />
      </Suspense>
    </div>
  );
}

function AppLayout() {
  return (
    <div className="app-shell">
      <Navbar />
      <div className="app-body">
        <Sidebar />
        <main className="main-content">
          <Routes>
            {vizRoutes.map((route) => (
              <Route
                key={route.path}
                path={route.path}
                element={
                  <VizRoute
                    slug={route.slug}
                    title={route.title}
                    description={route.description}
                  />
                }
              />
            ))}
            <Route path="/" element={<Navigate to={`/${visualizationRegistry[0]?.slug ?? "dot-product"}`} replace />} />
            <Route path="*" element={<Navigate to={`/${visualizationRegistry[0]?.slug ?? "dot-product"}`} replace />} />
          </Routes>
        </main>
      </div>
    </div>
  );
}

export default function App() {
  return (
    <BrowserRouter>
      <AppLayout />
    </BrowserRouter>
  );
}
