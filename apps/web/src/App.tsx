import { Suspense, lazy, useEffect, useState } from "react";
import { BrowserRouter, Routes, Route, Navigate, useLocation, useNavigate } from "react-router-dom";
import { visualizationRegistry } from "./registry";
import { useKeyDown } from "@ml-visual-lab/viz-kit";
import { useProgress } from "./lib/progress";
import Navbar from "./components/Navbar";
import Sidebar from "./components/Sidebar";
import VizPager from "./components/VizPager";
import Home from "./pages/Home";
import "./App.css";

const vizModules: Record<string, React.LazyExoticComponent<React.ComponentType>> = {
  "ml-pipeline": lazy(() => import("./visualizations/ml-pipeline/MLPipelineViz")),
  "dot-product": lazy(() => import("./visualizations/dot-product/DotProductViz")),
  "cosine-similarity": lazy(() => import("./visualizations/cosine-similarity/CosineSimilarityViz")),
  "gradient-descent": lazy(() => import("./visualizations/gradient-descent/GradientDescentViz")),
  "embedding-playground": lazy(() => import("./visualizations/embedding-playground/EmbeddingPlaygroundViz")),
  "contextual-embeddings": lazy(() => import("./visualizations/contextual-embeddings/ContextualViz")),
  "neural-network": lazy(() => import("./visualizations/neural-network/NeuralNetworkViz")),
  "attention-explorer": lazy(() => import("./visualizations/attention-explorer/AttentionExplorerViz")),
  "backprop-flow": lazy(() => import("./visualizations/backprop-flow/BackpropFlowViz")),
  "optimizer-benchmark": lazy(() => import("./visualizations/optimizer-benchmark/OptimizerBenchmarkViz")),
  "decision-boundary-lab": lazy(() => import("./visualizations/decision-boundary-lab/DecisionBoundaryLabViz")),
};

const vizRoutes = visualizationRegistry.map((viz) => ({
  path: `/${viz.slug}`,
  slug: viz.slug,
  title: viz.title,
  description: viz.description,
}));

function VizRoute({ slug, title, description }: { slug: string; title: string; description: string }) {
  const Component = vizModules[slug];
  return (
    <div className="viz-page">
      <div className="viz-header">
        <h1>{title}</h1>
        <p className="lede">{description}</p>
      </div>
      <Suspense fallback={<div className="viz-loading">Loading visualization…</div>}>
        <Component />
      </Suspense>
      <VizPager slug={slug} />
    </div>
  );
}

function AppLayout() {
  const location = useLocation();
  const navigate = useNavigate();
  const [sidebarOpen, setSidebarOpen] = useState(false);

  const pathSlug = location.pathname.slice(1);
  const activeSlug = visualizationRegistry.some((v) => v.slug === pathSlug) ? pathSlug : undefined;
  useProgress(activeSlug);

  useEffect(() => {
    setSidebarOpen(false);
  }, [location.pathname]);

  useKeyDown((event) => {
    const target = event.target as HTMLElement | null;
    if (!target) return;
    const tag = target.tagName;
    if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT" || target.isContentEditable) return;

    const idx = visualizationRegistry.findIndex((v) => v.slug === location.pathname.slice(1));
    if (event.key === "ArrowRight") {
      const next = visualizationRegistry[idx + 1];
      if (next) {
        event.preventDefault();
        navigate(`/${next.slug}`);
      }
    } else if (event.key === "ArrowLeft") {
      const prev = visualizationRegistry[idx - 1];
      if (prev) {
        event.preventDefault();
        navigate(`/${prev.slug}`);
      }
    } else if (event.key === " ") {
      event.preventDefault();
      window.dispatchEvent(new CustomEvent("deepsight:toggle-play"));
    }
  });

  return (
    <div className="app-shell">
      <Navbar onMenuClick={() => setSidebarOpen((o) => !o)} />
      <div className="app-body">
        <Sidebar open={sidebarOpen} onClose={() => setSidebarOpen(false)} />
        <main className="main-content">
          <Routes>
            <Route path="/" element={<Home />} />
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
            <Route path="*" element={<Navigate to="/" replace />} />
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