import { Link } from "react-router-dom";
import { visualizationRegistry, CATEGORY_LABELS, CATEGORY_ORDER } from "../registry";
import { useProgress } from "../lib/progress";
import "./Home.css";

const START_PATH = [
  "ml-pipeline",
  "dot-product",
  "cosine-similarity",
  "gradient-descent",
  "neural-network",
];

const HOW_IT_WORKS = [
  { label: "Drag", detail: "vectors, points, and tokens directly on the canvas" },
  { label: "Scrub", detail: "learning rates, temperatures, and seeds in real time" },
  { label: "Watch", detail: "optimizers race and networks learn, live in your browser" },
];

export default function Home() {
  const { isVisited } = useProgress();

  const groups = CATEGORY_ORDER.map((cat) => ({
    category: cat,
    label: CATEGORY_LABELS[cat],
    items: visualizationRegistry.filter((v) => v.category === cat),
  })).filter((g) => g.items.length > 0);

  const startItems = START_PATH
    .map((slug) => visualizationRegistry.find((v) => v.slug === slug))
    .filter((v): v is NonNullable<typeof v> => v !== undefined);

  return (
    <div className="home">
      <section className="home-hero">
        <h1>Learn machine learning by seeing it move.</h1>
        <p className="lede">
          DeepSight is an interactive visual laboratory. Instead of static diagrams,
          you drag vectors, scrub hyperparameters, and watch optimizers and networks
          evolve in real time — all running in your browser, no backend required.
        </p>
        <div className="home-hero-actions">
          {HOW_IT_WORKS.map((it) => (
            <div key={it.label} className="home-hero-action">
              <b>{it.label}</b>
              <span>{it.detail}</span>
            </div>
          ))}
        </div>
      </section>

      <section className="home-path">
        <h2 className="home-section-title">Where to start</h2>
        <p className="home-path-hint">
          A gentle learning path that builds from the core idea to a full network:
        </p>
        <div className="path-steps">
          {startItems.map((viz, i) => (
            <span key={viz.slug} className="path-step-wrapper">
              <Link to={`/${viz.slug}`} className="path-step">
                <span className="path-step-num">{i + 1}</span>
                {viz.title}
                {isVisited(viz.slug) && <span className="path-step-check">✓</span>}
              </Link>
              {i < startItems.length - 1 && <span className="path-arrow">→</span>}
            </span>
          ))}
        </div>
      </section>

      <section className="home-browse">
        <h2 className="home-section-title">Browse all visualizations</h2>
        {groups.map((group) => (
          <div key={group.category} className="home-cat-group">
            <h3 className="home-cat-heading">{group.label}</h3>
            <div className="viz-grid">
              {group.items.map((viz) => (
                <Link
                  key={viz.slug}
                  to={`/${viz.slug}`}
                  className={`viz-card ${isVisited(viz.slug) ? "viz-card--visited" : ""}`}
                >
                  {isVisited(viz.slug) && (
                    <span className="viz-card-check" title="Visited">✓</span>
                  )}
                  <span className="viz-card-cat">{group.label}</span>
                  <span className="viz-card-title">{viz.title}</span>
                  <p className="viz-card-desc">{viz.description}</p>
                  <span className="viz-card-go">{isVisited(viz.slug) ? "Visited · Open lab →" : "Open lab →"}</span>
                </Link>
              ))}
            </div>
          </div>
        ))}
      </section>
    </div>
  );
}