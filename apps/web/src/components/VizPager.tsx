import { Link } from "react-router-dom";
import { visualizationRegistry } from "../registry";
import "./VizPager.css";

export default function VizPager({ slug }: { slug: string }) {
  const idx = visualizationRegistry.findIndex((v) => v.slug === slug);
  if (idx === -1) return null;

  const prev = visualizationRegistry[idx - 1];
  const next = visualizationRegistry[idx + 1];

  return (
    <>
      {next && (
        <Link to={`/${next.slug}`} className="viz-nextup">
          <div className="viz-nextup-text">
            <span className="viz-nextup-kicker">Next up</span>
            <span className="viz-nextup-title">{next.title}</span>
            <span className="viz-nextup-desc">{next.description}</span>
          </div>
          <span className="viz-nextup-arrow">→</span>
        </Link>
      )}
      <nav className="viz-pager">
        {prev ? (
          <Link to={`/${prev.slug}`} className="viz-pager-link viz-pager-link--prev">
            <span className="viz-pager-dir">← Previous</span>
            <span className="viz-pager-title">{prev.title}</span>
          </Link>
        ) : (
          <span className="viz-pager-spacer" />
        )}
        {next ? (
          <Link to={`/${next.slug}`} className="viz-pager-link viz-pager-link--next">
            <span className="viz-pager-dir">Next →</span>
            <span className="viz-pager-title">{next.title}</span>
          </Link>
        ) : (
          <span className="viz-pager-spacer" />
        )}
      </nav>
    </>
  );
}