import { useState } from "react";
import { NavLink } from "react-router-dom";
import { visualizationRegistry, CATEGORY_LABELS, CATEGORY_ORDER, type VizCategory } from "../registry";

const CATEGORY_ICONS: Record<VizCategory, string> = {
  "pipeline": "PL",
  "linear-algebra": "LA",
  "optimization": "OPT",
  "embeddings": "EMB",
  "neural-networks": "NN",
};

export default function Sidebar() {
  const [collapsed, setCollapsed] = useState<Record<string, boolean>>({});

  const grouped = CATEGORY_ORDER.map((cat) => ({
    category: cat,
    label: CATEGORY_LABELS[cat],
    icon: CATEGORY_ICONS[cat],
    items: visualizationRegistry.filter((v) => v.category === cat),
  })).filter((g) => g.items.length > 0);

  const toggle = (cat: string) =>
    setCollapsed((prev) => ({ ...prev, [cat]: !prev[cat] }));

  return (
    <aside className="sidebar">
      <nav className="sidebar-nav">
        {grouped.map((group) => {
          const isCollapsed = collapsed[group.category] ?? false;
          return (
            <div key={group.category} className="sidebar-group">
              <button
                className="sidebar-group-header"
                onClick={() => toggle(group.category)}
                aria-expanded={!isCollapsed}
              >
                <span className="sidebar-group-icon">{group.icon}</span>
                <span className="sidebar-group-label">{group.label}</span>
                <svg
                  className={`sidebar-chevron ${isCollapsed ? "sidebar-chevron--collapsed" : ""}`}
                  width="14"
                  height="14"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2.5"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                >
                  <polyline points="6 9 12 15 18 9" />
                </svg>
              </button>
              {!isCollapsed && (
                <div className="sidebar-group-items">
                  {group.items.map((viz) => (
                    <NavLink
                      key={viz.slug}
                      to={`/${viz.slug}`}
                      className={({ isActive }) =>
                        `sidebar-link ${isActive ? "sidebar-link--active" : ""}`
                      }
                    >
                      <span className="sidebar-link-text">{viz.title}</span>
                    </NavLink>
                  ))}
                </div>
              )}
            </div>
          );
        })}
      </nav>
      <div className="sidebar-footer">
        <span className="sidebar-version">v0.1.0</span>
      </div>
    </aside>
  );
}
