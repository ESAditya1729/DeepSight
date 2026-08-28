import { useState } from "react";
import { dot, magnitude, normalize, type Vector } from "@ml-visual-lab/ml-core";
import { EquationBlock } from "@ml-visual-lab/viz-kit";
import "./CosineSimilarityViz.css";

const SIZE = 380;
const ORIGIN = SIZE / 2;
const RADIUS = 140;
const SCALE = RADIUS;

const COLOR_A = "#7c3aed";
const COLOR_B = "#2563eb";

const PRESETS: { label: string; a: Vector; b: Vector; icon: string }[] = [
  { label: "Same", a: [1, 0.2], b: [0.9, 0.4], icon: "0°" },
  { label: "Similar", a: [1, 0], b: [0.7, 0.7], icon: "45°" },
  { label: "Orthogonal", a: [1, 0], b: [0, 1], icon: "90°" },
  { label: "Opposite", a: [1, 0], b: [-1, 0], icon: "180°" },
];

function toScreen([x, y]: Vector): [number, number] {
  return [ORIGIN + x * SCALE, ORIGIN - y * SCALE];
}

function toWorld(px: number, py: number): Vector {
  return [(px - ORIGIN) / SCALE, (ORIGIN - py) / SCALE];
}

function clamp(v: number, lo: number, hi: number) { return Math.max(lo, Math.min(hi, v)); }

function simColor(s: number): string {
  if (s > 0.7) return "#22c55e";
  if (s > 0.3) return "#eab308";
  if (s > -0.3) return "#f97316";
  return "#ef4444";
}

function describeArc(cx: number, cy: number, r: number, startAngle: number, endAngle: number): string {
  let diff = endAngle - startAngle;
  if (diff > Math.PI) diff -= 2 * Math.PI;
  if (diff < -Math.PI) diff += 2 * Math.PI;
  const sweep = diff > 0;
  const end = startAngle + diff;
  const x1 = cx + r * Math.cos(startAngle);
  const y1 = cy - r * Math.sin(startAngle);
  const x2 = cx + r * Math.cos(end);
  const y2 = cy - r * Math.sin(end);
  return `M ${x1} ${y1} A ${r} ${r} 0 0 ${sweep ? 0 : 1} ${x2} ${y2}`;
}

function VectorArrow({ vector, color, markerId, label, onDrag, raw }: {
  vector: Vector; color: string; markerId: string; label: string;
  onDrag: (v: Vector) => void; raw?: Vector;
}) {
  const [sx, sy] = toScreen(vector);
  return (
    <g>
      <line x1={ORIGIN} y1={ORIGIN} x2={sx} y2={sy} stroke={color} strokeWidth={2.5} markerEnd={`url(#${markerId})`} />
      <text x={sx + 12} y={sy - 10} fill={color} fontSize={14} fontWeight={700} fontFamily="var(--mono)">
        {label}{raw ? ` (${raw[0].toFixed(1)}, ${raw[1].toFixed(1)})` : ""}
      </text>
      <circle
        cx={sx} cy={sy} r={9} fill={color} stroke="#fff" strokeWidth={2}
        style={{ cursor: "grab", filter: "drop-shadow(0 1px 2px rgba(0,0,0,0.15))" }}
        onPointerDown={(e) => e.currentTarget.setPointerCapture(e.pointerId)}
        onPointerMove={(e) => {
          const svg = e.currentTarget.ownerSVGElement;
          if (!svg) return;
          const rect = svg.getBoundingClientRect();
          onDrag(toWorld(e.clientX - rect.left, e.clientY - rect.top));
        }}
      />
    </g>
  );
}

export default function CosineSimilarityViz() {
  const [a, setA] = useState<Vector>(() => normalize([3, 2]));
  const [b, setB] = useState<Vector>(() => normalize([1, 3]));
  const [showProjection, setShowProjection] = useState(true);
  const [showRawVectors, setShowRawVectors] = useState(false);

  const [rawA, setRawA] = useState<Vector>([3, 2]);
  const [rawB, setRawB] = useState<Vector>([1, 3]);

  const sim = dot(a, b);
  const magA = magnitude(a);
  const magB = magnitude(b);
  const theta = magA > 0 && magB > 0
    ? (Math.acos(clamp(sim / (magA * magB), -1, 1)) * 180) / Math.PI
    : 0;

  const projLen = magB > 0 ? dot(a, b) / (magB * magB) : 0;
  const proj: Vector = [projLen * b[0], projLen * b[1]];
  const [ppx, ppy] = toScreen(proj);
  const [asx, asy] = toScreen(a);

  const angleA = Math.atan2(-a[1], a[0]);
  const angleB = Math.atan2(-b[1], b[0]);
  const arcR = 30;

  const fmt = (n: number) => n.toFixed(3);

  const handlePreset = (p: typeof PRESETS[number]) => {
    const na = normalize(p.a);
    const nb = normalize(p.b);
    setA(na);
    setB(nb);
    setRawA(p.a);
    setRawB(p.b);
  };

  const handleDragA = (v: Vector) => {
    const n = normalize(v);
    setA(n);
    setRawA(v);
  };

  const handleDragB = (v: Vector) => {
    const n = normalize(v);
    setB(n);
    setRawB(v);
  };

  return (
    <div className="cs-viz">
      <div className="cs-canvas-wrap">
        <svg width={SIZE} height={SIZE} className="cs-canvas">
          <defs>
            <marker id="cs-arrow-a" markerWidth="8" markerHeight="8" refX="6" refY="4" orient="auto">
              <path d="M0,0 L8,4 L0,8 Z" fill={COLOR_A} />
            </marker>
            <marker id="cs-arrow-b" markerWidth="8" markerHeight="8" refX="6" refY="4" orient="auto">
              <path d="M0,0 L8,4 L0,8 Z" fill={COLOR_B} />
            </marker>
            <marker id="cs-arrow-proj" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
              <path d="M0,0 L6,3 L0,6 Z" fill="#94a3b8" />
            </marker>
          </defs>

          {/* Unit circle */}
          <circle cx={ORIGIN} cy={ORIGIN} r={RADIUS} fill="none" stroke="var(--border)" strokeWidth={1} strokeDasharray="4 3" />
          {/* Axes */}
          <line x1={0} y1={ORIGIN} x2={SIZE} y2={ORIGIN} stroke="var(--border)" strokeWidth={0.8} />
          <line x1={ORIGIN} y1={0} x2={ORIGIN} y2={SIZE} stroke="var(--border)" strokeWidth={0.8} />

          {/* Angle arc */}
          {theta > 1 && (
            <path d={describeArc(ORIGIN, ORIGIN, arcR, angleA, angleB)}
              fill="none" stroke="#facc15" strokeWidth={2} strokeOpacity={0.7} />
          )}
          {theta > 5 && (
            <text
              x={ORIGIN + Math.cos((angleA + angleB) / 2) * (arcR + 14)}
              y={ORIGIN - Math.sin((angleA + angleB) / 2) * (arcR + 14)}
              fill="#facc15" fontSize={11} fontWeight={600} textAnchor="middle" dominantBaseline="middle"
            >
              {fmt(theta)}°
            </text>
          )}

          {/* Projection */}
          {showProjection && (
            <g>
              <line x1={ORIGIN} y1={ORIGIN} x2={ppx} y2={ppy}
                stroke="#94a3b8" strokeWidth={2} strokeDasharray="5 4" markerEnd="url(#cs-arrow-proj)" />
              <line x1={asx} y1={asy} x2={ppx} y2={ppy}
                stroke="#94a3b8" strokeWidth={1} strokeDasharray="3 3" opacity={0.5} />
            </g>
          )}

          {/* Vectors */}
          <VectorArrow vector={a} color={COLOR_A} markerId="cs-arrow-a" label="a" onDrag={handleDragA} raw={rawA} />
          <VectorArrow vector={b} color={COLOR_B} markerId="cs-arrow-b" label="b" onDrag={handleDragB} raw={rawB} />
        </svg>
      </div>

      <div className="cs-sidebar">
        {/* Similarity bar */}
        <div className="cs-readout">
          <h3>Cosine Similarity</h3>
          <div className="cs-sim-big" style={{ color: simColor(sim) }}>{fmt(sim)}</div>
          <div className="cs-similarity-bar">
            <div className="cs-similarity-bar-label">
              <span>-1 (opposite)</span>
              <span>+1 (same)</span>
            </div>
            <div className="cs-similarity-bar-track">
              <div className="cs-similarity-bar-center" />
              <div className="cs-similarity-bar-fill" style={{
                width: `${Math.abs(sim) * 50}%`,
                marginLeft: sim >= 0 ? "50%" : `${50 - Math.abs(sim) * 50}%`,
                background: simColor(sim),
              }} />
            </div>
          </div>
        </div>

        {/* Live equation */}
        <div className="cs-readout">
          <h3>Computation</h3>
          <EquationBlock tex={`\\cos\\theta = \\frac{\\vec{a} \\cdot \\vec{b}}{|a||b|} = \\frac{${fmt(dot(a, b))}}{${fmt(magA)} \\cdot ${fmt(magB)}} = \\mathbf{${fmt(sim)}}`} />
        </div>

        {/* Properties */}
        <div className="cs-readout">
          <h3>Properties</h3>
          <div className="cs-stats">
            <div className="cs-stat">
              <span className="cs-stat-label">|a|</span>
              <span className="cs-stat-value" style={{ color: COLOR_A }}>{fmt(magA)}</span>
            </div>
            <div className="cs-stat">
              <span className="cs-stat-label">|b|</span>
              <span className="cs-stat-value" style={{ color: COLOR_B }}>{fmt(magB)}</span>
            </div>
            <div className="cs-stat">
              <span className="cs-stat-label">Angle</span>
              <span className="cs-stat-value">{fmt(theta)}°</span>
            </div>
            <div className="cs-stat">
              <span className="cs-stat-label">cos(θ)</span>
              <span className="cs-stat-value" style={{ color: simColor(sim) }}>{fmt(sim)}</span>
            </div>
          </div>
        </div>

        {/* Presets */}
        <div className="cs-readout">
          <h3>Presets</h3>
          <div className="cs-presets">
            {PRESETS.map((p) => (
              <button key={p.label} className="cs-preset-btn" onClick={() => handlePreset(p)}>
                <span className="cs-preset-icon">{p.icon}</span>
                <span className="cs-preset-label">{p.label}</span>
              </button>
            ))}
          </div>
        </div>

        {/* Toggles */}
        <div className="cs-readout">
          <h3>Display</h3>
          <div className="cs-toggles">
            <label className="cs-toggle">
              <input type="checkbox" checked={showProjection} onChange={(e) => setShowProjection(e.target.checked)} />
              Show projection of a onto b
            </label>
            <label className="cs-toggle">
              <input type="checkbox" checked={showRawVectors} onChange={(e) => setShowRawVectors(e.target.checked)} />
              Show raw vector components
            </label>
          </div>
        </div>

        {/* Insight */}
        <div className="cs-readout">
          <h3>Key Insight</h3>
          <p style={{ fontSize: "0.85rem", lineHeight: 1.5, margin: 0 }}>
            Cosine similarity measures <strong>directional alignment</strong>, independent of magnitude.
            Drag the handles to change direction — notice the similarity doesn't change when you scale a vector.
            <strong> cos(θ) = 1</strong> means identical direction, <strong>0</strong> means perpendicular, <strong>-1</strong> means opposite.
          </p>
        </div>
      </div>
    </div>
  );
}
