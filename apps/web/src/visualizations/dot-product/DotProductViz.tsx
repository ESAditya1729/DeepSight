import { dot, magnitude, type Vector } from "@ml-visual-lab/ml-core";
import { EquationBlock } from "@ml-visual-lab/viz-kit";
import { useState } from "react";
import "./DotProductViz.css";

const SIZE = 380;
const ORIGIN = SIZE / 2;
const SCALE = 26;

const COLOR_A = "#7c3aed";
const COLOR_B = "#2563eb";

const PRESETS: { label: string; a: Vector; b: Vector; desc: string }[] = [
  { label: "Perpendicular", a: [0, 3], b: [3, 0], desc: "a · b = 0" },
  { label: "Parallel", a: [3, 2], b: [1.5, 1], desc: "Maximum positive" },
  { label: "Opposite", a: [2, 1], b: [-2, -1], desc: "Maximum negative" },
  { label: "Acute", a: [3, 1], b: [2, 2.5], desc: "Positive dot product" },
  { label: "Obtuse", a: [3, 0.5], b: [-1, 2.5], desc: "Negative dot product" },
];

function toScreen([x, y]: Vector): [number, number] {
  return [ORIGIN + x * SCALE, ORIGIN - y * SCALE];
}

function toWorld(px: number, py: number): Vector {
  return [(px - ORIGIN) / SCALE, (ORIGIN - py) / SCALE];
}

function snapToGrid(v: Vector, on: boolean): Vector {
  if (!on) return v;
  return [Math.round(v[0] * 2) / 2, Math.round(v[1] * 2) / 2];
}

function angleDeg(a: Vector, b: Vector): number {
  const d = dot(a, b);
  const ma = magnitude(a);
  const mb = magnitude(b);
  if (ma === 0 || mb === 0) return 0;
  return (Math.acos(Math.min(1, Math.max(-1, d / (ma * mb)))) * 180) / Math.PI;
}

function dotResultColor(v: number): string {
  if (v > 0) return "#22c55e";
  if (v < 0) return "#ef4444";
  return "#94a3b8";
}

function VectorArrow({ vector, color, markerId, label, onDrag }: {
  vector: Vector; color: string; markerId: string; label: string; onDrag: (v: Vector) => void;
}) {
  const [sx, sy] = toScreen(vector);
  return (
    <g>
      <line x1={ORIGIN} y1={ORIGIN} x2={sx} y2={sy} stroke={color} strokeWidth={2.5} markerEnd={`url(#${markerId})`} />
      <text x={sx + 12} y={sy - 10} fill={color} fontSize={14} fontWeight={700} fontFamily="var(--mono)">
        {label}
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

export default function DotProductViz() {
  const [a, setA] = useState<Vector>([4, 2]);
  const [b, setB] = useState<Vector>([1, 3]);
  const [showProjection, setShowProjection] = useState(true);
  const [gridSnap, setGridSnap] = useState(false);

  const dp = dot(a, b);
  const magA = magnitude(a);
  const magB = magnitude(b);
  const theta = angleDeg(a, b);

  const projLen = magB > 0 ? dp / (magB * magB) : 0;
  const proj: Vector = [projLen * b[0], projLen * b[1]];
  const [ppx, ppy] = toScreen(proj);
  const [asx, asy] = toScreen(a);

  const angleA = Math.atan2(-a[1], a[0]);
  const angleB = Math.atan2(-b[1], b[0]);
  const arcR = 30;

  const fmt = (n: number) => n.toFixed(2);

  const handlePreset = (preset: typeof PRESETS[number]) => {
    setA(preset.a);
    setB(preset.b);
  };

  return (
    <div className="dp-viz">
      <div className="dp-canvas-wrap">
        <svg width={SIZE} height={SIZE} className="dp-canvas">
          <defs>
            <marker id="dp-arr-a" markerWidth="8" markerHeight="8" refX="6" refY="4" orient="auto">
              <path d="M0,0 L8,4 L0,8 Z" fill={COLOR_A} />
            </marker>
            <marker id="dp-arr-b" markerWidth="8" markerHeight="8" refX="6" refY="4" orient="auto">
              <path d="M0,0 L8,4 L0,8 Z" fill={COLOR_B} />
            </marker>
            <marker id="dp-arr-proj" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
              <path d="M0,0 L6,3 L0,6 Z" fill="#94a3b8" />
            </marker>
          </defs>

          {/* Grid */}
          {Array.from({ length: Math.floor(SIZE / SCALE) + 1 }, (_, i) => {
            const pos = i * SCALE;
            return (
              <g key={i}>
                <line x1={pos} y1={0} x2={pos} y2={SIZE} stroke="var(--border)" strokeWidth={0.5} opacity={0.4} />
                <line x1={0} y1={pos} x2={SIZE} y2={pos} stroke="var(--border)" strokeWidth={0.5} opacity={0.4} />
              </g>
            );
          })}

          {/* Axes */}
          <line x1={0} y1={ORIGIN} x2={SIZE} y2={ORIGIN} stroke="var(--border)" strokeWidth={1} />
          <line x1={ORIGIN} y1={0} x2={ORIGIN} y2={SIZE} stroke="var(--border)" strokeWidth={1} />

          {/* Angle arc */}
          {theta > 1 && (
            <path
              d={describeArc(ORIGIN, ORIGIN, arcR, angleA, angleB)}
              fill="none" stroke="#facc15" strokeWidth={2} strokeOpacity={0.7}
            />
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
          {showProjection && magA > 0 && magB > 0 && (
            <g>
              <line x1={ORIGIN} y1={ORIGIN} x2={ppx} y2={ppy}
                stroke="#94a3b8" strokeWidth={2} strokeDasharray="5 4" markerEnd="url(#dp-arr-proj)" />
              <line x1={asx} y1={asy} x2={ppx} y2={ppy}
                stroke="#94a3b8" strokeWidth={1} strokeDasharray="3 3" opacity={0.5} />
              <text x={(ppx + asx) / 2 + 8} y={(ppy + asy) / 2 - 6}
                fill="#94a3b8" fontSize={10} fontFamily="var(--mono)">
                proj
              </text>
            </g>
          )}

          {/* Vectors */}
          <VectorArrow vector={a} color={COLOR_A} markerId="dp-arr-a" label="a"
            onDrag={(v) => setA(snapToGrid(v, gridSnap))} />
          <VectorArrow vector={b} color={COLOR_B} markerId="dp-arr-b" label="b"
            onDrag={(v) => setB(snapToGrid(v, gridSnap))} />
        </svg>
      </div>

      <div className="dp-sidebar">
        {/* Dot product result */}
        <div className="dp-result-card" style={{ borderColor: dotResultColor(dp) + "44" }}>
          <div className="dp-result-label">Dot Product</div>
          <div className="dp-result-value" style={{ color: dotResultColor(dp) }}>{fmt(dp)}</div>
          <div className="dp-result-bar">
            <div className="dp-result-bar-center" />
            <div
              className="dp-result-bar-fill"
              style={{
                width: `${Math.min(50, Math.abs(dp) / (magA * magB + 0.01) * 50)}%`,
                marginLeft: dp >= 0 ? "50%" : undefined,
                marginRight: dp < 0 ? "50%" : undefined,
                background: dotResultColor(dp),
              }}
            />
          </div>
          <div className="dp-result-bar-labels">
            <span>-1</span><span>0</span><span>+1</span>
          </div>
        </div>

        {/* Live equation */}
        <div className="dp-readout">
          <h3>Computation</h3>
          <EquationBlock
            tex={`\\vec{a} \\cdot \\vec{b} = (${fmt(a[0])})(${fmt(b[0])}) + (${fmt(a[1])})(${fmt(b[1])}) = \\mathbf{${fmt(dp)}}`}
          />
        </div>

        {/* Properties */}
        <div className="dp-readout">
          <h3>Properties</h3>
          <div className="dp-stats">
            <div className="dp-stat">
              <span className="dp-stat-label">|a|</span>
              <span className="dp-stat-value" style={{ color: COLOR_A }}>{fmt(magA)}</span>
            </div>
            <div className="dp-stat">
              <span className="dp-stat-label">|b|</span>
              <span className="dp-stat-value" style={{ color: COLOR_B }}>{fmt(magB)}</span>
            </div>
            <div className="dp-stat">
              <span className="dp-stat-label">Angle</span>
              <span className="dp-stat-value">{fmt(theta)}°</span>
            </div>
            <div className="dp-stat">
              <span className="dp-stat-label">|a||b|cos θ</span>
              <span className="dp-stat-value" style={{ color: dotResultColor(dp) }}>{fmt(magA * magB * Math.cos(theta * Math.PI / 180))}</span>
            </div>
          </div>
        </div>

        {/* Controls */}
        <div className="dp-readout">
          <h3>Controls</h3>
          <div className="dp-controls">
            <label className="dp-toggle">
              <input type="checkbox" checked={showProjection} onChange={(e) => setShowProjection(e.target.checked)} />
              Show projection of a onto b
            </label>
            <label className="dp-toggle">
              <input type="checkbox" checked={gridSnap} onChange={(e) => setGridSnap(e.target.checked)} />
              Snap to grid
            </label>
          </div>
        </div>

        {/* Presets */}
        <div className="dp-readout">
          <h3>Presets</h3>
          <div className="dp-presets">
            {PRESETS.map((p) => (
              <button key={p.label} className="dp-preset-btn" onClick={() => handlePreset(p)}>
                <span className="dp-preset-label">{p.label}</span>
                <span className="dp-preset-desc">{p.desc}</span>
              </button>
            ))}
          </div>
        </div>

        {/* Formula */}
        <div className="dp-readout">
          <h3>Formula</h3>
          <EquationBlock tex={`\\vec{a} \\cdot \\vec{b} = a_x b_x + a_y b_y = |a||b|\\cos\\theta`} />
        </div>
      </div>
    </div>
  );
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
