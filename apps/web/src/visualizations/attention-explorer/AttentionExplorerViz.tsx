import { useState, useMemo } from "react";
import { EquationBlock } from "@ml-visual-lab/viz-kit";
import "./AttentionExplorerViz.css";

const PRESETS = [
  { label: "The cat sat on the mat", tokens: ["The", "cat", "sat", "on", "the", "mat"] },
  { label: "She went to the bank to deposit money", tokens: ["She", "went", "to", "the", "bank", "to", "deposit", "money"] },
  { label: "Attention is all you need", tokens: ["Attention", "is", "all", "you", "need"] },
  { label: "The quick brown fox jumps", tokens: ["The", "quick", "brown", "fox", "jumps"] },
];

const HEAD_COLORS = [
  "#7c3aed", "#2563eb", "#059669", "#d97706",
  "#dc2626", "#ec4899", "#06b6d4", "#8b5cf6",
];

function softmax(arr: number[], temperature: number): number[] {
  const scaled = arr.map((x) => x / temperature);
  const max = Math.max(...scaled);
  const exps = scaled.map((x) => Math.exp(x - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((x) => x / sum);
}

function seededRandom(seed: number): () => number {
  let s = seed;
  return () => {
    s = (s * 16807 + 0) % 2147483647;
    return s / 2147483647;
  };
}

function generateAttentionWeights(
  numTokens: number,
  numHeads: number,
  seed: number,
): number[][][] {
  const rng = seededRandom(seed);
  const heads: number[][][] = [];
  for (let h = 0; h < numHeads; h++) {
    const weights: number[][] = [];
    for (let i = 0; i < numTokens; i++) {
      const raw: number[] = [];
      for (let j = 0; j < numTokens; j++) {
        raw.push((rng() - 0.3) * 4);
      }
      weights.push(softmax(raw, 1.0));
    }
    heads.push(weights);
  }
  return heads;
}

export default function AttentionExplorerViz() {
  const [presetIdx, setPresetIdx] = useState(0);
  const [selectedToken, setSelectedToken] = useState(0);
  const [numHeads, setNumHeads] = useState(2);
  const [temperature, setTemperature] = useState(1.0);
  const [activeHead, setActiveHead] = useState(0);
  const [viewMode, setViewMode] = useState<"connections" | "heatmap">("connections");
  const [seed, setSeed] = useState(42);

  const preset = PRESETS[presetIdx];
  const tokens = preset.tokens;
  const numTokens = tokens.length;

  const rawWeights = useMemo(
    () => generateAttentionWeights(numTokens, numHeads, seed),
    [numTokens, numHeads, seed],
  );

  const scaledWeights = useMemo(() => {
    return rawWeights.map((head) =>
      head.map((row) => softmax(row.map((v) => v * (1 / temperature)), 1.0)),
    );
  }, [rawWeights, temperature]);

  const currentHead = scaledWeights[Math.min(activeHead, numHeads - 1)];
  const attentionRow = currentHead?.[selectedToken] ?? [];

  const tokenWidth = Math.min(90, (560 - 40) / numTokens);
  const tokenStartX = (560 - numTokens * tokenWidth) / 2;
  const tokenY = 40;

  const fmt = (n: number) => n.toFixed(3);

  return (
    <div className="ae-viz">
      <div className="ae-canvas-wrap">
        <svg width={560} height={320} className="ae-canvas">
          <defs>
            {HEAD_COLORS.map((color, i) => (
              <marker key={i} id={`ae-arr-${i}`} markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
                <path d="M0,0 L6,3 L0,6 Z" fill={color} />
              </marker>
            ))}
          </defs>

          {/* Tokens */}
          {tokens.map((token, i) => {
            const x = tokenStartX + i * tokenWidth + tokenWidth / 2;
            const isSelected = i === selectedToken;
            return (
              <g key={i} onClick={() => setSelectedToken(i)} style={{ cursor: "pointer" }}>
                <rect
                  x={x - tokenWidth / 2 + 4} y={tokenY - 14}
                  width={tokenWidth - 8} height={28} rx={6}
                  fill={isSelected ? HEAD_COLORS[activeHead % HEAD_COLORS.length] : "var(--code-bg)"}
                  stroke={isSelected ? HEAD_COLORS[activeHead % HEAD_COLORS.length] : "var(--border)"}
                  strokeWidth={isSelected ? 2 : 1}
                />
                <text
                  x={x} y={tokenY + 4} textAnchor="middle"
                  fontSize={numTokens > 6 ? 11 : 12} fontWeight={isSelected ? 700 : 500}
                  fontFamily="var(--mono)"
                  fill={isSelected ? "#fff" : "var(--text-h)"}
                >
                  {token}
                </text>
                {/* Attention connections */}
                {viewMode === "connections" && isSelected && attentionRow.map((weight, j) => {
                  if (j === i || weight < 0.01) return null;
                  const tx = tokenStartX + j * tokenWidth + tokenWidth / 2;
                  const opacity = Math.min(1, weight * 2);
                  const strokeW = Math.max(1, weight * 6);
                  return (
                    <line key={j}
                      x1={x} y1={tokenY + 14}
                      x2={tx} y2={tokenY + 14}
                      stroke={HEAD_COLORS[activeHead % HEAD_COLORS.length]}
                      strokeWidth={strokeW}
                      strokeOpacity={opacity}
                    />
                  );
                })}
              </g>
            );
          })}

          {/* Heatmap view */}
          {viewMode === "heatmap" && (
            <g>
              {scaledWeights.slice(0, numHeads).map((head, h) => {
                const cellSize = Math.min(20, (500 / numTokens));
                const startX = (560 - numTokens * cellSize) / 2;
                const startY = 80 + h * (numTokens * cellSize + 30);
                return (
                  <g key={h}>
                    <text x={startX - 5} y={startY + numTokens * cellSize / 2}
                      fontSize={10} fill="var(--text)" fontWeight={600}
                      fontFamily="var(--mono)" textAnchor="end"
                      dominantBaseline="middle">
                      H{h}
                    </text>
                    {head.map((row, i) =>
                      row.map((val, j) => (
                        <rect key={`${i}-${j}`}
                          x={startX + j * cellSize} y={startY + i * cellSize}
                          width={cellSize - 1} height={cellSize - 1} rx={2}
                          fill={HEAD_COLORS[h % HEAD_COLORS.length]}
                          opacity={val}
                        />
                      )),
                    )}
                  </g>
                );
              })}
            </g>
          )}

          {/* Attention weight labels (connections mode) */}
          {viewMode === "connections" && tokens.map((_, j) => {
            if (j === selectedToken) return null;
            const weight = attentionRow[j];
            if (weight < 0.01) return null;
            const x = tokenStartX + j * tokenWidth + tokenWidth / 2;
            return (
              <text key={j} x={x} y={tokenY + 34}
                textAnchor="middle" fontSize={9} fontFamily="var(--mono)"
                fontWeight={600} fill={HEAD_COLORS[activeHead % HEAD_COLORS.length]}>
                {fmt(weight)}
              </text>
            );
          })}
        </svg>
      </div>

      <div className="ae-sidebar">
        {/* Sentence selector */}
        <div className="ae-readout">
          <h3>Input Sentence</h3>
          <div className="ae-presets">
            {PRESETS.map((p, i) => (
              <button key={i}
                className={`ae-preset-btn ${i === presetIdx ? "ae-preset-btn--active" : ""}`}
                onClick={() => { setPresetIdx(i); setSelectedToken(0); }}>
                {p.label}
              </button>
            ))}
          </div>
        </div>

        {/* Controls */}
        <div className="ae-readout">
          <h3>Controls</h3>
          <div className="ae-controls">
            <div className="ae-control-row">
              <span className="ae-control-label">Heads</span>
              <div className="ae-head-btns">
                {Array.from({ length: 8 }, (_, i) => (
                  <button key={i}
                    className={`ae-head-btn ${i < numHeads ? "ae-head-btn--active" : ""}`}
                    style={{ background: i < numHeads ? HEAD_COLORS[i] : undefined }}
                    onClick={() => {
                      setNumHeads(i + 1);
                      if (activeHead >= i + 1) setActiveHead(i);
                    }}>
                    {i + 1}
                  </button>
                ))}
              </div>
            </div>
            {numHeads > 1 && (
              <div className="ae-control-row">
                <span className="ae-control-label">Active Head</span>
                <div className="ae-head-btns">
                  {Array.from({ length: numHeads }, (_, i) => (
                    <button key={i}
                      className={`ae-head-btn ae-head-btn--selected ${i === activeHead ? "ae-head-btn--ring" : ""}`}
                      style={{ background: HEAD_COLORS[i] }}
                      onClick={() => setActiveHead(i)}>
                      H{i}
                    </button>
                  ))}
                </div>
              </div>
            )}
            <div className="ae-control-row">
              <span className="ae-control-label">Temperature</span>
              <input type="range" min={0.1} max={3} step={0.1} value={temperature}
                onChange={(e) => setTemperature(parseFloat(e.target.value))}
                className="ae-slider" />
              <span className="ae-control-value">{temperature.toFixed(1)}</span>
            </div>
            <div className="ae-control-row">
              <span className="ae-control-label">View</span>
              <div className="ae-view-btns">
                <button className={`ae-view-btn ${viewMode === "connections" ? "ae-view-btn--active" : ""}`}
                  onClick={() => setViewMode("connections")}>Connections</button>
                <button className={`ae-view-btn ${viewMode === "heatmap" ? "ae-view-btn--active" : ""}`}
                  onClick={() => setViewMode("heatmap")}>Heatmap</button>
              </div>
            </div>
            <div className="ae-control-row">
              <span className="ae-control-label">Seed</span>
              <input type="range" min={1} max={100} step={1} value={seed}
                onChange={(e) => setSeed(parseInt(e.target.value))}
                className="ae-slider" />
              <span className="ae-control-value">{seed}</span>
            </div>
          </div>
        </div>

        {/* Attention breakdown */}
        <div className="ae-readout">
          <h3>Attention Weights — "{tokens[selectedToken]}"</h3>
          <div className="ae-attn-bars">
            {tokens.map((token, i) => {
              const weight = attentionRow[i] ?? 0;
              return (
                <div key={i} className={`ae-attn-bar-row ${i === selectedToken ? "ae-attn-bar-row--self" : ""}`}>
                  <span className="ae-attn-bar-token">{token}</span>
                  <div className="ae-attn-bar-track">
                    <div className="ae-attn-bar-fill" style={{
                      width: `${weight * 100}%`,
                      background: HEAD_COLORS[activeHead % HEAD_COLORS.length],
                    }} />
                  </div>
                  <span className="ae-attn-bar-weight">{fmt(weight)}</span>
                </div>
              );
            })}
          </div>
        </div>

        {/* Formula */}
        <div className="ae-readout">
          <h3>Formula</h3>
          <EquationBlock tex={`\\text{Attention}(Q,K,V) = \\text{softmax}\\left(\\frac{QK^T}{\\sqrt{d_k}}\\right)V`} />
          <p className="ae-info">
            Temperature controls the sharpness of attention. Lower = more focused, higher = more uniform.
            Each head learns different attention patterns.
          </p>
        </div>
      </div>
    </div>
  );
}
