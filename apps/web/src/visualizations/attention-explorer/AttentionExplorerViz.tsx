import { useState, useMemo } from "react";
import { EquationBlock } from "@ml-visual-lab/viz-kit";
import { multiply, scaledDotProductAttention, type Matrix } from "@ml-visual-lab/ml-core";
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

// Fixed colors for Q/K/V — independent of head color, so "this is a Query"
// reads the same no matter which head is active.
const QKV_COLORS = { Q: "#3b82f6", K: "#d97706", V: "#059669" } as const;

const EMBED_DIM = 8;
const HEAD_DIM = 4;

function hashString(s: string): number {
  let h = 0;
  for (let i = 0; i < s.length; i++) {
    h = (h * 31 + s.charCodeAt(i)) >>> 0;
  }
  return h;
}

function seededRandom(seed: number): () => number {
  let s = seed % 2147483647;
  if (s <= 0) s += 2147483646;
  return () => {
    s = (s * 16807) % 2147483647;
    return s / 2147483647;
  };
}

// Toy per-token "embedding lookup": same word + same seed always yields the same
// base vector, then a sinusoidal positional encoding is added so repeated tokens
// (e.g. "the" at two different positions) still end up with distinct vectors.
function embedTokens(tokens: readonly string[], seed: number): Matrix {
  return tokens.map((token, pos) => {
    const tokenSeed = ((hashString(token.toLowerCase()) ^ seed) >>> 0) || 1;
    const rng = seededRandom(tokenSeed);
    return Array.from({ length: EMBED_DIM }, (_, dim) => {
      const base = (rng() - 0.5) * 2;
      const angle = pos / Math.pow(10000, (2 * Math.floor(dim / 2)) / EMBED_DIM);
      const posEncoding = dim % 2 === 0 ? Math.sin(angle) : Math.cos(angle);
      return base + posEncoding;
    });
  });
}

// Fixed (not learned) per-head projection matrix — independent per (seed, head, Q/K/V).
function generateProjection(seed: number, headIndex: number, tag: number): Matrix {
  const matSeed = seed * 1000 + headIndex * 17 + tag * 7 || 1;
  const rng = seededRandom(matSeed);
  return Array.from({ length: EMBED_DIM }, () =>
    Array.from({ length: HEAD_DIM }, () => (rng() - 0.5) * 0.8),
  );
}

interface HeadResult {
  Wq: Matrix;
  Wk: Matrix;
  Wv: Matrix;
  Q: Matrix;
  K: Matrix;
  V: Matrix;
  scores: Matrix;
  weights: Matrix;
}

function computeHeads(embeddings: Matrix, numHeads: number, seed: number, temperature: number): HeadResult[] {
  return Array.from({ length: numHeads }, (_, h) => {
    const Wq = generateProjection(seed, h, 1);
    const Wk = generateProjection(seed, h, 2);
    const Wv = generateProjection(seed, h, 3);
    const Q = multiply(embeddings, Wq);
    const K = multiply(embeddings, Wk);
    const V = multiply(embeddings, Wv);
    const { scores, weights } = scaledDotProductAttention(Q, K, V, temperature);
    return { Wq, Wk, Wv, Q, K, V, scores, weights };
  });
}

function WeightGrid({ matrix, color }: { matrix: Matrix; color: string }) {
  const maxAbs = Math.max(0.001, ...matrix.flatMap((row) => row.map((v) => Math.abs(v))));
  return (
    <div className="ae-weight-grid" style={{ gridTemplateColumns: `repeat(${matrix[0].length}, 1fr)` }}>
      {matrix.map((row, i) =>
        row.map((v, j) => (
          <div
            key={`${i}-${j}`}
            className="ae-weight-cell"
            style={{ background: color, opacity: Math.abs(v) / maxAbs }}
            title={v.toFixed(3)}
          />
        )),
      )}
    </div>
  );
}

export default function AttentionExplorerViz() {
  const [presetIdx, setPresetIdx] = useState(0);
  const [selectedToken, setSelectedToken] = useState(0);
  const [numHeads, setNumHeads] = useState(2);
  const [temperature, setTemperature] = useState(1.0);
  const [activeHead, setActiveHead] = useState(0);
  const [viewMode, setViewMode] = useState<"connections" | "heatmap">("connections");
  const [seed, setSeed] = useState(42);
  const [showWeights, setShowWeights] = useState(false);
  const [qkvPick, setQkvPick] = useState<"Q" | "K" | "V">("Q");
  const [qkvIndex, setQkvIndex] = useState(0);

  const preset = PRESETS[presetIdx];
  const tokens = preset.tokens;
  const numTokens = tokens.length;

  const embeddings = useMemo(() => embedTokens(tokens, seed), [tokens, seed]);

  const heads = useMemo(
    () => computeHeads(embeddings, numHeads, seed, temperature),
    [embeddings, numHeads, seed, temperature],
  );

  const active = heads[Math.min(activeHead, numHeads - 1)];
  const attentionRow = active?.weights[selectedToken] ?? [];
  const scoresRow = active?.scores[selectedToken] ?? [];
  const qVec = active?.Q[selectedToken] ?? [];
  const kVec = active?.K[selectedToken] ?? [];
  const vVec = active?.V[selectedToken] ?? [];

  // "How Q/K/V Are Built" calculator — recomputed live from the same source
  // data as qVec/kVec/vVec above, so its sum always matches them exactly.
  const embeddingVec = embeddings[selectedToken] ?? [];
  const qkvMatrix = active ? { Q: active.Wq, K: active.Wk, V: active.Wv }[qkvPick] : undefined;
  const qkvVector = { Q: qVec, K: kVec, V: vVec }[qkvPick];
  const weightColumn = qkvMatrix ? qkvMatrix.map((row) => row[qkvIndex] ?? 0) : [];
  const products = embeddingVec.map((v, i) => v * (weightColumn[i] ?? 0));
  const qkvSum = products.reduce((a, b) => a + b, 0);
  const qkvName = qkvPick === "Q" ? "Query" : qkvPick === "K" ? "Key" : "Value";
  const qkvMatrixName = qkvPick === "Q" ? "Wq" : qkvPick === "K" ? "Wk" : "Wv";

  const tokenWidth = Math.min(90, (560 - 40) / numTokens);
  const tokenStartX = (560 - numTokens * tokenWidth) / 2;
  const tokenY = 40;

  const fmt = (n: number) => n.toFixed(3);
  const activeColor = HEAD_COLORS[activeHead % HEAD_COLORS.length];

  return (
    <div className="ae-viz">
      <div className="ae-top">
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
                    fill={isSelected ? activeColor : "var(--code-bg)"}
                    stroke={isSelected ? activeColor : "var(--border)"}
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
                        stroke={activeColor}
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
                {heads.map(({ weights }, h) => {
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
                      {weights.map((row, i) =>
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
                  fontWeight={600} fill={activeColor}>
                  {fmt(weight)}
                </text>
              );
            })}
          </svg>
        </div>

        <div className="ae-controls-rail">
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
        </div>
      </div>

      <div className="ae-explain">
        {/* The story, told once, up front */}
        <div className="ae-intro">
          <p>
            Every word's meaning starts as a list of numbers (its <strong>embedding</strong>). To pay attention to
            other words, each word turns that same list into three things: a{" "}
            <strong style={{ color: QKV_COLORS.Q }}>Query</strong> — a question it's asking, a{" "}
            <strong style={{ color: QKV_COLORS.K }}>Key</strong> — a label it holds up, and a{" "}
            <strong style={{ color: QKV_COLORS.V }}>Value</strong> — the actual info it hands over.
          </p>
          <p>
            <strong>Wq, Wk, Wv</strong> are just recipes (grids of numbers) that reshape the same embedding into each
            of these three things. One word's Query is compared to every word's Key to see how well they match —
            softmax turns those matches into percentages that add up to 100%, and the word ends up with a blend of
            everyone's Value, weighted by those percentages.
          </p>
        </div>

        <div className="ae-explain-grid">
          {/* How Q, K, V are built — interactive calculator */}
          <div className="ae-readout ae-explain-span">
            <h3>How Q, K, V Are Built — "{tokens[selectedToken]}" (Head {activeHead})</h3>
            <div className="ae-qkv-picker">
              {(["Q", "K", "V"] as const).map((k) => (
                <button
                  key={k}
                  className={`ae-qkv-pick-btn ${qkvPick === k ? "ae-qkv-pick-btn--active" : ""}`}
                  style={{
                    borderColor: QKV_COLORS[k],
                    background: qkvPick === k ? QKV_COLORS[k] : "transparent",
                    color: qkvPick === k ? "#fff" : QKV_COLORS[k],
                  }}
                  onClick={() => setQkvPick(k)}
                >
                  {k}
                </button>
              ))}
              <div className="ae-qkv-index-picker">
                {Array.from({ length: HEAD_DIM }, (_, i) => (
                  <button
                    key={i}
                    className={`ae-index-btn ${qkvIndex === i ? "ae-index-btn--active" : ""}`}
                    style={{
                      borderColor: QKV_COLORS[qkvPick],
                      background: qkvIndex === i ? QKV_COLORS[qkvPick] : "transparent",
                      color: qkvIndex === i ? "#fff" : QKV_COLORS[qkvPick],
                    }}
                    onClick={() => setQkvIndex(i)}
                  >
                    #{i + 1}
                  </button>
                ))}
              </div>
            </div>

            <div className="ae-calc">
              <div className="ae-calc-row">
                <span className="ae-calc-row-label">embedding</span>
                <div className="ae-calc-chips">
                  {embeddingVec.map((v, i) => (
                    <span key={i} className="ae-calc-chip">{fmt(v)}</span>
                  ))}
                </div>
              </div>
              <div className="ae-calc-row">
                <span className="ae-calc-row-label">
                  {qkvMatrixName} col {qkvIndex + 1}
                </span>
                <div className="ae-calc-chips">
                  {weightColumn.map((v, i) => (
                    <span key={i} className="ae-calc-chip" style={{ borderColor: QKV_COLORS[qkvPick] }}>{fmt(v)}</span>
                  ))}
                </div>
              </div>
              <div className="ae-calc-divider" />
              <div className="ae-calc-row">
                <span className="ae-calc-row-label">products</span>
                <div className="ae-calc-chips">
                  {products.map((v, i) => (
                    <span key={i} className="ae-calc-chip ae-calc-chip--product">{fmt(v)}</span>
                  ))}
                </div>
              </div>
              <div className="ae-calc-sum">
                add them all up&nbsp;→&nbsp;
                <strong style={{ color: QKV_COLORS[qkvPick] }}>{qkvPick}[{qkvIndex + 1}] = {fmt(qkvSum)}</strong>
              </div>
            </div>

            <p className="ae-info">
              This is the recipe in action: multiply the word's meaning piece-by-piece against one column of the{" "}
              {qkvMatrixName} recipe, then add everything up. Do that for all {HEAD_DIM} columns and you get the full{" "}
              {qkvName} vector: [{qkvVector.map(fmt).join(", ")}]
            </p>
          </div>

          {/* Projection weight matrices */}
          <div className="ae-readout">
            <div className="ae-weights-header">
              <h3>Projection Weights — Head {activeHead}</h3>
              <button className="ae-toggle-btn" onClick={() => setShowWeights((v) => !v)}>
                {showWeights ? "Hide" : "Show"}
              </button>
            </div>
            {showWeights && active && (
              <div className="ae-weight-panels">
                <div className="ae-weight-panel">
                  <span className="ae-weight-panel-label" style={{ color: QKV_COLORS.Q }}>Wq</span>
                  <WeightGrid matrix={active.Wq} color={QKV_COLORS.Q} />
                </div>
                <div className="ae-weight-panel">
                  <span className="ae-weight-panel-label" style={{ color: QKV_COLORS.K }}>Wk</span>
                  <WeightGrid matrix={active.Wk} color={QKV_COLORS.K} />
                </div>
                <div className="ae-weight-panel">
                  <span className="ae-weight-panel-label" style={{ color: QKV_COLORS.V }}>Wv</span>
                  <WeightGrid matrix={active.Wv} color={QKV_COLORS.V} />
                </div>
              </div>
            )}
            <p className="ae-info">
              Wq, Wk, Wv are fixed recipes — grids of numbers, one column per output number. Darker cells matter more when building that column.
            </p>
          </div>

          {/* Attention breakdown */}
          <div className="ae-readout">
            <h3>Attention Weights — "{tokens[selectedToken]}"</h3>
            <div className="ae-attn-bars">
              <div className="ae-attn-bar-row ae-attn-bar-row--header">
                <span className="ae-attn-bar-token" />
                <span className="ae-attn-bar-score">score</span>
                <div className="ae-attn-bar-track" />
                <span className="ae-attn-bar-weight">weight</span>
              </div>
              {tokens.map((token, i) => {
                const weight = attentionRow[i] ?? 0;
                const score = scoresRow[i] ?? 0;
                return (
                  <div key={i} className={`ae-attn-bar-row ${i === selectedToken ? "ae-attn-bar-row--self" : ""}`}>
                    <span className="ae-attn-bar-token">{token}</span>
                    <span className="ae-attn-bar-score">{fmt(score)}</span>
                    <div className="ae-attn-bar-track">
                      <div className="ae-attn-bar-fill" style={{
                        width: `${weight * 100}%`,
                        background: activeColor,
                      }} />
                    </div>
                    <span className="ae-attn-bar-weight">{fmt(weight)}</span>
                  </div>
                );
              })}
            </div>
            <p className="ae-info">
              Score = how well this word's Query (its question) matches that word's Key (its label) — bigger means a
              better match. Weight = that match turned into a percentage — every row of percentages always adds up
              to 100%.
            </p>
          </div>

          {/* Formula */}
          <div className="ae-readout ae-explain-span">
            <h3>Formula</h3>
            <EquationBlock tex={`\\text{Attention}(Q,K,V) = \\text{softmax}\\left(\\frac{QK^T}{\\sqrt{d_k}}\\right)V`} />
            <ul className="ae-formula-legend">
              <li><strong style={{ color: QKV_COLORS.Q }}>Q</strong> = Query — the question</li>
              <li><strong style={{ color: QKV_COLORS.K }}>K</strong> = Key — the label</li>
              <li><strong style={{ color: QKV_COLORS.V }}>V</strong> = Value — the actual info</li>
              <li><code>softmax</code> = turn scores into percentages that add up to 100%</li>
              <li><code>/√d_k</code> = a scaling trick so the numbers don't get too big before softmax</li>
            </ul>
            <p className="ae-info">
              Every number above is computed live from this sentence's token embeddings and this head's Wq/Wk/Wv
              matrices. Temperature makes the attention more focused (low) or more spread out (high) before it's
              turned into percentages.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
