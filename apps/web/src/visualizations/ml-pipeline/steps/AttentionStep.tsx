import { useState } from "react";
import { EquationBlock } from "@ml-visual-lab/viz-kit";
import { categoryColor } from "../embeddingSpace";
import type { WordEmbedding } from "../MLPipelineViz";
import type { AttentionBreakdown } from "../pipelineCore";
import "./steps.css";

type Phase = "scores" | "scale" | "softmax" | "mix";

const PHASES: { key: Phase; label: string }[] = [
  { key: "scores", label: "1 · Scores" },
  { key: "scale", label: "2 · Scale" },
  { key: "softmax", label: "3 · Softmax" },
  { key: "mix", label: "4 · Mix" },
];

const INPUT_COLOR = "#7c3aed";
const OUTPUT_COLOR = "#22c55e";

const SIZE_W = 380;
const SIZE_H = 300;
const PADDING = 28;
const X_MIN = -6, X_MAX = 7, Y_MIN = -4, Y_MAX = 4;

function toScreen(x: number, y: number): [number, number] {
  return [
    PADDING + ((x - X_MIN) / (X_MAX - X_MIN)) * (SIZE_W - PADDING * 2),
    PADDING + ((Y_MAX - y) / (Y_MAX - Y_MIN)) * (SIZE_H - PADDING * 2),
  ];
}

function diamondPath(cx: number, cy: number, r: number): string {
  return `M ${cx} ${cy - r} L ${cx + r} ${cy} L ${cx} ${cy + r} L ${cx - r} ${cy} Z`;
}

interface AttentionStepProps {
  tokens: string[];
  embeddings: WordEmbedding[];
  breakdown: AttentionBreakdown;
  temperature: number;
  onTemperatureChange: (t: number) => void;
}

export default function AttentionStep({ tokens, embeddings, breakdown, temperature, onTemperatureChange }: AttentionStepProps) {
  const [selectedToken, setSelectedToken] = useState<number>(() => Math.max(0, tokens.length - 1));
  const [phase, setPhase] = useState<Phase>("softmax");

  if (tokens.length === 0 || embeddings.length === 0) {
    return <div className="pl-step-content"><p className="pl-info">No data. Go back to previous steps.</p></div>;
  }

  const { scores, scaled, weights, output, dK, scale } = breakdown;
  const n = tokens.length;
  const i = selectedToken;
  const maxW = Math.max(...weights.flat());
  const outX = output[i][0];
  const outY = output[i][1];

  const colActive = (col: Phase) => (phase === col ? "pl-at-cell--active" : "");
  const sel = { x: embeddings[i].x, y: embeddings[i].y };
  const [selSX, selSY] = toScreen(sel.x, sel.y);

  const phaseBody = () => {
    switch (phase) {
      case "scores":
        return (
          <>
            <p className="pl-info" style={{ marginBottom: "0.5rem" }}>
              Each word plays both roles at once: it is the <strong>Query</strong> (the question it asks)
              and the <strong>Key</strong> (the label it holds up). Here the projections are{" "}
              <strong>identity</strong>, so qₙ and kₙ are simply the word's own embedding from the Embed step —
              every number stays traceable.
            </p>
            <p className="pl-info">
              <code>score(i, j) = qᵢ · kⱼ</code> is a dot product: add <code>xᵢxⱼ + yᵢyⱼ</code>. Pointing the same
              way → large and positive; pointing apart →{" "}
              <span style={{ color: "#ef4444" }}>negative</span>. The diagonal is a word vs. itself = |embedding|².
            </p>
          </>
        );
      case "scale":
        return (
          <>
            <p className="pl-info" style={{ marginBottom: "0.5rem" }}>
              Dot products grow with the number of dimensions, and huge scores make softmax behave like
              all-or-nothing. Dividing by <code>√d_k</code> keeps the numbers in a friendly range.
            </p>
            <p className="pl-info">
              The pipeline's embeddings have <strong>d_k = {dK}</strong> dimensions, so every score is divided by{" "}
              <code>√{dK} ≈ {scale.toFixed(3)}</code>. Big/small scores shrink toward zero.
            </p>
          </>
        );
      case "softmax":
        return (
          <>
            <p className="pl-info" style={{ marginBottom: "0.5rem" }}>
              exp(...) + normalize: each row becomes <strong>percentages that always add up to 100%</strong>.
              Low temperature (<code>T &lt; 1</code>) → one winner dominates; high temperature → attention spread over everyone.
            </p>
            <p className="pl-info">
              Drag the <strong>Temperature</strong> slider and watch the <code>weight</code> column in the table
              change — this is exactly the knob real models tune.
            </p>
          </>
        );
      case "mix":
        return (
          <>
            <p className="pl-info" style={{ marginBottom: "0.5rem" }}>
              A word's new meaning is a <strong>weighted blend of everyone's Value</strong> — its dot physically
              slides toward the words it paid attention to (green diamonds on the canvas).
            </p>
            <p className="pl-info">
              The <strong>last token's output is the context</strong> the model hands to the Predict step.
              Bonus: a function word near the origin (like &quot;the&quot;) has tiny queries, attends almost
              uniformly, and its output is roughly just the sentence average.
            </p>
          </>
        );
    }
  };

  const phaseTex = () => {
    switch (phase) {
      case "scores": return "\\text{score}(i,j) = q_i \\cdot k_j = x_i x_j + y_i y_j";
      case "scale": return `\\text{scaled}(i,j) = \\frac{q_i \\cdot k_j}{\\sqrt{d_k}}, \\quad d_k = ${dK}`;
      case "softmax": return `w_{ij} = \\frac{e^{\\text{scaled}_{ij} / T}}{\\sum_{\\ell} e^{\\text{scaled}_{i\\ell} / T}}`;
      case "mix": return "\\text{out}_i = \\sum_j w_{ij}\\, v_j";
    }
  };

  return (
    <div className="pl-step-content">
      <div className="pl-step-left">
        <div className="pl-readout">
          <h3>Self-Attention — the core of transformers</h3>
          <p className="pl-info">
            Attention lets each word <strong>look at every other word</strong> and decide how much each one matters.
            It is computed in four steps — <strong>Scores → Scale → Softmax → Mix</strong> (tabs on the right).
            Pick any token below (or click its dot) and the table below the canvas walks its whole row through the math.
          </p>
          <div className="pl-attention-token-grid" style={{ marginTop: "0.5rem" }}>
            {tokens.map((token, t) => (
              <button key={t}
                className={`pl-attention-token-btn ${t === selectedToken ? "pl-attention-token-btn--active" : ""}`}
                onClick={() => setSelectedToken(t)}>
                <span className="pl-attention-token-idx">{t}</span>
                {token}
              </button>
            ))}
          </div>
        </div>

        <div className="pl-canvas-wrap">
          <svg width={SIZE_W} height={SIZE_H} className="pl-canvas">
            {/* Grid */}
            {Array.from({ length: 14 }, (_, g) => {
              const x = -6 + g;
              const [sx] = toScreen(x, 0);
              return <line key={`v${g}`} x1={sx} y1={PADDING} x2={sx} y2={SIZE_H - PADDING} stroke="var(--border)" strokeWidth={0.5} opacity={0.3} />;
            })}
            {Array.from({ length: 9 }, (_, g) => {
              const y = -4 + g;
              const [, sy] = toScreen(0, y);
              return <line key={`h${g}`} x1={PADDING} y1={sy} x2={SIZE_W - PADDING} y2={sy} stroke="var(--border)" strokeWidth={0.5} opacity={0.3} />;
            })}
            {(() => {
              const [ox, oy] = toScreen(0, 0);
              return (
                <>
                  <line x1={PADDING} y1={oy} x2={SIZE_W - PADDING} y2={oy} stroke="var(--border)" strokeWidth={0.8} />
                  <line x1={ox} y1={PADDING} x2={ox} y2={SIZE_H - PADDING} stroke="var(--border)" strokeWidth={0.8} />
                </>
              );
            })()}

            {/* Attention connections: the selected token's weights */}
            {tokens.map((_, j) => {
              if (j === selectedToken) return null;
              const w = weights[selectedToken][j];
              if (w < 0.005) return null;
              const [sx2, sy2] = toScreen(embeddings[j].x, embeddings[j].y);
              return (
                <line key={`c${j}`}
                  x1={selSX} y1={selSY} x2={sx2} y2={sy2}
                  stroke={OUTPUT_COLOR}
                  strokeWidth={Math.max(1, (w / maxW) * 5)}
                  opacity={0.2 + w * 0.8} />
              );
            })}

            {/* Output diamonds (Mix phase) */}
            {phase === "mix" && output.map((row, j) => {
              const [ox, oy] = toScreen(row[0], row[1]);
              const [tx, ty] = toScreen(embeddings[j].x, embeddings[j].y);
              return (
                <g key={`o${j}`}>
                  {j === selectedToken && (
                    <line x1={tx} y1={ty} x2={ox} y2={oy} stroke={OUTPUT_COLOR} strokeWidth={1.2} strokeDasharray="4 3" />
                  )}
                  <path d={diamondPath(ox, oy, 4)} fill={OUTPUT_COLOR} opacity={0.9} />
                  <text x={ox} y={oy - 8} textAnchor="middle" fontSize={8}
                    fontFamily="var(--mono)" fontWeight={700} fill={OUTPUT_COLOR}>
                    {j === selectedToken ? "out" : "o"}
                  </text>
                </g>
              );
            })}

            {/* Tokens */}
            {embeddings.map((e, j) => {
              const [sx, sy] = toScreen(e.x, e.y);
              const isSelected = j === selectedToken;
              const color = categoryColor(e.word);
              return (
                <g key={j} onClick={() => setSelectedToken(j)} style={{ cursor: "pointer" }}>
                  <circle cx={sx} cy={sy} r={isSelected ? 13 : 9}
                    fill={isSelected ? INPUT_COLOR : color}
                    stroke={isSelected ? "var(--text-h)" : color}
                    strokeWidth={isSelected ? 2 : 1}
                    opacity={0.95} />
                  <text x={sx} y={sy - 12} textAnchor="middle" fontSize={isSelected ? 10 : 9}
                    fontWeight={700} fontFamily="var(--mono)" fill={color}>
                    {e.word}
                  </text>
                </g>
              );
            })}
          </svg>
          <div className="pl-at-legend">
            <span><span className="pl-at-legend-dot" style={{ background: INPUT_COLOR }} /> word embedding</span>
            <span><span className="pl-at-legend-diamond" /> attention output</span>
            <span className="pl-at-legend-phase">{PHASES.find((p) => p.key === phase)?.label} · &quot;{tokens[selectedToken]}&quot; attends to others</span>
          </div>
        </div>

        <div className="pl-readout">
          <h3>Row walkthrough — how &quot;{tokens[selectedToken]}&quot; is transformed</h3>
          <div className="pl-at-table">
            <div className="pl-at-row pl-at-row--head">
              <span className="pl-at-cell pl-at-cell--token">token</span>
              <span className={`pl-at-cell ${colActive("scores")}`}>score <em>q·k</em></span>
              <span className={`pl-at-cell ${colActive("scale")}`}>÷√d<sub>k</sub></span>
              <span className={`pl-at-cell ${colActive("softmax")}`}>weight</span>
              <span className={`pl-at-cell ${colActive("mix")}`}>value × w</span>
            </div>
            {tokens.map((token, j) => {
              const isSelf = j === selectedToken;
              const score = scores[i][j];
              const sc = scaled[i][j];
              const w = weights[i][j];
              const vx = embeddings[j].x;
              const vy = embeddings[j].y;
              return (
                <div key={j} className={`pl-at-row ${isSelf ? "pl-at-row--self" : ""}`}>
                  <span className="pl-at-cell pl-at-cell--token">{token}{isSelf ? " (self)" : ""}</span>
                  <span className={`pl-at-cell ${colActive("scores")} ${score < 0 ? "pl-at-neg" : ""}`}>
                    {score.toFixed(2)}
                  </span>
                  <span className={`pl-at-cell ${colActive("scale")}`}>{sc.toFixed(2)}</span>
                  <span className={`pl-at-cell ${colActive("softmax")}`}>{(w * 100).toFixed(1)}%</span>
                  <span className={`pl-at-cell ${colActive("mix")}`}>[{ (w * vx).toFixed(2)}, { (w * vy).toFixed(2) }]</span>
                </div>
              );
            })}
            <div className="pl-at-row pl-at-row--foot">
              <span className="pl-at-cell pl-at-cell--token">output</span>
              <span className="pl-at-cell pl-at-foot-mid">+ all of the above</span>
              <span className="pl-at-cell pl-at-cell--out">[{outX.toFixed(2)}, {outY.toFixed(2)}]</span>
            </div>
          </div>
          <p className="pl-info" style={{ marginTop: "0.5rem" }}>
            So &quot;{tokens[selectedToken]}&quot; becomes the blended point{" "}
            <strong>[{outX.toFixed(2)}, {outY.toFixed(2)}]</strong> — pulled toward the words it attended to.
            {selectedToken === n - 1 && (
              <> This is the <strong style={{ color: OUTPUT_COLOR }}>attention context</strong> the Predict step uses.</>
            )}
          </p>
        </div>
      </div>

      <div className="pl-step-right">
        <div className="pl-readout">
          <h3>Tap into the four steps</h3>
          <div className="pl-phase-tabs">
            {PHASES.map((p) => (
              <button key={p.key}
                className={`pl-phase-btn ${phase === p.key ? "pl-phase-btn--active" : ""}`}
                onClick={() => setPhase(p.key)}>
                {p.label}
              </button>
            ))}
          </div>
        </div>

        <div className="pl-readout">
          <h3>Now showing: {PHASES.find((p) => p.key === phase)?.label}</h3>
          {phaseBody()}
          <EquationBlock tex={phaseTex()} />
        </div>

        <div className="pl-readout">
          <h3>Controls</h3>
          <div className="pl-control-row">
            <span className="pl-control-label">Temperature</span>
            <input type="range" min={0.1} max={3} step={0.1} value={temperature}
              onChange={(e) => onTemperatureChange(parseFloat(e.target.value))}
              className="pl-slider" />
            <span className="pl-control-value">{temperature.toFixed(1)}</span>
          </div>
          <p className="pl-info" style={{ marginTop: "0.4rem" }}>
            Temperature only affects the <strong>Softmax</strong> step. Lower → almost all weight on one word; higher → attention spread evenly.
          </p>
        </div>

        <div className="pl-readout">
          <h3>The full formula</h3>
          <EquationBlock tex={`\\text{Attention}(Q, K, V) = \\text{softmax}\\left(\\frac{QK^T}{\\sqrt{d_k}}\\right) V`} />
          <ul className="pl-formula-legend">
            <li><strong>Q / K / V</strong> — Query, Key, Value. In this toy Q = K = V = the embeddings (identity projections).</li>
            <li><strong>QK<sup>T</sup></strong> — dot every query against every key → the scores.</li>
            <li><strong>/√d<sub>k</sub></strong> — the Scale step, keeping scores softmax-friendly.</li>
            <li><strong>softmax</strong> — each row → percentages that sum to 100%.</li>
            <li><strong>· V</strong> — the Mix step: blend the values by those percentages.</li>
          </ul>
          <p className="pl-info" style={{ marginTop: "0.5rem" }}>
            Real transformers replace the identity with <strong>learned projection matrices</strong> W_q, W_k, W_v that
            reshape the embedding into q/k/v. See the <strong>Attention Explorer</strong> in the sidebar for the full machinery.
          </p>
        </div>
      </div>
    </div>
  );
}