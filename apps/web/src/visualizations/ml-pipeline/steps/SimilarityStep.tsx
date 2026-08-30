import { type WordEmbedding } from "../MLPipelineViz";
import "./steps.css";

const CELL_SIZE = 28;

function simColor(val: number): string {
  if (val > 0.7) return "#22c55e";
  if (val > 0.4) return "#eab308";
  if (val > 0.2) return "#f97316";
  return "#ef4444";
}

interface SimilarityStepProps {
  embeddings: WordEmbedding[];
  matrix: number[][];
}

export default function SimilarityStep({ embeddings, matrix }: SimilarityStepProps) {
  if (matrix.length === 0) {
    return <div className="pl-step-content"><p className="pl-info">No data. Go back to previous steps.</p></div>;
  }

  const n = matrix.length;
  const gridW = n * CELL_SIZE;
  const startX = 180;
  const startY = 30;

  return (
    <div className="pl-step-content">
      <div className="pl-step-left">
        <div className="pl-readout">
          <h3>Similarity</h3>
          <p className="pl-info">
            Every token pair is compared by <strong>distance in the embedding space</strong>.
            Close words (similar meaning) get a high score, far words get a low score.
            The matrix shows how the model "sees" relationships between all words simultaneously.
          </p>
          <div className="pl-sim-legend">
            <span className="pl-sim-legend-item"><span className="pl-sim-dot" style={{ background: "#ef4444" }} />Far (0)</span>
            <span className="pl-sim-legend-item"><span className="pl-sim-dot" style={{ background: "#f97316" }} />Low</span>
            <span className="pl-sim-legend-item"><span className="pl-sim-dot" style={{ background: "#eab308" }} />Medium</span>
            <span className="pl-sim-legend-item"><span className="pl-sim-dot" style={{ background: "#22c55e" }} />Close (1)</span>
          </div>
        </div>
        <div className="pl-canvas-wrap">
          <svg width={Math.max(420, startX + gridW + 20)} height={startY + gridW + 20} className="pl-canvas">
            {/* Column headers */}
            {embeddings.map((e, j) => (
              <text key={`ch${j}`}
                x={startX + j * CELL_SIZE + CELL_SIZE / 2}
                y={startY - 6}
                textAnchor="end"
                fontSize={9} fontFamily="var(--mono)" fontWeight={600}
                fill="var(--text)"
                transform={`rotate(-45, ${startX + j * CELL_SIZE + CELL_SIZE / 2}, ${startY - 6})`}>
                {e.word}
              </text>
            ))}
            {/* Row headers */}
            {embeddings.map((e, i) => (
              <text key={`rh${i}`}
                x={startX - 6}
                y={startY + i * CELL_SIZE + CELL_SIZE / 2 + 3}
                textAnchor="end"
                fontSize={9} fontFamily="var(--mono)" fontWeight={600}
                fill="var(--text)">
                {e.word}
              </text>
            ))}
            {/* Cells */}
            {matrix.map((row, i) =>
              row.map((val, j) => (
                <g key={`${i}-${j}`}>
                  <rect
                    x={startX + j * CELL_SIZE}
                    y={startY + i * CELL_SIZE}
                    width={CELL_SIZE - 1}
                    height={CELL_SIZE - 1}
                    rx={3}
                    fill={i === j ? "var(--accent)" : simColor(val)}
                    opacity={i === j ? 0.3 : Math.max(0.15, Math.abs(val))}
                  />
                  {CELL_SIZE >= 24 && (
                    <text
                      x={startX + j * CELL_SIZE + CELL_SIZE / 2}
                      y={startY + i * CELL_SIZE + CELL_SIZE / 2 + 3}
                      textAnchor="middle"
                      fontSize={7} fontFamily="var(--mono)"
                      fill={val > 0.5 ? "#fff" : "var(--text)"}
                      opacity={0.8}>
                      {val.toFixed(1)}
                    </text>
                  )}
                </g>
              )),
            )}
          </svg>
        </div>
      </div>
      <div className="pl-step-right">
        <div className="pl-readout">
          <h3>Top Similar Pairs</h3>
          <div className="pl-pairs-list">
            {(() => {
              const pairs: { i: number; j: number; sim: number }[] = [];
              for (let i = 0; i < n; i++) {
                for (let j = i + 1; j < n; j++) {
                  pairs.push({ i, j, sim: matrix[i][j] });
                }
              }
              pairs.sort((a, b) => b.sim - a.sim);
              return pairs.slice(0, 8).map((p, idx) => (
                <div key={idx} className="pl-pair-row">
                  <span className="pl-pair-words">
                    <span style={{ color: "var(--accent)" }}>{embeddings[p.i]?.word}</span>
                    {" ↔ "}
                    <span style={{ color: "var(--accent)" }}>{embeddings[p.j]?.word}</span>
                  </span>
                  <div className="pl-pair-bar">
                    <div className="pl-pair-bar-fill" style={{
                      width: `${Math.max(0, p.sim) * 100}%`,
                      background: simColor(p.sim),
                    }} />
                  </div>
                  <span className="pl-pair-sim" style={{ color: simColor(p.sim) }}>{p.sim.toFixed(3)}</span>
                </div>
              ));
            })()}
          </div>
        </div>
        <div className="pl-readout">
          <h3>How it works</h3>
          <p className="pl-info">
            <code>sim(a, b) = exp(−d(a, b)² / (2σ²)), σ = 2</code>
            <br /><br />
            A Gaussian "closeness" score on the map: identical points → <strong>1</strong>,
            far-apart points → <strong>~0</strong>. This is a warm-up intuition layer —
            it shows <em>which words sit near each other</em>.
          </p>
          <p className="pl-info" style={{ marginTop: "0.5rem" }}>
            Real attention does <strong>not</strong> reuse this matrix directly. It computes its own
            <strong> dot-product scores</strong> between queries and keys (the next step) — those can be{" "}
            <strong>negative</strong>, and softmax turns them into percentages.
          </p>
        </div>
      </div>
    </div>
  );
}
