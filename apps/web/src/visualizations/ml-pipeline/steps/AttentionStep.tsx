import { useState } from "react";
import "./steps.css";

interface AttentionStepProps {
  tokens: string[];
  weights: number[][];
}

const HEAD_COLORS = ["#7c3aed", "#2563eb", "#059669", "#d97706", "#dc2626", "#ec4899", "#06b6d4", "#f97316"];

export default function AttentionStep({ tokens, weights }: AttentionStepProps) {
  const [selectedToken, setSelectedToken] = useState<number | null>(0);

  if (tokens.length === 0 || weights.length === 0) {
    return <div className="pl-step-content"><p className="pl-info">No data. Go back to previous steps.</p></div>;
  }

  const n = tokens.length;
  const maxWeight = Math.max(...weights.flat());

  return (
    <div className="pl-step-content">
      <div className="pl-step-left">
        <div className="pl-readout">
          <h3>Self-Attention</h3>
          <p className="pl-info">
            Click any token to see how much attention it pays to every other token.
            <strong> Attention weights</strong> determine which words influence each other's representation.
            This is the core mechanism behind transformers (GPT, BERT, etc.).
          </p>
        </div>
        <div className="pl-canvas-wrap">
          <svg width={380} height={300} className="pl-canvas">
            {/* Token nodes */}
            {tokens.map((token, i) => {
              const x = 40 + (i / (n - 1)) * 300;
              const y = 40;
              const isSelected = i === selectedToken;
              return (
                <g key={i}>
                  <circle cx={x} cy={y} r={isSelected ? 16 : 12} fill={isSelected ? "var(--accent)" : "var(--surface)"}
                    stroke={isSelected ? "var(--accent)" : "var(--border)"} strokeWidth={2} />
                  <text x={x} y={y + 4} textAnchor="middle" fontSize={isSelected ? 9 : 8}
                    fontFamily="var(--mono)" fontWeight={600}
                    fill={isSelected ? "#fff" : "var(--text-h)"}>
                    {token}
                  </text>
                  {/* Attention lines to other tokens */}
                  {selectedToken !== null && selectedToken !== i && (
                    <>
                      <line x1={40 + (selectedToken / (n - 1)) * 300} y1={40} x2={x} y2={y}
                        stroke={HEAD_COLORS[i % HEAD_COLORS.length]}
                        strokeWidth={Math.max(1, weights[selectedToken][i] / maxWeight * 6)}
                        opacity={0.7} />
                      <text x={x} y={y + 30} textAnchor="middle" fontSize={8}
                        fontFamily="var(--mono)" fontWeight={600}
                        fill={HEAD_COLORS[i % HEAD_COLORS.length]}>
                        {(weights[selectedToken][i] * 100).toFixed(0)}%
                      </text>
                    </>
                  )}
                </g>
              );
            })}
            {/* Second row: attention distribution bar chart */}
            {selectedToken !== null && (
              <g>
                <text x={190} y={100} textAnchor="middle" fontSize={9} fontFamily="var(--mono)"
                  fontWeight={600} fill="var(--text)">
                  Attention from "{tokens[selectedToken]}"
                </text>
                {tokens.map((token, i) => {
                  const x = 20 + (i / n) * 340;
                  const barH = (weights[selectedToken][i] / maxWeight) * 140;
                  return (
                    <g key={i}>
                      <rect x={x} y={250 - barH} width={340 / n - 4} height={barH} rx={3}
                        fill={HEAD_COLORS[i % HEAD_COLORS.length]} opacity={0.8} />
                      <text x={x + (340 / n - 4) / 2} y={258} textAnchor="middle" fontSize={7}
                        fontFamily="var(--mono)" fontWeight={600} fill="var(--text)">
                        {token}
                      </text>
                      <text x={x + (340 / n - 4) / 2} y={250 - barH - 4} textAnchor="middle" fontSize={7}
                        fontFamily="var(--mono)" fill="var(--text-h)">
                        {(weights[selectedToken][i] * 100).toFixed(0)}%
                      </text>
                    </g>
                  );
                })}
              </g>
            )}
          </svg>
        </div>
      </div>
      <div className="pl-step-right">
        <div className="pl-readout">
          <h3>Select a Token</h3>
          <div className="pl-attention-token-grid">
            {tokens.map((token, i) => (
              <button key={i}
                className={`pl-attention-token-btn ${i === selectedToken ? "pl-attention-token-btn--active" : ""}`}
                onClick={() => setSelectedToken(i)}>
                <span className="pl-attention-token-idx">{i}</span>
                {token}
              </button>
            ))}
          </div>
        </div>
        <div className="pl-readout">
          <h3>Attention Weights</h3>
          {selectedToken !== null && (
            <div className="pl-attention-details">
              <p className="pl-info">How <strong>"{tokens[selectedToken]}"</strong> attends to others:</p>
              <div className="pl-attention-bar-list">
                {tokens.map((token, j) => {
                  const w = weights[selectedToken][j];
                  return (
                    <div key={j} className="pl-attention-bar-row">
                      <span className="pl-attention-bar-label" style={{
                        color: j === selectedToken ? "var(--accent)" : "var(--text)",
                        fontWeight: j === selectedToken ? 700 : 400,
                      }}>
                        {j === selectedToken ? `${token} (self)` : token}
                      </span>
                      <div className="pl-attention-bar-track">
                        <div className="pl-attention-bar-fill" style={{
                          width: `${(w / maxWeight) * 100}%`,
                          background: HEAD_COLORS[j % HEAD_COLORS.length],
                        }} />
                      </div>
                      <span className="pl-attention-bar-val" style={{ color: HEAD_COLORS[j % HEAD_COLORS.length] }}>
                        {(w * 100).toFixed(1)}%
                      </span>
                    </div>
                  );
                })}
              </div>
            </div>
          )}
        </div>
        <div className="pl-readout">
          <h3>How attention works</h3>
          <p className="pl-info">
            <code>Attention(Q,K,V) = softmax(QK<sup>T</sup> / √d) · V</code>
            <br /><br />
            Each token creates a <strong>Query</strong> (what I'm looking for), <strong>Key</strong> (what I contain),
            and <strong>Value</strong> (what I give). Attention scores are how well queries match keys.
            Softmax converts scores to weights that sum to 1.
          </p>
        </div>
      </div>
    </div>
  );
}
