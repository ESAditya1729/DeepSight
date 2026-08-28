import { type WordEmbedding } from "../MLPipelineViz";
import { categoryColor } from "../embeddingSpace";
import "./steps.css";

const SIZE = 360;
const PADDING = 30;
const USABLE = SIZE - PADDING * 2;
const X_MIN = -6, X_MAX = 7, Y_MIN = -4, Y_MAX = 4;

function toScreen(x: number, y: number): [number, number] {
  return [
    PADDING + ((x - X_MIN) / (X_MAX - X_MIN)) * USABLE,
    PADDING + ((Y_MAX - y) / (Y_MAX - Y_MIN)) * USABLE,
  ];
}

interface EmbedStepProps {
  embeddings: WordEmbedding[];
  seed: number;
  onSeedChange: (seed: number) => void;
}

export default function EmbedStep({ embeddings, seed, onSeedChange }: EmbedStepProps) {
  if (embeddings.length === 0) {
    return <div className="pl-step-content"><p className="pl-info">No tokens to embed. Go back to Input step.</p></div>;
  }

  return (
    <div className="pl-step-content">
      <div className="pl-step-left">
        <div className="pl-readout">
          <h3>Word Embeddings</h3>
          <p className="pl-info">
            Each token is mapped to a <strong>2D vector</strong> (embedding).
            Words with similar meanings appear close together.
            This is how machines understand that "king" and "queen" are related.
          </p>
        </div>
        <div className="pl-canvas-wrap">
          <svg width={SIZE} height={SIZE} className="pl-canvas">
            {/* Grid */}
            {Array.from({ length: 13 }, (_, i) => {
              const x = -6 + i;
              const [sx] = toScreen(x, 0);
              return <line key={`v${i}`} x1={sx} y1={PADDING} x2={sx} y2={SIZE - PADDING} stroke="var(--border)" strokeWidth={0.5} opacity={0.3} />;
            })}
            {Array.from({ length: 9 }, (_, i) => {
              const y = -4 + i;
              const [, sy] = toScreen(0, y);
              return <line key={`h${i}`} x1={PADDING} y1={sy} x2={SIZE - PADDING} y2={sy} stroke="var(--border)" strokeWidth={0.5} opacity={0.3} />;
            })}
            {/* Axes */}
            {(() => {
              const [ox, oy] = toScreen(0, 0);
              return <>
                <line x1={PADDING} y1={oy} x2={SIZE - PADDING} y2={oy} stroke="var(--border)" strokeWidth={0.8} />
                <line x1={ox} y1={PADDING} x2={ox} y2={SIZE - PADDING} stroke="var(--border)" strokeWidth={0.8} />;
              </>;
            })()}
            {/* Words */}
            {embeddings.map((e, i) => {
              const [sx, sy] = toScreen(e.x, e.y);
              const color = categoryColor(e.word);
              return (
                <g key={i}>
                  <circle cx={sx} cy={sy} r={5} fill={color} opacity={0.8} />
                  <text x={sx} y={sy - 8} textAnchor="middle" fontSize={10} fontWeight={600}
                    fontFamily="var(--mono)" fill={color}>
                    {e.word}
                  </text>
                </g>
              );
            })}
          </svg>
        </div>
      </div>
      <div className="pl-step-right">
        <div className="pl-readout">
          <h3>Embedding Vectors</h3>
          <div className="pl-embed-list">
            {embeddings.map((e, i) => (
              <div key={i} className="pl-embed-row">
                <span className="pl-embed-word" style={{ color: categoryColor(e.word) }}>{e.word}</span>
                <span className="pl-embed-vec">[{e.x.toFixed(2)}, {e.y.toFixed(2)}]</span>
              </div>
            ))}
          </div>
        </div>
        <div className="pl-readout">
          <h3>Controls</h3>
          <div className="pl-control-row">
            <span className="pl-control-label">Random Seed</span>
            <input type="range" min={1} max={100} step={1} value={seed}
              onChange={(e) => onSeedChange(parseInt(e.target.value))}
              className="pl-slider" />
            <span className="pl-control-value">{seed}</span>
          </div>
        </div>
        <div className="pl-readout">
          <h3>How embeddings work</h3>
          <p className="pl-info">
            In real systems, embeddings are learned during training.
            <strong> Word2Vec</strong>, <strong>GloVe</strong>, and <strong>transformer embeddings</strong> map words to
            high-dimensional vectors (768-4096 dims) where semantic relationships are encoded as geometric patterns.
          </p>
        </div>
      </div>
    </div>
  );
}
