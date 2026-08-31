import { useMemo, useRef, useState, useCallback } from "react";
import { useAnimationFrame, EquationBlock } from "@ml-visual-lab/viz-kit";
import {
  POLYSEMY_EXAMPLES,
  analyzePolysemy,
  type PolysemyExample,
  type SentenceAnalysis,
} from "./contextualCore";
import "./ContextualViz.css";

// ---------------------------------------------------------------------------
// World → screen mapping. All examples share this fixed window.
// ---------------------------------------------------------------------------
const SIZE = 320;
const PAD = 30;
const X_MIN = -1.2, X_MAX = 6.4, Y_MIN = -4.6, Y_MAX = 4.6;

function toScreen(x: number, y: number): [number, number] {
  return [
    PAD + ((x - X_MIN) / (X_MAX - X_MIN)) * (SIZE - PAD * 2),
    PAD + ((Y_MAX - y) / (Y_MAX - Y_MIN)) * (SIZE - PAD * 2),
  ];
}

const BASE_COLOR = "#8b8b96";
const CONTEXT_COLOR = "#22c55e";

// Region accent colors — first sense vs second sense.
const REGION_COLORS = ["#dc2626", "#2563eb"];

// ---------------------------------------------------------------------------

function RegionBackdrop({ example }: { example: PolysemyExample }) {
  const keys = Object.keys(example.regions);
  return (
    <g>
      {keys.map((key, i) => {
        const region = example.regions[key];
        // define a bounding box around the region's words to draw a soft pill
        const xs = region.nextWords.map((n) => n.near[0]);
        const ys = region.nextWords.map((n) => n.near[1]);
        const cx = (Math.min(...xs) + Math.max(...xs)) / 2;
        const cy = (Math.min(...ys) + Math.max(...ys)) / 2;
        const [sx, sy] = toScreen(cx, cy);
        const [t1, u1] = toScreen(Math.min(...xs) - 0.7, Math.max(...ys) + 0.7);
        const [t2, u2] = toScreen(Math.max(...xs) + 0.7, Math.min(...ys) - 0.7);
        const w = t2 - t1;
        const h = u1 - u2;
        return (
          <g key={key}>
            <rect
              x={sx - w / 2} y={sy - h / 2} width={w} height={h} rx={18}
              fill={REGION_COLORS[i]} opacity={0.08} stroke={REGION_COLORS[i]} strokeOpacity={0.35} strokeDasharray="4 3"
            />
            <text x={sx} y={sy - h / 2 - 6} textAnchor="middle" fontSize={11}
              fontWeight={700} fill={REGION_COLORS[i]} fontFamily="var(--sans)">
              {region.emoji} {region.label}
            </text>
          </g>
        );
      })}
    </g>
  );
}

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

function SentenceCanvas({
  analysis,
  example,
  tween,
  playing,
  onToggle,
}: {
  analysis: SentenceAnalysis;
  example: PolysemyExample;
  tween: number;
  playing: boolean;
  onToggle: () => void;
}) {
  const fmt = (n: number) => n.toFixed(2);
  const regionColor =
    Object.keys(example.regions)[0] === analysis.senseKey ? REGION_COLORS[0] : REGION_COLORS[1];

  const { tokens, targetIndex, targetContext, weights } = analysis;
  const target = tokens[targetIndex];
  // interpolate the target's displayed position from base → contextual as tween goes 0→1
  const shownX = lerp(target.base[0], targetContext[0], tween);
  const shownY = lerp(target.base[1], targetContext[1], tween);

  const [tbx, tby] = toScreen(target.base[0], target.base[1]);
  const [ctxX, ctxY] = toScreen(targetContext[0], targetContext[1]);
  const [shx, shy] = toScreen(shownX, shownY);

  return (
    <div className="ctx-sentence">
      <div className="ctx-sentence-head">
        <span className="ctx-sentence-index">{analysis.senseEmoji}</span>
        <span className="ctx-sentence-text">{analysis.label}</span>
        <span className="ctx-sense-pill" style={{ color: regionColor, borderColor: regionColor }}>
          {analysis.senseLabel}
        </span>
      </div>

      <svg width={SIZE} height={SIZE} className="ctx-canvas">
        {/* grid + axes */}
        {Array.from({ length: 9 }, (_, g) => {
          const x = -1 + g;
          const [sx] = toScreen(x, 0);
          return <line key={`v${g}`} x1={sx} y1={PAD} x2={sx} y2={SIZE - PAD} stroke="var(--border)" strokeWidth={0.5} opacity={0.28} />;
        })}
        {Array.from({ length: 10 }, (_, g) => {
          const y = -4 + g;
          const [, sy] = toScreen(0, y);
          return <line key={`h${g}`} x1={PAD} y1={sy} x2={SIZE - PAD} y2={sy} stroke="var(--border)" strokeWidth={0.5} opacity={0.28} />;
        })}
        {(() => {
          const [ox, oy] = toScreen(0, 0);
          return (
            <>
              <line x1={PAD} y1={oy} x2={SIZE - PAD} y2={oy} stroke="var(--border)" strokeWidth={1} />
              <line x1={ox} y1={PAD} x2={ox} y2={SIZE - PAD} stroke="var(--border)" strokeWidth={1} />
            </>
          );
        })()}

        <RegionBackdrop example={example} />

        {/* attention lines from target */}
        {targetIndex >= 0 &&
          tokens.map((tok, j) => {
            if (j === targetIndex) return null;
            const w = weights[targetIndex][j];
            if (w < 0.005) return null;
            const [jx, jy] = toScreen(tok.contextual[0], tok.contextual[1]);
            return (
              <line key={j}
                x1={shx} y1={shy} x2={jx} y2={jy}
                stroke={CONTEXT_COLOR} strokeWidth={Math.max(1, w * 6)} opacity={0.15 + w * 0.85} />
            );
          })}

        {/* the target's path: base (hollow) → contextual (filled) */}
        {targetIndex >= 0 && (
          <g>
            {/* base ring */}
            <circle cx={tbx} cy={tby} r={13} fill="none" stroke={BASE_COLOR} strokeWidth={1.5} strokeDasharray="4 3" />
            <text x={tbx} y={tby - 18} textAnchor="middle" fontSize={9} fontFamily="var(--mono)" fill={BASE_COLOR}>
              base
            </text>
            {/* contextual ring (final resting spot) */}
            <circle cx={ctxX} cy={ctxY} r={8} fill="none" stroke={CONTEXT_COLOR} strokeWidth={2} opacity={0.9} />
            {/* target word marker (animating) */}
            <g transform={`translate(${shx},${shy})`}>
              <circle r={9} fill={CONTEXT_COLOR} opacity={0.95} />
              <text y={-12} textAnchor="middle" fontSize={10} fontWeight={700} fontFamily="var(--mono)" fill={CONTEXT_COLOR}>
                {target.word}
              </text>
            </g>
          </g>
        )}

        {/* other tokens */}
        {tokens.map((tok, j) => {
          if (j === targetIndex) return null;
          const [px, py] = toScreen(tok.contextual[0], tok.contextual[1]);
          const isOrigin = tok.base[0] === 0 && tok.base[1] === 0;
          return (
            <g key={j}>
              <circle cx={px} cy={py} r={5} fill={isOrigin ? "#94a3b8" : regionColor} opacity={0.85} />
              <text x={px} y={py - 8} textAnchor="middle" fontSize={isOrigin ? 8 : 10} fontWeight={600}
                fontFamily="var(--mono)" fill={isOrigin ? "#94a3b8" : regionColor}>
                {tok.word}
              </text>
            </g>
          );
        })}

        {/* axis labels */}
        <text x={SIZE - PAD} y={toScreen(0, 0)[1] - 6} textAnchor="end" fontSize={9} fontFamily="var(--mono)" fill="var(--text)" opacity={0.5}>
          {analysis.dK}d · {analysis.senseLabel}
        </text>
      </svg>

      <div className="ctx-vec-row">
        <span className="ctx-vec"><span className="ctx-vec-dot" style={{ background: BASE_COLOR }} /> base <code>[{fmt(target.base[0])}, {fmt(target.base[1])}]</code></span>
        <span className="ctx-arrow">→</span>
        <span className="ctx-vec"><span className="ctx-vec-dot" style={{ background: CONTEXT_COLOR }} /> contextual <code>[{fmt(targetContext[0])}, {fmt(targetContext[1])}]</code></span>
      </div>
      <button className="ctx-playbtn" onClick={onToggle}>
        {playing ? "⏸ Pause tween" : "▶ Play tween"}
      </button>
    </div>
  );
}

function PredictionPanel({ analysis, color }: { analysis: SentenceAnalysis; color: string }) {
  const max = analysis.predictions[0]?.score ?? 1;
  return (
    <div className="ctx-predict">
      <div className="ctx-predict-head">
        <span>Model predicts the next word</span>
      </div>
      {analysis.predictions.map((p) => (
        <div key={p.word} className="ctx-predict-row">
          <span className="ctx-predict-word">{p.word}</span>
          <div className="ctx-predict-track">
            <div className="ctx-predict-fill" style={{ width: `${(p.score / max) * 100}%`, background: color }} />
          </div>
          <span className="ctx-predict-val">{(p.score * 100).toFixed(0)}%</span>
        </div>
      ))}
      <p className="ctx-predict-note">
        The predicted word is drawn from the <strong>{analysis.senseLabel}</strong> sense — the same sentence the
        target word is embedded in.
      </p>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Matrix panel
// ---------------------------------------------------------------------------
type MatrixStage = "scores" | "scaled" | "weights" | "output";

const STAGES: { key: MatrixStage; label: string; tex: string }[] = [
  { key: "scores", label: "Scores · QKᵀ", tex: "\\text{scores}_{ij} = q_i \\cdot k_j" },
  { key: "scaled", label: "Scaled · ÷√d_k", tex: "\\text{scaled}_{ij} = \\frac{q_i \\cdot k_j}{\\sqrt{d_k}}" },
  { key: "weights", label: "Weights · softmax", tex: "w_{ij} = \\frac{e^{\\text{scaled}_{ij}}}{\\sum_\\ell e^{\\text{scaled}_{i\\ell}}}" },
  { key: "output", label: "Output · weights·V", tex: "\\text{out}_i = \\sum_j w_{ij}\\, v_j" },
];

function MatrixPanel({ analysis }: { analysis: SentenceAnalysis }) {
  const [stage, setStage] = useState<MatrixStage>("weights");
  const useStage = STAGES.find((s) => s.key === stage) ?? STAGES[2];

  const targetIndex = analysis.targetIndex;
  const maxAbs = (mat: number[][]) => Math.max(0.0001, ...mat.flatMap((r) => r.map((v) => Math.abs(v))));

  const renderMatrix = (mat: number[][], format: (v: number) => string, highlightPositive: boolean) => {
    const mAbs = highlightPositive ? 1 : maxAbs(mat);
    return (
      <div className="ctx-matrix">
        <div className="ctx-matrix-head">
          <div className="ctx-mcell ctx-mcorner" />
          {analysis.tokens.map((t, i) => (
            <div key={i} className="ctx-mcell ctx-mtok" style={i === targetIndex ? { color: CONTEXT_COLOR } : undefined}>
              {t.word}
            </div>
          ))}
        </div>
        {mat.map((row, i) => (
          <div key={i} className="ctx-matrix-row" style={i === targetIndex ? { background: "var(--accent-bg)" } : undefined}>
            <div className="ctx-mcell ctx-mtok" style={i === targetIndex ? { color: CONTEXT_COLOR } : undefined}>
              {analysis.tokens[i].word}
            </div>
            {row.map((v, j) => {
              const isTargetCol = j === targetIndex;
              const bg = highlightPositive
                ? `rgba(34,197,94,${Math.min(1, Math.max(0, v)) * 0.85})`
                : v >= 0
                  ? `rgba(37,99,235,${(v / mAbs) * 0.5})`
                  : `rgba(220,38,38,${Math.min(0.6, (Math.abs(v) / mAbs) * 0.6)})`;
              return (
                <div
                  key={j}
                  className="ctx-mcell ctx-mval"
                  style={{ background: bg, fontWeight: isTargetCol && i === targetIndex ? 800 : 500 }}
                >
                  {format(v)}
                </div>
              );
            })}
          </div>
        ))}
      </div>
    );
  };

  const fmt1 = (v: number) => v.toFixed(1);
  const fmt2 = (v: number) => v.toFixed(2);
  const fmtPct = (v: number) => (v * 100).toFixed(0) + "%";

  const body =
    stage === "scores" ? renderMatrix(analysis.scores as unknown as number[][], fmt1, false) :
    stage === "scaled" ? renderMatrix(analysis.scaled as unknown as number[][], fmt2, false) :
    stage === "weights" ? renderMatrix(analysis.weights as unknown as number[][], fmtPct, true) :
    renderMatrix(analysis.outputs as unknown as number[][], fmt2, false);

  return (
    <div className="ctx-panel">
      <div className="ctx-panel-head">
        <span className="ctx-panel-title">Attention matrix · {analysis.senseEmoji} {analysis.senseLabel}</span>
        <span className="ctx-panel-sub">{analysis.label}</span>
      </div>
      <div className="ctx-stage-tabs">
        {STAGES.map((s) => (
          <button key={s.key} className={`ctx-stage-btn ${stage === s.key ? "ctx-stage-btn--active" : ""}`}
            onClick={() => setStage(s.key)}>
            {s.label}
          </button>
        ))}
      </div>
      <EquationBlock tex={useStage.tex} />
      <div className="ctx-matrix-scroll">{body}</div>
      <p className="ctx-matrix-note">
        Rows are <strong>queries</strong> (each word asking), columns are <strong>keys</strong> (each word answering).
        The highlighted row is "{analysis.tokens[targetIndex].word}" — see which words it leans on, and watch those
        same scores become the blend that becomes its contextual embedding.
      </p>
    </div>
  );
}

// ---------------------------------------------------------------------------

export default function ContextualViz() {
  const [exampleIdx, setExampleIdx] = useState(0);
  const [temperature, setTemperature] = useState(1);
  const [tween, setTween] = useState(1);
  const [playing, setPlaying] = useState(false);
  const tweenRef = useRef(1);
  const playingRef = useRef(false);

  const example: PolysemyExample = POLYSEMY_EXAMPLES[exampleIdx];

  const analysis = useMemo(() => analyzePolysemy(example, temperature), [example, temperature]);

  useAnimationFrame((deltaMs) => {
    if (!playingRef.current) return;
    let t = tweenRef.current + deltaMs / 1600;
    if (t >= 1) { t = 1; playingRef.current = false; setPlaying(false); }
    tweenRef.current = t;
    setTween(t);
  }, playing);

  const toggle = useCallback(() => {
    setPlaying((p) => {
      const next = !p;
      playingRef.current = next;
      if (next && tweenRef.current >= 1) { tweenRef.current = 0; setTween(0); }
      return next;
    });
  }, []);

  const aColor = example.sentences[0].senseKey === Object.keys(example.regions)[0] ? REGION_COLORS[0] : REGION_COLORS[1];
  const bColor = example.sentences[1].senseKey === Object.keys(example.regions)[0] ? REGION_COLORS[0] : REGION_COLORS[1];

  const baseVec = analysis.a.tokens[analysis.a.targetIndex].base;
  const fmt = (n: number) => n.toFixed(2);

  return (
    <div className="ctx-viz">
      {/* Intro */}
      <div className="ctx-intro">
        <p>
          Some words carry <strong>more than one meaning</strong> — “apple” is a fruit one sentence and a tech
          company the next. To a model, the word always starts at the <strong>same</strong> static embedding{" "}
          <em>(its dictionary entry)</em>. Then <strong>self-attention</strong> lets the word look at the rest of the
          sentence and <em>blend their meanings in</em>. The result: a <strong>contextual embedding</strong> that has
          drifted toward whichever meaning the sentence supports.
        </p>
        <p className="ctx-intro-sub">
          Same word · same base vector · different surroundings → two different contextual embeddings.
        </p>
      </div>

      {/* Example selector */}
      <div className="ctx-example-tabs">
        {POLYSEMY_EXAMPLES.map((ex, i) => (
          <button key={ex.slug}
            className={`ctx-example-tab ${i === exampleIdx ? "ctx-example-tab--active" : ""}`}
            onClick={() => { setExampleIdx(i); setTween(1); tweenRef.current = 1; setPlaying(false); playingRef.current = false; }}>
            {ex.target}
          </button>
        ))}
      </div>

      <div className="ctx-title">
        <h3><span className="ctx-target">{example.target}</span> {example.sentence}</h3>
        <p className="ctx-blurb">{example.blurb}</p>
      </div>

      {/* The reveal */}
      <div className="ctx-reveal">
        <div className="ctx-reveal-cell">
          <span className="ctx-reveal-label">Same base embedding</span>
          <code>[{fmt(baseVec[0])}, {fmt(baseVec[1])}]</code>
          <span className="ctx-reveal-tag">identical in both sentences</span>
        </div>
        <div className="ctx-reveal-arrow">→</div>
        <div className="ctx-reveal-cell">
          <span className="ctx-reveal-label">Different contextual embeddings</span>
          <div className="ctx-reveal-vecs">
            <span style={{ color: aColor }}>{analysis.a.senseEmoji} [{fmt(analysis.a.targetContext[0])}, {fmt(analysis.a.targetContext[1])}]</span>
            <span style={{ color: bColor }}>{analysis.b.senseEmoji} [{fmt(analysis.b.targetContext[0])}, {fmt(analysis.b.targetContext[1])}]</span>
          </div>
          <span className="ctx-reveal-tag">distance = {analysis.contextualDistance.toFixed(2)}</span>
        </div>
      </div>

      {/* Two sentences side by side */}
      <div className="ctx-grid">
        <SentenceCanvas analysis={analysis.a} example={example} tween={tween} playing={playing} onToggle={toggle} />
        <SentenceCanvas analysis={analysis.b} example={example} tween={tween} playing={playing} onToggle={toggle} />
      </div>

      {/* Predictions */}
      <div className="ctx-grid">
        <PredictionPanel analysis={analysis.a} color={aColor} />
        <PredictionPanel analysis={analysis.b} color={bColor} />
      </div>

      {/* Controls */}
      <div className="ctx-controls">
        <span className="ctx-control-label">Temperature</span>
        <input type="range" min={0.4} max={2.5} step={0.1} value={temperature}
          onChange={(e) => setTemperature(parseFloat(e.target.value))} className="ctx-slider" />
        <span className="ctx-control-value">{temperature.toFixed(1)}</span>
        <span className="ctx-control-hint">lower = sharper attention (fewer words move the target); higher = softer spread</span>
      </div>

      {/* Matrices */}
      <div className="ctx-grid">
        <MatrixPanel analysis={analysis.a} />
        <MatrixPanel analysis={analysis.b} />
      </div>

      <div className="ctx-formula">
        <EquationBlock tex={"\\text{ContextualEmbedding}(w) = \\text{Attention}(Q,K,V)_w = \\text{softmax}\\left(\\frac{QK^T}{\\sqrt{d_k}}\\right)V,\\quad Q=K=V=\\text{embeddings}"} />
        <p className="ctx-formula-note">
          In this toy we use identity projections (Q = K = V = the token embeddings), so every number is traceable.
          The target word's contextual embedding is literally its row of the attention output — a blend of the whole
          sentence. This is the same mechanism that real encoder models (BERT and friends) use to build
          context-aware word representations.
        </p>
      </div>
    </div>
  );
}
