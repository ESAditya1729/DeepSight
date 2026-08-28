import { useState, useRef, useEffect, useCallback } from "react";
import { createNetwork, forward, trainEpoch, type Network } from "@ml-visual-lab/ml-core";
import { PlaybackControls, EquationBlock } from "@ml-visual-lab/viz-kit";
import "./BackpropFlowViz.css";

const SIZE = 440;
const NEURON_R = 15;

type Scenario = "healthy" | "vanishing" | "exploding";

const SCENARIOS: Record<Scenario, {
  label: string;
  description: string;
  arch: number[];
  acts: string[];
  lr: number;
  input: number[];
  target: number[];
  tex: string;
}> = {
  healthy: {
    label: "Healthy Flow",
    description: "ReLU activations + moderate LR = stable gradient flow across all layers.",
    arch: [2, 4, 4, 1],
    acts: ["relu", "relu", "sigmoid"],
    lr: 0.5,
    input: [0.5, 0.8],
    target: [1],
    tex: "\\text{ReLU}: f(x) = \\max(0, x)",
  },
  vanishing: {
    label: "Vanishing Gradient",
    description: "Deep sigmoid network — gradients shrink exponentially toward input layers.",
    arch: [2, 4, 4, 4, 4, 1],
    acts: ["sigmoid", "sigmoid", "sigmoid", "sigmoid", "sigmoid"],
    lr: 1.0,
    input: [0.5, 0.8],
    target: [1],
    tex: "\\sigma'(x) = \\sigma(x)(1-\\sigma(x)) \\leq 0.25",
  },
  exploding: {
    label: "Exploding Gradient",
    description: "Large learning rate causes gradients to grow exponentially.",
    arch: [2, 4, 4, 1],
    acts: ["relu", "relu", "sigmoid"],
    lr: 8.0,
    input: [0.5, 0.8],
    target: [1],
    tex: "\\|g_t\\| > 1 \\implies \\text{weights grow exponentially}",
  },
};

function gradientMagnitude(w: number): number {
  return Math.min(1, Math.abs(w) * 2);
}

function gradientColor(mag: number): string {
  if (mag > 0.7) return "#22c55e";
  if (mag > 0.3) return "#eab308";
  if (mag > 0.1) return "#f97316";
  return "#ef4444";
}

export default function BackpropFlowViz() {
  const [scenario, setScenario] = useState<Scenario>("healthy");
  const [running, setRunning] = useState(false);
  const [epochCount, setEpochCount] = useState(0);
  const [lossHistory, setLossHistory] = useState<number[]>([]);
  const [gradients, setGradients] = useState<number[]>([]);
  const netRef = useRef<Network | null>(null);
  const rafRef = useRef(0);
  const lossCanvasRef = useRef<HTMLCanvasElement>(null);

  const config = SCENARIOS[scenario];

  const initNetwork = useCallback(() => {
    netRef.current = createNetwork(config.arch, config.acts);
    setEpochCount(0);
    setLossHistory([]);
    setGradients([]);
    evalGradients();
  }, [config]);

  useEffect(() => { initNetwork(); }, [initNetwork]);

  const evalGradients = useCallback(() => {
    const net = netRef.current;
    if (!net) return;
    const output = forward(net, config.input);
    const loss = (output[0] - config.target[0]) ** 2;

    // Compute gradient magnitudes per layer
    const mags: number[] = [];
    for (const layer of net.layers) {
      let maxMag = 0;
      for (const row of layer.weights) {
        for (const w of row) {
          maxMag = Math.max(maxMag, Math.abs(w));
        }
      }
      mags.push(maxMag);
    }
    setGradients(mags);

    setLossHistory((prev) => {
      const next = [...prev, loss];
      return next.length > 150 ? next.slice(-150) : next;
    });
  }, [config]);

  const reset = useCallback(() => {
    setRunning(false);
    cancelAnimationFrame(rafRef.current);
    initNetwork();
  }, [initNetwork]);

  const stepOnce = useCallback(() => {
    const net = netRef.current;
    if (!net) return;
    trainEpoch(net, [config.input], [config.target], config.lr);
    setEpochCount((c) => c + 1);
    evalGradients();
  }, [config, evalGradients]);

  useEffect(() => {
    if (!running) return;
    let rafId = 0, last = 0;
    const tick = (time: number) => {
      if (time - last >= 40) { last = time; stepOnce(); }
      rafId = requestAnimationFrame(tick);
    };
    rafId = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(rafId);
  }, [running, stepOnce]);

  // Loss chart
  useEffect(() => {
    const canvas = lossCanvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const W = 300, H = 80;
    canvas.width = W; canvas.height = H;
    ctx.clearRect(0, 0, W, H);
    if (lossHistory.length < 2) return;
    const maxLoss = Math.max(...lossHistory, 0.1);

    ctx.strokeStyle = "var(--border)"; ctx.lineWidth = 0.5;
    ctx.beginPath(); ctx.moveTo(25, 5); ctx.lineTo(25, H - 12); ctx.lineTo(W - 5, H - 12); ctx.stroke();

    const chartW = W - 30, chartH = H - 17, chartX = 27;
    ctx.beginPath();
    for (let i = 0; i < lossHistory.length; i++) {
      const x = chartX + (i / (lossHistory.length - 1)) * chartW;
      const y = 5 + chartH - (lossHistory[i] / maxLoss) * chartH;
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.strokeStyle = "#aa3bff"; ctx.lineWidth = 2; ctx.stroke();
  }, [lossHistory]);

  // Network layout
  const layerSizes = config.arch;
  const LAYER_X = layerSizes.map((_, i) => 60 + (i / (layerSizes.length - 1)) * (SIZE - 120));
  const neuronPositions: [number, number][][] = [];
  for (let l = 0; l < layerSizes.length; l++) {
    const count = layerSizes[l];
    const spacing = Math.min(45, (SIZE - 60) / count);
    const startY = SIZE / 2 - ((count - 1) * spacing) / 2;
    neuronPositions.push(Array.from({ length: count }, (_, n) => [LAYER_X[l], startY + n * spacing]));
  }

  const layerLabels = ["Input", ...layerSizes.slice(1, -1).map((_, i) => `Hidden ${i + 1}`), "Output"];

  return (
    <div className="bp-viz">
      <div className="bp-canvas-wrap">
        <svg width={SIZE} height={SIZE} className="bp-canvas">
          {/* Connections with gradient-colored width */}
          {netRef.current?.layers.map((layer, l) =>
            layer.weights.map((weights, oi) =>
              weights.map((w, ii) => {
                const [x1, y1] = neuronPositions[l][ii];
                const [x2, y2] = neuronPositions[l + 1][oi];
                const mag = gradientMagnitude(w);
                const col = gradientColor(mag);
                return (
                  <line key={`${l}-${ii}-${oi}`}
                    x1={x1} y1={y1} x2={x2} y2={y2}
                    stroke={col}
                    strokeWidth={Math.max(0.5, mag * 3)}
                    strokeOpacity={0.6}
                  />
                );
              }),
            ),
          )}

          {/* Neurons */}
          {neuronPositions.map((layer, l) =>
            layer.map(([x, y], n) => (
              <g key={`n-${l}-${n}`}>
                <circle cx={x} cy={y} r={NEURON_R}
                  fill={gradients[l] !== undefined ? gradientColor(gradientMagnitude(gradients[l])) : "var(--code-bg)"}
                  stroke="var(--border)" strokeWidth={2} fillOpacity={0.3}
                />
                <circle cx={x} cy={y} r={NEURON_R * 0.6}
                  fill={gradients[l] !== undefined ? gradientColor(gradientMagnitude(gradients[l])) : "#888"}
                />
              </g>
            )),
          )}

          {/* Layer labels */}
          {layerLabels.map((label, i) => (
            <text key={i} x={LAYER_X[i]} y={SIZE - 12} textAnchor="middle"
              fontSize={11} fill="var(--text)" fontWeight={600}>{label}</text>
          ))}

          {/* Gradient flow arrows (bottom) */}
          <text x={SIZE / 2} y={SIZE - 2} textAnchor="middle" fontSize={9} fill="var(--text)" opacity={0.5}>
            ← gradients flow backward
          </text>
        </svg>
      </div>

      <div className="bp-sidebar">
        {/* Scenario selector */}
        <div className="bp-readout">
          <h3>Scenario</h3>
          <div className="bp-scenarios">
            {(Object.keys(SCENARIOS) as Scenario[]).map((s) => (
              <button key={s}
                className={`bp-scenario-btn ${s === scenario ? "bp-scenario-btn--active" : ""}`}
                onClick={() => { setScenario(s); }}>
                {SCENARIOS[s].label}
              </button>
            ))}
          </div>
          <p className="bp-info">{config.description}</p>
        </div>

        {/* Gradient legend */}
        <div className="bp-readout">
          <h3>Gradient Magnitude</h3>
          <div className="bp-gradient-legend">
            <div className="bp-gradient-bar">
              {Array.from({ length: 20 }, (_, i) => {
                const t = i / 19;
                return <div key={i} className="bp-gradient-seg" style={{ background: gradientColor(t) }} />;
              })}
            </div>
            <div className="bp-gradient-labels">
              <span>Weak (vanishing)</span><span>Strong</span>
            </div>
          </div>
          <div className="bp-layer-mags">
            {gradients.map((mag, i) => (
              <div key={i} className="bp-layer-mag">
                <span className="bp-layer-mag-label">{layerLabels[i]}</span>
                <div className="bp-layer-mag-bar">
                  <div className="bp-layer-mag-fill" style={{
                    width: `${Math.min(100, mag * 100)}%`,
                    background: gradientColor(gradientMagnitude(mag)),
                  }} />
                </div>
                <span className="bp-layer-mag-value" style={{ color: gradientColor(gradientMagnitude(mag)) }}>
                  {mag.toFixed(3)}
                </span>
              </div>
            ))}
          </div>
        </div>

        {/* Controls */}
        <div className="bp-readout">
          <h3>Training</h3>
          <PlaybackControls
            isRunning={running}
            onPlayPause={() => setRunning((r) => !r)}
            onStep={stepOnce}
            onReset={reset} />
          <div className="bp-stats-row">
            <div className="bp-stat">
              <span className="bp-stat-label">Epoch</span>
              <span className="bp-stat-value">{epochCount}</span>
            </div>
            <div className="bp-stat">
              <span className="bp-stat-label">LR</span>
              <span className="bp-stat-value">{config.lr}</span>
            </div>
          </div>
        </div>

        {/* Loss chart */}
        <div className="bp-readout">
          <h3>Loss</h3>
          <canvas ref={lossCanvasRef} className="bp-loss-canvas" />
        </div>

        {/* Formula */}
        <div className="bp-readout">
          <h3>Key Insight</h3>
          <EquationBlock tex={config.tex} />
          <p className="bp-info" style={{ marginTop: "0.4rem" }}>
            {scenario === "vanishing" && "Sigmoid derivatives are ≤0.25. Multiplying many of these together drives gradients to zero in early layers."}
            {scenario === "exploding" && "When LR is too large, weight updates overshoot, causing gradients to grow exponentially across layers."}
            {scenario === "healthy" && "ReLU avoids vanishing gradients because its derivative is 1 for positive inputs — gradients flow cleanly through the network."}
          </p>
        </div>
      </div>
    </div>
  );
}
