import { useState, useRef, useEffect, useCallback } from "react";
import { createNetwork, forward, trainEpoch, type Network } from "@ml-visual-lab/ml-core";
import { PlaybackControls, Slider } from "@ml-visual-lab/viz-kit";
import "./DecisionBoundaryLabViz.css";

const SIZE = 420;
const PADDING = 30;
const USABLE = SIZE - PADDING * 2;

type DatasetName = "linear" | "xor" | "spiral" | "circles" | "moons";
type Activation = "relu" | "sigmoid" | "tanh";

interface Point { x: number; y: number; label: number; }

interface DatasetConfig {
  label: string;
  tex: string;
  generate: () => Point[];
}

function rotate(pts: Point[], angle: number): Point[] {
  const c = Math.cos(angle), s = Math.sin(angle);
  return pts.map((p) => ({ x: p.x * c - p.y * s, y: p.x * s + p.y * c, label: p.label }));
}

const DATASETS: Record<DatasetName, DatasetConfig> = {
  linear: {
    label: "Linear",
    tex: "\\text{Linearly separable}",
    generate: () => [
      ...Array.from({ length: 30 }, () => ({ x: Math.random() * 0.8 - 0.9, y: Math.random() * 0.8 + 0.1, label: 0 })),
      ...Array.from({ length: 30 }, () => ({ x: Math.random() * 0.8 + 0.1, y: Math.random() * 0.8 - 0.9, label: 1 })),
    ],
  },
  xor: {
    label: "XOR",
    tex: "\\text{XOR pattern}",
    generate: () => [
      ...Array.from({ length: 20 }, () => ({ x: Math.random() * 0.6 - 0.9, y: Math.random() * 0.6 - 0.9, label: 0 })),
      ...Array.from({ length: 20 }, () => ({ x: Math.random() * 0.6 + 0.3, y: Math.random() * 0.6 + 0.3, label: 0 })),
      ...Array.from({ length: 20 }, () => ({ x: Math.random() * 0.6 - 0.9, y: Math.random() * 0.6 + 0.3, label: 1 })),
      ...Array.from({ length: 20 }, () => ({ x: Math.random() * 0.6 + 0.3, y: Math.random() * 0.6 - 0.9, label: 1 })),
    ],
  },
  spiral: {
    label: "Spiral",
    tex: "\\text{Two spirals}",
    generate: () => {
      const pts: Point[] = [];
      for (let i = 0; i < 50; i++) {
        const t = i / 50 * 2.5;
        pts.push({ x: t * Math.cos(t * 2) / 3, y: t * Math.sin(t * 2) / 3, label: 0 });
        pts.push({ x: -t * Math.cos(t * 2) / 3, y: -t * Math.sin(t * 2) / 3, label: 1 });
      }
      return pts;
    },
  },
  circles: {
    label: "Circles",
    tex: "\\text{Concentric circles}",
    generate: () => {
      const pts: Point[] = [];
      for (let i = 0; i < 40; i++) {
        const a = Math.random() * Math.PI * 2;
        const r1 = 0.2 + Math.random() * 0.1;
        const r2 = 0.5 + Math.random() * 0.15;
        pts.push({ x: Math.cos(a) * r1, y: Math.sin(a) * r1, label: 0 });
        pts.push({ x: Math.cos(a) * r2, y: Math.sin(a) * r2, label: 1 });
      }
      return pts;
    },
  },
  moons: {
    label: "Moons",
    tex: "\\text{Two moons}",
    generate: () => {
      const pts: Point[] = [];
      for (let i = 0; i < 40; i++) {
        const a = Math.PI * (i / 40);
        pts.push({ x: Math.cos(a) * 0.5 + (Math.random() - 0.5) * 0.1, y: Math.sin(a) * 0.5 + (Math.random() - 0.5) * 0.1, label: 0 });
        pts.push({ x: 1 - Math.cos(a) * 0.5 + (Math.random() - 0.5) * 0.1, y: 1 - Math.sin(a) * 0.5 - 0.5 + (Math.random() - 0.5) * 0.1, label: 1 });
      }
      return rotate(pts, -0.3);
    },
  },
};

const ACTIVATIONS: { name: Activation; label: string }[] = [
  { name: "relu", label: "ReLU" },
  { name: "sigmoid", label: "Sigmoid" },
  { name: "tanh", label: "Tanh" },
];

const NET_CONFIGS: { label: string; layers: number[] }[] = [
  { label: "Shallow", layers: [2, 4, 1] },
  { label: "Medium", layers: [2, 8, 4, 1] },
  { label: "Deep", layers: [2, 8, 8, 4, 1] },
];

function worldToScreen(x: number, y: number): [number, number] {
  const xRange: [number, number] = [-1.5, 1.5];
  const yRange: [number, number] = [-1.5, 1.5];
  const sx = PADDING + ((x - xRange[0]) / (xRange[1] - xRange[0])) * USABLE;
  const sy = PADDING + ((yRange[1] - y) / (yRange[1] - yRange[0])) * USABLE;
  return [sx, sy];
}

export default function DecisionBoundaryLabViz() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [datasetName, setDatasetName] = useState<DatasetName>("spiral");
  const [activation, setActivation] = useState<Activation>("relu");
  const [netConfig, setNetConfig] = useState(1);
  const [lr, setLr] = useState(0.05);
  const [running, setRunning] = useState(false);
  const [epochCount, setEpochCount] = useState(0);
  const [currentLoss, setCurrentLoss] = useState(0);
  const [accuracy, setAccuracy] = useState(0);
  const [dataVersion, setDataVersion] = useState(0);
  const netRef = useRef<Network | null>(null);
  const dataRef = useRef<Point[]>([]);
  const rafRef = useRef(0);

  const config = NET_CONFIGS[netConfig];

  const generateData = useCallback(() => {
    dataRef.current = DATASETS[datasetName].generate();
    setDataVersion((v) => v + 1);
  }, [datasetName]);

  const createNet = useCallback(() => {
    const acts = config.layers.slice(1).map(() => activation);
    acts[acts.length - 1] = "sigmoid"; // output must stay a bounded 0-1 probability regardless of hidden activation
    netRef.current = createNetwork(config.layers, acts);
    setEpochCount(0);
    setCurrentLoss(0);
    setAccuracy(0);
  }, [config, activation]);

  useEffect(() => { generateData(); }, [generateData]);
  useEffect(() => { createNet(); }, [createNet]);

  const evalNet = useCallback(() => {
    const net = netRef.current;
    const data = dataRef.current;
    if (!net || data.length === 0) return;
    let loss = 0, correct = 0;
    for (const p of data) {
      const out = forward(net, [p.x, p.y]);
      loss += (out[0] - p.label) ** 2;
      if (Math.abs(out[0] - p.label) < 0.5) correct++;
    }
    setCurrentLoss(loss / data.length);
    setAccuracy(correct / data.length);
  }, []);

  useEffect(() => { evalNet(); }, [evalNet, dataVersion]);

  const drawHeatmap = useCallback(() => {
    const canvas = canvasRef.current;
    const net = netRef.current;
    if (!canvas || !net) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, SIZE, SIZE);

    // Background grid
    const gridRes = 60;
    const cellW = SIZE / gridRes;
    for (let iy = 0; iy < gridRes; iy++) {
      for (let ix = 0; ix < gridRes; ix++) {
        const wx = -1.5 + (ix / gridRes) * 3;
        const wy = 1.5 - (iy / gridRes) * 3;
        const out = forward(net, [wx, wy]);
        const prob = out[0];
        const r = Math.round(239 * (1 - prob) + 34 * prob);
        const g = Math.round(68 * (1 - prob) + 197 * prob);
        const b = Math.round(68 * (1 - prob) + 94 * prob);
        ctx.fillStyle = `rgba(${r},${g},${b},0.4)`;
        ctx.fillRect(ix * cellW, iy * cellW, cellW + 1, cellW + 1);
      }
    }

    // Decision boundary contour (prob ≈ 0.5)
    ctx.strokeStyle = "rgba(255,255,255,0.8)";
    ctx.lineWidth = 2;
    const step = 0.05;
    for (let x = -1.5; x < 1.5; x += step) {
      for (let y = -1.5; y < 1.5; y += step) {
        const v00 = forward(net, [x, y])[0];
        const v10 = forward(net, [x + step, y])[0];
        const v01 = forward(net, [x, y + step])[0];
        if ((v00 - 0.5) * (v10 - 0.5) < 0 || (v00 - 0.5) * (v01 - 0.5) < 0) {
          const [sx, sy] = worldToScreen(x, y);
          ctx.fillStyle = "rgba(255,255,255,0.9)";
          ctx.fillRect(sx, sy, 2, 2);
        }
      }
    }

    // Data points
    for (const p of dataRef.current) {
      const [sx, sy] = worldToScreen(p.x, p.y);
      ctx.beginPath();
      ctx.arc(sx, sy, 4, 0, Math.PI * 2);
      ctx.fillStyle = p.label === 1 ? "#22c55e" : "#ef4444";
      ctx.fill();
      ctx.strokeStyle = "rgba(255,255,255,0.8)";
      ctx.lineWidth = 1.5;
      ctx.stroke();
    }

    // Axes
    ctx.strokeStyle = "rgba(128,128,128,0.2)";
    ctx.lineWidth = 0.5;
    const [ox, oy] = worldToScreen(0, 0);
    ctx.beginPath(); ctx.moveTo(ox, PADDING); ctx.lineTo(ox, SIZE - PADDING); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(PADDING, oy); ctx.lineTo(SIZE - PADDING, oy); ctx.stroke();
  }, []);

  useEffect(() => { drawHeatmap(); }, [drawHeatmap, dataVersion]);

  const reset = useCallback(() => {
    setRunning(false);
    cancelAnimationFrame(rafRef.current);
    createNet();
    generateData();
    drawHeatmap();
  }, [createNet, generateData, drawHeatmap]);

  const stepOnce = useCallback(() => {
    const net = netRef.current;
    const data = dataRef.current;
    if (!net || data.length === 0) return;
    const inputs = data.map((p) => [p.x, p.y]);
    const targets = data.map((p) => [p.label]);
    trainEpoch(net, inputs, targets, lr);
    setEpochCount((c) => c + 1);
    evalNet();
    drawHeatmap();
  }, [lr, evalNet, drawHeatmap]);

  useEffect(() => {
    if (!running) { cancelAnimationFrame(rafRef.current); return; }
    let rafId = 0, last = 0;
    const tick = (time: number) => {
      if (time - last >= 30) { last = time; stepOnce(); }
      rafId = requestAnimationFrame(tick);
    };
    rafId = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(rafId);
  }, [running, stepOnce]);

  const fmt = (n: number) => n.toFixed(4);

  return (
    <div className="db-viz">
      <div className="db-canvas-wrap">
        <canvas ref={canvasRef} width={SIZE} height={SIZE} className="db-canvas" />
      </div>

      <div className="db-sidebar">
        <div className="db-readout">
          <h3>Dataset</h3>
          <div className="db-btn-group">
            {(Object.keys(DATASETS) as DatasetName[]).map((n) => (
              <button key={n} className={`db-btn ${n === datasetName ? "db-btn--active" : ""}`}
                onClick={() => setDatasetName(n)}>
                {DATASETS[n].label}
              </button>
            ))}
          </div>
        </div>

        <div className="db-readout">
          <h3>Network</h3>
          <div className="db-btn-group">
            {NET_CONFIGS.map((c, i) => (
              <button key={i} className={`db-btn ${i === netConfig ? "db-btn--active" : ""}`}
                onClick={() => setNetConfig(i)}>
                {c.label}
              </button>
            ))}
          </div>
          <div className="db-btn-group" style={{ marginTop: "0.3rem" }}>
            {ACTIVATIONS.map((a) => (
              <button key={a.name} className={`db-btn ${a.name === activation ? "db-btn--active" : ""}`}
                onClick={() => setActivation(a.name)}>
                {a.label}
              </button>
            ))}
          </div>
        </div>

        <div className="db-readout">
          <h3>Training</h3>
          <Slider label="Learning Rate" value={lr} min={0.001} max={0.5} step={0.001}
            onChange={setLr} format={(v) => v.toFixed(3)} />
          <PlaybackControls isRunning={running}
            onPlayPause={() => setRunning((r) => !r)}
            onStep={stepOnce}
            onReset={reset} />
        </div>

        <div className="db-readout">
          <h3>Metrics</h3>
          <div className="db-metrics">
            <div className="db-metric">
              <span className="db-metric-label">Epoch</span>
              <span className="db-metric-value">{epochCount}</span>
            </div>
            <div className="db-metric">
              <span className="db-metric-label">Loss</span>
              <span className="db-metric-value">{fmt(currentLoss)}</span>
            </div>
            <div className="db-metric">
              <span className="db-metric-label">Accuracy</span>
              <span className="db-metric-value" style={{ color: accuracy > 0.8 ? "#22c55e" : accuracy > 0.5 ? "#eab308" : "#ef4444" }}>
                {(accuracy * 100).toFixed(1)}%
              </span>
            </div>
          </div>
        </div>

        <div className="db-readout">
          <h3>Legend</h3>
          <div className="db-legend">
            <div className="db-legend-item">
              <span className="db-legend-dot" style={{ background: "#ef4444" }} />
              <span>Class 0</span>
            </div>
            <div className="db-legend-item">
              <span className="db-legend-dot" style={{ background: "#22c55e" }} />
              <span>Class 1</span>
            </div>
            <div className="db-legend-item">
              <span className="db-legend-dot" style={{ background: "rgba(34,197,94,0.4)" }} />
              <span>Class 1 region</span>
            </div>
            <div className="db-legend-item">
              <span className="db-legend-dot" style={{ background: "rgba(239,68,68,0.4)" }} />
              <span>Class 0 region</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
