import { useRef, useEffect, useCallback, useState } from "react";
import { createNetwork, forward, type Network } from "@ml-visual-lab/ml-core";
import { trainEpoch } from "@ml-visual-lab/ml-core";
import { Slider, PlaybackControls, EquationBlock } from "@ml-visual-lab/viz-kit";
import "./NeuralNetworkViz.css";

const SIZE = 440;
const NEURON_R = 16;

type ActivationName = "sigmoid" | "relu" | "tanh";
type DatasetName = "xor" | "and" | "or" | "circle";

interface Dataset {
  label: string;
  inputs: number[][];
  targets: number[][];
  tex: string;
}

const DATASETS: Record<DatasetName, Dataset> = {
  xor: { label: "XOR", inputs: [[0,0],[0,1],[1,0],[1,1]], targets: [[0],[1],[1],[0]], tex: "\\text{XOR}" },
  and: { label: "AND", inputs: [[0,0],[0,1],[1,0],[1,1]], targets: [[0],[0],[0],[1]], tex: "\\text{AND}" },
  or:  { label: "OR",  inputs: [[0,0],[0,1],[1,0],[1,1]], targets: [[0],[1],[1],[1]], tex: "\\text{OR}" },
  circle: {
    label: "Circle",
    inputs: [[0,0],[0.9,0],[0,0.9],[-0.9,0],[0,-0.9],[2,0],[0,2],[-2,0],[0,-2],[1.5,1.5],[-1.5,1.5],[1.5,-1.5],[-1.5,-1.5]],
    targets: [[1],[1],[1],[1],[1],[0],[0],[0],[0],[0],[0],[0],[0]],
    tex: "\\text{Circle}"
  },
};

const ACTIVATIONS: { name: ActivationName; label: string; tex: string }[] = [
  { name: "sigmoid", label: "Sigmoid", tex: "\\sigma" },
  { name: "relu", label: "ReLU", tex: "\\text{ReLU}" },
  { name: "tanh", label: "Tanh", tex: "\\tanh" },
];

function weightColor(w: number): string {
  if (w > 0) return `rgba(59, 130, 246, ${Math.min(1, Math.abs(w) * 2)})`;
  return `rgba(239, 68, 68, ${Math.min(1, Math.abs(w) * 2)})`;
}

function activationColor(a: number): string {
  const v = Math.round(a * 255);
  return `rgb(${v}, ${v}, ${v})`;
}

export default function NeuralNetworkViz() {
  const [hiddenLayers, setHiddenLayers] = useState([4]);
  const [activation, setActivation] = useState<ActivationName>("sigmoid");
  const [datasetName, setDatasetName] = useState<DatasetName>("xor");
  const netRef = useRef<Network | null>(null);
  const [epochCount, setEpochCount] = useState(0);
  const [lr, setLr] = useState(2);
  const [epochsPerTick, setEpochsPerTick] = useState(10);
  const [running, setRunning] = useState(false);
  const [activations, setActivations] = useState<number[][]>([]);
  const [lossHistory, setLossHistory] = useState<number[]>([]);
  const [currentLoss, setCurrentLoss] = useState(0);
  const rafRef = useRef(0);
  const lastTimeRef = useRef(0);
  const lossCanvasRef = useRef<HTMLCanvasElement>(null);
  const [showGradients, setShowGradients] = useState(false);

  const dataset = DATASETS[datasetName];

  const createNet = useCallback(() => {
    const arch = [2, ...hiddenLayers, 1];
    const acts = arch.slice(1).map(() => activation);
    acts[acts.length - 1] = "sigmoid"; // output must stay a bounded 0-1 probability regardless of hidden activation
    netRef.current = createNetwork(arch, acts);
    setEpochCount(0);
    setLossHistory([]);
    setCurrentLoss(0);
    setActivations([]);
  }, [hiddenLayers, activation]);

  useEffect(() => { createNet(); }, [createNet]);

  const evalNetwork = useCallback(() => {
    const net = netRef.current;
    if (!net) return;
    const allActs: number[][] = [];
    for (const input of dataset.inputs) {
      const output = forward(net, input);
      const hidden = getHiddenActivations(net);
      allActs.push([...input, ...hidden, ...output]);
    }
    setActivations(allActs);

    let loss = 0;
    for (let i = 0; i < dataset.inputs.length; i++) {
      const output = forward(net, dataset.inputs[i]);
      for (let j = 0; j < output.length; j++) {
        loss += (output[j] - dataset.targets[i][j]) ** 2;
      }
    }
    setCurrentLoss(loss / dataset.inputs.length);
  }, [dataset]);

  const reset = useCallback(() => {
    setRunning(false);
    cancelAnimationFrame(rafRef.current);
    createNet();
  }, [createNet]);

  const stepOnce = useCallback(() => {
    const net = netRef.current;
    if (!net) return;
    trainEpoch(net, dataset.inputs, dataset.targets, lr);
    setEpochCount((c) => c + 1);
    evalNetwork();
    setLossHistory((prev) => {
      const next = [...prev, currentLoss];
      return next.length > 200 ? next.slice(-200) : next;
    });
  }, [lr, dataset, currentLoss, evalNetwork]);

  useEffect(() => {
    if (!running) { cancelAnimationFrame(rafRef.current); return; }
    const tick = (time: number) => {
      if (time - lastTimeRef.current >= 30) {
        lastTimeRef.current = time;
        const net = netRef.current;
        if (net) {
          for (let i = 0; i < epochsPerTick; i++) {
            trainEpoch(net, dataset.inputs, dataset.targets, lr);
          }
          setEpochCount((c) => c + epochsPerTick);
          evalNetwork();
          setLossHistory((prev) => {
            const next = [...prev, currentLoss];
            return next.length > 200 ? next.slice(-200) : next;
          });
        }
      }
      rafRef.current = requestAnimationFrame(tick);
    };
    rafRef.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(rafRef.current);
  }, [running, lr, epochsPerTick, evalNetwork, dataset, currentLoss]);

  useEffect(() => { evalNetwork(); }, [evalNetwork]);

  // Draw loss chart
  useEffect(() => {
    const canvas = lossCanvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const W = 300, H = 80;
    canvas.width = W; canvas.height = H;
    ctx.clearRect(0, 0, W, H);
    if (lossHistory.length < 2) return;
    const maxLoss = Math.max(...lossHistory, 0.5);

    ctx.strokeStyle = "var(--border)";
    ctx.lineWidth = 0.5;
    ctx.beginPath(); ctx.moveTo(25, 5); ctx.lineTo(25, H - 12); ctx.lineTo(W - 5, H - 12); ctx.stroke();

    ctx.fillStyle = "rgba(128,128,128,0.5)";
    ctx.font = "8px monospace";
    ctx.textAlign = "right";
    ctx.fillText(maxLoss.toFixed(2), 23, 12);

    const chartW = W - 30, chartH = H - 17, chartX = 27;
    ctx.beginPath();
    for (let i = 0; i < lossHistory.length; i++) {
      const x = chartX + (i / (lossHistory.length - 1)) * chartW;
      const y = 5 + chartH - (lossHistory[i] / maxLoss) * chartH;
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.strokeStyle = "#aa3bff";
    ctx.lineWidth = 2;
    ctx.stroke();
  }, [lossHistory]);

  // Network layout — derive from the actual network to stay in sync
  const netLayers = netRef.current?.layers;
  const layerSizes = netLayers
    ? [netLayers[0].weights[0].length, ...netLayers.map((l) => l.biases.length)]
    : [2, ...hiddenLayers, 1];
  const totalLayers = layerSizes.length;

  // Dynamic X positions based on layer count
  const LAYER_X = layerSizes.map((_, i) => 60 + (i / (totalLayers - 1)) * (SIZE - 120));

  const neuronPositions: [number, number][][] = [];
  for (let l = 0; l < layerSizes.length; l++) {
    const count = layerSizes[l];
    const spacing = Math.min(50, (SIZE - 60) / count);
    const startY = SIZE / 2 - ((count - 1) * spacing) / 2;
    const positions: [number, number][] = [];
    for (let n = 0; n < count; n++) {
      positions.push([LAYER_X[l], startY + n * spacing]);
    }
    neuronPositions.push(positions);
  }

  const layerLabels = ["Input", ...layerSizes.slice(1, -1).map((_, i) => layerSizes.length > 3 ? `Hidden ${i + 1}` : "Hidden"), "Output"];

  const addHiddenLayer = () => {
    if (hiddenLayers.length >= 4) return;
    setHiddenLayers((prev) => [...prev, 4]);
  };

  const removeHiddenLayer = () => {
    if (hiddenLayers.length <= 1) return;
    setHiddenLayers((prev) => prev.slice(0, -1));
  };

  const updateLayerSize = (idx: number, size: number) => {
    setHiddenLayers((prev) => prev.map((s, i) => i === idx ? size : s));
  };

  const fmt = (n: number) => n.toFixed(3);
  const archTex = layerSizes.map((s, i) => {
    if (i === 0) return String(s);
    const actTex = i === layerSizes.length - 1 ? "\\sigma" : ACTIVATIONS.find((a) => a.name === activation)?.tex ?? "";
    return `${s}_{${actTex}}`;
  }).join("\\to");

  return (
    <div className="nn-viz">
      <div className="nn-canvas-wrap">
        <svg width={SIZE} height={SIZE} className="nn-canvas">
          {/* Connections */}
          {netRef.current?.layers.map((layer, l) =>
            layer.weights.map((weights, oi) =>
              weights.map((w, ii) => {
                const [x1, y1] = neuronPositions[l][ii];
                const [x2, y2] = neuronPositions[l + 1][oi];
                return (
                  <line key={`${l}-${ii}-${oi}`}
                    x1={x1} y1={y1} x2={x2} y2={y2}
                    stroke={weightColor(w)}
                    strokeWidth={Math.min(3.5, Math.abs(w) * 2.5)} />
                );
              }),
            ),
          )}

          {/* Neurons */}
          {neuronPositions.map((layer, l) =>
            layer.map(([x, y], n) => {
              const offset = layerSizes.slice(0, l).reduce((a, b) => a + b, 0) + n;
              const activation = activations.length > 0 ? activations[0]?.[offset] ?? 0 : 0;
              return (
                <g key={`neuron-${l}-${n}`}>
                  <circle cx={x} cy={y} r={NEURON_R}
                    fill={activationColor(activation)}
                    stroke="var(--border)" strokeWidth={2} />
                  <text x={x} y={y + 4} textAnchor="middle"
                    fontSize={9} fontFamily="var(--mono)" fontWeight={600}
                    fill={activation > 0.5 ? "#fff" : "#333"}>
                    {activation.toFixed(2)}
                  </text>
                </g>
              );
            }),
          )}

          {/* Layer labels */}
          {layerLabels.map((label, i) => (
            <text key={i} x={LAYER_X[i]} y={SIZE - 12} textAnchor="middle"
              fontSize={11} fill="var(--text)" fontWeight={600}>{label}</text>
          ))}
        </svg>
      </div>

      <div className="nn-sidebar">
        {/* Architecture controls */}
        <div className="nn-controls">
          <h3>Architecture</h3>
          <div className="nn-arch-config">
            <div className="nn-arch-row">
              <span className="nn-arch-label">Input</span>
              <span className="nn-arch-fixed">2</span>
            </div>
            {hiddenLayers.map((size, i) => (
              <div key={i} className="nn-arch-row">
                <span className="nn-arch-label">{hiddenLayers.length > 1 ? `Hidden ${i + 1}` : "Hidden"}</span>
                <input type="range" min={1} max={8} value={size}
                  onChange={(e) => updateLayerSize(i, parseInt(e.target.value))}
                  className="nn-size-slider" />
                <span className="nn-arch-value">{size}</span>
              </div>
            ))}
            <div className="nn-arch-row">
              <span className="nn-arch-label">Output</span>
              <span className="nn-arch-fixed">1</span>
            </div>
            <div className="nn-arch-btns">
              <button className="nn-arch-btn" onClick={addHiddenLayer} disabled={hiddenLayers.length >= 4}>+ Layer</button>
              <button className="nn-arch-btn" onClick={removeHiddenLayer} disabled={hiddenLayers.length <= 1}>- Layer</button>
            </div>
          </div>

          <div className="nn-act-select">
            {ACTIVATIONS.map((a) => (
              <button key={a.name}
                className={`nn-act-btn ${a.name === activation ? "nn-act-btn--active" : ""}`}
                onClick={() => setActivation(a.name)}>
                {a.label}
              </button>
            ))}
          </div>
        </div>

        {/* Dataset selector */}
        <div className="nn-controls">
          <h3>Dataset</h3>
          <div className="nn-dataset-select">
            {(Object.keys(DATASETS) as DatasetName[]).map((name) => (
              <button key={name}
                className={`nn-dataset-btn ${name === datasetName ? "nn-dataset-btn--active" : ""}`}
                onClick={() => { setDatasetName(name); }}>
                {DATASETS[name].label}
              </button>
            ))}
          </div>
        </div>

        {/* Training controls */}
        <div className="nn-controls">
          <h3>Training</h3>
          <Slider label="Learning Rate" value={lr} min={0.1} max={10} step={0.1}
            onChange={setLr} format={(v) => v.toFixed(1)} />
          <Slider label="Epochs/Tick" value={epochsPerTick} min={1} max={50} step={1}
            onChange={setEpochsPerTick} format={(v) => String(Math.round(v))} />
          <PlaybackControls
            isRunning={running}
            onPlayPause={() => setRunning((r) => !r)}
            onStep={stepOnce}
            onReset={reset} />
          <label className="nn-toggle">
            <input type="checkbox" checked={showGradients} onChange={(e) => setShowGradients(e.target.checked)} />
            Show weight magnitudes
          </label>
        </div>

        {/* Stats + loss chart */}
        <div className="nn-readout">
          <h3>Training Progress</h3>
          <div className="nn-stats">
            <div className="nn-stat">
              <span className="nn-stat-label">Epoch</span>
              <span className="nn-stat-value">{epochCount}</span>
            </div>
            <div className="nn-stat">
              <span className="nn-stat-label">Loss (MSE)</span>
              <span className="nn-stat-value">{fmt(currentLoss)}</span>
            </div>
          </div>
          <div className="nn-loss-chart">
            <canvas ref={lossCanvasRef} className="nn-loss-canvas" />
          </div>
        </div>

        {/* Predictions */}
        <div className="nn-readout">
          <h3>Predictions</h3>
          <div className="nn-input-grid">
            <span className="nn-input-header">Input</span>
            <span className="nn-input-header">Target</span>
            <span className="nn-input-header">Output</span>
            {dataset.inputs.map((input, i) => {
              const output = activations[i]?.[activations[i].length - 1] ?? 0;
              const target = dataset.targets[i][0];
              const correct = Math.abs(output - target) < 0.3;
              return (
                <div key={i} style={{ display: "contents" }}>
                  <span className="nn-input-cell nn-input-cell--input">{`[${input}]`}</span>
                  <span className="nn-input-cell nn-input-cell--target">{target}</span>
                  <span className={`nn-input-cell ${correct ? "nn-input-cell--correct" : "nn-input-cell--wrong"}`}>
                    {output.toFixed(3)}
                  </span>
                </div>
              );
            })}
          </div>
        </div>

        <div className="nn-readout">
          <h3>Architecture</h3>
          <EquationBlock tex={`\\text{Network: } ${archTex}`} />
        </div>
      </div>
    </div>
  );
}

function getHiddenActivations(net: Network): number[] {
  if (net.caches.length === 0) return [];
  const result: number[] = [];
  for (const cache of net.caches) {
    for (let i = 0; i < cache.a.length - 1; i++) {
      result.push(cache.a[i]);
    }
  }
  return result;
}
