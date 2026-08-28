import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { numericalGradient, type LossFunctionName } from "@ml-visual-lab/ml-core";
import { PlaybackControls, EquationBlock } from "@ml-visual-lab/viz-kit";
import "./GradientDescentViz.css";

const SIZE = 400;
const RESOLUTION = 200;

interface LossConfig {
  range: [number, number];
  tex: string;
}

const LOSS_CONFIGS: Record<LossFunctionName, LossConfig> = {
  quadratic: { range: [-5, 5], tex: "f(x) = x^2" },
  rosenbrock: { range: [-2, 3], tex: "f(x,y) = (1-x)^2 + 100(y-x^2)^2" },
  beale: { range: [-4.5, 4.5], tex: "f(x,y) = (1.5-x+xy)^2 + (2.25-x+xy^2)^2 + (2.625-x+xy^3)^2" },
  himmelblau: { range: [-5, 5], tex: "f(x,y) = (x^2+y-11)^2 + (x+y^2-7)^2" },
};

const LOSS_NAMES: LossFunctionName[] = ["quadratic", "rosenbrock", "beale", "himmelblau"];

type OptimizerName = "sgd" | "momentum" | "rmsprop" | "adam" | "adagrad";

interface OptimizerConfig {
  label: string;
  color: string;
  lr: number;
}

const OPTIMIZER_CONFIGS: Record<OptimizerName, OptimizerConfig> = {
  sgd: { label: "SGD", color: "#ef4444", lr: 0.01 },
  momentum: { label: "Momentum", color: "#f97316", lr: 0.01 },
  rmsprop: { label: "RMSProp", color: "#22c55e", lr: 0.005 },
  adam: { label: "Adam", color: "#7c3aed", lr: 0.005 },
  adagrad: { label: "AdaGrad", color: "#2563eb", lr: 0.05 },
};

const OPTIMIZER_NAMES: OptimizerName[] = ["sgd", "momentum", "rmsprop", "adam", "adagrad"];

interface OptimizerState {
  pos: [number, number];
  vel: [number, number];
  sqAcc: [number, number];
  m: [number, number];
  v: [number, number];
  t: number;
  trail: [number, number][];
  lossHistory: number[];
}

function evalLoss(name: LossFunctionName, ...args: number[]): number {
  if (name === "quadratic") return args[0] * args[0];
  if (name === "rosenbrock") return (1 - args[0]) ** 2 + 100 * (args[1] - args[0] ** 2) ** 2;
  if (name === "beale") {
    const [x, y] = args;
    return (1.5 - x + x * y) ** 2 + (2.25 - x + x * y ** 2) ** 2 + (2.625 - x + x * y ** 3) ** 2;
  }
  const [x, y] = args;
  return (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2;
}

function lossGradient(name: LossFunctionName, ...args: number[]): number[] {
  return numericalGradient((...a) => evalLoss(name, ...a), ...args);
}

function clamp(v: number, lo: number, hi: number) { return Math.max(lo, Math.min(hi, v)); }

function worldToPixel(wx: number, wy: number, range: [number, number]): [number, number] {
  const [lo, hi] = range;
  return [((wx - lo) / (hi - lo)) * SIZE, SIZE - ((wy - lo) / (hi - lo)) * SIZE];
}

function pixelToWorld(px: number, py: number, range: [number, number]): [number, number] {
  const [lo, hi] = range;
  return [(px / SIZE) * (hi - lo) + lo, ((SIZE - py) / SIZE) * (hi - lo) + lo];
}

function lossToColor(t: number): [number, number, number] {
  const stops = [
    [0.0, 30, 27, 75], [0.15, 49, 46, 129], [0.3, 67, 56, 202],
    [0.4, 99, 102, 241], [0.5, 129, 140, 248], [0.55, 200, 200, 230],
    [0.6, 254, 249, 195], [0.7, 253, 224, 71], [0.8, 234, 179, 8],
    [0.9, 161, 98, 7], [1.0, 66, 32, 6],
  ];
  for (let i = 1; i < stops.length; i++) {
    if (t <= stops[i][0]) {
      const [t0, r0, g0, b0] = stops[i - 1];
      const [t1, r1, g1, b1] = stops[i];
      const f = (t - t0) / (t1 - t0);
      return [Math.round(r0 + f * (r1 - r0)), Math.round(g0 + f * (g1 - g0)), Math.round(b0 + f * (b1 - b0))];
    }
  }
  return [66, 32, 6];
}

function buildHeatmapImage(ctx: CanvasRenderingContext2D, name: LossFunctionName, range: [number, number]) {
  const imageData = ctx.createImageData(RESOLUTION, RESOLUTION);
  const data = imageData.data;
  let min = Infinity, max = -Infinity;
  const raw = new Float64Array(RESOLUTION * RESOLUTION);
  for (let iy = 0; iy < RESOLUTION; iy++) {
    for (let ix = 0; ix < RESOLUTION; ix++) {
      const wx = range[0] + (ix / (RESOLUTION - 1)) * (range[1] - range[0]);
      const wy = range[0] + ((RESOLUTION - 1 - iy) / (RESOLUTION - 1)) * (range[1] - range[0]);
      const v = evalLoss(name, wx, wy);
      raw[iy * RESOLUTION + ix] = v;
      if (v < min) min = v;
      if (v > max) max = v;
    }
  }
  const logMin = Math.log(min + 1), logMax = Math.log(max + 1), logRange = logMax - logMin || 1;
  for (let i = 0; i < raw.length; i++) {
    const t = clamp((Math.log(raw[i] + 1) - logMin) / logRange, 0, 1);
    const [r, g, b] = lossToColor(t);
    const px = i * 4;
    data[px] = r; data[px + 1] = g; data[px + 2] = b; data[px + 3] = 255;
  }
  const tempCanvas = document.createElement("canvas");
  tempCanvas.width = RESOLUTION; tempCanvas.height = RESOLUTION;
  const tempCtx = tempCanvas.getContext("2d")!;
  tempCtx.putImageData(imageData, 0, 0);
  ctx.imageSmoothingEnabled = true;
  ctx.drawImage(tempCanvas, 0, 0, SIZE, SIZE);
}

function drawContourLines(ctx: CanvasRenderingContext2D, name: LossFunctionName, range: [number, number], isDark: boolean) {
  const levels: number[] = [];
  let min = Infinity, max = -Infinity;
  for (let iy = 0; iy < RESOLUTION; iy += 4) {
    for (let ix = 0; ix < RESOLUTION; ix += 4) {
      const wx = range[0] + (ix / (RESOLUTION - 1)) * (range[1] - range[0]);
      const wy = range[0] + ((RESOLUTION - 1 - iy) / (RESOLUTION - 1)) * (range[1] - range[0]);
      const v = evalLoss(name, wx, wy);
      if (v < min) min = v;
      if (v > max) max = v;
    }
  }
  const logMin = Math.log(min + 1), logMax = Math.log(max + 1);
  for (let i = 0; i < 12; i++) {
    const t = i / 11;
    levels.push(Math.exp(logMin + t * (logMax - logMin)) - 1);
  }

  ctx.strokeStyle = isDark ? "rgba(255,255,255,0.15)" : "rgba(0,0,0,0.12)";
  ctx.lineWidth = 0.7;
  for (const level of levels) {
    ctx.beginPath();
    let started = false;
    for (let ix = 0; ix < RESOLUTION; ix += 2) {
      const wx = range[0] + (ix / (RESOLUTION - 1)) * (range[1] - range[0]);
      for (let iy = 0; iy < RESOLUTION; iy += 2) {
        const wy = range[0] + ((RESOLUTION - 1 - iy) / (RESOLUTION - 1)) * (range[1] - range[0]);
        const v = evalLoss(name, wx, wy);
        if (Math.abs(v - level) < (max - min) * 0.015) {
          const [px, py] = worldToPixel(wx, wy, range);
          if (!started) { ctx.moveTo(px, py); started = true; }
          else ctx.lineTo(px, py);
        }
      }
    }
    ctx.stroke();
  }
}

function drawCanvas(
  ctx: CanvasRenderingContext2D, heatmapImage: HTMLCanvasElement | null,
  runners: Map<OptimizerName, OptimizerState>, activeOptimizers: OptimizerName[],
  range: [number, number], isDark: boolean, showContour: boolean, showGradient: boolean,
  lossName: LossFunctionName,
) {
  ctx.clearRect(0, 0, SIZE, SIZE);
  if (heatmapImage) ctx.drawImage(heatmapImage, 0, 0, SIZE, SIZE);

  if (showContour) drawContourLines(ctx, lossName, range, isDark);

  // Draw gradient field (subtle arrows)
  if (showGradient) {
    const step = Math.floor(SIZE / 20);
    ctx.strokeStyle = isDark ? "rgba(255,255,255,0.18)" : "rgba(0,0,0,0.15)";
    ctx.lineWidth = 1;
    for (let px = step / 2; px < SIZE; px += step) {
      for (let py = step / 2; py < SIZE; py += step) {
        const [wx, wy] = pixelToWorld(px, py, range);
        const [gx, gy] = lossGradient(lossName, wx, wy);
        const mag = Math.hypot(gx, gy);
        if (mag < 0.001) continue;
        const len = Math.min(step * 0.4, mag * 3);
        const nx = -gx / mag, ny = gy / mag;
        ctx.beginPath();
        ctx.moveTo(px, py);
        ctx.lineTo(px + nx * len, py - ny * len);
        ctx.stroke();
      }
    }
  }

  // Draw optimizer trails
  for (const optName of activeOptimizers) {
    const state = runners.get(optName);
    if (!state || state.trail.length < 1) continue;
    const color = OPTIMIZER_CONFIGS[optName].color;

    if (state.trail.length > 1) {
      ctx.beginPath();
      const [sx, sy] = worldToPixel(state.trail[0][0], state.trail[0][1], range);
      ctx.moveTo(sx, sy);
      for (let i = 1; i < state.trail.length; i++) {
        const [x, y] = worldToPixel(state.trail[i][0], state.trail[i][1], range);
        ctx.lineTo(x, y);
      }
      ctx.strokeStyle = color + "cc";
      ctx.lineWidth = 2;
      ctx.stroke();
    }

    const [mx, my] = worldToPixel(state.pos[0], state.pos[1], range);
    ctx.beginPath();
    ctx.arc(mx, my, 6, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.fill();
    ctx.strokeStyle = isDark ? "#1a1a2e" : "#ffffff";
    ctx.lineWidth = 2;
    ctx.stroke();

    // Label
    ctx.fillStyle = color;
    ctx.font = "bold 10px ui-monospace, monospace";
    ctx.textAlign = "center";
    ctx.fillText(OPTIMIZER_CONFIGS[optName].label, mx, my - 10);
  }

  // Axis labels
  ctx.fillStyle = isDark ? "rgba(255,255,255,0.3)" : "rgba(0,0,0,0.3)";
  ctx.font = "10px ui-monospace, monospace";
  ctx.textAlign = "center";
  const [lo, hi] = range;
  ctx.fillText(lo.toFixed(1), 15, SIZE - 4);
  ctx.fillText(hi.toFixed(1), SIZE - 15, SIZE - 4);
}

function initOptimizerState(startPos: [number, number]): OptimizerState {
  return {
    pos: [...startPos] as [number, number], vel: [0, 0] as [number, number], sqAcc: [0, 0] as [number, number],
    m: [0, 0] as [number, number], v: [0, 0] as [number, number], t: 0,
    trail: [[...startPos] as [number, number]], lossHistory: [evalLoss("rosenbrock", startPos[0], startPos[1])],
  };
}

function stepOptimizer(
  state: OptimizerState, name: OptimizerName, lr: number, lossName: LossFunctionName, range: [number, number],
): OptimizerState {
  const [gx, gy] = lossGradient(lossName, state.pos[0], state.pos[1]);
  let nx = state.pos[0], ny = state.pos[1];
  const newVel: [number, number] = [...state.vel];
  const newSqAcc: [number, number] = [...state.sqAcc];
  const newM: [number, number] = [...state.m];
  const newV: [number, number] = [...state.v];
  const t = state.t + 1;
  const beta1 = 0.9, beta2 = 0.999, eps = 1e-8;

  switch (name) {
    case "sgd":
      nx = state.pos[0] - lr * gx;
      ny = state.pos[1] - lr * gy;
      break;
    case "momentum":
      newVel[0] = beta1 * state.vel[0] + gx;
      newVel[1] = beta1 * state.vel[1] + gy;
      nx = state.pos[0] - lr * newVel[0];
      ny = state.pos[1] - lr * newVel[1];
      break;
    case "rmsprop":
      newSqAcc[0] = 0.9 * state.sqAcc[0] + 0.1 * gx * gx;
      newSqAcc[1] = 0.9 * state.sqAcc[1] + 0.1 * gy * gy;
      nx = state.pos[0] - lr * gx / (Math.sqrt(newSqAcc[0]) + eps);
      ny = state.pos[1] - lr * gy / (Math.sqrt(newSqAcc[1]) + eps);
      break;
    case "adam": {
      newM[0] = beta1 * state.m[0] + (1 - beta1) * gx;
      newM[1] = beta1 * state.m[1] + (1 - beta1) * gy;
      newV[0] = beta2 * state.v[0] + (1 - beta2) * gx * gx;
      newV[1] = beta2 * state.v[1] + (1 - beta2) * gy * gy;
      const mHat0 = newM[0] / (1 - beta1 ** t), mHat1 = newM[1] / (1 - beta1 ** t);
      const vHat0 = newV[0] / (1 - beta2 ** t), vHat1 = newV[1] / (1 - beta2 ** t);
      nx = state.pos[0] - lr * mHat0 / (Math.sqrt(vHat0) + eps);
      ny = state.pos[1] - lr * mHat1 / (Math.sqrt(vHat1) + eps);
      break;
    }
    case "adagrad":
      newSqAcc[0] = state.sqAcc[0] + gx * gx;
      newSqAcc[1] = state.sqAcc[1] + gy * gy;
      nx = state.pos[0] - lr * gx / (Math.sqrt(newSqAcc[0]) + eps);
      ny = state.pos[1] - lr * gy / (Math.sqrt(newSqAcc[1]) + eps);
      break;
  }

  nx = clamp(nx, range[0], range[1]);
  ny = clamp(ny, range[0], range[1]);
  const loss = evalLoss(lossName, nx, ny);
  const trail: [number, number][] = [...state.trail, [nx, ny]];
  if (trail.length > 300) trail.splice(0, trail.length - 300);
  const lossHistory = [...state.lossHistory, loss];
  if (lossHistory.length > 300) lossHistory.splice(0, lossHistory.length - 300);

  return { pos: [nx, ny] as [number, number], vel: newVel, sqAcc: newSqAcc, m: newM, v: newV, t, trail, lossHistory };
}

export default function GradientDescentViz() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [lossName, setLossName] = useState<LossFunctionName>("rosenbrock");
  const [running, setRunning] = useState(false);
  const [showContour, setShowContour] = useState(true);
  const [showGradient, setShowGradient] = useState(false);
  const [activeOptimizers, setActiveOptimizers] = useState<OptimizerName[]>(["sgd", "momentum", "adam"]);
  const [optLrs, setOptLrs] = useState<Record<OptimizerName, number>>(
    Object.fromEntries(OPTIMIZER_NAMES.map((n) => [n, OPTIMIZER_CONFIGS[n].lr])) as Record<OptimizerName, number>
  );
  const startPos: [number, number] = [-1, 1];

  const runnersRef = useRef<Map<OptimizerName, OptimizerState>>(new Map());
  const [runnersVersion, setRunnersVersion] = useState(0);

  const config = LOSS_CONFIGS[lossName];
  const range = config.range;

  const heatmapImage = useMemo(() => {
    const c = document.createElement("canvas");
    c.width = RESOLUTION; c.height = RESOLUTION;
    buildHeatmapImage(c.getContext("2d")!, lossName, range);
    return c;
  }, [lossName, range]);

  const isDark = typeof window !== "undefined" && window.matchMedia("(data-theme=dark)").matches
    || typeof window !== "undefined" && window.matchMedia("(prefers-color-scheme: dark)").matches;

  useEffect(() => {
    const ctx = canvasRef.current?.getContext("2d");
    if (!ctx) return;
    drawCanvas(ctx, heatmapImage, runnersRef.current, activeOptimizers, range, isDark, showContour, showGradient, lossName);
  }, [heatmapImage, activeOptimizers, runnersVersion, range, isDark, showContour, showGradient, lossName]);

  const reset = useCallback(() => {
    setRunning(false);
    runnersRef.current.clear();
    for (const name of activeOptimizers) {
      runnersRef.current.set(name, initOptimizerState(startPos));
    }
    setRunnersVersion((v) => v + 1);
  }, [activeOptimizers, startPos]);

  useEffect(() => {
    reset();
  }, [lossName]);

  const stepOnce = useCallback(() => {
    for (const name of activeOptimizers) {
      const state = runnersRef.current.get(name);
      if (state) {
        runnersRef.current.set(name, stepOptimizer(state, name, optLrs[name], lossName, range));
      }
    }
    setRunnersVersion((v) => v + 1);
  }, [activeOptimizers, optLrs, lossName, range]);

  useEffect(() => {
    if (!running) return;
    let rafId = 0;
    let lastTime = 0;
    const tick = (time: number) => {
      if (time - lastTime >= 30) {
        lastTime = time;
        stepOnce();
      }
      rafId = requestAnimationFrame(tick);
    };
    rafId = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(rafId);
  }, [running, stepOnce]);

  const handleCanvasPointerDown = useCallback((e: React.PointerEvent<HTMLCanvasElement>) => {
    if (running) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const px = (e.clientX - rect.left) * (SIZE / rect.width);
    const py = (e.clientY - rect.top) * (SIZE / rect.height);
    const [wx, wy] = pixelToWorld(px, py, range);
    const pos: [number, number] = [clamp(wx, range[0], range[1]), clamp(wy, range[0], range[1])];
    runnersRef.current.clear();
    for (const name of activeOptimizers) {
      runnersRef.current.set(name, initOptimizerState(pos));
    }
    setRunnersVersion((v) => v + 1);
    e.currentTarget.setPointerCapture(e.pointerId);
  }, [running, range, activeOptimizers]);

  const toggleOptimizer = (name: OptimizerName) => {
    setActiveOptimizers((prev) => {
      const next = prev.includes(name) ? prev.filter((n) => n !== name) : [...prev, name];
      if (!runnersRef.current.has(name)) {
        runnersRef.current.set(name, initOptimizerState(startPos));
      }
      return next;
    });
  };

  // Loss chart for race mode
  const lossCanvasRef = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const canvas = lossCanvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const W = 300, H = 100;
    canvas.width = W; canvas.height = H;
    ctx.clearRect(0, 0, W, H);

    let maxLoss = 0.5;
    for (const name of activeOptimizers) {
      const state = runnersRef.current.get(name);
      if (state) {
        for (const l of state.lossHistory) {
          if (l > maxLoss) maxLoss = l;
        }
      }
    }

    // Axes
    ctx.strokeStyle = isDark ? "rgba(255,255,255,0.2)" : "rgba(0,0,0,0.15)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(30, 5); ctx.lineTo(30, H - 15); ctx.lineTo(W - 5, H - 15);
    ctx.stroke();

    // Y axis labels
    ctx.fillStyle = isDark ? "rgba(255,255,255,0.4)" : "rgba(0,0,0,0.4)";
    ctx.font = "9px ui-monospace, monospace";
    ctx.textAlign = "right";
    ctx.fillText(maxLoss.toFixed(1), 27, 12);
    ctx.fillText("0", 27, H - 16);

    const chartW = W - 35;
    const chartH = H - 20;
    const chartX = 32;

    for (const name of activeOptimizers) {
      const state = runnersRef.current.get(name);
      if (!state || state.lossHistory.length < 2) continue;
      ctx.beginPath();
      for (let i = 0; i < state.lossHistory.length; i++) {
        const x = chartX + (i / Math.max(1, state.lossHistory.length - 1)) * chartW;
        const y = 5 + chartH - (state.lossHistory[i] / maxLoss) * chartH;
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }
      ctx.strokeStyle = OPTIMIZER_CONFIGS[name].color;
      ctx.lineWidth = 1.5;
      ctx.stroke();
    }
  }, [runnersVersion, activeOptimizers, isDark]);

  return (
    <div className="gd-viz">
      <div className="gd-canvas-wrap">
        <canvas ref={canvasRef} width={SIZE} height={SIZE} className="gd-canvas"
          onPointerDown={handleCanvasPointerDown} />
      </div>

      <div className="gd-sidebar">
        <div className="gd-controls">
          <h3>Loss Function</h3>
          <div className="gd-function-select">
            {LOSS_NAMES.map((name) => (
              <button key={name} type="button"
                className={`gd-function-btn ${name === lossName ? "gd-function-btn--active" : ""}`}
                onClick={() => setLossName(name)}>
                {name.charAt(0).toUpperCase() + name.slice(1)}
              </button>
            ))}
          </div>
        </div>

        <div className="gd-controls">
          <h3>Optimizers</h3>
          <div className="gd-opt-grid">
            {OPTIMIZER_NAMES.map((name) => (
              <button key={name}
                className={`gd-opt-btn ${activeOptimizers.includes(name) ? "gd-opt-btn--active" : ""}`}
                style={{ borderColor: activeOptimizers.includes(name) ? OPTIMIZER_CONFIGS[name].color : undefined }}
                onClick={() => toggleOptimizer(name)}>
                <span className="gd-opt-dot" style={{ background: OPTIMIZER_CONFIGS[name].color }} />
                {OPTIMIZER_CONFIGS[name].label}
              </button>
            ))}
          </div>
          <div className="gd-opt-lrs">
            {activeOptimizers.map((name) => (
              <div key={name} className="gd-opt-lr-row">
                <span className="gd-opt-lr-label" style={{ color: OPTIMIZER_CONFIGS[name].color }}>
                  {OPTIMIZER_CONFIGS[name].label}
                </span>
                <input type="range" min={0.0001} max={0.2} step={0.0001}
                  value={optLrs[name]}
                  onChange={(e) => setOptLrs((prev) => ({ ...prev, [name]: parseFloat(e.target.value) }))}
                  className="gd-lr-slider" />
                <span className="gd-opt-lr-value">{optLrs[name].toFixed(4)}</span>
              </div>
            ))}
          </div>
        </div>

        <div className="gd-controls">
          <h3>Controls</h3>
          <PlaybackControls
            isRunning={running}
            onPlayPause={() => {
              setRunning((r) => !r);
            }}
            onStep={stepOnce}
            onReset={reset}
          />
          <div className="gd-toggles">
            <label className="gd-toggle">
              <input type="checkbox" checked={showContour} onChange={(e) => setShowContour(e.target.checked)} />
              Contour lines
            </label>
            <label className="gd-toggle">
              <input type="checkbox" checked={showGradient} onChange={(e) => setShowGradient(e.target.checked)} />
              Gradient field
            </label>
          </div>
        </div>

        {/* Loss Race Chart */}
        <div className="gd-readout">
          <h3>Loss Over Time</h3>
          <canvas ref={lossCanvasRef} className="gd-loss-canvas" />
          <div className="gd-loss-legend">
            {activeOptimizers.map((name) => (
              <span key={name} className="gd-loss-legend-item">
                <span className="gd-loss-legend-dot" style={{ background: OPTIMIZER_CONFIGS[name].color }} />
                {OPTIMIZER_CONFIGS[name].label}
              </span>
            ))}
          </div>
        </div>

        <div className="gd-readout">
          <h3>Function</h3>
          <EquationBlock tex={config.tex} />
        </div>

        <div className="gd-readout">
          <h3>Update Rules</h3>
          <EquationBlock tex={`\\theta_{t+1} = \\theta_t - \\alpha \\nabla f(\\theta_t)`} />
        </div>
      </div>
    </div>
  );
}
