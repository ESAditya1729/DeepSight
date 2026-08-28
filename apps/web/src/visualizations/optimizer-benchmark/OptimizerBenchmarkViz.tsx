import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { numericalGradient } from "@ml-visual-lab/ml-core";
import { PlaybackControls, EquationBlock } from "@ml-visual-lab/viz-kit";
import "./OptimizerBenchmarkViz.css";

const SIZE = 360;

type Surface = "rosenbrock" | "himmelblau" | "beale" | "saddle" | "ravine" | "noisy";

interface SurfaceConfig {
  label: string;
  range: [number, number];
  tex: string;
  fn: (x: number, y: number) => number;
}

const SURFACES: Record<Surface, SurfaceConfig> = {
  rosenbrock: { label: "Rosenbrock", range: [-2, 3], tex: "(1-x)^2 + 100(y-x^2)^2", fn: (x, y) => (1 - x) ** 2 + 100 * (y - x ** 2) ** 2 },
  himmelblau: { label: "Himmelblau", range: [-5, 5], tex: "(x^2+y-11)^2 + (x+y^2-7)^2", fn: (x, y) => (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2 },
  beale: { label: "Beale", range: [-4.5, 4.5], tex: "(1.5-x+xy)^2 + (2.25-x+xy^2)^2 + (2.625-x+xy^3)^2", fn: (x, y) => (1.5 - x + x * y) ** 2 + (2.25 - x + x * y ** 2) ** 2 + (2.625 - x + x * y ** 3) ** 2 },
  saddle: { label: "Saddle", range: [-3, 3], tex: "x^2 - y^2", fn: (x, y) => x ** 2 - y ** 2 },
  ravine: { label: "Ravine", range: [-5, 5], tex: "0.1x^2 + 10y^2", fn: (x, y) => 0.1 * x ** 2 + 10 * y ** 2 },
  noisy: { label: "Noisy Bowl", range: [-5, 5], tex: "x^2 + y^2 + 3\\sin(3x)\\cos(3y)", fn: (x, y) => x ** 2 + y ** 2 + 3 * Math.sin(3 * x) * Math.cos(3 * y) },
};

type OptName = "sgd" | "momentum" | "nesterov" | "rmsprop" | "adam" | "adagrad";

interface OptConfig {
  label: string;
  color: string;
  lr: number;
}

const OPTS: Record<OptName, OptConfig> = {
  sgd: { label: "SGD", color: "#ef4444", lr: 0.01 },
  momentum: { label: "Momentum", color: "#f97316", lr: 0.01 },
  nesterov: { label: "Nesterov", color: "#eab308", lr: 0.01 },
  rmsprop: { label: "RMSProp", color: "#22c55e", lr: 0.005 },
  adam: { label: "Adam", color: "#7c3aed", lr: 0.005 },
  adagrad: { label: "AdaGrad", color: "#2563eb", lr: 0.05 },
};

const OPT_NAMES: OptName[] = ["sgd", "momentum", "nesterov", "rmsprop", "adam", "adagrad"];

interface RunnerState {
  pos: [number, number];
  vel: [number, number];
  sqAcc: [number, number];
  m: [number, number];
  v: [number, number];
  t: number;
  trail: [number, number][];
}

function clamp(v: number, lo: number, hi: number) { return Math.max(lo, Math.min(hi, v)); }

function worldToPixel(wx: number, wy: number, range: [number, number]): [number, number] {
  const [lo, hi] = range;
  return [((wx - lo) / (hi - lo)) * SIZE, SIZE - ((wy - lo) / (hi - lo)) * SIZE];
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

function buildHeatmap(ctx: CanvasRenderingContext2D, surfaceFn: (x: number, y: number) => number, range: [number, number]) {
  const res = 150;
  const imageData = ctx.createImageData(res, res);
  const data = imageData.data;
  const raw = new Float64Array(res * res);
  let min = Infinity, max = -Infinity;
  for (let iy = 0; iy < res; iy++) {
    for (let ix = 0; ix < res; ix++) {
      const wx = range[0] + (ix / (res - 1)) * (range[1] - range[0]);
      const wy = range[0] + ((res - 1 - iy) / (res - 1)) * (range[1] - range[0]);
      const v = surfaceFn(wx, wy);
      raw[iy * res + ix] = v;
      if (v < min) min = v;
      if (v > max) max = v;
    }
  }
  const logMin = Math.log(min + 1), logMax = Math.log(max + 1), logR = logMax - logMin || 1;
  for (let i = 0; i < raw.length; i++) {
    const t = clamp((Math.log(raw[i] + 1) - logMin) / logR, 0, 1);
    const [r, g, b] = lossToColor(t);
    const px = i * 4;
    data[px] = r; data[px + 1] = g; data[px + 2] = b; data[px + 3] = 255;
  }
  const tmp = document.createElement("canvas");
  tmp.width = res; tmp.height = res;
  tmp.getContext("2d")!.putImageData(imageData, 0, 0);
  ctx.imageSmoothingEnabled = true;
  ctx.drawImage(tmp, 0, 0, SIZE, SIZE);
}

function stepOpt(state: RunnerState, name: OptName, lr: number, grad: [number, number]): RunnerState {
  const [gx, gy] = grad;
  let nx = state.pos[0], ny = state.pos[1];
  let newVel = [...state.vel] as [number, number];
  let newSq = [...state.sqAcc] as [number, number];
  let newM = [...state.m] as [number, number];
  let newV = [...state.v] as [number, number];
  const t = state.t + 1;
  const b1 = 0.9, b2 = 0.999, eps = 1e-8;
  switch (name) {
    case "sgd": nx -= lr * gx; ny -= lr * gy; break;
    case "momentum": newVel = [b1 * newVel[0] + gx, b1 * newVel[1] + gy]; nx -= lr * newVel[0]; ny -= lr * newVel[1]; break;
    case "nesterov": {
      const vx = b1 * newVel[0] + gx, vy = b1 * newVel[1] + gy;
      nx -= lr * (gx + b1 * vx); ny -= lr * (gy + b1 * vy);
      newVel = [vx, vy]; break;
    }
    case "rmsprop":
      newSq = [0.9 * newSq[0] + 0.1 * gx * gx, 0.9 * newSq[1] + 0.1 * gy * gy];
      nx -= lr * gx / (Math.sqrt(newSq[0]) + eps); ny -= lr * gy / (Math.sqrt(newSq[1]) + eps); break;
    case "adam":
      newM = [b1 * newM[0] + (1 - b1) * gx, b1 * newM[1] + (1 - b1) * gy];
      newV = [b2 * newV[0] + (1 - b2) * gx * gx, b2 * newV[1] + (1 - b2) * gy * gy];
      nx -= lr * (newM[0] / (1 - b1 ** t)) / (Math.sqrt(newV[0] / (1 - b2 ** t)) + eps);
      ny -= lr * (newM[1] / (1 - b1 ** t)) / (Math.sqrt(newV[1] / (1 - b2 ** t)) + eps);
      break;
    case "adagrad":
      newSq = [newSq[0] + gx * gx, newSq[1] + gy * gy];
      nx -= lr * gx / (Math.sqrt(newSq[0]) + eps); ny -= lr * gy / (Math.sqrt(newSq[1]) + eps); break;
  }
  return { pos: [nx, ny], vel: newVel, sqAcc: newSq, m: newM, v: newV, t, trail: [...state.trail, [nx, ny]] };
}

export default function OptimizerBenchmarkViz() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [surface, setSurface] = useState<Surface>("rosenbrock");
  const [running, setRunning] = useState(false);
  const [startPos, setStartPos] = useState<[number, number]>([-1, 1]);
  const [activeOpts, setActiveOpts] = useState<OptName[]>(["sgd", "momentum", "adam", "rmsprop"]);
  const [optLrs, setOptLrs] = useState<Record<OptName, number>>(
    Object.fromEntries(OPT_NAMES.map((n) => [n, OPTS[n].lr])) as Record<OptName, number>,
  );
  const runnersRef = useRef<Map<OptName, RunnerState>>(new Map());
  const [version, setVersion] = useState(0);

  const config = SURFACES[surface];
  const range = config.range;

  const heatmapImage = useMemo(() => {
    const c = document.createElement("canvas");
    c.width = SIZE; c.height = SIZE;
    buildHeatmap(c.getContext("2d")!, config.fn, range);
    return c;
  }, [config.fn, range]);

  const isDark = typeof window !== "undefined" &&
    (window.matchMedia("(data-theme=dark)").matches || window.matchMedia("(prefers-color-scheme: dark)").matches);

  useEffect(() => {
    runnersRef.current.clear();
    for (const n of activeOpts) runnersRef.current.set(n, { pos: [...startPos], vel: [0, 0], sqAcc: [0, 0], m: [0, 0], v: [0, 0], t: 0, trail: [[...startPos]] });
    setVersion((v) => v + 1);
  }, [surface]);

  useEffect(() => {
    const ctx = canvasRef.current?.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, SIZE, SIZE);
    ctx.drawImage(heatmapImage, 0, 0);
    for (const name of activeOpts) {
      const state = runnersRef.current.get(name);
      if (!state || state.trail.length < 2) continue;
      const color = OPTS[name].color;
      ctx.beginPath();
      const [sx, sy] = worldToPixel(state.trail[0][0], state.trail[0][1], range);
      ctx.moveTo(sx, sy);
      for (let i = 1; i < state.trail.length; i++) {
        const [x, y] = worldToPixel(state.trail[i][0], state.trail[i][1], range);
        ctx.lineTo(x, y);
      }
      ctx.strokeStyle = color + "cc"; ctx.lineWidth = 2; ctx.stroke();
      const [mx, my] = worldToPixel(state.pos[0], state.pos[1], range);
      ctx.beginPath(); ctx.arc(mx, my, 5, 0, Math.PI * 2);
      ctx.fillStyle = color; ctx.fill();
      ctx.strokeStyle = isDark ? "#1a1a2e" : "#fff"; ctx.lineWidth = 2; ctx.stroke();
    }
  }, [heatmapImage, activeOpts, version, range, isDark]);

  const reset = useCallback(() => {
    setRunning(false);
    runnersRef.current.clear();
    for (const n of activeOpts) runnersRef.current.set(n, { pos: [...startPos], vel: [0, 0], sqAcc: [0, 0], m: [0, 0], v: [0, 0], t: 0, trail: [[...startPos]] });
    setVersion((v) => v + 1);
  }, [activeOpts, startPos]);

  useEffect(() => { reset(); }, [surface]);

  const stepOnce = useCallback(() => {
    for (const name of activeOpts) {
      const state = runnersRef.current.get(name);
      if (!state) continue;
      const g = numericalGradient((x: number, y: number) => config.fn(x, y), state.pos[0], state.pos[1]) as [number, number];
      runnersRef.current.set(name, stepOpt(state, name, optLrs[name], g));
    }
    setVersion((v) => v + 1);
  }, [activeOpts, optLrs, config.fn]);

  useEffect(() => {
    if (!running) return;
    let rafId = 0, last = 0;
    const tick = (time: number) => {
      if (time - last >= 30) { last = time; stepOnce(); }
      rafId = requestAnimationFrame(tick);
    };
    rafId = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(rafId);
  }, [running, stepOnce]);

  const handlePointerDown = useCallback((e: React.PointerEvent<HTMLCanvasElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const px = (e.clientX - rect.left) * (SIZE / rect.width);
    const py = (e.clientY - rect.top) * (SIZE / rect.height);
    const [lo, hi] = range;
    const wx = clamp((px / SIZE) * (hi - lo) + lo, lo, hi);
    const wy = clamp(((SIZE - py) / SIZE) * (hi - lo) + lo, lo, hi);
    setStartPos([wx, wy]);
    setRunning(false);
    runnersRef.current.clear();
    for (const n of activeOpts) runnersRef.current.set(n, { pos: [wx, wy], vel: [0, 0], sqAcc: [0, 0], m: [0, 0], v: [0, 0], t: 0, trail: [[wx, wy]] });
    setVersion((v) => v + 1);
  }, [activeOpts, range]);

  const toggleOpt = (name: OptName) => {
    setActiveOpts((prev) => {
      const next = prev.includes(name) ? prev.filter((n) => n !== name) : [...prev, name];
      if (!runnersRef.current.has(name)) {
        runnersRef.current.set(name, { pos: [...startPos], vel: [0, 0], sqAcc: [0, 0], m: [0, 0], v: [0, 0], t: 0, trail: [[...startPos]] });
      }
      return next;
    });
  };

  // Final loss bar chart
  const barCanvasRef = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const canvas = barCanvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const W = 300, H = 80;
    canvas.width = W; canvas.height = H;
    ctx.clearRect(0, 0, W, H);

    const losses = activeOpts.map((n) => {
      const state = runnersRef.current.get(n);
      return { name: n, loss: state ? config.fn(state.pos[0], state.pos[1]) : 0 };
    });
    const maxLoss = Math.max(...losses.map((l) => l.loss), 0.01);
    const barW = Math.min(30, (W - 20) / activeOpts.length - 8);

    losses.forEach((l, i) => {
      const x = 10 + i * (barW + 8);
      const h = Math.max(2, (l.loss / maxLoss) * (H - 25));
      ctx.fillStyle = OPTS[l.name].color;
      ctx.fillRect(x, H - 15 - h, barW, h);
      ctx.fillStyle = isDark ? "rgba(255,255,255,0.6)" : "rgba(0,0,0,0.6)";
      ctx.font = "9px monospace";
      ctx.textAlign = "center";
      ctx.fillText(OPTS[l.name].label, x + barW / 2, H - 3);
      ctx.fillText(l.loss.toFixed(2), x + barW / 2, H - 18 - h);
    });
  }, [version, activeOpts, config.fn, isDark]);

  return (
    <div className="ob-viz">
      <div className="ob-canvas-wrap">
        <canvas ref={canvasRef} width={SIZE} height={SIZE} className="ob-canvas"
          style={{ cursor: "crosshair" }} onPointerDown={handlePointerDown} />
      </div>

      <div className="ob-sidebar">
        <div className="ob-readout">
          <h3>Loss Surface</h3>
          <div className="ob-surface-select">
            {(Object.keys(SURFACES) as Surface[]).map((s) => (
              <button key={s} className={`ob-surface-btn ${s === surface ? "ob-surface-btn--active" : ""}`}
                onClick={() => setSurface(s)}>
                {SURFACES[s].label}
              </button>
            ))}
          </div>
        </div>

        <div className="ob-readout">
          <h3>Optimizers</h3>
          <div className="ob-opt-grid">
            {OPT_NAMES.map((n) => (
              <button key={n}
                className={`ob-opt-btn ${activeOpts.includes(n) ? "ob-opt-btn--active" : ""}`}
                style={{ borderColor: activeOpts.includes(n) ? OPTS[n].color : undefined }}
                onClick={() => toggleOpt(n)}>
                <span className="ob-opt-dot" style={{ background: OPTS[n].color }} />
                {OPTS[n].label}
              </button>
            ))}
          </div>
          {activeOpts.map((n) => (
            <div key={n} className="ob-lr-row">
              <span className="ob-lr-label" style={{ color: OPTS[n].color }}>{OPTS[n].label}</span>
              <input type="range" min={0.0001} max={0.2} step={0.0001}
                value={optLrs[n]}
                onChange={(e) => setOptLrs((prev) => ({ ...prev, [n]: parseFloat(e.target.value) }))}
                className="ob-slider" />
              <span className="ob-lr-value">{optLrs[n].toFixed(4)}</span>
            </div>
          ))}
        </div>

        <div className="ob-readout">
          <h3>Controls</h3>
          <PlaybackControls isRunning={running}
            onPlayPause={() => setRunning((r) => !r)}
            onStep={stepOnce}
            onReset={reset} />
          <p className="ob-info">Click canvas to set start point for all optimizers.</p>
        </div>

        <div className="ob-readout">
          <h3>Final Loss</h3>
          <canvas ref={barCanvasRef} className="ob-bar-canvas" />
        </div>

        <div className="ob-readout">
          <h3>Surface</h3>
          <EquationBlock tex={`f(x,y) = ${config.tex}`} />
        </div>
      </div>
    </div>
  );
}
