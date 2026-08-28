import { useState, useMemo, useRef, useCallback } from "react";
import { cosineSimilarity } from "@ml-visual-lab/ml-core";
import { EquationBlock } from "@ml-visual-lab/viz-kit";
import "./EmbeddingPlaygroundViz.css";

interface WordEmbedding {
  word: string;
  x: number;
  y: number;
  category: string;
}

const SIZE = 480;
const PADDING = 40;
const CANVAS_USABLE = SIZE - PADDING * 2;

const CATEGORIES: Record<string, { color: string; label: string }> = {
  royalty: { color: "#7c3aed", label: "Royalty" },
  gender: { color: "#ec4899", label: "Gender" },
  animals: { color: "#f97316", label: "Animals" },
  colors: { color: "#06b6d4", label: "Colors" },
  size: { color: "#22c55e", label: "Size" },
  speed: { color: "#eab308", label: "Speed" },
  emotion: { color: "#ef4444", label: "Emotion" },
};

const WORD_DATA: WordEmbedding[] = [
  { word: "king", x: 2.8, y: 3.2, category: "royalty" },
  { word: "queen", x: 3.2, y: -2.8, category: "royalty" },
  { word: "prince", x: 2.2, y: 2.5, category: "royalty" },
  { word: "princess", x: 2.6, y: -2.1, category: "royalty" },
  { word: "man", x: -2.0, y: 3.0, category: "gender" },
  { word: "woman", x: -1.6, y: -3.0, category: "gender" },
  { word: "boy", x: -3.0, y: 2.2, category: "gender" },
  { word: "girl", x: -2.6, y: -2.2, category: "gender" },
  { word: "father", x: -1.0, y: 3.5, category: "gender" },
  { word: "mother", x: -0.6, y: -3.5, category: "gender" },
  { word: "husband", x: -0.2, y: 2.8, category: "gender" },
  { word: "wife", x: 0.2, y: -2.8, category: "gender" },
  { word: "brother", x: -3.5, y: 3.2, category: "gender" },
  { word: "sister", x: -3.1, y: -3.2, category: "gender" },
  { word: "dog", x: 4.0, y: 0.5, category: "animals" },
  { word: "cat", x: 4.5, y: -0.5, category: "animals" },
  { word: "puppy", x: 3.5, y: 1.2, category: "animals" },
  { word: "kitten", x: 4.2, y: -1.2, category: "animals" },
  { word: "horse", x: 5.0, y: 0.0, category: "animals" },
  { word: "lion", x: 5.5, y: 0.8, category: "animals" },
  { word: "tiger", x: 5.8, y: -0.3, category: "animals" },
  { word: "red", x: -4.0, y: -0.5, category: "colors" },
  { word: "blue", x: -4.5, y: 0.5, category: "colors" },
  { word: "green", x: -4.2, y: 0.0, category: "colors" },
  { word: "yellow", x: -3.8, y: -1.0, category: "colors" },
  { word: "orange", x: -4.8, y: -0.3, category: "colors" },
  { word: "purple", x: -4.3, y: 0.8, category: "colors" },
  { word: "black", x: -5.0, y: -0.8, category: "colors" },
  { word: "white", x: -5.2, y: 0.3, category: "colors" },
  { word: "big", x: 0.5, y: 1.5, category: "size" },
  { word: "small", x: 0.3, y: -1.5, category: "size" },
  { word: "large", x: 0.8, y: 1.8, category: "size" },
  { word: "tiny", x: 0.1, y: -1.8, category: "size" },
  { word: "huge", x: 1.0, y: 2.0, category: "size" },
  { word: "little", x: -0.1, y: -2.0, category: "size" },
  { word: "fast", x: -0.5, y: 0.8, category: "speed" },
  { word: "slow", x: -0.3, y: -0.8, category: "speed" },
  { word: "quick", x: -0.8, y: 1.0, category: "speed" },
  { word: "rapid", x: -1.0, y: 1.2, category: "speed" },
  { word: "happy", x: 1.5, y: -0.5, category: "emotion" },
  { word: "sad", x: 1.3, y: -1.5, category: "emotion" },
  { word: "joyful", x: 1.8, y: -0.2, category: "emotion" },
  { word: "angry", x: 1.0, y: -1.8, category: "emotion" },
];

const ALL_WORDS = WORD_DATA.map((w) => w.word);
const WORD_MAP = new Map(WORD_DATA.map((w) => [w.word, w]));

function getNeighbors(word: string, embeddings: WordEmbedding[], n: number): { word: string; sim: number; category: string }[] {
  const target = WORD_MAP.get(word);
  if (!target) return [];
  return embeddings
    .filter((e) => e.word !== word)
    .map((e) => ({ word: e.word, sim: cosineSimilarity([target.x, target.y], [e.x, e.y]), category: e.category }))
    .sort((a, b) => b.sim - a.sim)
    .slice(0, n);
}

function findClosestWord(result: [number, number], embeddings: WordEmbedding[]): string {
  let best = embeddings[0]?.word ?? "";
  let bestD = Infinity;
  for (const e of embeddings) {
    const d = Math.hypot(result[0] - e.x, result[1] - e.y);
    if (d < bestD) { bestD = d; best = e.word; }
  }
  return best;
}

interface QueryOp { op: "+" | "-"; word: string; }

export default function EmbeddingPlaygroundViz() {
  const svgRef = useRef<SVGSVGElement>(null);
  const [selectedWord, setSelectedWord] = useState<string | null>("king");
  const [hoveredWord, setHoveredWord] = useState<string | null>(null);
  const [neighborCount, setNeighborCount] = useState(5);
  const [queryOps, setQueryOps] = useState<QueryOp[]>([{ op: "-", word: "man" }, { op: "+", word: "woman" }]);
  const [highlightCategory, setHighlightCategory] = useState<string | null>(null);
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState<[number, number]>([0, 0]);
  const isPanning = useRef(false);
  const panStart = useRef<[number, number]>([0, 0]);

  const displayWord = hoveredWord ?? selectedWord;
  const neighbors = useMemo(
    () => (displayWord ? getNeighbors(displayWord, WORD_DATA, neighborCount) : []),
    [displayWord, neighborCount],
  );
  const neighborSet = useMemo(() => new Set(neighbors.map((n) => n.word)), [neighbors]);

  const queryResult = useMemo(() => {
    if (!selectedWord) return null;
    const base = WORD_MAP.get(selectedWord);
    if (!base) return null;
    let rx = base.x, ry = base.y;
    for (const { op, word } of queryOps) {
      const target = WORD_MAP.get(word);
      if (!target) continue;
      if (op === "+") { rx += target.x; ry += target.y; }
      else { rx -= target.x; ry -= target.y; }
    }
    return { x: rx, y: ry, closest: findClosestWord([rx, ry], WORD_DATA) };
  }, [selectedWord, queryOps]);

  function dataToScreen(wx: number, wy: number): [number, number] {
    const xMin = -6, xMax = 7, yMin = -4, yMax = 4;
    const sx = PADDING + ((wx - xMin) / (xMax - xMin)) * CANVAS_USABLE;
    const sy = PADDING + ((yMax - wy) / (yMax - yMin)) * CANVAS_USABLE;
    return [(sx - SIZE / 2) * zoom + SIZE / 2 + pan[0], (sy - SIZE / 2) * zoom + SIZE / 2 + pan[1]];
  }

  const handleWheel = useCallback((e: React.WheelEvent) => {
    e.preventDefault();
    const delta = e.deltaY > 0 ? 0.9 : 1.1;
    setZoom((z) => Math.max(0.5, Math.min(3, z * delta)));
  }, []);

  const handlePointerDown = useCallback((e: React.PointerEvent<SVGSVGElement>) => {
    if (e.button === 1 || (e.button === 0 && e.shiftKey)) {
      isPanning.current = true;
      panStart.current = [e.clientX - pan[0], e.clientY - pan[1]];
      e.currentTarget.setPointerCapture(e.pointerId);
    }
  }, [pan]);

  const handlePointerMove = useCallback((e: React.PointerEvent<SVGSVGElement>) => {
    if (!isPanning.current) return;
    setPan([e.clientX - panStart.current[0], e.clientY - panStart.current[1]]);
  }, []);

  const handlePointerUp = useCallback(() => { isPanning.current = false; }, []);

  const addQueryOp = () => {
    setQueryOps((prev) => [...prev, { op: "+", word: "happy" }]);
  };

  const removeQueryOp = (idx: number) => {
    setQueryOps((prev) => prev.filter((_, i) => i !== idx));
  };

  return (
    <div className="ep-viz">
      <div className="ep-canvas-wrap">
        <svg ref={svgRef} width={SIZE} height={SIZE} className="ep-canvas"
          onWheel={handleWheel} onPointerDown={handlePointerDown}
          onPointerMove={handlePointerMove} onPointerUp={handlePointerUp}>

          {/* Category legend (inline) */}
          {Object.entries(CATEGORIES).map(([cat, { color, label }]) => {
            const words = WORD_DATA.filter((w) => w.category === cat);
            const avgX = words.reduce((s, w) => s + w.x, 0) / words.length;
            const avgY = words.reduce((s, w) => s + w.y, 0) / words.length;
            const [lx, ly] = dataToScreen(avgX, avgY);
            return (
              <text key={cat} x={lx} y={ly - 18 * zoom} textAnchor="middle"
                fontSize={9 * zoom} fill={color} opacity={0.6} fontWeight={600}>
                {label}
              </text>
            );
          })}

          {/* Words */}
          {WORD_DATA.map((w) => {
            const [sx, sy] = dataToScreen(w.x, w.y);
            const isSelected = w.word === displayWord;
            const isNeighbor = neighborSet.has(w.word) && displayWord !== null;
            const isQueryResult = queryResult?.closest === w.word;
            const catColor = CATEGORIES[w.category]?.color ?? "var(--text)";
            const dimmed = highlightCategory && w.category !== highlightCategory && !isSelected;
            return (
              <g key={w.word} opacity={dimmed ? 0.15 : 1} style={{ transition: "opacity 0.2s" }}>
                {isNeighbor && !isSelected && (
                  <line
                    x1={dataToScreen(WORD_MAP.get(displayWord!)!.x, WORD_MAP.get(displayWord!)!.y)[0]}
                    y1={dataToScreen(WORD_MAP.get(displayWord!)!.x, WORD_MAP.get(displayWord!)!.y)[1]}
                    x2={sx} y2={sy} stroke="var(--accent)" strokeWidth={1} strokeOpacity={0.3} />
                )}
                <circle cx={sx} cy={sy}
                  r={isSelected ? 7 * zoom : isQueryResult ? 6 * zoom : isNeighbor ? 5 * zoom : 3.5 * zoom}
                  fill={isSelected ? "#ef4444" : isQueryResult ? "#facc15" : isNeighbor ? "var(--accent)" : catColor}
                  opacity={isSelected || isNeighbor || isQueryResult ? 1 : 0.6}
                  style={{ cursor: "pointer", transition: "r 0.15s, fill 0.15s" }}
                  onPointerEnter={() => setHoveredWord(w.word)}
                  onPointerLeave={() => setHoveredWord(null)}
                  onClick={() => setSelectedWord(w.word)} />
                {(isSelected || isNeighbor || isQueryResult) && (
                  <text x={sx} y={sy - 10 * zoom} textAnchor="middle"
                    fontSize={(isSelected ? 12 : 10) * zoom} fontWeight={isSelected ? 700 : 600}
                    fill={isSelected ? "#ef4444" : isQueryResult ? "#facc15" : "var(--accent)"}
                    fontFamily="var(--mono)">{w.word}</text>
                )}
              </g>
            );
          })}

          {/* Query result */}
          {queryResult && selectedWord && (
            <g>
              <line
                x1={dataToScreen(WORD_MAP.get(selectedWord)!.x, WORD_MAP.get(selectedWord)!.y)[0]}
                y1={dataToScreen(WORD_MAP.get(selectedWord)!.x, WORD_MAP.get(selectedWord)!.y)[1]}
                x2={dataToScreen(queryResult.x, queryResult.y)[0]}
                y2={dataToScreen(queryResult.x, queryResult.y)[1]}
                stroke="#facc15" strokeWidth={2} strokeDasharray="6 4" strokeOpacity={0.6} />
              <circle cx={dataToScreen(queryResult.x, queryResult.y)[0]}
                cy={dataToScreen(queryResult.x, queryResult.y)[1]}
                r={10 * zoom} fill="none" stroke="#facc15" strokeWidth={2} strokeDasharray="4 3" />
            </g>
          )}
        </svg>
      </div>

      <div className="ep-sidebar">
        <div className="ep-readout">
          <h3>Selected Word</h3>
          <p className="ep-info">
            {displayWord ? (
              <>Click any word to explore. Currently: <strong style={{ fontFamily: "var(--mono)" }}>{displayWord}</strong>
                {highlightCategory && <> in <span style={{ color: CATEGORIES[highlightCategory]?.color }}>{CATEGORIES[highlightCategory]?.label}</span></>}
              </>
            ) : "Hover or click a word to explore."}
          </p>
        </div>

        {/* Category filter */}
        <div className="ep-readout">
          <h3>Filter by Category</h3>
          <div className="ep-categories">
            <button className={`ep-cat-btn ${!highlightCategory ? "ep-cat-btn--active" : ""}`}
              onClick={() => setHighlightCategory(null)}>All</button>
            {Object.entries(CATEGORIES).map(([cat, { color, label }]) => (
              <button key={cat}
                className={`ep-cat-btn ${highlightCategory === cat ? "ep-cat-btn--active" : ""}`}
                style={{ borderColor: highlightCategory === cat ? color : undefined, color: highlightCategory === cat ? color : undefined }}
                onClick={() => setHighlightCategory(highlightCategory === cat ? null : cat)}>
                <span className="ep-cat-dot" style={{ background: color }} />
                {label}
              </button>
            ))}
          </div>
        </div>

        {displayWord && (
          <div className="ep-readout">
            <h3>Nearest Neighbors</h3>
            <div className="ep-neighbors">
              {neighbors.map((n) => (
                <div key={n.word} className="ep-neighbor">
                  <span className="ep-neighbor-dot" style={{ background: CATEGORIES[n.category]?.color ?? "#888" }} />
                  <span className="ep-neighbor-word">{n.word}</span>
                  <div className="ep-neighbor-bar">
                    <div className="ep-neighbor-bar-fill" style={{ width: `${Math.max(0, n.sim) * 100}%`, background: CATEGORIES[n.category]?.color ?? "var(--accent)" }} />
                  </div>
                  <span className="ep-neighbor-sim">{n.sim.toFixed(3)}</span>
                </div>
              ))}
            </div>
          </div>
        )}

        <div className="ep-readout">
          <h3>Vector Arithmetic</h3>
          <p className="ep-info" style={{ marginBottom: "0.5rem" }}>
            Try: <strong>king - man + woman = ?</strong>
          </p>
          <div className="ep-query-section">
            <div className="ep-query-row">
              <span className="ep-query-op">=</span>
              <select className="ep-query-select" value={selectedWord ?? ""}
                onChange={(e) => setSelectedWord(e.target.value)}>
                {ALL_WORDS.map((w) => <option key={w} value={w}>{w}</option>)}
              </select>
            </div>
            {queryOps.map((q, i) => (
              <div key={i} className="ep-query-row">
                <select className="ep-query-select" value={q.op}
                  onChange={(e) => {
                    const next = [...queryOps];
                    next[i] = { ...next[i], op: e.target.value as "+" | "-" };
                    setQueryOps(next);
                  }}>
                  <option value="+">+</option>
                  <option value="-">-</option>
                </select>
                <select className="ep-query-select" value={q.word}
                  onChange={(e) => {
                    const next = [...queryOps];
                    next[i] = { ...next[i], word: e.target.value };
                    setQueryOps(next);
                  }}>
                  {ALL_WORDS.map((w) => <option key={w} value={w}>{w}</option>)}
                </select>
                {queryOps.length > 1 && (
                  <button className="ep-remove-btn" onClick={() => removeQueryOp(i)}>×</button>
                )}
              </div>
            ))}
            <button className="ep-add-op-btn" onClick={addQueryOp}>+ Add operation</button>
            {queryResult && (
              <div className="ep-query-result">
                <div className="ep-query-result-label">Closest word</div>
                <span className="ep-query-result-word">{queryResult.closest}</span>
              </div>
            )}
          </div>
        </div>

        <div className="ep-readout">
          <h3>Controls</h3>
          <div className="ep-controls">
            <label className="slider">
              <span className="slider-label">Neighbors</span>
              <input type="range" min={1} max={10} value={neighborCount}
                onChange={(e) => setNeighborCount(Number(e.target.value))} />
              <span className="slider-value">{neighborCount}</span>
            </label>
            <div className="ep-zoom-controls">
              <button className="ep-zoom-btn" onClick={() => setZoom((z) => Math.min(3, z * 1.2))}>+</button>
              <span className="ep-zoom-label">{Math.round(zoom * 100)}%</span>
              <button className="ep-zoom-btn" onClick={() => setZoom((z) => Math.max(0.5, z / 1.2))}>-</button>
              <button className="ep-zoom-btn" onClick={() => { setZoom(1); setPan([0, 0]); }}>Reset</button>
            </div>
          </div>
        </div>

        <div className="ep-readout">
          <h3>About</h3>
          <p className="ep-info">
            Words close in embedding space share similar meaning.
            <strong> king - man + woman</strong> lands near "queen" — the embedding captures gender as a dimension.
            Shift+drag to pan, scroll to zoom.
          </p>
          <EquationBlock tex={`\\vec{\\text{result}} = \\vec{\\text{king}} - \\vec{\\text{man}} + \\vec{\\text{woman}}`} />
        </div>
      </div>
    </div>
  );
}
