import { useState, useMemo } from "react";
import InputStep from "./steps/InputStep";
import TokenizeStep from "./steps/TokenizeStep";
import EmbedStep from "./steps/EmbedStep";
import SimilarityStep from "./steps/SimilarityStep";
import AttentionStep from "./steps/AttentionStep";
import PredictStep from "./steps/PredictStep";
import {
  generateEmbeddings,
  computeSimilarityMatrix,
  computeAttentionBreakdown,
  attentionContext,
  type WordEmbedding,
  type AttentionBreakdown,
} from "./pipelineCore";
import "./MLPipelineViz.css";

export type { WordEmbedding } from "./pipelineCore";

export interface PipelineData {
  rawText: string;
  tokens: string[];
  embeddings: WordEmbedding[];
  similarityMatrix: number[][];
  attention: AttentionBreakdown;
  context: [number, number];
}

const STEPS = ["Input", "Tokenize", "Embed", "Similarity", "Attention", "Predict"] as const;
type StepIndex = 0 | 1 | 2 | 3 | 4 | 5;

export default function MLPipelineViz() {
  const [currentStep, setCurrentStep] = useState<StepIndex>(0);
  const [seed, setSeed] = useState(42);
  const [temperature, setTemperature] = useState(1.0);

  const [rawText, setRawText] = useState("The cat sat on the mat");
  const [tokens, setTokens] = useState<string[]>(["The", "cat", "sat", "on", "the", "mat"]);

  // Auto-compute downstream data when upstream changes
  const computed = useMemo(() => {
    if (tokens.length === 0) return null;
    const emb = generateEmbeddings(tokens, seed);
    const sim = computeSimilarityMatrix(emb);
    const attention = computeAttentionBreakdown(emb, temperature);
    return { embeddings: emb, similarityMatrix: sim, attention };
  }, [tokens, seed, temperature]);

  const handleTextChange = (text: string) => {
    setRawText(text);
    const newTokens = text.split(/\s+/).filter((t) => t.length > 0);
    setTokens(newTokens);
  };

  const handlePreset = (text: string) => {
    setRawText(text);
    setTokens(text.split(/\s+/).filter((t) => t.length > 0));
  };

  const canGoNext = () => currentStep < STEPS.length - 1;
  const canGoPrev = () => currentStep > 0;
  const goNext = () => { if (canGoNext()) setCurrentStep((s) => (s + 1) as StepIndex); };
  const goPrev = () => { if (canGoPrev()) setCurrentStep((s) => (s - 1) as StepIndex); };

  const renderStep = () => {
    if (!computed) return <div className="pl-step-content"><p className="pl-info">No tokens to process. Go back to the Input step.</p></div>;
    switch (currentStep) {
      case 0: return <InputStep text={rawText} onTextChange={handleTextChange} onPreset={handlePreset} />;
      case 1: return <TokenizeStep tokens={tokens} />;
      case 2: return <EmbedStep embeddings={computed.embeddings} seed={seed} onSeedChange={setSeed} />;
      case 3: return <SimilarityStep embeddings={computed.embeddings} matrix={computed.similarityMatrix} />;
      case 4:
        return (
          <AttentionStep
            tokens={tokens}
            embeddings={computed.embeddings}
            breakdown={computed.attention}
            temperature={temperature}
            onTemperatureChange={setTemperature}
          />
        );
      case 5:
        return (
          <PredictStep
            embeddings={computed.embeddings}
            tokens={tokens}
            seed={seed}
            context={attentionContext(computed.attention)}
          />
        );
    }
  };

  const data: PipelineData | null = computed && {
    rawText,
    tokens,
    embeddings: computed.embeddings,
    similarityMatrix: computed.similarityMatrix,
    attention: computed.attention,
    context: attentionContext(computed.attention),
  };

  return (
    <div className="pl-viz">
      {/* Stepper */}
      <div className="pl-stepper">
        {STEPS.map((label, i) => (
          <div key={i} className="pl-step-wrapper">
            <button
              className={`pl-step ${i === currentStep ? "pl-step--active" : ""} ${i < currentStep ? "pl-step--done" : ""}`}
              onClick={() => setCurrentStep(i as StepIndex)}>
              <span className="pl-step-num">{i < currentStep ? "✓" : i + 1}</span>
              <span className="pl-step-label">{label}</span>
            </button>
            {i < STEPS.length - 1 && <div className={`pl-step-connector ${i < currentStep ? "pl-step-connector--done" : ""}`} />}
          </div>
        ))}
      </div>

      {/* Step content */}
      <div className="pl-content">
        {renderStep()}
      </div>

      {/* Navigation */}
      <div className="pl-nav">
        <button className="pl-nav-btn" disabled={!canGoPrev()} onClick={goPrev}>
          ← Previous
        </button>
        <span className="pl-nav-info">Step {currentStep + 1} of {STEPS.length}</span>
        <button className="pl-nav-btn" disabled={!canGoNext()} onClick={goNext}>
          Next →
        </button>
      </div>

      {/* Data preview */}
      <div className="pl-data-preview">
        <details className="pl-data-details">
          <summary className="pl-data-summary">View pipeline data</summary>
          <pre className="pl-data-pre">
            {data ? JSON.stringify({
              tokens: data.tokens,
              embeddings: data.embeddings.slice(0, 3).map((e) => `${e.word}: [${e.x.toFixed(2)}, ${e.y.toFixed(2)}]`),
              similarity: data.similarityMatrix.length > 0 ? `${data.similarityMatrix.length}×${data.similarityMatrix[0].length} matrix` : "empty",
              attention: `${data.attention.scores.length}×${data.attention.scores.length}`,
              attentionWeights: data.attention.weights.map((row) => row.map((v) => v.toFixed(2))),
              output: data.attention.output.map((row) => `[${row.map((v) => v.toFixed(2)).join(", ")}]`),
              context: `[${data.context[0].toFixed(2)}, ${data.context[1].toFixed(2)}]`,
            }, null, 2) : "no tokens"}
          </pre>
        </details>
      </div>
    </div>
  );
}