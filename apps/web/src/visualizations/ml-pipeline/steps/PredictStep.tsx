import { useMemo } from "react";
import { type WordEmbedding } from "../MLPipelineViz";
import { positionVocab, predictNextWord, contextVector, VOCAB_WORDS } from "../pipelineCore";
import { categoryColor } from "../embeddingSpace";
import "./steps.css";

interface PredictStepProps {
  embeddings: WordEmbedding[];
  tokens: string[];
  seed: number;
}

const CATEGORY_COLORS: Record<string, string> = {
  Surface: "#7c3aed",
  Place: "#2563eb",
  Animal: "#059669",
  Action: "#d97706",
  Size: "#ec4899",
  Color: "#dc2626",
  Emotion: "#f43f5e",
  Person: "#f97316",
  Speed: "#06b6d4",
  Money: "#8b5cf6",
};

export default function PredictStep({ embeddings, tokens, seed }: PredictStepProps) {
  // Position the vocabulary with the SAME seed and the SAME embedding space the
  // Embed step uses, so predictions are consistent with what's shown upstream.
  const vocab = useMemo(() => positionVocab(seed), [seed]);

  const predictions = useMemo(() => {
    if (embeddings.length === 0) return [];
    return predictNextWord(contextVector(embeddings), vocab, tokens);
  }, [embeddings, tokens, vocab]);

  const inputPreview = tokens.join(" ") + " ___";
  const topPrediction = predictions[0];
  const vocabularyPreview = VOCAB_WORDS.slice(0, 12);

  return (
    <div className="pl-step-content">
      <div className="pl-step-left">
        <div className="pl-readout">
          <h3>Next Word Prediction</h3>
          <p className="pl-info">
            The model looks at all the previous words and tries to guess <strong>what comes next</strong>.
            It uses the context from the attention step to find the word that best fits
            the meaning of the sentence so far.
          </p>
        </div>
        <div className="pl-canvas-wrap">
          <svg width={380} height={320} className="pl-canvas">
            {/* Input sentence */}
            <text x={190} y={25} textAnchor="middle" fontSize={10} fontFamily="var(--mono)"
              fontWeight={600} fill="var(--text)">
              Input:
            </text>
            <text x={190} y={42} textAnchor="middle" fontSize={12} fontFamily="var(--mono)"
              fontWeight={700} fill="var(--text-h)">
              {inputPreview}
            </text>

            {/* Prediction bars */}
            {predictions.map((pred, i) => {
              const x = 20;
              const y = 65 + i * 30;
              const barWidth = pred.score * 280;
              const color = CATEGORY_COLORS[pred.category] || categoryColor(pred.word);

              return (
                <g key={i}>
                  <text x={x} y={y + 12} fontSize={10} fontFamily="var(--mono)"
                    fontWeight={600} fill={color}>
                    {pred.word}
                  </text>
                  <rect x={x + 60} y={y} width={barWidth} height={18} rx={4}
                    fill={color} opacity={0.2} />
                  <rect x={x + 60} y={y} width={barWidth} height={18} rx={4}
                    fill={color} opacity={0.7} />
                  <text x={x + 65 + barWidth} y={y + 13} fontSize={9} fontFamily="var(--mono)"
                    fontWeight={600} fill={color}>
                    {(pred.score * 100).toFixed(0)}%
                  </text>
                  <text x={350} y={y + 12} fontSize={8} fontFamily="var(--mono)"
                    fill="var(--text)" opacity={0.6}>
                    {pred.category}
                  </text>
                </g>
              );
            })}

            {/* Top prediction highlight */}
            {topPrediction && (
              <g>
                <rect x={20} y={310} width={340} height={0} rx={0} fill="none" />
                <text x={190} y={310} textAnchor="middle" fontSize={11} fontFamily="var(--mono)"
                  fontWeight={700} fill={CATEGORY_COLORS[topPrediction.category] || categoryColor(topPrediction.word)}>
                  Most likely: &quot;{topPrediction.word}&quot;
                </text>
              </g>
            )}
          </svg>
        </div>
      </div>
      <div className="pl-step-right">
        <div className="pl-readout">
          <h3>How it works</h3>
          <p className="pl-info" style={{ marginBottom: "0.5rem" }}>
            This is a simplified version of how <strong>GPT</strong> and other language models work.
            Here&apos;s the process:
          </p>
          <ol className="pl-steps-list">
            <li>
              <strong>Context vector</strong> — combine the word embeddings into one &quot;meaning&quot; vector.
              We give the most weight to the last important word.
            </li>
            <li>
              <strong>Compare against the vocabulary</strong> — measure how close the context is to every
              possible next word in the vocabulary below.
            </li>
            <li>
              <strong>Rank by closeness</strong> — the closest word gets the highest score. This is like
              asking &quot;what word fits best here?&quot;
            </li>
          </ol>
        </div>
        <div className="pl-readout">
          <h3>The vocabulary ({VOCAB_WORDS.length} words)</h3>
          <div className="pl-vocab-list">
            {vocabularyPreview.map((v) => (
              <span key={v.word} className="pl-vocab-chip" style={{
                borderColor: CATEGORY_COLORS[v.category] || categoryColor(v.word),
                color: CATEGORY_COLORS[v.category] || categoryColor(v.word),
              }}>
                {v.word}
              </span>
            ))}
            {VOCAB_WORDS.length > vocabularyPreview.length && (
              <span className="pl-vocab-more">+{VOCAB_WORDS.length - vocabularyPreview.length} more</span>
            )}
          </div>
          <p className="pl-info" style={{ marginTop: "0.5rem" }}>
            Real language models have a vocabulary of <strong>50,000+ words</strong>. We use a small toy set
            so you can see how scoring works.
          </p>
        </div>
        <div className="pl-readout">
          <h3>Try it</h3>
          <p className="pl-info">
            Go back to the <strong>Input</strong> step and try different sentences:
          </p>
          <div className="pl-predict-examples">
            <code>&quot;The cat sat on the ___&quot;</code>
            <code>&quot;She went to the bank to ___&quot;</code>
            <code>&quot;The happy child played ___&quot;</code>
          </div>
          <p className="pl-info" style={{ marginTop: "0.5rem" }}>
            Watch how the predictions change based on the context!
          </p>
        </div>
      </div>
    </div>
  );
}
