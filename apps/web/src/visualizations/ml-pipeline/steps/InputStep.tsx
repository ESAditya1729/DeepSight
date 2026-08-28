import { useState } from "react";

const PRESETS = [
  { label: "The cat sat on the ___", text: "The cat sat on the" },
  { label: "She went to the bank to ___", text: "She went to the bank to" },
  { label: "The quick brown fox jumps ___", text: "The quick brown fox jumps" },
  { label: "A king and queen rule ___", text: "A king and queen rule" },
  { label: "The happy child played ___", text: "The happy child played" },
];

interface InputStepProps {
  text: string;
  onTextChange: (text: string) => void;
  onPreset: (text: string) => void;
}

export default function InputStep({ text, onTextChange, onPreset }: InputStepProps) {
  const [customText, setCustomText] = useState(text);

  return (
    <div className="pl-step-content">
      <div className="pl-step-left">
        <div className="pl-readout">
          <h3>Input Text</h3>
          <p className="pl-info">
            Type a sentence (without the last word) or choose a preset.
            The model will try to <strong>predict what comes next</strong> based on the context.
          </p>
          <textarea
            className="pl-textarea"
            value={customText}
            onChange={(e) => setCustomText(e.target.value)}
            onBlur={() => onTextChange(customText)}
            onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); onTextChange(customText); } }}
            placeholder="Type a sentence without the last word..."
            rows={3}
          />
          <button className="pl-apply-btn" onClick={() => onTextChange(customText)}>
            Apply
          </button>
        </div>
        <div className="pl-readout">
          <h3>Preset Sentences</h3>
          <div className="pl-preset-list">
            {PRESETS.map((p) => (
              <button key={p.label}
                className={`pl-preset-btn ${text === p.text ? "pl-preset-btn--active" : ""}`}
                onClick={() => { setCustomText(p.text); onPreset(p.text); }}>
                {p.label}
              </button>
            ))}
          </div>
        </div>
      </div>
      <div className="pl-step-right">
        <div className="pl-readout">
          <h3>Preview</h3>
          <div className="pl-input-preview">
            {text.split(/\s+/).filter(Boolean).map((word, i) => (
              <span key={i} className="pl-input-token">{word}</span>
            ))}
            <span className="pl-input-token pl-input-token--mask">___</span>
          </div>
          <p className="pl-info" style={{ marginTop: "0.5rem" }}>
            {text.split(/\s+/).filter(Boolean).length} tokens — predicting the next word
          </p>
        </div>
        <div className="pl-readout">
          <h3>What happens next?</h3>
          <div className="pl-flow-hint">
            <div className="pl-flow-step">
              <span className="pl-flow-icon">1</span>
              <span><strong>Tokenize</strong> — split into individual words</span>
            </div>
            <div className="pl-flow-step">
              <span className="pl-flow-icon">2</span>
              <span><strong>Embed</strong> — map words to vectors</span>
            </div>
            <div className="pl-flow-step">
              <span className="pl-flow-icon">3</span>
              <span><strong>Similarity</strong> — compare all pairs</span>
            </div>
            <div className="pl-flow-step">
              <span className="pl-flow-icon">4</span>
              <span><strong>Attention</strong> — learn what matters</span>
            </div>
            <div className="pl-flow-step">
              <span className="pl-flow-icon">5</span>
              <span><strong>Predict</strong> — guess the next word</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
