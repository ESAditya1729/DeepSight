import "./steps.css";

interface TokenizeStepProps {
  tokens: string[];
}

export default function TokenizeStep({ tokens }: TokenizeStepProps) {
  return (
    <div className="pl-step-content">
      <div className="pl-step-left">
        <div className="pl-readout">
          <h3>Tokenization</h3>
          <p className="pl-info">
            The input text is split into individual <strong>tokens</strong> (words).
            Each token gets a position index. This is the first step in any NLP pipeline —
            turning raw text into structured units the model can process.
          </p>
        </div>
        <div className="pl-readout">
          <h3>Tokens ({tokens.length})</h3>
          <div className="pl-tokens-grid">
            {tokens.map((token, i) => (
              <div key={i} className="pl-token-card">
                <span className="pl-token-index">{i}</span>
                <span className="pl-token-word">{token}</span>
                <span className="pl-token-len">{token.length} chars</span>
              </div>
            ))}
          </div>
        </div>
      </div>
      <div className="pl-step-right">
        <div className="pl-readout">
          <h3>Token Sequence</h3>
          <div className="pl-token-sequence">
            {tokens.map((token, i) => (
              <span key={i} className="pl-token-badge">
                <span className="pl-token-badge-idx">{i}</span>
                {token}
              </span>
            ))}
          </div>
        </div>
        <div className="pl-readout">
          <h3>How tokenization works</h3>
          <p className="pl-info">
            Simple whitespace splitting: <code>"The cat" → ["The", "cat"]</code>.
            Real systems use <strong>BPE</strong> (Byte Pair Encoding) or <strong>WordPiece</strong> to handle
            subwords, punctuation, and rare words. Each token maps to an integer ID in the vocabulary.
          </p>
          <div className="pl-formula">
            <code>tokenize("The cat sat") → ["The", "cat", "sat"]</code>
          </div>
        </div>
      </div>
    </div>
  );
}
