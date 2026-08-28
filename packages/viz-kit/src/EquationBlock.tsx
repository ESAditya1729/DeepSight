import katex from "katex";
import { useMemo } from "react";

interface EquationBlockProps {
  tex: string;
  displayMode?: boolean;
}

export function EquationBlock({ tex, displayMode = true }: EquationBlockProps) {
  const html = useMemo(
    () => katex.renderToString(tex, { throwOnError: false, displayMode }),
    [tex, displayMode],
  );
  return <div className="equation" dangerouslySetInnerHTML={{ __html: html }} />;
}
