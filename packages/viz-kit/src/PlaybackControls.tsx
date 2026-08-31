import { useEffect } from "react";
import { PLAYBACK_TOGGLE_EVENT } from "./useKeyDown.js";

interface PlaybackControlsProps {
  isRunning: boolean;
  onPlayPause: () => void;
  onStep: () => void;
  onReset: () => void;
}

export function PlaybackControls({ isRunning, onPlayPause, onStep, onReset }: PlaybackControlsProps) {
  useEffect(() => {
    const onToggle = () => onPlayPause();
    window.addEventListener(PLAYBACK_TOGGLE_EVENT, onToggle);
    return () => window.removeEventListener(PLAYBACK_TOGGLE_EVENT, onToggle);
  }, [onPlayPause]);
  return (
    <div className="playback-controls">
      <button type="button" className="playback-primary" onClick={onPlayPause}>
        {isRunning ? "Pause" : "Run"}
      </button>
      <button type="button" onClick={onStep} disabled={isRunning}>
        Step
      </button>
      <button type="button" onClick={onReset}>
        Reset
      </button>
    </div>
  );
}
