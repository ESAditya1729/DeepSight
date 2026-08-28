import { useEffect, useRef } from "react";

/**
 * Runs `callback` once per animation frame while `running` is true,
 * passing the elapsed time in milliseconds since the previous frame.
 */
export function useAnimationFrame(callback: (deltaMs: number) => void, running: boolean): void {
  const callbackRef = useRef(callback);
  callbackRef.current = callback;

  useEffect(() => {
    if (!running) return;

    let frameId: number;
    let lastTime = performance.now();

    const tick = (time: number) => {
      const deltaMs = time - lastTime;
      lastTime = time;
      callbackRef.current(deltaMs);
      frameId = requestAnimationFrame(tick);
    };

    frameId = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frameId);
  }, [running]);
}
