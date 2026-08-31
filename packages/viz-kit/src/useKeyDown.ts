import { useEffect, useRef } from "react";

/**
 * Attaches a window-level `keydown` listener. The handler stays fresh in a
 * ref, so it never needs to be re-registered across re-renders.
 */
export function useKeyDown(handler: (event: KeyboardEvent) => void): void {
  const handlerRef = useRef(handler);
  handlerRef.current = handler;

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => handlerRef.current(event);
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, []);
}

/** Window event used to toggle playback from a global keyboard shortcut. */
export const PLAYBACK_TOGGLE_EVENT = "deepsight:toggle-play";