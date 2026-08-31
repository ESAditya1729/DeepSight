import { useCallback, useEffect, useSyncExternalStore } from "react";

/**
 * Lightweight, framework-agnostic progress tracking persisted to localStorage.
 *
 * Tracks which visualizations a visitor has opened (first/last visited time).
 * The store is a tiny external store (module-level snapshot + subscriber set) so
 * any component — Sidebar, Home, etc. — observes updates without prop drilling.
 */

const STORAGE_KEY = "deepsight-progress";

export interface ProgressEntry {
  firstVisited: number;
  lastVisited: number;
}

export type ProgressMap = Record<string, ProgressEntry>;

function read(): ProgressMap {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? (JSON.parse(raw) as ProgressMap) : {};
  } catch {
    return {};
  }
}

let snapshot: ProgressMap | null = null;
const listeners = new Set<() => void>();

function emit() {
  for (const listener of listeners) listener();
}

function getSnapshot(): ProgressMap {
  if (!snapshot) snapshot = read();
  return snapshot;
}

function setMap(next: ProgressMap) {
  snapshot = next;
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(next));
  } catch {
    // Storage may be unavailable (private mode / full quota) — degrade gracefully.
  }
  emit();
}

function subscribe(listener: () => void): () => void {
  listeners.add(listener);
  const onStorage = (event: StorageEvent) => {
    if (event.key === STORAGE_KEY) {
      snapshot = null;
      emit();
    }
  };
  window.addEventListener("storage", onStorage);
  return () => {
    listeners.delete(listener);
    window.removeEventListener("storage", onStorage);
  };
}

/** Record that a visualization was opened (idempotent-ish: bumps lastVisited). */
export function recordVisit(slug: string): void {
  const map = getSnapshot();
  const now = Date.now();
  const entry = map[slug];
  setMap({
    ...map,
    [slug]: entry ? { ...entry, lastVisited: now } : { firstVisited: now, lastVisited: now },
  });
}

export function hasVisited(slug: string): boolean {
  return Object.prototype.hasOwnProperty.call(getSnapshot(), slug);
}

/**
 * Subscribe to progress, and when `slug` is provided, record that visit once
 * (on mount / whenever the slug changes). Returns the live progress map plus a
 * shorthand predicate for whether a given visualization has been visited.
 */
export function useProgress(slug?: string) {
  const progress = useSyncExternalStore(subscribe, getSnapshot);

  useEffect(() => {
    if (slug) recordVisit(slug);
  }, [slug]);

  const isVisited = useCallback(
    (s: string) => Object.prototype.hasOwnProperty.call(progress, s),
    [progress],
  );

  return { progress, isVisited };
}