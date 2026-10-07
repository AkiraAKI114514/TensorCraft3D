import type { Graph } from './types';

export const HISTORY_LIMIT = 40;
export type GraphHistory = { past: Graph[]; future: Graph[] };

type HistoryStep = { history: GraphHistory; graph: Graph | null };

export function recordHistory(history: GraphHistory, current: Graph): GraphHistory {
  return { past: [...history.past, current].slice(-HISTORY_LIMIT), future: [] };
}

export function undoHistory(history: GraphHistory, current: Graph): HistoryStep {
  const graph = history.past.at(-1);
  if (!graph) return { history, graph: null };
  return {
    graph,
    history: { past: history.past.slice(0, -1), future: [...history.future, current] }
  };
}

export function redoHistory(history: GraphHistory, current: Graph): HistoryStep {
  const graph = history.future.at(-1);
  if (!graph) return { history, graph: null };
  return {
    graph,
    history: { past: [...history.past, current].slice(-HISTORY_LIMIT), future: history.future.slice(0, -1) }
  };
}
