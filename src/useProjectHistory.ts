import { useCallback, useEffect, useState } from 'react';
import { validateGraph } from './analysis';
import { PRESETS } from './presets';
import { recordHistory, redoHistory, undoHistory, type GraphHistory } from './history';
import type { Graph } from './types';

function loadInitial() { try { const saved = localStorage.getItem('tensorlab-project'); if (saved) return validateGraph(JSON.parse(saved)); } catch { /* Invalid saved projects fall back to a valid preset. */ } return PRESETS.cnn(); }

type Options = { training: boolean; isTrainingActive: () => boolean; resetMetrics: () => void; notify: (message: string) => void };

export default function useProjectHistory({ training, isTrainingActive, resetMetrics, notify }: Options) {
  const [graph, setGraph] = useState<Graph>(loadInitial), [history, setHistory] = useState<GraphHistory>({ past: [], future: [] });
  // Write failures intentionally reach the existing recovery boundary.
  useEffect(() => { localStorage.setItem('tensorlab-project', JSON.stringify(graph)); }, [graph]);
  const commit = useCallback((next: Graph) => {
    if (isTrainingActive()) { notify('训练期间请先停止训练再修改模型'); return false; }
    setHistory(previous => recordHistory(previous, graph)); setGraph(next); resetMetrics(); return true;
  }, [graph, isTrainingActive, notify, resetMetrics]);
  const undo = useCallback(() => { if (isTrainingActive()) return; const step = undoHistory(history, graph); if (!step.graph) return; setHistory(step.history); setGraph(step.graph); resetMetrics(); }, [graph, history, isTrainingActive, resetMetrics]);
  const redo = useCallback(() => { if (isTrainingActive()) return; const step = redoHistory(history, graph); if (!step.graph) return; setHistory(step.history); setGraph(step.graph); resetMetrics(); }, [graph, history, isTrainingActive, resetMetrics]);
  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (training || isTrainingActive() || event.defaultPrevented || event.isComposing) return;
      const target = event.target as HTMLElement | null;
      if (target?.isContentEditable || ['INPUT', 'SELECT', 'TEXTAREA'].includes(target?.tagName ?? '')) return;
      if (!(event.ctrlKey || event.metaKey) || event.altKey) return;
      if (event.key.toLowerCase() === 'z') { event.preventDefault(); event.shiftKey ? redo() : undo(); }
      else if (event.key.toLowerCase() === 'y') { event.preventDefault(); redo(); }
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [redo, training, isTrainingActive, undo]);
  return { graph, setGraph, history, commit, undo, redo };
}
