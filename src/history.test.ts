import { describe, expect, it } from 'vitest';
import { HISTORY_LIMIT, recordHistory, redoHistory, undoHistory, type GraphHistory } from './history';
import type { Graph } from './types';

const graph = (name: string): Graph => ({ version: 1, name, nodes: [], edges: [] });

function history(...names: string[]): GraphHistory {
  return { past: names.map(graph), future: [] };
}

describe('graph history', () => {
  it('records a new edit and clears redo states', () => {
    const result = recordHistory({ past: [graph('before')], future: [graph('redo')] }, graph('current'));
    expect(result.past.map(item => item.name)).toEqual(['before', 'current']);
    expect(result.future).toEqual([]);
  });

  it('undoes and redoes without mutating the current history', () => {
    const before = history('first', 'second'), current = graph('third');
    const undone = undoHistory(before, current);
    expect(undone.graph?.name).toBe('second');
    expect(undone.history.past.map(item => item.name)).toEqual(['first']);
    expect(undone.history.future.map(item => item.name)).toEqual(['third']);

    const redone = redoHistory(undone.history, undone.graph!);
    expect(redone.graph?.name).toBe('third');
    expect(redone.history.past.map(item => item.name)).toEqual(['first', 'second']);
    expect(redone.history.future).toEqual([]);
    expect(before.past.map(item => item.name)).toEqual(['first', 'second']);
  });

  it('returns no-op steps when the selected direction is empty', () => {
    const empty = { past: [], future: [] };
    expect(undoHistory(empty, graph('current'))).toEqual({ history: empty, graph: null });
    expect(redoHistory(empty, graph('current'))).toEqual({ history: empty, graph: null });
  });

  it(`keeps at most ${HISTORY_LIMIT} past edits`, () => {
    const result = Array.from({ length: HISTORY_LIMIT + 5 }, (_, i) => i + 1).reduce(
      (state, i) => recordHistory(state, graph(String(i))),
      { past: [], future: [] } as GraphHistory
    );
    expect(result.past).toHaveLength(HISTORY_LIMIT);
    expect(result.past[0].name).toBe('6');
    expect(result.past.at(-1)?.name).toBe(String(HISTORY_LIMIT + 5));
  });
});
