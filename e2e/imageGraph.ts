import { readFileSync } from 'node:fs';
import type { Graph } from '../src/types';

// The export fixtures are the same graphs the workbench imports from disk, so a
// layout regression is measured against a model the user actually exports.
export function imageTestGraph(file = 'tests/fixtures/DualBranchCrossAttentionTransformer.json'): Graph {
  return JSON.parse(readFileSync(file, 'utf8')) as Graph;
}
