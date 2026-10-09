import { COLORS, isGroup, type Edge, type Graph, type Layer, type SubgraphDef } from './types';

/**
 * 结构签名的规范化：只保留决定语义的参数。
 *
 * 位置和名称不参与签名，否则两个真正同构的块会因为画布坐标不同而被判为不同。
 */
function signature(node: Layer): string {
  if (isGroup(node)) return `Group:${node.subgraph ?? ''}`;
  const params = Object.keys(node.params).sort().map(key => `${key}=${JSON.stringify(node.params[key])}`).join(',');
  return `${node.op}(${params})`;
}

const ALPHABET = 'abcdefghijklmnopqrstuvwxyz';
/** 节点 id 只允许 `[a-zA-Z][a-zA-Z0-9_]{0,63}`，自动生成的块名必须落在同一字符集里。 */
const blockName = (ordinal: number, hint: string) => `auto_${hint}_${ordinal < ALPHABET.length ? ALPHABET[ordinal] : `_${ordinal}`}`;

const isAttentionOp = (op: string) => op === 'Transformer' || op === 'MultiHeadAttention';
const hintFor = (block: Layer[]) => block.some(n => isAttentionOp(n.op)) ? 'attn' : block.some(n => n.op.startsWith('Conv')) ? 'conv' : block.some(n => n.op === 'Linear') ? 'linear' : 'block';
const labelFor = (block: Layer[]) => block.some(n => isAttentionOp(n.op)) ? 'Attention Block' : block.some(n => n.op.startsWith('Conv')) ? 'Conv Block' : block.some(n => n.op === 'Linear') ? 'Linear Block' : 'Block';

function topologicalOrder(graph: Graph): string[] {
  const incoming = new Map(graph.nodes.map(n => [n.id, 0]));
  for (const edge of graph.edges) incoming.set(edge.target, (incoming.get(edge.target) ?? 0) + 1);
  const queue = graph.nodes.filter(n => incoming.get(n.id) === 0).map(n => n.id);
  const order: string[] = [];
  while (queue.length) {
    const id = queue.shift()!;
    order.push(id);
    for (const edge of graph.edges) {
      if (edge.source !== id) continue;
      incoming.set(edge.target, incoming.get(edge.target)! - 1);
      if (incoming.get(edge.target) === 0) queue.push(edge.target);
    }
  }
  return order;
}

/**
 * 一个连续段能否作为子图实例被折叠。
 *
 * 子图实例只暴露唯一输入和唯一输出口，所以除了「块内唯一入口/出口」之外，
 * 还要求所有来自块外的边都落在入口上、所有离开块外的边都从出口发出。否则
 * 折叠会丢掉接线（例如从块外直接插进块中间那条残差）。
 */
function foldableAt(graph: Graph, order: string[], start: number, length: number): { entry: string; exit: string } | null {
  const inside = new Set(order.slice(start, start + length));
  if (inside.size !== length) return null;
  const internalIn = new Set<string>(), internalOut = new Set<string>();
  for (const edge of graph.edges) {
    if (!inside.has(edge.source) || !inside.has(edge.target)) continue;
    internalIn.add(edge.target);
    internalOut.add(edge.source);
  }
  const members = order.slice(start, start + length);
  const entries = members.filter(id => !internalIn.has(id));
  const exits = members.filter(id => !internalOut.has(id));
  if (entries.length !== 1 || exits.length !== 1) return null;
  const [entry] = entries, [exit] = exits;
  for (const edge of graph.edges) {
    const fromIn = inside.has(edge.source), toIn = inside.has(edge.target);
    if (fromIn === toIn) continue;
    if (toIn && edge.target !== entry) return null;
    if (fromIn && edge.source !== exit) return null;
  }
  return { entry, exit };
}

/** 块内边的相对偏移指纹：签名相同不代表连线相同，必须单独比对。 */
function edgeFingerprint(graph: Graph, order: string[], start: number, length: number): string {
  const index = new Map(order.map((id, i) => [id, i]));
  const inside = new Set(order.slice(start, start + length));
  return graph.edges
    .filter(e => inside.has(e.source) && inside.has(e.target))
    .map(e => `${index.get(e.source)! - start}->${index.get(e.target)! - start}:${e.sourcePort ?? ''}>${e.targetPort ?? ''}`)
    .sort()
    .join('|');
}

interface Block { start: number; length: number; repeat: number }

function candidateBlocks(graph: Graph, order: string[]): Block[] {
  const byId = new Map(graph.nodes.map(n => [n.id, n]));
  const sig = order.map(id => signature(byId.get(id)!));
  const sigKey = (start: number, length: number) => sig.slice(start, start + length).join('|');
  const found: Block[] = [];
  let cursor = 0;
  while (cursor < order.length) {
    let best: Block | null = null;
    for (let length = 2; length <= Math.floor((order.length - cursor) / 2); length++) {
      if (!foldableAt(graph, order, cursor, length)) continue;
      const pattern = edgeFingerprint(graph, order, cursor, length), key = sigKey(cursor, length);
      let repeat = 1, next = cursor + length;
      while (next + length <= order.length && sigKey(next, length) === key && edgeFingerprint(graph, order, next, length) === pattern) {
        const current = foldableAt(graph, order, next, length);
        const previous = foldableAt(graph, order, next - length, length);
        // 实例之间必须真的串联，否则它们不是同一个块的重复。
        if (!current || !previous || !graph.edges.some(e => e.source === previous.exit && e.target === current.entry)) break;
        repeat += 1;
        next += length;
      }
      // 重复次数优先、块更长其次：宁可折成 3 个大块，也不要 9 个碎块。
      if (repeat > 1 && (best === null || repeat > best.repeat || (repeat === best.repeat && length > best.length))) {
        best = { start: cursor, length, repeat };
      }
    }
    if (!best) { cursor += 1; continue; }
    found.push(best);
    cursor = best.start + best.length * best.repeat;
  }
  return found;
}

/**
 * 把签名重复、拓扑同构、且单入口单出口的连续段折成子图实例。
 *
 * 这是「自动识别」的那一半；手动分组的产物是同一个数据结构，下游无法区分。
 * 保守优先：宁可不分，不要错分——所有条件不满足就原样返回。
 */
export function autoGroup(graph: Graph): { graph: Graph; groups: number } {
  if (graph.nodes.some(isGroup)) return { graph, groups: 0 };
  const order = topologicalOrder(graph);
  // 拓扑排序必须覆盖所有节点，否则图里有环，交给分析器去报错。
  if (order.length !== graph.nodes.length || order.length < 4) return { graph, groups: 0 };
  const blocks = candidateBlocks(graph, order);
  if (!blocks.length) return { graph, groups: 0 };

  const byId = new Map(graph.nodes.map(n => [n.id, n]));
  const subgraphs: Record<string, SubgraphDef> = {};
  const groupOf = new Map<string, Layer>();
  const groups: Layer[] = [];

  blocks.forEach((block, ordinal) => {
    // unit 是单个实例的节点，用来定义子图；members 覆盖全部实例，用来消费外层图里的节点。
    // 只消费 unit 会把其余实例的节点留在外面，产生悬空边。
    const unit = order.slice(block.start, block.start + block.length);
    const members = order.slice(block.start, block.start + block.length * block.repeat);
    if (members.some(id => groupOf.has(id))) return;
    const inner = unit.map(id => byId.get(id)!);
    const name = blockName(ordinal, hintFor(inner));
    const inside = new Set(unit);
    const position = new Map(unit.map((id, i) => [id, i]));
    subgraphs[name] = {
      id: name, name: labelFor(inner), origin: 'auto',
      nodes: inner.map((n, i) => ({ ...n, id: `n${i}`, position: { x: i * 210, y: 100 } })),
      edges: graph.edges.filter(e => inside.has(e.source) && inside.has(e.target))
        .map(e => ({ ...e, id: `e_${e.id}`, source: `n${position.get(e.source)!}`, target: `n${position.get(e.target)!}` })),
    };
    const group: Layer = { id: `block_${name}`, name: `${labelFor(inner)} ×${block.repeat}`, op: 'Group', params: {}, position: { ...inner[0].position }, subgraph: name, repeat: block.repeat };
    for (const id of members) groupOf.set(id, group);
    groups.push(group);
  });
  if (!groups.length) return { graph, groups: 0 };

  const nodes = graph.nodes.filter(n => !groupOf.has(n.id)).concat(groups);
  const edges: Edge[] = [];
  for (const edge of graph.edges) {
    const sourceGroup = groupOf.get(edge.source), targetGroup = groupOf.get(edge.target);
    // 块内部的连线和实例之间的串联都由子图定义与 repeat 承担，不再留在外层图里。
    if (sourceGroup && targetGroup) continue;
    const source = sourceGroup ? sourceGroup.id : edge.source;
    const target = targetGroup ? targetGroup.id : edge.target;
    if (edges.some(e => e.source === source && e.target === target)) continue;
    edges.push({ ...edge, source, target });
  }
  return { graph: { ...graph, nodes, edges, subgraphs }, groups: groups.length };
}

/**
 * 只解散自动组：手动分组是用户显式建立的，任何自动过程都不动它。
 * 自动组的每个实例被内联回真实的算子节点。
 */
export function dissolveAutoGroups(graph: Graph): { graph: Graph; groups: number } {
  const definitions = graph.subgraphs ?? {};
  const auto = new Set(Object.entries(definitions).filter(([, def]) => def.origin === 'auto').map(([name]) => name));
  if (!auto.size) return { graph, groups: 0 };
  const nodes: Layer[] = [], edges: Edge[] = [];
  const first = new Map<string, string>(), last = new Map<string, string>();
  let dissolved = 0;

  for (const node of graph.nodes) {
    if (!isGroup(node) || !node.subgraph || !auto.has(node.subgraph)) { nodes.push(node); first.set(node.id, node.id); last.set(node.id, node.id); continue; }
    const def = definitions[node.subgraph];
    const count = node.repeat ?? 1;
    dissolved += 1;
    const entry = def.nodes.find(n => !def.edges.some(e => e.target === n.id)) ?? def.nodes[0];
    const exit = def.nodes.find(n => !def.edges.some(e => e.source === n.id)) ?? def.nodes[def.nodes.length - 1];
    for (let instance = 0; instance < count; instance++) {
      const scope = `${node.id}_${instance}`;
      def.nodes.forEach((inner, index) => nodes.push({ ...inner, id: `${scope}_${inner.id}`, position: { x: node.position.x + index * 160, y: node.position.y } }));
      for (const edge of def.edges) edges.push({ ...edge, id: `${scope}_${edge.id}`, source: `${scope}_${edge.source}`, target: `${scope}_${edge.target}` });
      if (instance > 0) edges.push({ id: `${scope}_chain`, source: last.get(`${node.id}_${instance - 1}`)!, target: `${scope}_${entry.id}` });
      first.set(`${node.id}_${instance}`, `${scope}_${entry.id}`);
      last.set(`${node.id}_${instance}`, `${scope}_${exit.id}`);
    }
    first.set(node.id, first.get(`${node.id}_0`)!);
    last.set(node.id, last.get(`${node.id}_${count - 1}`)!);
  }
  for (const edge of graph.edges) {
    const source = last.get(edge.source), target = first.get(edge.target);
    if (!source || !target) continue;
    edges.push({ ...edge, source, target });
  }
  const kept = Object.fromEntries(Object.entries(definitions).filter(([, def]) => def.origin === 'manual'));
  return { graph: { ...graph, nodes, edges, subgraphs: Object.keys(kept).length ? kept : undefined }, groups: dissolved };
}

export const groupColor = COLORS.Group;
