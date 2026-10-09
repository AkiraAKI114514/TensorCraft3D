import { MAX_SUBGRAPH_DEPTH, isGroup, repeatOf, type Edge, type Graph, type Layer, type SubgraphDef } from './types';

/** 展开后的节点 id 用斜杠分层：`<顶层节点 id>/<实例序号>/.../<内部节点 id>`。
 *  顶层叶子节点不带斜杠，所以「顶层 id」可以从展开 id 的第一段还原出来。 */
export const EXPAND_SEP = '/';
export const topLevelId = (expandedId: string) => expandedId.split(EXPAND_SEP)[0];

export class SubgraphError extends Error {}

/** 子图内部必须有唯一的入口和出口，否则实例串联和外部连线都没有确定的接点。 */
function endpoints(def: SubgraphDef, path: string): { entry: Layer; exit: Layer } {
  const ids = new Set(def.nodes.map(n => n.id));
  for (const edge of def.edges) {
    if (!ids.has(edge.source) || !ids.has(edge.target)) throw new SubgraphError(`子图 ${path} 的连线指向不存在的内部节点`);
  }
  const hasIncoming = new Set(def.edges.map(e => e.target));
  const hasOutgoing = new Set(def.edges.map(e => e.source));
  const entries = def.nodes.filter(n => !hasIncoming.has(n.id));
  const exits = def.nodes.filter(n => !hasOutgoing.has(n.id));
  if (def.nodes.length < 1) throw new SubgraphError(`子图 ${path} 是空的`);
  if (entries.length !== 1) throw new SubgraphError(`子图 ${path} 需要唯一入口，当前有 ${entries.length} 个`);
  if (exits.length !== 1) throw new SubgraphError(`子图 ${path} 需要唯一出口，当前有 ${exits.length} 个`);
  return { entry: entries[0], exit: exits[0] };
}

/**
 * 把带子图实例的图展开成等价的纯算子图。
 *
 * 展开后完全走既有路径：形状推导、参数计账、导出、训练都看不到子图。这是子图不引入
 * 新语义风险的关键——`analyze` 只在最后把结果按顶层节点重新聚合回去。
 *
 * 展开 id 形如 `g1/0/attn`，所以诊断和层信息都能通过 {@link topLevelId} 归位到顶层节点。
 */
export function expandGraph(graph: Graph): Graph {
  if (!graph.subgraphs || !graph.nodes.some(isGroup)) return graph;
  const nodes: Layer[] = [], edges: Edge[] = [];

  const expandInto = (list: Layer[], listEdges: Edge[], prefix: string, depth: number, ports: Map<string, { entry: string; exit: string }>) => {
    if (depth > MAX_SUBGRAPH_DEPTH) throw new SubgraphError(`子图嵌套超过 ${MAX_SUBGRAPH_DEPTH} 层`);
    for (const node of list) {
      const id = prefix ? `${prefix}${EXPAND_SEP}${node.id}` : node.id;
      if (!isGroup(node)) { nodes.push({ ...node, id }); ports.set(id, { entry: id, exit: id }); continue; }
      const name = node.subgraph;
      const def = name ? graph.subgraphs![name] : undefined;
      if (!def) throw new SubgraphError(`节点 ${node.name || node.id} 指向不存在的子图 ${String(name)}`);
      const { entry, exit } = endpoints(def, String(name));
      const count = repeatOf(node);
      for (let index = 0; index < count; index++) {
        const inner = `${id}${EXPAND_SEP}${index}`;
        const nested = new Map<string, { entry: string; exit: string }>();
        expandInto(def.nodes, def.edges, inner, depth + 1, nested);
        for (const edge of def.edges) {
          const source = nested.get(`${inner}${EXPAND_SEP}${edge.source}`);
          const target = nested.get(`${inner}${EXPAND_SEP}${edge.target}`);
          if (!source || !target) throw new SubgraphError(`子图 ${String(name)} 的连线端点缺失`);
          edges.push({ ...edge, id: `${inner}${EXPAND_SEP}${edge.id}`, source: source.exit, target: target.entry });
        }
        // 实例之间串联：第 i 个的输出接到第 i+1 个的输入，与 repeat 对单个算子的语义一致。
        if (index > 0) {
          const previous = ports.get(`${id}${EXPAND_SEP}${index - 1}`)!;
          const current = nested.get(`${inner}${EXPAND_SEP}${entry.id}`)!;
          edges.push({ id: `${inner}${EXPAND_SEP}chain`, source: previous.exit, target: current.entry });
        }
        const first = nested.get(`${inner}${EXPAND_SEP}${entry.id}`)!, last = nested.get(`${inner}${EXPAND_SEP}${exit.id}`)!;
        ports.set(`${id}${EXPAND_SEP}${index}`, { entry: first.entry, exit: last.exit });
      }
      // 实例节点自身对外暴露首实例的入口和末实例的出口。
      ports.set(id, { entry: ports.get(`${id}${EXPAND_SEP}0`)!.entry, exit: ports.get(`${id}${EXPAND_SEP}${count - 1}`)!.exit });
    }
  };

  const ports = new Map<string, { entry: string; exit: string }>();
  expandInto(graph.nodes, graph.edges, '', 0, ports);
  for (const edge of graph.edges) {
    const source = ports.get(edge.source), target = ports.get(edge.target);
    if (!source || !target) throw new SubgraphError(`连线 ${edge.id} 的端点缺失`);
    edges.push({ ...edge, source: source.exit, target: target.entry });
  }
  return { version: 1, name: graph.name, nodes, edges };
}

/** 子图实例在图里没有自己的算子语义，参数面板和导出都要按哨兵跳过。 */
export const isGroupNode = (layer: Layer) => isGroup(layer);
