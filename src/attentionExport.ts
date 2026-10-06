import * as THREE from 'three';
import { attentionLayout, visualBranches, type AttentionRoute, type Point3 } from './attentionLayout';
import { RESIDUAL_COLOR } from './flowGeometry';
import type { Graph, Layer } from './types';

// Heads that project one shared input into Q/K/V subspaces are drawn as a
// stepped stack of identical faces. A face whose own Q/K/V port is wired to
// another layer is no longer interchangeable with its siblings, so it keeps its
// own column; branches never stack, as each has its own weights and merge.
export function exportStacks(graph: Graph, node: Layer) {
  const wired = new Set(graph.edges.flatMap(e => [e.source === node.id ? e.sourcePort : undefined, e.target === node.id ? e.targetPort : undefined]).filter(Boolean));
  const faces = attentionLayout(node).faces;
  return Array.from({ length: visualBranches(node) }, (_, branch) => faces.filter(face => face.branch === branch && face.vertices.every(v => !wired.has(v.id))).map(face => face.index)).filter(stack => stack.length > 1);
}

// Image export draws attention modules from the flat layout instead of cloning
// the interactive ones: the live faces stand edge-on to a straight-on camera,
// so Q and K project onto the same point and the heads hide one another.
// The parts keep the names and export metadata the live module carries.
export function buildAttentionExport(node: Layer, expanded: boolean, options: { error?: boolean; overriddenPorts?: string[]; stacks?: number[][] } = {}) {
  const layout = attentionLayout(node, expanded, true, options.stacks), root = new THREE.Group(), resources: (THREE.BufferGeometry | THREE.Material)[] = [];
  const own = <T extends THREE.BufferGeometry | THREE.Material>(resource: T) => { resources.push(resource); return resource; };
  const text = (parent: THREE.Object3D, label: string, position: Point3, color = '#516e79') => {
    const anchor = new THREE.Object3D(); anchor.position.set(...position); anchor.userData = { exportText: label, exportColor: color }; parent.add(anchor);
  };
  const cube = (parent: THREE.Object3D, point: Point3, color: string, { part, label, labelOffset, size = 0.3, opacity = 0.85 }: { part?: string; label?: string; labelOffset?: Point3; size?: number; opacity?: number }) => {
    const group = new THREE.Group(); group.position.set(...point);
    const mesh = new THREE.Mesh(own(new THREE.BoxGeometry(size, size, size)), own(new THREE.MeshStandardMaterial({ color, transparent: true, opacity, roughness: 0.35 })));
    mesh.userData = { exportMesh: true, exportPart: part || label || 'port' }; group.add(mesh);
    if (label) text(group, label, labelOffset ?? [0, size * 0.8 + 0.2, 0], color);
    parent.add(group);
  };
  const route = (parent: THREE.Object3D, r: AttentionRoute, arrow: boolean, opacity = 0.65) => {
    const curve = new THREE.CubicBezierCurve3(...[r.start, r.control1, r.control2, r.end].map(p => new THREE.Vector3(...p)) as [THREE.Vector3, THREE.Vector3, THREE.Vector3, THREE.Vector3]);
    const line = new THREE.Mesh(own(new THREE.TubeGeometry(curve, 20, 0.009, 4, false)), own(new THREE.MeshBasicMaterial({ color: r.color, transparent: true, opacity })));
    line.userData = { exportCurve: { start: r.start, end: r.end, control1: r.control1, control2: r.control2, color: r.color, opacity } }; parent.add(line);
    if (!arrow && r.color !== RESIDUAL_COLOR) return;
    const head = new THREE.Mesh(own(new THREE.ConeGeometry(0.055, 0.14, 6)), own(new THREE.MeshBasicMaterial({ color: r.color })));
    head.position.copy(curve.getPoint(0.7)); head.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), curve.getTangent(0.7).normalize()); head.userData = { exportMesh: true };
    parent.add(head);
  };
  const ports = layout.ports, cross = node.params.attention_type === 'cross';
  if (node.op === 'Transformer') {
    cube(root, ports.norm1, '#bc8cc8', { label: 'LN', part: 'LN1' }); cube(root, ports.add1, RESIDUAL_COLOR, { label: '+', part: 'Add1', size: 0.22 });
    cube(root, ports.norm2, '#bc8cc8', { label: 'LN', part: 'LN2' }); cube(root, ports.ffn, '#e48369', { label: 'FFN', size: 0.55 }); cube(root, ports.add2, RESIDUAL_COLOR, { label: '+', part: 'Add2', size: 0.22 });
  }
  cube(root, ports.input, '#7d91a8', { part: 'input', label: cross ? 'Query' : undefined, size: 0.18 });
  if (cross) cube(root, ports.context, '#a17cbb', { part: 'context-input', label: 'Context', size: 0.18 });
  if (layout.branches > 1) layout.branchMerges.forEach((point, i) => cube(root, point, '#3c9b94', { label: `B${i + 1} · Wᵒ`, part: `branch-output-${i}`, size: 0.25 }));
  cube(root, ports.merge, '#3c9b94', { label: layout.branches > 1 ? 'Branch Mean' : 'Concat · Wᵒ', labelOffset: [0, -0.55, 0], size: 0.4 });
  layout.routes.forEach(r => route(root, r, true));
  const faceColor = options.error ? '#d96060' : '#77a8bd';
  layout.faces.forEach(face => {
    const group = new THREE.Group(); group.position.set(...face.position); root.add(group);
    const shape = new THREE.Shape(); shape.moveTo(face.vertices[0].point[0], face.vertices[0].point[1]); face.vertices.slice(1).forEach(v => shape.lineTo(v.point[0], v.point[1])); shape.closePath();
    if (face.stack) {
      // An opaque card under every stacked face hides the faces behind it, so
      // only the stepped edges of the back layers show. It sits behind the
      // front face's routes, which bow back to z = -0.25.
      const card = new THREE.Mesh(own(new THREE.ShapeGeometry(shape)), own(new THREE.MeshBasicMaterial({ color: '#edf3f6' })));
      card.position.z = -0.36; card.userData = { exportMesh: true, exportPart: 'stack-card' }; group.add(card);
    }
    if (face.depth) {
      const fill = new THREE.Mesh(own(new THREE.ShapeGeometry(shape)), own(new THREE.MeshBasicMaterial({ color: faceColor, transparent: true, opacity: 0.14, depthWrite: false })));
      fill.position.z = -0.35; fill.userData = { exportMesh: true, exportPart: 'stack-layer' }; group.add(fill);
      const outline = face.vertices.map(v => [v.point[0], v.point[1], -0.35] as Point3); outline.push(outline[0]);
      const line = new THREE.Line(own(new THREE.BufferGeometry().setFromPoints(outline.map(p => new THREE.Vector3(...p)))), own(new THREE.LineBasicMaterial({ color: faceColor, transparent: true, opacity: 0.8 })));
      line.userData = { exportOutline: outline, exportColor: faceColor }; group.add(line);
      if (face.depth === face.stack!.length - 1) {
        const top = face.vertices.reduce((best, v) => v.point[1] + v.point[0] * 0.3 > best.point[1] + best.point[0] * 0.3 ? v : best).point;
        text(group, `×${face.stack!.length}`, [top[0] + 0.3, top[1] + 0.3, 0.1], '#3f7f94');
      }
      return;
    }
    const members = face.stack?.map(index => layout.faces[index]) ?? [face], q = members.flatMap(f => f.qHeads);
    const label = !face.stack ? face.label : members.every(f => f.qHeads.length === 1) ? `H${q[0] + 1}–H${q.at(-1)! + 1}` : `Q${q[0] + 1}-Q${q.at(-1)! + 1} · KV${members[0].group + 1}-KV${members.at(-1)!.group + 1}`;
    text(group, `${layout.branches > 1 ? `B${face.branch + 1} · ` : ''}${label}`, [0, face.points.inlet[1] + 0.55, 0]);
    const geometry = own(new THREE.ExtrudeGeometry(shape, { depth: 0.06, bevelEnabled: false, steps: 1 }));
    const polygon = new THREE.Mesh(geometry, own(new THREE.MeshStandardMaterial({ color: faceColor, transparent: true, opacity: 0.14, roughness: 0.45, side: THREE.DoubleSide })));
    polygon.userData = { exportMesh: true, exportPart: 'qkv-face', exportSides: face.vertices.length }; group.add(polygon);
    group.add(new THREE.LineSegments(own(new THREE.EdgesGeometry(geometry)), own(new THREE.LineBasicMaterial({ color: faceColor, transparent: true, opacity: 0.8 }))));
    // Labels sit radially outside their vertex, so neighbouring Q/K/V names never stack.
    face.vertices.forEach(v => { const r = Math.hypot(v.point[0], v.point[1]); cube(group, v.point, v.color, { part: v.id, label: v.label, labelOffset: [v.point[0] / r * 0.4, v.point[1] / r * 0.4, 0.12], size: 0.26 }); });
    cube(group, face.points.inlet, '#728f9e', { part: 'head-input', size: 0.12 }); cube(group, face.points.outlet, '#36a18a', { part: 'head-output', size: 0.12 });
    if (cross) cube(group, face.points.context, '#a17cbb', { part: 'context-input', size: 0.12 });
    face.computations.forEach(c => { cube(group, c.score, '#539aa9', { part: `scores-${c.head}`, size: 0.16, opacity: 0.7 }); cube(group, c.weighted, '#36a18a', { part: `weighted-${c.head}`, size: 0.12, opacity: 0.8 }); });
    face.routes.filter(r => !r.projection || !options.overriddenPorts?.includes(r.projection)).forEach(r => route(group, r, false));
  });
  return { root, layout, dispose: () => resources.forEach(resource => resource.dispose()) };
}
