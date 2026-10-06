import { Component, useEffect, useMemo, useRef, useState, type ReactNode } from 'react';
import { Canvas, useThree } from '@react-three/fiber';
import { Html, OrbitControls, OrthographicCamera, Grid } from '@react-three/drei';
import * as THREE from 'three';
import { COLORS, type Graph, type Analysis, type Diagnostic, type Layer, type Edge } from './types';
import { shapeText } from './analysis';
import { EdgeLine, ParticleEdge, ResidualFlow, flowCurve } from './Flow';
import AttentionModule, { SceneText } from './AttentionModule';
import { isAttention, attentionLayout, qkvRadius } from './attentionLayout';
import { outerResidualCurve, residualEdges } from './graphRoutes';
import { crossInputRole, isCrossAttention, isProjectionPort } from './attentionConfig';

export interface SceneHandle { capture: (size: number, transparent: boolean) => Promise<Blob>; svg: () => string; reset: () => void; focus: (nodeId: string) => void; focusHead: (nodeId: string, index: number) => void; }
interface Props { graph: Graph; analysis: Analysis; selected: string | null; onSelect: (id: string | null) => void; selectedHead: number | null; selectedProjection: string | null; onSelectProjection: (id: string, port: string) => void; onSelectHead: (id: string, index: number) => void; rotating: boolean; playing: boolean; direction: 'forward' | 'backward'; speed: number; expanded: boolean; diagnostics: Diagnostic[]; onReady: (handle: SceneHandle) => void; onGpu: (name: string) => void; }
interface SceneRoute extends Edge { start: THREE.Vector3; end: THREE.Vector3; control1?: THREE.Vector3; control2?: THREE.Vector3; residual: boolean; }

function layout(graph: Graph, analysis: Analysis, expanded: boolean) {
  const rank: Record<string, number> = {};
  for (const id of analysis.order) rank[id] = Math.max(-1, ...graph.edges.filter(e => e.target === id).map(e => rank[e.source] ?? -1)) + 1;
  const rows: Record<number, number> = {};
  const positions: Record<string, THREE.Vector3> = {};
  const max = Math.max(1, ...Object.values(rank));
  const minimumWidth = graph.nodes.some(n => isAttention(n.op)) ? (expanded ? 4 : 3.5) : (expanded ? 2.8 : 2.1);
  const widths = Array.from({ length: max + 1 }, (_, r) => Math.max(minimumWidth, ...graph.nodes.filter(n => rank[n.id] === r && isAttention(n.op)).map(n => attentionLayout(n, expanded).dimensions[0] + 1.6)));
  const centers = widths.map((w, r) => widths.slice(0, r).reduce((a, b) => a + b, 0) + w / 2 - widths.reduce((a, b) => a + b, 0) / 2);
  graph.nodes.forEach((n, i) => {
    const r = rank[n.id] ?? i, row = rows[r] || 0; rows[r] = row + 1;
    const height = Math.max(3, ...graph.nodes.filter(layer => rank[layer.id] === r && isAttention(layer.op)).map(layer => attentionLayout(layer, expanded).dimensions[1] + 1.8));
    const elevation = isAttention(n.op) ? attentionLayout(n, expanded).dimensions[1] / 2 + 1.05 : 1.65;
    positions[n.id] = new THREE.Vector3(centers[r] ?? (r - max / 2) * 2.1, elevation + row * height, 0);
  });
  return positions;
}

function contentsDimensions(shape: number[] | undefined, op: string, node?: Layer, expanded = false): [number, number, number] {
  if (node && isAttention(op)) return attentionLayout(node, expanded).dimensions;
  if (op === 'Add') return [0.3, 0.3, 0.3];
  if (!shape) return [0.5, 1, 1];
  if (shape.length === 4) return [Math.max(0.22, Math.log2(shape[1] + 1) * 0.15), Math.max(0.6, Math.log2(shape[2] + 1) * 0.38), Math.max(0.6, Math.log2(shape[3] + 1) * 0.38)];
  return [['ReLU', 'GELU', 'Dropout', 'Flatten'].includes(op) ? 0.18 : 0.55, Math.min(2.5, Math.max(0.65, Math.log2(shape.at(-1)! + 1) * 0.25)), 0.8];
}

function nodePort(node: Layer, position: THREE.Vector3, side: 'input' | 'output' | 'context', expanded: boolean, shape?: number[]) {
  const offset = isAttention(node.op) ? new THREE.Vector3(...attentionLayout(node, expanded).ports[side]) : new THREE.Vector3((side === 'input' ? -1 : 1) * contentsDimensions(shape, node.op)[0] / 2, 0, 0);
  return position.clone().add(offset);
}

function layerLabel(node: Layer, dims: [number, number, number], selected: string | null, head: number | null, expanded: boolean) {
  const headPosition = selected === node.id && head !== null && isAttention(node.op) ? attentionLayout(node, expanded).heads[head] : undefined;
  const attn = headPosition ? attentionLayout(node, expanded) : null, meta = head !== null ? attn?.headMeta[head] : undefined;
  return { position: new THREE.Vector3(headPosition?.[0] ?? 0, headPosition ? headPosition[1] - 2.45 : -dims[1] / 2 - 0.6, 0), name: `${node.name}${meta ? ` · ${attn!.branches > 1 ? `B${meta.branch + 1} · ` : ''}H${meta.head + 1}` : ''}` };
}

function boxCorners(box: THREE.Box3) {
  return Array.from({ length: 8 }, (_, i) => new THREE.Vector3(i & 1 ? box.max.x : box.min.x, i & 2 ? box.max.y : box.min.y, i & 4 ? box.max.z : box.min.z));
}

function fitCamera(camera: THREE.OrthographicCamera, bounds: THREE.Box3, width: number, height: number) {
  camera.updateMatrixWorld(true);
  const view = new THREE.Box3().setFromPoints(boxCorners(bounds).map(p => p.applyMatrix4(camera.matrixWorldInverse)));
  // Fit around the orbit target even when residual branches lie on one side.
  const dimensions = new THREE.Vector2(2 * Math.max(Math.abs(view.min.x), Math.abs(view.max.x)), 2 * Math.max(Math.abs(view.min.y), Math.abs(view.max.y)));
  // Leave room for the scene title and the bottom playback controls.
  camera.zoom = Math.min(Math.max(100, width - 70) / (dimensions.x + 1.5), Math.max(100, height - 145) / (dimensions.y + 1), 110);
  camera.updateProjectionMatrix();
}

function svgAttention(root: THREE.Object3D | undefined, camera: THREE.Camera, width: number, height: number, escape: (s: string) => string) {
  if (!root) return '';
  root.updateWorldMatrix(true, true);
  const project = (p: THREE.Vector3) => { const v = p.clone().project(camera); return [(v.x * 0.5 + 0.5) * width, (-v.y * 0.5 + 0.5) * height]; };
  const polygons: { depth: number; markup: string }[] = [], texts: string[] = [], paths: string[] = [];
  root.traverse(object => {
    if (object.userData.exportMesh && object instanceof THREE.Mesh) {
      const positions = object.geometry.getAttribute('position'), indices = object.geometry.index;
      const material = object.material as THREE.MeshStandardMaterial;
      const center = project(object.getWorldPosition(new THREE.Vector3()));
      const identity = ` data-part="${escape(object.userData.exportPart || 'mesh')}" data-center-x="${center[0]}" data-center-y="${center[1]}"${object.userData.exportSides ? ` data-sides="${object.userData.exportSides}"` : ''}`;
      for (let i = 0; i < (indices?.count ?? positions.count); i += 3) {
        const points = [0, 1, 2].map(offset => new THREE.Vector3().fromBufferAttribute(positions, indices ? indices.getX(i + offset) : i + offset).applyMatrix4(object.matrixWorld));
        const depth = points.reduce((sum, p) => sum + p.clone().applyMatrix4(camera.matrixWorldInverse).z, 0) / 3;
        polygons.push({ depth, markup: `<polygon${identity} points="${points.map(p => project(p).join(',')).join(' ')}" fill="#${material.color.getHexString()}" fill-opacity="${material.opacity}" stroke="#${material.color.getHexString()}" stroke-width="0.4"/>` });
      }
    }
    if (object.userData.exportText) {
      const p = project(object.getWorldPosition(new THREE.Vector3()));
      texts.push(`<text x="${p[0]}" y="${p[1]}" text-anchor="middle" font-family="Arial,sans-serif" font-size="9" fill="${object.userData.exportColor}">${escape(object.userData.exportText)}</text>`);
    }
    if (object.userData.exportCurve) {
      const curve = object.userData.exportCurve, a = new THREE.Vector3(...curve.start as [number, number, number]), b = new THREE.Vector3(...curve.end as [number, number, number]);
      const points = [a, new THREE.Vector3(...curve.control1 as [number, number, number]), new THREE.Vector3(...curve.control2 as [number, number, number]), b].map(p => project(p.clone().applyMatrix4(object.matrixWorld)).join(','));
      paths.push(`<path d="M${points[0]} C${points[1]} ${points[2]} ${points[3]}" fill="none" stroke="${curve.color}" stroke-width="0.8" stroke-opacity="${curve.opacity}"/>`);
    }
  });
  return paths.join('') + polygons.sort((a, b) => a.depth - b.depth).map(p => p.markup).join('') + texts.join('');
}

function World(props: Props) {
  const { gl, camera, scene, size } = useThree();
  // Keep label roots in one DOM container when the canvas event target changes.
  const labelPortal = useMemo(() => ({ current: gl.domElement.parentElement! }), [gl]);
  const controls = useRef<any>(null);
  const selection = useRef({ selected: props.selected, head: props.selectedHead });
  selection.current = { selected: props.selected, head: props.selectedHead };
  const positions = useMemo(() => layout(props.graph, props.analysis, props.expanded), [props.graph, props.analysis, props.expanded]);
  const routes = useMemo(() => {
    const shortcuts = residualEdges(props.graph), lanes = new Map<string, number>();
    return props.graph.edges.flatMap<SceneRoute>(e => {
      if (!positions[e.source] || !positions[e.target]) return [];
      const source = props.graph.nodes.find(n => n.id === e.source)!, target = props.graph.nodes.find(n => n.id === e.target)!;
      const projectionPoint = (node: Layer, port?: string) => isAttention(node.op) && port ? attentionLayout(node, props.expanded).projectionPositions[port] : undefined;
      const startPoint = projectionPoint(source, e.sourcePort), endPoint = projectionPoint(target, e.targetPort);
      const start = startPoint ? positions[e.source].clone().add(new THREE.Vector3(...startPoint)) : nodePort(source, positions[e.source], 'output', props.expanded, props.analysis.layers[e.source]?.output);
      const side = !isProjectionPort(e.targetPort) && isCrossAttention(target) && crossInputRole(props.graph, target, e.id) === 'context' ? 'context' : 'input';
      const end = endPoint ? positions[e.target].clone().add(new THREE.Vector3(...endPoint)) : nodePort(target, positions[e.target], side, props.expanded, props.analysis.layers[e.target]?.output);
      const residual = shortcuts.has(e.id);
      if (residual) {
        end.copy(positions[e.target]).add(new THREE.Vector3(0, 0.15, 0));
        const obstacles = props.graph.nodes.filter(n => n.id !== e.source && n.id !== e.target).map(n => {
          const dims = new THREE.Vector3(...contentsDimensions(props.analysis.layers[n.id]?.output, n.op, n, props.expanded));
          const box = new THREE.Box3().setFromCenterAndSize(positions[n.id], dims);
          return { min: box.min.toArray(), max: box.max.toArray() };
        }).filter(box => box.max[0] > start.x && box.min[0] < end.x);
        const lane = lanes.get(e.target) ?? 0; lanes.set(e.target, lane + 1);
        const path = outerResidualCurve(start.toArray(), end.toArray(), obstacles, lane);
        return [{ ...e, start, end, residual, control1: new THREE.Vector3(...path.control1), control2: new THREE.Vector3(...path.control2) }];
      }
      return [{ ...e, start, end, residual, control1: undefined, control2: undefined }];
    });
  }, [props.graph, props.analysis, props.expanded, positions]);
  const bounds = useMemo(() => {
    const box = new THREE.Box3();
    props.graph.nodes.forEach(n => {
      const dims = new THREE.Vector3(...contentsDimensions(props.analysis.layers[n.id]?.output, n.op, n, props.expanded));
      const nodeBox = new THREE.Box3().setFromCenterAndSize(positions[n.id], dims);
      nodeBox.min.y -= 1.1;
      box.union(nodeBox);
    });
    routes.filter(route => route.residual).forEach(route => flowCurve(route).getPoints(40).forEach(p => box.expandByPoint(p)));
    return box.isEmpty() ? new THREE.Box3(new THREE.Vector3(-5, 0, -2), new THREE.Vector3(5, 5, 2)) : box;
  }, [props.graph, props.analysis, props.expanded, positions, routes]);
  const attentionView = props.graph.nodes.some(n => isAttention(n.op));
  const [hover, setHover] = useState<string | null>(null);
  const centerView = (center: THREE.Vector3, dimensions: THREE.Vector3, offset: THREE.Vector3, bounds?: THREE.Box3) => {
    camera.position.copy(center.clone().add(offset)); camera.lookAt(center);
    if (controls.current) { controls.current.target.copy(center); controls.current.update(); }
    fitCamera(camera as THREE.OrthographicCamera, bounds ?? new THREE.Box3().setFromCenterAndSize(center, dimensions), size.width, size.height);
  };
  const focus = (id: string, head?: number) => {
    const target = positions[id], node = props.graph.nodes.find(n => n.id === id); if (!target || !node) return;
    const attention = isAttention(node.op) ? attentionLayout(node, props.expanded) : null;
    const headPoint = head !== undefined ? attention?.heads[head] : undefined;
    const focusBox = new THREE.Box3().setFromCenterAndSize(target, new THREE.Vector3(...contentsDimensions(props.analysis.layers[id]?.output, node.op, node, props.expanded)));
    if (node.op === 'Add') routes.filter(route => route.target === id).forEach(route => {
      flowCurve(route).getPoints(40).forEach(p => focusBox.expandByPoint(p));
      props.graph.nodes.filter(n => positions[n.id].x >= route.start.x && positions[n.id].x <= route.end.x).forEach(n => focusBox.union(new THREE.Box3().setFromCenterAndSize(positions[n.id], new THREE.Vector3(...contentsDimensions(props.analysis.layers[n.id]?.output, n.op, n, props.expanded)))));
    });
    const center = headPoint ? target.clone().add(new THREE.Vector3(...headPoint)) : target.clone();
    const radius = attention ? qkvRadius(attention.qkvCount) : 0;
    const dims = headPoint ? new THREE.Vector3(2.1, Math.max(5, radius * 2 + 2.8), Math.max(4.4, radius * 2 + 1.3)) : new THREE.Vector3(...contentsDimensions(props.analysis.layers[id]?.output, node.op, node, props.expanded));
    centerView(center, dims, attention ? (headPoint ? new THREE.Vector3(24, 5, 18) : new THREE.Vector3(18, 7, 24)) : new THREE.Vector3(4, 8, 12), node.op === 'Add' ? focusBox : undefined);
  };
  useEffect(() => {
    const context = gl.getContext(), ext = context.getExtension('WEBGL_debug_renderer_info');
    const name = ext ? context.getParameter(ext.UNMASKED_RENDERER_WEBGL) : 'WebGL 2'; props.onGpu(String(name));
  }, [gl, props.onGpu]);
  useEffect(() => {
    const reset = () => {
      const center = bounds.getCenter(new THREE.Vector3());
      camera.position.copy(center.clone().add(attentionView ? new THREE.Vector3(14, 8, 26) : new THREE.Vector3(15, 9, 18))); camera.lookAt(center);
      if (controls.current) { controls.current.target.copy(center); controls.current.update(); }
      fitCamera(camera as THREE.OrthographicCamera, bounds, size.width, size.height);
    };
    reset();
    props.onReady({
      reset,
      focus: id => focus(id),
      focusHead: (id, index) => focus(id, index),
      capture: async (width, transparent) => {
        const oldSize = gl.getSize(new THREE.Vector2()), oldRatio = gl.getPixelRatio(), background = scene.background;
        const height = Math.round(width * size.height / size.width);
        try {
          gl.setPixelRatio(1); gl.setSize(width, height, false);
          if (transparent) { scene.background = null; gl.setClearColor(0x000000, 0); }
          gl.render(scene, camera);
          const output = document.createElement('canvas'); output.width = width; output.height = height;
          const context = output.getContext('2d')!; context.drawImage(gl.domElement, 0, 0);
          const ratio = width / size.width;
          context.textAlign = 'center';
          props.graph.nodes.forEach(n => {
            const dims = contentsDimensions(props.analysis.layers[n.id]?.output, n.op, n, props.expanded);
            const label = layerLabel(n, dims, selection.current.selected, selection.current.head, props.expanded);
            const point = positions[n.id].clone().add(label.position).project(camera);
            const x = (point.x * 0.5 + 0.5) * width, y = (-point.y * 0.5 + 0.5) * height;
            context.font = `500 ${9 * ratio}px "Segoe UI", "Microsoft YaHei", sans-serif`;
            context.fillStyle = '#506d78'; context.fillText(label.name, x, y, 125 * ratio);
            context.font = `${8 * ratio}px "Segoe UI", sans-serif`; context.fillStyle = '#7d949e';
            context.fillText(shapeText(props.analysis.layers[n.id]?.output), x, y + 15 * ratio, 125 * ratio);
          });
          return await new Promise<Blob>((resolve, reject) => output.toBlob(blob => blob ? resolve(blob) : reject(new Error('图片导出失败')), 'image/png'));
        } finally { scene.background = background; gl.setPixelRatio(oldRatio); gl.setSize(oldSize.x, oldSize.y, false); gl.render(scene, camera); }
      },
      svg: () => {
        const width = size.width, height = size.height;
        const project = (v: THREE.Vector3) => { const p = v.clone().project(camera); return [(p.x * 0.5 + 0.5) * width, (-p.y * 0.5 + 0.5) * height]; };
        const escape = (s: string) => s.replace(/[<>&"']/g, c => ({ '<': '&lt;', '>': '&gt;', '&': '&amp;', '"': '&quot;', "'": '&apos;' })[c]!);
        const lines = routes.map(route => `<g data-edge-id="${escape(route.id)}" data-flow-kind="${route.residual ? 'residual' : 'main'}">${svgAttention(scene.getObjectByName(`edge_${route.id}`), camera, width, height, escape)}</g>`).join('');
        const boxes = props.graph.nodes.map(n => {
          const pos = positions[n.id], dims = contentsDimensions(props.analysis.layers[n.id]?.output, n.op, n, props.expanded);
          const points = Array.from({ length: 8 }, (_, i) => project(pos.clone().add(new THREE.Vector3((i & 1 ? 1 : -1) * dims[0] / 2, (i & 2 ? 1 : -1) * dims[1] / 2, (i & 4 ? 1 : -1) * dims[2] / 2))));
          const faces = isAttention(n.op) || n.op === 'Add' ? svgAttention(scene.getObjectByName(`${n.op === 'Add' ? 'add' : 'attention'}_${n.id}`), camera, width, height, escape) : [[0, 1, 3, 2], [0, 2, 6, 4], [2, 3, 7, 6], [4, 5, 7, 6]].map(indices => `<polygon points="${indices.map(i => points[i].join(',')).join(' ')}" fill="${COLORS[n.op]}" fill-opacity="0.35" stroke="${COLORS[n.op]}" stroke-width="1"/>`).join('');
          const label = layerLabel(n, dims, selection.current.selected, selection.current.head, props.expanded), p = project(pos.clone().add(label.position));
          const center = project(pos);
          return `<g data-node-id="${escape(n.id)}" data-center-x="${center[0]}" data-center-y="${center[1]}">${faces}<text x="${p[0]}" y="${p[1]}" font-family="Arial, sans-serif" font-size="11" text-anchor="middle" fill="#293e43">${escape(label.name)}</text><text x="${p[0]}" y="${p[1] + 16}" font-family="Arial, sans-serif" font-size="10" text-anchor="middle" fill="#61777d">${escape(shapeText(props.analysis.layers[n.id]?.output))}</text></g>`;
        }).join('');
        return `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}"><title>${escape(props.graph.name)}</title>${lines}${boxes}</svg>`;
      }
    });
  }, [camera, gl, scene, positions, routes, size, bounds, attentionView, props.expanded, props.graph, props.analysis, props.onReady]);
  useEffect(() => { if (props.selected && props.selectedHead !== null && isAttention(props.graph.nodes.find(n => n.id === props.selected)?.op || '')) focus(props.selected, props.selectedHead); }, [camera, props.selected, props.selectedHead, props.expanded, positions, size]);
  return <>
    <color attach="background" args={['#f5f8fa']} />
    <ambientLight intensity={1.5} /><directionalLight position={[8, 14, 10]} intensity={2.2} />
    <Grid position={[0, -0.1, 0]} args={[80, 40]} cellSize={1} cellThickness={0.5} cellColor="#d9e2e5" sectionSize={5} sectionThickness={0.8} sectionColor="#bacacf" fadeDistance={60} fadeStrength={1.3} infiniteGrid />
    <OrbitControls ref={controls} makeDefault enableDamping={false} minZoom={1.5} maxZoom={180} />
    {routes.map(route => {
      const flow = { start: route.start, end: route.end, control1: route.control1, control2: route.control2 };
      return <group key={route.id} name={`edge_${route.id}`}>{route.residual ? <ResidualFlow {...flow} active={props.playing} direction={props.direction} speed={props.speed} /> : <><EdgeLine {...flow} color={props.analysis.layers[route.target] ? '#7095a6' : '#e26969'} /><ParticleEdge {...flow} active={props.playing} direction={props.direction} speed={props.speed} color={props.direction === 'forward' ? '#31adce' : '#ee706d'} intensity={Math.log2(props.analysis.layers[route.source]?.output.reduce((a, b) => a * b, 1) || 1)} /></>}</group>;
    })}
    {props.graph.nodes.map(n => {
      const pos = positions[n.id], dims = contentsDimensions(props.analysis.layers[n.id]?.output, n.op, n, props.expanded);
      const issue = props.diagnostics.find(d => d.nodeId === n.id && d.level === 'error'), selected = props.selected === n.id;
      const color = issue ? '#e45555' : COLORS[n.op];
      const label = layerLabel(n, dims, props.selected, props.selectedHead, props.expanded);
      return <group key={n.id} position={pos} name={isAttention(n.op) ? `attention_${n.id}` : n.op === 'Add' ? `add_${n.id}` : undefined} onClick={e => { e.stopPropagation(); props.onSelect(n.id); focus(n.id); }} onPointerOver={e => { e.stopPropagation(); setHover(n.id); document.body.style.cursor = 'pointer'; }} onPointerOut={() => { setHover(null); document.body.style.cursor = ''; }}>
        {isAttention(n.op) ? <AttentionModule node={n} selected={selected} selectedHead={selected ? props.selectedHead : null} selectedProjection={selected ? props.selectedProjection : null} onSelectProjection={(port, center) => { props.onSelectProjection(n.id, port); centerView(center, new THREE.Vector3(3.2, 3.8, 3.2), new THREE.Vector3(24, 5, 18)); }} overriddenPorts={props.graph.edges.filter(e => e.target === n.id && isProjectionPort(e.targetPort)).map(e => e.targetPort!)} attachedPorts={props.graph.edges.some(e => e.source === n.id && e.sourcePort || e.target === n.id && isProjectionPort(e.targetPort))} onSelectHead={index => { props.onSelectHead(n.id, index); focus(n.id, index); }} onFocusPart={center => { props.onSelect(n.id); centerView(center, new THREE.Vector3(2.1, 2.4, 2.1), new THREE.Vector3(18, 7, 24)); }} playing={props.playing} rotating={props.rotating} direction={props.direction} speed={props.speed} expanded={props.expanded} error={Boolean(issue)} /> : <>
        <mesh userData={{ exportMesh: n.op === 'Add' }}>
          <boxGeometry args={dims} /><meshStandardMaterial color={color} transparent opacity={selected || hover === n.id ? 0.78 : 0.48} roughness={0.28} metalness={0.12} emissive={color} emissiveIntensity={issue ? 0.5 : selected ? 0.16 : 0} />
        </mesh>
        <lineSegments raycast={() => {}}><edgesGeometry args={[new THREE.BoxGeometry(...dims)]} /><lineBasicMaterial color={selected ? '#223b43' : color} transparent opacity={0.85} /></lineSegments>{n.op === 'Add' && <SceneText text="+" position={new THREE.Vector3(0, 0.44, 0)} color={color} scale={0.38} />}</>}
        {selected && !isAttention(n.op) && <mesh raycast={() => {}} position={[0, -dims[1] / 2 - 0.1, 0]} rotation={[-Math.PI / 2, 0, 0]}><ringGeometry args={[0.45, 0.48, 40]} /><meshBasicMaterial color="#233f47" transparent opacity={0.7} /></mesh>}
        <Html portal={labelPortal} position={label.position} center zIndexRange={[10, 0]} style={{ pointerEvents: 'none' }}><div className={`scene-label ${selected ? 'selected' : ''}`}><strong>{label.name}</strong><span>{shapeText(props.analysis.layers[n.id]?.output)}</span></div></Html>
      </group>;
    })}
  </>;
}

class SceneBoundary extends Component<{ children: ReactNode }, { error: boolean }> {
  state = { error: false };
  static getDerivedStateFromError() { return { error: true }; }
  render() { return this.state.error ? <div className="empty-state">WebGL 初始化失败。请启用浏览器硬件加速，或切换到拓扑图。</div> : this.props.children; }
}
export default function Scene(props: Props) {
  return <SceneBoundary><Canvas gl={{ antialias: true, alpha: true, preserveDrawingBuffer: true, powerPreference: 'high-performance' }} dpr={[1, 2]} onPointerMissed={() => props.onSelect(null)}><OrthographicCamera makeDefault position={[15, 11, 18]} near={0.1} far={200} zoom={35} /><World {...props} /></Canvas></SceneBoundary>;
}
