import { Component, useEffect, useMemo, useRef, useState, type ReactNode, type RefObject } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { Html, OrbitControls, OrthographicCamera, Grid } from '@react-three/drei';
import * as THREE from 'three';
import { COLORS, repeatOf, type Graph, type Analysis, type Diagnostic, type Layer, type Edge } from './types';
import { product, shapeText } from './analysis';
import { EdgeLine, ParticleEdge, ResidualFlow, flowCurve } from './Flow';
import AttentionModule, { SceneText } from './AttentionModule';
import { isAttention, attentionLayout, qkvRadius } from './attentionLayout';
import { outerResidualCurve, residualEdges } from './graphRoutes';
import { crossInputRole, isCrossAttention, isProjectionPort } from './attentionConfig';
import { captureSceneImage, exportSceneSvg } from './sceneImageExport';
import { blank, createTensorMaterial, floatTexture, majorGroup, tensorCells, type TensorUniforms } from './tensorBlock';
import { blockPaint, type BlockPaint, type SceneObservation } from './sceneObservation';
import { receptiveField, WINDOW_OPS, type Field } from './receptiveField';
import { tensorNumber } from './inferenceTypes';
import type { ImageExportOptions } from './imageLayout';
import { useI18n } from './i18n';

export interface SceneHandle { capture: (size: number, transparent: boolean, options?: ImageExportOptions) => Promise<Blob>; svg: (options?: ImageExportOptions, width?: number) => string; reset: () => void; focus: (nodeId: string) => void; focusHead: (nodeId: string, index: number) => void; }
interface Props { graph: Graph; analysis: Analysis; selected: string | null; onSelect: (id: string | null) => void; selectedHead: number | null; selectedProjection: string | null; onSelectProjection: (id: string, port: string) => void; onSelectHead: (id: string, index: number) => void; rotating: boolean; playing: boolean; direction: 'forward' | 'backward'; speed: number; expanded: boolean; diagnostics: Diagnostic[]; onReady: (handle: SceneHandle) => void; onGpu: (name: string) => void;
  /** Measured values for this level's layers, keyed by their IDs here; painted while present. */
  observation?: SceneObservation | null;
  /** One kernel window per convolution or pooling layer; on unless turned off. */
  showKernels?: boolean; }
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
  if (op === 'Add' || op === 'Multiply' || op === 'Divide') return [0.3, 0.3, 0.3];
  // Latent attention draws three parts side by side (input, latent, output), so it is wider; MoE
  // draws its gate, a column of experts, its shared experts and the output, wider still.
  if (op === 'LatentAttention' || op === 'MoE') return [op === 'MoE' ? 1.3 : 1.15, Math.min(2.5, Math.max(0.65, Math.log2((shape?.at(-1) ?? 64) + 1) * 0.25)), 0.8];
  if (!shape) return [0.5, 1, 1];
  if (shape.length === 4) return [Math.max(0.22, Math.log2(shape[1] + 1) * 0.15), Math.max(0.6, Math.log2(shape[2] + 1) * 0.38), Math.max(0.6, Math.log2(shape[3] + 1) * 0.38)];
  return [['ReLU', 'GELU', 'Dropout', 'Flatten', 'Unflatten'].includes(op) ? 0.18 : 0.55, Math.min(2.5, Math.max(0.65, Math.log2(shape.at(-1)! + 1) * 0.25)), 0.8];
}

function nodePort(node: Layer, position: THREE.Vector3, side: 'input' | 'output' | 'context', expanded: boolean, shape?: number[]) {
  const offset = isAttention(node.op) ? new THREE.Vector3(...attentionLayout(node, expanded).ports[side]) : new THREE.Vector3((side === 'input' ? -1 : 1) * contentsDimensions(shape, node.op)[0] / 2, 0, 0);
  return position.clone().add(offset);
}

function layerLabel(node: Layer, dims: [number, number, number], selected: string | null, head: number | null, expanded: boolean) {
  const headPosition = selected === node.id && head !== null && isAttention(node.op) ? attentionLayout(node, expanded).heads[head] : undefined;
  const attn = headPosition ? attentionLayout(node, expanded) : null, meta = head !== null ? attn?.headMeta[head] : undefined;
  // A folded layer stacks `repeat` identical instances; the ×N badge flags it.
  const repeat = repeatOf(node);
  return { position: new THREE.Vector3(headPosition?.[0] ?? 0, headPosition ? headPosition[1] - 2.45 : -dims[1] / 2 - 0.6, 0), name: `${node.name}${repeat > 1 ? ` ×${repeat}` : ''}${meta ? ` · ${attn!.branches > 1 ? `B${meta.branch + 1} · ` : ''}H${meta.head + 1}` : ''}` };
}

function boxCorners(box: THREE.Box3) {
  return Array.from({ length: 8 }, (_, i) => new THREE.Vector3(i & 1 ? box.max.x : box.min.x, i & 2 ? box.max.y : box.min.y, i & 4 ? box.max.z : box.min.z));
}

// The canvas area the model is framed in, in CSS pixels: the title block sits above it, the
// legend and the playback bar below.
const SAFE_AREA = { top: 112, bottom: 72, left: 28, right: 28 };

/**
 * Fit the camera to `bounds`.
 *
 * A focused part (no `overview` target) stays exactly at the centre of the canvas, which is what
 * clicking it promises, so the zoom is fitted around the orbit target.
 *
 * The whole-model overview pans instead: it zooms so the projected bounds fill the safe area,
 * then moves camera and orbit target together so the bounds sit in its middle. Fitting the
 * actual extent, rather than one mirrored around the target, keeps a model with a residual loop
 * on one side from being drawn half size, and keeps its first blocks clear of the title.
 */
function fitCamera(camera: THREE.OrthographicCamera, bounds: THREE.Box3, width: number, height: number, overview?: THREE.Vector3) {
  camera.updateMatrixWorld(true);
  const view = new THREE.Box3().setFromPoints(boxCorners(bounds).map(p => p.applyMatrix4(camera.matrixWorldInverse)));
  if (!overview) {
    const dimensions = new THREE.Vector2(2 * Math.max(Math.abs(view.min.x), Math.abs(view.max.x)), 2 * Math.max(Math.abs(view.min.y), Math.abs(view.max.y)));
    // Leave room for the scene title and the bottom playback controls.
    camera.zoom = Math.min(Math.max(100, width - 70) / (dimensions.x + 1.5), Math.max(100, height - 145) / (dimensions.y + 1), 110);
    camera.updateProjectionMatrix();
    return;
  }
  const extent = view.getSize(new THREE.Vector3()), center = view.getCenter(new THREE.Vector3());
  const safeWidth = Math.max(100, width - SAFE_AREA.left - SAFE_AREA.right), safeHeight = Math.max(100, height - SAFE_AREA.top - SAFE_AREA.bottom);
  camera.zoom = Math.min(safeWidth / (extent.x + 0.8), safeHeight / (extent.y + 0.6), 110);
  const shift = new THREE.Vector3(center.x - (SAFE_AREA.left - SAFE_AREA.right) / 2 / camera.zoom, center.y - (SAFE_AREA.bottom - SAFE_AREA.top) / 2 / camera.zoom, 0).applyQuaternion(camera.quaternion);
  camera.position.add(shift); overview.add(shift);
  camera.updateProjectionMatrix(); camera.updateMatrixWorld(true);
}

// 盒子轮廓的边几何体必须缓存：写在 JSX 里的 new THREE.BoxGeometry(...dims) 会在每次渲染时
// 重新分配并重算 EdgesGeometry，节点一多就是每帧的分配风暴。
function NodeOutline({ dims, color }: { dims: [number, number, number]; color: string }) {
  const geometry = useMemo(() => new THREE.EdgesGeometry(new THREE.BoxGeometry(dims[0], dims[1], dims[2])), [dims[0], dims[1], dims[2]]);
  useEffect(() => () => geometry.dispose(), [geometry]);
  return <lineSegments raycast={() => {}} geometry={geometry}><lineBasicMaterial color={color} transparent opacity={0.9} /></lineSegments>;
}

/** The block's own hue, darker: edges that read as the block's, not as a second colour. */
const edgeTone = (color: string) => `#${new THREE.Color(color).multiplyScalar(0.62).getHexString()}`;

/**
 * A layer drawn as the tensor it outputs: a solid block whose faces carry that tensor's cell
 * grid, so channels, rows and columns are counted on the block instead of implied by its size.
 * The footprint is unchanged, which keeps the layout, routes and image export where they were.
 */
function TensorBlock({ dims, color, glow, cells, paint }: { dims: [number, number, number]; color: string; glow: number; cells: [number, number, number]; paint: BlockPaint | null }) {
  const block = useMemo(createTensorMaterial, []);
  useEffect(() => { paintedBlocks.add(block.uniforms); return () => { paintedBlocks.delete(block.uniforms); block.material.dispose(); }; }, [block]);
  const textures = useMemo(() => ({ map: paint?.map ? floatTexture(paint.map.values, paint.map.columns, paint.map.rows) : null, strip: paint?.strip ? floatTexture(paint.strip.values, paint.strip.count, 1) : null }), [paint]);
  useEffect(() => () => { textures.map?.dispose(); textures.strip?.dispose(); }, [textures]);
  block.material.color.set(color); block.material.emissive.set(color); block.material.emissiveIntensity = glow;
  const u = block.uniforms;
  u.uCells.value.set(...cells); u.uGroups.value.set(...cells.map(majorGroup) as [number, number, number]);
  u.uHalf.value.set(dims[0] / 2, dims[1] / 2, dims[2] / 2); u.uLine.value.set(edgeTone(color));
  u.uObserve.value = paint ? 1 : 0; u.uLimit.value = paint?.limit ?? 1;
  u.uMap.value = textures.map ?? blank(); u.uStrip.value = textures.strip ?? blank();
  if (paint?.map) { u.uMapSize.value.set(paint.map.columns, paint.map.rows); u.uMapTotal.value.set(paint.map.uTotal, paint.map.vTotal); u.uMapU.value.copy(AXES[paint.map.u]); u.uMapV.value.copy(AXES[paint.map.v]); }
  else u.uMapSize.value.set(0, 0);
  if (paint?.strip) { u.uStripAxis.value.copy(AXES[paint.strip.axis]); u.uStripRange.value.set(paint.strip.total, paint.strip.start, paint.strip.count); }
  else u.uStripRange.value.set(1, 0, 0);
  return <mesh userData={{ exportMesh: true, exportShade: true, exportCells: cells }} material={block.material}><boxGeometry args={dims} /></mesh>;
}

/**
 * Multi-head Latent Attention as a waist: the input tensor, the narrower K/V latent it is squeezed
 * through (its height is kv_lora_rank over embed_dim) and the output tensor, inside the layer's own
 * footprint so the layout, routes and image export keep their places. Measured values paint the
 * output part, which is the tensor the layer reports.
 */
function LatentAttentionBlock({ dims, color, glow, input, output, latent, paint, outline }: { dims: [number, number, number]; color: string; glow: number; input?: number[]; output?: number[]; latent: number; paint: BlockPaint | null; outline: string }) {
  const [width, height, depth] = dims, gap = 0.05, side = 0.28, waist = 0.22, rest = width - side - waist - 2 * gap;
  const ratio = Math.min(0.85, Math.max(0.2, latent / Math.max(1, output?.at(-1) ?? 64)));
  const latentShape = output ? [output[0], output[1], latent] : undefined;
  const tint = (amount: number) => `#${new THREE.Color(color).lerp(new THREE.Color('#ffffff'), amount).getHexString()}`;
  const parts: { x: number; dims: [number, number, number]; color: string; cells: [number, number, number]; paint: BlockPaint | null }[] = [
    { x: -width / 2 + side / 2, dims: [side, height, depth], color: tint(0.28), cells: tensorCells(input), paint: null },
    { x: -width / 2 + side + gap + waist / 2, dims: [waist, height * ratio, depth], color: `#${new THREE.Color(color).multiplyScalar(0.86).getHexString()}`, cells: tensorCells(latentShape), paint: null },
    { x: width / 2 - rest / 2, dims: [rest, height, depth], color, cells: tensorCells(output), paint }
  ];
  return <>{parts.map((part, index) => <group key={index} position={[part.x, 0, 0]}><TensorBlock dims={part.dims} color={part.color} glow={glow} cells={part.cells} paint={part.paint} /><NodeOutline dims={part.dims} color={outline} /></group>)}</>;
}

type BlockPart = { x: number; y?: number; dims: [number, number, number]; color: string; cells: [number, number, number]; paint: BlockPaint | null };
const tintOf = (color: string, amount: number) => `#${new THREE.Color(color).lerp(new THREE.Color('#ffffff'), amount).getHexString()}`;
const shadeOf = (color: string, amount: number) => `#${new THREE.Color(color).multiplyScalar(amount).getHexString()}`;
const drawParts = (parts: BlockPart[], glow: number, outline: string) => <>{parts.map((part, index) => <group key={index} position={[part.x, part.y ?? 0, 0]}><TensorBlock dims={part.dims} color={part.color} glow={glow} cells={part.cells} paint={part.paint} /><NodeOutline dims={part.dims} color={outline} /></group>)}</>;

/** At most this many bars stand for the routed experts; a larger layer gives each bar a run of them. */
const MAX_DRAWN_EXPERTS = 16;
/**
 * A mixture-of-experts layer as its parts, inside its own footprint: the gate (its grid is tokens by
 * experts), one bar per routed expert (or per run of experts past 16), the shared experts when there
 * are any, and the output tensor, which measured values paint. A measured routing shades each bar by
 * the tokens its experts received, from pale (none) to the full colour (the busiest bar).
 */
function MoEBlock({ dims, color, glow, output, experts, shared, paint, outline, load }: { dims: [number, number, number]; color: string; glow: number; output?: number[]; experts: number; shared: number; paint: BlockPaint | null; outline: string; load?: number[] }) {
  const [width, height, depth] = dims, gap = 0.05, gate = 0.16, column = 0.42, sharedWidth = shared ? 0.12 : 0;
  const rest = width - gate - column - sharedWidth - gap * (shared ? 3 : 2);
  const per = Math.ceil(Math.max(1, experts) / MAX_DRAWN_EXPERTS), drawn = Math.ceil(Math.max(1, experts) / per);
  const spacing = Math.min(0.04, height * 0.15 / drawn), bar = (height - spacing * (drawn - 1)) / drawn;
  const barLoads = load ? Array.from({ length: drawn }, (_, index) => load.slice(index * per, (index + 1) * per).reduce((sum, count) => sum + count, 0)) : null;
  const busiest = barLoads ? Math.max(1, ...barLoads) : 1;
  const barColor = (index: number) => barLoads ? `#${new THREE.Color(tintOf(color, 0.82)).lerp(new THREE.Color(shadeOf(color, 0.82)), barLoads[index] / busiest).getHexString()}` : color;
  const columnX = -width / 2 + gate + gap + column / 2;
  const parts: BlockPart[] = [
    { x: -width / 2 + gate / 2, dims: [gate, height, depth], color: tintOf(color, 0.32), cells: tensorCells(output ? [output[0], product(output.slice(1, -1)), experts] : undefined), paint: null },
    ...Array.from({ length: drawn }, (_, index): BlockPart => ({ x: columnX, y: height / 2 - bar / 2 - index * (bar + spacing), dims: [column, bar, depth * 0.85], color: barColor(index), cells: [1, 1, 1], paint: null })),
    ...(shared ? [{ x: columnX + column / 2 + gap + sharedWidth / 2, dims: [sharedWidth, height * 0.9, depth] as [number, number, number], color: shadeOf(color, 0.78), cells: [1, 1, 1] as [number, number, number], paint: null }] : []),
    { x: width / 2 - rest / 2, dims: [rest, height, depth], color, cells: tensorCells(output), paint }
  ];
  return drawParts(parts, glow, outline);
}

const AXES = [new THREE.Vector3(1, 0, 0), new THREE.Vector3(0, 1, 0), new THREE.Vector3(0, 0, 1)];
// Every mounted block, so an image export can leave out measured values: an exported diagram
// shows the architecture, not one sample's activations.
const paintedBlocks = new Set<TensorUniforms>();
function withoutPaint<T>(work: () => T): T {
  const saved = [...paintedBlocks].map(uniforms => [uniforms, uniforms.uObserve.value] as const);
  saved.forEach(([uniforms]) => { uniforms.uObserve.value = 0; });
  try { return work(); } finally { saved.forEach(([uniforms, value]) => { uniforms.uObserve.value = value; }); }
}

const GROUND_Y = -0.09;
let shadowTexture: THREE.CanvasTexture | null = null;
/** One soft radial falloff shared by every shadow; its green channel is read as alpha. */
function groundShadowTexture() {
  if (shadowTexture) return shadowTexture;
  const canvas = document.createElement('canvas'); canvas.width = canvas.height = 128;
  const context = canvas.getContext('2d')!, gradient = context.createRadialGradient(64, 64, 6, 64, 64, 63);
  gradient.addColorStop(0, '#ffffff'); gradient.addColorStop(0.45, '#8c8c8c'); gradient.addColorStop(1, '#000000');
  context.fillStyle = gradient; context.fillRect(0, 0, 128, 128);
  shadowTexture = new THREE.CanvasTexture(canvas);
  return shadowTexture;
}

/**
 * A soft shadow on the ground under a block, so the model reads as standing in space rather than
 * floating. It is decoration only: not clickable, not part of the image export's geometry, and
 * edge-on (invisible) in the straight-on compact export.
 */
function GroundShadow({ width, depth, lift }: { width: number; depth: number; lift: number }) {
  const texture = useMemo(groundShadowTexture, []);
  // Higher above the ground, wider and fainter, the way a contact shadow behaves.
  const spread = 1.3 + Math.min(lift, 6) * 0.08, opacity = Math.max(0.1, 0.26 - Math.min(lift, 6) * 0.025);
  return <mesh raycast={() => {}} rotation={[-Math.PI / 2, 0, 0]} position={[0, -lift, 0]} renderOrder={-1}>
    <planeGeometry args={[width * spread + 0.4, depth * spread + 0.4]} />
    <meshBasicMaterial color="#3f5c67" alphaMap={texture} transparent opacity={opacity} depthWrite={false} />
  </mesh>;
}

// 每节点一个 <Html> 标签是最重的成本：它们不受 three.js 视锥剔除约束，节点一多就
// 全是常驻 DOM。按缩放分三档，远景只留骨架，这是「低缩放聚合」的渲染侧。
type Detail = 'far' | 'mid' | 'near';
const KERNEL_COLOR = '#f97316';
const UNIT_EDGES = new THREE.EdgesGeometry(new THREE.BoxGeometry(1, 1, 1));

/**
 * The block-local box of a range of elements of a `[1,C,H,W]` block, laid out as tensorBlock.ts
 * draws its cells: channel 0 at the front, row 0 at the top, column 0 nearest the viewer. A
 * small lift keeps the highlight just outside the faces it covers.
 */
function elementBox(dims: [number, number, number], shape: number[], field: Field) {
  const along = (size: number, count: number, [first, last]: [number, number]) => {
    const top = size / 2 - first * size / count, bottom = size / 2 - (last + 1) * size / count;
    return { center: (top + bottom) / 2, size: top - bottom + 0.012 };
  };
  const x = along(dims[0], shape[1], field.channels), y = along(dims[1], shape[2], field.rows), z = along(dims[2], shape[3], field.columns);
  return { center: new THREE.Vector3(x.center, y.center, z.center), size: new THREE.Vector3(x.size, y.size, z.size) };
}

interface KernelProps { node: Layer; input: number[]; output: number[]; from: THREE.Vector3; to: THREE.Vector3; fromDims: [number, number, number]; toDims: [number, number, number] }

/**
 * One output element of a convolution or pooling layer and the input window it reads: the
 * window on the input block (through every channel a convolution reads, or the one channel a
 * pooling reads), the element on the layer's own block, and four beams between them. One window
 * per layer: it sweeps the output grid while the data flow plays and holds still while paused.
 * With measured values painted, the element shows its measured value.
 */
function KernelWindow({ node, input, output, from, to, fromDims, toDims, playing, speed, measured }: KernelProps & { playing: boolean; speed: number; measured?: (row: number, column: number) => number | null | undefined }) {
  const [step, setStep] = useState(0);
  const clock = useRef(0);
  const total = output[2] * output[3];
  useFrame((_, delta) => {
    if (!playing) return;
    clock.current += delta * speed;
    const next = Math.floor(clock.current * 1.5) % total;
    setStep(current => current === next ? current : next);
  });
  const beams = useMemo(() => new THREE.BufferGeometry(), []);
  useEffect(() => () => beams.dispose(), [beams]);
  const row = Math.floor(step / output[3]) % output[2], column = step % output[3];
  const field = receptiveField(node, input, output, row, column);
  if (!field) return null;
  const window = elementBox(fromDims, input, field), cell = elementBox(toDims, output, { channels: [0, 0], rows: [row, row], columns: [column, column] });
  const windowCenter = from.clone().add(window.center), cellCenter = to.clone().add(cell.center);
  // From the window's face toward the layer to the facing side of the element.
  const corners: number[] = [];
  for (const [dy, dz] of [[1, 1], [1, -1], [-1, 1], [-1, -1]]) corners.push(
    windowCenter.x + window.size.x / 2, windowCenter.y + dy * window.size.y / 2, windowCenter.z + dz * window.size.z / 2,
    cellCenter.x - cell.size.x / 2, cellCenter.y + dy * cell.size.y / 2, cellCenter.z + dz * cell.size.z / 2);
  beams.setAttribute('position', new THREE.Float32BufferAttribute(corners, 3));
  const value = measured?.(row, column);
  return <group name="scene_overlay">
    <mesh raycast={() => {}} position={windowCenter} scale={window.size} renderOrder={2}><boxGeometry /><meshBasicMaterial color={KERNEL_COLOR} transparent opacity={0.22} depthWrite={false} /></mesh>
    <lineSegments raycast={() => {}} position={windowCenter} scale={window.size} geometry={UNIT_EDGES}><lineBasicMaterial color={KERNEL_COLOR} /></lineSegments>
    <mesh raycast={() => {}} position={cellCenter} scale={cell.size}><boxGeometry /><meshBasicMaterial color={KERNEL_COLOR} /></mesh>
    <lineSegments raycast={() => {}} geometry={beams} frustumCulled={false}><lineBasicMaterial color={KERNEL_COLOR} transparent opacity={0.7} /></lineSegments>
    {value !== undefined && <Html position={[cellCenter.x, cellCenter.y + cell.size.y / 2 + 0.16, cellCenter.z]} center zIndexRange={[11, 0]} style={{ pointerEvents: 'none' }}><div className="kernel-value">= {value === null ? '—' : tensorNumber(value)}</div></Html>}
  </group>;
}

// The default framing of a template lands around zoom 35–45; every layer keeps its label there.
const DETAIL_THRESHOLDS: [number, number] = [16, 32];

function useDetailTier(): Detail {
  const camera = useThree(state => state.camera);
  const [tier, setTier] = useState<Detail>('near');
  // 分档必须量化：useFrame 每帧都跑，而缩放值在拖拽时每帧都变，
  // 直接 setState 会让整个场景每帧重渲染。
  useFrame(() => {
    const zoom = (camera as THREE.OrthographicCamera).zoom;
    const next: Detail = zoom < DETAIL_THRESHOLDS[0] ? 'far' : zoom < DETAIL_THRESHOLDS[1] ? 'mid' : 'near';
    setTier(current => (current === next ? current : next));
  });
  return tier;
}

/** 远景只标注骨架：结构块、输入输出和当前选中项，其余算子的标签不建立 DOM。 */
function labelVisible(node: Layer, tier: Detail, selectedId: string | null, hovered: string | null): boolean {
  if (tier === 'near') return true;
  if (node.op === 'Group' || node.op === 'Input' || node.op === 'Output') return true;
  if (node.id === selectedId || node.id === hovered) return true;
  return tier === 'mid' && repeatOf(node) > 1;
}

/**
 * Hide the labels that would cover a more important one. A few times a second, the labels are
 * placed greedily by priority and graph order, and one that overlaps an already placed label is
 * hidden until zooming in makes room. Along a chain this keeps every other label, spread evenly,
 * instead of a staircase of pills covering each other's shapes.
 */
function useLabelDeclutter(labels: RefObject<Map<string, HTMLDivElement>>, rank: (id: string) => number) {
  const frame = useRef(0);
  useFrame(() => {
    if (++frame.current % 8) return;
    const placed: DOMRect[] = [];
    const ordered = [...labels.current.entries()].sort(([a], [b]) => rank(a) - rank(b));
    for (const [, element] of ordered) {
      const box = element.getBoundingClientRect();
      const blocked = placed.some(other => box.left < other.right + 3 && box.right > other.left - 3 && box.top < other.bottom + 2 && box.bottom > other.top - 2);
      // An attribute rather than a class: React owns className and rewrites it on re-render.
      element.toggleAttribute('data-decluttered', blocked);
      if (!blocked) placed.push(box);
    }
  });
}

function World(props: Props) {
  const { gl, camera, scene, size } = useThree();
  const detail = useDetailTier();
  // Keep label roots in one DOM container when the canvas event target changes.
  const labelPortal = useMemo(() => ({ current: gl.domElement.parentElement! }), [gl]);
  const controls = useRef<any>(null);
  const flowDirection = useRef(props.direction); flowDirection.current = props.direction;
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
  // A convolution or pooling layer fed by a feature map gets its one kernel window.
  const kernels = useMemo(() => props.showKernels === false ? [] : props.graph.nodes.flatMap<KernelProps>(node => {
    if (!WINDOW_OPS.has(node.op)) return [];
    const edge = props.graph.edges.find(e => e.target === node.id && !e.targetPort), source = edge && props.graph.nodes.find(n => n.id === edge.source);
    const input = edge && props.analysis.layers[edge.source]?.output, output = props.analysis.layers[node.id]?.output;
    if (!edge || !source || !input || !output || input.length !== 4 || output.length !== 4 || !positions[source.id] || !positions[node.id]) return [];
    return [{ node, input, output, from: positions[source.id], to: positions[node.id], fromDims: contentsDimensions(input, source.op, source, props.expanded), toDims: contentsDimensions(output, node.op, node, props.expanded) }];
  }), [props.graph, props.analysis, positions, props.expanded, props.showKernels]);
  // The measured value of a layer's output element, when the painted window holds channel 0 there.
  const measuredAt = (id: string) => {
    const tensor = props.observation?.tensors[id];
    if (!tensor || tensor.shape.length !== 4 || tensor.slice.indices[1] !== 0) return undefined;
    return (row: number, column: number) => row < tensor.slice.rows && column < tensor.slice.columns ? tensor.slice.values[row][column] : undefined;
  };
  // One paint per observed layer, kept stable per observation so block textures are built once.
  const paints = useMemo(() => {
    const observation = props.observation;
    if (!observation) return {} as Record<string, BlockPaint | null>;
    return Object.fromEntries(props.graph.nodes.map(node => [node.id, observation.tensors[node.id] ? blockPaint(observation.tensors[node.id]) : null]));
  }, [props.observation, props.graph]);
  // Selected first, then hovered, then the skeleton (inputs, outputs, blocks), then graph order.
  const labelElements = useRef(new Map<string, HTMLDivElement>());
  const labelOrder = useMemo(() => new Map(props.graph.nodes.map((node, index) => [node.id, { index, skeleton: node.op === 'Input' || node.op === 'Output' || node.op === 'Group' }])), [props.graph]);
  useLabelDeclutter(labelElements, id => {
    const entry = labelOrder.get(id);
    const tier = id === props.selected ? 0 : id === hover ? 1 : entry?.skeleton ? 2 : 3;
    return tier * 100000 + (entry?.index ?? 99999);
  });
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
    const merge = node.op === 'Add' || node.op === 'Multiply' || node.op === 'Divide';
    if (merge) routes.filter(route => route.target === id).forEach(route => {
      flowCurve(route).getPoints(40).forEach(p => focusBox.expandByPoint(p));
      props.graph.nodes.filter(n => positions[n.id].x >= route.start.x && positions[n.id].x <= route.end.x).forEach(n => focusBox.union(new THREE.Box3().setFromCenterAndSize(positions[n.id], new THREE.Vector3(...contentsDimensions(props.analysis.layers[n.id]?.output, n.op, n, props.expanded)))));
    });
    const center = headPoint ? target.clone().add(new THREE.Vector3(...headPoint)) : target.clone();
    const radius = attention ? qkvRadius(attention.qkvCount) : 0;
    const dims = headPoint ? new THREE.Vector3(2.1, Math.max(5, radius * 2 + 2.8), Math.max(4.4, radius * 2 + 1.3)) : new THREE.Vector3(...contentsDimensions(props.analysis.layers[id]?.output, node.op, node, props.expanded));
    centerView(center, dims, attention ? (headPoint ? new THREE.Vector3(24, 5, 18) : new THREE.Vector3(18, 7, 24)) : new THREE.Vector3(4, 8, 12), merge ? focusBox : undefined);
  };
  useEffect(() => {
    const context = gl.getContext(), ext = context.getExtension('WEBGL_debug_renderer_info');
    const name = ext ? context.getParameter(ext.UNMASKED_RENDERER_WEBGL) : 'WebGL 2'; props.onGpu(String(name));
  }, [gl, props.onGpu]);
  useEffect(() => {
    const reset = () => {
      const center = bounds.getCenter(new THREE.Vector3());
      camera.position.copy(center.clone().add(attentionView ? new THREE.Vector3(14, 8, 26) : new THREE.Vector3(15, 9, 18))); camera.lookAt(center);
      const target = center.clone();
      fitCamera(camera as THREE.OrthographicCamera, bounds, size.width, size.height, target);
      if (controls.current) { controls.current.target.copy(target); controls.current.update(); }
    };
    reset();
    const exportSource = { scene, camera: camera as THREE.OrthographicCamera, graph: props.graph, analysis: props.analysis, expanded: props.expanded, viewport: size, direction: () => flowDirection.current,
      dimensions: Object.fromEntries(props.graph.nodes.map(node => [node.id, contentsDimensions(props.analysis.layers[node.id]?.output, node.op, node, props.expanded)])) };
    props.onReady({
      reset,
      focus: id => focus(id),
      focusHead: (id, index) => focus(id, index),
      // The render happens before captureSceneImage first awaits, so the paint is back right after.
      capture: (width, transparent, options) => withoutPaint(() => captureSceneImage(gl, exportSource, width, transparent, options)),
      svg: (options, exportWidth = 1920) => exportSceneSvg(exportSource, exportWidth, options)
    });
  }, [camera, gl, scene, positions, routes, size, bounds, attentionView, props.expanded, props.graph, props.analysis, props.onReady]);
  useEffect(() => { if (props.selected && props.selectedHead !== null && isAttention(props.graph.nodes.find(n => n.id === props.selected)?.op || '')) focus(props.selected, props.selectedHead); }, [camera, props.selected, props.selectedHead, props.expanded, positions, size]);
  return <>
    {/* No scene background: the canvas is transparent over the workspace's soft gradient. */}
    {/* Sky/ground fill plus a key light from the upper right and a weak rim from behind, so a
        block's top, front and side read as three distinct tones of its one colour. */}
    <hemisphereLight args={['#ffffff', '#b9c8cf', 1.7]} /><directionalLight position={[9, 16, 11]} intensity={1.9} /><directionalLight position={[-10, 5, -8]} intensity={0.55} />
    <Grid name="export_grid" position={[0, -0.1, 0]} args={[80, 40]} cellSize={1} cellThickness={0.5} cellColor="#e0e8eb" sectionSize={5} sectionThickness={0.85} sectionColor="#cad6db" fadeDistance={55} fadeStrength={1.6} infiniteGrid />
    <OrbitControls ref={controls} makeDefault enableDamping={false} minZoom={1.5} maxZoom={180} />
    {routes.map(route => {
      const flow = { start: route.start, end: route.end, control1: route.control1, control2: route.control2 };
      return <group key={route.id} name={`edge_${route.id}`}>{route.residual ? <ResidualFlow {...flow} active={props.playing} direction={props.direction} speed={props.speed} /> : <><EdgeLine {...flow} color={props.analysis.layers[route.target] ? '#7095a6' : '#e26969'} />{detail !== 'far' && <ParticleEdge {...flow} active={props.playing} direction={props.direction} speed={props.speed} color={props.direction === 'forward' ? '#31adce' : '#ee706d'} intensity={Math.log2(props.analysis.layers[route.source]?.output.reduce((a, b) => a * b, 1) || 1)} compact={detail === 'mid'} />}</>}</group>;
    })}
    {kernels.map(kernel => <KernelWindow key={kernel.node.id} {...kernel} playing={props.playing} speed={props.speed} measured={measuredAt(kernel.node.id)} />)}
    {props.graph.nodes.map(n => {
      const pos = positions[n.id], dims = contentsDimensions(props.analysis.layers[n.id]?.output, n.op, n, props.expanded);
      const issue = props.diagnostics.find(d => d.nodeId === n.id && d.level === 'error'), selected = props.selected === n.id;
      const color = issue ? '#e45555' : COLORS[n.op];
      const label = layerLabel(n, dims, props.selected, props.selectedHead, props.expanded);
      return <group key={n.id} position={pos} name={isAttention(n.op) ? `attention_${n.id}` : n.op === 'Add' ? `add_${n.id}` : `node_${n.id}`} onClick={e => { e.stopPropagation(); props.onSelect(n.id); focus(n.id); }} onPointerOver={e => { e.stopPropagation(); setHover(n.id); document.body.style.cursor = 'pointer'; }} onPointerOut={() => { setHover(null); document.body.style.cursor = ''; }}>
        {isAttention(n.op) ? <AttentionModule node={n} selected={selected} selectedHead={selected ? props.selectedHead : null} selectedProjection={selected ? props.selectedProjection : null} onSelectProjection={(port, center) => { props.onSelectProjection(n.id, port); centerView(center, new THREE.Vector3(3.2, 3.8, 3.2), new THREE.Vector3(24, 5, 18)); }} overriddenPorts={props.graph.edges.filter(e => e.target === n.id && isProjectionPort(e.targetPort)).map(e => e.targetPort!)} attachedPorts={props.graph.edges.some(e => e.source === n.id && e.sourcePort || e.target === n.id && isProjectionPort(e.targetPort))} onSelectHead={index => { props.onSelectHead(n.id, index); focus(n.id, index); }} onFocusPart={center => { props.onSelect(n.id); centerView(center, new THREE.Vector3(2.1, 2.4, 2.1), new THREE.Vector3(18, 7, 24)); }} playing={props.playing} rotating={props.rotating} direction={props.direction} speed={props.speed} expanded={props.expanded} error={Boolean(issue)} /> : <>
        {/* Solid, matte blocks: a translucent block blended with whatever stood behind it, which
            muddied both colours wherever neighbours overlapped. A merge node's small cube stands
            for the operation, not for a tensor, so it has no grid. */}
        {n.op === 'LatentAttention' ? <LatentAttentionBlock dims={dims} color={color} glow={issue ? 0.4 : selected ? 0.22 : hover === n.id ? 0.12 : 0} input={props.analysis.layers[n.id]?.input[0]} output={props.analysis.layers[n.id]?.output} latent={Number(n.params.kv_lora_rank ?? 32)} paint={paints[n.id] ?? null} outline={selected ? '#1d3943' : edgeTone(color)} /> :
        n.op === 'MoE' ? <MoEBlock dims={dims} color={color} glow={issue ? 0.4 : selected ? 0.22 : hover === n.id ? 0.12 : 0} output={props.analysis.layers[n.id]?.output} experts={Number(n.params.experts ?? 8)} shared={Number(n.params.shared_experts ?? 1)} paint={paints[n.id] ?? null} outline={selected ? '#1d3943' : edgeTone(color)} load={props.observation?.loads[n.id]} /> : <>
        <TensorBlock dims={dims} color={color} glow={issue ? 0.4 : selected ? 0.22 : hover === n.id ? 0.12 : 0} cells={n.op === 'Add' || n.op === 'Multiply' || n.op === 'Divide' ? [1, 1, 1] : tensorCells(props.analysis.layers[n.id]?.output)} paint={paints[n.id] ?? null} />
        <NodeOutline dims={dims} color={selected ? '#1d3943' : edgeTone(color)} /></>}{n.op === 'Add' && <SceneText text="+" position={new THREE.Vector3(0, 0.44, 0)} color={color} scale={0.38} />}{n.op === 'Multiply' && <SceneText text="×" position={new THREE.Vector3(0, 0.44, 0)} color={color} scale={0.38} />}{n.op === 'Divide' && <SceneText text="÷" position={new THREE.Vector3(0, 0.44, 0)} color={color} scale={0.38} />}</>}
        {/* An attention module is an open frame of small parts; one shadow under its whole footprint read as a smudge. */}
        {!isAttention(n.op) && <GroundShadow width={dims[0]} depth={dims[2]} lift={pos.y - GROUND_Y} />}
        {selected && !isAttention(n.op) && <mesh raycast={() => {}} position={[0, -dims[1] / 2 - 0.1, 0]} rotation={[-Math.PI / 2, 0, 0]}><ringGeometry args={[0.45, 0.49, 48]} /><meshBasicMaterial color="#1d3943" transparent opacity={0.75} /></mesh>}
        {labelVisible(n, detail, props.selected, hover) && <Html portal={labelPortal} position={label.position} center zIndexRange={[10, 0]} style={{ pointerEvents: 'none' }}><div ref={element => { if (element) labelElements.current.set(n.id, element); else labelElements.current.delete(n.id); }} className={`scene-label ${selected ? 'selected' : ''}`}><strong>{label.name}</strong>{detail === 'near' && <span>{shapeText(props.analysis.layers[n.id]?.output)}</span>}</div></Html>}
      </group>;
    })}
  </>;
}

class SceneBoundary extends Component<{ children: ReactNode; failure: string }, { error: boolean }> {
  state = { error: false };
  static getDerivedStateFromError() { return { error: true }; }
  render() { return this.state.error ? <div className="empty-state">{this.props.failure}</div> : this.props.children; }
}
export default function Scene(props: Props) {
  const { t } = useI18n();
  return <SceneBoundary failure={t('WebGL 初始化失败。请启用浏览器硬件加速，或切换到拓扑图。')}><Canvas className="scene-canvas" flat gl={{ antialias: true, alpha: true, preserveDrawingBuffer: true, powerPreference: 'high-performance' }} dpr={[1, 2]} onPointerMissed={() => props.onSelect(null)}><OrthographicCamera makeDefault position={[15, 11, 18]} near={0.1} far={200} zoom={35} /><World {...props} /></Canvas></SceneBoundary>;
}
