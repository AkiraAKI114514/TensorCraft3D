import { Component, useEffect, useMemo, useRef, useState, type CSSProperties, type ReactNode, type RefObject } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { Html, OrbitControls, OrthographicCamera, Grid } from '@react-three/drei';
import * as THREE from 'three';
import { RoomEnvironment } from 'three/examples/jsm/environments/RoomEnvironment.js';
import { EffectComposer } from 'three/examples/jsm/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/examples/jsm/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/examples/jsm/postprocessing/UnrealBloomPass.js';
import { OutputPass } from 'three/examples/jsm/postprocessing/OutputPass.js';
import { COLORS, repeatOf, type Graph, type Analysis, type Diagnostic, type Layer } from './types';
import { product, shapeText } from './analysis';
import { EdgeLine, ParticleEdge, ResidualFlow } from './Flow';
import { FLOW_GLOW } from './flowGeometry';
import { SceneThemeContext, useSceneLook, type SceneTheme } from './sceneTheme';
import { EDGE_COLOR, forExport, LOOKS } from './sceneLook';
import AttentionModule, { SceneText } from './AttentionModule';
import { isAttention, attentionLayout, qkvRadius } from './attentionLayout';
import { isProjectionPort } from './attentionConfig';
import { flowCurve, isMerge, OVERVIEW_OFFSETS, sceneLayout, type SceneNode } from './sceneLayout';
import { captureSceneImage, exportSceneSvg } from './sceneImageExport';
import { blank, createTensorMaterial, floatTexture, majorGroup, NIGHT, tensorCells, type TensorUniforms } from './tensorBlock';
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
  showKernels?: boolean;
  /** Pixels at the right edge an overlay covers (the open legend); the overview keeps the model clear of them. */
  insetRight?: number;
  /** The live look; exported images always use the day look. */
  theme?: SceneTheme; }
function layerLabel(entry: SceneNode, selected: string | null, head: number | null, expanded: boolean) {
  const { node } = entry;
  const headPosition = selected === node.id && head !== null && isAttention(node.op) ? attentionLayout(node, expanded).heads[head] : undefined;
  const attn = headPosition ? attentionLayout(node, expanded) : null, meta = head !== null ? attn?.headMeta[head] : undefined;
  // A folded layer stacks `repeat` identical instances; the ×N badge flags it.
  const repeat = repeatOf(node);
  // A selected head is labelled at the head; every other label hangs on its row's baseline.
  return { position: new THREE.Vector3(headPosition?.[0] ?? 0, headPosition ? headPosition[1] - 2.45 : entry.labelY - entry.position.y, 0), name: `${node.name}${repeat > 1 ? ` ×${repeat}` : ''}${meta ? ` · ${attn!.branches > 1 ? `B${meta.branch + 1} · ` : ''}H${meta.head + 1}` : ''}` };
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
function fitCamera(camera: THREE.OrthographicCamera, bounds: THREE.Box3, width: number, height: number, overview?: THREE.Vector3, insetRight = 0) {
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
  // An open legend covers the right edge; on a narrow canvas the model keeps at least half the width.
  const right = Math.max(SAFE_AREA.right, Math.min(insetRight, width / 2 - SAFE_AREA.left));
  const safeWidth = Math.max(100, width - SAFE_AREA.left - right), safeHeight = Math.max(100, height - SAFE_AREA.top - SAFE_AREA.bottom);
  camera.zoom = Math.min(safeWidth / (extent.x + 0.8), safeHeight / (extent.y + 0.6), 110);
  const shift = new THREE.Vector3(center.x - (SAFE_AREA.left - right) / 2 / camera.zoom, center.y - (SAFE_AREA.bottom - SAFE_AREA.top) / 2 / camera.zoom, 0).applyQuaternion(camera.quaternion);
  camera.position.add(shift); overview.add(shift);
  camera.updateProjectionMatrix(); camera.updateMatrixWorld(true);
}

// 盒子轮廓的边几何体必须缓存：写在 JSX 里的 new THREE.BoxGeometry(...dims) 会在每次渲染时
// 重新分配并重算 EdgesGeometry，节点一多就是每帧的分配风暴。
/** The outline of the selected layer, dark by day and luminous at night; any other outline is a faint tone of its own block. */
const SELECTED_OUTLINE = '#1d3943', NIGHT_SELECTED_OUTLINE = '#9cecff';
function NodeOutline({ dims, color }: { dims: [number, number, number]; color: string }) {
  const night = useSceneLook() === 'night';
  const geometry = useMemo(() => new THREE.EdgesGeometry(new THREE.BoxGeometry(dims[0], dims[1], dims[2])), [dims[0], dims[1], dims[2]]);
  useEffect(() => () => geometry.dispose(), [geometry]);
  // Dark, opaque edges on every block drew the model like a cartoon; shading separates the faces
  // by day, and at night the faces' own glowing borders do.
  const selected = color === SELECTED_OUTLINE || color === NIGHT_SELECTED_OUTLINE;
  return <lineSegments raycast={() => {}} geometry={geometry} userData={{ dayMaterial: { color: selected ? SELECTED_OUTLINE : color, opacity: selected ? 0.95 : 0.38 } }}><lineBasicMaterial color={color} transparent opacity={selected ? 0.95 : night ? 0.16 : 0.38} /></lineSegments>;
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

const MERGE_SYMBOLS: Record<string, string> = { Add: '+', Multiply: '×', Divide: '÷' };
const MERGE_SYMBOL_POSITION = new THREE.Vector3(0, 0.44, 0);

/**
 * A layer other than an attention module, as its tensor: a solid, matte block (a translucent one
 * blended with whatever stood behind it and muddied both colours), or its parts for latent
 * attention and MoE. A merge's small cube stands for the operation, not a tensor, so it has no
 * grid and carries its symbol instead.
 */
function LayerBody({ node, dims, color, glow, outline, input, output, paint, load }: { node: Layer; dims: [number, number, number]; color: string; glow: number; outline: string; input?: number[]; output?: number[]; paint: BlockPaint | null; load?: number[] }) {
  if (node.op === 'LatentAttention') return <LatentAttentionBlock dims={dims} color={color} glow={glow} input={input} output={output} latent={Number(node.params.kv_lora_rank ?? 32)} paint={paint} outline={outline} />;
  if (node.op === 'MoE') return <MoEBlock dims={dims} color={color} glow={glow} output={output} experts={Number(node.params.experts ?? 8)} shared={Number(node.params.shared_experts ?? 1)} paint={paint} outline={outline} load={load} />;
  const symbol = MERGE_SYMBOLS[node.op];
  return <>
    <TensorBlock dims={dims} color={color} glow={glow} cells={symbol ? [1, 1, 1] : tensorCells(output)} paint={paint} />
    <NodeOutline dims={dims} color={outline} />
    {symbol && <SceneText text={symbol} position={MERGE_SYMBOL_POSITION} color={color} scale={0.38} />}
  </>;
}

const AXES = [new THREE.Vector3(1, 0, 0), new THREE.Vector3(0, 1, 0), new THREE.Vector3(0, 0, 1)];
// Every mounted block, so an image export can leave out measured values: an exported diagram
// shows the architecture, not one sample's activations.
const paintedBlocks = new Set<TensorUniforms>();

const GROUND_Y = -0.09;
/** The night stage: a deep blue backdrop that lifts behind the model, drawn as the scene background. */
function NightBackdrop() {
  const scene = useThree(state => state.scene);
  useEffect(() => {
    const canvas = document.createElement('canvas'); canvas.width = canvas.height = 512;
    const context = canvas.getContext('2d')!, gradient = context.createRadialGradient(256, 210, 16, 256, 256, 400);
    gradient.addColorStop(0, '#173650'); gradient.addColorStop(0.42, '#0b1b2a'); gradient.addColorStop(1, '#03070c');
    context.fillStyle = gradient; context.fillRect(0, 0, 512, 512);
    const texture = new THREE.CanvasTexture(canvas); texture.colorSpace = THREE.SRGBColorSpace;
    const previous = scene.background; scene.background = texture;
    return () => { scene.background = previous; texture.dispose(); };
  }, [scene]);
  return null;
}

/**
 * Bloom at night: the scene is rendered into a half-float target and everything brighter than the
 * glowing lines' threshold spills into soft light, then written out in the canvas's colour space.
 * Rendering this way replaces react-three-fiber's own render each frame while it is mounted.
 */
function Bloom() {
  const gl = useThree(state => state.gl), scene = useThree(state => state.scene), camera = useThree(state => state.camera), size = useThree(state => state.size);
  const composer = useMemo(() => {
    const composer = new EffectComposer(gl);
    composer.addPass(new RenderPass(scene, camera));
    composer.addPass(new UnrealBloomPass(new THREE.Vector2(256, 256), 0.72, 0.5, 0.56));
    composer.addPass(new OutputPass());
    return composer;
  }, [gl, scene, camera]);
  useEffect(() => { composer.setPixelRatio(gl.getPixelRatio()); composer.setSize(size.width, size.height); }, [composer, gl, size]);
  useEffect(() => () => composer.dispose(), [composer]);
  useFrame(() => composer.render(), 1);
  return null;
}
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
function GroundShadow({ width, depth, lift, color }: { width: number; depth: number; lift: number; color: string }) {
  const night = useSceneLook() === 'night';
  const texture = useMemo(groundShadowTexture, []);
  // Higher above the ground, wider and fainter, the way a contact shadow behaves; a block well
  // above the floor (a row raised for an attention module) casts none, rather than a stray smudge.
  const fade = 1 - THREE.MathUtils.smoothstep(lift, 1.9, 2.8), spread = 1.4 + Math.min(lift, 6) * 0.1;
  if (fade <= 0.02) return null;
  // At night a block casts light, not shadow: a pool of its own colour on the dark floor, which the
  // export leaves out like every night-only decoration. The day shadow stays mounted, hidden, so an
  // export draws it whatever the live look.
  return <>
    {night && <mesh name="scene_overlay" raycast={() => {}} rotation={[-Math.PI / 2, 0, 0]} position={[0, -lift, 0]} renderOrder={-1}>
      <planeGeometry args={[width * spread * 1.5 + 1, depth * spread * 1.5 + 1]} />
      <meshBasicMaterial color={color} alphaMap={texture} transparent opacity={0.42 * fade} blending={THREE.AdditiveBlending} depthWrite={false} />
    </mesh>}
    <mesh raycast={() => {}} visible={!night} userData={{ dayVisible: true }} rotation={[-Math.PI / 2, 0, 0]} position={[0, -lift, 0]} renderOrder={-1}>
      <planeGeometry args={[width * spread + 0.5, depth * spread + 0.5]} />
      <meshBasicMaterial color="#1c2d35" alphaMap={texture} transparent opacity={0.22 * fade} depthWrite={false} />
    </mesh>
  </>;
}

/** At night, a wide cyan glow on the floor under the whole model, the stage it stands on. */
function StageGlow({ bounds }: { bounds: THREE.Box3 }) {
  const texture = useMemo(groundShadowTexture, []);
  const center = bounds.getCenter(new THREE.Vector3()), size = bounds.getSize(new THREE.Vector3());
  return <mesh name="scene_overlay" raycast={() => {}} rotation={[-Math.PI / 2, 0, 0]} position={[center.x, GROUND_Y - 0.005, center.z]} renderOrder={-2}>
    <planeGeometry args={[size.x * 1.35 + 10, size.z + 16]} />
    <meshBasicMaterial color="#1d6f9a" alphaMap={texture} transparent opacity={0.38} blending={THREE.AdditiveBlending} depthWrite={false} />
  </mesh>;
}

/**
 * Soft studio light: a room environment, prefiltered once, lights every lit material from all
 * sides, so a block's faces shade smoothly instead of as three flat tones under bare lamps. It is
 * built locally from three.js's own room model; nothing is downloaded.
 */
function StudioEnvironment() {
  const gl = useThree(state => state.gl), scene = useThree(state => state.scene);
  useEffect(() => {
    const generator = new THREE.PMREMGenerator(gl), room = new RoomEnvironment();
    const texture = generator.fromScene(room, 0.04).texture;
    scene.environment = texture;
    return () => { scene.environment = null; texture.dispose(); generator.dispose(); room.dispose(); };
  }, [gl, scene]);
  return null;
}

// 每节点一个 <Html> 标签是最重的成本：它们不受 three.js 视锥剔除约束，节点一多就
// 全是常驻 DOM。按相邻阶段在屏幕上的间距分三档：远景只留骨架，中景每层只标名称，
// 近景再加形状；重叠的由 useLabelDeclutter 隐藏。
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

/**
 * On-screen pixels between neighbouring stages below which labels thin out: under the first only
 * the skeleton keeps one, under the second a label says the name alone. A label is about 70–90 px
 * wide, so from the first up there is room for a name under every other stage, and declutter
 * hides the rest; from the second, for a name and a shape under most stages.
 */
const DETAIL_PIXELS: [number, number] = [30, 66];

function useDetailTier(pitch: number): Detail {
  const camera = useThree(state => state.camera);
  const [tier, setTier] = useState<Detail>('near');
  // 分档必须量化：useFrame 每帧都跑，而缩放值在拖拽时每帧都变，
  // 直接 setState 会让整个场景每帧重渲染。
  useFrame(() => {
    // An orthographic camera's zoom is its pixels per world unit.
    const spacing = (camera as THREE.OrthographicCamera).zoom * pitch;
    const next: Detail = spacing < DETAIL_PIXELS[0] ? 'far' : spacing < DETAIL_PIXELS[1] ? 'mid' : 'near';
    setTier(current => (current === next ? current : next));
  });
  return tier;
}

/** 远景只标注骨架：结构块、输入输出、折叠层和当前选中/悬停项，其余算子的标签不建立 DOM。 */
function labelVisible(node: Layer, tier: Detail, selectedId: string | null, hovered: string | null): boolean {
  if (tier !== 'far') return true;
  if (node.op === 'Group' || node.op === 'Input' || node.op === 'Output') return true;
  if (node.id === selectedId || node.id === hovered) return true;
  return repeatOf(node) > 1;
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
  // Every layer's position, size and ports, the routes between them and the bounds, worked out once.
  const model = useMemo(() => sceneLayout(props.graph, props.analysis, props.expanded), [props.graph, props.analysis, props.expanded]);
  const detail = useDetailTier(model.pitch);
  // Keep label roots in one DOM container when the canvas event target changes.
  const labelPortal = useMemo(() => ({ current: gl.domElement.parentElement! }), [gl]);
  const controls = useRef<any>(null);
  const flowDirection = useRef(props.direction); flowDirection.current = props.direction;
  // Read when the overview is framed, not a reason to reframe: opening the legend keeps the user's view.
  const insetRight = useRef(props.insetRight ?? 0); insetRight.current = props.insetRight ?? 0;
  const theme: SceneTheme = props.theme ?? 'day', look = LOOKS[theme];
  // Bloom renders the scene several times over; under software rendering it would cost more than it shows.
  const [software, setSoftware] = useState(false);
  useEffect(() => {
    NIGHT.value = theme === 'night' ? 1 : 0; FLOW_GLOW.value = look.flowGlow; scene.environmentIntensity = look.environment;
  }, [theme, look, scene]);
  const { routes, bounds } = model;
  const attentionView = props.graph.nodes.some(n => isAttention(n.op));
  const [hover, setHover] = useState<string | null>(null);
  // A convolution or pooling layer fed by a feature map gets its one kernel window.
  const kernels = useMemo(() => props.showKernels === false ? [] : props.graph.nodes.flatMap<KernelProps>(node => {
    if (!WINDOW_OPS.has(node.op)) return [];
    const edge = props.graph.edges.find(e => e.target === node.id && !e.targetPort), source = edge && model.nodes[edge.source], target = model.nodes[node.id];
    const input = edge && props.analysis.layers[edge.source]?.output, output = props.analysis.layers[node.id]?.output;
    if (!edge || !source || !target || !input || !output || input.length !== 4 || output.length !== 4) return [];
    return [{ node, input, output, from: source.position, to: target.position, fromDims: source.dims, toDims: target.dims }];
  }), [props.graph, props.analysis, model, props.showKernels]);
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
  const boxOf = (entry: SceneNode) => new THREE.Box3().setFromCenterAndSize(entry.position, new THREE.Vector3(...entry.dims));
  const focus = (id: string, head?: number) => {
    const entry = model.nodes[id]; if (!entry) return;
    const { node, position: target, attention } = entry;
    const headPoint = head !== undefined ? attention?.heads[head] : undefined;
    const focusBox = boxOf(entry);
    const merge = isMerge(node.op);
    // A merge is framed with the routes into it and every layer they span.
    if (merge) routes.filter(route => route.target === id).forEach(route => {
      flowCurve(route).getPoints(40).forEach(p => focusBox.expandByPoint(p));
      Object.values(model.nodes).filter(other => other.position.x >= route.start.x && other.position.x <= route.end.x).forEach(other => focusBox.union(boxOf(other)));
    });
    const center = headPoint ? target.clone().add(new THREE.Vector3(...headPoint)) : target.clone();
    const radius = attention ? qkvRadius(attention.qkvCount) : 0;
    const dims = headPoint ? new THREE.Vector3(2.1, Math.max(5, radius * 2 + 2.8), Math.max(4.4, radius * 2 + 1.3)) : new THREE.Vector3(...entry.dims);
    centerView(center, dims, attention ? (headPoint ? new THREE.Vector3(24, 5, 18) : new THREE.Vector3(18, 7, 24)) : new THREE.Vector3(4, 8, 12), merge ? focusBox : undefined);
  };
  useEffect(() => {
    const context = gl.getContext(), ext = context.getExtension('WEBGL_debug_renderer_info');
    const name = ext ? context.getParameter(ext.UNMASKED_RENDERER_WEBGL) : 'WebGL 2'; props.onGpu(String(name));
    setSoftware(/swiftshader|software|llvmpipe|basic render/i.test(String(name)));
  }, [gl, props.onGpu]);
  useEffect(() => {
    const reset = () => {
      const center = bounds.getCenter(new THREE.Vector3());
      camera.position.copy(center.clone().add(new THREE.Vector3(...(attentionView ? OVERVIEW_OFFSETS.attention : OVERVIEW_OFFSETS.blocks)))); camera.lookAt(center);
      const target = center.clone();
      fitCamera(camera as THREE.OrthographicCamera, bounds, size.width, size.height, target, insetRight.current);
      if (controls.current) { controls.current.target.copy(target); controls.current.update(); }
    };
    reset();
    const exportSource = { scene, camera: camera as THREE.OrthographicCamera, graph: props.graph, analysis: props.analysis, expanded: props.expanded, viewport: size, direction: () => flowDirection.current, dimensions: model.dimensions };
    props.onReady({
      reset,
      focus: id => focus(id),
      focusHead: (id, index) => focus(id, index),
      // The render happens before captureSceneImage first awaits, so the paint is back right after.
      capture: (width, transparent, options) => forExport(scene, paintedBlocks, () => captureSceneImage(gl, exportSource, width, transparent, options)),
      svg: (options, exportWidth = 1920) => exportSceneSvg(exportSource, exportWidth, options)
    });
  }, [camera, gl, scene, model, size, attentionView, props.expanded, props.graph, props.analysis, props.onReady]);
  useEffect(() => { if (props.selected && props.selectedHead !== null && model.nodes[props.selected]?.attention) focus(props.selected, props.selectedHead); }, [camera, props.selected, props.selectedHead, props.expanded, model, size]);
  return <SceneThemeContext.Provider value={theme}>
    {/* By day the canvas is transparent over the workspace's soft gradient; at night the backdrop is the scene's background. */}
    {/* The studio environment fills from every side; a key light from the upper right and a weak
        rim from behind keep a block's top, front and side distinct tones of its one colour. */}
    <StudioEnvironment />
    {theme === 'night' && <><NightBackdrop /><StageGlow bounds={bounds} />{!software && <Bloom />}</>}
    <hemisphereLight args={[look.sky, look.ground, look.hemisphere]} userData={{ dayLight: { intensity: LOOKS.day.hemisphere, color: LOOKS.day.sky, groundColor: LOOKS.day.ground } }} />
    <directionalLight position={[9, 16, 11]} intensity={look.key} userData={{ dayLight: { intensity: LOOKS.day.key, color: '#ffffff' } }} />
    <directionalLight position={[-10, 5, -8]} color={look.rimColor} intensity={look.rim} userData={{ dayLight: { intensity: LOOKS.day.rim, color: LOOKS.day.rimColor } }} />
    {/* A quiet floor: fine lines that fade out close to the model, not engineering paper to the horizon. */}
    <Grid name="export_grid" position={[0, -0.1, 0]} args={[80, 40]} cellSize={1} cellThickness={0.4} cellColor={look.cell} sectionSize={5} sectionThickness={0.7} sectionColor={look.section} fadeDistance={40} fadeStrength={2.2} infiniteGrid />
    <OrbitControls ref={controls} makeDefault enableDamping={false} minZoom={1.5} maxZoom={180} />
    {routes.map(route => {
      const flow = { start: route.start, end: route.end, control1: route.control1, control2: route.control2 };
      return <group key={route.id} name={`edge_${route.id}`}>{route.residual ? <ResidualFlow {...flow} active={props.playing} direction={props.direction} speed={props.speed} /> : <><EdgeLine {...flow} color={props.analysis.layers[route.target] ? look.edge : '#e26969'} exportColor={props.analysis.layers[route.target] ? EDGE_COLOR : '#e26969'} opacity={look.edgeOpacity} exportOpacity={LOOKS.day.edgeOpacity} />{detail !== 'far' && <ParticleEdge {...flow} active={props.playing} direction={props.direction} speed={props.speed} color={props.direction === 'forward' ? '#31adce' : '#ee706d'} intensity={Math.log2(props.analysis.layers[route.source]?.output.reduce((a, b) => a * b, 1) || 1)} compact={detail === 'mid'} />}</>}</group>;
    })}
    {kernels.map(kernel => <KernelWindow key={kernel.node.id} {...kernel} playing={props.playing} speed={props.speed} measured={measuredAt(kernel.node.id)} />)}
    {props.graph.nodes.map(n => {
      const entry = model.nodes[n.id], { position: pos, dims } = entry;
      const issue = props.diagnostics.find(d => d.nodeId === n.id && d.level === 'error'), selected = props.selected === n.id;
      const color = issue ? '#e45555' : COLORS[n.op];
      const label = layerLabel(entry, props.selected, props.selectedHead, props.expanded);
      return <group key={n.id} position={pos} name={isAttention(n.op) ? `attention_${n.id}` : n.op === 'Add' ? `add_${n.id}` : `node_${n.id}`} onClick={e => { e.stopPropagation(); props.onSelect(n.id); focus(n.id); }} onPointerOver={e => { e.stopPropagation(); setHover(n.id); document.body.style.cursor = 'pointer'; }} onPointerOut={() => { setHover(null); document.body.style.cursor = ''; }}>
        {isAttention(n.op) ? <AttentionModule node={n} selected={selected} selectedHead={selected ? props.selectedHead : null} selectedProjection={selected ? props.selectedProjection : null} onSelectProjection={(port, center) => { props.onSelectProjection(n.id, port); centerView(center, new THREE.Vector3(3.2, 3.8, 3.2), new THREE.Vector3(24, 5, 18)); }} overriddenPorts={props.graph.edges.filter(e => e.target === n.id && isProjectionPort(e.targetPort)).map(e => e.targetPort!)} attachedPorts={props.graph.edges.some(e => e.source === n.id && e.sourcePort || e.target === n.id && isProjectionPort(e.targetPort))} onSelectHead={index => { props.onSelectHead(n.id, index); focus(n.id, index); }} onFocusPart={center => { props.onSelect(n.id); centerView(center, new THREE.Vector3(2.1, 2.4, 2.1), new THREE.Vector3(18, 7, 24)); }} playing={props.playing} rotating={props.rotating} direction={props.direction} speed={props.speed} expanded={props.expanded} error={Boolean(issue)} />
          : <LayerBody node={n} dims={dims} color={color} glow={issue ? 0.4 : selected ? 0.22 : hover === n.id ? 0.12 : 0} outline={selected ? (theme === 'night' ? NIGHT_SELECTED_OUTLINE : SELECTED_OUTLINE) : edgeTone(color)} input={props.analysis.layers[n.id]?.input[0]} output={props.analysis.layers[n.id]?.output} paint={paints[n.id] ?? null} load={props.observation?.loads[n.id]} />}
        {/* An attention module is an open frame of small parts; one shadow under its whole footprint
            read as a smudge. A wrapped model's upper bands float above the lowest one, and a shadow
            cast that far down would land on the band below, so only the lowest band casts them. */}
        {!isAttention(n.op) && entry.grounded && <GroundShadow width={dims[0]} depth={dims[2]} lift={pos.y - GROUND_Y} color={color} />}
        {selected && !isAttention(n.op) && <mesh raycast={() => {}} position={[0, -dims[1] / 2 - 0.1, 0]} rotation={[-Math.PI / 2, 0, 0]} userData={{ dayMaterial: { color: SELECTED_OUTLINE } }}><ringGeometry args={[0.45, 0.49, 48]} /><meshBasicMaterial color={theme === 'night' ? NIGHT_SELECTED_OUTLINE : SELECTED_OUTLINE} transparent opacity={0.75} /></mesh>}
        {labelVisible(n, detail, props.selected, hover) && <Html portal={labelPortal} position={label.position} center zIndexRange={[10, 0]} style={{ pointerEvents: 'none' }}><div ref={element => { if (element) labelElements.current.set(n.id, element); else labelElements.current.delete(n.id); }} className={`scene-label ${selected ? 'selected' : ''}`} style={{ '--accent': color } as CSSProperties}><strong>{label.name}</strong>{detail === 'near' && <span>{shapeText(props.analysis.layers[n.id]?.output)}</span>}</div></Html>}
      </group>;
    })}
  </SceneThemeContext.Provider>;
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
