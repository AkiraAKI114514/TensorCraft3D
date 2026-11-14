import { useEffect, useMemo, useRef, useState } from 'react';
import { useFrame, type ThreeEvent } from '@react-three/fiber';
import * as THREE from 'three';
import { FlowArrow, FlowLine, ResidualFlow } from './Flow';
import { RESIDUAL_COLOR } from './flowGeometry';
import { attentionLayout, HEAD_FACING, type AttentionRoute, type Point3 } from './attentionLayout';
import type { Layer } from './types';
import { useSceneLook } from './sceneTheme';
import { edgeTone, NodeOutline, TensorBlock } from './TensorBlockMesh';
import { gridCount } from './tensorBlock';
export { isAttention, visualHeads } from './attentionLayout';

/** A part's colour, darker, for text in it: pale parts (LN, residual) gave labels that barely read. */
const inkOf = (color: string) => `#${new THREE.Color(color).multiplyScalar(0.68).getHexString()}`;
/** The same colour, lighter, for text and rims on the night stage. */
const glowOf = (color: string) => `#${new THREE.Color(color).lerp(new THREE.Color('#ffffff'), 0.45).getHexString()}`;

/**
 * Text in the scene, drawn on a sprite at twice the old resolution, semibold, in a darker tone of
 * its colour and on a soft white halo, so a label stays sharp and legible where it crosses an edge
 * or a head's panel. The image export reads `exportText`/`exportColor` and sets its own type.
 */
export function SceneText({ text, position, color = '#516e79', scale = 0.24 }: { text: string; position: THREE.Vector3; color?: string; scale?: number }) {
  const night = useSceneLook() === 'night';
  const texture = useMemo(() => {
    const canvas = document.createElement('canvas'); canvas.width = 1024; canvas.height = 160;
    const ctx = canvas.getContext('2d')!; ctx.font = '600 76px "Segoe UI", "Microsoft YaHei", sans-serif';
    ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.lineJoin = 'round';
    // By day, dark ink on a white halo; at night, light ink on a dark one.
    ctx.lineWidth = 16; ctx.strokeStyle = night ? 'rgba(3, 9, 15, 0.82)' : 'rgba(255, 255, 255, 0.9)'; ctx.strokeText(text, 512, 80, 1000);
    ctx.fillStyle = night ? glowOf(color) : inkOf(color); ctx.fillText(text, 512, 80, 1000);
    const map = new THREE.CanvasTexture(canvas); map.colorSpace = THREE.SRGBColorSpace; map.anisotropy = 4; return map;
  }, [text, color, night]);
  useEffect(() => () => texture.dispose(), [texture]);
  return <sprite raycast={() => {}} position={position} scale={[scale * 6.4, scale, 1]} userData={{ exportText: text, exportColor: color }}><spriteMaterial map={texture} transparent depthTest={false} /></sprite>;
}
type Dims = [number, number, number];
const cube = (size: number): Dims => [size, size, size];
/**
 * One part of a module, drawn as a small tensor block like the layer blocks and the parts of latent
 * attention and MoE: `dims` is its shape (thin along the flow for a projection tile, a norm plate or
 * a score matrix), `cells` the grid its faces carry. It shades by day, glows at night through the
 * block shader, and keeps the export name and position the image export and click-centring use. A
 * part only turns translucent while another head is focused.
 */
function Part({ point, color, label, labelOffset, part, dims, cells = [1, 1, 1], opacity = 1, glow = 0, onFocus }: { point: Point3; color: string; label?: string; labelOffset?: Point3; part?: string; dims: Dims; cells?: Dims; opacity?: number; glow?: number; onFocus?: (center: THREE.Vector3) => void }) {
  return <group position={point} onClick={onFocus ? e => { e.stopPropagation(); onFocus(e.eventObject.getWorldPosition(new THREE.Vector3())); } : undefined}>
    <TensorBlock dims={dims} color={color} glow={glow} cells={cells} paint={null} part={part || label || 'port'} opacity={opacity} />
    {opacity > 0.5 && <NodeOutline dims={dims} color={edgeTone(color)} />}
    {label && <SceneText text={label} position={new THREE.Vector3(...(labelOffset ?? [0, dims[1] / 2 + 0.22, 0]))} scale={0.38} color={color} />}
  </group>;
}
/** The sizes a module's parts are drawn from: the head width, the positions and the feed-forward width. */
interface Sizes { head: number; positions: number; embed: number; feedForward: number }
interface Props { node: Layer; selected: boolean; selectedHead: number | null; selectedProjection: string | null; onSelectProjection: (port: string, center: THREE.Vector3) => void; overriddenPorts: string[]; attachedPorts: boolean; onSelectHead: (index: number) => void; onFocusPart: (center: THREE.Vector3) => void; playing: boolean; rotating: boolean; direction: string; speed: number; expanded: boolean; error: boolean;
  /** Positions in the sequence the module attends over, which its score matrices are drawn with. */
  positions?: number; }
function Route({ route, playing, direction, speed, arrow = false, opacity = 1 }: { route: AttentionRoute; playing: boolean; direction: string; speed: number; arrow?: boolean; opacity?: number }) {
  const points = useMemo(() => ({ start: new THREE.Vector3(...route.start), end: new THREE.Vector3(...route.end), control1: new THREE.Vector3(...route.control1), control2: new THREE.Vector3(...route.control2) }), [route]);
  const color = direction === 'forward' ? route.color : '#ee706d';
  // The module's own residual loops arch wide around its heads; at full strength they outweighed
  // the heads themselves, so they are drawn at half strength in the module's own pink.
  if (route.color === RESIDUAL_COLOR) return <ResidualFlow {...points} active={playing} direction={direction} speed={speed} opacity={0.5} />;
  return <group><FlowLine {...points} color={route.color} pulseColor={color} opacity={opacity * 0.65} pulseStrength={opacity} active={playing} direction={direction} speed={speed} compact />{arrow && <FlowArrow {...points} color={color} direction={direction} />}</group>;
}
type Face = ReturnType<typeof attentionLayout>['faces'][number];
function FaceModule({ face, phase, sizes, ...props }: Props & { face: Face; phase: React.RefObject<number>; sizes: Sizes }) {
  const group = useRef<THREE.Group>(null), [hovered, setHovered] = useState(false);
  const focused = props.selected && (props.selectedHead !== null && face.computations.some(c => c.head === props.selectedHead) || face.vertices.some(v => v.id === props.selectedProjection));
  const dimmed = props.selected && (props.selectedHead !== null || props.selectedProjection !== null) && !focused;
  const geometry = useMemo(() => {
    const shape = new THREE.Shape(); shape.moveTo(face.vertices[0].point[0], face.vertices[0].point[1]); face.vertices.slice(1).forEach(v => shape.lineTo(v.point[0], v.point[1])); shape.closePath();
    return new THREE.ExtrudeGeometry(shape, { depth: 0.06, bevelEnabled: false, steps: 1 });
  }, [face]);
  useEffect(() => () => geometry.dispose(), [geometry]);
  useFrame(() => { if (group.current) group.current.rotation.y = props.attachedPorts || props.selected || hovered ? 0 : props.playing && props.rotating ? Math.sin(phase.current * 0.35) * 0.16 : group.current.rotation.y; });
  const color = props.error ? '#d96060' : focused ? '#299eae' : '#8ab6c7';
  const click = (e: ThreeEvent<MouseEvent>) => { e.stopPropagation(); props.onSelectHead(face.computations[0].head); };
  return <group position={face.position} onClick={click}>
    <SceneText text={`${Number(props.node.params.branches ?? 1) > 1 ? `B${face.branch + 1} · ` : ''}${face.label}`} position={new THREE.Vector3(0, face.points.inlet[1] + 0.55, 0)} scale={0.38} />
    <group rotation={HEAD_FACING}><group ref={group} onPointerOver={() => setHovered(true)} onPointerOut={() => setHovered(false)}>
      {/* A frosted panel: the head reads as one surface its projections sit on, with a light rim. */}
      <mesh geometry={geometry} userData={{ exportMesh: true, exportPart: 'qkv-face', exportSides: face.vertices.length }}><meshStandardMaterial color={color} transparent opacity={dimmed ? 0.06 : focused ? 0.28 : 0.18} roughness={0.35} metalness={0} side={THREE.DoubleSide} depthWrite={false} /></mesh>
      <lineSegments raycast={() => {}}><edgesGeometry args={[geometry]} /><lineBasicMaterial color={inkOf(color)} transparent opacity={dimmed ? 0.15 : focused ? 0.75 : 0.45} /></lineSegments>
      {/* Each Q, K or V is a projection block, its grid one head wide by the positions. It keeps depth:
          the polygon is often seen nearly edge-on, and a thin tile there would be too slim to click. */}
      {face.vertices.map(v => { const chosen = props.selectedProjection === v.id, side = chosen ? 0.36 : 0.27; return <Part key={v.id} point={v.point} color={v.color} part={v.id} label={v.label} labelOffset={[v.point[0] / Math.hypot(v.point[0], v.point[1]) * 0.44, v.point[1] / Math.hypot(v.point[0], v.point[1]) * 0.44, 0.12]} dims={[side, side, side * 0.75]} cells={[gridCount(sizes.head), gridCount(sizes.positions), 1]} glow={chosen ? 0.25 : 0} opacity={dimmed ? 0.22 : 1} onFocus={center => props.onSelectProjection(v.id, center)} />; })}
      <Part point={face.points.inlet} color="#728f9e" part="head-input" dims={cube(0.14)} onFocus={props.onFocusPart} />
      <Part point={face.points.outlet} color="#36a18a" part="head-output" dims={cube(0.14)} onFocus={props.onFocusPart} />
      {props.node.params.attention_type === 'cross' && <Part point={face.points.context} color="#a17cbb" part="context-input" dims={cube(0.14)} onFocus={props.onFocusPart} />}
      {/* Per head: its score matrix, positions by positions, and the weighted sum of its values; no larger
          than the cubes they replace, so they never cover a projection tile someone clicks. */}
      {face.computations.map(c => <group key={c.head}>
        <Part point={c.score} color="#539aa9" part={`scores-${c.head}`} dims={[0.21, 0.21, 0.035]} cells={[gridCount(sizes.positions), gridCount(sizes.positions), 1]} glow={focused && props.selectedHead === c.head ? 0.25 : 0} opacity={dimmed ? 0.15 : 1} onFocus={center => { props.onSelectHead(c.head); props.onFocusPart(center); }} />
        <Part point={c.weighted} color="#36a18a" part={`weighted-${c.head}`} dims={[0.12, 0.17, 0.06]} cells={[1, gridCount(sizes.head), 1]} opacity={dimmed ? 0.15 : 1} onFocus={props.onFocusPart} />
      </group>)}
      {focused && <SceneText text="QKᵀ → Softmax → ×V" position={new THREE.Vector3(0, -face.points.inlet[1] + 0.32, 0.3)} scale={0.26} />}
      {face.routes.filter(r => !r.projection || !props.overriddenPorts.includes(r.projection)).map(route => <Route key={route.id} route={route} playing={props.playing} direction={props.direction} speed={props.speed} opacity={dimmed ? 0.18 : 1} />)}
    </group></group>
  </group>;
}
/**
 * A frosted card standing behind the module, a little larger than its parts, so the heads, LN, FFN
 * and residual adds read as one layer rather than loose parts in the air. It is decoration: not
 * clickable, and named so the image export leaves it out.
 */
function ModuleCard({ width, height, depth, selected }: { width: number; height: number; depth: number; selected: boolean }) {
  const night = useSceneLook() === 'night';
  const { shape, outline } = useMemo(() => {
    const w = width + 0.5, h = height + 0.25, r = Math.min(0.4, h / 4), shape = new THREE.Shape();
    shape.moveTo(-w / 2 + r, -h / 2); shape.lineTo(w / 2 - r, -h / 2); shape.quadraticCurveTo(w / 2, -h / 2, w / 2, -h / 2 + r);
    shape.lineTo(w / 2, h / 2 - r); shape.quadraticCurveTo(w / 2, h / 2, w / 2 - r, h / 2); shape.lineTo(-w / 2 + r, h / 2);
    shape.quadraticCurveTo(-w / 2, h / 2, -w / 2, h / 2 - r); shape.lineTo(-w / 2, -h / 2 + r); shape.quadraticCurveTo(-w / 2, -h / 2, -w / 2 + r, -h / 2);
    return { shape: new THREE.ShapeGeometry(shape, 12), outline: new THREE.BufferGeometry().setFromPoints(shape.getPoints(12)) };
  }, [width, height]);
  useEffect(() => () => { shape.dispose(); outline.dispose(); }, [shape, outline]);
  return <group name="scene_overlay" position={[0, 0, -depth / 2 - 0.35]}>
    {/* By day a frosted white card; at night dark glass with a cyan rim, brighter when selected. */}
    <mesh raycast={() => {}} geometry={shape} renderOrder={-2}><meshBasicMaterial color={night ? '#0b2232' : '#ffffff'} transparent opacity={night ? (selected ? 0.62 : 0.48) : selected ? 0.62 : 0.46} depthWrite={false} /></mesh>
    <lineLoop raycast={() => {}} geometry={outline}><lineBasicMaterial color={night ? (selected ? '#9cecff' : '#2b86b3') : selected ? '#7fa3b1' : '#c3d2d8'} transparent opacity={0.9} /></lineLoop>
  </group>;
}

export default function AttentionModule(props: Props) {
  const layout = useMemo(() => attentionLayout(props.node, props.expanded), [props.node, props.expanded]), phase = useRef(0);
  useFrame((_, delta) => { if (props.playing && props.rotating && !props.selected) phase.current += delta * props.speed; });
  const ports = layout.ports;
  const embed = Number(props.node.params.embed_dim ?? 64), heads = Math.max(1, Number(props.node.params.num_heads ?? 1));
  const sizes: Sizes = { head: Math.max(1, Math.round(embed / heads)), positions: Math.max(1, props.positions ?? 8), embed, feedForward: Number(props.node.params.ff_dim ?? embed * 2) };
  // A tensor travelling through the module is a slab one position deep per cell row; FFN is wider in
  // proportion to its hidden width, and LN a thin plate, as RMSNorm is drawn in latent attention.
  const slab = (height: number): Dims => [0.12, height, 0.5], tensorCells: Dims = [1, gridCount(sizes.embed), gridCount(sizes.positions)];
  const norm = '#bc8cc8';
  return <>
    <ModuleCard width={layout.dimensions[0]} height={layout.dimensions[1]} depth={layout.dimensions[2]} selected={props.selected} />
    {props.node.op === 'Transformer' && <>
      <Part point={ports.norm1} color={norm} label="LN" part="LN1" dims={[0.05, 0.56, 0.46]} onFocus={props.onFocusPart} />
      <Part point={ports.add1} color={RESIDUAL_COLOR} label="+" part="Add1" dims={cube(0.24)} onFocus={props.onFocusPart} />
      <Part point={ports.norm2} color={norm} label="LN" part="LN2" dims={[0.05, 0.56, 0.46]} onFocus={props.onFocusPart} />
      <Part point={ports.ffn} color="#e48369" label="FFN" dims={[0.3, Math.min(1.1, 0.6 + 0.08 * Math.log2(Math.max(1, sizes.feedForward / sizes.embed) + 1)), 0.5]} cells={[1, gridCount(sizes.feedForward), gridCount(sizes.positions)]} onFocus={props.onFocusPart} />
      <Part point={ports.add2} color={RESIDUAL_COLOR} label="+" part="Add2" dims={cube(0.24)} onFocus={props.onFocusPart} />
    </>}
    <Part point={ports.input} color="#7d91a8" part="input" label={props.node.params.attention_type === 'cross' ? 'Query' : undefined} dims={slab(0.62)} cells={tensorCells} onFocus={props.onFocusPart} />
    {props.node.params.attention_type === 'cross' && <Part point={ports.context} color="#a17cbb" part="context-input" label="Context" dims={slab(0.5)} cells={tensorCells} onFocus={props.onFocusPart} />}
    {layout.branches > 1 && layout.branchMerges.map((point, i) => <Part key={i} point={point} color="#3c9b94" label={`B${i + 1} · Wᵒ`} part={`branch-output-${i}`} dims={[0.18, 0.5, 0.4]} cells={tensorCells} onFocus={props.onFocusPart} />)}
    <Part point={ports.merge} color="#3c9b94" label={layout.branches > 1 ? 'Branch Mean' : 'Concat · Wᵒ'} labelOffset={[0, -0.6, 0]} dims={[0.22, 0.66, 0.5]} cells={tensorCells} onFocus={props.onFocusPart} />
    {layout.routes.map(route => <Route key={route.id} route={route} playing={props.playing} direction={props.direction} speed={props.speed} arrow />)}
    {layout.faces.map(face => <FaceModule key={`${face.branch}-${face.group}`} {...props} face={face} phase={phase} sizes={sizes} />)}
  </>;
}
