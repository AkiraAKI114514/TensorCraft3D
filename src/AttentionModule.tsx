import { useEffect, useMemo, useRef, useState } from 'react';
import { useFrame, type ThreeEvent } from '@react-three/fiber';
import * as THREE from 'three';
import { FlowArrow, FlowLine, ResidualFlow } from './Flow';
import { RESIDUAL_COLOR } from './flowGeometry';
import { attentionLayout, HEAD_FACING, type AttentionRoute, type Point3 } from './attentionLayout';
import type { Layer } from './types';
import { useSceneLook } from './sceneTheme';
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
const cubeEdges = new Map<number, THREE.EdgesGeometry>();
/** One shared outline per cube size; the parts of every module reuse them. */
function edgesOf(size: number) {
  let edges = cubeEdges.get(size);
  if (!edges) { edges = new THREE.EdgesGeometry(new THREE.BoxGeometry(size, size, size)); cubeEdges.set(size, edges); }
  return edges;
}
/**
 * One part of a module (a projection, a score, a port, LN, FFN, a residual add): a solid, matte
 * cube with a faint outline in its own colour, like the layer blocks, rather than a glassy speck.
 * A part only turns translucent while another head is focused.
 */
function Cube({ point, color, label, labelOffset, part, size = 0.3, opacity = 1, onFocus }: { point: Point3; color: string; label?: string; labelOffset?: Point3; part?: string; size?: number; opacity?: number; onFocus?: (center: THREE.Vector3) => void }) {
  const night = useSceneLook() === 'night';
  // At night a part glows faintly in its own colour, enough for the bloom pass to catch; the image
  // export clears the glow (`nightGlow`) and draws the day look.
  return <group position={point} onClick={onFocus ? e => { e.stopPropagation(); onFocus(e.eventObject.getWorldPosition(new THREE.Vector3())); } : undefined}>
    <mesh userData={{ exportMesh: true, exportPart: part || label || 'port', nightGlow: true }}><boxGeometry args={[size, size, size]} /><meshStandardMaterial color={color} emissive={color} emissiveIntensity={night ? 0.55 * opacity : 0} transparent={opacity < 1} opacity={opacity} roughness={0.5} metalness={0} /></mesh>
    <lineSegments raycast={() => {}} geometry={edgesOf(size)} userData={{ dayMaterial: { color: inkOf(color), opacity: 0.4 * opacity } }}><lineBasicMaterial color={night ? glowOf(color) : inkOf(color)} transparent opacity={(night ? 0.7 : 0.4) * opacity} /></lineSegments>
    {label && <SceneText text={label} position={new THREE.Vector3(...(labelOffset ?? [0, size * 0.8 + 0.2, 0]))} scale={0.38} color={color} />}
  </group>;
}
interface Props { node: Layer; selected: boolean; selectedHead: number | null; selectedProjection: string | null; onSelectProjection: (port: string, center: THREE.Vector3) => void; overriddenPorts: string[]; attachedPorts: boolean; onSelectHead: (index: number) => void; onFocusPart: (center: THREE.Vector3) => void; playing: boolean; rotating: boolean; direction: string; speed: number; expanded: boolean; error: boolean; }
function Route({ route, playing, direction, speed, arrow = false, opacity = 1 }: { route: AttentionRoute; playing: boolean; direction: string; speed: number; arrow?: boolean; opacity?: number }) {
  const points = useMemo(() => ({ start: new THREE.Vector3(...route.start), end: new THREE.Vector3(...route.end), control1: new THREE.Vector3(...route.control1), control2: new THREE.Vector3(...route.control2) }), [route]);
  const color = direction === 'forward' ? route.color : '#ee706d';
  // The module's own residual loops arch wide around its heads; at full strength they outweighed
  // the heads themselves, so they are drawn at half strength in the module's own pink.
  if (route.color === RESIDUAL_COLOR) return <ResidualFlow {...points} active={playing} direction={direction} speed={speed} opacity={0.5} />;
  return <group><FlowLine {...points} color={route.color} pulseColor={color} opacity={opacity * 0.65} pulseStrength={opacity} active={playing} direction={direction} speed={speed} compact />{arrow && <FlowArrow {...points} color={color} direction={direction} />}</group>;
}
type Face = ReturnType<typeof attentionLayout>['faces'][number];
function FaceModule({ face, phase, ...props }: Props & { face: Face; phase: React.RefObject<number> }) {
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
      {face.vertices.map(v => <Cube key={v.id} point={v.point} color={v.color} part={v.id} label={v.label} labelOffset={[v.point[0] / Math.hypot(v.point[0], v.point[1]) * 0.42, v.point[1] / Math.hypot(v.point[0], v.point[1]) * 0.42, 0.12]} size={props.selectedProjection === v.id ? 0.38 : 0.28} opacity={dimmed ? 0.22 : 1} onFocus={center => props.onSelectProjection(v.id, center)} />)}
      <Cube point={face.points.inlet} color="#728f9e" part="head-input" size={0.15} onFocus={props.onFocusPart} />
      <Cube point={face.points.outlet} color="#36a18a" part="head-output" size={0.15} onFocus={props.onFocusPart} />
      {props.node.params.attention_type === 'cross' && <Cube point={face.points.context} color="#a17cbb" part="context-input" size={0.15} onFocus={props.onFocusPart} />}
      {face.computations.map(c => <group key={c.head}>
        <Cube point={c.score} color="#539aa9" part={`scores-${c.head}`} size={0.2} opacity={dimmed ? 0.15 : 1} onFocus={center => { props.onSelectHead(c.head); props.onFocusPart(center); }} />
        <Cube point={c.weighted} color="#36a18a" part={`weighted-${c.head}`} size={0.15} opacity={dimmed ? 0.15 : 1} onFocus={props.onFocusPart} />
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
  return <>
    <ModuleCard width={layout.dimensions[0]} height={layout.dimensions[1]} depth={layout.dimensions[2]} selected={props.selected} />
    {props.node.op === 'Transformer' && <><Cube point={ports.norm1} color="#bc8cc8" label="LN" part="LN1" size={0.36} onFocus={props.onFocusPart} /><Cube point={ports.add1} color={RESIDUAL_COLOR} label="+" part="Add1" size={0.26} onFocus={props.onFocusPart} /><Cube point={ports.norm2} color="#bc8cc8" label="LN" part="LN2" size={0.36} onFocus={props.onFocusPart} /><Cube point={ports.ffn} color="#e48369" label="FFN" size={0.6} onFocus={props.onFocusPart} /><Cube point={ports.add2} color={RESIDUAL_COLOR} label="+" part="Add2" size={0.26} onFocus={props.onFocusPart} /></>}
    <Cube point={ports.input} color="#7d91a8" part="input" label={props.node.params.attention_type === 'cross' ? 'Query' : undefined} size={0.22} onFocus={props.onFocusPart} />
    {props.node.params.attention_type === 'cross' && <Cube point={ports.context} color="#a17cbb" part="context-input" label="Context" size={0.22} onFocus={props.onFocusPart} />}
    {layout.branches > 1 && layout.branchMerges.map((point, i) => <Cube key={i} point={point} color="#3c9b94" label={`B${i + 1} · Wᵒ`} part={`branch-output-${i}`} size={0.28} onFocus={props.onFocusPart} />)}
    <Cube point={ports.merge} color="#3c9b94" label={layout.branches > 1 ? 'Branch Mean' : 'Concat · Wᵒ'} labelOffset={[0, -0.6, 0]} size={0.44} onFocus={props.onFocusPart} />
    {layout.routes.map(route => <Route key={route.id} route={route} playing={props.playing} direction={props.direction} speed={props.speed} arrow />)}
    {layout.faces.map(face => <FaceModule key={`${face.branch}-${face.group}`} {...props} face={face} phase={phase} />)}
  </>;
}
