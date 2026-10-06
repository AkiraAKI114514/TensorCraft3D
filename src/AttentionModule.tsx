import { useEffect, useMemo, useRef, useState } from 'react';
import { useFrame, type ThreeEvent } from '@react-three/fiber';
import * as THREE from 'three';
import { EdgeLine, FlowArrow, ParticleEdge, ResidualFlow } from './Flow';
import { RESIDUAL_COLOR } from './flowGeometry';
import { attentionLayout, HEAD_FACING, type AttentionRoute, type Point3 } from './attentionLayout';
import type { Layer } from './types';
export { isAttention, visualHeads } from './attentionLayout';

export function SceneText({ text, position, color = '#516e79', scale = 0.24 }: { text: string; position: THREE.Vector3; color?: string; scale?: number }) {
  const texture = useMemo(() => {
    const canvas = document.createElement('canvas'); canvas.width = 512; canvas.height = 80;
    const ctx = canvas.getContext('2d')!; ctx.font = '500 40px "Segoe UI", "Microsoft YaHei", sans-serif';
    ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.fillStyle = color; ctx.fillText(text, 256, 40, 505);
    const map = new THREE.CanvasTexture(canvas); map.colorSpace = THREE.SRGBColorSpace; return map;
  }, [text, color]);
  useEffect(() => () => texture.dispose(), [texture]);
  return <sprite raycast={() => {}} position={position} scale={[scale * 6.4, scale, 1]} userData={{ exportText: text, exportColor: color }}><spriteMaterial map={texture} transparent depthTest={false} /></sprite>;
}
function Cube({ point, color, label, labelOffset, part, size = 0.3, opacity = 0.85, onFocus }: { point: Point3; color: string; label?: string; labelOffset?: Point3; part?: string; size?: number; opacity?: number; onFocus?: (center: THREE.Vector3) => void }) {
  return <group position={point} onClick={onFocus ? e => { e.stopPropagation(); onFocus(e.eventObject.getWorldPosition(new THREE.Vector3())); } : undefined}><mesh userData={{ exportMesh: true, exportPart: part || label || 'port' }}><boxGeometry args={[size, size, size]} /><meshStandardMaterial color={color} transparent opacity={opacity} roughness={0.35} /></mesh>{label && <SceneText text={label} position={new THREE.Vector3(...(labelOffset ?? [0, size * 0.8 + 0.2, 0]))} scale={0.32} color={color} />}</group>;
}
interface Props { node: Layer; selected: boolean; selectedHead: number | null; selectedProjection: string | null; onSelectProjection: (port: string, center: THREE.Vector3) => void; overriddenPorts: string[]; attachedPorts: boolean; onSelectHead: (index: number) => void; onFocusPart: (center: THREE.Vector3) => void; playing: boolean; rotating: boolean; direction: string; speed: number; expanded: boolean; error: boolean; }
function Route({ route, playing, direction, speed, arrow = false, opacity = 1 }: { route: AttentionRoute; playing: boolean; direction: string; speed: number; arrow?: boolean; opacity?: number }) {
  const points = useMemo(() => ({ start: new THREE.Vector3(...route.start), end: new THREE.Vector3(...route.end), control1: new THREE.Vector3(...route.control1), control2: new THREE.Vector3(...route.control2) }), [route]);
  const color = direction === 'forward' ? route.color : '#ee706d';
  if (route.color === RESIDUAL_COLOR) return <ResidualFlow {...points} active={playing} direction={direction} speed={speed} />;
  return <group><EdgeLine {...points} color={route.color} opacity={opacity * 0.65} compact /><ParticleEdge {...points} color={color} opacity={opacity * 0.85} active={playing} direction={direction} speed={speed} compact />{arrow && <FlowArrow {...points} color={color} direction={direction} />}</group>;
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
  const color = props.error ? '#d96060' : focused ? '#299eae' : '#77a8bd';
  const click = (e: ThreeEvent<MouseEvent>) => { e.stopPropagation(); props.onSelectHead(face.computations[0].head); };
  return <group position={face.position} onClick={click}>
    <SceneText text={`${Number(props.node.params.branches ?? 1) > 1 ? `B${face.branch + 1} · ` : ''}${face.label}`} position={new THREE.Vector3(0, face.points.inlet[1] + 0.55, 0)} scale={0.38} />
    <group rotation={HEAD_FACING}><group ref={group} onPointerOver={() => setHovered(true)} onPointerOut={() => setHovered(false)}>
      <mesh geometry={geometry} userData={{ exportMesh: true, exportPart: 'qkv-face', exportSides: face.vertices.length }}><meshStandardMaterial color={color} transparent opacity={dimmed ? 0.06 : focused ? 0.24 : 0.14} roughness={0.45} side={THREE.DoubleSide} /></mesh>
      <lineSegments raycast={() => {}}><edgesGeometry args={[geometry]} /><lineBasicMaterial color={color} transparent opacity={dimmed ? 0.18 : 0.8} /></lineSegments>
      {face.vertices.map(v => <Cube key={v.id} point={v.point} color={v.color} part={v.id} label={v.label} labelOffset={[v.point[0] / Math.hypot(v.point[0], v.point[1]) * 0.4, v.point[1] / Math.hypot(v.point[0], v.point[1]) * 0.4, 0.12]} size={props.selectedProjection === v.id ? 0.38 : 0.26} opacity={dimmed ? 0.22 : 0.85} onFocus={center => props.onSelectProjection(v.id, center)} />)}
      <Cube point={face.points.inlet} color="#728f9e" part="head-input" size={0.12} onFocus={props.onFocusPart} />
      <Cube point={face.points.outlet} color="#36a18a" part="head-output" size={0.12} onFocus={props.onFocusPart} />
      {props.node.params.attention_type === 'cross' && <Cube point={face.points.context} color="#a17cbb" part="context-input" size={0.12} onFocus={props.onFocusPart} />}
      {face.computations.map(c => <group key={c.head}>
        <Cube point={c.score} color="#539aa9" part={`scores-${c.head}`} size={0.16} opacity={dimmed ? 0.15 : 0.7} onFocus={center => { props.onSelectHead(c.head); props.onFocusPart(center); }} />
        <Cube point={c.weighted} color="#36a18a" part={`weighted-${c.head}`} size={0.12} opacity={dimmed ? 0.15 : 0.8} onFocus={props.onFocusPart} />
      </group>)}
      {focused && <SceneText text="QKᵀ → Softmax → ×V" position={new THREE.Vector3(0, -face.points.inlet[1] + 0.32, 0.3)} scale={0.26} />}
      {face.routes.filter(r => !r.projection || !props.overriddenPorts.includes(r.projection)).map(route => <Route key={route.id} route={route} playing={props.playing} direction={props.direction} speed={props.speed} opacity={dimmed ? 0.18 : 1} />)}
    </group></group>
  </group>;
}
export default function AttentionModule(props: Props) {
  const layout = useMemo(() => attentionLayout(props.node, props.expanded), [props.node, props.expanded]), phase = useRef(0);
  useFrame((_, delta) => { if (props.playing && props.rotating && !props.selected) phase.current += delta * props.speed; });
  const ports = layout.ports;
  return <>
    {props.node.op === 'Transformer' && <><Cube point={ports.norm1} color="#bc8cc8" label="LN" part="LN1" onFocus={props.onFocusPart} /><Cube point={ports.add1} color={RESIDUAL_COLOR} label="+" part="Add1" size={0.22} onFocus={props.onFocusPart} /><Cube point={ports.norm2} color="#bc8cc8" label="LN" part="LN2" onFocus={props.onFocusPart} /><Cube point={ports.ffn} color="#e48369" label="FFN" size={0.55} onFocus={props.onFocusPart} /><Cube point={ports.add2} color={RESIDUAL_COLOR} label="+" part="Add2" size={0.22} onFocus={props.onFocusPart} /></>}
    <Cube point={ports.input} color="#7d91a8" part="input" label={props.node.params.attention_type === 'cross' ? 'Query' : undefined} size={0.18} onFocus={props.onFocusPart} />
    {props.node.params.attention_type === 'cross' && <Cube point={ports.context} color="#a17cbb" part="context-input" label="Context" size={0.18} onFocus={props.onFocusPart} />}
    {layout.branches > 1 && layout.branchMerges.map((point, i) => <Cube key={i} point={point} color="#3c9b94" label={`B${i + 1} · Wᵒ`} part={`branch-output-${i}`} size={0.25} onFocus={props.onFocusPart} />)}
    <Cube point={ports.merge} color="#3c9b94" label={layout.branches > 1 ? 'Branch Mean' : 'Concat · Wᵒ'} labelOffset={[0, -0.55, 0]} size={0.4} onFocus={props.onFocusPart} />
    {layout.routes.map(route => <Route key={route.id} route={route} playing={props.playing} direction={props.direction} speed={props.speed} arrow />)}
    {layout.faces.map(face => <FaceModule key={`${face.branch}-${face.group}`} {...props} face={face} phase={phase} />)}
  </>;
}
