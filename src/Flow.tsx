import { useEffect, useMemo, useRef } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';
import { RESIDUAL_COLOR } from './flowGeometry';

export interface FlowProps { start: THREE.Vector3; end: THREE.Vector3; control1?: THREE.Vector3; control2?: THREE.Vector3; active: boolean; direction: string; speed: number; color: string; intensity?: number; compact?: boolean; opacity?: number; }

export function flowCurve({ start, end, control1, control2, compact = false }: Pick<FlowProps, 'start' | 'end' | 'control1' | 'control2' | 'compact'>) {
  const d = end.x - start.x, bend = compact ? 0.05 : 0.4;
  return new THREE.CubicBezierCurve3(start, control1 ?? start.clone().add(new THREE.Vector3(d * 0.36, bend, 0)), control2 ?? end.clone().add(new THREE.Vector3(-d * 0.36, bend, 0)), end);
}

// Each instance follows its curve in the vertex shader, without CPU particle updates.
export function ParticleEdge({ start, end, control1, control2, active, direction, speed, color, intensity = 0, compact = false, opacity = 0.85 }: FlowProps) {
  const clock = useRef(0);
  const { geometry, material } = useMemo(() => {
    const base = new THREE.SphereGeometry(compact ? 0.024 : 0.046, 6, 4);
    const geometry = new THREE.InstancedBufferGeometry(); geometry.index = base.index;
    geometry.setAttribute('position', base.getAttribute('position').clone());
    const count = compact ? 8 : 28 + Math.min(40, Math.max(0, Math.floor(intensity)));
    geometry.setAttribute('aPhase', new THREE.InstancedBufferAttribute(Float32Array.from({ length: count }, (_, i) => i / count), 1));
    geometry.setAttribute('aOffset', new THREE.InstancedBufferAttribute(Float32Array.from({ length: count * 2 }, (_, i) => Math.sin(i * 9.13) * (compact ? 0.025 : 0.13)), 2));
    geometry.instanceCount = count; base.dispose();
    const curve = flowCurve({ start, end, control1, control2, compact });
    const material = new THREE.ShaderMaterial({
      uniforms: { uTime: { value: 0 }, uStart: { value: start }, uEnd: { value: end }, uControl1: { value: curve.v1 }, uControl2: { value: curve.v2 }, uDirection: { value: 1 }, uOpacity: { value: opacity }, uColor: { value: new THREE.Color(color) } },
      vertexShader: `attribute float aPhase; attribute vec2 aOffset; uniform float uTime; uniform vec3 uStart; uniform vec3 uEnd; uniform vec3 uControl1; uniform vec3 uControl2; uniform float uDirection; varying float vGlow;
      void main(){ float t=fract(aPhase+uTime*0.22); if(uDirection<0.0)t=1.0-t; float r=1.0-t; vec3 p=r*r*r*uStart+3.0*r*r*t*uControl1+3.0*r*t*t*uControl2+t*t*t*uEnd; p.yz+=aOffset; vGlow=sin(t*3.14159)*0.5+0.5; gl_Position=projectionMatrix*modelViewMatrix*vec4(p+position,1.0); }`,
      fragmentShader: `uniform vec3 uColor; uniform float uOpacity; varying float vGlow; void main(){gl_FragColor=vec4(uColor*(0.75+vGlow*0.45),uOpacity);}`,
      transparent: true, depthWrite: false
    });
    return { geometry, material };
  }, [start, end, control1, control2, intensity, color, compact, opacity]);
  useEffect(() => () => { geometry.dispose(); material.dispose(); }, [geometry, material]);
  useFrame((_, delta) => { if (active) clock.current += delta * speed; material.uniforms.uTime.value = clock.current; material.uniforms.uDirection.value = direction === 'forward' ? 1 : -1; });
  return <mesh raycast={() => {}} geometry={geometry} material={material} frustumCulled={false} />;
}

export function EdgeLine({ start, end, control1, control2, color, compact = false, opacity = compact ? 0.65 : 0.45 }: Pick<FlowProps, 'start' | 'end' | 'control1' | 'control2' | 'color' | 'compact' | 'opacity'>) {
  const geometry = useMemo(() => {
    const curve = flowCurve({ start, end, control1, control2, compact });
    return new THREE.TubeGeometry(curve, 20, compact ? 0.009 : 0.013, 4, false);
  }, [start, end, control1, control2, compact]);
  useEffect(() => () => geometry.dispose(), [geometry]);
  const curve = flowCurve({ start, end, control1, control2, compact });
  return <mesh raycast={() => {}} geometry={geometry} userData={{ exportCurve: { start: start.toArray(), end: end.toArray(), control1: curve.v1.toArray(), control2: curve.v2.toArray(), color, opacity } }}><meshBasicMaterial color={color} transparent opacity={opacity} /></mesh>;
}

export function FlowArrow(props: Pick<FlowProps, 'start' | 'end' | 'control1' | 'control2' | 'color' | 'direction'>) {
  const { position, quaternion } = useMemo(() => {
    const curve = flowCurve({ ...props, compact: true }), t = props.direction === 'forward' ? 0.7 : 0.3;
    const tangent = curve.getTangent(t).multiplyScalar(props.direction === 'forward' ? 1 : -1).normalize();
    return { position: curve.getPoint(t), quaternion: new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 1, 0), tangent) };
  }, [props.start, props.end, props.control1, props.control2, props.direction]);
  return <mesh raycast={() => {}} position={position} quaternion={quaternion} userData={{ exportMesh: true }}><coneGeometry args={[0.055, 0.14, 6]} /><meshBasicMaterial color={props.color} /></mesh>;
}

export function ResidualFlow(props: Omit<FlowProps, 'color' | 'compact' | 'intensity'>) {
  return <><EdgeLine {...props} color={RESIDUAL_COLOR} compact /><ParticleEdge {...props} color={RESIDUAL_COLOR} compact /><FlowArrow {...props} color={RESIDUAL_COLOR} /></>;
}
