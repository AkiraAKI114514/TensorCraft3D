import { useEffect, useMemo, useRef } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';
import { FLOW_GLOW, FLOW_PULSES, RESIDUAL_COLOR } from './flowGeometry';
import { flowCurve } from './sceneLayout';

export { flowCurve };
export { FLOW_GLOW };
export interface FlowProps { start: THREE.Vector3; end: THREE.Vector3; control1?: THREE.Vector3; control2?: THREE.Vector3; active: boolean; direction: string; speed: number; color: string; intensity?: number; compact?: boolean; opacity?: number; }

/** World units between two pulses, and how fast they travel at 1×, so every edge flows alike whatever its length. */
const SPACING = { main: 1.7, compact: 0.95 };
const SPEED = 1.15;

const VERTEX = `varying float vAlong;
void main() { vAlong = uv.x; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`;
// A pulse is a bright head with a tail fading behind it. For a point `distance` along the edge, `behind`
// is how far (as a share of the spacing) the last head has gone past it: 0 at the head, near 1 just
// before the next head arrives, which is why the head has a sharp front and a soft tail.
const FRAGMENT = `uniform vec3 uBase; uniform vec3 uPulse; uniform float uOpacity; uniform float uTime; uniform float uDirection;
uniform float uLength; uniform float uSpacing; uniform float uSpeed; uniform float uTail; uniform float uGlow; uniform float uPulses;
varying float vAlong;
void main() {
  float distance = (uDirection > 0.0 ? vAlong : 1.0 - vAlong) * uLength;
  float behind = fract((uTime * uSpeed - distance) / uSpacing);
  float pulse = exp(-behind * uTail) * uPulses;
  gl_FragColor = vec4(mix(uBase, uPulse * uGlow * 1.25, pulse), max(uOpacity, pulse));
  #include <tonemapping_fragment>
  #include <colorspace_fragment>
}`;

/**
 * An edge drawn as a tube that the data flows through: pulses of light, each a bright head with a
 * fading tail, travel along it at one speed and one spacing in world units, forward or back. A
 * larger tensor leaves a longer tail. At night the pulses pass the bloom threshold and glow; an
 * image export draws the tube in its day colour without pulses. `exportColor`/`exportOpacity` are
 * what the SVG export and the day look use when the live colour differs.
 */
export function FlowLine({ start, end, control1, control2, active, direction, speed, color, pulseColor = color, intensity = 0, compact = false, opacity = compact ? 0.75 : 0.72, exportColor = color, exportOpacity = opacity, pulses = true, pulseStrength = 1 }: FlowProps & { pulseColor?: string; exportColor?: string; exportOpacity?: number; pulses?: boolean; pulseStrength?: number }) {
  const clock = useRef(0);
  const { geometry, curve, material } = useMemo(() => {
    const curve = flowCurve({ start, end, control1, control2, compact });
    const geometry = new THREE.TubeGeometry(curve, 64, compact ? 0.016 : 0.03, 10, false);
    const material = new THREE.ShaderMaterial({
      uniforms: {
        uBase: { value: new THREE.Color() }, uPulse: { value: new THREE.Color() }, uOpacity: { value: opacity }, uTime: { value: 0 }, uDirection: { value: 1 },
        uLength: { value: curve.getLength() }, uSpacing: { value: compact ? SPACING.compact : SPACING.main }, uSpeed: { value: SPEED },
        uTail: { value: compact ? 7 : 7 - Math.min(3.5, Math.max(0, intensity) / 6) }, uGlow: FLOW_GLOW, uPulses: { value: 1 }
      },
      vertexShader: VERTEX, fragmentShader: FRAGMENT, transparent: true
    });
    return { geometry, curve, material };
  }, [start, end, control1, control2, compact, intensity]);
  useEffect(() => () => { geometry.dispose(); material.dispose(); }, [geometry, material]);
  const u = material.uniforms;
  u.uBase.value.set(color); u.uPulse.value.set(pulseColor); u.uOpacity.value = opacity;
  useFrame((_, delta) => {
    if (active) clock.current += delta * speed;
    u.uTime.value = clock.current; u.uDirection.value = direction === 'forward' ? 1 : -1;
    u.uPulses.value = pulses ? FLOW_PULSES.value * pulseStrength : 0;
  });
  return <mesh raycast={() => {}} geometry={geometry} material={material}
    userData={{ exportCurve: { start: start.toArray(), end: end.toArray(), control1: curve.v1.toArray(), control2: curve.v2.toArray(), color: exportColor, opacity: exportOpacity }, dayUniforms: { uBase: exportColor, uOpacity: exportOpacity, uPulses: 0 } }} />;
}

export function FlowArrow(props: Pick<FlowProps, 'start' | 'end' | 'control1' | 'control2' | 'color' | 'direction'>) {
  const { position, quaternion } = useMemo(() => {
    const curve = flowCurve({ ...props, compact: true }), t = props.direction === 'forward' ? 0.7 : 0.3;
    const tangent = curve.getTangent(t).multiplyScalar(props.direction === 'forward' ? 1 : -1).normalize();
    return { position: curve.getPoint(t), quaternion: new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 1, 0), tangent) };
  }, [props.start, props.end, props.control1, props.control2, props.direction]);
  return <mesh raycast={() => {}} position={position} quaternion={quaternion} userData={{ exportMesh: true }}><coneGeometry args={[0.055, 0.14, 6]} /><meshBasicMaterial color={props.color} /></mesh>;
}

/** A residual shortcut: its pink tube with pink pulses, and an arrow toward the addition. */
export function ResidualFlow(props: Omit<FlowProps, 'color' | 'compact' | 'intensity'> & { pulses?: boolean }) {
  return <><FlowLine {...props} color={RESIDUAL_COLOR} compact /><FlowArrow {...props} color={RESIDUAL_COLOR} /></>;
}
