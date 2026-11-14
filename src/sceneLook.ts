import * as THREE from 'three';
import { FLOW_GLOW } from './flowGeometry';
import { NIGHT, type TensorUniforms } from './tensorBlock';

/** Edges between layers by day: a calm slate that reads on the floor without competing with the blocks. */
export const EDGE_COLOR = '#5f7c89';

/**
 * Lights, environment, floor, edges and flow brightness for each look (src/sceneTheme.ts). By day
 * the scene is a light studio; at night a dark stage lit low from above and rimmed in cyan from
 * behind, so the blocks' glowing lines and the data flow carry the picture.
 */
export const LOOKS = {
  day: { sky: '#ffffff', ground: '#c3d0d6', hemisphere: 0.85, key: 1.55, rimColor: '#ffffff', rim: 0.4, environment: 0.38, cell: '#e7edf0', section: '#d6e0e4', edge: EDGE_COLOR, edgeOpacity: 0.72, flowGlow: 1 },
  night: { sky: '#a6c8ff', ground: '#04080d', hemisphere: 0.34, key: 0.85, rimColor: '#4cc3ff', rim: 1.6, environment: 0.16, cell: '#0f2938', section: '#1a4c63', edge: '#3aa7d4', edgeOpacity: 0.62, flowGlow: 2.6 }
} as const;

/** What a light is set to while an image is exported. */
export interface DayLight { intensity: number; color: string; groundColor?: string }

/** A material's day colour and opacity, for objects that draw differently at night. */
export interface DayMaterial { color?: string; opacity?: number }

/**
 * An image export shows the architecture in the day look, whatever the live view shows. For the
 * export's synchronous render, these are set to day, and afterwards put back as they were:
 * - measured values, the night switch, the flow glow and the environment's strength;
 * - every light marked `userData.dayLight`, and every part's night glow (`userData.nightGlow`);
 * - every material marked `userData.dayMaterial`;
 * - every day-only object (`userData.dayVisible`), shown.
 */
export function forExport<T>(scene: THREE.Scene, painted: Iterable<TensorUniforms>, work: () => T): T {
  const restore: (() => void)[] = [];
  const set = <O extends object, K extends keyof O>(object: O, key: K, value: O[K]) => { const previous = object[key]; restore.push(() => { object[key] = previous; }); object[key] = value; };
  for (const uniforms of painted) set(uniforms.uObserve, 'value', 0);
  set(NIGHT, 'value', 0); set(FLOW_GLOW, 'value', 1);
  set(scene, 'environmentIntensity', LOOKS.day.environment);
  scene.traverse(object => {
    const day = object.userData.dayLight as DayLight | undefined;
    if (day && object instanceof THREE.Light) {
      set(object, 'intensity', day.intensity); set(object, 'color', new THREE.Color(day.color));
      if (object instanceof THREE.HemisphereLight && day.groundColor) set(object, 'groundColor', new THREE.Color(day.groundColor));
    }
    if (object.userData.nightGlow && object instanceof THREE.Mesh && object.material instanceof THREE.MeshStandardMaterial) set(object.material, 'emissiveIntensity', 0);
    const material = (object as THREE.Mesh).material as (THREE.Material & { color?: THREE.Color }) | undefined, dayMaterial = object.userData.dayMaterial as DayMaterial | undefined;
    if (dayMaterial && material && !Array.isArray(material)) {
      if (dayMaterial.color && material.color) set(material, 'color', new THREE.Color(dayMaterial.color));
      if (dayMaterial.opacity !== undefined) set(material, 'opacity', dayMaterial.opacity);
    }
    if (object.userData.dayVisible) set(object, 'visible', true);
  });
  try { return work(); } finally { restore.reverse().forEach(undo => undo()); }
}
