import { describe, expect, it } from 'vitest';
import * as THREE from 'three';
import { FLOW_GLOW, FLOW_PULSES } from './flowGeometry';
import { forExport, LOOKS } from './sceneLook';
import { createTensorMaterial, NIGHT } from './tensorBlock';

/** A scene lit and set up for the night look, the way Scene.tsx sets it up. */
function nightScene() {
  const scene = new THREE.Scene();
  const sky = new THREE.HemisphereLight(LOOKS.night.sky, LOOKS.night.ground, LOOKS.night.hemisphere);
  sky.userData.dayLight = { intensity: LOOKS.day.hemisphere, color: LOOKS.day.sky, groundColor: LOOKS.day.ground };
  const rim = new THREE.DirectionalLight(LOOKS.night.rimColor, LOOKS.night.rim);
  rim.userData.dayLight = { intensity: LOOKS.day.rim, color: LOOKS.day.rimColor };
  const part = new THREE.Mesh(new THREE.BoxGeometry(), new THREE.MeshStandardMaterial({ emissive: '#319cac', emissiveIntensity: 0.55 }));
  part.userData.nightGlow = true;
  // A flow tube: its night colour and opacity live in shader uniforms.
  const tube = new THREE.Mesh(new THREE.BoxGeometry(), new THREE.ShaderMaterial({ uniforms: { uBase: { value: new THREE.Color(LOOKS.night.edge) }, uOpacity: { value: LOOKS.night.edgeOpacity } } }));
  tube.userData.dayUniforms = { uBase: LOOKS.day.edge, uOpacity: LOOKS.day.edgeOpacity };
  scene.add(sky, rim, part, tube);
  scene.environmentIntensity = LOOKS.night.environment;
  NIGHT.value = 1; FLOW_GLOW.value = LOOKS.night.flowGlow;
  return { scene, sky, rim, part: part.material as THREE.MeshStandardMaterial, tube: tube.material as THREE.ShaderMaterial };
}

describe('scene looks', () => {
  it('exports in the day look and puts the night look back afterwards', () => {
    const { scene, sky, rim, part, tube } = nightScene(), painted = createTensorMaterial().uniforms;
    painted.uObserve.value = 1;
    const during = forExport(scene, [painted], () => ({
      night: NIGHT.value, glow: FLOW_GLOW.value, pulses: FLOW_PULSES.value, tube: [`#${tube.uniforms.uBase.value.getHexString()}`, tube.uniforms.uOpacity.value], environment: scene.environmentIntensity, observe: painted.uObserve.value,
      sky: [sky.intensity, `#${sky.color.getHexString()}`, `#${sky.groundColor.getHexString()}`], rim: [rim.intensity, `#${rim.color.getHexString()}`], part: part.emissiveIntensity
    }));
    expect(during).toEqual({ night: 0, glow: 1, pulses: 0, tube: [LOOKS.day.edge, LOOKS.day.edgeOpacity], environment: LOOKS.day.environment, observe: 0, sky: [LOOKS.day.hemisphere, LOOKS.day.sky, LOOKS.day.ground], rim: [LOOKS.day.rim, LOOKS.day.rimColor], part: 0 });
    expect([NIGHT.value, FLOW_GLOW.value, FLOW_PULSES.value, scene.environmentIntensity, painted.uObserve.value]).toEqual([1, LOOKS.night.flowGlow, 1, LOOKS.night.environment, 1]);
    expect([`#${tube.uniforms.uBase.value.getHexString()}`, tube.uniforms.uOpacity.value]).toEqual([LOOKS.night.edge, LOOKS.night.edgeOpacity]);
    expect([sky.intensity, `#${sky.color.getHexString()}`, `#${sky.groundColor.getHexString()}`]).toEqual([LOOKS.night.hemisphere, LOOKS.night.sky, LOOKS.night.ground]);
    expect([rim.intensity, `#${rim.color.getHexString()}`, part.emissiveIntensity]).toEqual([LOOKS.night.rim, LOOKS.night.rimColor, 0.55]);
    NIGHT.value = 0; FLOW_GLOW.value = 1;
  });

  it('puts the night look back even when the export fails', () => {
    const { scene, rim } = nightScene();
    expect(() => forExport(scene, [], () => { throw new Error('export failed'); })).toThrow('export failed');
    expect([NIGHT.value, rim.intensity]).toEqual([1, LOOKS.night.rim]);
    NIGHT.value = 0; FLOW_GLOW.value = 1;
  });
});
