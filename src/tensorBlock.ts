import * as THREE from 'three';

/**
 * Cells along a block's three axes for the tensor it stands for, so its faces show the tensor
 * itself rather than a plain box. A feature map `[1,C,H,W]` gets C along x (the block's
 * thickness, the direction data flows in), H along y and W along z. A sequence `[1,S,E]` gets
 * its features along y and positions along z, and a vector `[1,F]` its features along y. Other
 * ranks, and blocks that are not drawn from their shape, have no grid.
 */
export function tensorCells(shape: number[] | undefined): [number, number, number] {
  if (!shape) return [1, 1, 1];
  if (shape.length === 4) return [gridCount(shape[1]), gridCount(shape[2]), gridCount(shape[3])];
  if (shape.length === 3) return [1, gridCount(shape[2]), gridCount(shape[1])];
  if (shape.length === 2) return [1, gridCount(shape[1]), 1];
  return [1, 1, 1];
}

/**
 * How many cells to draw for an axis of `n` elements: all of them up to 16. Beyond that, the
 * largest divisor of `n` that is at most 16, so every cell stands for the same whole number of
 * elements; an axis with no such divisor of at least 4 gets 16 approximate cells.
 */
export function gridCount(n: number) {
  if (!Number.isFinite(n) || n < 1) return 1;
  if (n <= 16) return Math.floor(n);
  for (let divisor = 16; divisor >= 4; divisor--) if (n % divisor === 0) return divisor;
  return 16;
}

/**
 * Cells per major line, like the heavier lines of graph paper: an axis of eight or more cells
 * gets one every 4 (or 3, or 2) cells, so the grid still reads from the overview, where single
 * cells are too small to draw.
 */
export function majorGroup(cells: number) {
  if (cells < 8) return 1;
  return [4, 3, 2].find(group => cells % group === 0) ?? 1;
}

export interface TensorUniforms {
  uCells: { value: THREE.Vector3 };
  uGroups: { value: THREE.Vector3 };
  uHalf: { value: THREE.Vector3 };
  uLine: { value: THREE.Color };
  uLineStrength: { value: number };
}

const VERTEX_HEAD = 'varying vec3 vTensorPosition;\nvarying vec3 vTensorNormal;';
const VERTEX_BODY = 'vTensorPosition = position;\nvTensorNormal = normal;';
const FRAGMENT_HEAD = 'uniform vec3 uCells;\nuniform vec3 uGroups;\nuniform vec3 uHalf;\nuniform vec3 uLine;\nuniform float uLineStrength;\nvarying vec3 vTensorPosition;\nvarying vec3 vTensorNormal;';
// A face shows the grid of the two axes it spans, never of the axis it faces. Cell lines are
// about one pixel wide at any zoom and fade out as cells shrink from five to three pixels, where
// a dense grid reads as a smear; major lines are a little heavier and last down to smaller
// groups, so the structure stays visible from the overview.
//
// All three axes are computed at once, without a loop or a branch: `fwidth` is undefined in
// control flow that may diverge, and on Direct3D (ANGLE, Windows) a per-axis `continue` made
// every line weight come out as zero.
const FRAGMENT_GRID = `{
  vec3 tensorCoord = vTensorPosition / max(2.0 * uHalf, vec3(1e-4)) + 0.5;
  vec3 tensorAxes = (1.0 - step(0.5, abs(vTensorNormal))) * step(1.5, uCells);
  vec3 tensorCell = tensorCoord * uCells;
  vec3 tensorRate = max(fwidth(tensorCell), vec3(1e-5));
  vec3 tensorMinor = (1.0 - clamp(abs(fract(tensorCell + 0.5) - 0.5) / tensorRate - 0.35, 0.0, 1.0)) * (1.0 - smoothstep(0.2, 0.34, tensorRate));
  vec3 tensorGroup = tensorCell / max(uGroups, vec3(1.0));
  vec3 tensorGroupRate = max(fwidth(tensorGroup), vec3(1e-5));
  vec3 tensorMajor = (1.0 - clamp(abs(fract(tensorGroup + 0.5) - 0.5) / tensorGroupRate - 0.6, 0.0, 1.0)) * (1.0 - smoothstep(0.16, 0.3, tensorGroupRate)) * step(1.5, uGroups);
  vec3 tensorLines = max(tensorMinor * 0.6, tensorMajor) * tensorAxes;
  float tensorLine = max(tensorLines.x, max(tensorLines.y, tensorLines.z));
  diffuseColor.rgb = mix(diffuseColor.rgb, uLine, tensorLine * uLineStrength);
  // The selection glow is added after lighting; dim it on the lines too, or a selected block loses its grid.
  totalEmissiveRadiance *= 1.0 - tensorLine * uLineStrength;
}`;

/**
 * A lit, matte block material whose faces carry a tensor's cell grid. It is an ordinary
 * MeshStandardMaterial underneath, so lighting, emissive selection and the image export, which
 * reads `color` and `opacity`, behave as for any other block; the grid only darkens the
 * diffuse colour along cell boundaries. Every block compiles to the same program.
 */
export function createTensorMaterial() {
  const uniforms: TensorUniforms = {
    uCells: { value: new THREE.Vector3(1, 1, 1) },
    uGroups: { value: new THREE.Vector3(1, 1, 1) },
    uHalf: { value: new THREE.Vector3(0.5, 0.5, 0.5) },
    uLine: { value: new THREE.Color() },
    uLineStrength: { value: 0.55 },
  };
  const material = new THREE.MeshStandardMaterial({ roughness: 0.62, metalness: 0.04 });
  material.onBeforeCompile = shader => {
    Object.assign(shader.uniforms, uniforms);
    shader.vertexShader = shader.vertexShader
      .replace('#include <common>', `#include <common>\n${VERTEX_HEAD}`)
      .replace('#include <begin_vertex>', `#include <begin_vertex>\n${VERTEX_BODY}`);
    shader.fragmentShader = shader.fragmentShader
      .replace('#include <common>', `#include <common>\n${FRAGMENT_HEAD}`)
      .replace('#include <color_fragment>', `#include <color_fragment>\n${FRAGMENT_GRID}`);
  };
  material.customProgramCacheKey = () => 'tensor-block-1';
  return { material, uniforms };
}
