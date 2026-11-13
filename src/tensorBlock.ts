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
  /** 1 while measured values are painted (see sceneObservation.ts), 0 for the plain block. */
  uObserve: { value: number };
  uMap: { value: THREE.Texture };
  /** The window's columns and rows, and how many elements the u and v axes hold in all. */
  uMapSize: { value: THREE.Vector2 };
  uMapTotal: { value: THREE.Vector2 };
  uMapU: { value: THREE.Vector3 };
  uMapV: { value: THREE.Vector3 };
  uStrip: { value: THREE.Texture };
  uStripAxis: { value: THREE.Vector3 };
  /** Channels in all, the first captured one, and how many were captured. */
  uStripRange: { value: THREE.Vector3 };
  uLimit: { value: number };
}

// The inspector's diverging scale (AttentionInspector.css): blue above zero, red below, a warm
// light grey at zero, and its own grey for a non-finite element.
const SCALE = { neutral: '#f0efec', positiveMid: '#3987e5', positiveDeep: '#0d366b', negativeMid: '#e34948', negativeDeep: '#651b1b', missing: '#8f8e89' };

const VERTEX_HEAD = 'varying vec3 vTensorPosition;\nvarying vec3 vTensorNormal;';
const VERTEX_BODY = 'vTensorPosition = position;\nvTensorNormal = normal;';
const FRAGMENT_HEAD = `uniform vec3 uCells;
uniform vec3 uGroups;
uniform vec3 uHalf;
uniform vec3 uLine;
uniform float uLineStrength;
uniform float uObserve;
uniform sampler2D uMap;
uniform vec2 uMapSize;
uniform vec2 uMapTotal;
uniform vec3 uMapU;
uniform vec3 uMapV;
uniform sampler2D uStrip;
uniform vec3 uStripAxis;
uniform vec3 uStripRange;
uniform float uLimit;
varying vec3 vTensorPosition;
varying vec3 vTensorNormal;
vec3 tensorHeat(float value) {
  if (value > 1e29) return vec3(${rgb(SCALE.missing)});
  float t = clamp(value / max(uLimit, 1e-12), -1.0, 1.0), a = abs(t);
  vec3 mid = t >= 0.0 ? vec3(${rgb(SCALE.positiveMid)}) : vec3(${rgb(SCALE.negativeMid)});
  vec3 deep = t >= 0.0 ? vec3(${rgb(SCALE.positiveDeep)}) : vec3(${rgb(SCALE.negativeDeep)});
  return a < 0.5 ? mix(vec3(${rgb(SCALE.neutral)}), mid, a * 2.0) : mix(mid, deep, a * 2.0 - 1.0);
}`;
// Measured values first, then the grid on top of them, so cells stay countable.
//
// Painting: element 0 of every axis sits at the top, front corner (see sceneObservation.ts). The
// window covers the faces spanned by its two axes, as far as it reaches; channel means cover the
// faces that span the channel axis. Faces without data are muted, so the measured ones stand out.
// Textures are read with `texelFetch` and the face is chosen by blending, not branching.
//
// Grid: a face shows the grid of the two axes it spans, never of the axis it faces. Cell lines
// are about one pixel wide at any zoom and fade out as cells shrink from five to three pixels,
// where a dense grid reads as a smear; major lines are a little heavier and last down to smaller
// groups, so the structure stays visible from the overview.
//
// All three axes are computed at once, without a loop or a branch: `fwidth` is undefined in
// control flow that may diverge, and on Direct3D (ANGLE, Windows) a per-axis `continue` made
// every line weight come out as zero.
const FRAGMENT_GRID = `{
  vec3 tensorCoord = vTensorPosition / max(2.0 * uHalf, vec3(1e-4)) + 0.5;
  vec3 tensorFace = step(0.5, abs(vTensorNormal));
  vec3 tensorFrom = 1.0 - tensorCoord;
  vec2 mapCell = floor(vec2(dot(tensorFrom, uMapU), dot(tensorFrom, uMapV)) * uMapTotal);
  float mapFace = step(0.5, dot(tensorFace, abs(cross(uMapU, uMapV))));
  float inMap = mapFace * step(mapCell.x + 0.5, uMapSize.x) * step(mapCell.y + 0.5, uMapSize.y);
  float stripCell = floor(dot(tensorFrom, uStripAxis) * uStripRange.x);
  float inStrip = (1.0 - mapFace) * (1.0 - step(0.5, dot(tensorFace, uStripAxis))) * step(0.5, uStripRange.z)
    * step(uStripRange.y, stripCell) * step(stripCell + 0.5, uStripRange.y + uStripRange.z);
  float mapValue = texelFetch(uMap, ivec2(clamp(mapCell, vec2(0.0), max(uMapSize - 1.0, vec2(0.0)))), 0).r;
  float stripValue = texelFetch(uStrip, ivec2(int(clamp(stripCell - uStripRange.y, 0.0, max(uStripRange.z - 1.0, 0.0))), 0), 0).r;
  vec3 observed = mix(mix(diffuseColor.rgb, vec3(${rgb(SCALE.neutral)}), 0.72), tensorHeat(stripValue), inStrip);
  observed = mix(observed, tensorHeat(mapValue), inMap);
  diffuseColor.rgb = mix(diffuseColor.rgb, observed, step(0.5, uObserve));
  vec3 tensorAxes = (1.0 - tensorFace) * step(1.5, uCells);
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

/** A colour as GLSL `r, g, b` in linear space, which is where the shader mixes. */
function rgb(hex: string) {
  const color = new THREE.Color(hex);
  return [color.r, color.g, color.b].map(value => value.toFixed(4)).join(', ');
}

let blankTexture: THREE.DataTexture | null = null;
/** One float texel, bound wherever a block has no measured window or strip. */
export function blank() {
  blankTexture ??= floatTexture(new Float32Array(1), 1, 1);
  return blankTexture;
}

/** A single-channel float texture, read cell by cell with `texelFetch`. */
export function floatTexture(values: Float32Array, width: number, height: number) {
  const texture = new THREE.DataTexture(values, width, height, THREE.RedFormat, THREE.FloatType);
  texture.minFilter = texture.magFilter = THREE.NearestFilter;
  texture.needsUpdate = true;
  return texture;
}

/**
 * A lit, matte block material whose faces carry a tensor's cell grid, and in observation mode
 * the layer's measured values. It is an ordinary MeshStandardMaterial underneath, so lighting,
 * emissive selection and the SVG export, which reads `color` and `opacity`, behave as for any
 * other block; the grid only darkens the diffuse colour along cell boundaries, and measured
 * values replace it. Every block compiles to the same program.
 */
export function createTensorMaterial() {
  const uniforms: TensorUniforms = {
    uCells: { value: new THREE.Vector3(1, 1, 1) },
    uGroups: { value: new THREE.Vector3(1, 1, 1) },
    uHalf: { value: new THREE.Vector3(0.5, 0.5, 0.5) },
    uLine: { value: new THREE.Color() },
    uLineStrength: { value: 0.55 },
    uObserve: { value: 0 },
    uMap: { value: blank() },
    uMapSize: { value: new THREE.Vector2(0, 0) },
    uMapTotal: { value: new THREE.Vector2(1, 1) },
    uMapU: { value: new THREE.Vector3(0, 0, 1) },
    uMapV: { value: new THREE.Vector3(0, 1, 0) },
    uStrip: { value: blank() },
    uStripAxis: { value: new THREE.Vector3(1, 0, 0) },
    uStripRange: { value: new THREE.Vector3(1, 0, 0) },
    uLimit: { value: 1 },
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
  material.customProgramCacheKey = () => 'tensor-block-2';
  return { material, uniforms };
}
