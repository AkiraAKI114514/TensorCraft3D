import { useEffect, useMemo } from 'react';
import * as THREE from 'three';
import { blank, createTensorMaterial, floatTexture, majorGroup, type TensorUniforms } from './tensorBlock';
import type { BlockPaint } from './sceneObservation';
import { useSceneLook } from './sceneTheme';

/**
 * Tensor blocks and their outlines, shared by the scene's layers and by the parts of attention
 * modules, so every part of the model is drawn, lit, painted and exported the same way.
 */

const AXES = [new THREE.Vector3(1, 0, 0), new THREE.Vector3(0, 1, 0), new THREE.Vector3(0, 0, 1)];
// Every mounted block, so an image export can leave out measured values: an exported diagram
// shows the architecture, not one sample's activations.
export const paintedBlocks = new Set<TensorUniforms>();

// 盒子轮廓的边几何体必须缓存：写在 JSX 里的 new THREE.BoxGeometry(...dims) 会在每次渲染时
// 重新分配并重算 EdgesGeometry，节点一多就是每帧的分配风暴。
/** The outline of the selected layer, dark by day and luminous at night; any other outline is a faint tone of its own block. */
export const SELECTED_OUTLINE = '#1d3943', NIGHT_SELECTED_OUTLINE = '#9cecff';
export function NodeOutline({ dims, color }: { dims: [number, number, number]; color: string }) {
  const night = useSceneLook() === 'night';
  const geometry = useMemo(() => new THREE.EdgesGeometry(new THREE.BoxGeometry(dims[0], dims[1], dims[2])), [dims[0], dims[1], dims[2]]);
  useEffect(() => () => geometry.dispose(), [geometry]);
  // Dark, opaque edges on every block drew the model like a cartoon; shading separates the faces
  // by day, and at night the faces' own glowing borders do.
  const selected = color === SELECTED_OUTLINE || color === NIGHT_SELECTED_OUTLINE;
  return <lineSegments raycast={() => {}} geometry={geometry} userData={{ dayMaterial: { color: selected ? SELECTED_OUTLINE : color, opacity: selected ? 0.95 : 0.38 } }}><lineBasicMaterial color={color} transparent opacity={selected ? 0.95 : night ? 0.16 : 0.38} /></lineSegments>;
}

/** The block's own hue, darker: edges that read as the block's, not as a second colour. */
export const edgeTone = (color: string) => `#${new THREE.Color(color).multiplyScalar(0.62).getHexString()}`;

/**
 * A layer drawn as the tensor it outputs: a solid block whose faces carry that tensor's cell
 * grid, so channels, rows and columns are counted on the block instead of implied by its size.
 * The footprint is unchanged, which keeps the layout, routes and image export where they were.
 */
export function TensorBlock({ dims, color, glow, cells, paint, part, opacity = 1 }: { dims: [number, number, number]; color: string; glow: number; cells: [number, number, number]; paint: BlockPaint | null; part?: string; opacity?: number }) {
  const block = useMemo(createTensorMaterial, []);
  useEffect(() => { paintedBlocks.add(block.uniforms); return () => { paintedBlocks.delete(block.uniforms); block.material.dispose(); }; }, [block]);
  const textures = useMemo(() => ({ map: paint?.map ? floatTexture(paint.map.values, paint.map.columns, paint.map.rows) : null, strip: paint?.strip ? floatTexture(paint.strip.values, paint.strip.count, 1) : null }), [paint]);
  useEffect(() => () => { textures.map?.dispose(); textures.strip?.dispose(); }, [textures]);
  block.material.color.set(color); block.material.emissive.set(color); block.material.emissiveIntensity = glow;
  block.material.transparent = opacity < 1; block.material.opacity = opacity;
  const u = block.uniforms;
  u.uCells.value.set(...cells); u.uGroups.value.set(...cells.map(majorGroup) as [number, number, number]);
  u.uHalf.value.set(dims[0] / 2, dims[1] / 2, dims[2] / 2); u.uLine.value.set(edgeTone(color));
  u.uObserve.value = paint ? 1 : 0; u.uLimit.value = paint?.limit ?? 1;
  u.uMap.value = textures.map ?? blank(); u.uStrip.value = textures.strip ?? blank();
  if (paint?.map) { u.uMapSize.value.set(paint.map.columns, paint.map.rows); u.uMapTotal.value.set(paint.map.uTotal, paint.map.vTotal); u.uMapU.value.copy(AXES[paint.map.u]); u.uMapV.value.copy(AXES[paint.map.v]); }
  else u.uMapSize.value.set(0, 0);
  if (paint?.strip) { u.uStripAxis.value.copy(AXES[paint.strip.axis]); u.uStripRange.value.set(paint.strip.total, paint.strip.start, paint.strip.count); }
  else u.uStripRange.value.set(1, 0, 0);
  return <mesh userData={{ exportMesh: true, exportShade: true, exportCells: cells, ...(part ? { exportPart: part } : {}) }} material={block.material}><boxGeometry args={dims} /></mesh>;
}
