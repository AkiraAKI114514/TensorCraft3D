import { describe, expect, it } from 'vitest';
import * as THREE from 'three';
import { createTensorMaterial, gridCount, majorGroup, tensorCells } from './tensorBlock';

describe('tensor blocks', () => {
  it('counts every element up to 16 and whole groups of elements beyond', () => {
    expect([0, 1, 3, 10, 16].map(gridCount)).toEqual([1, 1, 3, 10, 16]);
    // 32 rows draw as 16 cells of 2, 28 as 14 of 2, 100 as 10 of 10, 784 as 16 of 49.
    expect([32, 28, 100, 784, 1024].map(gridCount)).toEqual([16, 14, 10, 16, 16]);
    // A prime has no whole grouping that fits, so its 16 cells are approximate.
    expect(gridCount(17)).toBe(16);
  });

  it('draws a heavier line every few cells once an axis has eight or more', () => {
    expect([1, 7, 13].map(majorGroup)).toEqual([1, 1, 1]);
    expect([16, 12, 9, 10, 14].map(majorGroup)).toEqual([4, 4, 3, 2, 2]);
  });

  it('maps each tensor axis to the block axis its size is drawn from', () => {
    // Feature map: channels along the thickness, rows up, columns across.
    expect(tensorCells([1, 16, 32, 8])).toEqual([16, 16, 8]);
    // Sequence: features up (the block height follows them), positions across.
    expect(tensorCells([1, 10, 64])).toEqual([1, 16, 10]);
    expect(tensorCells([1, 10])).toEqual([1, 10, 1]);
    expect(tensorCells(undefined)).toEqual([1, 1, 1]);
    expect(tensorCells([1, 2, 3, 4, 5])).toEqual([1, 1, 1]);
  });

  it('adds the grid to the standard material, so a renamed shader chunk cannot drop it silently', () => {
    const { material, uniforms } = createTensorMaterial();
    const shader = { uniforms: {} as Record<string, unknown>, vertexShader: THREE.ShaderLib.standard.vertexShader, fragmentShader: THREE.ShaderLib.standard.fragmentShader };
    material.onBeforeCompile(shader as unknown as THREE.WebGLProgramParametersWithUniforms, undefined as unknown as THREE.WebGLRenderer);
    expect(shader.vertexShader).toContain('vTensorPosition = position;');
    expect(shader.fragmentShader).toContain('uniform vec3 uCells;');
    expect(shader.fragmentShader).toContain('diffuseColor.rgb = mix(diffuseColor.rgb, uLine');
    expect(shader.uniforms.uCells).toBe(uniforms.uCells);
    expect(material.customProgramCacheKey()).toBe('tensor-block-1');
  });
});
