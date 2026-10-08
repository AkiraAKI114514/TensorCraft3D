import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { test, expect } from '@playwright/test';
import type { Graph } from '../src/types';
import type { InferenceReport } from '../src/inferenceTypes';

type Example = { filename: string; model_name: string; input_shapes: Record<string, number[]>; expected_output: number[]; expected_parameters: number };
const { examples } = JSON.parse(readFileSync(resolve('examples/import_models/manifest.json'), 'utf8')) as { examples: Example[] };

for (const example of examples) test(`imports bundled ${example.filename}, exports it and observes a real sample`, async ({ page }) => {
  const errors: string[] = []; page.on('pageerror', error => errors.push(error.message));
  await page.goto('/');
  await expect(page.locator('.statusbar')).toContainText('PyTorch');
  await page.getByRole('button', { name: '导入 PyTorch 代码', exact: true }).click();
  const dialog = page.getByRole('dialog', { name: '导入 PyTorch 代码', exact: true });
  await dialog.locator('input[type="file"]').setInputFiles(resolve('examples/import_models', example.filename));
  await dialog.getByRole('button', { name: '解析代码', exact: true }).click();
  for (const [name, shape] of Object.entries(example.input_shapes)) {
    await expect(dialog.getByLabel(`导入输入形状 ${name}`)).toBeVisible();
    await dialog.getByLabel(`导入输入形状 ${name}`).fill(shape.join(', '));
  }
  await expect(dialog.getByLabel('导入模型对象')).toHaveValue(example.model_name);
  await dialog.getByRole('button', { name: '解析代码', exact: true }).click();
  await expect(dialog.locator('.import-preview')).toBeVisible();
  await expect(dialog.getByRole('alert')).toHaveCount(0);
  await dialog.getByRole('button', { name: '导入模型', exact: true }).click();
  await expect(page.locator('.validation-badge')).toContainText('形状校验通过');
  const graph = await page.evaluate(() => JSON.parse(localStorage.getItem('tensorlab-project')!) as Graph);
  const selected = graph.nodes.find(node => node.op === 'MultiHeadAttention' || node.op === 'Transformer') ?? graph.nodes.find(node => node.op === 'Output')!;
  await page.getByRole('button', { name: '拓扑图', exact: true }).click();
  await page.locator(`.react-flow__node[data-id="${selected.id}"]`).click();
  await page.getByRole('button', { name: '导出代码', exact: true }).click();
  await expect(page.locator('.code-preview')).toContainText('class VisualModel');
  const download = page.waitForEvent('download');
  await page.getByRole('button', { name: '下载 .py', exact: true }).click();
  expect((await download).suggestedFilename()).toMatch(/\.py$/);
  await page.getByRole('button', { name: '关闭对话框', exact: true }).click();
  await page.getByRole('button', { name: '张量观测', exact: true }).click();
  const response = page.waitForResponse('**/api/infer');
  await page.getByRole('button', { name: '运行单样本推理', exact: true }).click();
  const result = await response; expect(result.ok(), await result.text()).toBe(true);
  const report = await result.json() as InferenceReport;
  expect(report.tensors.find(tensor => tensor.nodeId === graph.nodes.find(node => node.op === 'Output')!.id)!.shape).toEqual([1, ...example.expected_output.slice(1)]);
  if (selected.op === 'MultiHeadAttention' || selected.op === 'Transformer') {
    expect(report.attentions).toHaveLength(1);
    await expect(page.getByRole('region', { name: 'Attention Inspector', exact: true })).toBeVisible();
    if (selected.op === 'Transformer') await expect(page.getByRole('region', { name: 'Attention Inspector', exact: true })).toContainText('最终输出还包含残差与 FFN');
  }
  expect(errors).toEqual([]);
});

for (const [op, shape] of [['BatchNorm1d', [2, 4]], ['BatchNorm3d', [2, 4, 2, 3, 3]]] as const) test(`imports and inspects ${op} with its correct rank`, async ({ page }) => {
  await page.goto('/');
  await page.getByRole('button', { name: '导入 PyTorch 代码', exact: true }).click();
  const source = `from torch import nn\nmodel = nn.Sequential(nn.${op}(4))\n`;
  await page.getByRole('textbox', { name: 'PyTorch 源代码', exact: true }).fill(source);
  await page.getByRole('button', { name: '解析代码', exact: true }).click();
  await page.getByLabel('导入输入形状 x').fill(shape.join(', '));
  await page.getByRole('button', { name: '解析代码', exact: true }).click();
  await expect(page.locator('.import-preview')).toContainText(op);
  await page.getByRole('button', { name: '导入模型', exact: true }).click();
  await expect(page.locator('.validation-badge')).toContainText('形状校验通过');
  const graph = await page.evaluate(() => JSON.parse(localStorage.getItem('tensorlab-project')!) as Graph);
  const normalization = graph.nodes.find(node => node.op === op)!;
  await page.getByRole('button', { name: '拓扑图', exact: true }).click();
  await page.locator(`.react-flow__node[data-id="${normalization.id}"]`).click();
  await page.getByRole('button', { name: '张量观测', exact: true }).click();
  const response = page.waitForResponse('**/api/infer');
  await page.getByRole('button', { name: '运行单样本推理', exact: true }).click();
  const result = await response; expect(result.ok(), await result.text()).toBe(true);
  const report = await result.json() as InferenceReport;
  expect(report.tensors.find(tensor => tensor.nodeId === normalization.id)!.shape).toEqual([1, ...shape.slice(1)]);
});
