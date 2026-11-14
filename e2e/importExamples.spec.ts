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

// DeepSeek-V3's own configuration appended to the vendored official file, which is only parsed.
const FULL_DEEPSEEK = 'args = ModelArgs(vocab_size=129280, dim=7168, inter_dim=18432, moe_inter_dim=2048, n_layers=61, n_dense_layers=3, n_heads=128, n_routed_experts=256, n_shared_experts=1, n_activated_experts=8, n_expert_groups=8, n_limited_groups=4, route_scale=2.5, score_func="sigmoid", q_lora_rank=1536, kv_lora_rank=512, qk_nope_head_dim=128, qk_rope_head_dim=64, v_head_dim=128, dtype="fp8")\nmodel = Transformer(args)\n';

test('imports the full 671B DeepSeek-V3 for its structure, and never trains it, runs it or exports it as Python', async ({ page }) => {
  const errors: string[] = []; page.on('pageerror', error => errors.push(error.message));
  await page.goto('/');
  await expect(page.locator('.statusbar')).toContainText('PyTorch');
  await page.getByRole('button', { name: '导入 PyTorch 代码', exact: true }).click();
  const dialog = page.getByRole('dialog', { name: '导入 PyTorch 代码', exact: true });
  const source = `${readFileSync(resolve('tests/fixtures/deepseek_v3/model.py'), 'utf8')}\n\n${FULL_DEEPSEEK}`;
  await dialog.locator('input[type="file"]').setInputFiles({ name: 'deepseek_v3_671b.py', mimeType: 'text/x-python', buffer: Buffer.from(source) });
  await dialog.getByRole('button', { name: '解析代码', exact: true }).click();
  await expect(dialog.locator('.import-preview')).toContainText('671.03B');
  await expect(dialog.getByRole('alert')).toHaveCount(0);
  await dialog.getByRole('button', { name: '导入模型', exact: true }).click();
  await expect(page.locator('.validation-badge')).toContainText('形状校验通过');
  await expect(page.locator('.structure-only-tag')).toHaveText('仅结构 · 超出本地运行上限');
  await expect(page.getByRole('button', { name: '训练', exact: true })).toBeDisabled();
  await page.getByRole('button', { name: '导出代码', exact: true }).click();
  await expect(page.locator('.code-preview')).toContainText('仅结构模型不能导出可运行的 Python');
  await expect(page.getByRole('button', { name: '下载 .py', exact: true })).toBeDisabled();
  await page.getByRole('button', { name: 'Graph JSON', exact: true }).click();
  await expect(page.getByRole('button', { name: '下载 .json', exact: true })).toBeEnabled();
  await page.getByRole('button', { name: '关闭对话框', exact: true }).click();
  // Inside the MoE block every layer has its shape: the level is analysed with what enters it.
  const graph = await page.evaluate(() => JSON.parse(localStorage.getItem('tensorlab-project')!) as Graph);
  expect(graph.structureOnly).toBe(true);
  const block = graph.nodes.find(node => node.subgraph === 'moe_layers')!;
  expect(block.repeat).toBe(58);
  await page.getByRole('button', { name: '拓扑图', exact: true }).click();
  await page.locator(`.react-flow__node[data-id="${block.id}"]`).click();
  await page.getByRole('button', { name: '进入结构块', exact: true }).click();
  await expect(page.locator('.validation-badge')).toContainText('形状校验通过');
  await expect(page.locator('.structure-only-tag')).toBeVisible();
  expect(errors).toEqual([]);
});
