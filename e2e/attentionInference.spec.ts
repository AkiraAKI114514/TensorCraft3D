import { test, expect } from '@playwright/test';
import { PRESETS } from '../src/presets';
import type { AttentionSnapshot, InferenceReport } from '../src/inferenceTypes';

const graph = PRESETS.cross_attention();
graph.name = 'Attention fixture';
graph.nodes.find(node => node.op === 'Input')!.params.shape = [2, 20, 64];
graph.nodes.find(node => node.id === 'context')!.params.shape = [2, 24, 64];
graph.nodes.find(node => node.op === 'MultiHeadAttention')!.params = { attention_type: 'cross', embed_dim: 64, num_heads: 4, kv_heads: 2, branches: 2, dropout: 0.2 };

async function waitForReport(response: Awaited<ReturnType<import('@playwright/test').Page['waitForResponse']>>) {
  expect(response.ok(), await response.text()).toBe(true);
  return response.json() as Promise<InferenceReport>;
}

function validateAttention(value: AttentionSnapshot) {
  expect(value.scoreSource).toBe('projected-qk'); expect(value.outputSource).toBe('scaled_dot_product_attention');
  expect(value.tensors.q.shape).toEqual([20, 16]);
  expect(value.tensors.k.shape).toEqual([24, 16]);
  expect(value.tensors.probabilities.shape).toEqual([20, 24]);
  expect(value.tensors.probabilities.slice.truncated).toBe(true);
  expect(value.tensors.probabilities.nonFiniteCount).toBe(0);
  expect(value.tensors.probabilities.stats.min).toBeGreaterThanOrEqual(0);
  expect(value.tensors.probabilities.stats.max).toBeLessThanOrEqual(1);
}

test.beforeEach(async ({ page }) => {
  await page.addInitScript(value => localStorage.setItem('tensorlab-project', JSON.stringify(value)), graph);
});

test('observes actual cross-attention heads, offsets, KV groups and trained state on CPU/CUDA', async ({ page }, testInfo) => {
  const errors: string[] = []; page.on('pageerror', error => errors.push(error.message));
  let calls = 0; page.on('request', request => { if (request.url().endsWith('/api/infer')) calls++; });
  await page.goto('/');
  await expect(page.locator('.statusbar')).toContainText('PyTorch');
  await page.getByRole('button', { name: '张量观测', exact: true }).click();
  const panel = page.getByRole('region', { name: 'Tensor Inspector', exact: true });
  const run = panel.getByRole('button', { name: '运行单样本推理', exact: true });
  await expect(panel.getByLabel('Attention Head')).toBeEnabled();
  expect(calls).toBe(0);
  let response = page.waitForResponse('**/api/infer'); await run.click();
  const random = await waitForReport(await response);
  validateAttention(random.attentions[0]);
  expect(random.weights).toBe('random-initialized'); expect(random.attentions[0].kvHead).toBe(0);
  const attention = panel.getByRole('region', { name: 'Attention Inspector', exact: true });
  await expect(attention).toContainText('融合 SDPA 有浮点差异');
  await attention.getByText('矩阵数值表 · 当前窗口', { exact: true }).click();
  await expect(attention.locator('table')).toBeVisible();
  await attention.locator('g[tabindex]').first().focus();
  await expect(attention.locator('.tensor-hover')).toContainText('Query 0 · Key 0');
  await panel.getByLabel('Attention 分支').selectOption('1');
  await panel.getByLabel('Attention Head').selectOption('3');
  await panel.getByLabel('Attention Query 起点').fill('4');
  await panel.getByLabel('Attention Key 起点').fill('9');
  await expect(panel.locator('.tensor-stale')).toContainText('旧快照已失效');
  await expect(attention).toHaveCount(0);
  response = page.waitForResponse('**/api/infer'); await run.click();
  const windowed = await waitForReport(await response), selected = windowed.attentions[0];
  validateAttention(selected);
  expect(selected.branch).toBe(1); expect(selected.head).toBe(3); expect(selected.kvHead).toBe(1);
  expect(selected.tensors.probabilities.rowStart).toBe(4); expect(selected.tensors.probabilities.columnStart).toBe(9);
  expect(selected.tensors.probabilities.slice.rows).toBe(16); expect(selected.tensors.probabilities.slice.columns).toBe(15);
  await attention.getByLabel('Attention 内部张量').selectOption('scores');
  await expect(attention).toContainText('缩放分数');
  await attention.getByLabel('Attention 内部张量').selectOption('headOutput');
  await expect(attention).toContainText('实际 SDPA');
  await attention.getByLabel('Attention 内部张量').selectOption('probabilities');
  await page.evaluate(() => document.documentElement.dataset.theme = 'dark');
  await attention.locator('.attention-matrix').screenshot({ path: testInfo.outputPath('attention-dark.png') });
  await page.evaluate(() => document.documentElement.dataset.theme = 'light');
  await attention.locator('.attention-matrix').screenshot({ path: testInfo.outputPath('attention-light.png') });
  await page.setViewportSize({ width: 390, height: 844 });
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  expect(await panel.evaluate(element => element.scrollWidth <= element.clientWidth + 1)).toBe(true);
  await attention.locator('.attention-matrix').screenshot({ path: testInfo.outputPath('attention-mobile.png') });
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.getByRole('button', { name: '训练', exact: true }).click();
  await page.getByLabel('训练轮次', { exact: true }).fill('2');
  await page.getByLabel('合成样本数', { exact: true }).fill('64');
  await page.locator('label.field-label').filter({ hasText: '计算设备' }).locator('select').selectOption('cpu');
  await page.getByRole('button', { name: '开始真实训练', exact: true }).click();
  await expect(page.locator('.train-status')).toContainText('训练完成', { timeout: 30000 });
  await expect(panel.getByLabel('推理权重来源')).toHaveValue('trained');
  response = page.waitForResponse('**/api/infer'); await run.click();
  const trained = await waitForReport(await response);
  expect(trained.weights).toBe('trained'); expect(trained.model!.modelId).toBeTruthy();
  expect(trained.attentions[0].tensors.q.slice.values).not.toEqual(selected.tensors.q.slice.values);
  const health = await (await page.request.get('/api/health')).json();
  if (health.cuda) {
    await panel.getByLabel('推理设备').selectOption('cuda');
    response = page.waitForResponse('**/api/infer'); await run.click();
    const cuda = await waitForReport(await response);
    expect(cuda.device).toBe('CUDA'); expect(cuda.model).toEqual(trained.model);
    const cpuValues = trained.attentions[0].tensors.probabilities.slice.values, gpuValues = cuda.attentions[0].tensors.probabilities.slice.values;
    for (let i = 0; i < cpuValues.length; i++) for (let j = 0; j < cpuValues[i].length; j++) expect(gpuValues[i][j]).toBeCloseTo(cpuValues[i][j]!, 4);
  }
  expect(errors).toEqual([]);
});

test('keeps attention requests bounded and reports resource errors without a silent fallback', async ({ page }) => {
  await page.goto('/');
  await page.getByRole('button', { name: '张量观测', exact: true }).click();
  const panel = page.getByRole('region', { name: 'Tensor Inspector', exact: true });
  await panel.getByLabel('Attention Key 起点').fill('24');
  await expect(panel.getByRole('button', { name: '运行单样本推理', exact: true })).toBeDisabled();
  await panel.getByLabel('Attention Key 起点').fill('0');
  await page.route('**/api/infer', route => route.fulfill({ status: 422, json: { detail: 'Selected attention score matrix exceeds inspection limit' } }));
  await panel.getByRole('button', { name: '运行单样本推理', exact: true }).click();
  await expect(panel.getByRole('alert')).toContainText('inspection limit');
  await expect(panel.getByRole('region', { name: 'Attention Inspector', exact: true })).toHaveCount(0);
  await page.unroute('**/api/infer');
  const response = page.waitForResponse('**/api/infer');
  await panel.getByRole('button', { name: '运行单样本推理', exact: true }).click();
  validateAttention((await waitForReport(await response)).attentions[0]);
});
