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

// This file sorts first, so the training below is the first one the CI server process ever
// runs. The first optimizer step lazily imports torch._dynamo and sympy, about 790 modules; the
// two epochs themselves take roughly 20 ms each, so that import is the whole cost. On a hosted
// Windows runner the whole test measured 49 s on two green runs and then ran out of its 30 s
// training wait on a third, with the status still at "准备数据" and nothing else wrong. That is
// a budget too tight for a cold start, not a regression, so give the cold start its own budget
// the same way the slow scene cases in editing.spec.ts have one.
const coldTrainingTimeout = 90_000;

test('observes actual cross-attention heads, offsets, KV groups and trained state on CPU/CUDA', async ({ page }, testInfo) => {
  test.setTimeout(180_000);
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
  const detail = attention.locator('.attention-matrix:not(.head-overview)'), overview = attention.locator('.head-overview');
  await expect(attention).toContainText('融合 SDPA 有浮点差异');
  await attention.getByText('矩阵数值表 · 当前窗口', { exact: true }).click();
  await expect(detail.locator('table')).toBeVisible();
  await detail.locator('g[tabindex]').first().focus();
  await expect(detail.locator('.tensor-hover')).toContainText('Query 0 · Key 0');
  // Every query head of the branch sits above the detail; the shown head's tile is its window.
  const heads = random.attentions[0].overview!;
  expect(heads.map(entry => [entry.head, entry.kvHead])).toEqual([[0, 0], [1, 0], [2, 1], [3, 1]]);
  expect(heads[0].values).toEqual(random.attentions[0].tensors.probabilities.slice.values);
  for (const entry of heads) {
    expect([entry.rows, entry.columns]).toEqual([16, 16]);
    expect(entry.entropy!).toBeGreaterThan(0); expect(entry.entropy!).toBeLessThanOrEqual(1);
    expect(entry.peak!).toBeGreaterThanOrEqual(1 / 24); expect(entry.peak!).toBeLessThanOrEqual(1);
  }
  const tiles = overview.getByRole('group', { name: '全部 Head', exact: true }).getByRole('button');
  await expect(tiles).toHaveCount(4);
  await expect(tiles.nth(0)).toHaveAttribute('aria-pressed', 'true');
  await tiles.nth(2).focus();
  await expect(overview.locator('.tensor-hover')).toContainText('H3：平均熵');
  await overview.screenshot({ path: testInfo.outputPath('attention-overview.png') });
  // Choosing a tile picks the head the detail shows, which needs a new run.
  await tiles.nth(2).click();
  await expect(panel.getByLabel('Attention Head')).toHaveValue('2');
  await expect(panel.locator('.tensor-stale')).toContainText('旧快照已失效');
  await expect(attention).toHaveCount(0);
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
  expect(selected.overview![3].values).toEqual(selected.tensors.probabilities.slice.values);
  await expect(tiles).toHaveCount(4);
  await expect(tiles.nth(3)).toHaveAttribute('aria-pressed', 'true');
  await attention.getByLabel('Attention 内部张量').selectOption('scores');
  await expect(attention).toContainText('缩放分数');
  await attention.getByLabel('Attention 内部张量').selectOption('headOutput');
  await expect(attention).toContainText('实际 SDPA');
  await attention.getByLabel('Attention 内部张量').selectOption('probabilities');
  await page.evaluate(() => document.documentElement.dataset.theme = 'dark');
  await detail.screenshot({ path: testInfo.outputPath('attention-dark.png') });
  await page.evaluate(() => document.documentElement.dataset.theme = 'light');
  await detail.screenshot({ path: testInfo.outputPath('attention-light.png') });
  await page.setViewportSize({ width: 390, height: 844 });
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  expect(await panel.evaluate(element => element.scrollWidth <= element.clientWidth + 1)).toBe(true);
  await detail.screenshot({ path: testInfo.outputPath('attention-mobile.png') });
  await overview.screenshot({ path: testInfo.outputPath('attention-overview-mobile.png') });
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.getByRole('button', { name: '训练', exact: true }).click();
  await page.getByLabel('训练轮次', { exact: true }).fill('2');
  await page.getByLabel('合成样本数', { exact: true }).fill('64');
  await page.locator('label.field-label').filter({ hasText: '计算设备' }).locator('select').selectOption('cpu');
  await page.getByRole('button', { name: '开始真实训练', exact: true }).click();
  await expect(page.locator('.train-status')).toContainText('训练完成', { timeout: coldTrainingTimeout });
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

test('turns the causal mask on in the properties panel and observes it in the executed head', async ({ page }) => {
  const errors: string[] = []; page.on('pageerror', error => errors.push(error.message));
  const causal = PRESETS.mqa(); causal.name = 'Causal fixture';
  await page.addInitScript(value => localStorage.setItem('tensorlab-project', JSON.stringify(value)), causal);
  await page.goto('/');
  await expect(page.locator('.statusbar')).toContainText('PyTorch');
  await page.getByLabel('causal', { exact: true }).selectOption('1');
  await expect(page.locator('.validation-badge')).toContainText('形状校验通过');
  await page.getByRole('button', { name: '张量观测', exact: true }).click();
  const panel = page.getByRole('region', { name: 'Tensor Inspector', exact: true });
  const response = page.waitForResponse('**/api/infer');
  await panel.getByRole('button', { name: '运行单样本推理', exact: true }).click();
  const item = (await waitForReport(await response)).attentions[0];
  expect(item.mask).toBe('causal');
  // 16 positions: every key after the query's own is exactly zero and each row still sums to one.
  const values = item.tensors.probabilities.slice.values;
  expect(values).toHaveLength(16);
  values.forEach((row, query) => {
    row.forEach((value, key) => { if (key > query) expect(value).toBe(0); else expect(value).toBeGreaterThan(0); });
    expect(row.reduce<number>((sum, value) => sum + value!, 0)).toBeCloseTo(1, 4);
  });
  expect(item.tensors.scores.nonFiniteCount).toBe(0);
  const attention = panel.getByRole('region', { name: 'Attention Inspector', exact: true });
  await expect(attention).toContainText('causal 掩码 · dropout=0');
  await expect(attention.locator('.matrix-mask-layer')).toHaveCount(16 * 15 / 2);
  // Projections are not masked, only what the softmax consumes and produces.
  await attention.getByLabel('Attention 内部张量').selectOption('q');
  await expect(attention.locator('.matrix-mask-layer')).toHaveCount(0);
  // The exported module carries the flag, so the standalone code computes the same thing.
  await page.getByRole('button', { name: '导出代码', exact: true }).click();
  await expect(page.locator('.code-preview, pre').first()).toContainText('causal=True');
  expect(errors).toEqual([]);
});

test('reads every head of a latent attention layer, masked like a decoder', async ({ page }) => {
  const errors: string[] = []; page.on('pageerror', error => errors.push(error.message));
  const layer = (id: string, op: string, params: Record<string, unknown>, x: number) => ({ id, name: id, op, params, position: { x, y: 100 } });
  // `layer_1` is the layer the workbench selects on load, so the inspector observes the MLA.
  const latent = {
    version: 1, name: 'Latent fixture',
    nodes: [layer('layer_0', 'Input', { shape: [1, 12, 32] }, 0), layer('layer_1', 'LatentAttention', { embed_dim: 32, num_heads: 4, q_lora_rank: 16, kv_lora_rank: 8, qk_nope_head_dim: 8, qk_rope_head_dim: 4, v_head_dim: 8, rope_theta: 10000, causal: 1, eps: 1e-6 }, 210), layer('layer_2', 'Flatten', {}, 420), layer('layer_3', 'Linear', { out_features: 3 }, 630), layer('layer_4', 'Output', {}, 840)],
    edges: [0, 1, 2, 3].map(i => ({ id: `e${i}`, source: `layer_${i}`, target: `layer_${i + 1}` }))
  };
  await page.addInitScript(value => localStorage.setItem('tensorlab-project', JSON.stringify(value)), latent);
  await page.goto('/');
  await expect(page.locator('.statusbar')).toContainText('PyTorch');
  await expect(page.locator('.validation-badge')).toContainText('形状校验通过');
  await expect(page.locator('.scene-legend')).toContainText('MLA');
  await page.getByRole('button', { name: '张量观测', exact: true }).click();
  const panel = page.getByRole('region', { name: 'Tensor Inspector', exact: true });
  // MLA is its own operator, but the inspector offers its heads like any attention layer's.
  await expect(panel.getByLabel('Attention Head').locator('option')).toHaveCount(4);
  const response = page.waitForResponse('**/api/infer');
  await panel.getByRole('button', { name: '运行单样本推理', exact: true }).click();
  const item = (await waitForReport(await response)).attentions[0];
  expect([item.numHeads, item.kvHeads, item.branches, item.headDim]).toEqual([4, 4, 1, 12]);
  expect(item.mask).toBe('causal');
  // Q/K heads are the position-free part plus the rotary part (8 + 4); V heads are 8 wide.
  expect(item.tensors.q.shape).toEqual([12, 12]); expect(item.tensors.v.shape).toEqual([12, 8]);
  item.tensors.probabilities.slice.values.forEach((row, query) => row.forEach((value, key) => { if (key > query) expect(value).toBe(0); }));
  const attention = panel.getByRole('region', { name: 'Attention Inspector', exact: true });
  await expect(attention.locator('.head-overview').getByRole('group', { name: '全部 Head', exact: true }).getByRole('button')).toHaveCount(4);
  await page.getByRole('button', { name: '导出代码', exact: true }).click();
  await expect(page.locator('.code-preview, pre').first()).toContainText('TensorLabLatentAttention(embed_dim=32');
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
