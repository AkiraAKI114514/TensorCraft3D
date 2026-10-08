import { test, expect } from '@playwright/test';
import type { Graph } from '../src/types';
import type { InferenceReport } from '../src/inferenceTypes';

const graph: Graph = {
  version: 1, name: 'Inspector fixture',
  nodes: [
    { id: 'layer_0', name: 'Input image', op: 'Input', params: { shape: [4, 2, 20, 20] }, position: { x: 0, y: 100 } },
    { id: 'layer_1', name: 'Measured ReLU', op: 'ReLU', params: {}, position: { x: 210, y: 100 } },
    { id: 'layer_2', name: 'Output image', op: 'Output', params: {}, position: { x: 420, y: 100 } }
  ],
  edges: [{ id: 'e0', source: 'layer_0', target: 'layer_1' }, { id: 'e1', source: 'layer_1', target: 'layer_2' }]
};

test.beforeEach(async ({ page }) => {
  await page.addInitScript(value => localStorage.setItem('tensorlab-project', JSON.stringify(value)), graph);
});

test('inspects real single-sample tensors, slices and stale snapshots on desktop and mobile', async ({ page }, testInfo) => {
  const errors: string[] = [], calls: string[] = [];
  page.on('pageerror', error => errors.push(error.message));
  page.on('request', request => { if (request.url().endsWith('/api/infer')) calls.push(request.url()); });
  await page.goto('/');
  await expect(page.locator('.statusbar')).toContainText('PyTorch');
  await page.getByRole('button', { name: '张量观测', exact: true }).click();
  const panel = page.getByRole('region', { name: 'Tensor Inspector', exact: true });
  const run = panel.getByRole('button', { name: '运行单样本推理', exact: true });
  await expect(run).toBeEnabled();
  expect(calls).toHaveLength(0);
  const response = page.waitForResponse('**/api/infer');
  await run.click();
  const firstResponse = await response;
  expect(firstResponse.ok(), await firstResponse.text()).toBe(true);
  const report = await firstResponse.json() as InferenceReport;
  await expect(panel.locator('.tensor-provenance')).toContainText('实测快照 · CPU · eval');
  expect(report.tensors.find(tensor => tensor.nodeId === 'layer_1')!.shape).toEqual([1, 2, 20, 20]);
  expect(report.weights).toBe('random-initialized');
  await expect(panel).toContainText('已截断');
  expect(report.tensors.every(tensor => tensor.elements === 800 && tensor.finiteCount === 800)).toBe(true);
  const samples = Array.from({ length: 800 }, (_, i) => i % 5 - 2);
  await panel.getByLabel('推理输入来源').selectOption('provided');
  await panel.getByLabel('推理输入 JSON').fill(JSON.stringify({ layer_0: samples }));
  await expect(panel.locator('.tensor-stale')).toContainText('旧快照已失效');
  await expect(panel.locator('.tensor-slice')).toHaveCount(0);
  const provided = page.waitForResponse('**/api/infer');
  await run.click();
  const observed = await (await provided).json() as InferenceReport;
  const relu = observed.tensors.find(tensor => tensor.nodeId === 'layer_1')!;
  expect(relu.stats.min).toBe(0); expect(relu.stats.max).toBe(2); expect(relu.stats.mean).toBeCloseTo(0.6);
  expect(relu.histogram.counts.reduce((a, b) => a + b, 0)).toBe(800);
  expect(relu.slice.values[0].slice(0, 5)).toEqual([0, 0, 0, 1, 2]);
  await expect(panel).toContainText('自定义样本');
  const health = await (await page.request.get('/api/health')).json();
  if (health.cuda) {
    await panel.getByLabel('推理设备').selectOption('cuda');
    const gpuResponse = page.waitForResponse('**/api/infer');
    await run.click();
    const gpuReport = await (await gpuResponse).json() as InferenceReport;
    expect(gpuReport.device).toBe('CUDA');
    expect(gpuReport.tensors.find(tensor => tensor.nodeId === 'layer_1')!.slice.values).toEqual(relu.slice.values);
    await expect(panel.locator('.tensor-provenance')).toContainText('CUDA');
    await panel.getByLabel('推理设备').selectOption('cpu');
  }
  await panel.getByLabel('张量切片轴 1').fill('1');
  await expect(panel.locator('.tensor-stale')).toContainText('旧快照已失效');
  const sliced = page.waitForResponse('**/api/infer');
  await run.click();
  const slicedReport = await (await sliced).json() as InferenceReport;
  expect(slicedReport.tensors.find(tensor => tensor.nodeId === 'layer_1')!.slice.indices).toEqual([0, 1]);
  await panel.getByText('分布数据表', { exact: true }).click();
  await panel.locator('.tensor-histogram g').first().focus();
  await expect(panel.locator('.tensor-hover')).toContainText('个元素');
  await panel.locator('.tensor-histogram').scrollIntoViewIfNeeded();
  await page.locator('.inspector').screenshot({ path: testInfo.outputPath('tensor-inspector-desktop.png') });
  await panel.locator('.tensor-histogram').screenshot({ path: testInfo.outputPath('tensor-histogram-desktop.png') });
  await page.setViewportSize({ width: 390, height: 844 });
  await panel.scrollIntoViewIfNeeded();
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  expect(await panel.evaluate(element => element.scrollWidth <= element.clientWidth + 1)).toBe(true);
  await panel.locator('.tensor-histogram').screenshot({ path: testInfo.outputPath('tensor-histogram-mobile.png') });
  await page.screenshot({ path: testInfo.outputPath('tensor-inspector-mobile.png') });
  await page.getByRole('button', { name: '层属性', exact: true }).click();
  await page.getByRole('button', { name: '打开层库', exact: true }).click();
  await page.getByLabel('模型模板').selectOption('mlp');
  await page.getByRole('button', { name: '关闭层库', exact: true }).click();
  await page.getByRole('button', { name: '张量观测', exact: true }).click();
  await expect(panel.locator('.tensor-stale')).toContainText('旧快照已失效');
  await expect(run).toBeDisabled();
  expect(errors).toEqual([]);
});

test('keeps pending inference exclusive across tabs and discards responses for edited graphs', async ({ page }) => {
  await page.goto('/');
  await page.getByRole('button', { name: '张量观测', exact: true }).click();
  const panel = page.getByRole('region', { name: 'Tensor Inspector', exact: true });
  let release!: () => void;
  const held = new Promise<void>(resolve => { release = resolve; });
  await page.route('**/api/infer', async route => { const response = await route.fetch(); await held; await route.fulfill({ response }); });
  await panel.getByRole('button', { name: '运行单样本推理', exact: true }).click();
  await expect(panel.getByRole('button', { name: '单样本推理中' })).toBeDisabled();
  await page.getByRole('button', { name: '训练', exact: true }).click();
  await expect(page.getByRole('button', { name: '开始真实训练', exact: true })).toBeDisabled();
  await page.getByRole('button', { name: '关闭对话框', exact: true }).click();
  await page.getByRole('button', { name: '层属性', exact: true }).click();
  await page.getByLabel('模型模板').selectOption('mlp');
  release();
  await page.getByRole('button', { name: '张量观测', exact: true }).click();
  await expect(panel.locator('.tensor-stale')).toContainText('旧快照已失效');
  await expect(panel.locator('.tensor-provenance')).toHaveCount(0);
  await page.getByRole('button', { name: '训练', exact: true }).click();
  await expect(page.getByRole('button', { name: '开始真实训练', exact: true })).toBeEnabled();
});

test('reports inference errors and recovers while preserving Python and JSON export', async ({ page }) => {
  await page.goto('/');
  await page.getByRole('button', { name: '暂停数据流', exact: true }).click();
  await page.getByRole('button', { name: '拓扑图', exact: true }).click();
  await page.getByRole('button', { name: '导出代码', exact: true }).click();
  await expect(page.locator('.code-preview')).toContainText('class VisualModel');
  await page.getByRole('button', { name: 'Graph JSON', exact: true }).click();
  await expect(page.locator('.code-preview')).toContainText('Inspector fixture');
  const download = page.waitForEvent('download');
  await page.getByRole('button', { name: '下载 .json', exact: true }).click();
  expect((await download).suggestedFilename()).toMatch(/\.json$/);
  await page.getByRole('button', { name: '关闭对话框', exact: true }).click();
  await page.getByRole('button', { name: '张量观测', exact: true }).click();
  const panel = page.getByRole('region', { name: 'Tensor Inspector', exact: true });
  await page.route('**/api/infer', route => route.fulfill({ status: 409, json: { detail: '训练或诊断期间无法执行单样本推理' } }));
  await panel.getByRole('button', { name: '运行单样本推理', exact: true }).click();
  await expect(panel.getByRole('alert')).toContainText('无法执行单样本推理');
  await expect(panel.getByRole('button', { name: '运行单样本推理', exact: true })).toBeEnabled();
  await page.unroute('**/api/infer');
  await panel.getByRole('button', { name: '运行单样本推理', exact: true }).click();
  await expect(panel.locator('.tensor-provenance')).toBeVisible();
  await expect(panel.getByRole('alert')).toHaveCount(0);
});
