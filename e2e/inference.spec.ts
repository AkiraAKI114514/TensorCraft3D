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
  // The same bounded window read as a feature map: a heatmap of the selected channel plus one
  // tile per channel. Picking a tile only moves the slice index; it never runs inference itself.
  const semantic = panel.locator('.tensor-semantic');
  await expect(semantic).toContainText('特征图 · 通道 0');
  // The colour legend is a bar between its two end labels; outside the attention inspector it
  // once collapsed into the two numbers run together.
  expect(await semantic.locator('.attention-scale i').first().evaluate(bar => bar.getBoundingClientRect().width)).toBeGreaterThan(40);
  const tiles = semantic.locator('.channel-grid-items button');
  await expect(tiles).toHaveCount(2);
  await expect(tiles.nth(0)).toHaveAttribute('aria-pressed', 'true');
  const measured = report.tensors.find(tensor => tensor.nodeId === 'layer_1')!;
  expect(measured.channels!.maps[0]).toEqual(measured.slice.values);
  await tiles.nth(1).click();
  await expect(panel.getByLabel('张量切片轴 1')).toHaveValue('1');
  await expect(panel.locator('.tensor-stale')).toContainText('旧快照已失效');
  expect(calls).toHaveLength(1);
  await panel.getByLabel('张量切片轴 1').fill('0');
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
  await expect(panel.locator('.tensor-histogram .tensor-hover')).toContainText('个元素');
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

test('paints every layer\'s measured values on the 3D blocks and pauses them after an edit', async ({ page }) => {
  const errors: string[] = [], requests: Record<string, unknown>[] = [];
  page.on('pageerror', error => errors.push(error.message));
  page.on('request', request => { if (request.url().endsWith('/api/infer')) requests.push(request.postDataJSON()); });
  await page.goto('/');
  await expect(page.locator('.statusbar')).toContainText('PyTorch');
  // Paused, so the only thing that can change the frame is the paint.
  await page.getByRole('button', { name: '暂停数据流', exact: true }).click();
  const checksum = () => page.locator('canvas').evaluate(canvas => new Promise<number>(resolve => requestAnimationFrame(() => {
    const c = canvas as HTMLCanvasElement, gl = c.getContext('webgl2')!, buffer = new Uint8Array(c.width * c.height * 4);
    gl.readPixels(0, 0, c.width, c.height, gl.RGBA, gl.UNSIGNED_BYTE, buffer);
    let sum = 0; for (let i = 0; i < buffer.length; i += 4) sum = (sum + buffer[i] * (1 + i % 31) + buffer[i + 2]) >>> 0;
    resolve(sum);
  })));
  const plain = await checksum();
  await page.getByRole('button', { name: '张量观测', exact: true }).click();
  await page.getByRole('button', { name: '观测全部层，画到三维视图', exact: true }).click();
  const legend = page.locator('.scene-observation');
  await expect(legend).toContainText('实测值 · 单样本');
  await expect(legend).toContainText('工作台推理 · 随机初始化');
  // One request for every layer, naming none of them.
  expect(requests.at(-1)).toMatchObject({ allNodes: true });
  expect(requests.at(-1)).not.toHaveProperty('nodeIds');
  await expect.poll(checksum).not.toBe(plain);
  // Another graph pauses the paint; undo brings back the graph that was measured.
  await page.getByLabel('模型模板').selectOption('mlp');
  await expect(legend).toContainText('模型已修改，实测着色已暂停');
  await legend.locator('strong').click();
  await page.keyboard.press('Control+z');
  await expect(legend).not.toContainText('模型已修改');
  await expect(legend.locator('.scene-observation-scale')).toBeVisible();
  await legend.getByRole('button', { name: '关闭实测着色', exact: true }).click();
  await expect(legend).toHaveCount(0);
  expect(errors).toEqual([]);
});

const convolution: Graph = {
  version: 1, name: 'Kernel fixture',
  nodes: [
    { id: 'layer_0', name: 'Input image', op: 'Input', params: { shape: [2, 2, 6, 6] }, position: { x: 0, y: 100 } },
    { id: 'layer_1', name: 'Conv', op: 'Conv2d', params: { out_channels: 3, kernel_size: 3, stride: 1, padding: 1, bias: 1 }, position: { x: 210, y: 100 } },
    { id: 'layer_2', name: 'Flatten', op: 'Flatten', params: {}, position: { x: 420, y: 100 } },
    { id: 'layer_3', name: 'Head', op: 'Linear', params: { out_features: 2 }, position: { x: 630, y: 100 } },
    { id: 'layer_4', name: 'Output', op: 'Output', params: {}, position: { x: 840, y: 100 } },
  ],
  edges: [0, 1, 2, 3].map(i => ({ id: `e${i}`, source: `layer_${i}`, target: `layer_${i + 1}` })),
};

test('shows the weights and biases the observed layer ran with, one filter at a time', async ({ page }) => {
  const errors: string[] = []; page.on('pageerror', error => errors.push(error.message));
  await page.addInitScript(value => localStorage.setItem('tensorlab-project', JSON.stringify(value)), convolution);
  await page.goto('/');
  await expect(page.locator('.statusbar')).toContainText('PyTorch');
  await page.getByRole('button', { name: '拓扑图', exact: true }).click();
  await page.locator('.react-flow__node[data-id="layer_1"]').click();
  await page.getByRole('button', { name: '张量观测', exact: true }).click();
  const panel = page.getByRole('region', { name: 'Tensor Inspector', exact: true }), weights = panel.locator('.tensor-parameters');
  await panel.getByRole('button', { name: '运行单样本推理', exact: true }).click();
  const tensors = panel.getByLabel('查看的参数张量', { exact: true });
  await expect(tensors.locator('option')).toHaveText(['weight · [3, 2, 3, 3]', 'bias · [3]']);
  await expect(weights).toContainText('卷积核 · 滤波器 0 · 输入通道 0');
  await expect(weights.locator('.channel-grid-items button')).toHaveCount(2);
  // Another filter is another slice: it needs a rerun, like the activation's.
  await panel.getByLabel('参数切片轴 0', { exact: true }).fill('2');
  await expect(weights).not.toContainText('卷积核');
  await panel.getByRole('button', { name: '运行单样本推理', exact: true }).click();
  await expect(weights).toContainText('卷积核 · 滤波器 2 · 输入通道 0');
  await tensors.selectOption({ label: 'bias · [3]' });
  await expect(panel.getByRole('region', { name: '参数切片', exact: true }).locator('td')).toHaveCount(3);
  expect(errors).toEqual([]);
});

test('explains one output value as the measured inputs times weights plus the bias', async ({ page }, testInfo) => {
  const errors: string[] = []; page.on('pageerror', error => errors.push(error.message));
  await page.addInitScript(value => localStorage.setItem('tensorlab-project', JSON.stringify(value)), convolution);
  await page.goto('/');
  await expect(page.locator('.statusbar')).toContainText('PyTorch');
  await page.getByRole('button', { name: '拓扑图', exact: true }).click();
  await page.locator('.react-flow__node[data-id="layer_1"]').click();
  await page.getByRole('button', { name: '张量观测', exact: true }).click();
  const panel = page.getByRole('region', { name: 'Tensor Inspector', exact: true }), run = panel.getByRole('button', { name: '运行单样本推理', exact: true });
  const position = (axis: string) => panel.getByLabel(`要解释的值 · ${axis}`, { exact: true });
  // It starts at the first channel, in the middle of the 6 x 6 map.
  await expect(position('输出通道')).toHaveValue('0');
  await expect(position('输出行')).toHaveValue('2');
  await expect(position('输出列')).toHaveValue('2');
  let response = page.waitForResponse('**/api/infer'); await run.click();
  let report = await (await response).json() as InferenceReport;
  let explanation = report.explanation!;
  expect(explanation).toMatchObject({ nodeId: 'layer_1', index: [0, 2, 2], rule: 'weighted-sum', count: 18, padded: 0, inputNode: 'layer_0' });
  // The value is the activation the slice shows there, and its terms rebuild it.
  expect(report.tensors.find(tensor => tensor.nodeId === 'layer_1')!.slice.values[2][2]).toBe(explanation.value);
  expect(explanation.recomputed!).toBeCloseTo(explanation.value!, 5);
  const view = panel.getByRole('region', { name: '数值解释', exact: true });
  await expect(view).toContainText('y[0, 2, 2] = Σ 18 项 x·w + b');
  await expect(view.getByRole('table', { name: '贡献最大的项', exact: true }).locator('tbody tr')).toHaveCount(16);
  await expect(view).toContainText('其余 2 项');
  // At the top edge the window reads the zero padding; a new position needs a rerun.
  await position('输出行').fill('0');
  await expect(view).toHaveCount(0);
  await expect(panel.locator('.tensor-stale')).toContainText('旧快照已失效');
  response = page.waitForResponse('**/api/infer'); await run.click();
  report = await (await response).json() as InferenceReport;
  explanation = report.explanation!;
  expect(explanation).toMatchObject({ index: [0, 0, 2], padded: 6 });
  expect(explanation.recomputed!).toBeCloseTo(explanation.value!, 5);
  await expect(view).toContainText('其中 6 项落在补零区域');
  await expect(view.getByText('补零', { exact: true }).first()).toBeVisible();
  await view.screenshot({ path: testInfo.outputPath('value-explanation.png') });
  // A layer that is not a weighted sum or a pooling window offers nothing to explain.
  await page.locator('.react-flow__node[data-id="layer_2"]').click();
  await expect(position('输出通道')).toHaveCount(0);
  await expect(view).toHaveCount(0);
  expect(errors).toEqual([]);
});
test('backpropagates one sample against a chosen class to the observed layer and its weights', async ({ page }) => {
  const errors: string[] = []; page.on('pageerror', error => errors.push(error.message));
  await page.addInitScript(value => localStorage.setItem('tensorlab-project', JSON.stringify(value)), convolution);
  await page.goto('/');
  await expect(page.locator('.statusbar')).toContainText('PyTorch');
  await page.getByRole('button', { name: '拓扑图', exact: true }).click();
  await page.locator('.react-flow__node[data-id="layer_1"]').click();
  await page.getByRole('button', { name: '张量观测', exact: true }).click();
  const panel = page.getByRole('region', { name: 'Tensor Inspector', exact: true }), run = panel.getByRole('button', { name: '运行单样本推理', exact: true });
  const backward = panel.getByRole('checkbox', { name: '同时反向传播一次', exact: true });
  await expect(backward).not.toBeChecked();
  await backward.check();
  await panel.getByLabel('梯度目标类别', { exact: true }).selectOption('1');
  let response = page.waitForResponse('**/api/infer'); await run.click();
  let report = await (await response).json() as InferenceReport;
  const gradient = report.gradient!;
  expect(gradient).toMatchObject({ target: 1, classes: 2 });
  expect(gradient.tensors.map(tensor => tensor.nodeId).sort()).toEqual(['layer_1', 'layer_4']);
  // At the logits the gradient is softmax minus the one-hot target: it sums to zero.
  const logits = gradient.tensors.find(tensor => tensor.nodeId === 'layer_4')!.slice.values[0];
  expect(logits[0]! + logits[1]!).toBeCloseTo(0, 6);
  expect(logits[1]!).toBeCloseTo(gradient.probability! - 1, 6);
  expect(gradient.parameters.map(entry => [entry.name, entry.shape])).toEqual([['weight', [3, 2, 3, 3]], ['bias', [3]]]);
  await expect(panel.locator('.tensor-gradient-summary')).toContainText('单样本梯度 · 交叉熵对类别 1');
  await expect(panel.getByRole('region', { name: '梯度切片', exact: true })).toBeVisible();
  await expect(panel.getByRole('region', { name: '参数梯度切片', exact: true })).toBeVisible();
  // The values are the ones a run without the backward pass reports.
  await panel.getByLabel('梯度目标类别', { exact: true }).selectOption('0');
  await expect(panel.locator('.tensor-gradient')).toHaveCount(0);
  await expect(panel.locator('.tensor-stale')).toContainText('旧快照已失效');
  await backward.uncheck();
  response = page.waitForResponse('**/api/infer'); await run.click();
  const plain = await (await response).json() as InferenceReport;
  expect(plain.gradient).toBeNull();
  expect(plain.tensors).toEqual(report.tensors);
  expect(errors).toEqual([]);
});
