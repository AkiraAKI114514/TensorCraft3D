import { test, expect, type Page, type WebSocketRoute } from '@playwright/test';
import type { Graph } from '../src/types';
import type { InferenceReport } from '../src/inferenceTypes';
import type { TrainedModelMetadata } from '../src/trainedModel';

const graph: Graph = {
  version: 1, name: 'Trained classifier',
  nodes: [
    { id: 'layer_0', name: 'Input', op: 'Input', params: { shape: [4, 1, 2, 2] }, position: { x: 0, y: 100 } },
    { id: 'layer_1', name: 'Dense', op: 'Linear', params: { out_features: 8 }, position: { x: 210, y: 100 } },
    { id: 'layer_2', name: 'BN', op: 'BatchNorm2d', params: {}, position: { x: 420, y: 100 } },
    { id: 'layer_3', name: 'Flatten', op: 'Flatten', params: {}, position: { x: 630, y: 100 } },
    { id: 'layer_4', name: 'Logits', op: 'Linear', params: { out_features: 2 }, position: { x: 840, y: 100 } },
    { id: 'layer_5', name: 'Output', op: 'Output', params: {}, position: { x: 1050, y: 100 } }
  ],
  edges: Array.from({ length: 5 }, (_, i) => ({ id: `e${i}`, source: `layer_${i}`, target: `layer_${i + 1}` }))
};

async function train(page: Page, dataset: 'synthetic' | 'csv' = 'synthetic', device = 'cpu') {
  await expect(page.locator('.validation-badge')).toContainText('形状校验通过');
  const done = new Promise<TrainedModelMetadata>(resolve => page.once('websocket', socket => socket.on('framereceived', frame => {
    const message = JSON.parse(frame.payload.toString());
    if (message.type === 'done') resolve(message.model);
  })));
  await page.getByRole('button', { name: '训练', exact: true }).click();
  await page.getByLabel('训练轮次', { exact: true }).fill('2');
  await page.getByLabel('合成样本数', { exact: true }).fill('64');
  await page.locator('label.field-label').filter({ hasText: '计算设备' }).locator('select').selectOption(device);
  await page.locator('label.field-label').filter({ hasText: '数据集' }).locator('select').selectOption(dataset);
  if (dataset === 'csv') {
    const rows = Array.from({ length: 32 }, (_, i) => [100 + i, 200 + i * 2, 300 - i, 50 + i % 3, i % 2].join(',')).join('\n');
    await page.locator('input[type="file"][accept=".csv"]').setInputFiles({ name: 'samples.csv', mimeType: 'text/csv', buffer: Buffer.from(rows) });
  }
  await expect(page.getByRole('button', { name: '开始真实训练', exact: true })).toBeEnabled();
  await page.getByRole('button', { name: '开始真实训练', exact: true }).click();
  await expect(page.locator('.train-status')).toContainText('训练完成', { timeout: 30000 });
  const metadata = await done;
  expect(metadata.modelId).toBeTruthy();
  return metadata;
}

async function infer(page: Page): Promise<InferenceReport> {
  const response = page.waitForResponse('**/api/infer');
  await page.getByRole('button', { name: '运行单样本推理', exact: true }).click();
  const result = await response;
  expect(result.ok(), await result.text()).toBe(true);
  return result.json();
}

test.beforeEach(async ({ page }) => {
  await page.addInitScript(value => localStorage.setItem('tensorlab-project', JSON.stringify(value)), graph);
});

test('observes trained weights, preserves identity on rename, invalidates edits, and never silently falls back', async ({ page }, testInfo) => {
  const errors: string[] = []; page.on('pageerror', error => errors.push(error.message));
  await page.goto('/');
  const metadata = await train(page);
  await page.getByRole('button', { name: '张量观测', exact: true }).click();
  const panel = page.getByRole('region', { name: 'Tensor Inspector', exact: true });
  await expect(panel.getByLabel('推理权重来源')).toHaveValue('trained');
  await expect(panel.locator('.tensor-trained-model')).toContainText(metadata.modelId);
  await expect(panel.locator('.tensor-provenance:not(.tensor-trained-model)')).toHaveCount(0);
  await panel.getByLabel('推理输入来源').selectOption('provided');
  await panel.getByLabel('推理输入 JSON').fill(JSON.stringify({ layer_0: [1, 2, 3, 4] }));
  const first = await infer(page);
  expect(first.weights).toBe('trained'); expect(first.model).toEqual(metadata);
  expect(first.inputTransform).toBe('none');
  expect(first.tensors.find(tensor => tensor.nodeId === 'layer_5')!.shape).toEqual([1, 2]);
  await panel.getByLabel('推理随机种子').fill('7');
  const second = await infer(page);
  expect(second.tensors).toEqual(first.tensors);
  await page.getByLabel('项目名称').fill('Renamed classifier');
  await expect(panel.getByRole('button', { name: '运行单样本推理', exact: true })).toBeEnabled();
  expect((await infer(page)).model!.modelId).toBe(metadata.modelId);
  await page.getByRole('button', { name: '层属性', exact: true }).click();
  await page.getByLabel('out_features', { exact: true }).fill('9');
  await page.getByRole('button', { name: '张量观测', exact: true }).click();
  await expect(panel).toContainText('当前计算图没有已确认的训练权重');
  await expect(panel.getByRole('button', { name: '运行单样本推理', exact: true })).toBeDisabled();
  await expect(panel.locator('.tensor-provenance:not(.tensor-trained-model)')).toHaveCount(0);
  await page.getByRole('button', { name: '撤销', exact: true }).click();
  await expect(panel.getByRole('button', { name: '运行单样本推理', exact: true })).toBeEnabled();
  // Simulate a restart/eviction response; the UI must not retry with random weights.
  let calls = 0;
  await page.route('**/api/infer', route => { calls++; return route.fulfill({ status: 422, json: { detail: 'Trained model snapshot is unavailable (backend restarted)' } }); });
  await panel.getByRole('button', { name: '运行单样本推理', exact: true }).click();
  await expect(panel.getByRole('alert')).toContainText('snapshot is unavailable');
  await expect(panel.getByLabel('推理权重来源')).toHaveValue('trained');
  expect(calls).toBe(1);
  await expect(panel.locator('.tensor-provenance:not(.tensor-trained-model)')).toHaveCount(0);
  await page.unroute('**/api/infer');
  await page.route('**/api/infer', async route => {
    const response = await route.fetch();
    const body = await response.json();
    await route.fulfill({ response, json: { ...body, weights: 'random-initialized', model: null } });
  });
  await panel.getByRole('button', { name: '运行单样本推理', exact: true }).click();
  await expect(panel.getByRole('alert')).toContainText('权重来源与请求不一致');
  await expect(panel.locator('.tensor-provenance:not(.tensor-trained-model)')).toHaveCount(0);
  await page.unroute('**/api/infer');
  await panel.getByLabel('推理权重来源').selectOption('random');
  const random = await infer(page);
  expect(random.weights).toBe('random-initialized'); expect(random.model).toBeNull();
  expect(random.tensors).not.toEqual(first.tensors);
  await panel.getByLabel('推理权重来源').selectOption('trained');
  await infer(page);
  await panel.locator('.tensor-provenance:not(.tensor-trained-model)').scrollIntoViewIfNeeded();
  await page.locator('.inspector').screenshot({ path: testInfo.outputPath('trained-inspector.png') });
  await page.setViewportSize({ width: 390, height: 844 });
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  expect(await panel.evaluate(element => element.scrollWidth <= element.clientWidth + 1)).toBe(true);
  await page.screenshot({ path: testInfo.outputPath('trained-inspector-mobile.png') });
  expect(errors).toEqual([]);
});

test('replays fitted CSV preprocessing and uses the same trained state on CPU and CUDA', async ({ page }) => {
  await page.goto('/');
  const metadata = await train(page, 'csv');
  expect(metadata.preprocessing).toBe('csv-standardized');
  await page.getByRole('button', { name: '张量观测', exact: true }).click();
  const panel = page.getByRole('region', { name: 'Tensor Inspector', exact: true });
  await panel.getByLabel('推理输入来源').selectOption('provided');
  await panel.getByLabel('推理输入 JSON').fill(JSON.stringify({ layer_0: [110, 220, 290, 51] }));
  await expect(panel).toContainText('输入原始 CSV 数值');
  const cpu = await infer(page);
  expect(cpu.inputTransform).toBe('csv-standardized');
  const health = await (await page.request.get('/api/health')).json();
  if (health.cuda) {
    await panel.getByLabel('推理设备').selectOption('cuda');
    const gpu = await infer(page);
    expect(gpu.device).toBe('CUDA'); expect(gpu.model).toEqual(cpu.model);
    for (let i = 0; i < cpu.tensors.length; i++) for (let j = 0; j < cpu.tensors[i].slice.values[0].length; j++)
      expect(gpu.tensors[i].slice.values[0][j]).toBeCloseTo(cpu.tensors[i].slice.values[0][j]!, 4);
    const cudaTrained = await train(page, 'synthetic', 'cuda');
    expect(cudaTrained.device).toBe('CUDA');
    await panel.getByLabel('推理设备').selectOption('cpu');
    expect((await infer(page)).model!.modelId).toBe(cudaTrained.modelId);
  }
});

test('retraining invalidates a pending snapshot and page reload does not claim persisted weights', async ({ page }) => {
  await page.goto('/');
  const first = await train(page);
  await page.getByRole('button', { name: '张量观测', exact: true }).click();
  const panel = page.getByRole('region', { name: 'Tensor Inspector', exact: true });
  let release!: () => void;
  const held = new Promise<void>(resolve => { release = resolve; });
  await page.route('**/api/infer', async route => { const response = await route.fetch(); await held; await route.fulfill({ response }); });
  await panel.getByRole('button', { name: '运行单样本推理', exact: true }).click();
  await page.getByRole('button', { name: '训练', exact: true }).click();
  await expect(page.getByRole('button', { name: '开始真实训练', exact: true })).toBeDisabled();
  await page.getByRole('button', { name: '关闭对话框', exact: true }).click();
  release();
  await expect(panel.locator('.tensor-provenance:not(.tensor-trained-model)')).toBeVisible();
  await page.unroute('**/api/infer');
  const second = await train(page);
  expect(second.modelId).not.toBe(first.modelId);
  await expect(panel.locator('.tensor-trained-model')).toContainText(second.modelId);
  await expect(panel.locator('.tensor-provenance:not(.tensor-trained-model)')).toHaveCount(0);
  await expect(panel.locator('.tensor-stale')).toContainText('旧快照已失效');
  await infer(page);
  await page.reload();
  await page.getByRole('button', { name: '张量观测', exact: true }).click();
  await panel.getByLabel('推理权重来源').selectOption('trained');
  await expect(panel).toContainText('当前计算图没有已确认的训练权重');
  await expect(panel.getByRole('button', { name: '运行单样本推理', exact: true })).toBeDisabled();
});

test('does not confirm weights after stopped, failed or disconnected training sessions', async ({ page }) => {
  const errors: string[] = []; page.on('pageerror', error => errors.push(error.message));
  let socket: WebSocketRoute | undefined;
  let started = 0;
  await page.routeWebSocket('**/api/train', ws => {
    socket = ws;
    ws.onMessage(data => {
      const message = JSON.parse(data.toString());
      if (message.graph) started++;
      if (message.type === 'stop') ws.send(JSON.stringify({ type: 'done', reason: 'stopped' }));
    });
  });
  await page.goto('/');
  await expect(page.locator('.statusbar')).toContainText('PyTorch');
  const start = async () => {
    const previous = started;
    await page.getByRole('button', { name: '训练', exact: true }).click();
    await page.getByRole('button', { name: '开始真实训练', exact: true }).click();
    await expect.poll(() => started).toBe(previous + 1);
  };
  const checkNoModel = async () => {
    await page.getByRole('button', { name: '张量观测', exact: true }).click();
    const panel = page.getByRole('region', { name: 'Tensor Inspector', exact: true });
    await panel.getByLabel('推理权重来源').selectOption('trained');
    await expect(panel).toContainText('当前计算图没有已确认的训练权重');
    await expect(panel.getByRole('button', { name: '运行单样本推理', exact: true })).toBeDisabled();
  };
  await start();
  await page.getByRole('button', { name: '停止', exact: true }).click();
  await expect(page.locator('.train-status')).toContainText('已停止');
  await checkNoModel();
  await start();
  socket!.send(JSON.stringify({ type: 'error', message: 'fixture training error' }));
  await expect(page.locator('.train-status')).toContainText('训练失败');
  await checkNoModel();
  await start();
  await socket!.close({ code: 1011, reason: 'fixture disconnect' });
  await expect(page.locator('.train-status')).toContainText('连接中断 · 未收到训练结果');
  await checkNoModel();
  await expect(page.getByRole('button', { name: '训练', exact: true })).toBeEnabled();
  expect(errors).toEqual([]);
});
