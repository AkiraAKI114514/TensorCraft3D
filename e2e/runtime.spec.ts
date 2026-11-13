import { execFileSync } from 'node:child_process';
import { test, expect } from '@playwright/test';
import type { InferenceReport, TensorSnapshot } from '../src/inferenceTypes';
import type { Graph, Layer } from '../src/types';

const source = `import math
import torch
from torch import nn

class PositionClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        position = torch.arange(8, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, 4, 2, dtype=torch.float32) * (-math.log(10000.0) / 4))
        pe = torch.zeros(1, 8, 4)
        pe[0, :, 0::2] = torch.sin(position * div)
        pe[0, :, 1::2] = torch.cos(position * div)
        self.register_buffer("pe", pe)
        self.head = nn.Linear(4, 2)
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.head(x[:, 1::2][:, -1])
`;

test('imports, edits, exports and trains a classifier with actual position buffers and slices', async ({ page }, testInfo) => {
  const errors: string[] = []; page.on('pageerror', e => errors.push(e.message));
  const environmentRequests: string[] = []; page.on('request', request => { if (request.url().includes('/api/environment')) environmentRequests.push(request.url()); });
  await page.goto('/');
  await expect(page.locator('.statusbar')).toContainText('PyTorch');
  await page.getByRole('button', { name: '导入 PyTorch 代码', exact: true }).click();
  await page.getByRole('textbox', { name: 'PyTorch 源代码', exact: true }).fill(source);
  await page.getByRole('button', { name: '解析代码', exact: true }).click();
  await expect(page.getByLabel('导入输入形状 x')).toBeVisible();
  await page.getByLabel('导入输入形状 x').fill('2, 6, 4');
  await page.getByRole('button', { name: '解析代码', exact: true }).click();
  await expect(page.locator('.import-preview')).toContainText('ConstantAdd');
  await expect(page.locator('.import-preview')).toContainText('Slice');
  await expect(page.locator('.import-preview')).toContainText('Select');
  await page.getByRole('button', { name: '导入模型', exact: true }).click();
  await expect(page.locator('.validation-badge')).toContainText('形状校验通过');
  const graph = await page.evaluate(() => JSON.parse(localStorage.getItem('tensorlab-project')!));
  const constant = graph.nodes.find((n: { op: string }) => n.op === 'ConstantAdd');
  expect(constant.params.values.some((v: number) => v !== 0)).toBe(true);
  await page.getByRole('button', { name: '拓扑图', exact: true }).click();
  await page.locator(`.react-flow__node[data-id="${constant.id}"]`).click();
  const values = page.getByLabel('values', { exact: true });
  await expect(values).toHaveValue(JSON.stringify(constant.params.values));
  const changed = [...constant.params.values]; changed[0] = 0.5;
  await values.fill(JSON.stringify(changed)); await values.blur();
  await page.getByRole('button', { name: '撤销', exact: true }).click();
  await expect(values).toHaveValue(JSON.stringify(constant.params.values));
  await expect(page.locator('.validation-badge')).toContainText('形状校验通过');
  await page.getByRole('button', { name: '导出代码', exact: true }).click();
  await expect(page.locator('.code-preview')).toContainText('TensorLabConstantAdd');
  await expect(page.locator('.code-preview')).toContainText('torch.select');
  const download = page.waitForEvent('download');
  await page.getByRole('button', { name: '下载 .py', exact: true }).click();
  expect((await download).suggestedFilename()).toMatch(/\.py$/);
  await page.getByRole('button', { name: '关闭对话框', exact: true }).click();
  await page.getByRole('button', { name: '训练', exact: true }).click();
  await page.getByLabel('训练轮次', { exact: true }).fill('2');
  await page.getByLabel('合成样本数', { exact: true }).fill('64');
  await page.locator('label.field-label').filter({ hasText: '计算设备' }).locator('select').selectOption('cpu');
  await page.getByRole('button', { name: '开始真实训练', exact: true }).click();
  await expect(page.locator('.train-status')).toContainText('训练完成', { timeout: 30000 });
  await expect(page.locator('.source-badge').first()).toContainText('真实训练');
  const health = await (await page.request.get('/api/health')).json();
  if (health.cuda) {
    const metrics: { device: string; metric: { trainLoss: number; valLoss: number; gradNorm: number; source: string } }[] = [];
    const trainingErrors: string[] = [];
    page.on('websocket', socket => socket.on('framereceived', frame => {
      const message = JSON.parse(frame.payload.toString());
      if (message.type === 'metric') metrics.push(message);
      if (message.type === 'error') trainingErrors.push(message.message);
    }));
    await page.getByRole('button', { name: '训练', exact: true }).click();
    await page.locator('label.field-label').filter({ hasText: '计算设备' }).locator('select').selectOption('cuda');
    await page.getByRole('button', { name: '开始真实训练', exact: true }).click();
    await expect.poll(() => ({ count: metrics.length, errors: trainingErrors }), { timeout: 30000 }).toEqual({ count: 2, errors: [] });
    await expect(page.locator('.train-status')).toContainText('训练完成');
    expect(metrics.every(m => m.device === 'CUDA' && m.metric.source === 'training' && Number.isFinite(m.metric.trainLoss) && Number.isFinite(m.metric.valLoss) && Number.isFinite(m.metric.gradNorm) && m.metric.gradNorm > 0)).toBe(true);
  }
  expect(environmentRequests).toEqual([]);
  await page.getByRole('button', { name: '三维视图', exact: true }).click();
  await expect(page.locator('canvas')).toBeVisible();
  const coloredPixels = () => page.locator('canvas').evaluate(canvas => new Promise<number>(resolve => requestAnimationFrame(() => {
    const element = canvas as HTMLCanvasElement, gl = element.getContext('webgl2')!;
    const pixels = new Uint8Array(element.width * element.height * 4);
    gl.readPixels(0, 0, element.width, element.height, gl.RGBA, gl.UNSIGNED_BYTE, pixels);
    let colored = 0;
    for (let i = 0; i < pixels.length; i += 4) if (Math.max(pixels[i], pixels[i + 1], pixels[i + 2]) - Math.min(pixels[i], pixels[i + 1], pixels[i + 2]) > 30) colored++;
    resolve(colored);
  })));
  await expect.poll(coloredPixels, { timeout: 15000 }).toBeGreaterThan(500);
  await page.screenshot({ path: testInfo.outputPath('position-classifier.png') });
  await page.setViewportSize({ width: 390, height: 844 });
  await expect.poll(coloredPixels, { timeout: 15000 }).toBeGreaterThan(200);
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  await page.screenshot({ path: testInfo.outputPath('position-classifier-mobile.png') });
  expect(errors).toEqual([]);
});

test('shows CUDA diagnostics and setup without automatically running GPU tests', async ({ page, request }, testInfo) => {
  const response = await request.get('/api/environment'); expect(response.ok()).toBe(true);
  const report = await response.json();
  let smokeCalls = 0; page.on('request', request => { if (request.url().endsWith('/api/environment/smoke')) smokeCalls++; });
  const errors: string[] = []; page.on('pageerror', e => errors.push(e.message));
  await page.goto('/');
  await page.getByRole('button', { name: '训练', exact: true }).click();
  const start = page.getByRole('button', { name: '开始真实训练', exact: true });
  await expect(start).toBeEnabled();
  await page.getByText('CUDA 环境与配置', { exact: true }).click();
  const panel = page.getByRole('region', { name: 'CUDA 环境诊断', exact: true });
  await expect(panel).toContainText(report.torch.version);
  if (report.gpu.present) await expect(panel).toContainText(report.gpu.devices[0].name);
  await expect(panel).toContainText('.\\setup-training.ps1 -Variant cu128');
  const smoke = panel.getByRole('button', { name: '测试 GPU 前向与反向', exact: true });
  if (report.cuda.available) await expect(smoke).toBeEnabled(); else await expect(smoke).toBeDisabled();
  expect(smokeCalls).toBe(0);
  let releaseRefresh!: () => void;
  const refreshHeld = new Promise<void>(resolve => { releaseRefresh = resolve; });
  await page.route('**/api/environment', async route => { const response = await route.fetch(); await refreshHeld; await route.fulfill({ response }); });
  await panel.getByRole('button', { name: '刷新', exact: true }).click();
  await expect(start).toBeDisabled();
  await page.getByText('CUDA 环境与配置', { exact: true }).click();
  await expect(start).toBeDisabled();
  await page.getByRole('button', { name: '关闭对话框', exact: true }).click();
  await page.getByRole('button', { name: '训练', exact: true }).click();
  await expect(start).toBeDisabled();
  releaseRefresh();
  await expect(start).toBeEnabled();
  await page.getByText('CUDA 环境与配置', { exact: true }).click();
  await expect(panel).toContainText(report.torch.version);
  await expect(panel.getByRole('button', { name: '刷新', exact: true })).toBeEnabled();
  expect(smokeCalls).toBe(0);
  if (report.cuda.available) {
    let releaseSmoke!: () => void;
    const smokeHeld = new Promise<void>(resolve => { releaseSmoke = resolve; });
    await page.route('**/api/environment/smoke', async route => { const response = await route.fetch(); await smokeHeld; await route.fulfill({ response }); });
    await smoke.click();
    await expect(start).toBeDisabled();
    releaseSmoke();
    await expect(panel.getByRole('status')).toContainText('GPU 前向/反向成功', { timeout: 15000 });
    await expect(start).toBeEnabled();
    expect(smokeCalls).toBe(1);
  } else {
    const failure = await request.post('/api/environment/smoke');
    expect(failure.status()).toBe(422); expect((await failure.json()).detail).toContain('CUDA');
  }
  await page.screenshot({ path: testInfo.outputPath('cuda-desktop.png') });
  await page.setViewportSize({ width: 390, height: 844 });
  await expect(panel).toBeVisible();
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  expect(await panel.evaluate(element => element.scrollWidth <= element.clientWidth + 1)).toBe(true);
  await page.screenshot({ path: testInfo.outputPath('cuda-mobile.png') });
  expect(errors).toEqual([]);
});

test('the getting-started guide follows what the user does and never runs anything itself', async ({ page }) => {
  const errors: string[] = []; page.on('pageerror', e => errors.push(e.message));
  const inferCalls: string[] = []; page.on('request', request => { if (request.url().endsWith('/api/infer')) inferCalls.push(request.url()); });
  let sockets = 0; page.on('websocket', () => { sockets++; });
  await page.goto('/');
  await expect(page.locator('.statusbar')).toContainText('PyTorch');
  await page.getByRole('button', { name: '上手引导', exact: true }).click();
  const guide = page.getByRole('region', { name: '上手引导', exact: true });
  await expect(guide).toContainText('已完成 0 / 5 步');
  // Each button only navigates: here it moves focus to the template picker.
  await guide.getByRole('button', { name: '去选模板', exact: true }).click();
  await expect(page.getByLabel('模型模板')).toBeFocused();
  await page.getByLabel('模型模板').selectOption('mlp');
  await expect(guide).toContainText('已完成 1 / 5 步');
  await page.getByRole('button', { name: '拓扑图', exact: true }).click();
  await page.locator('.react-flow__node').first().click();
  await expect(guide).toContainText('已完成 2 / 5 步');
  // Opening the inspector is not an observation; only the user's own run completes the step.
  await guide.getByRole('button', { name: '打开张量观测', exact: true }).click();
  const panel = page.getByRole('region', { name: 'Tensor Inspector', exact: true });
  const run = panel.getByRole('button', { name: '运行单样本推理', exact: true });
  await expect(run).toBeEnabled();
  await expect(guide).toContainText('已完成 2 / 5 步');
  expect(inferCalls).toHaveLength(0);
  const response = page.waitForResponse('**/api/infer');
  await run.click();
  expect((await response).ok()).toBe(true);
  await expect(guide).toContainText('已完成 3 / 5 步');
  // The training step opens the dialog and stops there: no socket, no progress.
  await guide.getByRole('button', { name: '打开训练设置', exact: true }).click();
  await expect(page.getByRole('button', { name: '开始真实训练', exact: true })).toBeVisible();
  await page.getByRole('button', { name: '关闭对话框', exact: true }).click();
  expect(sockets).toBe(0);
  await expect(guide).toContainText('已完成 3 / 5 步');
  await guide.getByRole('button', { name: '打开代码导出', exact: true }).click();
  await page.getByRole('button', { name: '关闭对话框', exact: true }).click();
  await expect(guide).toContainText('已完成 4 / 5 步');
  // Progress is the user's, so it survives a reload and can be started over.
  await page.reload();
  await page.getByRole('button', { name: '上手引导', exact: true }).click();
  await expect(guide).toContainText('已完成 4 / 5 步');
  await guide.getByRole('button', { name: '重新开始', exact: true }).click();
  await expect(guide).toContainText('已完成 0 / 5 步');
  expect(errors).toEqual([]);
});

test('imports a layer called twice as shared weights and lets the panel untie and retie it', async ({ page }) => {
  const errors: string[] = []; page.on('pageerror', e => errors.push(e.message));
  const tied = `from torch import nn
import torch.nn.functional as F

class TiedNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.cell = nn.Linear(8, 8)
        self.head = nn.Linear(8, 2)
    def forward(self, x):
        x = F.relu(self.cell(x))
        x = F.relu(self.cell(x))
        return self.head(x)
`;
  await page.goto('/');
  await expect(page.locator('.statusbar')).toContainText('PyTorch');
  await page.getByRole('button', { name: '导入 PyTorch 代码', exact: true }).click();
  const dialog = page.getByRole('dialog', { name: '导入 PyTorch 代码', exact: true });
  await dialog.getByRole('textbox', { name: 'PyTorch 源代码', exact: true }).fill(tied);
  await dialog.getByRole('button', { name: '解析代码', exact: true }).click();
  await dialog.getByLabel('导入输入形状 x').fill('2, 8');
  await dialog.getByRole('button', { name: '解析代码', exact: true }).click();
  await expect(dialog.locator('.diagnostic-item.info', { hasText: 'SHARED_WEIGHTS' })).toContainText('已导入为权重共享');
  await expect(dialog.getByRole('alert')).toHaveCount(0);
  await dialog.getByRole('button', { name: '导入模型', exact: true }).click();
  await expect(page.locator('.validation-badge')).toContainText('形状校验通过');
  // 8*8+8 for the cell, counted once, plus 8*2+2 for the head.
  const total = page.locator('.model-summary strong').first();
  await expect(total).toHaveText('90');
  const graph = await page.evaluate(() => JSON.parse(localStorage.getItem('tensorlab-project')!) as Graph);
  const sharing = graph.nodes.filter(node => node.shares !== undefined);
  expect(sharing).toHaveLength(1);
  const [{ id, shares: owner }] = sharing;
  await page.getByRole('button', { name: '拓扑图', exact: true }).click();
  await page.locator(`.react-flow__node[data-id="${id}"]`).click();
  const share = page.getByLabel('共享权重自', { exact: true });
  await expect(share).toHaveValue(owner!);
  await share.selectOption('');
  await expect(total).toHaveText('162');
  await share.selectOption(owner!);
  await expect(total).toHaveText('90');
  await page.getByRole('button', { name: '导出代码', exact: true }).click();
  await expect(page.locator('.code-preview')).toContainText(`values["${id}"] = self.layers["${owner}"](`);
  expect(errors).toEqual([]);
});

test('opens an observation snapshot written by an outside Python process, read-only and bound to its graph', async ({ page }) => {
  const errors: string[] = []; page.on('pageerror', e => errors.push(e.message));
  const inferences: string[] = []; page.on('request', request => { if (request.url().includes('/api/infer')) inferences.push(request.url()); });
  const layer = (id: string, op: Layer['op'], params: Layer['params'], x: number): Layer => ({ id, name: id, op, params, position: { x, y: 160 } });
  const nodes = [layer('image', 'Input', { shape: [2, 1, 6, 6] }, 0), layer('conv', 'Conv2d', { out_channels: 4, kernel_size: 3, stride: 1, padding: 1, bias: 1 }, 180), layer('relu', 'ReLU', {}, 360), layer('flat', 'Flatten', {}, 540), layer('head', 'Linear', { out_features: 2 }, 720), layer('out', 'Output', {}, 900)];
  const graph: Graph = { version: 1, name: 'Observed CNN', nodes, edges: nodes.slice(1).map((node, index) => ({ id: `e${index}`, source: nodes[index].id, target: node.id })) };
  // The outside process: weights it trained (here: seeded) and one real sample, written by the
  // snapshot writer. The browser receives only the file.
  const snapshot = execFileSync('.venv/Scripts/python.exe', ['-c', `import json, sys, torch
from backend.observation_snapshot import capture_snapshot, snapshot_json
from backend.training import build_model
graph = json.load(sys.stdin)
torch.manual_seed(5)
model, _ = build_model(graph)
print(snapshot_json(capture_snapshot(graph, model.state_dict(), torch.randn(1, 1, 6, 6), label="one sample from the outside run")))
`], { input: JSON.stringify(graph), encoding: 'utf8', timeout: 60000 });
  await page.goto('/');
  await page.getByRole('button', { name: '张量观测', exact: true }).click();
  const inspector = page.getByRole('region', { name: 'Tensor Inspector', exact: true });
  await inspector.getByLabel('观测来源', { exact: true }).selectOption('offline');
  await inspector.getByLabel('观测快照文件', { exact: true }).setInputFiles({ name: 'sample.observation.json', mimeType: 'application/json', buffer: Buffer.from(snapshot) });
  await expect(inspector.locator('.tensor-offline')).toContainText('离线快照 · CPU · eval');
  await expect(inspector.locator('.tensor-offline')).toContainText('one sample from the outside run');
  await expect(inspector.locator('.tensor-offline')).toContainText('外部进程加载的权重');
  // The starting project is another graph, so nothing is shown until its own graph is opened.
  const unbound = inspector.locator('.tensor-stale');
  await expect(unbound).toContainText('快照测量的是另一张计算图');
  await inspector.getByRole('button', { name: '打开快照中的模型', exact: true }).click();
  await expect(page.locator('.validation-badge')).toContainText('形状校验通过');
  await expect(unbound).toHaveCount(0);
  await page.getByRole('button', { name: '拓扑图', exact: true }).click();
  await page.locator('.react-flow__node[data-id="conv"]').click();
  await page.getByRole('button', { name: '张量观测', exact: true }).click();
  await expect(inspector).toContainText('conv · 实测输出');
  await expect(inspector.locator('.tensor-activation .tensor-stats')).toContainText('[1, 4, 6, 6]');
  // All four captured channels are in the grid; choosing one moves the window without a run.
  const channels = inspector.locator('.tensor-activation .channel-grid-items button');
  await expect(channels).toHaveCount(4);
  await expect(inspector.locator('.tensor-activation .channel-grid')).toContainText('点选通道直接查看它的窗口');
  await channels.nth(2).click();
  await expect(inspector.locator('.tensor-activation .tensor-semantic')).toContainText('特征图 · 通道 2');
  await expect(inspector.getByRole('region', { name: '实测张量切片', exact: true }).locator('td').first()).toHaveAttribute('title', /^\[0, 2, 0, 0\] = /);
  // Undo returns to the earlier project, which the snapshot does not describe; redo binds it again.
  await page.keyboard.press('Control+z');
  await expect(unbound).toContainText('快照测量的是另一张计算图');
  await expect(inspector.locator('.tensor-stats')).toHaveCount(0);
  await page.keyboard.press('Control+y');
  await expect(unbound).toHaveCount(0);
  await inspector.getByRole('button', { name: '关闭快照', exact: true }).click();
  await expect(inspector.locator('.tensor-offline')).toHaveCount(0);
  expect(inferences).toEqual([]);
  expect(errors).toEqual([]);
});

test('observes a layer inside a repeated block live and from a snapshot, at the same chosen instance', async ({ page }) => {
  const errors: string[] = []; page.on('pageerror', e => errors.push(e.message));
  const inferences: string[][] = []; page.on('request', request => { if (request.url().endsWith('/api/infer')) inferences.push(request.postDataJSON().nodeIds); });
  const layer = (id: string, op: Layer['op'], params: Layer['params'], x: number, extra: Partial<Layer> = {}): Layer => ({ id, name: id, op, params, position: { x, y: 160 }, ...extra });
  const nodes = [layer('features', 'Input', { shape: [2, 8] }, 0), layer('stack', 'Group', {}, 220, { name: 'Cell ×3', subgraph: 'cell', repeat: 3 }), layer('head', 'Linear', { out_features: 2 }, 440), layer('out', 'Output', {}, 660)];
  const graph: Graph = {
    version: 1, name: 'Folded cells', nodes, edges: nodes.slice(1).map((node, index) => ({ id: `e${index}`, source: nodes[index].id, target: node.id })),
    subgraphs: { cell: { id: 'cell', name: 'Cell', origin: 'manual', nodes: [layer('fc', 'Linear', { out_features: 8 }, 0), layer('act', 'Tanh', {}, 220)], edges: [{ id: 'c0', source: 'fc', target: 'act' }] } },
  };
  // The outside writer with the inspector's own defaults (seed 42, seeded weights, synthetic
  // sample), so the snapshot holds the numbers a live run measures.
  const file = execFileSync('.venv/Scripts/python.exe', ['-c', `import json, sys
from backend.observation_snapshot import capture_snapshot, snapshot_json
print(snapshot_json(capture_snapshot(json.load(sys.stdin), seed=42)))
`], { input: JSON.stringify(graph), encoding: 'utf8', timeout: 60000 });
  const written = JSON.parse(file) as { tensors: TensorSnapshot[] };
  const offlineTensor = (nodeId: string) => written.tensors.find(tensor => tensor.nodeId === nodeId)!;
  await page.addInitScript(value => localStorage.setItem('tensorlab-project', JSON.stringify(value)), graph);
  await page.goto('/');
  await expect(page.locator('.statusbar')).toContainText('PyTorch');
  await page.getByRole('button', { name: '拓扑图', exact: true }).click();
  await page.locator('.react-flow__node[data-id="stack"]').click();
  await page.getByRole('button', { name: '张量观测', exact: true }).click();
  const panel = page.getByRole('region', { name: 'Tensor Inspector', exact: true });
  const run = panel.getByRole('button', { name: '运行单样本推理', exact: true });
  // The block itself is what leaves it: its last instance's exit layer.
  await expect(panel).toContainText('结构块的输出来自最后一个实例的出口层 stack/2/act。');
  const blockRun = page.waitForResponse('**/api/infer');
  await run.click();
  const blockReport = await (await blockRun).json() as InferenceReport;
  expect(blockReport.tensors.find(tensor => tensor.nodeId === 'stack/2/act')).toEqual(offlineTensor('stack/2/act'));
  await expect(panel).toContainText('Cell ×3 · 实测输出');
  const liveBlock = await panel.locator('.tensor-activation .tensor-stats').innerText();
  // The snapshot is looked up through the same mapping, so it shows the same layer.
  const source = panel.getByLabel('观测来源', { exact: true });
  await source.selectOption('offline');
  await panel.getByLabel('观测快照文件', { exact: true }).setInputFiles({ name: 'cells.observation.json', mimeType: 'application/json', buffer: Buffer.from(file) });
  await expect(panel.locator('.tensor-offline')).toContainText('离线快照 · CPU · eval');
  await expect(panel.locator('.tensor-stale')).toHaveCount(0);
  await expect(panel.locator('.tensor-activation .tensor-stats')).toHaveText(liveBlock, { useInnerText: true });
  await source.selectOption('live');

  // Inside the block a layer runs once per instance; instance 0 is observed until another is chosen.
  await page.getByRole('button', { name: '层属性', exact: true }).click();
  await page.getByRole('button', { name: '进入结构块', exact: true }).click();
  await page.locator('.react-flow__node[data-id="fc"]').click();
  await page.getByRole('button', { name: '张量观测', exact: true }).click();
  const instance = panel.getByLabel('观测实例 · Cell', { exact: true });
  await expect(instance).toHaveValue('0');
  await expect(instance).toHaveAttribute('max', '2');
  await expect(panel).toContainText('观测展开后的层 stack/0/fc。');
  await expect(panel).toContainText('当前层未包含在本次采样中');
  await instance.fill('1');
  await expect(panel).toContainText('观测展开后的层 stack/1/fc。');
  const layerRun = page.waitForResponse('**/api/infer');
  await run.click();
  const layerReport = await (await layerRun).json() as InferenceReport;
  expect(layerReport.tensors.find(tensor => tensor.nodeId === 'stack/1/fc')).toEqual(offlineTensor('stack/1/fc'));
  await expect(panel).toContainText('fc · 实测输出');
  await expect(panel.locator('.tensor-activation .tensor-stats')).toContainText('[1, 8]');
  const liveLayer = await panel.locator('.tensor-activation .tensor-stats').innerText();
  expect(inferences).toEqual([['stack/2/act', 'out'], ['stack/1/fc', 'out']]);

  // The snapshot, still open, shows the same chosen instance.
  await source.selectOption('offline');
  await expect(instance).toHaveValue('1');
  await expect(panel.locator('.tensor-activation .tensor-stats')).toHaveText(liveLayer, { useInnerText: true });
  // Another instance needs no run offline: the snapshot holds every expanded layer.
  await instance.fill('2');
  await expect(panel).toContainText('观测展开后的层 stack/2/fc。');
  const cell = panel.getByRole('region', { name: '实测张量切片', exact: true }).locator('td').first();
  await expect(cell).toHaveAttribute('title', `[0, 0] = ${offlineTensor('stack/2/fc').slice.values[0][0]}`);
  expect(inferences).toHaveLength(2);
  expect(errors).toEqual([]);
});

test('shows one kernel window per convolution and pooling layer, with its measured output value', async ({ page }) => {
  const errors: string[] = []; page.on('pageerror', e => errors.push(e.message));
  await page.goto('/');
  await expect(page.locator('.validation-badge')).toContainText('形状校验通过');
  await expect(page.locator('.statusbar')).toContainText('PyTorch');
  // Paused, so only the toggle can change the frame.
  await page.getByRole('button', { name: '暂停数据流', exact: true }).click();
  const checksum = () => page.locator('canvas').evaluate(canvas => new Promise<number>(resolve => requestAnimationFrame(() => {
    const c = canvas as HTMLCanvasElement, gl = c.getContext('webgl2')!, buffer = new Uint8Array(c.width * c.height * 4);
    gl.readPixels(0, 0, c.width, c.height, gl.RGBA, gl.UNSIGNED_BYTE, buffer);
    let sum = 0; for (let i = 0; i < buffer.length; i += 4) sum = (sum + buffer[i] * (1 + i % 31) + buffer[i + 2]) >>> 0;
    resolve(sum);
  })));
  const shown = await checksum();
  await page.getByRole('button', { name: '隐藏卷积窗口', exact: true }).click();
  await expect.poll(checksum).not.toBe(shown);
  const hidden = await checksum();
  await page.getByRole('button', { name: '显示卷积窗口', exact: true }).click();
  await expect.poll(checksum).not.toBe(hidden);
  // With every layer measured, an output element inside the painted window shows its value.
  await expect(page.locator('.kernel-value')).toHaveCount(0);
  await page.getByRole('button', { name: '张量观测', exact: true }).click();
  await page.getByRole('button', { name: '观测全部层，画到三维视图', exact: true }).click();
  await expect(page.locator('.scene-observation')).toContainText('实测值 · 单样本');
  await expect(page.locator('.kernel-value').first()).toHaveText(/^= -?\d/);
  expect(errors).toEqual([]);
});
test('trains a mixture-of-experts layer, balancing its gate only when asked, and measures it', async ({ page }) => {
  const errors: string[] = []; page.on('pageerror', e => errors.push(e.message));
  const layer = (id: string, op: string, params: Layer['params']): Layer => ({ id, name: id, op: op as Layer['op'], params, position: { x: 0, y: 0 } });
  // `layer_1` is the layer the workbench selects on load.
  const graph: Graph = {
    version: 1, name: 'MoE fixture',
    nodes: [layer('layer_0', 'Input', { shape: [1, 6, 16] }), layer('layer_1', 'MoE', { experts: 4, top_k: 2, shared_experts: 1, expert_dim: 8, score_func: 'sigmoid', route_scale: 1, groups: 1, topk_groups: 1, gate_bias: 1 }), layer('layer_2', 'Flatten', {}), layer('layer_3', 'Linear', { out_features: 3 }), layer('layer_4', 'Output', {})],
    edges: [0, 1, 2, 3].map(i => ({ id: `e${i}`, source: `layer_${i}`, target: `layer_${i + 1}` }))
  };
  await page.addInitScript(value => localStorage.setItem('tensorlab-project', JSON.stringify(value)), graph);
  let sent: Record<string, unknown> | undefined;
  page.on('websocket', socket => socket.on('framesent', frame => { const message = JSON.parse(String(frame.payload)); if (message.config) sent = message.config; }));
  await page.goto('/');
  await expect(page.locator('.statusbar')).toContainText('PyTorch');
  await expect(page.locator('.validation-badge')).toContainText('形状校验通过');
  // The layer has a gate bias, so the training dialog offers the balancing switch, off by default.
  await page.getByRole('button', { name: '训练', exact: true }).click();
  const balancing = page.getByLabel('MoE 负载均衡：每步按专家负载调整门控偏置（无辅助损失）');
  await expect(balancing).not.toBeChecked();
  await balancing.check();
  await page.getByLabel('训练轮次', { exact: true }).fill('2');
  await page.getByLabel('合成样本数', { exact: true }).fill('64');
  await page.locator('label.field-label').filter({ hasText: '计算设备' }).locator('select').selectOption('cpu');
  await page.getByRole('button', { name: '开始真实训练', exact: true }).click();
  await expect(page.locator('.train-status')).toContainText('训练完成', { timeout: 90000 });
  expect(sent?.moeBalancing).toBe(true);
  await page.getByRole('button', { name: '张量观测', exact: true }).click();
  const panel = page.getByRole('region', { name: 'Tensor Inspector', exact: true });
  let response = page.waitForResponse('**/api/infer');
  await panel.getByRole('button', { name: '运行单样本推理', exact: true }).click();
  const report = await (await response).json() as InferenceReport;
  expect(report.weights).toBe('trained');
  expect(report.tensors.find(tensor => tensor.nodeId === 'layer_1')!.shape).toEqual([1, 6, 16]);
  // The routing of the trained layer: every position keeps two of the four experts.
  const routing = report.routings!.find(entry => entry.nodeId === 'layer_1')!;
  expect([routing.positions, routing.experts, routing.topK, routing.sharedExperts]).toEqual([6, 4, 2, 1]);
  expect(routing.load.reduce((sum, count) => sum + count, 0)).toBe(12);
  const view = panel.getByRole('region', { name: 'MoE 路由', exact: true });
  await expect(view.locator('.moe-load-bar')).toHaveCount(4);
  await expect(view).toContainText('窗口内每个位置选中的专家');
  // Moving the window is part of the run, so it needs a new one.
  await panel.getByLabel('路由位置起点').fill('2');
  await expect(panel.locator('.tensor-stale')).toContainText('旧快照已失效');
  response = page.waitForResponse('**/api/infer');
  await panel.getByRole('button', { name: '运行单样本推理', exact: true }).click();
  const moved = (await (await response).json() as InferenceReport).routings![0];
  expect([moved.positionStart, moved.weights.rowStart, moved.chosen.length]).toEqual([2, 2, 4]);
  // Every layer at once carries the routing too, which shades the expert bars in the 3D view.
  response = page.waitForResponse('**/api/infer');
  await panel.getByRole('button', { name: '观测全部层，画到三维视图', exact: true }).click();
  expect((await (await response).json() as InferenceReport).routings!.map(entry => entry.nodeId)).toEqual(['layer_1']);
  await expect(page.locator('.scene-observation')).toContainText('实测值 · 单样本');
  await page.getByRole('button', { name: '导出代码', exact: true }).click();
  await expect(page.locator('.code-preview, pre').first()).toContainText('TensorLabMoE(dim=16');
  expect(errors).toEqual([]);
});
