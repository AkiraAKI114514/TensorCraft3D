import { test, expect } from '@playwright/test';
import type { Graph } from '../src/types';

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
