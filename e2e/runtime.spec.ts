import { test, expect } from '@playwright/test';

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
