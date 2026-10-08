import { test, expect } from '@playwright/test';

test.beforeEach(async ({ page }) => {
  await page.route('**/api/health', route => route.fulfill({ json: { torch: false, cuda: false } }));
});

test('switches languages live, preserves graph/history/metrics, and remembers the preference', async ({ page }) => {
  await page.goto('/');
  await expect(page.locator('html')).toHaveAttribute('lang', 'zh-CN');
  await page.getByLabel('模型模板').selectOption('mlp');
  await page.getByRole('button', { name: '拓扑图', exact: true }).click();
  await page.locator('.react-flow__node[data-id="layer_1"]').click();
  const features = page.getByLabel('out_features', { exact: true });
  await features.fill('12');
  await features.blur();
  await page.getByRole('button', { name: '演示', exact: true }).click();
  await page.getByRole('button', { name: '层属性', exact: true }).click();
  const saved = await page.evaluate(() => localStorage.getItem('tensorlab-project'));
  const nodeElements = await page.locator('.react-flow__node').count();
  await page.evaluate(() => { (window as unknown as { languageNode?: Element | null }).languageNode = document.querySelector('.react-flow__node[data-id="layer_1"]'); });

  await page.getByLabel('界面语言').selectOption('en');
  await expect(page.locator('html')).toHaveAttribute('lang', 'en');
  await expect(page.getByRole('button', { name: 'Export code', exact: true })).toBeVisible();
  await expect(page.getByRole('button', { name: 'Undo', exact: true })).toBeEnabled();
  await expect(features).toHaveValue('12');
  await expect(page.locator('.train-status')).toHaveText('Demo complete');
  expect(await page.evaluate(() => localStorage.getItem('tensorlab-project'))).toBe(saved);
  expect(await page.locator('.react-flow__node').count()).toBe(nodeElements);
  expect(await page.evaluate(() => (window as unknown as { languageNode?: Element | null }).languageNode === document.querySelector('.react-flow__node[data-id="layer_1"]'))).toBe(true);
  await page.getByRole('button', { name: 'Diagnostics', exact: false }).click();
  await expect(page.locator('.diagnostic-item').first()).not.toContainText(/[㐀-鿿]/);
  await page.getByRole('button', { name: 'Properties', exact: true }).click();
  await page.getByRole('button', { name: 'Undo', exact: true }).click();
  await expect(features).toHaveValue('64');
  await page.getByRole('button', { name: 'Redo', exact: true }).click();
  await expect(features).toHaveValue('12');

  await page.getByRole('button', { name: 'Export code', exact: true }).click();
  await expect(page.getByRole('dialog', { name: 'Export code', exact: true })).toBeVisible();
  await expect(page.getByRole('heading', { name: 'Export model code' })).toBeVisible();
  await page.getByLabel('Interface language').selectOption('zh');
  await expect(page.getByRole('dialog', { name: '导出代码', exact: true })).toBeVisible();
  await page.getByRole('button', { name: '关闭对话框', exact: true }).click();
  await expect(features).toHaveValue('12');
  await page.getByLabel('界面语言').selectOption('en');
  await page.reload();
  await expect(page.getByLabel('Interface language')).toHaveValue('en');
  await expect(page.locator('html')).toHaveAttribute('lang', 'en');
  expect(await page.evaluate(() => localStorage.getItem('tensorlab-project'))).toBe(saved);
});

test('keeps the switch and English controls within a narrow viewport', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto('/');
  await page.getByLabel('界面语言').selectOption('en');
  await expect(page.getByLabel('Interface language')).toBeVisible();
  const switchBounds = await page.getByLabel('Interface language').boundingBox();
  expect(switchBounds).not.toBeNull();
  expect(switchBounds!.x + switchBounds!.width).toBeLessThanOrEqual(390);
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  await page.getByRole('button', { name: 'Train', exact: true }).click();
  await expect(page.getByRole('dialog', { name: 'Training settings' })).toBeVisible();
  await expect(page.getByText('CUDA environment & setup', { exact: true })).toBeVisible();
  await page.getByRole('button', { name: 'Close dialog', exact: true }).click();
  await page.getByRole('button', { name: '3D view', exact: true }).click();
  await expect(page.locator('canvas').first()).toBeVisible();
  if (process.env.CLAUDE_JOB_DIR) await page.screenshot({ path: `${process.env.CLAUDE_JOB_DIR.replaceAll('\\', '/')}/tmp/language-en-mobile.png`, fullPage: true });
  for (const width of [320, 700, 820, 1024]) {
    await page.setViewportSize({ width, height: 844 });
    const bounds = await page.getByLabel('Interface language').boundingBox();
    expect(bounds, `Language selector at ${width}px`).not.toBeNull();
    expect(bounds!.x + bounds!.width, `Language selector at ${width}px`).toBeLessThanOrEqual(width);
    const tabs = await page.locator('.inspector-tabs').evaluate(element => [...element.querySelectorAll('button')].map(button => button.getBoundingClientRect().right));
    expect(Math.max(...tabs), `Inspector tabs at ${width}px`).toBeLessThanOrEqual(width);
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), `Page overflow at ${width}px`).toBe(true);
  }
  await page.getByLabel('Interface language').selectOption('zh');
  await expect(page.getByRole('button', { name: '三维视图', exact: true })).toBeVisible();
});

test('changing language does not repeat environment or inference requests', async ({ page }) => {
  let environmentRequests = 0, inferenceRequests = 0;
  await page.route('**/api/environment', route => { environmentRequests++; return route.fulfill({ json: { status: 'unavailable', torch: { installed: false, version: null, cudaBuild: null }, cuda: { available: false, deviceCount: 0, devices: [], error: null }, gpu: { present: false, driver: null, devices: [] }, interpreter: 'test', environment: {}, diagnosis: [], setup: { pytorch: 'https://pytorch.org', nvidia: 'https://nvidia.com', note: '' } } }); });
  await page.route('**/api/infer', route => { inferenceRequests++; return route.fulfill({ status: 400, json: { error: 'test' } }); });
  await page.goto('/');
  await page.getByRole('button', { name: '训练', exact: true }).click();
  await page.getByText('CUDA 环境与配置', { exact: true }).click();
  await expect.poll(() => environmentRequests).toBe(1);
  await page.getByLabel('界面语言').selectOption('en');
  await expect(page.getByText('CUDA environment & setup', { exact: true })).toBeVisible();
  await page.getByLabel('Interface language').selectOption('zh');
  expect(environmentRequests).toBe(1);
  expect(inferenceRequests).toBe(0);
});

test('switches language during training without reconnecting or resetting metrics', async ({ page }) => {
  await page.route('**/api/health', route => route.fulfill({ json: { torch: true, cuda: false } }));
  await page.addInitScript(() => {
    class TrainingSocket {
      static OPEN = 1;
      readyState = 1;
      onopen: (() => void) | null = null;
      onmessage: ((event: { data: string }) => void) | null = null;
      onclose: (() => void) | null = null;
      constructor() {
        (window as unknown as { trainingSocket?: TrainingSocket }).trainingSocket = this;
        setTimeout(() => this.onopen?.(), 0);
      }
      send() {}
      close() { this.onclose?.(); }
    }
    Object.assign(window, { WebSocket: TrainingSocket });
  });
  await page.goto('/');
  await page.getByRole('button', { name: '训练', exact: true }).click();
  await page.getByRole('button', { name: '开始真实训练', exact: true }).click();
  await page.evaluate(() => {
    const socket = (window as unknown as { trainingSocket: { onmessage: (event: { data: string }) => void } }).trainingSocket;
    socket.onmessage({ data: JSON.stringify({ type: 'metric', device: 'CPU', metric: { epoch: 1, trainLoss: 0.5, valLoss: 0.6, accuracy: 0.7, gradNorm: 0.8, source: 'training' } }) });
    (window as unknown as { originalTrainingSocket?: unknown }).originalTrainingSocket = socket;
  });
  await expect(page.locator('.train-status')).toHaveText('训练中 · CPU');
  await page.getByLabel('界面语言').selectOption('en');
  await expect(page.locator('.train-status')).toHaveText('Training · CPU');
  await expect(page.getByRole('button', { name: 'Stop', exact: true })).toBeVisible();
  await expect(page.locator('.metric-grid')).toContainText('0.5000');
  expect(await page.evaluate(() => (window as unknown as { trainingSocket?: unknown }).trainingSocket === (window as unknown as { originalTrainingSocket?: unknown }).originalTrainingSocket)).toBe(true);
  await page.getByLabel('Interface language').selectOption('zh');
  await expect(page.locator('.train-status')).toHaveText('训练中 · CPU');
  await page.evaluate(() => (window as unknown as { trainingSocket: { onmessage: (event: { data: string }) => void } }).trainingSocket.onmessage({ data: JSON.stringify({ type: 'done', reason: 'completed' }) }));
  await expect(page.locator('.train-status')).toHaveText('训练完成');
  await page.getByLabel('界面语言').selectOption('en');
  await expect(page.locator('.train-status')).toHaveText('Training complete');
  await expect(page.locator('.toast')).toContainText('Training ended');
});

test('keeps the recovery screen bilingual without losing its saved project', async ({ page }) => {
  await page.addInitScript(() => {
    const toFixed = Number.prototype.toFixed;
    Number.prototype.toFixed = function (digits) { if (digits === 2) throw new Error('test rendering failure'); return toFixed.call(this, digits); };
  });
  await page.goto('/');
  await expect(page.getByRole('alert')).toContainText('工作台发生错误');
  await page.getByLabel('界面语言').selectOption('en');
  await expect(page.getByRole('heading', { name: 'Workbench error' })).toBeVisible();
  await expect(page.getByRole('button', { name: 'Download saved project' })).toBeEnabled();
  await page.getByRole('button', { name: 'Download saved project' }).click();
  await expect(page.getByRole('status')).toHaveText('No saved project is available to back up.');
  await page.getByLabel('Interface language').selectOption('zh');
  await expect(page.getByRole('status')).toHaveText('没有可备份的已保存项目。');
});

test('translates the import dialog live while preserving source and deferred parsing', async ({ page }) => {
  let imports = 0;
  await page.route('**/api/import/pytorch', route => { imports++; return route.fulfill({ status: 400, json: { detail: 'test' } }); });
  await page.goto('/');
  await page.getByRole('button', { name: '导入 PyTorch 代码', exact: true }).click();
  await page.locator('textarea.import-source').fill('from torch import nn\nmodel = nn.Linear(16, 4)');
  await page.getByLabel('界面语言').selectOption('en');
  await expect(page.getByRole('dialog', { name: 'Import PyTorch code' })).toBeVisible();
  await expect(page.getByRole('heading', { name: 'Import code · automatic modeling' })).toBeVisible();
  await expect(page.locator('textarea.import-source')).toHaveValue('from torch import nn\nmodel = nn.Linear(16, 4)');
  await page.getByText('Supported scope and import method', { exact: true }).click();
  await expect(page.getByText(/Only the structure is statically parsed/)).toBeVisible();
  await page.getByLabel('Interface language').selectOption('zh');
  await expect(page.getByRole('dialog', { name: '导入 PyTorch 代码' })).toBeVisible();
  expect(imports).toBe(0);
});

test('language preference storage failures do not crash the workbench', async ({ page }) => {
  await page.addInitScript(() => {
    const read = Storage.prototype.getItem, write = Storage.prototype.setItem;
    Storage.prototype.getItem = function (key) { if (key === 'tensorlab-language') throw new Error('blocked language preference'); return read.call(this, key); };
    Storage.prototype.setItem = function (key, value) { if (key === 'tensorlab-language') throw new Error('blocked language preference'); write.call(this, key, value); };
  });
  await page.goto('/');
  await page.getByLabel('界面语言').selectOption('en');
  await expect(page.getByRole('button', { name: 'Export code', exact: true })).toBeVisible();
  await expect(page.locator('html')).toHaveAttribute('lang', 'en');
  await page.reload();
  await expect(page.getByLabel('界面语言')).toHaveValue('zh');
});
