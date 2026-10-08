import { test, expect, type Page } from '@playwright/test';
import { readFileSync } from 'node:fs';

type SvgWindow = Window & { focusSvg?: Promise<string> };

test.beforeEach(async ({ page }) => {
  // Focus checks need the exported coordinates, not Edge's native download UI.
  // Real SVG/PNG downloads are covered separately in imageExport.spec.ts.
  await page.addInitScript(() => {
    const createObjectURL = URL.createObjectURL, click = HTMLAnchorElement.prototype.click;
    const blobs = new Map<string, Blob>();
    URL.createObjectURL = function(blob) {
      const url = createObjectURL.call(this, blob);
      if (blob instanceof Blob) blobs.set(url, blob);
      return url;
    };
    HTMLAnchorElement.prototype.click = function() {
      const blob = blobs.get(this.href);
      if (!this.download.endsWith('.svg') || !blob) return click.call(this);
      (window as SvgWindow).focusSvg = blob.text();
    };
  });
});

async function exportSvg(page: Page) {
  await page.evaluate(() => { delete (window as SvgWindow).focusSvg; });
  await page.getByRole('button', { name: '导出图像', exact: true }).click();
  await page.getByLabel('图像布局').selectOption('current');
  await page.getByRole('button', { name: 'SVG 矢量图', exact: true }).click();
  await page.waitForFunction(() => !!(window as SvgWindow).focusSvg);
  return page.evaluate(() => (window as SvgWindow).focusSvg!);
}

async function anchor(page: Page, svg: string, selector: string) {
  return page.evaluate(({ svg, selector }) => {
    const document = new DOMParser().parseFromString(svg, 'image/svg+xml'), el = document.querySelector(selector)!;
    if (!el) throw new Error(`Missing exported module: ${selector}`);
    return { x: Number(el.getAttribute('data-center-x')), y: Number(el.getAttribute('data-center-y')), width: Number(document.documentElement.getAttribute('width')), height: Number(document.documentElement.getAttribute('height')) };
  }, { svg, selector });
}

async function clickAndCheckCenter(page: Page, selector: string, op?: string) {
  const before = await anchor(page, await exportSvg(page), selector), canvas = (await page.locator('canvas').boundingBox())!;
  await page.mouse.click(canvas.x + before.x / before.width * canvas.width, canvas.y + before.y / before.height * canvas.height);
  if (op) await expect(page.locator('.selected-layer h2')).toHaveText(op);
  const after = await anchor(page, await exportSvg(page), selector);
  expect(Math.abs(after.x - after.width / 2), `${selector} should be centered horizontally`).toBeLessThan(3);
  expect(Math.abs(after.y - after.height / 2), `${selector} should be centered vertically`).toBeLessThan(3);
}

const focusCases: [string, string[][]][] = [
  ['cnn', [['layer_0', 'Input'], ['layer_1', 'Conv2d'], ['layer_2', 'BatchNorm2d'], ['layer_3', 'ReLU'], ['layer_4', 'MaxPool2d'], ['layer_7', 'AdaptiveAvgPool2d'], ['layer_8', 'Flatten'], ['layer_9', 'Linear'], ['layer_10', 'Output']]],
  ['mlp', [['layer_3', 'Dropout'], ['layer_5', 'GELU']]],
  ['residual', [['layer_5', 'Add']]]
];

for (const [preset, entries] of focusCases) for (const [id, op] of entries) test(`centers ${preset} ${id} (${op}) when clicked`, async ({ page }) => {
  const errors: string[] = []; page.on('pageerror', e => errors.push(e.stack || e.message));
  await page.goto('/');
  await page.getByRole('button', { name: '暂停数据流', exact: true }).click();
  await page.getByLabel('模型模板').selectOption(preset);
  await page.getByRole('button', { name: '重置视角', exact: true }).click();
  await clickAndCheckCenter(page, `g[data-node-id="${id}"]`, op);
  expect(errors).toEqual([]);
});

for (const part of ['b0:q0', 'b0:k0', 'b0:v0', 'b0:q1', 'b0:q2', 'b0:q3', 'FFN', 'Concat · Wᵒ', 'Add1', 'Add2', 'LN1', 'LN2', 'scores-0', 'weighted-0', 'input', 'head-input', 'head-output']) test(`centers attention ${part} when clicked`, async ({ page }) => {
  const errors: string[] = []; page.on('pageerror', e => errors.push(e.stack || e.message));
  await page.goto('/');
  await page.getByRole('button', { name: '暂停数据流', exact: true }).click();
  await page.getByLabel('模型模板').selectOption('transformer');
  await page.getByRole('button', { name: '拓扑图', exact: true }).click();
  await page.locator('.react-flow__node[data-id="layer_1"]').click();
  await expect(page.getByLabel('num_heads', { exact: true })).toHaveValue('1');
  await page.getByLabel('attention_type', { exact: true }).selectOption('multi_query');
  await page.getByLabel('num_heads', { exact: true }).fill('4');
  await page.getByRole('button', { name: '三维视图', exact: true }).click();
  await page.getByRole('button', { name: '聚焦选中层', exact: true }).click();
  const svg = await exportSvg(page);
  expect(svg).toContain('data-sides="6"');
  for (const label of ['Q1', 'Q2', 'Q3', 'Q4', 'K1', 'V1']) expect(svg).toContain(`>${label}</text>`);
  await page.getByRole('button', { name: '聚焦选中层', exact: true }).click();
  await clickAndCheckCenter(page, `g[data-node-id="layer_1"] polygon[data-part="${part}"]`);
  expect(errors).toEqual([]);
});

test('centers an imported Concat when clicked', async ({ page }) => {
  const errors: string[] = []; page.on('pageerror', e => errors.push(e.stack || e.message));
  await page.goto('/');
  await page.getByRole('button', { name: '暂停数据流', exact: true }).click();
  // A saved project can also expose Concat as an ordinary selectable module.
  const project = { version: 1, name: 'Concat focus', nodes: [
    { id: 'input', name: 'input', op: 'Input', params: { shape: [1, 16] }, position: { x: 0, y: 100 } },
    { id: 'a', name: 'a', op: 'ReLU', params: {}, position: { x: 210, y: 100 } },
    { id: 'b', name: 'b', op: 'ReLU', params: {}, position: { x: 210, y: 250 } },
    { id: 'concat', name: 'concat', op: 'Concat', params: { dim: 1 }, position: { x: 420, y: 100 } },
    { id: 'output', name: 'output', op: 'Output', params: {}, position: { x: 630, y: 100 } }
  ], edges: [['input', 'a'], ['input', 'b'], ['a', 'concat'], ['b', 'concat'], ['concat', 'output']].map(([source, target], i) => ({ id: `e${i}`, source, target })) };
  await page.locator('input[type="file"][accept=".json"]').setInputFiles({ name: 'concat.json', mimeType: 'application/json', buffer: Buffer.from(JSON.stringify(project)) });
  await expect(page.locator('.validation-badge')).toContainText('形状校验通过');
  await clickAndCheckCenter(page, 'g[data-node-id="concat"]', 'Concat');
  expect(errors).toEqual([]);
});

test('Input/output objects and free connections report and recover from structural errors', async ({ page, request }, testInfo) => {
  await page.goto('/'); await page.getByLabel('模型模板').selectOption('mlp');
  await page.getByRole('button', { name: '拓扑图', exact: true }).click();
  await page.getByRole('button', { name: '添加输入对象', exact: true }).click();
  await expect(page.locator('.flow-layer')).toHaveCount(9);
  await expect(page.locator('.validation-badge')).toContainText('结构错误');
  await page.getByRole('button', { name: '删除选中对象', exact: true }).click();
  await expect(page.locator('.validation-badge')).toContainText('形状校验通过');
  await page.getByRole('button', { name: '添加输出对象', exact: true }).click();
  await expect(page.locator('.flow-layer')).toHaveCount(9);
  await expect(page.locator('.validation-badge')).toContainText('结构错误');
  await page.getByRole('button', { name: '删除选中对象', exact: true }).click();
  await expect(page.locator('.validation-badge')).toContainText('形状校验通过');
  await page.locator('.react-flow__node[data-id="layer_1"]').click();
  await page.getByLabel('选择输入对象').selectOption('layer_2');
  await page.getByRole('button', { name: '添加输入连接', exact: true }).click();
  await expect(page.locator('.toast')).toContainText('结构错误：计算图存在循环');
  await expect(page.locator('.validation-badge')).toContainText('结构错误');
  const cycle = await page.evaluate(() => JSON.parse(localStorage.getItem('tensorlab-project')!));
  expect((await request.post('/api/analyze', { data: cycle })).status()).toBe(422);
  await page.getByRole('button', { name: '导出代码', exact: true }).click();
  await expect(page.getByRole('button', { name: '下载 .py' })).toBeDisabled();
  await page.getByRole('button', { name: '关闭对话框', exact: true }).click();
  await page.getByRole('button', { name: '取消输入对象 ReLU_2', exact: true }).click();
  await expect(page.locator('.validation-badge')).toContainText('形状校验通过');
  await page.getByLabel('选择输出对象').selectOption('layer_3');
  await page.getByRole('button', { name: '添加输出连接', exact: true }).click();
  await expect(page.locator('.toast')).toContainText('结构错误');
  await expect(page.locator('.validation-badge')).toContainText('结构错误');
  await page.getByRole('button', { name: '取消输出对象 Dropout_3', exact: true }).click();
  await expect(page.locator('.validation-badge')).toContainText('形状校验通过');
  // Removing and restoring an existing connection is also supported.
  await page.getByRole('button', { name: '取消输入对象 特征输入', exact: true }).click();
  await expect(page.locator('.validation-badge')).toContainText('结构错误');
  await page.getByLabel('选择输入对象').selectOption('layer_0');
  await page.getByRole('button', { name: '添加输入连接', exact: true }).click();
  await expect(page.locator('.validation-badge')).toContainText('形状校验通过');
  const graph = await page.evaluate(() => JSON.parse(localStorage.getItem('tensorlab-project')!));
  expect((await request.post('/api/analyze', { data: graph })).ok()).toBe(true);
  const screenshot = testInfo.outputPath('free-connections.png');
  await page.screenshot({ path: screenshot });
  expect(readFileSync(screenshot).length).toBeGreaterThan(10000);
});
