import { expect, test, type Page } from '@playwright/test';
import { mkdirSync, readFileSync } from 'node:fs';
import { execFileSync } from 'node:child_process';

async function exportedSvg(page: Page) {
  await page.getByRole('button', { name: '导出图像', exact: true }).click();
  const download = page.waitForEvent('download');
  await page.getByRole('button', { name: 'SVG 矢量图', exact: true }).click();
  const stream = await (await download).createReadStream(), chunks: Buffer[] = [];
  for await (const chunk of stream!) chunks.push(Buffer.from(chunk));
  return Buffer.concat(chunks).toString('utf8');
}

test('MQA/GQA parameters, shared KV visualization and real exported branches', async ({ page, request }) => {
  const errors: string[] = []; page.on('pageerror', e => errors.push(e.message));
  await page.goto('/'); await page.getByLabel('模型模板').selectOption('mqa');
  await page.getByRole('button', { name: '拓扑图', exact: true }).click();
  await page.locator('.react-flow__node[data-id="layer_1"]').click();
  await expect(page.getByLabel('attention_type', { exact: true })).toHaveValue('multi_query');
  await expect(page.getByLabel('kv_heads', { exact: true })).toHaveValue('1');
  await expect(page.getByLabel('kv_heads', { exact: true })).toBeDisabled();
  await page.getByLabel('num_heads', { exact: true }).fill('8');
  await page.getByLabel('attention_type', { exact: true }).selectOption('grouped_query');
  await page.getByLabel('kv_heads', { exact: true }).fill('3');
  await expect(page.locator('.validation-badge')).toContainText('结构错误');
  await page.getByLabel('kv_heads', { exact: true }).fill('2');
  await page.getByLabel('branches', { exact: true }).fill('2');
  await expect(page.locator('.validation-badge')).toContainText('形状校验通过');
  const graph = await page.evaluate(() => JSON.parse(localStorage.getItem('tensorlab-project')!));
  expect((await request.post('/api/analyze', { data: graph })).ok()).toBe(true);
  await page.getByRole('button', { name: '三维视图', exact: true }).click();
  await page.getByRole('button', { name: '聚焦选中层', exact: true }).click();
  const svg = await exportedSvg(page);
  // Each branch's two KV groups export as one stack: the front group keeps its
  // Q/K/V cubes, the group behind it shows as a stepped layer.
  expect(svg).toContain('data-part="b0:k0"'); expect(svg).toContain('data-part="b1:k0"'); expect(svg).toContain('data-part="stack-layer"'); expect(svg).toContain('Branch Mean');
  expect(svg).not.toContain('shared-K'); expect(svg).toContain('data-sides="6"');
  await page.getByRole('button', { name: 'B2 · H8', exact: true }).click();
  mkdirSync('artifacts', { recursive: true });
  await page.screenshot({ path: 'artifacts/gqa-branches.png' });
  await page.getByRole('button', { name: '导出代码', exact: true }).click();
  const download = page.waitForEvent('download'); await page.getByRole('button', { name: '下载 .py' }).click();
  await (await download).saveAs('artifacts/gqa-branches.py');
  const code = readFileSync('artifacts/gqa-branches.py', 'utf8');
  expect(code).toContain('num_heads=8, kv_heads=2'); expect(code).toContain('attention_type="grouped_query", branches=2');
  expect(execFileSync('.venv/Scripts/python.exe', ['artifacts/gqa-branches.py'], { encoding: 'utf8' })).toContain('Output shape: (1, 10)');
  await page.getByRole('button', { name: '关闭对话框', exact: true }).click();
  // Type changes update locked KV heads and make true independent branches.
  await page.getByLabel('attention_type', { exact: true }).selectOption('self');
  await expect(page.getByLabel('kv_heads', { exact: true })).toHaveValue('8');
  await page.getByLabel('num_heads', { exact: true }).fill('4');
  await expect(page.getByLabel('kv_heads', { exact: true })).toHaveValue('4');
  await page.getByLabel('attention_type', { exact: true }).selectOption('multi_branch');
  await expect(page.getByLabel('branches', { exact: true })).toHaveValue('2');
  await expect(page.locator('.validation-badge')).toContainText('形状校验通过');
  expect(errors).toEqual([]);
});

test('Cross-Attention ports, roles, exports and measured dual-input training', async ({ page }) => {
  const errors: string[] = []; page.on('pageerror', e => errors.push(e.message));
  await page.goto('/'); await page.getByLabel('模型模板').selectOption('cross_attention');
  await page.getByRole('button', { name: '拓扑图', exact: true }).click();
  await page.locator('.react-flow__node[data-id="layer_1"]').click();
  await expect(page.locator('.validation-badge')).toContainText('形状校验通过');
  await expect(page.getByLabel('输入角色 Query 输入', { exact: true })).toHaveValue('query');
  await expect(page.getByLabel('输入角色 Context 输入', { exact: true })).toHaveValue('context');
  await page.getByLabel('输入角色 Query 输入', { exact: true }).selectOption('context');
  await expect(page.getByLabel('输入角色 Context 输入', { exact: true })).toHaveValue('query');
  await expect(page.locator('.tensor-box')).toContainText('1 × 12 × 64');
  await page.getByLabel('输入角色 Query 输入', { exact: true }).selectOption('query');
  await page.getByRole('button', { name: '取消输入对象 Context 输入', exact: true }).click();
  await expect(page.locator('.validation-badge')).toContainText('结构错误');
  await page.getByLabel('选择输入对象', { exact: true }).selectOption('context');
  await expect(page.getByLabel('新增输入角色', { exact: true })).toHaveValue('context');
  await page.getByRole('button', { name: '添加输入连接', exact: true }).click();
  await expect(page.locator('.validation-badge')).toContainText('形状校验通过');
  await page.getByLabel('kv_heads', { exact: true }).fill('1');
  await page.getByRole('button', { name: '三维视图', exact: true }).click();
  await page.getByRole('button', { name: '聚焦选中层', exact: true }).click();
  const svg = await exportedSvg(page);
  expect(svg).toContain('context-input'); expect(svg).toContain('>Query</text>'); expect(svg).toContain('>Context</text>');
  mkdirSync('artifacts', { recursive: true });
  await page.screenshot({ path: 'artifacts/cross-attention.png' });
  await page.getByRole('button', { name: '导出代码', exact: true }).click();
  const download = page.waitForEvent('download'); await page.getByRole('button', { name: '下载 .py' }).click();
  await (await download).saveAs('artifacts/cross-attention.py');
  expect(execFileSync('.venv/Scripts/python.exe', ['artifacts/cross-attention.py'], { encoding: 'utf8' })).toContain('Output shape: (1, 10)');
  await page.getByRole('button', { name: '关闭对话框', exact: true }).click();
  await page.getByRole('button', { name: '训练', exact: true }).click();
  await page.getByLabel('训练轮次', { exact: true }).fill('2'); await page.getByLabel('合成样本数', { exact: true }).fill('64');
  await page.getByRole('button', { name: '开始真实训练', exact: true }).click();
  await expect(page.locator('.train-status')).toContainText('训练完成', { timeout: 60000 });
  const metricsEvent = page.waitForEvent('download'); await page.getByRole('button', { name: '导出训练指标', exact: true }).click();
  await (await metricsEvent).saveAs('artifacts/cross-attention-metrics.json');
  const metrics = JSON.parse(readFileSync('artifacts/cross-attention-metrics.json', 'utf8'));
  expect(metrics).toHaveLength(2); expect(metrics[0].layerGradients.layer_1).toBeGreaterThan(0);
  expect(errors).toEqual([]);
});
