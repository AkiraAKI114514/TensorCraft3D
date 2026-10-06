import { test, expect } from '@playwright/test';
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { execFileSync } from 'node:child_process';

test('Coaxial attention slices, exports and measured training', async ({ page }) => {
  const errors: string[] = []; page.on('pageerror', e => errors.push(e.message));
  await page.goto('/'); await page.getByLabel('模型模板').selectOption('transformer');
  await expect(page.getByRole('heading', { name: 'Transformer Encoder', exact: true })).toBeVisible();
  await expect(page.locator('.validation-badge')).toContainText('形状校验通过');
  const pixels = () => page.locator('canvas').evaluate(canvas => new Promise<{ colored: number; checksum: number }>(resolve => requestAnimationFrame(() => {
    const c = canvas as HTMLCanvasElement, gl = c.getContext('webgl2')!;
    const buffer = new Uint8Array(c.width * c.height * 4); gl.readPixels(0, 0, c.width, c.height, gl.RGBA, gl.UNSIGNED_BYTE, buffer);
    let colored = 0, checksum = 0; for (let i = 0; i < buffer.length; i += 4) { if (Math.max(buffer[i], buffer[i+1], buffer[i+2]) - Math.min(buffer[i], buffer[i+1], buffer[i+2]) > 30) colored++; checksum = (checksum + buffer[i] * (1 + i % 31) + buffer[i+1]) >>> 0; }
    resolve({ colored, checksum });
  })));
  await expect.poll(async () => (await pixels()).colored).toBeGreaterThan(1500);
  const a = await pixels(); await page.waitForTimeout(350); expect((await pixels()).checksum).not.toBe(a.checksum);
  mkdirSync('artifacts', { recursive: true });
  await page.screenshot({ path: 'artifacts/transformer-desktop.png' });
  await page.getByRole('button', { name: '拓扑图', exact: true }).click();
  await page.locator('.flow-layer').filter({ hasText: '多头注意力层' }).first().click();
  await page.getByRole('button', { name: '三维视图', exact: true }).click();
  await expect(page.getByLabel('num_heads', { exact: true })).toHaveValue('1');
  await expect(page.locator('.head-selector button')).toHaveCount(1);
  await page.getByLabel('num_heads', { exact: true }).fill('4');
  await expect(page.locator('.head-selector button')).toHaveCount(4);
  await page.getByRole('button', { name: '聚焦选中层', exact: true }).click();
  await page.waitForTimeout(100); await expect.poll(async () => (await pixels()).colored).toBeGreaterThan(1500);
  await page.screenshot({ path: 'artifacts/transformer-coaxial.png' });
  await page.getByRole('button', { name: 'H2', exact: true }).click(); await expect(page.getByRole('button', { name: 'H2', exact: true })).toHaveAttribute('aria-pressed', 'true');
  await page.waitForTimeout(150); await expect.poll(async () => (await pixels()).colored).toBeGreaterThan(1500);
  await page.screenshot({ path: 'artifacts/transformer-head-focus.png' });
  await page.getByLabel('num_heads', { exact: true }).fill('3'); await expect(page.locator('.validation-badge')).toContainText('结构错误');
  for (const count of [1, 8, 16]) {
    await page.getByLabel('num_heads', { exact: true }).fill(String(count));
    await expect(page.locator('.head-selector button')).toHaveCount(count);
    await expect(page.locator('.validation-badge')).toContainText('形状校验通过');
    await page.getByRole('button', { name: '聚焦选中层', exact: true }).click();
    await expect.poll(async () => (await pixels()).colored).toBeGreaterThan(1000);
    await page.screenshot({ path: `artifacts/transformer-${count}-heads.png` });
    await page.getByRole('button', { name: '导出图像', exact: true }).click();
    const svgEvent = page.waitForEvent('download'); await page.getByRole('button', { name: 'SVG 矢量图', exact: true }).click();
    const svgDownload = await svgEvent, svgFile = `artifacts/transformer-${count}-heads.svg`; await svgDownload.saveAs(svgFile);
    const projected = await page.evaluate(svg => {
      const document = new DOMParser().parseFromString(svg, 'image/svg+xml');
      const root = document.documentElement, width = Number(root.getAttribute('width')), height = Number(root.getAttribute('height'));
      return { cards: document.querySelectorAll('g[data-node-id="layer_1"] polygon[data-part="stack-card"]').length, texts: Array.from(document.querySelectorAll('g[data-label-id^="layer_1:"] text')).map(t => ({ text: t.textContent || '', x: Number(t.getAttribute('x')), y: Number(t.getAttribute('y')), width, height })) };
    }, readFileSync(svgFile, 'utf8'));
    const visible = projected.texts.filter(p => p.x >= 0 && p.x <= p.width && p.y >= 0 && p.y <= p.height);
    // Heads that split one input are exported as a single stepped stack: the
    // front face names the head range and the back layers carry the count.
    expect(visible.filter(p => /^H\d+/.test(p.text)).map(p => p.text)).toEqual([count === 1 ? 'H1' : `H1–H${count}`]);
    expect(visible.filter(p => /^×\d+$/.test(p.text)).map(p => p.text)).toEqual(count === 1 ? [] : [`×${count}`]);
    expect(projected.cards > 0).toBe(count > 1);
  }
  await page.getByLabel('num_heads', { exact: true }).fill('4');
  await page.getByRole('button', { name: '展开层间距', exact: true }).click();
  await page.getByRole('button', { name: '聚焦选中层', exact: true }).click();
  await page.getByRole('button', { name: '暂停数据流', exact: true }).click(); await page.waitForTimeout(300);
  const paused = await pixels(); await page.waitForTimeout(250); expect(Math.abs((await pixels()).checksum - paused.checksum)).toBeLessThan(paused.checksum * 0.0001);
  await page.screenshot({ path: 'artifacts/transformer-selected.png' });
  await page.getByRole('button', { name: '反向', exact: true }).click();
  expect((await pixels()).checksum).not.toBe(paused.checksum);
  await page.getByRole('button', { name: '播放数据流', exact: true }).click(); const backward = await pixels(); await page.waitForTimeout(250); expect((await pixels()).checksum).not.toBe(backward.checksum);
  await page.getByRole('button', { name: '暂停数据流', exact: true }).click(); await page.getByRole('button', { name: '前向', exact: true }).click();
  await page.getByRole('button', { name: '导出代码', exact: true }).click();
  const pythonEvent = page.waitForEvent('download'); await page.getByRole('button', { name: '下载 .py' }).click(); await (await pythonEvent).saveAs('artifacts/transformer-model.py');
  const pythonOutput = execFileSync('.venv/Scripts/python.exe', ['artifacts/transformer-model.py'], { encoding: 'utf8' }); expect(pythonOutput).toContain('Output shape: (1, 10)'); expect(pythonOutput).toContain('TensorLabTransformer');
  await page.getByRole('button', { name: '关闭对话框', exact: true }).click();
  await page.getByRole('button', { name: '导出图像', exact: true }).click(); const svgEvent = page.waitForEvent('download'); await page.getByRole('button', { name: 'SVG 矢量图', exact: true }).click(); await (await svgEvent).saveAs('artifacts/transformer.svg');
  const svg = readFileSync('artifacts/transformer.svg', 'utf8'); expect(svg).toContain('FFN'); expect(svg).toContain('Concat · Wᵒ'); expect(svg.match(/<polygon/g)!.length).toBeGreaterThan(400);
  await page.getByRole('button', { name: '导出图像', exact: true }).click(); const imageEvent = page.waitForEvent('download'); await page.getByRole('button', { name: '导出 PNG', exact: true }).click(); await (await imageEvent).saveAs('artifacts/transformer-4k.png'); expect(readFileSync('artifacts/transformer-4k.png').readUInt32BE(16)).toBe(3840);
  await page.getByRole('button', { name: '训练', exact: true }).click(); await page.getByLabel('训练轮次', { exact: true }).fill('2'); await page.getByLabel('合成样本数', { exact: true }).fill('64'); await page.getByLabel('学习率', { exact: true }).fill('0.001');
  await page.getByRole('button', { name: '开始真实训练', exact: true }).click(); await expect(page.locator('.train-status')).toContainText('训练完成', { timeout: 60000 }); await expect(page.locator('.monitor .source-badge')).toContainText('真实训练');
  const metricEvent = page.waitForEvent('download'); await page.getByRole('button', { name: '导出训练指标', exact: true }).click(); await (await metricEvent).saveAs('artifacts/transformer-metrics.json');
  const metrics = JSON.parse(readFileSync('artifacts/transformer-metrics.json', 'utf8')); expect(metrics).toHaveLength(2); expect(metrics[0].layerGradients.layer_1).toBeGreaterThan(0);
  await page.setViewportSize({ width: 390, height: 844 }); await page.getByRole('button', { name: '聚焦选中层', exact: true }).click(); await page.waitForTimeout(300); expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true); expect((await pixels()).colored).toBeGreaterThan(300);
  await page.screenshot({ path: 'artifacts/transformer-mobile.png', fullPage: true });
  await page.getByRole('button', { name: 'H2', exact: true }).click(); await page.waitForTimeout(100); expect((await pixels()).colored).toBeGreaterThan(1500);
  await page.screenshot({ path: 'artifacts/transformer-mobile-head.png', fullPage: true });
  expect(errors).toEqual([]); writeFileSync('artifacts/attention-verification.json', JSON.stringify({ desktop: a, mobile: await pixels(), runtimeErrors: errors, pythonOutput, measuredEpochs: metrics.length }, null, 2));
});
