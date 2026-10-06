import { test, expect } from '@playwright/test';
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';

test('Pink residual bypass, addition, motion and matching image exports', async ({ page }) => {
  const errors: string[] = []; page.on('pageerror', e => errors.push(e.message));
  await page.goto('/'); await page.getByLabel('模型模板').selectOption('residual');
  await expect(page.locator('.validation-badge')).toContainText('形状校验通过');
  const pixels = () => page.locator('canvas').evaluate(canvas => new Promise<{ colored: number; pink: number; checksum: number; pinkChecksum: number }>(resolve => requestAnimationFrame(() => {
    const c = canvas as HTMLCanvasElement, gl = c.getContext('webgl2')!;
    const buffer = new Uint8Array(c.width * c.height * 4); gl.readPixels(0, 0, c.width, c.height, gl.RGBA, gl.UNSIGNED_BYTE, buffer);
    let colored = 0, pink = 0, checksum = 0, pinkChecksum = 0;
    for (let i = 0; i < buffer.length; i += 4) {
      const r = buffer[i], g = buffer[i + 1], b = buffer[i + 2];
      if (Math.max(r, g, b) - Math.min(r, g, b) > 30) colored++;
      checksum = (checksum + r * (1 + i % 31) + g) >>> 0;
      if (r > g + 15 && b > g + 10 && r > b + 8) { pink++; pinkChecksum = (pinkChecksum + r * (1 + i % 31) + b) >>> 0; }
    }
    resolve({ colored, pink, checksum, pinkChecksum });
  })));
  await expect.poll(async () => (await pixels()).colored).toBeGreaterThan(1000);
  await expect.poll(async () => (await pixels()).pink).toBeGreaterThan(20);
  mkdirSync('artifacts', { recursive: true });
  await page.screenshot({ path: 'artifacts/residual-desktop.png' });
  await page.getByRole('button', { name: '拓扑图', exact: true }).click();
  const skip = page.locator('.react-flow__edge[data-id="skip"]');
  await expect(skip.locator('[data-flow-kind="residual"] path.react-flow__edge-path')).toHaveCSS('stroke', 'rgb(208, 115, 168)');
  await expect(page.locator('.react-flow__edge[data-id="edge_4"] [data-flow-kind="residual"]')).toHaveCount(0);
  await page.screenshot({ path: 'artifacts/residual-topology.png' });
  await page.locator('.flow-layer').filter({ hasText: '残差融合' }).click();
  await page.getByRole('button', { name: '三维视图', exact: true }).click();
  await page.getByRole('button', { name: '聚焦选中层', exact: true }).click();
  await expect.poll(async () => (await pixels()).pink).toBeGreaterThan(100);
  const forward = await pixels(); await page.waitForTimeout(300); expect((await pixels()).pinkChecksum).not.toBe(forward.pinkChecksum);
  await page.getByRole('button', { name: '暂停数据流', exact: true }).click(); await page.waitForTimeout(300);
  const paused = await pixels(); await page.waitForTimeout(250); expect(Math.abs((await pixels()).checksum - paused.checksum)).toBeLessThan(paused.checksum * 0.0001);
  await page.screenshot({ path: 'artifacts/residual-focused.png' });
  const exportSvg = async (file: string) => {
    await page.getByRole('button', { name: '导出图像', exact: true }).click();
    const event = page.waitForEvent('download'); await page.getByRole('button', { name: 'SVG 矢量图', exact: true }).click();
    await (await event).saveAs(file); return readFileSync(file, 'utf8');
  };
  const inspectSvg = (svg: string) => page.evaluate(markup => {
    const doc = new DOMParser().parseFromString(markup, 'image/svg+xml');
    const edge = doc.querySelector('g[data-edge-id="skip"]')!;
    return { kind: edge.getAttribute('data-flow-kind'), path: edge.querySelector('path')?.getAttribute('d'), stroke: edge.querySelector('path')?.getAttribute('stroke'), arrows: Array.from(edge.querySelectorAll('polygon')).map(p => p.getAttribute('points')), plus: Array.from(doc.querySelectorAll('g[data-label-id^="layer_5:"] text')).some(t => t.textContent === '+'), main: doc.querySelector('g[data-edge-id="edge_4"]')?.getAttribute('data-flow-kind') };
  }, svg);
  const forwardSvg = await inspectSvg(await exportSvg('artifacts/residual.svg'));
  expect(forwardSvg).toMatchObject({ kind: 'residual', stroke: '#d073a8', plus: true, main: 'main' });
  expect(forwardSvg.arrows.length).toBeGreaterThan(0);
  await page.getByRole('button', { name: '反向', exact: true }).click();
  const backwardSvg = await inspectSvg(await exportSvg('artifacts/residual-backward.svg'));
  expect(backwardSvg.stroke).toBe(forwardSvg.stroke); expect(backwardSvg.path).toBe(forwardSvg.path); expect(backwardSvg.arrows).not.toEqual(forwardSvg.arrows);
  await page.getByRole('button', { name: '播放数据流', exact: true }).click();
  const backward = await pixels(); await page.waitForTimeout(300); expect((await pixels()).pinkChecksum).not.toBe(backward.pinkChecksum);
  await page.getByRole('button', { name: '暂停数据流', exact: true }).click(); await page.getByRole('button', { name: '前向', exact: true }).click();
  await page.getByRole('button', { name: '导出图像', exact: true }).click();
  const pngEvent = page.waitForEvent('download'); await page.getByRole('button', { name: '导出 PNG', exact: true }).click();
  await (await pngEvent).saveAs('artifacts/residual-4k.png'); expect(readFileSync('artifacts/residual-4k.png').readUInt32BE(16)).toBe(3840);
  await page.setViewportSize({ width: 390, height: 844 }); await page.getByRole('button', { name: '重置视角', exact: true }).click(); await page.waitForTimeout(300);
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true); expect((await pixels()).colored).toBeGreaterThan(300);
  await page.screenshot({ path: 'artifacts/residual-mobile.png', fullPage: true });
  await page.getByRole('button', { name: '聚焦选中层', exact: true }).click(); await page.waitForTimeout(300);
  expect((await pixels()).pink).toBeGreaterThan(30); await page.screenshot({ path: 'artifacts/residual-mobile-focus.png', fullPage: true });
  expect(errors).toEqual([]); writeFileSync('artifacts/residual-verification.json', JSON.stringify({ desktop: forward, mobile: await pixels(), forwardSvg, backwardSvg, runtimeErrors: errors }, null, 2));
});
