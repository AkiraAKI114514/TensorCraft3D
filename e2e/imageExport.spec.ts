import { test, expect } from '@playwright/test';
import { mkdirSync, readFileSync } from 'node:fs';
import { imageTestGraph } from './imageGraph';

test('exports a compact dual-branch model with readable labels and a fixed aspect ratio', async ({ page }) => {
  const graph = imageTestGraph(), errors: string[] = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.addInitScript(graph => localStorage.setItem('tensorlab-project', JSON.stringify(graph)), graph);
  await page.goto('/');
  await expect(page.locator('.validation-badge')).toContainText('形状校验通过');
  await page.getByRole('button', { name: '暂停数据流', exact: true }).click();
  mkdirSync('artifacts', { recursive: true });
  await page.getByRole('button', { name: '导出图像', exact: true }).click();
  await page.getByLabel('PNG 分辨率').selectOption('1920');
  await expect(page.getByLabel('图像布局')).toHaveValue('compact');
  const svgEvent = page.waitForEvent('download'); await page.getByRole('button', { name: 'SVG 矢量图', exact: true }).click();
  await (await svgEvent).saveAs('artifacts/compact-model.svg');
  const measurements = await page.evaluate(svg => {
    const document = new DOMParser().parseFromString(svg, 'image/svg+xml'), root = document.documentElement;
    const labels = Array.from(document.querySelectorAll('g[data-label-id]')).map(label => ({ id: label.getAttribute('data-label-id')!, x: Number(label.getAttribute('data-x')), y: Number(label.getAttribute('data-y')), width: Number(label.getAttribute('data-width')), height: Number(label.getAttribute('data-height')) }));
    const centers = Array.from(document.querySelectorAll('g[data-node-id]')).map(node => ({ x: Number(node.getAttribute('data-center-x')), y: Number(node.getAttribute('data-center-y')) }));
    return { width: Number(root.getAttribute('width')), height: Number(root.getAttribute('height')), bands: Number(root.getAttribute('data-bands')), labels, centers };
  }, readFileSync('artifacts/compact-model.svg', 'utf8'));
  expect(measurements.width).toBe(1920); expect(measurements.height).toBe(1080); expect(measurements.bands).toBeGreaterThan(1);
  expect(measurements.labels.filter(label => graph.nodes.some(node => node.id === label.id))).toHaveLength(graph.nodes.length);
  for (const [i, label] of measurements.labels.entries()) {
    expect(label.x).toBeGreaterThanOrEqual(0); expect(label.y).toBeGreaterThanOrEqual(0);
    expect(label.x + label.width).toBeLessThanOrEqual(1920); expect(label.y + label.height).toBeLessThanOrEqual(1080);
    for (const other of measurements.labels.slice(i + 1)) expect(label.x < other.x + other.width && label.x + label.width > other.x && label.y < other.y + other.height && label.y + label.height > other.y, `${label.id} overlaps ${other.id}`).toBe(false);
  }
  measurements.centers.forEach(center => { expect(center.x).toBeGreaterThan(0); expect(center.x).toBeLessThan(1920); expect(center.y).toBeGreaterThan(0); expect(center.y).toBeLessThan(1080); });
  await page.getByRole('button', { name: '导出图像', exact: true }).click();
  const pngEvent = page.waitForEvent('download'); await page.getByRole('button', { name: '导出 PNG', exact: true }).click();
  await (await pngEvent).saveAs('artifacts/compact-model-1920.png');
  const bytes = readFileSync('artifacts/compact-model-1920.png'); expect(bytes.readUInt32BE(16)).toBe(1920); expect(bytes.readUInt32BE(20)).toBe(1080);
  expect(await page.evaluate(() => JSON.parse(localStorage.getItem('tensorlab-project')!))).toEqual(graph);
  await page.getByRole('button', { name: '导出图像', exact: true }).click();
  await page.getByLabel('画幅比例').selectOption('1'); await page.getByLabel('标注字号').fill('9'); await page.getByLabel('透明背景').check();
  const squareEvent = page.waitForEvent('download'); await page.getByRole('button', { name: '导出 PNG', exact: true }).click(); await (await squareEvent).saveAs('artifacts/compact-model-square.png');
  const square = readFileSync('artifacts/compact-model-square.png'); expect(square.readUInt32BE(16)).toBe(1920); expect(square.readUInt32BE(20)).toBe(1920);
  const alpha = await page.evaluate(async url => {
    const image = new Image(); image.src = url; await image.decode();
    const canvas = document.createElement('canvas'); canvas.width = image.width; canvas.height = image.height;
    const context = canvas.getContext('2d')!; context.drawImage(image, 0, 0);
    const pixels = context.getImageData(0, 0, canvas.width, canvas.height).data;
    let visible = 0; for (let i = 3; i < pixels.length; i += 4) if (pixels[i] > 0) visible++;
    return { corner: pixels[3], visible };
  }, `data:image/png;base64,${square.toString('base64')}`);
  expect(alpha.corner).toBe(0); expect(alpha.visible).toBeGreaterThan(10000);
  expect(errors).toEqual([]);
});
