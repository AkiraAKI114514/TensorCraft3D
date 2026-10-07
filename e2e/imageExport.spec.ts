import { test, expect } from '@playwright/test';
import { mkdirSync, readFileSync } from 'node:fs';
import { imageTestGraph } from './imageGraph';
import { PRESETS } from '../src/presets';

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
    const bounds = (polygons: Element[]) => {
      const points = polygons.flatMap(p => p.getAttribute('points')!.trim().split(/\s+/).map(point => point.split(',').map(Number)));
      return { left: Math.min(...points.map(p => p[0])), right: Math.max(...points.map(p => p[0])), top: Math.min(...points.map(p => p[1])), bottom: Math.max(...points.map(p => p[1])) };
    };
    const centers = Array.from(document.querySelectorAll('g[data-node-id]')).map(node => ({ id: node.getAttribute('data-node-id')!, x: Number(node.getAttribute('data-center-x')), y: Number(node.getAttribute('data-center-y')), ...bounds(Array.from(node.querySelectorAll('polygon'))) }));
    const face = bounds(Array.from(document.querySelectorAll('g[data-node-id="import_4"] polygon[data-part="qkv-face"]')));
    return { width: Number(root.getAttribute('width')), height: Number(root.getAttribute('height')), bands: Number(root.getAttribute('data-bands')), labels, centers, faceWidth: face.right - face.left };
  }, readFileSync('artifacts/compact-model.svg', 'utf8'));
  expect(measurements.width).toBe(1920); expect(measurements.height).toBe(1080); expect(measurements.bands).toBe(2);
  // These normalized centers record the layout in compact-model-before-size.png.
  // Camera fitting can change the zoom, but enlargement must not move the nodes.
  const origin = measurements.centers[0], slab = measurements.centers.find(node => node.id === 'import_2')!, pitch = slab.x - origin.x;
  const reference = [[0, 0], [0, 3.368], [1, 0], [2, 0], [5.064, 0], [10.192, 0], [1, 3.368], [2, 3.368], [5.064, 3.368], [10.192, 3.368], [0.128, 6.736], ...Array.from({ length: 12 }, (_, i) => [2.332 + i, 6.736])];
  measurements.centers.forEach((node, i) => {
    expect((node.x - origin.x) / pitch).toBeCloseTo(reference[i][0], 4);
    expect((node.y - origin.y) / pitch).toBeCloseTo(reference[i][1], 4);
    expect(node.left).toBeGreaterThan(0); expect(node.right).toBeLessThan(1920);
    expect(node.top).toBeGreaterThan(0); expect(node.bottom).toBeLessThan(1080);
    for (const other of measurements.centers.slice(i + 1)) expect(node.left < other.right && node.right > other.left && node.top < other.bottom && node.bottom > other.top, `${node.id} overlaps ${other.id}`).toBe(false);
  });
  expect((slab.right - slab.left) / pitch).toBeGreaterThan(0.42);
  expect(measurements.faceWidth / pitch).toBeGreaterThan(0.62);
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
  // The compact layout must fill the frame: measure the ink bounding box so a
  // layout that shrinks the model into a corner fails here. Horizontal slack is
  // expected when cross-band routes need room, so it is bounded more loosely.
  const fill = await page.evaluate(async url => {
    const image = new Image(); image.src = url; await image.decode();
    const canvas = document.createElement('canvas'); canvas.width = image.width; canvas.height = image.height;
    const context = canvas.getContext('2d')!; context.drawImage(image, 0, 0);
    const pixels = context.getImageData(0, 0, canvas.width, canvas.height).data;
    let left = canvas.width, right = 0, top = canvas.height, bottom = 0;
    for (let y = 0; y < canvas.height; y++) for (let x = 0; x < canvas.width; x++) {
      const i = (y * canvas.width + x) * 4;
      if (Math.abs(pixels[i] - 245) + Math.abs(pixels[i + 1] - 248) + Math.abs(pixels[i + 2] - 250) < 12) continue;
      if (x < left) left = x; if (x > right) right = x; if (y < top) top = y; if (y > bottom) bottom = y;
    }
    return { left, right, top, bottom, width: canvas.width, height: canvas.height };
  }, `data:image/png;base64,${bytes.toString('base64')}`);
  expect(fill.bottom - fill.top).toBeGreaterThan(fill.height * 0.85);
  expect(fill.right - fill.left).toBeGreaterThan(fill.width * 0.8);
  expect(fill.left).toBeLessThan(fill.width * 0.05);
  expect(fill.top).toBeLessThan(fill.height * 0.05);
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

test('keeps exported attention edges attached to enlarged query, context and projection ports', async ({ page }, testInfo) => {
  const graph = PRESETS.cross_attention(), errors: string[] = [];
  graph.nodes[1].params.num_heads = 16; graph.nodes[1].params.kv_heads = 16;
  for (let head = 0; head < 16; head++) graph.edges.push({ id: `query_${head}`, source: 'layer_0', target: 'layer_1', targetPort: `b0:q${head}` });
  for (const role of ['k', 'v']) graph.edges.push({ id: `context_${role}`, source: 'context', target: 'layer_1', targetPort: `b0:${role}0` });
  graph.edges.find(edge => edge.source === 'layer_1')!.sourcePort = 'b0:q0';
  page.on('pageerror', error => errors.push(error.message));
  await page.addInitScript(graph => localStorage.setItem('tensorlab-project', JSON.stringify(graph)), graph);
  await page.goto('/');
  await expect(page.locator('.validation-badge')).toContainText('形状校验通过');
  await page.getByRole('button', { name: '暂停数据流', exact: true }).click();
  await page.getByRole('button', { name: '导出图像', exact: true }).click();
  await page.getByLabel('PNG 分辨率').selectOption('1920');
  const event = page.waitForEvent('download'); await page.getByRole('button', { name: 'SVG 矢量图', exact: true }).click();
  const file = testInfo.outputPath('attention-port-alignment.svg'); await (await event).saveAs(file);
  const distances = await page.evaluate(({ svg, edges }) => {
    const document = new DOMParser().parseFromString(svg, 'image/svg+xml');
    const center = (part: string) => {
      const mesh = document.querySelector(`g[data-node-id="layer_1"] polygon[data-part="${part}"]`)!;
      return [Number(mesh.getAttribute('data-center-x')), Number(mesh.getAttribute('data-center-y'))];
    };
    return edges.filter(edge => edge.target === 'layer_1' || edge.source === 'layer_1').map(edge => {
      const path = document.querySelector(`g[data-edge-id="${edge.id}"] path`)!.getAttribute('d')!.trim().split(/\s+/);
      const source = edge.source === 'layer_1';
      const endpoint = path[source ? 0 : path.length - 1].slice(1).split(',').map(Number);
      const port = center(source ? edge.sourcePort! : edge.targetPort === 'query' ? 'input' : edge.targetPort === 'context' ? 'context-input' : edge.targetPort!);
      return { id: edge.id, distance: Math.hypot(endpoint[0] - port[0], endpoint[1] - port[1]) };
    });
  }, { svg: readFileSync(file, 'utf8'), edges: graph.edges });
  expect(distances).toHaveLength(21);
  distances.forEach(edge => expect(edge.distance, edge.id).toBeLessThan(0.01));
  expect(await page.evaluate(() => JSON.parse(localStorage.getItem('tensorlab-project')!))).toEqual(graph);
  expect(errors).toEqual([]);
});
