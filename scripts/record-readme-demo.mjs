// Records docs/images/workbench-demo.gif, the README's first image, from a local build.
//
// It needs what the browser tests need, plus PyTorch and ffmpeg: a fresh `npm run build` served
// by run.py with PyTorch installed (the recording trains and measures for real), Microsoft Edge,
// and ffmpeg on PATH. With the service running:
//
//   node scripts/record-readme-demo.mjs [--out <file.gif>] [--keep-frames]
//
// Like the browser tests, it connects to http://127.0.0.1:8765 unless TENSORLAB_TEST_URL names
// another server. The session runs at real speed and the GIF keeps the recorded timing:
// - the Residual CNN template in the English UI;
// - its last convolution cut to 8 channels, which the residual add rejects, then one Undo;
// - a training run with the default settings;
// - every layer measured on one sample of concentric rings and painted onto the 3D blocks,
//   then a focus on the selected layer and a short orbit.
// The cursor is an overlay drawn for the recording, because headless screenshots have none.
import { chromium } from '@playwright/test';
import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const args = process.argv.slice(2);
const outIndex = args.indexOf('--out');
const keepFrames = args.includes('--keep-frames');
const baseURL = process.env.TENSORLAB_TEST_URL || 'http://127.0.0.1:8765';
const WIDTH = 1280, HEIGHT = 800, FPS = 15, LAST_FRAME_SECONDS = 1.2;

function fail(message) {
  console.error(message);
  process.exit(1);
}

if (outIndex >= 0 && !args[outIndex + 1]) fail('--out needs a file path.');
const out = path.resolve(outIndex >= 0 ? args[outIndex + 1] : path.join(root, 'docs/images/workbench-demo.gif'));

if (spawnSync('ffmpeg', ['-version'], { stdio: 'ignore' }).status !== 0) fail('ffmpeg was not found on PATH.');
const health = await fetch(new URL('/api/health', baseURL)).then(response => response.json()).catch(() => null);
if (!health) fail(`No workbench service answered at ${baseURL}. Build, start run.py, or set TENSORLAB_TEST_URL.`);
if (!health.torch) fail('The service has no PyTorch; the recording trains a model and measures every layer.');

// Runs in the page: draws the cursor and a ring on each click.
function installCursor() {
  const install = () => {
    if (document.getElementById('demo-cursor')) return;
    const cursor = document.createElement('div');
    cursor.id = 'demo-cursor';
    cursor.innerHTML = '<svg width="24" height="30" viewBox="0 0 24 30" xmlns="http://www.w3.org/2000/svg"><path d="M3 2.5 L3 23 L8.4 18.1 L12 26.6 L15.6 25.1 L12.1 16.8 L19.6 16.8 Z" fill="#ffffff" stroke="#14211d" stroke-width="1.7" stroke-linejoin="round"/></svg>';
    Object.assign(cursor.style, { position: 'fixed', left: '0', top: '0', width: '24px', height: '30px', pointerEvents: 'none', zIndex: '2147483647', transform: 'translate(-200px, -200px)', filter: 'drop-shadow(0 1px 2px rgba(0, 0, 0, .35))' });
    const ring = document.createElement('div');
    Object.assign(ring.style, { position: 'fixed', left: '-19px', top: '-19px', width: '38px', height: '38px', borderRadius: '50%', border: '3px solid rgba(31, 128, 101, .9)', background: 'rgba(31, 128, 101, .12)', pointerEvents: 'none', zIndex: '2147483646', opacity: '0' });
    document.documentElement.append(ring, cursor);
    addEventListener('mousemove', event => { cursor.style.transform = `translate(${event.clientX - 3}px, ${event.clientY - 2}px)`; }, true);
    addEventListener('mousedown', event => {
      const at = `translate(${event.clientX}px, ${event.clientY}px)`;
      ring.animate([{ opacity: 1, transform: `${at} scale(.35)` }, { opacity: 0, transform: `${at} scale(1.25)` }], { duration: 520, easing: 'ease-out' });
    }, true);
  };
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', install); else install();
}

// Runs in the page: keeps an exported SVG in memory instead of downloading it. The SVG
// export carries each block's projected centre, which is where the recording clicks.
function keepExportedSvg() {
  const createObjectURL = URL.createObjectURL, click = HTMLAnchorElement.prototype.click, blobs = new Map();
  URL.createObjectURL = function(blob) { const url = createObjectURL.call(this, blob); if (blob instanceof Blob) blobs.set(url, blob); return url; };
  HTMLAnchorElement.prototype.click = function() { const blob = blobs.get(this.href); if (!this.download.endsWith('.svg') || !blob) return click.call(this); window.exportedSvg = blob.text(); };
}

async function blockCentres(page) {
  await page.evaluate(() => { delete window.exportedSvg; });
  await page.getByRole('button', { name: 'Export image', exact: true }).click();
  await page.getByLabel('Image layout').selectOption('current');
  await page.getByRole('button', { name: 'SVG vector image', exact: true }).click();
  await page.waitForFunction(() => !!window.exportedSvg);
  const svg = await page.evaluate(() => window.exportedSvg);
  const canvas = await page.locator('canvas').boundingBox();
  return page.evaluate(({ svg, canvas }) => {
    const doc = new DOMParser().parseFromString(svg, 'image/svg+xml');
    const width = Number(doc.documentElement.getAttribute('width')), height = Number(doc.documentElement.getAttribute('height'));
    return Object.fromEntries([...doc.querySelectorAll('g[data-node-id]')].map(g => [g.getAttribute('data-node-id'), {
      x: canvas.x + Number(g.getAttribute('data-center-x')) / width * canvas.width,
      y: canvas.y + Number(g.getAttribute('data-center-y')) / height * canvas.height
    }]));
  }, { svg, canvas });
}

async function ready(page) {
  await page.locator('.validation-badge').filter({ hasText: 'Shapes validated' }).waitFor();
  await page.locator('.statusbar').filter({ hasText: 'PyTorch' }).waitFor();
}

const framesDir = fs.mkdtempSync(path.join(os.tmpdir(), 'tensorcraft-demo-'));
const browser = await chromium.launch({ channel: 'msedge', headless: true, args: ['--enable-webgl', '--ignore-gpu-blocklist'] });
const frames = [], errors = [];
try {
  const context = await browser.newContext({ viewport: { width: WIDTH, height: HEIGHT }, deviceScaleFactor: 1 });
  const page = await context.newPage();
  page.on('pageerror', error => errors.push(error.message));
  await page.addInitScript(() => { try { localStorage.setItem('tensorlab-language', 'en'); } catch { /* Documents without storage. */ } });
  await page.addInitScript(keepExportedSvg);
  await page.addInitScript(installCursor);

  // Setup, not recorded. Layer names are project content and stay as authored in either
  // language, so the template's residual add gets an English name.
  await page.goto(baseURL);
  await page.getByLabel('Model template').selectOption('residual');
  await page.waitForFunction(() => JSON.parse(localStorage.getItem('tensorlab-project') || '{}').name === 'Residual CNN');
  const graph = await page.evaluate(() => {
    const graph = JSON.parse(localStorage.getItem('tensorlab-project'));
    for (const node of graph.nodes) if (node.op === 'Add') node.name = 'Residual add';
    localStorage.setItem('tensorlab-project', JSON.stringify(graph));
    return graph;
  });
  const input = graph.nodes.find(node => node.op === 'Input'), conv = graph.nodes.filter(node => node.op === 'Conv2d').at(-1);
  await page.reload();
  await ready(page);
  await page.waitForTimeout(800);
  const centres = await blockCentres(page);
  // A fresh load drops the export's toast and returns to the same default camera.
  await page.reload();
  await ready(page);
  // One sample of concentric rings centred in the top-left 16 × 16 window, the window each
  // block face shows, so the same shape can be followed from layer to layer. The inspector
  // keeps this setting while another panel tab is shown.
  const [, channels, rows, columns] = input.params.shape;
  const cy = (Math.min(16, rows) - 1) / 2, cx = (Math.min(16, columns) - 1) / 2, sample = [];
  for (let c = 0; c < channels; c++) for (let y = 0; y < rows; y++) for (let x = 0; x < columns; x++)
    sample.push(Number(Math.cos(Math.hypot(x - cx, y - cy) * 1.15 + c * 0.7).toFixed(3)));
  await page.getByRole('button', { name: 'Tensors', exact: true }).click();
  const panel = page.getByRole('region', { name: 'Tensor Inspector', exact: true });
  await panel.getByLabel('Inference input source').selectOption('provided');
  await panel.getByLabel('Inference input JSON').fill(JSON.stringify({ [input.id]: sample }));
  await page.getByRole('button', { name: 'Properties', exact: true }).click();
  await page.evaluate(() => document.querySelectorAll('*').forEach(element => { if (element.scrollTop > 0) element.scrollTop = 0; }));
  await page.waitForTimeout(1500);

  let mouse = { x: WIDTH * 0.62, y: HEIGHT * 0.42 };
  const glide = async (x, y, ms = 650) => {
    const steps = Math.max(10, Math.round(ms / 18)), from = { ...mouse };
    for (let i = 1; i <= steps; i++) {
      const p = i / steps, eased = p < 0.5 ? 4 * p * p * p : 1 - Math.pow(-2 * p + 2, 3) / 2;
      await page.mouse.move(from.x + (x - from.x) * eased, from.y + (y - from.y) * eased);
      await page.waitForTimeout(ms / steps);
    }
    mouse = { x, y };
  };
  const clickAt = async (x, y, ms) => { await glide(x, y, ms); await page.waitForTimeout(140); await page.mouse.down(); await page.waitForTimeout(80); await page.mouse.up(); };
  const clickOn = async (locator, ms) => { const box = await locator.boundingBox(); await clickAt(box.x + box.width / 2, box.y + box.height / 2, ms); };
  const hold = ms => page.waitForTimeout(ms);

  // Recording: DevTools screencast frames with their timestamps.
  const cdp = await context.newCDPSession(page);
  let writes = Promise.resolve();
  cdp.on('Page.screencastFrame', ({ data, metadata, sessionId }) => {
    const file = `${String(frames.length + 1).padStart(6, '0')}.png`;
    frames.push({ file, t: metadata.timestamp });
    writes = writes.then(() => fs.promises.writeFile(path.join(framesDir, file), Buffer.from(data, 'base64')));
    cdp.send('Page.screencastFrameAck', { sessionId }).catch(() => {});
  });
  await page.mouse.move(mouse.x, mouse.y);
  await cdp.send('Page.startScreencast', { format: 'png', everyNthFrame: 3 });
  await hold(1300);

  // Build: cut the convolution to 8 channels; the residual add rejects it; one Undo restores it.
  await clickAt(centres[conv.id].x, centres[conv.id].y, 750);
  await page.waitForFunction(name => [...document.querySelectorAll('label.field-label')].some(label => label.firstChild?.textContent === 'Layer name' && label.querySelector('input')?.value === name), conv.name);
  await hold(1000);
  await clickOn(page.getByLabel('out_channels', { exact: true }), 650);
  await page.keyboard.press('Control+A');
  // Step off the field so the new value stays readable. One keystroke is one history entry.
  await glide(mouse.x - 70, mouse.y + 34, 260);
  await page.keyboard.type('8', { delay: 120 });
  await page.locator('.validation-badge').filter({ hasText: 'structure error' }).waitFor();
  await hold(1600);
  await clickOn(page.getByRole('button', { name: 'Undo', exact: true }), 700);
  await page.locator('.validation-badge').filter({ hasText: 'Shapes validated' }).waitFor();
  await hold(900);

  // Run: train with the default settings.
  await clickOn(page.getByRole('button', { name: 'Train', exact: true }), 750);
  const start = page.getByRole('button', { name: 'Start training', exact: true });
  await start.waitFor();
  await hold(600);
  await clickOn(start, 600);
  await glide(WIDTH * 0.45, HEIGHT * 0.86, 700);
  await page.locator('.train-status').filter({ hasText: 'Training complete' }).waitFor({ timeout: 300000 });
  await hold(1000);

  // Understand: measure every layer on the ring sample and paint it onto the blocks.
  await clickOn(page.getByRole('button', { name: 'Tensors', exact: true }), 700);
  await hold(300);
  const observe = panel.getByRole('button', { name: 'Observe every layer and paint it in 3D', exact: true });
  await glide(WIDTH - 140, HEIGHT * 0.55, 350);
  for (let i = 0; i < 40; i++) {
    const box = await observe.boundingBox();
    if (box && box.y + box.height < HEIGHT - 150) break;
    await page.mouse.wheel(0, 110);
    await page.waitForTimeout(20);
  }
  await hold(250);
  await clickOn(observe, 550);
  await page.locator('.scene-observation').filter({ hasText: 'Measured values · one sample' }).waitFor({ timeout: 60000 });
  await hold(1600);
  await clickOn(page.getByRole('button', { name: 'Focus selected layer', exact: true }), 700);
  await hold(1900);
  const canvas = await page.locator('canvas').boundingBox();
  const ox = canvas.x + canvas.width * 0.5, oy = canvas.y + canvas.height * 0.86;
  await glide(ox, oy, 500);
  await page.mouse.down();
  for (let i = 1; i <= 65; i++) {
    const p = i / 65, eased = p < 0.5 ? 2 * p * p : 1 - Math.pow(-2 * p + 2, 2) / 2;
    await page.mouse.move(ox + 45 * eased, oy - 6 * eased);
    await page.waitForTimeout(24);
  }
  await page.mouse.up();
  await hold(1500);

  await cdp.send('Page.stopScreencast');
  await writes;
} catch (error) {
  if (!keepFrames) fs.rmSync(framesDir, { recursive: true, force: true });
  throw error;
} finally {
  await browser.close();
}
if (errors.length || frames.length < 2) {
  if (!keepFrames) fs.rmSync(framesDir, { recursive: true, force: true });
  fail(errors.length ? `The page reported errors, so no GIF was written:\n${errors.join('\n')}` : 'The screencast produced no frames.');
}

// Encode with the recorded timing. The concat demuxer resolves these names next to the list,
// and it only honours the last duration when the last file is listed again.
const lines = frames.flatMap((frame, i) => [`file '${frame.file}'`, `duration ${(i + 1 < frames.length ? Math.max(0.001, frames[i + 1].t - frame.t) : LAST_FRAME_SECONDS).toFixed(4)}`]);
lines.push(`file '${frames.at(-1).file}'`);
const list = path.join(framesDir, 'frames.txt');
fs.writeFileSync(list, lines.join('\n'));
fs.mkdirSync(path.dirname(out), { recursive: true });
const filter = `fps=${FPS},split[a][b];[a]palettegen=max_colors=256:stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle`;
const encoded = spawnSync('ffmpeg', ['-y', '-hide_banner', '-loglevel', 'error', '-f', 'concat', '-safe', '0', '-i', list, '-vf', filter, '-loop', '0', out], { stdio: 'inherit' });
if (encoded.status !== 0) fail(`ffmpeg failed; the frames are kept in ${framesDir}`);
if (keepFrames) console.log(`Frames kept in ${framesDir}`);
else fs.rmSync(framesDir, { recursive: true, force: true });
const seconds = frames.at(-1).t - frames[0].t + LAST_FRAME_SECONDS;
console.log(`Wrote ${path.relative(root, out) || out}: ${WIDTH} × ${HEIGHT}, ${FPS} fps, ${seconds.toFixed(1)} s, ${(fs.statSync(out).size / 1024 / 1024).toFixed(2)} MiB`);
