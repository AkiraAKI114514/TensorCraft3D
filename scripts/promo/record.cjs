// Records the promo video's shots from a running local build (see docs/workbench-guide.md, "Promo video").
//
//   node scripts/promo/record.cjs <zh|en> all|<shot>... [--keep-frames]
//
// Like the browser tests it connects to http://127.0.0.1:8765 unless TENSORLAB_TEST_URL names another
// server; the service needs PyTorch (some shots train and measure for real). Each shot is set up off
// camera, then recorded as DevTools screencast frames with their timestamps and encoded to a 1920 x 1080,
// 30 fps clip in artifacts/promo/build/clips/<lang>/<shot>.mp4. 3D shots run "cine": the scene pinned over
// the whole viewport with its overlays hidden by injected CSS; the app itself is unchanged.
const { chromium } = require('@playwright/test');
const fs = require('fs'), path = require('path'), { spawnSync } = require('child_process');

const BASE = process.env.TENSORLAB_TEST_URL || 'http://127.0.0.1:8765/';
const ROOT = __dirname;
const BUILD = path.resolve(__dirname, '../../artifacts/promo/build');
const keepFrames = process.argv.includes('--keep-frames');
const [lang, ...shots] = process.argv.slice(2).filter(arg => arg !== '--keep-frames');
if (!['zh', 'en'].includes(lang) || !shots.length) { console.error('usage: node scripts/promo/record.cjs <zh|en> all|<shot>... [--keep-frames]'); process.exit(1); }
const T = (zh, en) => (lang === 'zh' ? zh : en);

// English names for the templates' Chinese layer names (layer names are project content).
const EN = {
  'Token 输入': 'Input IDs', 'Query 输入': 'Query input', 'Context 输入': 'Context input', '图像输入': 'Image input', '特征输入': 'Feature input', '输入': 'Input', '输出': 'Output',
  'Logits 输出': 'Logits', '分类输出': 'Logits', '分类器': 'Classifier', '输出头': 'Head', '卷积块 01': 'Conv block 01', '卷积块 02': 'Conv block 02', '批归一化': 'BatchNorm',
  '激活': 'Activation', '下采样': 'Downsample', '全局池化': 'Global pool', '展平': 'Flatten', '序列展平': 'Flatten', '残差融合': 'Residual add', '多头注意力层': 'Encoder layer 1',
  '多头注意力层 02': 'Encoder layer 2', '共享 KV 注意力': 'Multi-query attention', '分组 KV 编码器': 'Grouped-query encoder', 'GQA 注意力': 'GQA attention', '前置归一化': 'Pre-norm',
  '门控投影': 'Gate projection', '值投影': 'Value projection', '门控激活': 'Gate activation', '逐元素门控': 'Gating', '下投影': 'Down projection', '词嵌入': 'Embedding',
  '最终归一化': 'Final norm', '取最后位置': 'Last position'
};
const english = name => EN[name] ?? name
  .replace(/^注意力归一化 (\d)/, 'Attention norm $1').replace(/^前馈归一化 (\d)/, 'FFN norm $1').replace(/^残差 (\d[ab])/, 'Residual $1')
  .replace(/^门控投影 (\d)/, 'Gate $1').replace(/^上投影 (\d)/, 'Up $1').replace(/^门控激活 (\d)/, 'SiLU $1').replace(/^下投影 (\d)/, 'Down $1');

const CINE_CSS = `.scene-workspace{position:fixed!important;inset:0!important;z-index:2147483000!important;margin:0!important;border:0!important;border-radius:0!important;width:100vw!important;height:100vh!important;max-width:none!important;max-height:none!important}
.scene-workspace>.scene-heading,.scene-workspace>.validation-badge,.scene-workspace>.scene-legend,.scene-workspace>.flow-controls,.scene-workspace>.gpu-badge,.toast-stack,.toast,[role=status].toast{display:none!important}`;
const NO_LABELS = `.scene-label{display:none!important}`;

function installCursor() {
  const install = () => {
    if (document.getElementById('demo-cursor')) return;
    const cursor = document.createElement('div'); cursor.id = 'demo-cursor';
    cursor.innerHTML = '<svg width="26" height="32" viewBox="0 0 24 30" xmlns="http://www.w3.org/2000/svg"><path d="M3 2.5 L3 23 L8.4 18.1 L12 26.6 L15.6 25.1 L12.1 16.8 L19.6 16.8 Z" fill="#ffffff" stroke="#14211d" stroke-width="1.7" stroke-linejoin="round"/></svg>';
    Object.assign(cursor.style, { position: 'fixed', left: '0', top: '0', width: '26px', height: '32px', pointerEvents: 'none', zIndex: '2147483647', transform: 'translate(-200px, -200px)', filter: 'drop-shadow(0 1px 2px rgba(0,0,0,.35))' });
    const ring = document.createElement('div');
    Object.assign(ring.style, { position: 'fixed', left: '-19px', top: '-19px', width: '38px', height: '38px', borderRadius: '50%', border: '3px solid rgba(60, 200, 170, .9)', background: 'rgba(60, 200, 170, .15)', pointerEvents: 'none', zIndex: '2147483646', opacity: '0' });
    document.documentElement.append(ring, cursor);
    addEventListener('mousemove', e => { cursor.style.transform = `translate(${e.clientX - 3}px, ${e.clientY - 2}px)`; }, true);
    addEventListener('mousedown', e => { const at = `translate(${e.clientX}px, ${e.clientY}px)`; ring.animate([{ opacity: 1, transform: `${at} scale(.35)` }, { opacity: 0, transform: `${at} scale(1.25)` }], { duration: 520, easing: 'ease-out' }); }, true);
  };
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', install); else install();
}

let browser, context, page, errors = [];
async function open({ ui = false } = {}) {
  browser = await chromium.launch({ channel: 'msedge', headless: true, args: ['--enable-webgl', '--ignore-gpu-blocklist'] });
  context = ui ? await browser.newContext({ viewport: { width: 1440, height: 810 }, deviceScaleFactor: 4 / 3 })
    : await browser.newContext({ viewport: { width: 1280, height: 720 }, deviceScaleFactor: 1.5 });
  await context.addInitScript(language => { try { localStorage.setItem('tensorlab-language', language); localStorage.setItem('tensorlab-legend-open', '0'); localStorage.setItem('tensorlab-scene-theme', 'night'); } catch { } }, lang);
  if (ui) await context.addInitScript(installCursor);
  page = await context.newPage();
  errors = []; page.on('pageerror', e => errors.push(e.message));
  await page.goto(BASE);
  await page.locator('.validation-badge').waitFor();
}
async function close() { await browser.close(); }
const wait = ms => page.waitForTimeout(ms);
const badgeOk = () => page.locator('.validation-badge.success').waitFor({ timeout: 20000 });

/** Loads a template (by key) or a graph object, renaming layers to English in the English cut. */
async function load(source) {
  let graph = source;
  if (typeof source === 'string') {
    await page.locator('.preset-selector select').selectOption(source);
    await page.waitForFunction(() => !!localStorage.getItem('tensorlab-project'));
    await wait(300);
    graph = await page.evaluate(() => JSON.parse(localStorage.getItem('tensorlab-project')));
  }
  graph = JSON.parse(JSON.stringify(graph));
  if (lang === 'en') for (const node of graph.nodes) node.name = english(node.name);
  await page.evaluate(g => localStorage.setItem('tensorlab-project', JSON.stringify(g)), graph);
  await page.reload();
  await badgeOk();
  await wait(600);
  return graph;
}
const showcase = name => JSON.parse(fs.readFileSync(path.join(ROOT, 'graphs', `showcase-${name}.json`), 'utf8'));

async function view(which) { await page.locator('.view-tabs button').nth(which === '3d' ? 0 : 1).click(); await wait(500); }
async function select(id) {
  await view('graph');
  await page.locator(`.react-flow__node[data-id="${id}"]`).click({ force: true });
  await wait(200);
  await view('3d');
}
async function cine(on = true, labels = true) {
  await page.evaluate(([on, css]) => {
    document.getElementById('promo-cine')?.remove();
    if (!on) return;
    const style = document.createElement('style'); style.id = 'promo-cine'; style.textContent = css; document.head.append(style);
  }, [on, CINE_CSS + (labels ? '' : NO_LABELS)]);
  await wait(900);
}
const clickTitle = titles => page.evaluate(titles => { for (const title of titles) { const b = document.querySelector(`button[title="${title}"]`); if (b) { b.click(); return true; } } return false; }, titles);
const resetView = () => clickTitle(['重置视角', 'Reset view']);
const focusSelected = () => clickTitle(['聚焦选中层', 'Focus selected layer']);
async function centre() { const box = await page.locator('canvas').first().boundingBox(); return { x: box.x + box.width / 2, y: box.y + box.height / 2, box }; }

const ease = p => (p < 0.5 ? 4 * p * p * p : 1 - Math.pow(-2 * p + 2, 3) / 2);
const linear = p => p;
/** One camera move over `ms` of wall time: orbit (left drag) by [dx, dy] px, or pan (right drag), and dolly by `zoom` wheel units (negative = in). */
async function move({ ms, orbit = [0, 0], pan = [0, 0], zoom = 0, curve = ease }) {
  const c = await centre();
  const drag = (orbit[0] || orbit[1]) ? ['left', orbit] : (pan[0] || pan[1]) ? ['right', pan] : null;
  if (drag) { await page.mouse.move(c.x, c.y); await page.mouse.down({ button: drag[0] }); }
  const start = Date.now(); let zoomed = 0;
  for (;;) {
    const p = Math.min(1, (Date.now() - start) / ms), q = curve(p);
    if (drag) await page.mouse.move(c.x + drag[1][0] * q, c.y + drag[1][1] * q);
    if (zoom) { const target = Math.round(zoom * q); if (target !== zoomed) { await page.mouse.wheel(0, target - zoomed); zoomed = target; } }
    if (p >= 1) break;
    await wait(12);
  }
  if (drag) await page.mouse.up({ button: drag[0] });
}
const hold = ms => wait(ms);
/** Wheel zoom in quick steps, not recorded framing. */
async function zoomBy(units) { const c = await centre(); await page.mouse.move(c.x, c.y); const n = Math.ceil(Math.abs(units) / 100); for (let i = 0; i < n; i++) { await page.mouse.wheel(0, units / n); await wait(30); } await wait(300); }

let glideFrom = { x: 700, y: 400 };
async function glide(x, y, ms = 600) {
  const steps = Math.max(10, Math.round(ms / 16)), from = { ...glideFrom };
  for (let i = 1; i <= steps; i++) { const p = ease(i / steps); await page.mouse.move(from.x + (x - from.x) * p, from.y + (y - from.y) * p); await wait(ms / steps); }
  glideFrom = { x, y };
}
async function clickOn(locator, ms = 600) { const b = await locator.boundingBox(); await glide(b.x + b.width / 2, b.y + b.height / 2, ms); await wait(120); await page.mouse.down(); await wait(70); await page.mouse.up(); }

/** Records `action` as one clip. */
async function record(name, action) {
  const dir = path.join(BUILD, 'frames', lang, name);
  fs.rmSync(dir, { recursive: true, force: true }); fs.mkdirSync(dir, { recursive: true });
  const cdp = await context.newCDPSession(page), frames = [];
  let writes = Promise.resolve();
  cdp.on('Page.screencastFrame', ({ data, metadata, sessionId }) => {
    const file = `${String(frames.length + 1).padStart(6, '0')}.jpg`;
    frames.push({ file, t: metadata.timestamp });
    writes = writes.then(() => fs.promises.writeFile(path.join(dir, file), Buffer.from(data, 'base64')));
    cdp.send('Page.screencastFrameAck', { sessionId }).catch(() => {});
  });
  await cdp.send('Page.startScreencast', { format: 'jpeg', quality: 94, maxWidth: 1920, maxHeight: 1080, everyNthFrame: 1 });
  await wait(250);
  await action();
  await cdp.send('Page.stopScreencast'); await wait(300); await writes; await cdp.detach(); await writes;
  if (frames.length < 2) throw new Error(`${name}: no frames`);
  const lines = frames.flatMap((f, i) => [`file '${f.file}'`, `duration ${(i + 1 < frames.length ? Math.max(0.001, frames[i + 1].t - f.t) : 0.05).toFixed(4)}`]);
  lines.push(`file '${frames.at(-1).file}'`);
  fs.writeFileSync(path.join(dir, 'frames.txt'), lines.join('\n'));
  const out = path.join(BUILD, 'clips', lang, `${name}.mp4`); fs.mkdirSync(path.dirname(out), { recursive: true });
  const r = spawnSync('ffmpeg', ['-y', '-hide_banner', '-loglevel', 'error', '-f', 'concat', '-safe', '0', '-i', path.join(dir, 'frames.txt'),
    '-vf', 'fps=30,scale=1920:1080:flags=lanczos,format=yuv420p', '-c:v', 'libx264', '-preset', 'slow', '-crf', '12', '-movflags', '+faststart', out], { stdio: 'inherit' });
  if (r.status !== 0) throw new Error(`${name}: ffmpeg failed; frames kept in ${dir}`);
  if (!keepFrames) fs.rmSync(dir, { recursive: true, force: true });
  const seconds = frames.at(-1).t - frames[0].t;
  console.log(`${lang}/${name}: ${frames.length} frames, ${seconds.toFixed(2)} s, ${(frames.length / seconds).toFixed(1)} fps captured`);
  if (errors.length) console.log(`  page errors: ${errors.join(' | ')}`);
}

const SHOTS = {};
const tab = (zh, en) => page.getByRole('button', { name: T(zh, en), exact: true });
async function ringSample(graph) {
  const input = graph.nodes.find(n => n.op === 'Input'); const [, channels, rows, columns] = input.params.shape;
  const cy = (Math.min(16, rows) - 1) / 2, cx = (Math.min(16, columns) - 1) / 2, sample = [];
  for (let c = 0; c < channels; c++) for (let y = 0; y < rows; y++) for (let x = 0; x < columns; x++) sample.push(Number(Math.cos(Math.hypot(x - cx, y - cy) * 1.15 + c * 0.7).toFixed(3)));
  await tab('张量观测', 'Tensors').click();
  const panel = page.getByRole('region', { name: 'Tensor Inspector', exact: true });
  await panel.getByLabel(T('推理输入来源', 'Inference input source')).selectOption('provided');
  await panel.getByLabel(T('推理输入 JSON', 'Inference input JSON')).fill(JSON.stringify({ [input.id]: sample }));
  await tab('层属性', 'Properties').click();
  return panel;
}
async function train(recorded = false) {
  const go = recorded ? clickOn : (l => l.click());
  await go(tab('训练', 'Train')); const start = page.getByRole('button', { name: T('开始真实训练', 'Start training'), exact: true }); await start.waitFor(); if (recorded) await hold(500);
  await go(start); await page.locator('.train-status').filter({ hasText: T('训练完成', 'Training complete') }).waitFor({ timeout: 300000 });
}
async function hideObservationCard() { await page.addStyleTag({ content: '.scene-observation{display:none!important}' }); }

// ---- 3D, cinematic ----
SHOTS.open_pulses = async () => {
  await open(); await load('deepseek'); await select('residual_1b'); await cine(true, false); await focusSelected(); await wait(1800); await zoomBy(-1400);
  await record('open_pulses', () => move({ ms: 5000, pan: [-240, -70], curve: linear })); await close();
};
SHOTS.open_mla = async () => {
  await open(); await load('deepseek'); await select('mla_2'); await cine(true, false); await focusSelected(); await wait(1800); await zoomBy(-1000);
  await record('open_mla', () => move({ ms: 5000, orbit: [-150, 10], curve: linear })); await close();
};
SHOTS.open_moe = async () => {
  await open(); await load('deepseek'); await select('moe_3'); await cine(true, false); await focusSelected(); await wait(1800); await zoomBy(-1000);
  await record('open_moe', () => move({ ms: 5000, orbit: [150, -10], curve: linear })); await close();
};
SHOTS.hero = async () => {
  await open(); await load('deepseek'); await cine(true, false); await resetView(); await wait(1200); await zoomBy(-300);
  await record('hero', () => move({ ms: 8000, orbit: [60, 0], curve: linear })); await close();
};
SHOTS.outro = async () => {
  await open(); await load('deepseek'); await cine(true, false); await resetView(); await wait(1200); await zoomBy(-200);
  await record('outro', () => move({ ms: 7000, orbit: [70, 0], curve: linear })); await close();
};
SHOTS.attn_gqa = async () => {
  await open(); await load('gqa'); await select('layer_1'); await cine(true, false); await focusSelected(); await wait(1800); await zoomBy(-300);
  await record('attn_gqa', () => move({ ms: 6000, orbit: [-150, 0], curve: linear })); await close();
};
// Attention written out by hand: Einsum QKᵀ, Scale, CausalMask and Softmax, each drawn as what it does.
SHOTS.attn_by_hand = async () => {
  await open(); await load(showcase('attention')); await select('mask'); await cine(true, true); await focusSelected(); await wait(1800); await zoomBy(200);
  await record('attn_by_hand', () => move({ ms: 6000, orbit: [-70, 0], curve: linear })); await close();
};
SHOTS.attn_cross = async () => {
  await open(); await load('cross_attention'); await select('layer_1'); await cine(true, false); await focusSelected(); await wait(1800); await zoomBy(-300);
  await record('attn_cross', () => move({ ms: 5000, orbit: [130, 0], curve: linear })); await close();
};
SHOTS.wide_signal = async () => {
  await open(); await load(showcase('signal')); await cine(true, true); await resetView(); await wait(1200); await zoomBy(-300);
  await record('wide_signal', () => move({ ms: 6000, orbit: [-70, 0], curve: linear })); await close();
};
SHOTS.wide_sequence = async () => {
  await open(); await load(showcase('sequence')); await cine(true, true); await resetView(); await wait(1200); await zoomBy(-300);
  await record('wide_sequence', () => move({ ms: 6000, orbit: [70, 0], curve: linear })); await close();
};
// One fly-in per operator: from the overview to the layer, then a small drift.
// Grouped by graph so each opens once; the operators drawn as what they do (Dropout2d, LogSoftmax,
// Einsum, Scale, CausalMask, Softmax, AlphaDropout) are among them.
const PARADE = [['vision', 'c0_0'], ['vision', 'c0_3'], ['vision', 'c0_10'], ['vision', 'c0_19'], ['sequence', 'c0_1'], ['sequence', 'c0_4'], ['sequence', 'c0_5'],
  ['attention', 'scores'], ['attention', 'scale'], ['attention', 'mask'], ['attention', 'softmax'], ['signal', 'alpha']];
SHOTS.parade = async () => {
  let current = null;
  for (const [graphName, id] of PARADE) {
    if (graphName !== current) { if (current) await close(); await open(); await load(showcase(graphName)); current = graphName; }
    await cine(false); await select(id); await cine(true, true); await resetView(); await wait(900);
    const c = await centre(); await page.mouse.move(c.x, c.y);
    // The flight lands on a framing shared with the neighbours; ten quick wheel steps as it lands
    // carry it on into a close-up, so each operator fills its flash.
    await record(`parade_${graphName}_${id}`, async () => { await focusSelected(); await hold(380); for (let i = 0; i < 10; i++) { await page.mouse.wheel(0, -120); await wait(40); } await hold(900); await move({ ms: 1500, orbit: [-70, 0], curve: linear }); });
  }
  await close();
};
SHOTS.rings = async () => {
  await open(); const graph = await load('residual'); await ringSample(graph); await train(); await tab('张量观测', 'Tensors').click();
  await page.getByRole('region', { name: 'Tensor Inspector', exact: true }).getByRole('button', { name: T('观测全部层，画到三维视图', 'Observe every layer and paint it in 3D'), exact: true }).click();
  await page.locator('.scene-observation').waitFor({ timeout: 60000 });
  await select('layer_3'); await hideObservationCard(); await cine(true, true); await focusSelected(); await wait(1800); await zoomBy(300);
  await record('rings', () => move({ ms: 6000, orbit: [-80, 0], curve: linear })); await close();
};

// ---- UI ----
SHOTS.ui_build = async () => {
  await open({ ui: true }); await load('blank'); await select('layer_0'); await wait(600);
  await record('ui_build', async () => {
    await hold(500);
    for (const op of ['Conv2d', 'BatchNorm2d', 'ReLU', 'MaxPool2d', 'Flatten', 'Linear']) {
      const item = page.locator('.library button').filter({ has: page.locator(`text="${op}"`) }).first();
      await item.evaluate(el => el.scrollIntoView({ block: 'center', behavior: 'smooth' })); await hold(380);
      await clickOn(item, 420); await hold(480);
    }
    await badgeOk(); await hold(900);
  });
  await close();
};
SHOTS.ui_error = async () => {
  await open({ ui: true }); await load('residual'); await select('layer_3'); await wait(800);
  await record('ui_error', async () => {
    await hold(400);
    await clickOn(page.getByLabel('out_channels', { exact: true }), 600);
    await page.keyboard.press('Control+A'); await glide(glideFrom.x - 70, glideFrom.y + 34, 240); await page.keyboard.type('8', { delay: 100 });
    await page.locator('.validation-badge').filter({ hasText: T('结构错误', 'structure error') }).waitFor(); await hold(1700);
    await clickOn(page.locator(`button[title="${T('撤销', 'Undo')}"]`), 650); await badgeOk(); await hold(900);
  });
  await close();
};
SHOTS.ui_import = async () => {
  await open({ ui: true }); await load('mlp'); await wait(500);
  await record('ui_import', async () => {
    await hold(300);
    await clickOn(page.locator(`button[title="${T('导入 PyTorch 代码', 'Import PyTorch code')}"]`), 600);
    const dialog = page.getByRole('dialog', { name: T('导入 PyTorch 代码', 'Import PyTorch code'), exact: true }); await dialog.waitFor(); await hold(400);
    const example = dialog.getByLabel(T('PyTorch 导入示例', 'PyTorch import example'));
    const box = await example.boundingBox(); await glide(box.x + box.width / 2, box.y + box.height / 2, 500); await example.selectOption('transformer'); await hold(900);
    await clickOn(dialog.getByRole('button', { name: T('解析代码', 'Parse code'), exact: true }), 600);
    await dialog.locator('.import-preview').waitFor({ timeout: 20000 }); await hold(1300);
    await clickOn(dialog.getByRole('button', { name: T('导入模型', 'Import model'), exact: true }), 600);
    await dialog.waitFor({ state: 'detached' }); await badgeOk(); await hold(1800);
  });
  await close();
};
SHOTS.ui_train = async () => {
  await open({ ui: true }); await load('residual'); await wait(500);
  await record('ui_train', async () => { await hold(300); await train(true); await hold(1500); });
  await close();
};
SHOTS.ui_inspect = async () => {
  await open({ ui: true }); const graph = await load('residual'); await ringSample(graph); await train(); await select('layer_1'); await wait(600);
  await record('ui_inspect', async () => {
    await hold(300);
    await clickOn(tab('张量观测', 'Tensors'), 600);
    const panel = page.getByRole('region', { name: 'Tensor Inspector', exact: true });
    const run = panel.getByRole('button', { name: T('运行单样本推理', 'Run single-sample inference'), exact: true });
    await run.evaluate(el => el.scrollIntoView({ block: 'center', behavior: 'smooth' })); await hold(450);
    await clickOn(run, 600);
    const grid = page.locator('figure.channel-grid:not(.head-overview)').first(); await grid.waitFor({ timeout: 60000 }); await hold(500);
    await grid.evaluate(el => el.scrollIntoView({ behavior: 'smooth', block: 'center' })); await hold(1600);
    const observe = panel.getByRole('button', { name: T('观测全部层，画到三维视图', 'Observe every layer and paint it in 3D'), exact: true });
    await observe.evaluate(el => el.scrollIntoView({ behavior: 'smooth', block: 'center' })); await hold(700);
    await clickOn(observe, 600); await page.locator('.scene-observation').waitFor({ timeout: 60000 }); await hold(2200);
  });
  await close();
};
SHOTS.ui_attention = async () => {
  await open({ ui: true }); await load('gqa'); await select('layer_1'); await wait(600);
  await record('ui_attention', async () => {
    await hold(300);
    await clickOn(tab('张量观测', 'Tensors'), 600);
    const panel = page.getByRole('region', { name: 'Tensor Inspector', exact: true });
    const run = panel.getByRole('button', { name: T('运行单样本推理', 'Run single-sample inference'), exact: true });
    await run.evaluate(el => el.scrollIntoView({ block: 'center', behavior: 'smooth' })); await hold(450);
    await clickOn(run, 600);
    const attention = panel.getByRole('region', { name: 'Attention Inspector', exact: true }); await attention.waitFor({ timeout: 60000 });
    await attention.locator('.head-overview').evaluate(el => el.scrollIntoView({ behavior: 'smooth', block: 'start' })); await hold(1800);
    await attention.locator('.attention-matrix:not(.head-overview)').first().evaluate(el => el.scrollIntoView({ behavior: 'smooth', block: 'center' })); await hold(1800);
  });
  await close();
};
SHOTS.ui_export = async () => {
  await open({ ui: true }); await load('transformer'); await wait(600);
  await record('ui_export', async () => {
    await hold(300);
    await clickOn(page.getByRole('button', { name: T('导出代码', 'Export code'), exact: true }), 600); await page.locator('.code-preview').waitFor({ timeout: 20000 }); await hold(2000);
    await clickOn(page.getByRole('button', { name: T('关闭对话框', 'Close dialog'), exact: true }).first(), 500); await hold(300);
    await clickOn(page.getByRole('button', { name: T('导出图像', 'Export image'), exact: true }), 600); await hold(2400);
  });
  await close();
};
const ALL_3D = ['open_pulses', 'open_mla', 'open_moe', 'hero', 'outro', 'attn_gqa', 'attn_by_hand', 'attn_cross', 'wide_signal', 'wide_sequence', 'parade', 'rings'];
const ALL_UI = ['ui_build', 'ui_error', 'ui_import', 'ui_train', 'ui_inspect', 'ui_attention', 'ui_export'];
SHOTS.all = async () => { for (const name of [...ALL_3D, ...ALL_UI]) { try { await SHOTS[name](); } catch (error) { console.error(`${lang}/${name} failed: ${error.stack || error}`); process.exitCode = 1; try { await close(); } catch { } } } };

(async () => {
  for (const name of shots) {
    if (!SHOTS[name]) { console.error(`unknown shot ${name}`); process.exitCode = 1; continue; }
    try { await SHOTS[name](); } catch (error) { console.error(`${lang}/${name} failed: ${error.stack || error}`); process.exitCode = 1; try { await close(); } catch { } }
  }
})();
