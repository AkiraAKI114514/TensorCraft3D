// The promo video's title cards, rendered frame by frame to transparent PNGs (see docs/workbench-guide.md).
//
//   node scripts/promo/cards.cjs <zh|en> [card...]
//
// Writes artifacts/promo/build/titles/<lang>/<card>/0001.png ... and a manifest of frame counts. Each card
// is HTML whose every animated property is set for each frame's time, so the motion is exact. Fonts:
// Segoe UI Variable for Latin text, Noto Sans SC (else Microsoft YaHei UI) for Chinese.
const { chromium } = require('@playwright/test');
const fs = require('fs'), path = require('path');
const BUILD = path.resolve(__dirname, '../../artifacts/promo/build'), FPS = 30;
const [lang, ...only] = process.argv.slice(2);
if (!['zh', 'en'].includes(lang)) { console.error('usage: node scripts/promo/cards.cjs <zh|en> [card...]'); process.exit(1); }
// The operator count on the counter card: every layer operator, without Input, Output and Group.
const OPERATORS = (() => {
  const source = fs.readFileSync(path.resolve(__dirname, '../../src/types.ts'), 'utf8');
  const list = source.match(/export const OPS = \[([\s\S]*?)\] as const/)[1];
  return [...list.matchAll(/'(\w+)'/g)].map(m => m[1]).filter(op => !['Input', 'Output', 'Group'].includes(op)).length;
})();
const L = (zh, en) => (lang === 'zh' ? zh : en);
// The release the end card names; set it to the version the video ships with.
const RELEASE = 'v1.0.0-beta.3';

// Plain wording, few punctuation marks: no separators, no full stops at the end of a line.
const FEATURES = {
  f1: [L('01 搭建', '01 BUILD'), L('像搭积木一样搭模型', 'Build a model block by block'), L('参数改错了，出问题的层会直接变红', 'Get a shape wrong and the layers it breaks turn red')],
  f3: [L('03 结构', '03 INSIDE'), L('把注意力拆开来看', 'Open up attention'), L('每个注意力头和每个专家都画了出来', 'Every attention head and every expert drawn out')],
  f4: [L('04 导入', '04 IMPORT'), L('直接导入 PyTorch 代码', 'Import your PyTorch code'), L('只读取结构，不会执行代码', 'Read for its structure and never executed')],
  f5: [L('05 训练', '05 TRAIN'), L('在本地真正训练一遍', 'Train it for real on your own machine'), L('CPU 和 CUDA 都可以', 'On CPU or CUDA')],
  f6: [L('06 观测', '06 OBSERVE'), L('看看每一层算出了什么', 'See what every layer computes'), L('喂进一个样本，真实数值会画在方块上', 'Feed in one sample and its real values appear on the blocks')],
  f6b: [L('06 观测', '06 OBSERVE'), L('比较每一个注意力头', 'Compare every attention head'), L('同一分支的所有头放在一起看', 'All heads of a branch side by side')],
  f7: [L('07 导出', '07 EXPORT'), L('导出代码和图片', 'Export code and figures'), L('可运行的 PyTorch 代码和 PNG SVG 图片', 'Runnable PyTorch code plus PNG and SVG figures')]
};
const OPS = {
  op_conv2d: ['Conv2d', L('卷积', 'Convolution')], op_maxpool: ['MaxPool2d', L('池化', 'Pooling')], op_upsample: ['Upsample', L('上采样', 'Resampling')],
  op_convt: ['ConvTranspose2d', L('转置卷积', 'Transposed convolution')], op_conv3d: ['Conv3d', L('三维卷积', '3D convolution')], op_bilinear: ['Bilinear', L('双线性层', 'Bilinear layer')],
  op_transformer: ['Transformer', L('编码层 注意力加前馈', 'Encoder with attention and feed-forward')], op_mla: ['LatentAttention', L('多头潜在注意力', 'Multi-head latent attention')],
  op_moe: ['MoE', L('混合专家', 'Mixture of experts')], op_glu: ['GLU', L('门控线性单元', 'Gated linear unit')], op_unflatten: ['Unflatten', L('维度拆分', 'Axis split')],
  op_dropout2d: ['Dropout2d', L('通道随机失活', 'Channel dropout')], op_logsoftmax: ['LogSoftmax', L('对数概率', 'Log probabilities')],
  op_scale: ['Scale', L('缩放 1/√d', 'Scaling by 1/√d')], op_softmax: ['Softmax', L('注意力概率', 'Attention probabilities')], op_alpha: ['AlphaDropout', L('α 随机失活', 'Alpha dropout')],
  op_einsum: ['Einsum', L('爱因斯坦求和', 'Einstein summation')], op_mask: ['CausalMask', L('因果掩码', 'Causal mask')], op_multiply: ['Multiply', L('逐元素乘', 'Elementwise product')]
};

const CARDS = {
  k1: { seconds: 2.9, kind: 'kicker', text: L('从一个张量开始', 'It starts with a tensor') },
  k2: { seconds: 2.7, kind: 'kicker', text: L('一层一层搭起来', 'Built layer by layer') },
  k3: { seconds: 2.5, kind: 'kicker', text: L('看数据怎么流过去', 'Watch the data flow through') },
  k4: { seconds: 2.6, kind: 'kicker', text: L('从搭建到看懂', 'From building to understanding') },
  title: { seconds: 5.6, kind: 'title', text: 'TensorCraft3D', sub: L('本地运行的神经网络三维工作台', 'A local 3D workbench for neural networks') },
  ops: { seconds: 3.2, kind: 'counter', value: OPERATORS, unit: L('种算子', 'operators'), sub: L('从卷积到注意力，再到 MLA 和 MoE', 'From convolutions to attention, MLA and MoE'), eyebrow: L('02 算子', '02 OPERATORS') },
  end: { seconds: 6.0, kind: 'end', text: 'TensorCraft3D', sub: L('在你自己的电脑上运行', 'Runs on your own machine'), meta: RELEASE, url: 'github.com/AkiraAKI114514/TensorCraft3D' },
  mask: { seconds: 1 / FPS, kind: 'mask' }, shadow: { seconds: 1 / FPS, kind: 'shadow' }
};
for (const [key, [eyebrow, head, sub]] of Object.entries(FEATURES)) CARDS[key] = { seconds: 4.6, kind: 'feature', eyebrow, head, sub, place: key === 'f3' ? 'lower' : 'top' };
for (const [key, [name, family]] of Object.entries(OPS)) CARDS[key] = { seconds: 1.05, kind: 'op', name, family };

const FONT = lang === 'zh' ? `"Noto Sans SC", "Microsoft YaHei UI", sans-serif` : `"Segoe UI Variable Display", "Segoe UI Variable", "Segoe UI", sans-serif`;
const LATIN = `"Segoe UI Variable Display", "Segoe UI Variable", "Segoe UI", sans-serif`;
const HTML = `<!doctype html><html><head><meta charset="utf-8"><style>
html,body{margin:0;width:1920px;height:1080px;background:transparent;overflow:hidden}
#stage{position:absolute;inset:0;font-family:${FONT};color:#fff;-webkit-font-smoothing:antialiased}
.glow{text-shadow:0 2px 16px rgba(0,0,0,.55)}
</style></head><body><div id="stage"></div></body></html>`;

// Each kind builds its DOM once, then `frame(t)` sets every animated property for time t (seconds).
const smooth = (a, b, t) => { const x = Math.min(1, Math.max(0, (t - a) / (b - a))); return x * x * (3 - 2 * x); };
const outExpo = x => (x >= 1 ? 1 : 1 - Math.pow(2, -10 * x));
function setup(card) {
  const stage = document.getElementById('stage'); stage.innerHTML = '';
  const el = (tag, css, text, parent = stage) => { const e = document.createElement(tag); e.style.cssText = css; if (text !== undefined) e.textContent = text; parent.append(e); return e; };
  const T = card.seconds;
  const halo = (w, h) => el('div', `position:absolute;inset:0;background:radial-gradient(${w}% ${h}% at 50% 50%,rgba(3,8,14,.66) 0,rgba(3,8,14,.3) 50%,rgba(3,8,14,0) 78%)`);
  if (card.kind === 'kicker') {
    const scrim = halo(62, 44);
    const box = el('div', 'position:absolute;inset:0;display:flex;align-items:center;justify-content:center');
    const t = el('div', `font-size:${window.LANG === 'zh' ? 118 : 124}px;font-weight:650;letter-spacing:-0.01em;white-space:nowrap`, card.text, box); t.className = 'glow';
    return time => { const a = smooth(0.05, 0.9, time) * (1 - smooth(T - 0.55, T - 0.05, time)); const p = outExpo(Math.min(1, time / 1.1));
      scrim.style.opacity = a; t.style.opacity = a; t.style.filter = `blur(${(1 - p) * 4}px)`;
      t.style.transform = `translateY(${(1 - p) * 14}px)`; };
  }
  if (card.kind === 'title') {
    const scrim = halo(70, 58);
    const box = el('div', 'position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:26px');
    const t = el('div', `font-family:${window.LATIN};font-size:164px;font-weight:700;letter-spacing:-0.035em;color:#fff;text-shadow:0 3px 24px rgba(0,0,0,.5)`, card.text, box);
    const s = el('div', 'font-size:46px;font-weight:500;color:rgba(235,245,255,.86);letter-spacing:0.02em', card.sub, box); s.className = 'glow';
    return time => { const a = smooth(0.1, 1.2, time) * (1 - smooth(T - 0.6, T - 0.05, time)); const p = outExpo(Math.min(1, time / 1.4));
      scrim.style.opacity = a; t.style.opacity = a; t.style.transform = `translateY(${(1 - p) * 26}px) scale(${0.97 + 0.03 * p})`; t.style.backgroundPosition = `${100 - (time / T) * 100}% 0`;
      t.style.filter = `blur(${(1 - p) * 5}px)`;
      const b = smooth(1.0, 1.9, time) * (1 - smooth(T - 0.6, T - 0.05, time)); s.style.opacity = b; s.style.transform = `translateY(${(1 - smooth(1.0, 2.0, time)) * 18}px)`; };
  }
  if (card.kind === 'feature') {
    const top = card.place === 'top';
    const scrim = el('div', top ? 'position:absolute;inset:0;background:linear-gradient(180deg,rgba(4,10,18,.0) 0,rgba(4,10,18,0) 100%)' : 'position:absolute;inset:0;background:radial-gradient(120% 70% at 0% 100%,rgba(3,8,14,.78) 0,rgba(3,8,14,.35) 45%,rgba(3,8,14,0) 70%)');
    const box = el('div', top ? 'position:absolute;left:240px;top:58px;right:240px' : 'position:absolute;left:120px;bottom:118px;right:560px');
    const eyebrow = el('div', `font-family:${window.LATIN};font-size:22px;font-weight:650;letter-spacing:.34em;color:#7fe3d1;margin-bottom:${top ? 10 : 16}px`, card.eyebrow, box);
    if (window.LANG === 'zh') eyebrow.style.fontFamily = `"Noto Sans SC", ${window.LATIN}`;
    const head = el('div', `font-size:${top ? 58 : 74}px;font-weight:680;letter-spacing:-0.012em;line-height:1.08;white-space:nowrap`, card.head, box); head.className = 'glow';
    const sub = el('div', `font-size:${top ? 25 : 30}px;font-weight:450;color:rgba(225,238,248,.8);margin-top:${top ? 10 : 18}px`, card.sub, box); sub.className = 'glow';
    return time => { const out = 1 - smooth(T - 0.45, T - 0.05, time);
      scrim.style.opacity = smooth(0, 0.6, time) * out;
      [eyebrow, head, sub].forEach((e, i) => { const d = 0.08 + i * 0.14, p = outExpo(Math.min(1, Math.max(0, (time - d) / 0.9)));
        e.style.opacity = smooth(d, d + 0.5, time) * out; e.style.transform = `translateY(${(1 - p) * 22}px)`; e.style.filter = `blur(${(1 - p) * 6}px)`; }); };
  }
  if (card.kind === 'counter') {
    const scrim = el('div', 'position:absolute;inset:0;background:radial-gradient(70% 60% at 50% 50%,rgba(3,8,14,.62) 0,rgba(3,8,14,.25) 55%,rgba(3,8,14,0) 80%)');
    const box = el('div', 'position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center');
    const eyebrow = el('div', `font-family:${window.LANG === 'zh' ? '"Noto Sans SC",' : ''}${window.LATIN};font-size:24px;font-weight:650;letter-spacing:.34em;color:#7fe3d1;margin-bottom:6px`, card.eyebrow, box);
    const row = el('div', 'display:flex;align-items:baseline;gap:26px', undefined, box);
    const n = el('div', `font-family:${window.LATIN};font-size:250px;font-weight:700;letter-spacing:-0.04em;font-variant-numeric:tabular-nums;line-height:1`, '0', row); n.className = 'glow';
    const u = el('div', 'font-size:84px;font-weight:650', card.unit, row); u.className = 'glow';
    const sub = el('div', 'font-size:30px;font-weight:450;color:rgba(225,238,248,.82);margin-top:14px', card.sub, box); sub.className = 'glow';
    return time => { const out = 1 - smooth(T - 0.4, T - 0.03, time), a = smooth(0, 0.35, time) * out;
      scrim.style.opacity = a; eyebrow.style.opacity = a; n.style.opacity = a; u.style.opacity = smooth(0.2, 0.6, time) * out; sub.style.opacity = smooth(0.7, 1.2, time) * out;
      n.textContent = String(Math.round(card.value * outExpo(Math.min(1, time / 1.3)))); n.style.transform = `scale(${1 + 0.06 * (1 - outExpo(Math.min(1, time / 0.6)))})`; };
  }
  if (card.kind === 'op') {
    const box = el('div', 'position:absolute;left:110px;bottom:104px');
    const bar = el('div', 'width:0;height:5px;background:#5fd3c0;border-radius:3px;margin-bottom:18px', undefined, box);
    const n = el('div', `font-family:${window.LATIN};font-size:104px;font-weight:700;letter-spacing:-0.03em;line-height:1;white-space:nowrap`, card.name, box); n.className = 'glow';
    const f = el('div', 'font-size:30px;font-weight:500;color:rgba(200,236,248,.85);margin-top:12px;letter-spacing:0.02em', card.family, box); f.className = 'glow';
    return time => { const p = outExpo(Math.min(1, time / 0.32)), out = 1 - smooth(T - 0.16, T, time);
      bar.style.width = `${180 * p}px`; bar.style.opacity = out;
      n.style.opacity = p * out; n.style.transform = `translateX(${(1 - p) * -30}px)`;
      f.style.opacity = smooth(0.12, 0.35, time) * out; f.style.transform = `translateX(${(1 - smooth(0.12, 0.45, time)) * -30}px)`; };
  }
  if (card.kind === 'end') {
    const scrim = halo(72, 66);
    const box = el('div', 'position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:18px');
    const t = el('div', `font-family:${window.LATIN};font-size:150px;font-weight:700;letter-spacing:-0.035em;color:#fff;text-shadow:0 3px 24px rgba(0,0,0,.5)`, card.text, box);
    const s = el('div', 'font-size:38px;font-weight:500;color:rgba(235,245,255,.88)', card.sub, box); s.className = 'glow';
    const m = el('div', `font-family:${window.LATIN};font-size:26px;font-weight:600;letter-spacing:.3em;color:#7fe3d1;margin-top:20px`, card.meta, box);
    const u = el('div', `font-family:${window.LATIN};font-size:30px;font-weight:500;color:rgba(235,245,255,.75);letter-spacing:.02em`, card.url, box); u.className = 'glow';
    return time => { const p = outExpo(Math.min(1, time / 1.4));
      scrim.style.opacity = smooth(0, 1.0, time); t.style.opacity = smooth(0.1, 1.1, time); t.style.transform = `scale(${0.96 + 0.04 * p})`; t.style.filter = `blur(${(1 - p) * 5}px)`;
      [s, m, u].forEach((e, i) => { const d = 0.9 + i * 0.25; e.style.opacity = smooth(d, d + 0.6, time); e.style.transform = `translateY(${(1 - smooth(d, d + 0.8, time)) * 16}px)`; }); };
  }
  if (card.kind === 'mask') { el('div', 'position:absolute;left:0;top:0;width:1440px;height:810px;background:#fff;border-radius:22px'); return () => {}; }
  if (card.kind === 'shadow') { el('div', 'position:absolute;left:240px;top:222px;width:1440px;height:810px;border-radius:22px;background:transparent;box-shadow:0 30px 90px rgba(0,0,0,.65),0 0 0 1px rgba(160,220,255,.16),0 0 60px rgba(90,200,255,.12)'); return () => {}; }
  throw new Error(`unknown card kind ${card.kind}`);
}

(async () => {
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  // Laid out at 1920 x 1080 and rendered at PROMO_SCALE times the pixels (2 = 4K, the default), like the shots.
  const page = await browser.newPage({ viewport: { width: 1920, height: 1080 }, deviceScaleFactor: Number(process.env.PROMO_SCALE || 2) });
  await page.setContent(HTML); await page.evaluate(([lang, latin]) => { window.LANG = lang; window.LATIN = latin; }, [lang, LATIN]);
  await page.addScriptTag({ content: `window.smooth=${smooth};\nwindow.outExpo=${outExpo};\nwindow.setup=${setup};` });
  await page.evaluate(() => document.fonts.ready);
  const manifestPath = path.join(BUILD, 'titles', lang, 'manifest.json');
  const manifest = fs.existsSync(manifestPath) ? JSON.parse(fs.readFileSync(manifestPath, 'utf8')) : {};
  for (const [key, card] of Object.entries(CARDS)) {
    if (only.length && !only.includes(key)) continue;
    const dir = path.join(BUILD, 'titles', lang, key); fs.rmSync(dir, { recursive: true, force: true }); fs.mkdirSync(dir, { recursive: true });
    await page.evaluate(card => { window.frameFn = window.setup(card); }, card);
    await page.evaluate(() => document.fonts.ready);
    const frames = Math.max(1, Math.round(card.seconds * FPS));
    for (let i = 0; i < frames; i++) {
      await page.evaluate(t => window.frameFn(t), i / FPS);
      await page.screenshot({ path: path.join(dir, `${String(i + 1).padStart(4, '0')}.png`), omitBackground: true });
    }
    manifest[key] = { frames, seconds: frames / FPS };
    console.log(`${lang}/${key}: ${frames} frames`);
  }
  fs.mkdirSync(path.dirname(manifestPath), { recursive: true }); fs.writeFileSync(manifestPath, JSON.stringify(manifest, null, 1));
  await browser.close();
})();
