import { useMemo, useState, type KeyboardEvent, type MouseEvent } from 'react';
import { useI18n } from './i18n';
import { distributionKeys, distributionSeries, formatMetricValue, layerSeries, samplePoints, type DistributionPoint, type LayerPoint } from './layerMetricSeries';
import type { Metric } from './types';
import './LayerMetrics.css';

const WIDTH = 240, LEFT = 36, RIGHT = 8, TOP = 10, BASE = 78, PLOT = WIDTH - LEFT - RIGHT;
const HISTOGRAM_TOP = 4, HISTOGRAM_BASE = 46, HISTOGRAM_HEIGHT = 58;
// Weight distribution bands, outside in: min–max, then the middle 87%, 68% and 38% of the values.
const BANDS: [number, number][] = [[0, 8], [1, 7], [2, 6], [3, 5]];
const MAX_DISTRIBUTION_CHOICES = 256;

/** 整张图只占一个 Tab 停靠点，方向键逐轮移动：上百个可聚焦的点会把键盘用户困在图里。 */
function useEpochScrub(last: number) {
  const [hovered, setHovered] = useState<number | null>(null);
  const active = Math.min(hovered ?? last, last);
  const move = (event: MouseEvent<HTMLDivElement>) => {
    const box = event.currentTarget.getBoundingClientRect();
    if (!box.width || last < 1) { setHovered(last); return; }
    const ratio = ((event.clientX - box.left) / box.width * WIDTH - LEFT) / PLOT;
    setHovered(Math.min(last, Math.max(0, Math.round(ratio * last))));
  };
  const key = (event: KeyboardEvent<HTMLDivElement>) => {
    const next = event.key === 'ArrowLeft' ? active - 1 : event.key === 'ArrowRight' ? active + 1 : event.key === 'Home' ? 0 : event.key === 'End' ? last : null;
    if (next === null) return;
    event.preventDefault(); setHovered(Math.min(last, Math.max(0, next)));
  };
  return { hovered, active, move, key, leave: () => setHovered(null) };
}

function LayerChart({ title, points, percent = false }: { title: string; points: LayerPoint[]; percent?: boolean }) {
  const { t } = useI18n();
  const shown = samplePoints(points), last = shown.length - 1;
  const { hovered, active, move, key, leave } = useEpochScrub(last);
  const peak = Math.max(...shown.map(point => point.max));
  // 两个量的量纲不同，所以各用一张图而不是共用一根纵轴；比例固定到 100%，范数按峰值。
  const ceiling = percent ? 1 : peak > 0 ? peak : 1;
  const x = (index: number) => LEFT + (last > 0 ? index / last * PLOT : PLOT / 2);
  const y = (value: number) => BASE - Math.min(1, Math.max(0, value / ceiling)) * (BASE - TOP);
  const banded = shown.some(point => point.count > 1);
  const label = (index: number) => {
    const point = shown[index];
    return point.count > 1
      ? t('第 {epoch} 轮：均值 {value}，范围 {min} – {max}，{count} 个展开层', { epoch: point.epoch, value: formatMetricValue(point.mean, percent), min: formatMetricValue(point.min, percent), max: formatMetricValue(point.max, percent), count: point.count })
      : t('第 {epoch} 轮：{value}', { epoch: point.epoch, value: formatMetricValue(point.mean, percent) });
  };
  const line = shown.map((point, index) => `${x(index)},${y(point.mean)}`).join(' ');
  const band = `M${shown.map((point, index) => `${x(index)} ${y(point.max)}`).join('L')}L${shown.map((point, index) => `${x(index)} ${y(point.min)}`).reverse().join('L')}Z`;
  return <figure className="layer-metric-chart">
    <figcaption>{title}</figcaption>
    <div className="layer-metric-plot" role="slider" tabIndex={0} aria-label={title} aria-valuemin={shown[0].epoch} aria-valuemax={shown[last].epoch} aria-valuenow={shown[active].epoch} aria-valuetext={label(active)} onMouseMove={move} onMouseLeave={leave} onKeyDown={key} onBlur={leave}>
      <svg viewBox={`0 0 ${WIDTH} 96`} aria-hidden="true">
        <line x1={LEFT} x2={WIDTH - RIGHT} y1={TOP} y2={TOP} className="lm-grid" />
        <line x1={LEFT} x2={WIDTH - RIGHT} y1={BASE} y2={BASE} className="lm-axis" />
        <text x={LEFT - 4} y={TOP + 3} textAnchor="end">{formatMetricValue(ceiling, percent)}</text><text x={LEFT - 4} y={BASE + 1} textAnchor="end">0</text>
        <text x={LEFT} y="92">{shown[0].epoch}</text><text x={WIDTH - RIGHT} y="92" textAnchor="end">Epoch {shown[last].epoch}</text>
        {banded && last > 0 && <path d={band} className="lm-band" />}
        {last > 0 && <polyline points={line} className="lm-line" />}
        {hovered !== null && <line x1={x(active)} x2={x(active)} y1={TOP} y2={BASE} className="lm-crosshair" />}
        <circle cx={x(active)} cy={y(shown[active].mean)} r="4" className="lm-marker" />
      </svg>
    </div>
    <p className="layer-metric-readout">{label(active)}</p>
    {shown.length < points.length && <p className="layer-metric-note">{t('曲线与数据表等距抽样显示 {shown} / {total} 轮', { shown: shown.length, total: points.length })}</p>}
    <details><summary>{t('指标数据表')}</summary><table><thead><tr><th>{t('轮次')}</th>{banded ? <><th>{t('均值')}</th><th>{t('最小')}</th><th>{t('最大')}</th></> : <th>{t('数值')}</th>}</tr></thead><tbody>{shown.map(point => <tr key={point.epoch}><td>{point.epoch}</td><td>{formatMetricValue(point.mean, percent)}</td>{banded && <><td>{formatMetricValue(point.min, percent)}</td><td>{formatMetricValue(point.max, percent)}</td></>}</tr>)}</tbody></table></details>
  </figure>;
}

/** Equal bins from the minimum to the maximum, as densities so epochs with other ranges compare. */
function bins(point: DistributionPoint) {
  const total = point.counts.reduce((sum, count) => sum + count, 0) || 1;
  const low = point.quantiles[0], width = (point.quantiles[8] - low) / point.counts.length;
  return point.counts.map((count, index) => ({ start: low + index * width, end: low + (index + 1) * width, density: width > 0 ? count / total / width : 0 }));
}

/**
 * The active epoch's histogram as bars, over the first epoch's as a dashed outline, on one value
 * axis. Heights are densities, so a distribution that spreads out flattens instead of growing.
 * A layer whose weights are all equal has no width to spread over and is drawn as one spike.
 */
function Histogram({ point, first }: { point: DistributionPoint; first: DistributionPoint | null }) {
  const both = first ? [point, first] : [point];
  const low = Math.min(...both.map(entry => entry.quantiles[0])), high = Math.max(...both.map(entry => entry.quantiles[8]));
  const x = (value: number) => LEFT + (high > low ? (value - low) / (high - low) : 0.5) * PLOT;
  const peak = Math.max(...both.flatMap(entry => bins(entry).map(bin => bin.density))) || 1;
  const y = (density: number) => HISTOGRAM_BASE - density / peak * (HISTOGRAM_BASE - HISTOGRAM_TOP);
  const spike = (entry: DistributionPoint) => entry.quantiles[8] === entry.quantiles[0];
  const outline = (entry: DistributionPoint) => spike(entry)
    ? `M${x(entry.quantiles[0])} ${HISTOGRAM_BASE}V${HISTOGRAM_TOP}`
    : `M${x(entry.quantiles[0])} ${HISTOGRAM_BASE}${bins(entry).map(bin => `V${y(bin.density)}H${x(bin.end)}`).join('')}V${HISTOGRAM_BASE}`;
  return <svg className="lm-histogram" viewBox={`0 0 ${WIDTH} ${HISTOGRAM_HEIGHT}`} aria-hidden="true">
    <line x1={LEFT} x2={WIDTH - RIGHT} y1={HISTOGRAM_BASE} y2={HISTOGRAM_BASE} className="lm-axis" />
    {spike(point)
      ? <rect x={x(point.quantiles[0]) - 1.5} width={3} y={HISTOGRAM_TOP} height={HISTOGRAM_BASE - HISTOGRAM_TOP} className="lm-hist" />
      : bins(point).map((bin, index) => <rect key={index} x={x(bin.start)} width={Math.max(0.5, x(bin.end) - x(bin.start) - 0.5)} y={y(bin.density)} height={HISTOGRAM_BASE - y(bin.density)} className="lm-hist" />)}
    {first && <path d={outline(first)} className="lm-hist-first" />}
    <text x={LEFT} y={HISTOGRAM_HEIGHT - 2}>{formatMetricValue(low)}</text>
    <text x={WIDTH - RIGHT} y={HISTOGRAM_HEIGHT - 2} textAnchor="end">{formatMetricValue(high)}</text>
  </svg>;
}

/**
 * A layer's weights at the end of every epoch, the way TensorBoard's distribution view draws
 * them: nested quantile bands around the median over the epochs, then the chosen epoch's
 * histogram against the first epoch's. One tab stop scrubs through the epochs.
 */
function DistributionChart({ title, points }: { title: string; points: DistributionPoint[] }) {
  const { t } = useI18n();
  const shown = samplePoints(points), last = shown.length - 1;
  const { hovered, active, move, key, leave } = useEpochScrub(last);
  const low = Math.min(...shown.map(point => point.quantiles[0])), high = Math.max(...shown.map(point => point.quantiles[8]));
  // Equal weights have no spread; centre them in a range around their value.
  const pad = high > low ? 0 : Math.max(Math.abs(high), 1) / 2, bottom = low - pad, top = high + pad;
  const x = (index: number) => LEFT + (last > 0 ? index / last * PLOT : PLOT / 2);
  const y = (value: number) => BASE - (value - bottom) / (top - bottom) * (BASE - TOP);
  const band = (lower: number, upper: number) => `M${shown.map((point, index) => `${x(index)} ${y(point.quantiles[upper])}`).join('L')}L${shown.map((point, index) => `${x(index)} ${y(point.quantiles[lower])}`).reverse().join('L')}Z`;
  const median = shown.map((point, index) => `${x(index)},${y(point.quantiles[4])}`).join(' ');
  const label = (index: number) => {
    const { epoch, quantiles } = shown[index];
    return t('第 {epoch} 轮：中位数 {median}，68% 落在 {low} – {high}，范围 {min} – {max}', { epoch, median: formatMetricValue(quantiles[4]), low: formatMetricValue(quantiles[2]), high: formatMetricValue(quantiles[6]), min: formatMetricValue(quantiles[0]), max: formatMetricValue(quantiles[8]) });
  };
  const current = shown[active], first = active > 0 ? shown[0] : null;
  return <figure className="layer-metric-chart">
    <figcaption>{title}</figcaption>
    <div className="layer-metric-plot" role="slider" tabIndex={0} aria-label={title} aria-valuemin={shown[0].epoch} aria-valuemax={shown[last].epoch} aria-valuenow={current.epoch} aria-valuetext={label(active)} onMouseMove={move} onMouseLeave={leave} onKeyDown={key} onBlur={leave}>
      <svg viewBox={`0 0 ${WIDTH} 96`} aria-hidden="true">
        <line x1={LEFT} x2={WIDTH - RIGHT} y1={TOP} y2={TOP} className="lm-grid" />
        <line x1={LEFT} x2={WIDTH - RIGHT} y1={BASE} y2={BASE} className="lm-axis" />
        <text x={LEFT - 4} y={TOP + 3} textAnchor="end">{formatMetricValue(top)}</text><text x={LEFT - 4} y={BASE + 1} textAnchor="end">{formatMetricValue(bottom)}</text>
        <text x={LEFT} y="92">{shown[0].epoch}</text><text x={WIDTH - RIGHT} y="92" textAnchor="end">Epoch {shown[last].epoch}</text>
        {BANDS.map(([lower, upper], level) => last > 0
          ? <path key={level} d={band(lower, upper)} className={`lm-dist lm-dist-${level}`} />
          : <rect key={level} x={x(0) - 4} width={8} y={y(current.quantiles[upper])} height={Math.max(0.5, y(current.quantiles[lower]) - y(current.quantiles[upper]))} className={`lm-dist lm-dist-${level}`} />)}
        {last > 0 && <polyline points={median} className="lm-line" />}
        {hovered !== null && <line x1={x(active)} x2={x(active)} y1={TOP} y2={BASE} className="lm-crosshair" />}
        <circle cx={x(active)} cy={y(current.quantiles[4])} r="4" className="lm-marker" />
      </svg>
    </div>
    <p className="layer-metric-readout">{label(active)}</p>
    <p className="layer-metric-note">{t('色带由外到内是最小–最大与居中 87%、68%、38% 的权重，线为中位数。')}</p>
    <p className="layer-metric-note">{first ? t('第 {epoch} 轮的直方图 · 虚线为第 {first} 轮', { epoch: current.epoch, first: first.epoch }) : t('第 {epoch} 轮的直方图', { epoch: current.epoch })}</p>
    <Histogram point={current} first={first} />
    {shown.length < points.length && <p className="layer-metric-note">{t('曲线与数据表等距抽样显示 {shown} / {total} 轮', { shown: shown.length, total: points.length })}</p>}
    <details><summary>{t('指标数据表')}</summary><div className="layer-metric-table"><table><thead><tr><th>{t('轮次')}</th><th>{t('中位数')}</th><th>{t('68% 区间')}</th><th>{t('最小')}</th><th>{t('最大')}</th><th>{t('标准差')}</th></tr></thead><tbody>{shown.map(point => <tr key={point.epoch}><td>{point.epoch}</td><td>{formatMetricValue(point.quantiles[4])}</td><td>{`${formatMetricValue(point.quantiles[2])} – ${formatMetricValue(point.quantiles[6])}`}</td><td>{formatMetricValue(point.quantiles[0])}</td><td>{formatMetricValue(point.quantiles[8])}</td><td>{formatMetricValue(point.std)}</td></tr>)}</tbody></table></div></details>
  </figure>;
}

/**
 * 选中层在训练中的实测指标。后端每轮都会推送逐层梯度范数与 ReLU 零激活比例，
 * 这里把它们落到被选中的层上；三维场景里的配色与粒子仍是示意，所以实测值只出现在面板里。
 *
 * 任何图编辑都会清空训练指标，所以实测曲线总是对应屏幕上的这张图；只有导入的指标
 * 没有在这张图上测过，它们按层 ID 对应并在面板里注明。
 */
export default function LayerMetrics({ metrics, keys, eligible }: { metrics: Metric[]; keys: string[]; eligible: boolean }) {
  const { t } = useI18n();
  const gradients = useMemo(() => layerSeries(metrics, keys, 'layerGradients'), [metrics, keys]);
  const dead = useMemo(() => layerSeries(metrics, keys, 'deadRelu'), [metrics, keys]);
  // Quantiles of different layers do not merge, so several expanded layers offer a choice.
  const distributed = useMemo(() => distributionKeys(metrics, keys).slice(0, MAX_DISTRIBUTION_CHOICES), [metrics, keys]);
  const [chosen, setChosen] = useState<string | null>(null);
  const distributionKey = chosen !== null && distributed.includes(chosen) ? chosen : distributed[0];
  const distribution = useMemo(() => distributionKey === undefined ? [] : distributionSeries(metrics, distributionKey), [metrics, distributionKey]);
  if (!gradients.length && !dead.length && !distribution.length) {
    if (!eligible) return null;
    return <div className="inspector-section layer-metrics"><h3>{t('本层训练指标')}</h3><p className="muted-text">{t('真实训练后，这里显示本层的梯度范数、权重分布与 ReLU 零激活比例。')}</p></div>;
  }
  const instances = Math.max(...gradients.map(point => point.count), ...dead.map(point => point.count));
  const measured = metrics.at(-1)?.source === 'training';
  return <div className="inspector-section layer-metrics">
    <h3>{t('本层训练指标')}<span>{measured ? t('实测') : t('导入指标')}</span></h3>
    {!measured && <p className="layer-metric-note">{t('这些指标没有在当前计算图上测量，仅按层 ID 对应。')}</p>}
    {instances > 1 && <p className="layer-metric-note">{t('{count} 个展开层 · 曲线为均值，色带为最小值到最大值', { count: instances })}</p>}
    {gradients.length > 0 && <LayerChart title={t('梯度范数 · 每轮批次均值，裁剪前')} points={gradients} />}
    {dead.length > 0 && <LayerChart title={t('ReLU 零激活比例 · 每轮批次均值')} points={dead} percent />}
    {distribution.length > 0 && <>
      {distributed.length > 1 && <label className="field-label layer-metric-key">{t('权重分布 · 展开层')}<select aria-label={t('查看权重分布的展开层')} value={distributionKey} onChange={event => setChosen(event.target.value)}>{distributed.map(key => <option key={key} value={key}>{key}</option>)}</select></label>}
      <DistributionChart key={distributionKey} title={t('权重分布 · 每轮结束时，不含偏置')} points={distribution} />
    </>}
  </div>;
}
