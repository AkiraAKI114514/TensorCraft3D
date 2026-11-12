import { useMemo, useState, type KeyboardEvent, type MouseEvent } from 'react';
import { useI18n } from './i18n';
import { formatMetricValue, layerSeries, samplePoints, type LayerPoint } from './layerMetricSeries';
import type { Metric } from './types';
import './LayerMetrics.css';

const WIDTH = 240, LEFT = 36, RIGHT = 8, TOP = 10, BASE = 78, PLOT = WIDTH - LEFT - RIGHT;

function LayerChart({ title, points, percent = false }: { title: string; points: LayerPoint[]; percent?: boolean }) {
  const { t } = useI18n();
  const [hovered, setHovered] = useState<number | null>(null);
  const shown = samplePoints(points), last = shown.length - 1;
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
  const active = Math.min(hovered ?? last, last);
  const move = (event: MouseEvent<HTMLDivElement>) => {
    const box = event.currentTarget.getBoundingClientRect();
    if (!box.width || last < 1) { setHovered(last); return; }
    const ratio = ((event.clientX - box.left) / box.width * WIDTH - LEFT) / PLOT;
    setHovered(Math.min(last, Math.max(0, Math.round(ratio * last))));
  };
  // 整张图只占一个 Tab 停靠点，方向键逐轮移动：上百个可聚焦的点会把键盘用户困在图里。
  const key = (event: KeyboardEvent<HTMLDivElement>) => {
    const next = event.key === 'ArrowLeft' ? active - 1 : event.key === 'ArrowRight' ? active + 1 : event.key === 'Home' ? 0 : event.key === 'End' ? last : null;
    if (next === null) return;
    event.preventDefault(); setHovered(Math.min(last, Math.max(0, next)));
  };
  const line = shown.map((point, index) => `${x(index)},${y(point.mean)}`).join(' ');
  const band = `M${shown.map((point, index) => `${x(index)} ${y(point.max)}`).join('L')}L${shown.map((point, index) => `${x(index)} ${y(point.min)}`).reverse().join('L')}Z`;
  return <figure className="layer-metric-chart">
    <figcaption>{title}</figcaption>
    <div className="layer-metric-plot" role="slider" tabIndex={0} aria-label={title} aria-valuemin={shown[0].epoch} aria-valuemax={shown[last].epoch} aria-valuenow={shown[active].epoch} aria-valuetext={label(active)} onMouseMove={move} onMouseLeave={() => setHovered(null)} onKeyDown={key} onBlur={() => setHovered(null)}>
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
  if (!gradients.length && !dead.length) {
    if (!eligible) return null;
    return <div className="inspector-section layer-metrics"><h3>{t('本层训练指标')}</h3><p className="muted-text">{t('真实训练后，这里显示本层的梯度范数与 ReLU 零激活比例。')}</p></div>;
  }
  const instances = Math.max(...gradients.map(point => point.count), ...dead.map(point => point.count));
  const measured = metrics.at(-1)?.source === 'training';
  return <div className="inspector-section layer-metrics">
    <h3>{t('本层训练指标')}<span>{measured ? t('实测') : t('导入指标')}</span></h3>
    {!measured && <p className="layer-metric-note">{t('这些指标没有在当前计算图上测量，仅按层 ID 对应。')}</p>}
    {instances > 1 && <p className="layer-metric-note">{t('{count} 个展开层 · 曲线为均值，色带为最小值到最大值', { count: instances })}</p>}
    {gradients.length > 0 && <LayerChart title={t('梯度范数 · 每轮批次均值，裁剪前')} points={gradients} />}
    {dead.length > 0 && <LayerChart title={t('ReLU 零激活比例 · 每轮批次均值')} points={dead} percent />}
  </div>;
}
