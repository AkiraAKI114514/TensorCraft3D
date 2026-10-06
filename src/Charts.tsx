import type { Metric } from './types';
export default function LossChart({ metrics }: { metrics: Metric[] }) {
  const width = 640, height = 140, left = 42, top = 15, bottom = 25;
  const finite = metrics.filter(m => Number.isFinite(m.trainLoss) && Number.isFinite(m.valLoss));
  const max = Math.max(1, ...finite.map(m => Math.max(m.trainLoss, m.valLoss))) * 1.08;
  const x = (i: number) => left + i / Math.max(1, metrics.length - 1) * (width - left - 15);
  const y = (v: number) => top + (1 - v / max) * (height - top - bottom);
  const points = (key: 'trainLoss' | 'valLoss') => metrics.map((m, i) => Number.isFinite(m[key]) ? `${x(i)},${y(m[key])}` : '').filter(Boolean).join(' ');
  return <svg className="loss-chart" viewBox={`0 0 ${width} ${height}`} role="img" aria-label="训练与验证损失曲线">
    {[0, 0.5, 1].map(t => <g key={t}><line x1={left} y1={y(max * t)} x2={width - 15} y2={y(max * t)} stroke="#e7ecee" strokeDasharray="3 4" /><text x={left - 8} y={y(max * t) + 3} textAnchor="end" fill="#87959b" fontSize="10">{(max * t).toFixed(1)}</text></g>)}
    {metrics.length > 0 && <><polyline points={points('trainLoss')} fill="none" stroke="#249d85" strokeWidth="2.2" /><polyline points={points('valLoss')} fill="none" stroke="#df866b" strokeWidth="2.2" /><text x={left} y={height - 6} fontSize="10" fill="#87959b">1</text><text x={width - 15} y={height - 6} textAnchor="end" fontSize="10" fill="#87959b">Epoch {metrics.at(-1)?.epoch}</text></>}
    {!metrics.length && <text x={width / 2} y={height / 2} textAnchor="middle" fill="#8c9da4" fontSize="12">等待训练指标</text>}
  </svg>;
}
