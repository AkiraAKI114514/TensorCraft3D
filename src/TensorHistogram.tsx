import { useState } from 'react';
import type { TensorSnapshot } from './inferenceTypes';
import { tensorNumber } from './inferenceTypes';

export default function TensorHistogram({ histogram }: { histogram: TensorSnapshot['histogram'] }) {
  const [hovered, setHovered] = useState<number | null>(null), [texture, setTexture] = useState(false);
  const max = Math.max(1, ...histogram.counts), width = 240, left = 25, plot = width - left - 6;
  const step = plot / histogram.counts.length;
  const label = (i: number) => `${tensorNumber(histogram.edges[i])} – ${tensorNumber(histogram.edges[i + 1])}：${histogram.counts[i]} 个元素`;
  return <figure className={`tensor-histogram ${texture ? 'textured' : ''}`}>
    <figcaption>有限元素分布 · 全张量</figcaption>
    <svg viewBox={`0 0 ${width} 126`} role="img" aria-label="全张量有限元素直方图">
      <defs><pattern id="tensor-bin-texture" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><rect width="6" height="6" className="bin-fill" /><path d="M0 0V6" className="bin-texture" /></pattern></defs>
      <line x1={left} x2={width - 6} y1="103" y2="103" className="hist-axis" />
      <line x1={left} x2={width - 6} y1="18" y2="18" className="hist-grid" />
      <text x={left - 4} y="21" textAnchor="end">{new Intl.NumberFormat('en', { notation: 'compact', maximumFractionDigits: 1 }).format(max)}</text><text x={left - 4} y="104" textAnchor="end">0</text>
      <text x={left} y="119">{tensorNumber(histogram.edges[0])}</text>
      <text x={width - 6} y="119" textAnchor="end">{tensorNumber(histogram.edges.at(-1) ?? null)}</text>
      {histogram.counts.map((count, i) => {
        const height = count / max * 85, x = left + i * step + 1, barWidth = Math.min(24, step - 2), y = 103 - height;
        return <g key={i} tabIndex={0} role="img" aria-label={label(i)} onMouseEnter={() => setHovered(i)} onMouseLeave={() => setHovered(null)} onFocus={() => setHovered(i)} onBlur={() => setHovered(null)}>
          <rect x={left + i * step} y="15" width={step} height="88" fill="transparent" />
          <path d={`M${x} 103V${y + Math.min(4, height)}Q${x} ${y} ${x + Math.min(4, barWidth / 2)} ${y}H${x + barWidth - Math.min(4, barWidth / 2)}Q${x + barWidth} ${y} ${x + barWidth} ${y + Math.min(4, height)}V103Z`} className="hist-bar" />
          <title>{label(i)}</title>
        </g>;
      })}
    </svg>
    <p className="tensor-hover" role="status">{hovered === null ? '悬停或聚焦柱条查看区间与数量' : label(hovered)}</p>
    <label className="checkbox-label"><input type="checkbox" checked={texture} onChange={e => setTexture(e.target.checked)} />纹理辅助</label>
    <details><summary>分布数据表</summary><table><thead><tr><th>区间</th><th>数量</th></tr></thead><tbody>{histogram.counts.map((count, i) => <tr key={i}><td>{tensorNumber(histogram.edges[i])} – {tensorNumber(histogram.edges[i + 1])}</td><td>{count}</td></tr>)}</tbody></table></details>
  </figure>;
}
