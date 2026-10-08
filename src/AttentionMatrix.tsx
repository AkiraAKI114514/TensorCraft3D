import { useId, useState } from 'react';
import type { AttentionMatrix as Matrix } from './inferenceTypes';
import { tensorNumber } from './inferenceTypes';
import './AttentionInspector.css';

export function matrixStep(value: number | null, limit: number, probability: boolean) {
  if (value === null || !Number.isFinite(value)) return null;
  const magnitude = probability ? Math.max(0, Math.min(1, value)) : limit > 0 ? Math.min(1, Math.abs(value) / limit) : 0;
  return Math.min(6, Math.floor(magnitude * 6));
}

export default function AttentionMatrix({ matrix, title, probability = false, columns = '特征', rows = 'Query' }: { matrix: Matrix; title: string; probability?: boolean; columns?: string; rows?: string }) {
  const id = useId().replaceAll(':', ''), [hover, setHover] = useState<[number, number] | null>(null), [focused, setFocused] = useState<[number, number] | null>(null), [texture, setTexture] = useState(false);
  const readout = focused ?? hover;
  const limit = probability ? 1 : Math.max(Math.abs(matrix.stats.min ?? 0), Math.abs(matrix.stats.max ?? 0));
  const cell = 24, left = 34, top = 25, width = Math.max(230, left + matrix.slice.columns * cell + 8), height = top + matrix.slice.rows * cell + 22;
  const label = (row: number, column: number) => `${title} · ${rows} ${matrix.rowStart + row} · ${columns} ${matrix.columnStart + column}：${tensorNumber(matrix.slice.values[row][column])}`;
  return <figure className={`attention-matrix ${texture ? 'textured' : ''}`}>
    <figcaption>{title}</figcaption>
    <p className="attention-caption">全矩阵 {`[${matrix.shape.join(', ')}]`} · 行 {matrix.rowStart}–{matrix.rowStart + matrix.slice.rows - 1} · {columns} {matrix.columnStart}–{matrix.columnStart + matrix.slice.columns - 1}{matrix.slice.truncated ? ' · 有界窗口，非完整矩阵' : ''}</p>
    <div className="attention-plot-scroll" tabIndex={0} role="region" aria-label={`${title} 热力图窗口`}>
      <svg viewBox={`0 0 ${width} ${height}`} style={{ minWidth: width }} role="img" aria-label={`${title} 实测矩阵热力图`}>
        <defs>{[0, 1, 2, 3, 4, 5, 6].flatMap(step => ['positive', 'negative'].map(sign => <pattern key={`${step}-${sign}`} id={`${id}-${step}-${sign}`} width={8 - step} height={8 - step} patternUnits="userSpaceOnUse" patternTransform={`rotate(${sign === 'negative' ? 135 : 45})`}><path d={`M0 0V${8 - step}`} className="attention-texture" /></pattern>))}</defs>
        {Array.from({ length: matrix.slice.columns }, (_, column) => <text key={column} x={left + column * cell + cell / 2} y="16" textAnchor="middle">{matrix.columnStart + column}</text>)}
        {matrix.slice.values.map((row, i) => <g key={i}><text x={left - 7} y={top + i * cell + 15} textAnchor="end">{matrix.rowStart + i}</text>{row.map((value, j) => {
          const step = matrixStep(value, limit, probability), sign = !probability && value !== null && value < 0 ? 'negative' : 'positive';
          const fill = step === null ? 'matrix-nonfinite' : !probability && value === 0 ? 'matrix-neutral' : `matrix-${sign}-${step}`;
          return <g key={j} tabIndex={0} role="img" aria-label={label(i, j)} onMouseEnter={() => setHover([i, j])} onMouseLeave={() => setHover(null)} onFocus={() => setFocused([i, j])} onBlur={() => setFocused(null)}>
            <rect x={left + j * cell} y={top + i * cell} width={cell} height={cell} fill="transparent" />
            <rect x={left + j * cell + 1} y={top + i * cell + 1} width={cell - 2} height={cell - 2} rx="2" className={`matrix-cell ${fill}`} />
            {step !== null && <rect x={left + j * cell + 1} y={top + i * cell + 1} width={cell - 2} height={cell - 2} fill={`url(#${id}-${step}-${sign})`} className="matrix-texture-layer" />}
            {step === null && <text x={left + j * cell + cell / 2} y={top + i * cell + 15} textAnchor="middle">×</text>}
            <title>{label(i, j)}</title>
          </g>;
        })}</g>)}
      </svg>
    </div>
    <div className={`attention-scale ${probability ? 'probability' : 'signed'}`} aria-label={probability ? '概率色阶：固定 0 到 1' : `对称色阶：${tensorNumber(-limit)} 到 ${tensorNumber(limit)}`}><span>{probability ? '0' : tensorNumber(-limit)}</span><i /><span>{probability ? '1' : tensorNumber(limit)}</span></div>
    <p className="tensor-hover" role="status">{readout ? label(...readout) : '悬停或键盘聚焦单元格查看精确值；支持横向滚动。'}</p>
    <label className="checkbox-label"><input type="checkbox" checked={texture} onChange={event => setTexture(event.target.checked)} />纹理辅助 · 密度表示幅度，方向表示正负</label>
    <details><summary>矩阵数值表 · 当前窗口</summary><div className="tensor-table-scroll"><table className="tensor-slice"><thead><tr><th>行 / {columns}</th>{Array.from({ length: matrix.slice.columns }, (_, column) => <th key={column}>{matrix.columnStart + column}</th>)}</tr></thead><tbody>{matrix.slice.values.map((row, i) => <tr key={i}><th>{matrix.rowStart + i}</th>{row.map((value, j) => <td key={j}>{tensorNumber(value)}</td>)}</tr>)}</tbody></table></div></details>
  </figure>;
}
