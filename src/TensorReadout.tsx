import TensorHistogram from './TensorHistogram';
import TensorSemanticView from './TensorSemanticView';
import { tensorNumber, type TensorSnapshot } from './inferenceTypes';
import { useI18n } from './i18n';

type Props = {
  title: string;
  snapshot: TensorSnapshot;
  /** Identifies the run or file and the tensor; a new one remounts the views. */
  viewKey: string;
  /** A weight is labelled as a kernel or matrix, not as a feature map. */
  subject?: 'activation' | 'parameter';
  /** Accessible name of the value table; each readout on the panel needs its own. */
  tableLabel: string;
  disabled: boolean;
  rerun: boolean;
  onSelectChannel: (channel: number) => void;
};

/**
 * Everything the inspector shows for one measured tensor: shape and statistics, the finite
 * distribution, the semantic view of its window and the window as a table. Activations and
 * weights read the same way.
 */
export default function TensorReadout({ title, snapshot, viewKey, subject = 'activation', tableLabel, disabled, rerun, onSelectChannel }: Props) {
  const { t } = useI18n();
  return <>
    <div className="inspector-section"><h3>{title}</h3>
      <dl className="tensor-stats">
        <div><dt>Shape</dt><dd>{`[${snapshot.shape.join(', ')}]`}</dd></div><div><dt>Dtype</dt><dd>{snapshot.dtype}</dd></div>
        <div><dt>{t('元素数')}</dt><dd>{snapshot.elements}</dd></div><div><dt>{t('非有限元素')}</dt><dd>{snapshot.nonFiniteCount}</dd></div>
        <div><dt>Min</dt><dd>{t(tensorNumber(snapshot.stats.min))}</dd></div><div><dt>Max</dt><dd>{t(tensorNumber(snapshot.stats.max))}</dd></div>
        <div><dt>Mean</dt><dd>{t(tensorNumber(snapshot.stats.mean))}</dd></div><div><dt>{t('Std · 总体')}</dt><dd>{t(tensorNumber(snapshot.stats.std))}</dd></div>
      </dl>
      {snapshot.nonFiniteCount > 0 && <p className="tensor-stale">{t('存在 NaN/Inf；统计和分布只包含有限元素，切片中以“非有限值”标注。')}</p>}
    </div>
    <TensorHistogram key={`${viewKey}histogram`} histogram={snapshot.histogram} />
    <TensorSemanticView key={`${viewKey}semantic`} snapshot={snapshot} subject={subject} disabled={disabled} rerun={rerun} onSelectChannel={onSelectChannel} />
    <div className="inspector-section"><h3>{t('张量切片')}<span>{t('最多 16 × 16')}</span></h3>
      <p className="tensor-note">{t('固定前置轴，显示末尾{count}维 {shape} 的左上区域。', { count: snapshot.slice.shape.length, shape: `[${snapshot.slice.shape.join(', ')}]` })}{snapshot.slice.truncated ? t('已截断，表格不是完整张量。') : t('该切片已完整显示。')}</p>
      <div className="tensor-table-scroll" tabIndex={0} role="region" aria-label={tableLabel}>
        <table className="tensor-slice"><thead><tr><th>{t('索引')}</th>{Array.from({ length: snapshot.slice.columns }, (_, column) => <th key={column}>{column}</th>)}</tr></thead><tbody>{snapshot.slice.values.map((row, i) => <tr key={i}><th>{i}</th>{row.map((value, j) => <td key={j} title={`[${[...snapshot.slice.indices, ...(snapshot.slice.shape.length > 1 ? [i] : []), j].join(', ')}] = ${value === null ? t('非有限值') : value}`}>{t(tensorNumber(value))}</td>)}</tr>)}</tbody></table>
      </div>
    </div>
  </>;
}
