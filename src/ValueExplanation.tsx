import type { ValueExplanation } from './inferenceTypes';
import { tensorNumber } from './inferenceTypes';
import { useI18n } from './i18n';

/** Layers whose output values the backend splits into terms (`EXPLAINED_RULES` in inference.py). */
export const EXPLAINED_OPS = new Set([
  'Linear', 'Conv1d', 'Conv2d', 'Conv3d',
  'AvgPool1d', 'AvgPool2d', 'AvgPool3d', 'AdaptiveAvgPool1d', 'AdaptiveAvgPool2d', 'AdaptiveAvgPool3d',
  'MaxPool1d', 'MaxPool2d', 'MaxPool3d', 'AdaptiveMaxPool1d', 'AdaptiveMaxPool2d', 'AdaptiveMaxPool3d',
]);

/** Where to start: the first channel or feature, and the middle of every spatial axis. */
export function defaultExplainIndex(op: string, shape: number[]) {
  return shape.slice(1).map((size, axis) => op !== 'Linear' && axis > 0 ? Math.floor((size - 1) / 2) : 0);
}

/** A name for each output axis after the batch axis; an empty name falls back to its number. */
export function explainAxes(op: string, rank: number) {
  const count = Math.max(0, rank - 1);
  if (op === 'Linear') return Array.from({ length: count }, (_, axis) => axis === count - 1 ? '输出特征' : count === 2 ? '输出位置' : '');
  return ['输出通道', ...([[], ['输出位置'], ['输出行', '输出列'], ['输出深度', '输出行', '输出列']][count - 1] ?? Array.from({ length: count - 1 }, () => ''))];
}

const at = (index: number[]) => `[${index.join(', ')}]`;

/**
 * One output value as the measured terms it is made of: Σ x·w + b for a Linear or convolution
 * layer, Σ x ÷ divisor for average pooling, the largest window value for max pooling. The terms
 * come from the input and weights of the same run; the largest are listed, the rest summed, and
 * the float64 total is set against the float32 value the layer produced.
 */
export default function ValueExplanationView({ explanation, layer, input }: { explanation: ValueExplanation; layer: string; input: string }) {
  const { t } = useI18n();
  const { rule, terms, rest, channels, channelRest } = explanation;
  const weighted = rule === 'weighted-sum', max = rule === 'max';
  const number = (value: number | null) => t(tensorNumber(value));
  const formula = weighted
    ? explanation.bias === null ? t('Σ {count} 项 x·w', { count: explanation.count }) : t('Σ {count} 项 x·w + b', { count: explanation.count })
    : max ? t('max（{count} 个窗口值）', { count: explanation.count }) : t('Σ {count} 项 x ÷ {divisor}', { count: explanation.count, divisor: explanation.divisor ?? 1 });
  const difference = explanation.value !== null && explanation.recomputed !== null ? explanation.recomputed - explanation.value : null;
  const largest = Math.max(0, ...terms.map(term => Math.abs(term.term ?? 0)));
  // The panel is narrow, so the contribution comes first; max pooling's term is x itself.
  const columns = weighted ? 4 : max ? 2 : 3;
  const filler = columns > 2 ? <td colSpan={columns - 2} /> : null;
  return <section className="value-explanation" aria-label={t('数值解释')}>
    <h3>{t('{layer} · 输出 {index} 的组成', { layer, index: at(explanation.index) })}<span>{explanation.op}</span></h3>
    <p className="value-formula"><code>{`y${at(explanation.index)} = ${formula}`}</code></p>
    <dl className="tensor-stats">
      <div><dt>{t('实测输出 · float32')}</dt><dd>{number(explanation.value)}</dd></div>
      <div><dt>{t('由各项重算 · float64')}</dt><dd>{number(explanation.recomputed)}</dd></div>
      <div><dt>{t('重算 − 实测')}</dt><dd>{number(difference)}</dd></div>
      {explanation.bias !== null && <div><dt>{t('偏置 b')}</dt><dd>{number(explanation.bias)}</dd></div>}
    </dl>
    <p className="tensor-note">{t('x 是 {input} 在这次运行中的实测输出 {shape}；', { input, shape: at(explanation.inputShape) })}{weighted ? t('w 是同一个模型里这一层的权重。') : max ? t('最大池化没有权重，输出就是窗口里最大的值。') : t('平均池化没有权重，每项是 x 除以 PyTorch 计入的窗口大小。')}{t('各项用 float64 重算，与 float32 实测值可能有舍入差异。')}</p>
    {explanation.padded > 0 && <p className="tensor-note">{max ? t('窗口有 {count} 个位置落在边界外，按 −∞ 处理，不参与比较。', { count: explanation.padded }) : weighted ? t('其中 {count} 项落在补零区域，x 为 0。', { count: explanation.padded }) : t('窗口有 {count} 个位置落在补零区域，按 PyTorch 默认计入分母。', { count: explanation.padded })}</p>}
    {explanation.instances > 1 && <p className="tensor-note">{t('折叠层共 {count} 个实例，这里拆的是最后一个（编号 {instance}），它的输入是前一个实例的输出。', { count: explanation.instances, instance: explanation.instance })}</p>}
    {explanation.owner && <p className="tensor-note">{t('这一层共享 {owner} 的权重。', { owner: explanation.owner })}</p>}
    <div className="tensor-table-scroll"><table className="tensor-slice value-terms" aria-label={max ? t('窗口里最大的值') : t('贡献最大的项')}>
      <thead><tr><th>{t('输入位置')}</th><th>{weighted ? 'x·w' : max ? t('窗口值') : 'x ÷ n'}</th>{!max && <th>x</th>}{weighted && <th>w</th>}</tr></thead>
      <tbody>{terms.map((term, row) => {
        const share = largest > 0 && term.term !== null ? Math.abs(term.term) / largest * 100 : 0;
        return <tr key={row} className={max && row === 0 ? 'value-winner' : undefined}>
          <th>{term.input ? at(term.input) : t('补零')}{max && row === 0 ? ` · ${t('最大')}` : ''}</th>
          <td className="value-term">{number(term.term)}<span className="value-bar" data-sign={(term.term ?? 0) < 0 ? 'negative' : 'positive'} style={{ width: `${share}%` }} aria-hidden="true" /></td>
          {!max && <td>{number(term.x)}</td>}
          {weighted && <td title={term.parameter ? `W${at(term.parameter)}` : undefined}>{number(term.w)}</td>}
        </tr>;
      })}</tbody>
      <tfoot>
        {rest.count > 0 && <tr><th>{t('其余 {count} 项', { count: rest.count })}</th><td>{max ? t('都不大于最大值') : number(rest.sum)}</td>{filler}</tr>}
        {explanation.bias !== null && <tr><th>{t('偏置 b')}</th><td>{number(explanation.bias)}</td>{filler}</tr>}
        <tr><th>{max ? t('最大值 · float64 重算') : t('合计 · float64 重算')}</th><td>{number(explanation.recomputed)}</td>{filler}</tr>
      </tfoot>
    </table></div>
    {channels.length > 0 && <details className="value-channels"><summary>{t('按输入通道汇总 · {count} 个通道', { count: channels.length + (channelRest?.count ?? 0) })}</summary>
      <div className="tensor-table-scroll"><table className="tensor-slice" aria-label={t('按输入通道汇总')}>
        <thead><tr><th>{t('输入通道')}</th><th>Σ x·w</th></tr></thead>
        <tbody>{channels.map(entry => <tr key={entry.channel}><th>{entry.channel}</th><td>{number(entry.sum)}</td></tr>)}</tbody>
        {channelRest && channelRest.count > 0 && <tfoot><tr><th>{t('其余 {count} 个通道', { count: channelRest.count })}</th><td>{number(channelRest.sum)}</td></tr></tfoot>}
      </table></div>
    </details>}
  </section>;
}
