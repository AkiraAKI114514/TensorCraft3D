import AttentionMatrix from './AttentionMatrix';
import { useI18n } from './i18n';
import type { RoutingSnapshot } from './inferenceTypes';
import './MoERouting.css';

/**
 * One MoE layer's measured routing for the sample: how many positions each routed expert received
 * (its load), the weight each position gave each expert, and the gate's scores, at the selected
 * window. A position is a row the layer routes; without tokenizer metadata it is not a token.
 */
export default function MoERouting({ routing }: { routing: RoutingSnapshot }) {
  const { t } = useI18n();
  const busiest = Math.max(1, ...routing.load), mean = routing.load.reduce((sum, count) => sum + count, 0) / routing.experts;
  const left = 30, width = Math.max(240, left + routing.experts * 14 + 8), bar = (width - left - 8) / routing.experts;
  const y = (count: number) => 100 - (count / busiest) * 88;
  return <section className="moe-routing" aria-label={t('MoE 路由')}>
    <h3>{t('MoE 路由 · 每个位置选 {k} 个，共 {experts} 个专家', { k: routing.topK, experts: routing.experts })}</h3>
    <p className="tensor-note">{t('本样本共 {positions} 个位置。负载是每个路由专家收到的位置数，合计为位置数 × top_k；虚线为平均值。{shared} 个共享专家处理所有位置，不参与路由。', { positions: routing.positions, shared: routing.sharedExperts })}</p>
    <figure className="moe-load">
      <figcaption>{t('专家负载')}</figcaption>
      <svg viewBox={`0 0 ${width} 120`} role="img" aria-label={t('专家负载：{loads}', { loads: routing.load.join(', ') })}>
        {routing.load.map((count, expert) => <g key={expert}>
          <rect x={left + expert * bar + 1} y={y(count)} width={Math.max(1, bar - 2)} height={100 - y(count)} className="moe-load-bar"><title>{t('专家 {expert}：{count} 个位置', { expert, count })}</title></rect>
          {routing.experts <= 16 && <text x={left + expert * bar + bar / 2} y={114} textAnchor="middle">{expert}</text>}
        </g>)}
        <line x1={left} x2={width - 8} y1={y(mean)} y2={y(mean)} className="moe-load-mean" />
        <text x={left - 4} y={16} textAnchor="end">{busiest}</text><text x={left - 4} y={103} textAnchor="end">0</text>
      </svg>
    </figure>
    <AttentionMatrix matrix={routing.weights} title={t('路由权重 · 未选中的专家为 0')} rows="位置号" columns="专家" />
    <AttentionMatrix matrix={routing.scores} title={t('门控分数 · {func}，偏置前', { func: routing.scoreFunc })} probability rows="位置号" columns="专家" />
    <details><summary>{t('窗口内每个位置选中的专家')}</summary><ul className="moe-chosen">{routing.chosen.map((experts, index) => <li key={index}>{t('位置 {position}：专家 {experts}', { position: routing.positionStart + index, experts: experts.join(', ') })}</li>)}</ul></details>
  </section>;
}
