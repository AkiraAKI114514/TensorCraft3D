import { useCallback, useState } from 'react';
import { Check, X } from 'lucide-react';
import { useI18n } from './i18n';
import { GUIDE_STEPS, completeStep, emptyGuide, guideProgress, loadGuide, nextGuideStep, saveGuide, type GuideState, type GuideStep } from './guideState';
import './GuidePanel.css';

const COPY: Record<GuideStep, { title: string; body: string; action: string }> = {
  template: { title: '选一个模板', body: '从层库顶部的模板里选一个现成结构，比如 Residual CNN 或 Transformer Encoder。', action: '去选模板' },
  select: { title: '点一层看它的形状', body: '在三维视图或拓扑图里点任意一层，右侧会显示它的输入输出形状、参数和连接。', action: '回到三维视图' },
  inspect: { title: '跑一次单样本，看真实数值', body: '打开张量观测并运行单样本推理：统计、分布、特征图和注意力矩阵都来自真实计算。', action: '打开张量观测' },
  train: { title: '训练几轮', body: '用合成数据训练并观察损失曲线；选中某一层还能看到它自己的梯度范数。', action: '打开训练设置' },
  export: { title: '把成果带走', body: '导出可运行的 PyTorch 代码，或者导出 PNG / SVG 结构图。', action: '打开代码导出' },
};

/** 进度保存在浏览器里；`complete` 对已完成的步骤是空操作，所以可以放心地在每次相关事件上调用。 */
export function useGuide() {
  const [state, setState] = useState<GuideState>(loadGuide), [open, setOpenState] = useState(false);
  const update = useCallback((change: (previous: GuideState) => GuideState) => setState(previous => { const next = change(previous); if (next !== previous) saveGuide(next); return next; }), []);
  const complete = useCallback((step: GuideStep) => update(previous => completeStep(previous, step)), [update]);
  const setOpen = useCallback((value: boolean) => { setOpenState(value); if (value) update(previous => previous.seen ? previous : { ...previous, seen: true }); }, [update]);
  const reset = useCallback(() => update(() => ({ ...emptyGuide(), seen: true })), [update]);
  return { state, open, setOpen, complete, reset };
}

/**
 * 非模态的上手清单。每一步的按钮只负责把人带到对应的位置——切视图、切标签或打开对话框——
 * 推理和训练始终由用户自己在那里点下去，所以引导不会触发任何计算。
 */
export default function GuidePanel({ state, onAction, onClose, onReset }: { state: GuideState; onAction: (step: GuideStep) => void; onClose: () => void; onReset: () => void }) {
  const { t } = useI18n();
  const done = guideProgress(state), next = nextGuideStep(state);
  return <section className="guide-panel" aria-label={t('上手引导')}>
    <div className="guide-heading"><div><span className="eyebrow">GETTING STARTED</span><h2>{t('上手引导')}</h2></div><button type="button" className="icon-button" aria-label={t('关闭引导')} title={t('关闭引导')} onClick={onClose}><X size={16} /></button></div>
    <p className="guide-progress" role="status">{next ? t('已完成 {done} / {total} 步', { done, total: GUIDE_STEPS.length }) : t('五步都走完了，接下来随意探索。')}</p>
    <ol className="guide-steps">{GUIDE_STEPS.map((step, index) => {
      const finished = !!state.done[step];
      return <li key={step} className={finished ? 'done' : step === next ? 'current' : ''} aria-current={step === next ? 'step' : undefined}>
        <span className="guide-marker" aria-hidden="true">{finished ? <Check size={13} /> : index + 1}</span>
        <div><h3>{t(COPY[step].title)}{finished && <span className="guide-done-label"> · {t('已完成')}</span>}</h3><p>{t(COPY[step].body)}</p>
          <button type="button" className="button subtle" onClick={() => onAction(step)}>{t(COPY[step].action)}</button></div>
      </li>;
    })}</ol>
    <p className="guide-note">{t('引导只把你带到对应的位置，不会替你运行推理或训练。')}</p>
    <button type="button" className="button text-button" onClick={onReset} disabled={!done}>{t('重新开始')}</button>
  </section>;
}
