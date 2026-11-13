/** 上手引导的进度。它只记录用户自己做过什么，从不替用户执行推理或训练。 */
export const GUIDE_STORAGE_KEY = 'tensorcraft-guide';
export const GUIDE_STEPS = ['template', 'select', 'inspect', 'train', 'export'] as const;
export type GuideStep = typeof GUIDE_STEPS[number];
export interface GuideState { seen: boolean; done: Partial<Record<GuideStep, boolean>>; }

export const emptyGuide = (): GuideState => ({ seen: false, done: {} });

/** 存储里的内容可能来自旧版本或被手改过：只接受认识的步骤和布尔值，其余一律当作没做过。 */
export function parseGuide(raw: string | null): GuideState {
  if (!raw) return emptyGuide();
  try {
    const value: unknown = JSON.parse(raw);
    if (!value || typeof value !== 'object' || Array.isArray(value)) return emptyGuide();
    const { seen, done } = value as { seen?: unknown; done?: unknown };
    const record = done && typeof done === 'object' && !Array.isArray(done) ? done as Record<string, unknown> : {};
    return { seen: seen === true, done: Object.fromEntries(GUIDE_STEPS.filter(step => record[step] === true).map(step => [step, true])) };
  } catch { return emptyGuide(); }
}

export function loadGuide(): GuideState {
  try { return typeof window === 'undefined' ? emptyGuide() : parseGuide(window.localStorage.getItem(GUIDE_STORAGE_KEY)); }
  catch { return emptyGuide(); }
}

export function saveGuide(state: GuideState) {
  try { if (typeof window !== 'undefined') window.localStorage.setItem(GUIDE_STORAGE_KEY, JSON.stringify(state)); }
  catch { /* The guide still works for this session when browser storage is unavailable. */ }
}

/** 已完成的步骤返回同一个对象，调用方可以据此跳过一次无意义的重渲染和写盘。 */
export function completeStep(state: GuideState, step: GuideStep): GuideState {
  return state.done[step] ? state : { ...state, done: { ...state.done, [step]: true } };
}

export const guideProgress = (state: GuideState) => GUIDE_STEPS.filter(step => state.done[step]).length;
/** 下一步是顺序里第一个没做的；步骤可以乱序完成，所以它不一定是最后完成那步的后一个。 */
export const nextGuideStep = (state: GuideState): GuideStep | null => GUIDE_STEPS.find(step => !state.done[step]) ?? null;
