import { lazy, Suspense, type ComponentProps } from 'react';
import type SceneModule from './Scene';
import RendererBoundary from './RendererBoundary';
import { useI18n } from './i18n';

export type { SceneHandle } from './Scene';

type SceneProps = ComponentProps<typeof SceneModule>;
const LazyScene = lazy(() => import('./Scene'));

function LoadingState() {
  const { t } = useI18n();
  return <div className="renderer-loading" role="status" aria-live="polite">{t('正在加载三维渲染器…')}</div>;
}

export default function SceneView(props: SceneProps) {
  return <RendererBoundary label="三维视图"><Suspense fallback={<LoadingState />}><LazyScene {...props} /></Suspense></RendererBoundary>;
}
