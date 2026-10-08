import { lazy, Suspense, type ComponentProps } from 'react';
import type TopologyModule from './Topology';
import RendererBoundary from './RendererBoundary';

type TopologyProps = ComponentProps<typeof TopologyModule>;
const LazyTopology = lazy(() => import('./Topology'));

function LoadingState() {
  return <div className="renderer-loading" role="status" aria-live="polite">正在加载拓扑渲染器…</div>;
}

export default function TopologyView(props: TopologyProps) {
  return <RendererBoundary label="拓扑图"><Suspense fallback={<LoadingState />}><LazyTopology {...props} /></Suspense></RendererBoundary>;
}
