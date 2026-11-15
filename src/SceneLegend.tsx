import { useMemo, useState } from 'react';
import { ChevronDown } from 'lucide-react';
import type { Analysis, Graph } from './types';
import { HEAD_COLORS, legendRows, type Glyph, type LegendRow } from './operatorStyle';
import { ROPE_COLOR } from './layerAnatomy';
import { useI18n } from './i18n';

/** A hex colour moved toward white (`t` > 0) or black (`t` < 0). */
function tone(hex: string, t: number) {
  const value = parseInt(hex.slice(1), 16), channels = [value >> 16, (value >> 8) & 255, value & 255];
  const target = t > 0 ? 255 : 0, amount = Math.abs(t);
  return `#${channels.map(c => Math.round(c + (target - c) * amount).toString(16).padStart(2, '0')).join('')}`;
}

/**
 * A small isometric box: front face in the colour itself, top lighter, side darker, the way the
 * scene's light shades a block. `x, y` is the front face's lower left corner.
 */
function Box({ x, y, w, h, d, color, opacity = 1 }: { x: number; y: number; w: number; h: number; d: number; color: string; opacity?: number }) {
  const dx = d * 0.62, dy = -d * 0.42;
  return <g opacity={opacity}>
    <path d={`M${x} ${y - h}L${x + dx} ${y - h + dy}L${x + w + dx} ${y - h + dy}L${x + w} ${y - h}Z`} fill={tone(color, 0.38)} />
    <path d={`M${x + w} ${y - h}L${x + w + dx} ${y - h + dy}L${x + w + dx} ${y + dy}L${x + w} ${y}Z`} fill={tone(color, -0.28)} />
    <rect x={x} y={y - h} width={w} height={h} fill={color} />
  </g>;
}

/** The operator's shape in miniature, matching how the 3D scene draws it. */
export function LegendGlyph({ glyph, color, secondary }: { glyph: Glyph; color: string; secondary?: string }) {
  const body = (() => {
    switch (glyph) {
      case 'tensor': return <Box x={10} y={20} w={11} h={13} d={9} color={color} />;
      case 'io': return secondary ? <><Box x={4} y={20} w={8} h={12} d={7} color={color} /><Box x={19} y={20} w={8} h={12} d={7} color={secondary} /></> : <Box x={10} y={20} w={11} h={13} d={9} color={color} />;
      case 'slab': return <Box x={14} y={21} w={4} h={16} d={11} color={color} />;
      case 'membrane': return <Box x={15} y={21} w={2} h={16} d={11} color={color} opacity={0.9} />;
      case 'shape': return <Box x={15} y={20} w={3} h={10} d={9} color={color} />;
      case 'table': return <><Box x={11} y={21} w={10} h={15} d={9} color={color} />{[0, 1, 2, 3].map(i => <line key={i} x1={11} x2={21} y1={9.5 + i * 3} y2={9.5 + i * 3} stroke={tone(color, -0.35)} strokeWidth={0.7} />)}</>;
      // Multi-head latent attention: the input, the Q and K/V latents it is squeezed through, heads with their RoPE part on top, the output.
      case 'waist': return <><Box x={2} y={21} w={3} h={15} d={7} color={tone(color, 0.28)} /><Box x={8} y={12} w={2.5} h={5} d={6} color={tone(color, 0.1)} /><Box x={8} y={20} w={2.5} h={4} d={6} color={tone(color, -0.15)} />{[0, 1, 2].map(i => <g key={i}><Box x={14 + i * 2.6} y={21 - i * 0.9} w={1.6} h={9} d={2} color={tone(color, 0.18)} /><Box x={14 + i * 2.6} y={12 - i * 0.9} w={1.6} h={4} d={2} color={ROPE_COLOR} /></g>)}<Box x={25} y={21} w={4} h={15} d={7} color={color} /></>;
      // Mixture of experts: the router, a bank of experts (two of them routed), the output.
      case 'experts': return <><Box x={3} y={21} w={2.5} h={15} d={8} color={tone(color, 0.32)} />{[0, 1, 2, 3].map(i => <Box key={i} x={9 + (i % 2) * 6} y={13 + Math.floor(i / 2) * 7} w={4.5} h={4.5} d={4} color={i === 0 || i === 3 ? tone(color, -0.12) : tone(color, 0.6)} />)}<Box x={24} y={21} w={5} h={15} d={8} color={color} /></>;
      // An attention head: Q, K and V around a frosted panel.
      case 'heads': return <><path d="M9 19L18 4L27 19Z" fill={tone(color, 0.75)} stroke={tone(color, -0.1)} strokeWidth={0.9} strokeLinejoin="round" /><circle cx={18} cy={5} r={2.6} fill={HEAD_COLORS.q} /><circle cx={9} cy={19} r={2.6} fill={HEAD_COLORS.k} /><circle cx={27} cy={19} r={2.6} fill={HEAD_COLORS.v} /><circle cx={18} cy={14} r={1.8} fill={HEAD_COLORS.merge} /></>;
      // A shortcut arching over the layers it skips, onto its addition.
      case 'residual': return <><path d="M3 18C3 3 25 3 25 15" fill="none" stroke={color} strokeWidth={1.6} strokeLinecap="round" /><path d="M22.4 12.6L25 16.2L27.4 12.4" fill="none" stroke={color} strokeWidth={1.4} strokeLinecap="round" strokeLinejoin="round" /><Box x={22} y={22} w={5} h={5} d={4} color={color} /></>;
      case 'merge': return <><Box x={13} y={19} w={7} h={7} d={6} color={color} /></>;
      // Operand plates meeting at a contraction node, then the output block.
      case 'contract': return <><Box x={3} y={11.5} w={2.5} h={7} d={6} color={tone(color, 0.48)} /><Box x={3} y={21} w={2.5} h={7} d={6} color={tone(color, 0.25)} /><path d="M6.5 8.5L12 14M6.5 18L12 14M14.5 14L18 14" stroke={tone(color, -0.2)} strokeWidth={0.9} fill="none" /><rect x={11.5} y={12.6} width={3} height={3} fill={tone(color, -0.2)} /><Box x={18} y={21} w={8} h={14} d={8} color={color} /></>;
      // Three plates shrinking toward the output, like a zoom.
      case 'scale': return <><Box x={5} y={21} w={3} h={15} d={10} color={tone(color, 0.45)} /><Box x={13} y={19.5} w={3} h={12} d={8} color={tone(color, 0.22)} /><Box x={21} y={18} w={3} h={9} d={6} color={color} /></>;
      // The kept lower triangle as a staircase, the masked scores dark behind it.
      case 'mask': return <><Box x={8} y={21} w={16} h={16} d={2} color={tone(color, -0.45)} />{[0, 1, 2, 3].map(i => <Box key={i} x={8} y={9 + i * 4} w={4 * (i + 1)} h={4} d={5} color={tone(color, 0.36 - i * 0.12)} />)}</>;
      case 'group': return <><Box x={7} y={21} w={8} h={11} d={7} color={tone(color, 0.3)} /><Box x={17} y={21} w={8} h={11} d={7} color={color} /></>;
      case 'flow': return <><path d="M2 14C10 8 22 20 30 12" fill="none" stroke={tone(color, 0.55)} strokeWidth={1.2} />{[4, 10, 16, 22].map(x => <circle key={x} cx={x + 2} cy={13 - Math.sin((x - 2) / 28 * Math.PI * 1.6) * 3.4} r={1.4} fill={color} />)}<path d="M28 9.6L31.6 11.6L28.6 14.6" fill="none" stroke={color} strokeWidth={1.2} strokeLinecap="round" strokeLinejoin="round" /></>;
      case 'kernel': return <><rect x={4} y={6} width={9} height={9} fill={tone(color, 0.75)} stroke={color} strokeWidth={1.1} /><path d="M13 6L25 11M13 15L25 15" stroke={color} strokeWidth={0.8} opacity={0.75} /><rect x={25} y={10} width={5} height={5} fill={color} /></>;
    }
  })();
  return <svg className="legend-glyph" viewBox="0 0 34 24" width={34} height={24} aria-hidden="true">{body}</svg>;
}

const STORAGE_KEY = 'tensorlab-legend-open';
/** The width the open legend covers at the scene's right edge, card and margin, for the overview framing. */
export const LEGEND_INSET = 276;
function initiallyOpen() {
  try {
    const saved = localStorage.getItem(STORAGE_KEY);
    if (saved === '0' || saved === '1') return saved === '1';
  } catch { /* storage unavailable: fall through to the default */ }
  // On a phone the legend would cover the model; it starts folded there.
  return typeof window === 'undefined' || !window.matchMedia?.('(max-width: 700px)').matches;
}

/**
 * The 3D view's legend, built from the operators actually in the shown level, in the order data
 * reaches them: each row's glyph is the operator's shape as the scene draws it, in its colour.
 */
/** Whether the legend is open, remembered per browser; the workbench also frames the overview around it. */
export function useLegendOpen(): [boolean, () => void] {
  const [open, setOpen] = useState(initiallyOpen);
  const toggle = () => { setOpen(!open); try { localStorage.setItem(STORAGE_KEY, open ? '0' : '1'); } catch { /* not remembered */ } };
  return [open, toggle];
}

export default function SceneLegend({ graph, analysis, showKernels, open, onToggle }: { graph: Graph; analysis: Pick<Analysis, 'order'>; showKernels: boolean; open: boolean; onToggle: () => void }) {
  const { t } = useI18n();
  const rows = useMemo(() => legendRows(graph, analysis, { showKernels }), [graph, analysis, showKernels]);
  const toggle = onToggle;
  if (!rows.length) return null;
  const note = (row: LegendRow) => row.note ? (row.note.literal ? row.note.text : t(row.note.text)) : null;
  return <section className={`scene-legend ${open ? '' : 'collapsed'}`} aria-label={t('图例')}>
    <button type="button" className="scene-legend-toggle" aria-expanded={open} aria-controls="scene-legend-rows" onClick={toggle} title={open ? t('收起图例') : t('展开图例')}>
      <span>{t('图例')}</span><small>{rows.length}</small><ChevronDown size={13} />
    </button>
    <ul id="scene-legend-rows" hidden={!open}>
      {rows.map(row => <li key={row.key}>
        <LegendGlyph glyph={row.glyph} color={row.color} secondary={row.secondary} />
        <span><strong>{row.names.map(name => t(name)).join(' · ')}</strong>{note(row) && <small>{note(row)}</small>}</span>
      </li>)}
    </ul>
  </section>;
}
