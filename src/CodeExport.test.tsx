import { beforeEach, describe, expect, it, vi } from 'vitest';
import { renderToStaticMarkup } from 'react-dom/server';
import { generatePython } from './export';
import { PRESETS } from './presets';
import CodeExport from './CodeExport';
import App from './App';

vi.mock('./export', async importOriginal => ({ ...await importOriginal<typeof import('./export')>(), generatePython: vi.fn(() => '# generated on demand') }));
vi.mock('./Scene', () => ({ default: () => null }));
vi.mock('./Topology', () => ({ default: () => null }));
vi.mock('./Charts', () => ({ default: () => null }));

beforeEach(() => { vi.mocked(generatePython).mockReset().mockReturnValue('# generated on demand'); });

describe('on-demand code export', () => {
  it('does not generate Python when the workbench export dialog is closed', () => {
    const markup = renderToStaticMarkup(<App />);
    expect(markup).toContain('TensorCraft3D');
    expect(generatePython).not.toHaveBeenCalled();
  });

  it('generates Python only for an active Python preview', () => {
    const graph = PRESETS.mlp();
    const markup = renderToStaticMarkup(<CodeExport graph={graph} valid tab="python" onTab={() => {}} onExport={() => {}} />);
    expect(generatePython).toHaveBeenCalledExactlyOnceWith(graph);
    expect(markup).toContain('# generated on demand');
    expect(markup).toContain('下载 .py');
  });

  it('renders Graph JSON without generating Python', () => {
    const graph = PRESETS.mlp();
    const markup = renderToStaticMarkup(<CodeExport graph={graph} valid tab="json" onTab={() => {}} onExport={() => {}} />);
    expect(generatePython).not.toHaveBeenCalled();
    expect(markup).toContain(graph.name);
    expect(markup).toContain('下载 .json');
  });

  it('keeps generator failures visible and disables Python download', () => {
    vi.mocked(generatePython).mockImplementation(() => { throw new Error('invalid export'); });
    const markup = renderToStaticMarkup(<CodeExport graph={PRESETS.blank()} valid tab="python" onTab={() => {}} onExport={() => {}} />);
    expect(markup).toContain('# invalid export');
    expect(markup).toContain('disabled=""');
  });
});
