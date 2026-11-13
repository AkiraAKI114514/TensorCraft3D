import { describe, expect, it } from 'vitest';
import { receptiveField } from './receptiveField';

const conv = (params: Record<string, number> = {}) => ({ op: 'Conv2d' as const, params: { out_channels: 8, kernel_size: 3, stride: 1, padding: 1, ...params } });

describe('the input window behind one output element', () => {
  it('reads a kernel through every input channel, clipped at the padding', () => {
    expect(receptiveField(conv(), [1, 4, 8, 8], [1, 8, 8, 8], 0, 0)).toEqual({ channels: [0, 3], rows: [0, 1], columns: [0, 1] });
    expect(receptiveField(conv(), [1, 4, 8, 8], [1, 8, 8, 8], 5, 7)).toEqual({ channels: [0, 3], rows: [4, 6], columns: [6, 7] });
    // Stride 2 without padding: output (2, 1) starts at input (4, 2).
    expect(receptiveField(conv({ stride: 2, padding: 0 }), [1, 4, 9, 9], [1, 8, 4, 4], 2, 1)).toEqual({ channels: [0, 3], rows: [4, 6], columns: [2, 4] });
    // Output channel 0 sits in the first of two groups.
    expect(receptiveField(conv({ groups: 2 }), [1, 4, 8, 8], [1, 8, 8, 8], 1, 1)!.channels).toEqual([0, 1]);
  });

  it('reads a pooling window in its own channel, with the shape analysis defaults', () => {
    expect(receptiveField({ op: 'MaxPool2d', params: {} }, [1, 16, 32, 32], [1, 16, 16, 16], 3, 4)).toEqual({ channels: [0, 0], rows: [6, 7], columns: [8, 9] });
    expect(receptiveField({ op: 'AvgPool2d', params: { kernel_size: 3, stride: 3, padding: 0 } }, [1, 2, 9, 9], [1, 2, 3, 3], 2, 0)).toEqual({ channels: [0, 0], rows: [6, 8], columns: [0, 2] });
  });

  it('reads the slice adaptive pooling assigns to an output position', () => {
    expect(receptiveField({ op: 'AdaptiveAvgPool2d', params: { output_size: 1 } }, [1, 16, 32, 32], [1, 16, 1, 1], 0, 0)).toEqual({ channels: [0, 0], rows: [0, 31], columns: [0, 31] });
    // 10 rows into 3: [0, 3], [3, 6], [6, 9], overlapping as in PyTorch.
    expect(receptiveField({ op: 'AdaptiveMaxPool2d', params: { output_size: 3 } }, [1, 1, 10, 10], [1, 1, 3, 3], 1, 2)).toEqual({ channels: [0, 0], rows: [3, 6], columns: [6, 9] });
  });

  it('has no window for other operations or ranks', () => {
    expect(receptiveField({ op: 'Linear', params: {} }, [1, 4, 8, 8], [1, 4, 8, 8], 0, 0)).toBeNull();
    expect(receptiveField(conv(), [1, 4, 8], [1, 8, 8], 0, 0)).toBeNull();
  });
});
