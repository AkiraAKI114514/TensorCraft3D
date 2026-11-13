import type { Layer } from './types';

/** Inclusive element ranges of an input tensor `[1,C,H,W]`. */
export interface Field { channels: [number, number]; rows: [number, number]; columns: [number, number] }

/** Operations whose output element reads a window of their input, as the 3D view can show. */
export const WINDOW_OPS = new Set(['Conv2d', 'MaxPool2d', 'AvgPool2d', 'AdaptiveAvgPool2d', 'AdaptiveMaxPool2d']);

const clamp = (value: number, low: number, high: number) => Math.min(high, Math.max(low, value));
const span = (start: number, end: number, size: number): [number, number] => [clamp(start, 0, size - 1), clamp(end, 0, size - 1)];

/**
 * The input elements that output element (channel 0, `row`, `column`) of a layer reads, with the
 * same parameters and defaults as the shape analysis. A convolution reads a kernel-sized window
 * through every input channel of its group; the window is clipped where it hangs over the
 * padding. Pooling reads its window in the same channel, and adaptive pooling the slice PyTorch
 * assigns to that output position. Anything else, or tensors that are not `[1,C,H,W]`, has no
 * window to show.
 */
export function receptiveField(node: Pick<Layer, 'op' | 'params'>, input: number[], output: number[], row: number, column: number): Field | null {
  if (!WINDOW_OPS.has(node.op) || input.length !== 4 || output.length !== 4) return null;
  const [, channels, height, width] = input, [, , outHeight, outWidth] = output;
  const number = (key: string, fallback: number) => Number(node.params[key] ?? fallback);
  if (node.op.startsWith('Adaptive')) {
    const rows = span(Math.floor(row * height / outHeight), Math.ceil((row + 1) * height / outHeight) - 1, height);
    return { channels: [0, 0], rows, columns: span(Math.floor(column * width / outWidth), Math.ceil((column + 1) * width / outWidth) - 1, width) };
  }
  const convolution = node.op === 'Conv2d';
  const kernel = number('kernel_size', convolution ? 3 : 2), stride = number('stride', convolution ? 1 : 2), padding = number('padding', convolution ? 1 : 0);
  const window = (index: number, size: number) => span(index * stride - padding, index * stride - padding + kernel - 1, size);
  // Output channel 0 belongs to the first group, which reads the first channels / groups inputs.
  const groups = convolution ? number('groups', 1) : 1;
  return { channels: convolution ? [0, channels / groups - 1] : [0, 0], rows: window(row, height), columns: window(column, width) };
}
