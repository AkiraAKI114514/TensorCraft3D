export type Point3 = [number, number, number];
export interface FlowCurve { start: Point3; end: Point3; control1: Point3; control2: Point3; }

export const RESIDUAL_COLOR = '#d073a8';

/**
 * How bright flow pulses are, shared by every edge: above 1 at night, so the bloom pass turns the
 * pulses into light; 1 by day and while an image is exported.
 */
export const FLOW_GLOW = { value: 1 };

/** Whether edges show their travelling pulses, shared by every edge: 0 while an image is exported. */
export const FLOW_PULSES = { value: 1 };

export function residualCurve(start: Point3, end: Point3, height: number, depth: number): FlowCurve & { color: string } {
  return { start, end, control1: [start[0], height, depth], control2: [end[0], height, depth], color: RESIDUAL_COLOR };
}
