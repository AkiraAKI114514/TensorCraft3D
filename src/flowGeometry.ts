export type Point3 = [number, number, number];
export interface FlowCurve { start: Point3; end: Point3; control1: Point3; control2: Point3; }

export const RESIDUAL_COLOR = '#d073a8';

export function residualCurve(start: Point3, end: Point3, height: number, depth: number): FlowCurve & { color: string } {
  return { start, end, control1: [start[0], height, depth], control2: [end[0], height, depth], color: RESIDUAL_COLOR };
}
