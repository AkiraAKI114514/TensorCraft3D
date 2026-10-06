import * as THREE from 'three';
import { attentionLayout } from './attentionLayout';
import { crossInputRole, isAttention, isCrossAttention, isProjectionPort } from './attentionConfig';
import { residualEdges } from './graphRoutes';
import { RESIDUAL_COLOR, type Point3 } from './flowGeometry';
import { compactImageLayout, DEFAULT_IMAGE_OPTIONS, placeImageLabels, type ImageExportOptions, type ImageLabel, type LabelRequest, type Rect } from './imageLayout';
import { shapeText } from './analysis';
import type { Analysis, Graph, Layer } from './types';

interface ExportSource { scene: THREE.Scene; camera: THREE.OrthographicCamera; graph: Graph; analysis: Analysis; dimensions: Record<string, Point3>; expanded: boolean; viewport: { width: number; height: number }; direction?: () => string; }
interface PreparedImage { scene: THREE.Scene; camera: THREE.OrthographicCamera; labels: ImageLabel[]; width: number; height: number; bands: number; dispose: () => void; }
const fontFamily = '"Segoe UI", "Microsoft YaHei", sans-serif';
const escape = (text: string) => text.replace(/[<>&"']/g, char => ({ '<': '&lt;', '>': '&gt;', '&': '&amp;', '"': '&quot;', "'": '&apos;' })[char]!);
const nodeRoot = (scene: THREE.Scene, node: Layer) => scene.getObjectByName(`${isAttention(node.op) ? 'attention' : node.op === 'Add' ? 'add' : 'node'}_${node.id}`);
const corners = (box: THREE.Box3) => Array.from({ length: 8 }, (_, i) => new THREE.Vector3(i & 1 ? box.max.x : box.min.x, i & 2 ? box.max.y : box.min.y, i & 4 ? box.max.z : box.min.z));

function roundedRoute(points: THREE.Vector3[]) {
  const path = new THREE.CurvePath<THREE.Vector3>(); let previous = points[0];
  for (let i = 1; i < points.length - 1; i++) {
    const before = points[i - 1], point = points[i], after = points[i + 1];
    const radius = Math.min(0.35, before.distanceTo(point) / 3, after.distanceTo(point) / 3);
    const entry = point.clone().add(before.clone().sub(point).normalize().multiplyScalar(radius));
    const exit = point.clone().add(after.clone().sub(point).normalize().multiplyScalar(radius));
    path.add(new THREE.LineCurve3(previous, entry));
    path.add(new THREE.QuadraticBezierCurve3(entry, point, exit)); previous = exit;
  }
  path.add(new THREE.LineCurve3(previous, points.at(-1)!));
  return path;
}

function prepareImage(source: ExportSource, width: number, options: ImageExportOptions): PreparedImage {
  const height = Math.round(width / (options.layout === 'current' ? source.viewport.width / source.viewport.height : options.aspect));
  const scene = source.scene.clone(true), camera = source.camera.clone(), allocated: (THREE.BufferGeometry | THREE.Material)[] = [];
  scene.background = new THREE.Color('#f5f8fa');
  const grid = scene.getObjectByName('export_grid'); if (grid) grid.visible = false;
  const compact = options.layout === 'compact' ? compactImageLayout(source.graph, source.analysis.order, source.dimensions, options.aspect) : undefined;
  const positions: Record<string, THREE.Vector3> = {};
  source.graph.nodes.forEach(node => {
    const root = nodeRoot(scene, node);
    if (!root) return;
    if (compact) root.position.set(...compact.positions[node.id]);
    positions[node.id] = root.position.clone();
  });
  if (compact) {
    const shortcuts = residualEdges(source.graph), lanes = new Map<string, number>();
    const port = (node: Layer, side: 'input' | 'output' | 'context', projection?: string) => {
      const attention = isAttention(node.op) ? attentionLayout(node, source.expanded) : undefined;
      const offset = attention ? (projection ? attention.projectionPositions[projection] : attention.ports[side]) : [(side === 'input' ? -1 : 1) * source.dimensions[node.id][0] / 2, 0, 0];
      return positions[node.id].clone().add(new THREE.Vector3(...offset as Point3));
    };
    for (const edge of source.graph.edges) {
      const old = scene.getObjectByName(`edge_${edge.id}`); old?.removeFromParent();
      const from = source.graph.nodes.find(n => n.id === edge.source)!, to = source.graph.nodes.find(n => n.id === edge.target)!;
      const start = port(from, 'output', edge.sourcePort);
      const side = !isProjectionPort(edge.targetPort) && isCrossAttention(to) && crossInputRole(source.graph, to, edge.id) === 'context' ? 'context' : 'input';
      const end = port(to, side, isProjectionPort(edge.targetPort) ? edge.targetPort : undefined);
      const residual = shortcuts.has(edge.id), fromBand = compact.bandByNode[from.id], toBand = compact.bandByNode[to.id];
      const key = `${fromBand}-${toBand}`, lane = lanes.get(key) ?? 0;
      if (residual || fromBand !== toBand) lanes.set(key, lane + 1);
      const offset = 0.55 + (lane % 6) * 0.18;
      let curve: THREE.Curve<THREE.Vector3>;
      if (fromBand !== toBand) {
        const right = Math.max(...compact.bands.slice(Math.min(fromBand, toBand), Math.max(fromBand, toBand) + 1).map(b => b.right)) + offset;
        const left = Math.min(...compact.bands.slice(Math.min(fromBand, toBand), Math.max(fromBand, toBand) + 1).map(b => b.left)) - offset;
        const corridor = compact.bands[toBand].top - 0.35 - (lane % 4) * 0.18;
        curve = roundedRoute([start, new THREE.Vector3(right, start.y, 0.5), new THREE.Vector3(right, corridor, 0.5), new THREE.Vector3(left, corridor, 0.5), new THREE.Vector3(left, end.y, 0.5), end]);
      } else if (residual) {
        const corridor = compact.bands[fromBand].top - 0.35 - (lane % 4) * 0.18;
        curve = roundedRoute([start, new THREE.Vector3(start.x + 0.45, start.y, 0.5), new THREE.Vector3(start.x + 0.45, corridor, 0.5), new THREE.Vector3(end.x - 0.5, corridor, 0.5), new THREE.Vector3(end.x - 0.5, end.y, 0.5), end]);
      } else {
        const dx = end.x - start.x;
        curve = new THREE.CubicBezierCurve3(start, start.clone().add(new THREE.Vector3(dx * 0.4, 0, 0)), end.clone().sub(new THREE.Vector3(dx * 0.4, 0, 0)), end);
      }
      const group = new THREE.Group(); group.name = `edge_${edge.id}`; group.userData.flowKind = residual ? 'residual' : 'main';
      const color = residual ? RESIDUAL_COLOR : '#7095a6';
      const geometry = new THREE.TubeGeometry(curve, 100, 0.018, 5, false), material = new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.8 });
      allocated.push(geometry, material);
      const line = new THREE.Mesh(geometry, material); line.userData.exportPolyline = curve.getPoints(100).map(p => p.toArray()); line.userData.exportColor = color; group.add(line);
      const arrowGeometry = new THREE.ConeGeometry(0.09, 0.23, 8), arrowMaterial = new THREE.MeshBasicMaterial({ color }); allocated.push(arrowGeometry, arrowMaterial);
      const backward = source.direction?.() === 'backward', t = backward ? 0.14 : 0.86;
      const arrow = new THREE.Mesh(arrowGeometry, arrowMaterial); arrow.position.copy(curve.getPoint(t)); arrow.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), curve.getTangent(t).multiplyScalar(backward ? -1 : 1).normalize()); arrow.userData.exportMesh = true; arrow.userData.exportPart = 'flow-arrow'; group.add(arrow);
      scene.add(group);
    }
  }
  scene.updateMatrixWorld(true);
  camera.left = -width / 2; camera.right = width / 2; camera.top = height / 2; camera.bottom = -height / 2;
  if (compact) {
    // Measure the drawn geometry itself. Attention blocks fan labels out past
    // their nominal cube and can sit off-centre, so fitting the nominal boxes
    // leaves the model small inside a mostly empty frame.
    const bounds = new THREE.Box3();
    source.graph.nodes.forEach(node => {
      const root = nodeRoot(scene, node);
      if (!root) return;
      root.traverse(object => {
        if (object instanceof THREE.Mesh && object.userData.exportMesh && object.geometry.getAttribute('position')) {
          object.geometry.computeBoundingBox(); bounds.union(object.geometry.boundingBox!.clone().applyMatrix4(object.matrixWorld));
        }
        if (object.userData.exportText) { const point = object.getWorldPosition(new THREE.Vector3()); bounds.expandByPoint(point.clone().setY(point.y + 0.7)); bounds.expandByPoint(point.setY(point.y - 2.1)); }
      });
    });
    scene.traverse(object => {
      if (object.userData.exportPolyline) object.userData.exportPolyline.forEach((p: Point3) => bounds.expandByPoint(new THREE.Vector3(...p)));
    });
    if (bounds.isEmpty()) bounds.set(new THREE.Vector3(-3, -2, -1), new THREE.Vector3(3, 2, 1));
    const center = bounds.getCenter(new THREE.Vector3()), span = bounds.getSize(new THREE.Vector3()), distance = Math.max(80, span.length() * 1.5);
    camera.near = 0.1; camera.far = distance * 3 + 100;
    // A straight-on view keeps the frame's aspect equal to the layout's: a
    // three-quarter angle folds module depth into the projected height and
    // leaves the sides empty even when the rows themselves are wide.
    camera.position.copy(center).add(new THREE.Vector3(0, 0, distance * 2)); camera.up.set(0, 1, 0); camera.lookAt(center); camera.updateMatrixWorld(true);
    const view = new THREE.Box3().setFromPoints(corners(bounds).map(p => p.applyMatrix4(camera.matrixWorldInverse)));
    const size = view.getSize(new THREE.Vector3()), margin = 0.97;
    // One uniform margin keeps the model filling the frame in both directions.
    camera.zoom = Math.min(width * margin / Math.max(1, size.x), height * margin / Math.max(1, size.y));
    // Center the projected bounds, including wraparound routes.
    const viewCenter = view.getCenter(new THREE.Vector3());
    camera.position.add(new THREE.Vector3(viewCenter.x, viewCenter.y, 0).applyQuaternion(camera.quaternion)); camera.updateMatrixWorld(true);
  } else camera.zoom = source.camera.zoom * width / source.viewport.width;
  camera.updateProjectionMatrix();
  const project = (point: THREE.Vector3) => { const p = point.clone().project(camera); return [(p.x * 0.5 + 0.5) * width, (-p.y * 0.5 + 0.5) * height]; };
  const scale = width / 1920, baseSize = options.fontSize * scale, requests: LabelRequest[] = [], obstacles: Rect[] = [];
  source.graph.nodes.forEach(node => {
    const root = nodeRoot(scene, node)!;
    const box = new THREE.Box3();
    root.traverse(object => {
      if (object instanceof THREE.Mesh && object.userData.exportMesh) {
        object.geometry.computeBoundingBox();
        box.union(object.geometry.boundingBox!.clone().applyMatrix4(object.matrixWorld));
      }
      const route = object.userData.exportCurve;
      if (route) {
        const curve = new THREE.CubicBezierCurve3(...[route.start, route.control1, route.control2, route.end].map(p => new THREE.Vector3(...p as Point3)) as [THREE.Vector3, THREE.Vector3, THREE.Vector3, THREE.Vector3]);
        curve.getPoints(24).forEach(p => box.expandByPoint(p.applyMatrix4(object.matrixWorld)));
      }
    });
    if (box.isEmpty()) box.setFromCenterAndSize(positions[node.id], new THREE.Vector3(...source.dimensions[node.id]));
    const points = corners(box).map(project), xs = points.map(p => p[0]), ys = points.map(p => p[1]);
    const rect = { x: Math.min(...xs), y: Math.min(...ys), width: Math.max(...xs) - Math.min(...xs), height: Math.max(...ys) - Math.min(...ys) }; obstacles.push(rect);
    requests.push({ id: node.id, text: node.name, x: rect.x + rect.width / 2, y: rect.y + rect.height + 8 * scale, maxWidth: 145 * scale, fontSize: baseSize, secondary: shapeText(source.analysis.layers[node.id]?.output), color: '#334f5b' });
    // PNG previously used sprite text, SVG used a different constant font size.
    // Both now use the same pixel-space labels and collision handling.
    root.traverse(object => {
      if (object.userData.exportText) {
        const p = project(object.getWorldPosition(new THREE.Vector3()));
        requests.push({ id: `${node.id}:${requests.length}`, text: String(object.userData.exportText), x: p[0], y: p[1], maxWidth: 80 * scale, fontSize: Math.max(6, options.fontSize * 0.7) * scale, color: object.userData.exportColor || '#506d78' });
        object.visible = false;
      }
    });
  });
  // Reserve internal text first; node captions can move to nearby callouts.
  requests.sort((a, b) => Number(Boolean(a.secondary)) - Number(Boolean(b.secondary)));
  const measuringCanvas = document.createElement('canvas'), measuringContext = measuringCanvas.getContext('2d')!;
  const labels = placeImageLabels(requests.filter(label => !label.secondary), [], width, height, (text, size) => { measuringContext.font = `${size}px ${fontFamily}`; return measuringContext.measureText(text).width; });
  // Captions avoid the model boxes; internal labels belong inside their module.
  const internal = labels.filter(label => !label.secondary);
  const captions = placeImageLabels(requests.filter(label => Boolean(label.secondary)), [...obstacles, ...internal], width, height, (text, size) => { measuringContext.font = `${size}px ${fontFamily}`; return measuringContext.measureText(text).width; });
  return { scene, camera, labels: [...internal, ...captions], width, height, bands: compact?.bands.length ?? 1, dispose: () => allocated.forEach(resource => resource.dispose()) };
}

function drawLabels(context: CanvasRenderingContext2D, image: PreparedImage, title: string) {
  const scale = image.width / 1920;
  context.textAlign = 'center'; context.textBaseline = 'top';
  for (const label of image.labels) {
    const x = label.x + label.width / 2, lineHeight = label.fontSize * 1.35;
    if (Math.hypot(x - label.anchor[0], label.y - label.anchor[1]) > label.height) {
      context.beginPath(); context.moveTo(label.anchor[0], label.anchor[1]); context.lineTo(x, label.y); context.strokeStyle = '#9aaeb980'; context.lineWidth = scale * 0.7; context.stroke();
    }
    context.fillStyle = '#f5f8faed'; context.fillRect(label.x, label.y, label.width, label.height);
    context.font = `500 ${label.fontSize}px ${fontFamily}`; context.fillStyle = label.color;
    label.lines.forEach((line, index) => context.fillText(line, x, label.y + 4 + index * lineHeight));
    if (label.secondary) { context.font = `${label.fontSize * 0.85}px ${fontFamily}`; context.fillStyle = '#67818d'; context.fillText(label.secondary, x, label.y + 4 + label.lines.length * lineHeight); }
  }
  context.textAlign = 'left'; context.font = `600 ${18 * scale}px ${fontFamily}`;
  context.fillStyle = '#f5f8faed'; context.fillRect(20 * scale, 12 * scale, Math.min(image.width - 40 * scale, context.measureText(title).width + 16 * scale), 30 * scale);
  context.fillStyle = '#334f5b'; context.fillText(title, 28 * scale, 18 * scale, image.width - 56 * scale);
}

export async function captureSceneImage(renderer: THREE.WebGLRenderer, source: ExportSource, width: number, transparent: boolean, options = DEFAULT_IMAGE_OPTIONS): Promise<Blob> {
  const height = Math.round(width / (options.layout === 'current' ? source.viewport.width / source.viewport.height : options.aspect));
  if (Math.max(width, height) > renderer.capabilities.maxTextureSize) throw new Error(`当前 GPU 最大支持 ${renderer.capabilities.maxTextureSize} 像素，请选择较低分辨率`);
  const image = prepareImage(source, width, options), target = new THREE.WebGLRenderTarget(image.width, image.height, { samples: 4 });
  target.texture.colorSpace = THREE.SRGBColorSpace;
  const oldTarget = renderer.getRenderTarget(), color = renderer.getClearColor(new THREE.Color()), alpha = renderer.getClearAlpha();
  const output = document.createElement('canvas'); output.width = image.width; output.height = image.height;
  const context = output.getContext('2d')!;
  try {
    if (transparent) image.scene.background = null;
    renderer.setClearColor('#f5f8fa', transparent ? 0 : 1); renderer.setRenderTarget(target); renderer.clear(); renderer.render(image.scene, image.camera);
    const pixels = new Uint8Array(image.width * image.height * 4); renderer.readRenderTargetPixels(target, 0, 0, image.width, image.height, pixels);
    const data = context.createImageData(image.width, image.height), stride = image.width * 4;
    for (let row = 0; row < image.height; row++) data.data.set(pixels.subarray((image.height - row - 1) * stride, (image.height - row) * stride), row * stride);
    // Framebuffer blending stores premultiplied color; ImageData expects
    // straight alpha, otherwise translucent modules acquire dark edges.
    if (transparent) for (let index = 0; index < data.data.length; index += 4) {
      const opacity = data.data[index + 3];
      if (opacity > 0 && opacity < 255) for (let channel = 0; channel < 3; channel++) data.data[index + channel] = Math.min(255, data.data[index + channel] * 255 / opacity);
    }
    context.putImageData(data, 0, 0); drawLabels(context, image, source.graph.name);
  } finally { renderer.setRenderTarget(oldTarget); renderer.setClearColor(color, alpha); target.dispose(); image.dispose(); }
  return await new Promise<Blob>((resolve, reject) => output.toBlob(blob => blob ? resolve(blob) : reject(new Error('图片导出失败')), 'image/png'));
}

export function exportSceneSvg(source: ExportSource, width: number, options = DEFAULT_IMAGE_OPTIONS): string {
  const image = prepareImage(source, width, options);
  const project = (point: THREE.Vector3) => { const p = point.clone().project(image.camera); return [(p.x * 0.5 + 0.5) * image.width, (-p.y * 0.5 + 0.5) * image.height]; };
  const geometry = (root?: THREE.Object3D) => {
    if (!root) return '';
    const polygons: { depth: number; text: string }[] = [], paths: string[] = [];
    root.traverse(object => {
      if (object.userData.exportCurve || object.userData.exportPolyline) {
        const route = object.userData.exportCurve;
        const points: Point3[] = object.userData.exportPolyline ?? new THREE.CubicBezierCurve3(...[route.start, route.control1, route.control2, route.end].map(p => new THREE.Vector3(...p as Point3)) as [THREE.Vector3, THREE.Vector3, THREE.Vector3, THREE.Vector3]).getPoints(60).map(p => p.toArray());
        paths.push(`<path d="${points.map((p, i) => `${i ? 'L' : 'M'}${project(new THREE.Vector3(...p).applyMatrix4(object.matrixWorld)).join(',')}`).join(' ')}" fill="none" stroke="${route?.color ?? object.userData.exportColor}" stroke-width="${Math.max(0.8, image.width / 1920)}" stroke-opacity="0.8"/>`);
      }
      if (object.userData.exportMesh && object instanceof THREE.Mesh) {
        const positions = object.geometry.getAttribute('position'), indices = object.geometry.index, material = object.material as THREE.MeshStandardMaterial;
        const center = project(object.getWorldPosition(new THREE.Vector3()));
        const identity = ` data-part="${escape(object.userData.exportPart || 'mesh')}" data-center-x="${center[0]}" data-center-y="${center[1]}"${object.userData.exportSides ? ` data-sides="${object.userData.exportSides}"` : ''}`;
        for (let i = 0; i < (indices?.count ?? positions.count); i += 3) {
          const points = [0, 1, 2].map(offset => new THREE.Vector3().fromBufferAttribute(positions, indices ? indices.getX(i + offset) : i + offset).applyMatrix4(object.matrixWorld));
          const depth = points.reduce((sum, p) => sum + p.clone().applyMatrix4(image.camera.matrixWorldInverse).z, 0) / 3;
          polygons.push({ depth, text: `<polygon${identity} points="${points.map(p => project(p).join(',')).join(' ')}" fill="#${material.color.getHexString()}" fill-opacity="${material.opacity}" stroke="#${material.color.getHexString()}" stroke-width="0.4"/>` });
        }
      }
    });
    return paths.join('') + polygons.sort((a, b) => a.depth - b.depth).map(p => p.text).join('');
  };
  try {
    const shortcuts = residualEdges(source.graph);
    const lines = source.graph.edges.map(edge => `<g data-edge-id="${escape(edge.id)}" data-flow-kind="${shortcuts.has(edge.id) ? 'residual' : 'main'}">${geometry(image.scene.getObjectByName(`edge_${edge.id}`))}</g>`).join('');
    const boxes = source.graph.nodes.map(node => {
      const root = nodeRoot(image.scene, node)!, center = project(root.getWorldPosition(new THREE.Vector3()));
      return `<g data-node-id="${escape(node.id)}" data-center-x="${center[0]}" data-center-y="${center[1]}">${geometry(root)}</g>`;
    }).join('');
    const labels = image.labels.map(label => {
      const x = label.x + label.width / 2, dy = label.fontSize * 1.35;
      const leader = Math.hypot(x - label.anchor[0], label.y - label.anchor[1]) > label.height ? `<path d="M${label.anchor.join(',')} L${x},${label.y}" stroke="#9aaeb9" stroke-opacity="0.5" fill="none"/>` : '';
      const texts = label.lines.map((line, index) => `<text x="${x}" y="${label.y + 4 + label.fontSize + index * dy}" font-size="${label.fontSize}" fill="${label.color}">${escape(line)}</text>`).join('');
      const secondary = label.secondary ? `<text x="${x}" y="${label.y + 4 + label.fontSize + label.lines.length * dy}" font-size="${label.fontSize * 0.85}" fill="#67818d">${escape(label.secondary)}</text>` : '';
      return `<g data-label-id="${escape(label.id)}" data-x="${label.x}" data-y="${label.y}" data-width="${label.width}" data-height="${label.height}" font-family="Segoe UI,Microsoft YaHei,sans-serif" text-anchor="middle"><title>${escape(label.text)}</title>${leader}<rect x="${label.x}" y="${label.y}" width="${label.width}" height="${label.height}" rx="3" fill="#f5f8fa" fill-opacity="0.93"/>${texts}${secondary}</g>`;
    }).join('');
    return `<svg xmlns="http://www.w3.org/2000/svg" width="${image.width}" height="${image.height}" viewBox="0 0 ${image.width} ${image.height}" data-layout="${options.layout}" data-bands="${image.bands}"><title>${escape(source.graph.name)}</title>${lines}${boxes}${labels}<text x="${28 * width / 1920}" y="${36 * width / 1920}" font-family="Segoe UI,Microsoft YaHei,sans-serif" font-size="${18 * width / 1920}" fill="#334f5b">${escape(source.graph.name)}</text></svg>`;
  } finally { image.dispose(); }
}
