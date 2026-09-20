import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  centroid, accuracyPx, spreadPx, noiseS2SPx, sampleRateHz,
  cssPxPerCm, pxToDeg, degToPx, cellOf, regionLabel, scoreGrid, summarize, toCsv,
} from '../js/metrics.js';
import { spreadPoints } from '../js/metrics.js';

const close = (a, b, tol = 1e-6) =>
  assert.ok(Math.abs(a - b) <= tol, `expected ${a} within ${tol} of ${b}`);

test('centroid averages both axes', () => {
  assert.deepEqual(centroid([{ x: 0, y: 0 }, { x: 10, y: 20 }]), { x: 5, y: 10 });
  assert.equal(centroid([]), null);
});

test('accuracy is centroid-to-target distance, not mean of distances', () => {
  // Samples straddle the target symmetrically: centroid is dead on, so accuracy
  // is 0 even though every individual sample is 10px off.
  const samples = [{ x: -10, y: 0 }, { x: 10, y: 0 }];
  close(accuracyPx(samples, { x: 0, y: 0 }), 0);
  close(spreadPx(samples), 10); // ...and the spread catches what accuracy misses
});

test('spread is RMS radius about the centroid', () => {
  const s = [{ x: 3, y: 0 }, { x: -3, y: 0 }, { x: 0, y: 4 }, { x: 0, y: -4 }];
  close(spreadPx(s), Math.sqrt((9 + 9 + 16 + 16) / 4));
  assert.ok(Number.isNaN(spreadPx([{ x: 1, y: 1 }])));
});

test('sample-to-sample noise ignores absolute position', () => {
  const a = [{ x: 0, y: 0 }, { x: 3, y: 4 }, { x: 6, y: 8 }];
  const b = a.map((p) => ({ x: p.x + 500, y: p.y + 500 }));
  close(noiseS2SPx(a), 5);
  close(noiseS2SPx(b), noiseS2SPx(a));
});

test('sample rate uses the interval count, not the sample count', () => {
  const s = [0, 40, 80, 120].map((t) => ({ x: 0, y: 0, t }));
  close(sampleRateHz(s), 25); // 3 gaps over 120ms
});

test('css px per cm divides the device pixel ratio back out', () => {
  // 15.4" panel, 2560x1600 device px reported as 1280x800 CSS px at dpr 2.
  const hidpi = cssPxPerCm({ diagonalIn: 15.4, screenWpx: 1280, screenHpx: 800, dpr: 2 });
  const same1x = cssPxPerCm({ diagonalIn: 15.4, screenWpx: 2560, screenHpx: 1600, dpr: 1 });
  close(hidpi, same1x / 2, 1e-9); // a CSS px is twice as wide, so half as many per cm
  close(hidpi, 38.6, 0.1);
});

test('degree conversion round-trips and 1 deg is ~1cm at 57cm', () => {
  const pxPerCm = 40, d = 60;
  close(pxToDeg(degToPx(1.7, pxPerCm, d), pxPerCm, d), 1.7, 1e-9);
  close(degToPx(1, 40, 57) / 40, 0.995, 0.01); // the "1cm per degree at 57cm" rule of thumb
  assert.ok(Number.isNaN(pxToDeg(10, 0, 60)));
});

test('cell lookup clamps the far edge and rejects outside points', () => {
  const g = { w: 300, h: 300, cols: 3, rows: 3 };
  assert.equal(cellOf({ x: 10, y: 10 }, g), 0);
  assert.equal(cellOf({ x: 150, y: 150 }, g), 4);
  assert.equal(cellOf({ x: 299.9, y: 299.9 }, g), 8);
  assert.equal(cellOf({ x: 300, y: 150 }, g), -1);
  assert.equal(cellOf({ x: -1, y: 0 }, g), -1);
});

test('region labels name the screen third', () => {
  assert.equal(regionLabel({ x: 50, y: 50 }, 900, 900), 'top left');
  assert.equal(regionLabel({ x: 450, y: 450 }, 900, 900), 'centre');
  assert.equal(regionLabel({ x: 850, y: 850 }, 900, 900), 'bottom right');
});

test('summary reports loss, bias and per-region breakdown', () => {
  const geom = { pxPerCm: 40, distanceCm: 60, viewportW: 900, viewportH: 900 };
  const pts = [
    // Consistently 40px right of target -> a rightward bias, ~1 degree of error.
    { target: { x: 100, y: 100 }, samples: [{ x: 140, y: 100, t: 0 }, { x: 140, y: 100, t: 40 }], nulls: 0 },
    { target: { x: 800, y: 800 }, samples: [{ x: 840, y: 800, t: 0 }, { x: 840, y: 800, t: 40 }], nulls: 2 },
    { target: { x: 450, y: 450 }, samples: [], nulls: 4 }, // total tracking failure
  ];
  const s = summarize(pts, geom);
  assert.equal(s.n, 3);
  assert.equal(s.measured, 2);
  assert.equal(s.dropped, 1);
  close(s.accuracyPx, 40);
  close(s.accuracyDeg, pxToDeg(40, 40, 60));
  close(s.bias.x, 40);
  close(s.bias.y, 0);
  close(s.lossPct, (0 + 50 + 100) / 3);
  assert.deepEqual(s.regions.map((r) => r.region).sort(), ['bottom right', 'top left']);
  assert.equal(toCsv(s).split('\n').length, 4); // header + 3 points
  assert.ok(toCsv(s).split('\n')[3].endsWith('centre')); // dropped point still reported
});

test('grid scoring separates sample hit rate from trials acquired', () => {
  const grid = { w: 300, h: 300, cols: 3, rows: 3 };
  const r = scoreGrid([
    { cell: 0, samples: [{ x: 10, y: 10 }, { x: 20, y: 20 }, { x: 150, y: 150 }] }, // 2/3 on target
    { cell: 4, samples: [{ x: 10, y: 10 }, { x: 20, y: 20 }] },                     // missed entirely
  ], grid);
  close(r.hitRate, 2 / 5);
  assert.equal(r.trialsAcquired, 1);
  assert.equal(r.trials, 2);
  assert.deepEqual(r.tilePx, { w: 100, h: 100 });
});

test('spreadPoints keeps picks apart and inside the margin', () => {
  let seed = 7;
  const rand = () => { seed = (seed * 16807) % 2147483647; return seed / 2147483647; };
  const w = 1500, h = 1000;
  const pts = spreadPoints(6, { w, h, margin: 0.1, rand });
  assert.equal(pts.length, 6);
  for (const p of pts) { assert.ok(p.x >= 150 && p.x <= 1350 && p.y >= 100 && p.y <= 900); }
  let minD = Infinity;
  for (let i = 0; i < pts.length; i++) for (let j = i + 1; j < pts.length; j++) minD = Math.min(minD, Math.hypot(pts[i].x - pts[j].x, pts[i].y - pts[j].y));
  assert.ok(minD > 300, `min separation ${minD}`);
  // respects existing points too
  const more = spreadPoints(1, { w, h, existing: pts, rand });
  assert.ok(Math.min(...pts.map((q) => Math.hypot(q.x - more[0].x, q.y - more[0].y))) > 200);
});
