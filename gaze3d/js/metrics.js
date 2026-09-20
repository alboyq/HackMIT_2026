/**
 * Pure gaze-measurement math. No DOM, no WebGazer — importable from the
 * browser and from `node --test` so the numbers on screen are verifiable.
 *
 * Coordinate space throughout: CSS pixels in the browser viewport, which is
 * the space WebGazer reports predictions in.
 */

const DEG = 180 / Math.PI;

export const mean = (xs) => (xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : NaN);

export const dist = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);

export function centroid(pts) {
  if (!pts.length) return null;
  let sx = 0, sy = 0;
  for (const p of pts) { sx += p.x; sy += p.y; }
  return { x: sx / pts.length, y: sy / pts.length };
}

/** Offset between where they looked (sample centroid) and the target. */
export function accuracyPx(samples, target) {
  const c = centroid(samples);
  return c ? dist(c, target) : NaN;
}

/** Spread of samples about their own centroid — RMS radius. Tracker steadiness. */
export function spreadPx(samples) {
  const c = centroid(samples);
  if (!c || samples.length < 2) return NaN;
  return Math.sqrt(mean(samples.map((s) => dist(s, c) ** 2)));
}

/** Sample-to-sample RMS. High-frequency noise, independent of slow drift. */
export function noiseS2SPx(samples) {
  if (samples.length < 2) return NaN;
  const d = [];
  for (let i = 1; i < samples.length; i++) d.push(dist(samples[i], samples[i - 1]) ** 2);
  return Math.sqrt(mean(d));
}

/** Samples per second from wall-clock timestamps on the samples themselves. */
export function sampleRateHz(samples) {
  if (samples.length < 2) return NaN;
  const span = samples[samples.length - 1].t - samples[0].t;
  return span > 0 ? ((samples.length - 1) / span) * 1000 : NaN;
}

/**
 * CSS pixels per centimetre. Gaze coordinates are CSS px, so the device-pixel
 * ratio has to be divided back out of the panel's physical PPI.
 */
export function cssPxPerCm({ diagonalIn, screenWpx, screenHpx, dpr = 1 }) {
  const devDiagPx = Math.hypot(screenWpx * dpr, screenHpx * dpr);
  const devicePpi = devDiagPx / diagonalIn;
  return devicePpi / dpr / 2.54;
}

/** Screen distance -> angle subtended at the eye. The unit that lets you
 *  compare against a real eye tracker's spec sheet. */
export function pxToDeg(px, pxPerCm, distanceCm) {
  if (!(pxPerCm > 0) || !(distanceCm > 0)) return NaN;
  return 2 * Math.atan(px / pxPerCm / (2 * distanceCm)) * DEG;
}

/** Pixels subtended by one degree at this geometry. Used for the setup diagram. */
export function degToPx(deg, pxPerCm, distanceCm) {
  if (!(pxPerCm > 0) || !(distanceCm > 0)) return NaN;
  return 2 * distanceCm * Math.tan((deg / 2) / DEG) * pxPerCm;
}

/** Which cell of a rows x cols grid a point falls in. -1 if outside. */
export function cellOf(p, { w, h, cols, rows }) {
  if (p.x < 0 || p.y < 0 || p.x >= w || p.y >= h) return -1;
  const c = Math.min(cols - 1, Math.floor((p.x / w) * cols));
  const r = Math.min(rows - 1, Math.floor((p.y / h) * rows));
  return r * cols + c;
}

const THIRD = (v, size) => (v < size / 3 ? 0 : v < (2 * size) / 3 ? 1 : 2);
const BAND_X = ['left', 'centre', 'right'];
const BAND_Y = ['top', 'middle', 'bottom'];

/** Human-readable screen third, e.g. "top left". WebGazer degrades at the edges,
 *  so error has to be reported per region or the average hides the problem. */
export function regionLabel(p, w, h) {
  const y = BAND_Y[THIRD(p.y, h)], x = BAND_X[THIRD(p.x, w)];
  return y === 'middle' && x === 'centre' ? 'centre' : `${y} ${x}`;
}

/**
 * Per-target metrics for one validation point.
 * @param {{target:{x,y}, samples:Array<{x,y,t}>, nulls:number, windowMs:number}} pt
 */
export function scorePoint(pt, geom) {
  const { target, samples, nulls = 0 } = pt;
  const errPx = accuracyPx(samples, target);
  const c = centroid(samples);
  const total = samples.length + nulls;
  return {
    target,
    centroid: c,
    n: samples.length,
    lossPct: total ? (nulls / total) * 100 : 100,
    errPx,
    errDeg: pxToDeg(errPx, geom.pxPerCm, geom.distanceCm),
    dx: c ? c.x - target.x : NaN,
    dy: c ? c.y - target.y : NaN,
    spreadPx: spreadPx(samples),
    spreadDeg: pxToDeg(spreadPx(samples), geom.pxPerCm, geom.distanceCm),
    noisePx: noiseS2SPx(samples),
    hz: sampleRateHz(samples),
    region: regionLabel(target, geom.viewportW, geom.viewportH),
    samples,
  };
}

const finite = (xs) => xs.filter((v) => Number.isFinite(v));
const p95 = (xs) => {
  const s = finite(xs).sort((a, b) => a - b);
  return s.length ? s[Math.min(s.length - 1, Math.floor(0.95 * (s.length - 1)))] : NaN;
};

/** Aggregate a whole validation run into the headline numbers. */
export function summarize(points, geom) {
  const scored = points.map((p) => scorePoint(p, geom));
  const withData = scored.filter((s) => s.n > 0);
  const errs = withData.map((s) => s.errPx);

  const byRegion = new Map();
  for (const s of withData) {
    if (!byRegion.has(s.region)) byRegion.set(s.region, []);
    byRegion.get(s.region).push(s);
  }

  return {
    points: scored,
    n: scored.length,
    measured: withData.length,
    dropped: scored.length - withData.length,
    accuracyPx: mean(errs),
    accuracyDeg: pxToDeg(mean(errs), geom.pxPerCm, geom.distanceCm),
    worstPx: errs.length ? Math.max(...errs) : NaN,
    worstDeg: pxToDeg(errs.length ? Math.max(...errs) : NaN, geom.pxPerCm, geom.distanceCm),
    p95Px: p95(errs),
    spreadPx: mean(withData.map((s) => s.spreadPx)),
    spreadDeg: pxToDeg(mean(withData.map((s) => s.spreadPx)), geom.pxPerCm, geom.distanceCm),
    noisePx: mean(withData.map((s) => s.noisePx)),
    hz: mean(withData.map((s) => s.hz)),
    lossPct: mean(scored.map((s) => s.lossPct)),
    bias: { x: mean(withData.map((s) => s.dx)), y: mean(withData.map((s) => s.dy)) },
    regions: [...byRegion].map(([region, ss]) => ({
      region,
      n: ss.length,
      errPx: mean(ss.map((s) => s.errPx)),
      errDeg: pxToDeg(mean(ss.map((s) => s.errPx)), geom.pxPerCm, geom.distanceCm),
    })),
    geom,
  };
}

/**
 * Fraction of gaze samples that landed in the tile that was actually lit.
 * This is the number that decides whether you can build a UI on this: it maps
 * directly onto "how big does a button have to be".
 */
export function scoreGrid(trials, grid) {
  const per = trials.map((t) => {
    const hits = t.samples.filter((s) => cellOf(s, grid) === t.cell).length;
    return { cell: t.cell, n: t.samples.length, hits, rate: t.samples.length ? hits / t.samples.length : 0 };
  });
  const n = per.reduce((a, b) => a + b.n, 0);
  const hits = per.reduce((a, b) => a + b.hits, 0);
  return {
    per,
    grid,
    tilePx: { w: grid.w / grid.cols, h: grid.h / grid.rows },
    hitRate: n ? hits / n : 0,
    /** A trial counts as "acquired" if the majority of its samples were on target. */
    trialsAcquired: per.filter((p) => p.rate > 0.5).length,
    trials: per.length,
  };
}

export function toCsv(summary) {
  const head = [
    'index', 'target_x', 'target_y', 'gaze_x', 'gaze_y', 'dx_px', 'dy_px',
    'error_px', 'error_deg', 'spread_px', 'noise_s2s_px', 'samples', 'loss_pct', 'hz', 'region',
  ];
  const num = (v, d = 2) => (Number.isFinite(v) ? v.toFixed(d) : '');
  const rows = summary.points.map((p, i) => [
    i + 1, num(p.target.x, 1), num(p.target.y, 1),
    num(p.centroid?.x, 1), num(p.centroid?.y, 1), num(p.dx, 1), num(p.dy, 1),
    num(p.errPx, 1), num(p.errDeg, 3), num(p.spreadPx, 1), num(p.noisePx, 1),
    p.n, num(p.lossPct, 1), num(p.hz, 1), p.region,
  ]);
  return [head, ...rows].map((r) => r.join(',')).join('\n');
}

/**
 * Pick `n` random points in a w x h box (inside `margin` fraction of each edge) that are
 * far from each other and from `existing`: best-candidate sampling, each pick maximising
 * the distance to everything chosen so far. `rand` is injectable for tests.
 */
export function spreadPoints(n, { w, h, margin = 0.1, existing = [], candidates = 40, rand = Math.random }) {
  const out = [];
  const all = existing.slice();
  const x0 = w * margin, y0 = h * margin, x1 = w * (1 - margin), y1 = h * (1 - margin);
  for (let i = 0; i < n; i++) {
    let best = null, bestD = -1;
    for (let c = 0; c < candidates; c++) {
      const p = { x: x0 + rand() * (x1 - x0), y: y0 + rand() * (y1 - y0) };
      const d = all.length ? Math.min(...all.map((q) => dist(p, q))) : Infinity;
      if (d > bestD) { bestD = d; best = p; }
    }
    best = { x: Math.round(best.x), y: Math.round(best.y) };
    out.push(best); all.push(best);
  }
  return out;
}
