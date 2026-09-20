/**
 * Gaze Accuracy Bench — drives WebGazer through calibration, validation and a
 * tile hit test, then reports the error in pixels and degrees of visual angle.
 */

import { summarize, scoreGrid, toCsv, cssPxPerCm, pxToDeg, degToPx, cellOf } from './metrics.js';

const $ = (id) => document.getElementById(id);
const el = (sel, root = document) => root.querySelector(sel);
const wait = (ms) => new Promise((r) => setTimeout(r, ms));
const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
const fmt = (v, d = 2) => (Number.isFinite(v) ? v.toFixed(d) : '—');
const reduceMotion = matchMedia('(prefers-reduced-motion: reduce)').matches;

/* ── state ─────────────────────────────────────────────────────────────── */

const S = {
  running: false,
  calibrated: false,
  mode: 'idle',          // idle | calibrate | validate | live | grid
  gaze: null,            // { x, y, t }
  stamps: [],            // recent sample times, for the rate meter
  hits: 0, misses: 0,    // rolling signal-quality tally
  trail: [],
  runs: [],
  summary: null,
  gridResult: null,
  synthetic: false,
  abort: false,
  skip: false,
};

/** Fed by the gaze listener while a measurement window is open. */
const collector = { on: false, samples: [], nulls: 0 };
/** When set, every frame trains the model toward this viewport coordinate. */
let trainTo = null;

/* ── viewing geometry ──────────────────────────────────────────────────── */

function geometry() {
  const distanceCm = Number($('inDist').value) || 60;
  const diagonalIn = Number($('inDiag').value) || 15.4;
  const pxPerCm = cssPxPerCm({
    diagonalIn,
    screenWpx: screen.width,
    screenHpx: screen.height,
    dpr: devicePixelRatio || 1,
  });
  return {
    distanceCm, diagonalIn, pxPerCm,
    viewportW: innerWidth, viewportH: innerHeight,
    dpr: devicePixelRatio || 1,
  };
}

function paintGeometry() {
  const g = geometry();
  const onePx = degToPx(1, g.pxPerCm, g.distanceCm);
  $('roNative').textContent = `${Math.round(screen.width * g.dpr)}×${Math.round(screen.height * g.dpr)}`;
  $('roPxCm').innerHTML = `${fmt(g.pxPerCm, 1)}<small>px/cm</small>`;
  $('roDegPx').innerHTML = `${fmt(onePx, 1)}<small>px</small>`;

  // Wedge drawn to scale: the eye sits at x=44, the screen plane at x=400.
  const EYE_X = 44, SCREEN_X = 400, MID = 100, RUN = SCREEN_X - EYE_X;
  const halfW = g.viewportW / 2 / g.pxPerCm;                 // cm
  const halfAngle = Math.atan(halfW / g.distanceCm);
  const half = clamp(Math.tan(halfAngle) * RUN, 8, 88);
  $('geoWedge').setAttribute('points',
    `${EYE_X},${MID} ${SCREEN_X},${MID - half} ${SCREEN_X},${MID + half}`);
  $('geoScreen').setAttribute('y1', MID - half);
  $('geoScreen').setAttribute('y2', MID + half);
  $('geoDist').textContent = `${g.distanceCm} cm`;
  $('geoSpan').textContent = `${fmt(halfAngle * 2 * 180 / Math.PI, 0)}°`;
}

/* ── phase rail ────────────────────────────────────────────────────────── */

function phase(name, status, read) {
  const li = el(`.phase[data-phase="${name}"]`);
  if (!li) return;
  if (status) li.dataset.status = status;
  if (read !== undefined) el('[data-read]', li).textContent = read;
}

function view(name) {
  for (const v of document.querySelectorAll('.view')) v.hidden = v.dataset.view !== name;
  $('stage').scrollTop = 0;
}

function pill(state, text) {
  $('pill').dataset.state = state;
  $('pillText').textContent = text;
}

/* ── camera ────────────────────────────────────────────────────────────── */

/** Swap the note under Start camera between quiet fineprint and a loud alert. */
function notice(kind, html) {
  const n = $('startNote');
  n.className = kind ? `notice notice--${kind}` : 'fineprint';
  n.innerHTML = html;
}

async function startCamera() {
  const btn = $('btnStart');
  btn.disabled = true;
  btn.textContent = 'Waiting for permission…';
  pill('warn', 'allow camera');
  notice('wait', '<b>Chrome is asking to use your camera.</b> Click <b>Allow</b> in the bubble ' +
    'just under the address bar. The video stays in this page — nothing is uploaded.');

  try {
    webgazer.saveDataAcrossSessions(false);       // every session is a fresh measurement
    webgazer.setRegression($('inReg').value);
    webgazer.setTracker('TFFacemesh');
    webgazer.params.showGazeDot = false;          // we draw our own marker
    webgazer.setGazeListener(onGaze);

    await webgazer.begin();

    webgazer.showVideoPreview(true);
    webgazer.showFaceOverlay(true);
    webgazer.showFaceFeedbackBox(true);
    webgazer.showPredictionPoints(false);
    webgazer.applyKalmanFilter($('inKalman').checked);
    applyClickLock();
    await adoptVideo();

    S.running = true;
    pill('live', 'tracking');
    phase('camera', 'done', 'live');
    phase('calibrate', 'pending', `0/${$('inCalib').value}`);
    notice(null, 'Nothing leaves this machine — inference runs in the page.');
    view('camera');
    syncCalibButton();
    syncControls();
  } catch (err) {
    console.error(err);
    btn.disabled = false;
    btn.textContent = 'Start camera';
    pill('error', 'camera blocked');
    notice('bad', cameraError(err));
  }
}

function cameraError(err) {
  const name = err?.name || '';
  if (!isSecureContext) {
    return '<b>The camera needs a secure origin.</b> This page is on ' +
      `<code>${location.protocol}</code>, which Chrome will not give camera access to. ` +
      'Run <code>./serve.sh</code> and open <code>http://localhost:8000</code> instead.';
  }
  if (name === 'NotAllowedError') {
    return '<b>Chrome blocked the camera.</b> Click the camera icon in the address bar, ' +
      'choose <b>Allow</b>, then reload. If no icon is there, the prompt was dismissed — ' +
      'reload and click Allow when it appears.';
  }
  if (name === 'NotFoundError') return '<b>No camera found</b> on this machine.';
  if (name === 'NotReadableError') {
    return '<b>The camera is busy.</b> Another app (Zoom, Photo Booth, Meet) is holding it. ' +
      'Quit that app and reload.';
  }
  return `<b>Could not start the camera.</b> ${err?.message || err}`;
}

/** WebGazer appends its preview to <body> as a fixed overlay. Move it into the rail. */
async function adoptVideo() {
  for (let i = 0; i < 60; i++) {
    const box = $('webgazerVideoContainer');
    if (box) {
      $('camslot').textContent = '';
      $('camslot').appendChild(box);
      sizeVideo();
      return;
    }
    await wait(100);
  }
  $('camslot').innerHTML =
    '<p class="camslot__idle">Camera started but no preview appeared. Tracking may still work.</p>';
}

/** Resize through WebGazer so the video, overlay and feedback box stay aligned. */
function sizeVideo() {
  if (!$('webgazerVideoContainer')) return;
  const w = Math.round($('camslot').clientWidth);
  if (w > 0) webgazer.setVideoViewerSize(w, Math.round(w * 3 / 4));
}

/** WebGazer trains on every document click by default, which silently
 *  contaminates a run whenever you press a button. */
function applyClickLock() {
  if (!S.running && !webgazer.params) return;
  if ($('inLock').checked) webgazer.removeMouseEventListeners();
  else webgazer.addMouseEventListeners();
}

/* ── gaze bus ──────────────────────────────────────────────────────────── */

function onGaze(data, _clock) {
  const t = performance.now();

  if (trainTo) webgazer.recordScreenPosition(trainTo.x, trainTo.y, 'click');

  if (!data || !Number.isFinite(data.x) || !Number.isFinite(data.y)) {
    S.misses++;
    if (collector.on) collector.nulls++;
    return;
  }

  S.hits++;
  S.gaze = { x: data.x, y: data.y, t };
  S.stamps.push(t);
  if (S.stamps.length > 40) S.stamps.shift();
  if (collector.on) collector.samples.push({ x: data.x, y: data.y, t });

  S.trail.push({ x: data.x, y: data.y, t });
  if (S.trail.length > 24) S.trail.shift();
}

function sampleRate() {
  if (S.stamps.length < 4) return NaN;
  const span = S.stamps[S.stamps.length - 1] - S.stamps[0];
  return span > 0 ? ((S.stamps.length - 1) / span) * 1000 : NaN;
}

function openWindow() { collector.on = true; collector.samples = []; collector.nulls = 0; }
function closeWindow() {
  collector.on = false;
  return { samples: collector.samples.slice(), nulls: collector.nulls };
}

/* ── live readout + gaze marker ────────────────────────────────────────── */

const layer = $('gazelayer');
const lctx = layer.getContext('2d');

function fitCanvas(canvas, ctx) {
  const dpr = devicePixelRatio || 1;
  const w = canvas.clientWidth || innerWidth;
  const h = canvas.clientHeight || innerHeight;
  if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
  }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { w, h };
}

const heat = $('heat');
const hctx = heat.getContext('2d');
let heatArmed = false;

function frame() {
  requestAnimationFrame(frame);

  const hz = sampleRate();
  const total = S.hits + S.misses;
  $('roX').textContent = S.gaze ? Math.round(S.gaze.x) : '—';
  $('roY').textContent = S.gaze ? Math.round(S.gaze.y) : '—';
  $('roHz').innerHTML = `${fmt(hz, 1)}<small>Hz</small>`;
  $('roLoss').innerHTML = S.calibrated && total
    ? `${fmt((S.hits / total) * 100, 0)}<small>%</small>`
    : '—<small>%</small>';

  const showMarker = S.mode === 'live' || S.mode === 'grid';
  layer.classList.toggle('gazelayer--on', showMarker);
  if (!showMarker) return;

  const { w, h } = fitCanvas(layer, lctx);
  lctx.clearRect(0, 0, w, h);
  if (!S.gaze) return;

  const now = performance.now();
  const fresh = S.trail.filter((p) => now - p.t < 420);

  // trail
  lctx.lineCap = 'round';
  for (let i = 1; i < fresh.length; i++) {
    const a = (i / fresh.length) * 0.5;
    lctx.strokeStyle = `rgba(213,232,90,${a.toFixed(3)})`;
    lctx.lineWidth = 1 + (i / fresh.length) * 3;
    lctx.beginPath();
    lctx.moveTo(fresh[i - 1].x, fresh[i - 1].y);
    lctx.lineTo(fresh[i].x, fresh[i].y);
    lctx.stroke();
  }

  // marker: crosshair + uncertainty ring sized to the measured error
  const { x, y } = S.gaze;
  const r = S.summary && Number.isFinite(S.summary.accuracyPx)
    ? clamp(S.summary.accuracyPx, 12, 260) : 34;
  lctx.strokeStyle = 'rgba(76,125,255,.5)';
  lctx.lineWidth = 1;
  lctx.setLineDash([4, 5]);
  lctx.beginPath(); lctx.arc(x, y, r, 0, Math.PI * 2); lctx.stroke();
  lctx.setLineDash([]);

  lctx.strokeStyle = 'rgba(213,232,90,.9)';
  lctx.lineWidth = 1.2;
  lctx.beginPath();
  lctx.moveTo(x - 11, y); lctx.lineTo(x + 11, y);
  lctx.moveTo(x, y - 11); lctx.lineTo(x, y + 11);
  lctx.stroke();
  lctx.fillStyle = '#d5e85a';
  lctx.beginPath(); lctx.arc(x, y, 3.2, 0, Math.PI * 2); lctx.fill();

  if (heatArmed) paintHeat(x, y);
}

function paintHeat(x, y) {
  fitCanvas(heat, hctx);
  const g = hctx.createRadialGradient(x, y, 0, x, y, 46);
  g.addColorStop(0, 'rgba(213,232,90,.055)');
  g.addColorStop(1, 'rgba(213,232,90,0)');
  hctx.fillStyle = g;
  hctx.beginPath(); hctx.arc(x, y, 46, 0, Math.PI * 2); hctx.fill();
}

function clearHeat() {
  const { w, h } = fitCanvas(heat, hctx);
  hctx.clearRect(0, 0, w, h);
}

/* ── the chamber: one shared fixation-target runner ────────────────────── */

const chamber = $('chamber');
const targetEl = $('target');
const ringEl = $('targetRing');

function openChamber(mode, hud, keys = 'space · skip point') {
  S.mode = mode;
  S.abort = false;
  chamber.hidden = false;
  $('chamberHud').textContent = hud;
  $('chamberKeys').textContent = keys;
}

function closeChamber() {
  chamber.hidden = true;
  targetEl.hidden = true;
  $('chamberGrid').hidden = true;
  $('chamberGrid').textContent = '';
  heatArmed = false;
  clearHeat();
  S.mode = 'idle';
}

function placeTarget(p) {
  targetEl.style.left = `${p.x}px`;
  targetEl.style.top = `${p.y}px`;
  targetEl.hidden = false;
}

/** Contract the ring over `ms`. The shrinking radius is the "hold still now" cue. */
function contractRing(ms, from = 50, to = 13) {
  return new Promise((resolve) => {
    if (reduceMotion) { ringEl.setAttribute('r', to); setTimeout(resolve, ms); return; }
    const t0 = performance.now();
    const step = () => {
      const k = clamp((performance.now() - t0) / ms, 0, 1);
      ringEl.setAttribute('r', (from + (to - from) * (1 - (1 - k) ** 3)).toFixed(1));
      if (k < 1 && !S.abort && !S.skip) requestAnimationFrame(step);
      else resolve();
    };
    requestAnimationFrame(step);
  });
}

const SETTLE_MS = 900;   // saccade to the target and settle
const DWELL_MS = 800;    // measurement window
const GAP_MS = 130;

/**
 * Walk a list of viewport points. Each point gets a settle phase, then an
 * armed window during which samples are collected and — when `train` is set —
 * fed back to WebGazer as training data.
 */
async function runPoints(points, { train = false, onPoint, progress }) {
  for (let i = 0; i < points.length; i++) {
    if (S.abort) return false;
    S.skip = false;
    const p = points[i];

    targetEl.classList.remove('target--armed');
    placeTarget(p);
    progress?.(i, points.length);
    await contractRing(SETTLE_MS);
    if (S.abort) return false;

    targetEl.classList.add('target--armed');
    openWindow();
    if (train) trainTo = p;
    await wait(DWELL_MS);
    trainTo = null;
    const got = closeWindow();
    if (S.abort) return false;

    onPoint?.(p, got, i);
    targetEl.hidden = true;
    await wait(GAP_MS);
  }
  progress?.(points.length, points.length);
  return true;
}

/* ── point layouts ─────────────────────────────────────────────────────── */

const spread = (fracs) => {
  const w = innerWidth, h = innerHeight;
  return fracs.map(([fx, fy]) => ({
    x: Math.round(clamp(fx * w, 44, w - 44)),
    y: Math.round(clamp(fy * h, 44, h - 44)),
  }));
};

const GRID3 = (a, b, c) => [[a, a], [b, a], [c, a], [a, b], [b, b], [c, b], [a, c], [b, c], [c, c]];

const CALIB = {
  5: [[.5, .5], [.15, .15], [.85, .15], [.15, .85], [.85, .85]],
  9: GRID3(.1, .5, .9),
  13: [...GRID3(.1, .5, .9), [.3, .3], [.7, .3], [.3, .7], [.7, .7]],
};

// Validation points sit *between* the calibration points, so the run measures
// how well the model generalises rather than how well it memorised.
const VALID = {
  5: [[.5, .5], [.25, .25], [.75, .25], [.25, .75], [.75, .75]],
  9: GRID3(.25, .5, .75),
  13: [...GRID3(.25, .5, .75), [.05, .05], [.95, .05], [.05, .95], [.95, .95]],
};

const shuffle = (a) => a.map((v) => [Math.random(), v]).sort((x, y) => x[0] - y[0]).map(([, v]) => v);

/* ── calibration ───────────────────────────────────────────────────────── */

async function calibrate() {
  const n = Number($('inCalib').value);
  const pts = spread(CALIB[n]);

  webgazer.clearData();
  S.calibrated = false;
  S.summary = null;
  S.gridResult = null;
  S.synthetic = false;
  S.runs = [];            // drift is only meaningful within one calibration
  $('tileOut')?.remove();
  $('runs').hidden = true;
  syncControls();
  phase('calibrate', 'active', `0/${n}`);
  phase('measure', 'locked', '—');
  phase('report', 'locked', '—');

  openChamber('calibrate',
    'Look straight at the centre of each target. When the ring locks and turns yellow, hold your gaze there — that is when it learns.');

  const ok = await runPoints(pts, {
    train: true,
    progress: (i) => {
      phase('calibrate', 'active', `${i}/${n}`);
      $('chamberHud').textContent = i < n
        ? `Point ${i + 1} of ${n} — look at the centre and hold.`
        : 'Calibration complete.';
    },
  });

  closeChamber();
  if (!ok) { phase('calibrate', 'pending', `0/${n}`); view('camera'); return; }

  S.calibrated = true;
  S.hits = S.misses = 0;   // nulls from before training are not signal loss
  phase('calibrate', 'done', `${n} pts`);
  phase('measure', 'pending', `0/${$('inValid').value}`);
  syncControls();
  await validate();
}

/* ── validation ────────────────────────────────────────────────────────── */

async function validate() {
  const n = Number($('inValid').value);
  const pts = shuffle(spread(VALID[n]));
  const collected = [];

  phase('measure', 'active', `0/${n}`);
  openChamber('validate',
    'Same again, but nothing is being learned now — the error at each target is being recorded. No marker is shown, so you cannot chase it.');

  const ok = await runPoints(pts, {
    train: false,
    onPoint: (p, got) => collected.push({ target: p, samples: got.samples, nulls: got.nulls, windowMs: DWELL_MS }),
    progress: (i) => {
      phase('measure', 'active', `${i}/${n}`);
      $('chamberHud').textContent = `Target ${Math.min(i + 1, n)} of ${n} — recording.`;
    },
  });

  closeChamber();
  if (!ok) { phase('measure', 'pending', `0/${n}`); view(S.summary ? 'report' : 'camera'); return; }

  S.summary = summarize(collected, geometry());
  S.runs.push({ at: new Date(), acc: S.summary.accuracyDeg, px: S.summary.accuracyPx, hz: S.summary.hz });
  phase('measure', 'done', `${n} pts`);
  phase('report', 'done', `${fmt(S.summary.accuracyDeg, 1)}°`);
  syncControls();
  view('report');      // must be visible before the plot can measure itself
  renderReport();
}

/* ── report ────────────────────────────────────────────────────────────── */

const grade = (deg) => (deg <= 1.5 ? '' : deg <= 3 ? 'mid' : 'bad');

function renderReport() {
  const s = S.summary;
  $('repStamp').textContent = S.synthetic
    ? 'synthetic preview'
    : S.runs[S.runs.length - 1].at.toLocaleTimeString();
  $('vAcc').textContent = fmt(s.accuracyDeg, 2);
  $('vAccPx').textContent = fmt(s.accuracyPx, 0);
  $('vWorst').textContent = fmt(s.worstDeg, 2);
  $('vHz').textContent = fmt(s.hz, 1);
  $('vRead').innerHTML = verdict(s);

  // error by screen third
  const order = ['top left', 'top centre', 'top right', 'middle left', 'centre', 'middle right',
    'bottom left', 'bottom centre', 'bottom right'];
  const rows = s.regions.slice().sort((a, b) => order.indexOf(a.region) - order.indexOf(b.region));
  el('tbody', $('tblRegion')).innerHTML =
    `<tr><th>region</th><th>error</th><th>px</th><th>n</th></tr>` +
    rows.map((r) => `<tr><td>${r.region}</td>` +
      `<td class="${grade(r.errDeg)}"><b>${fmt(r.errDeg, 2)}°</b></td>` +
      `<td>${fmt(r.errPx, 0)}</td><td>${r.n}</td></tr>`).join('');

  const g = s.geom;
  const both = (px) => `${fmt(pxToDeg(px, g.pxPerCm, g.distanceCm), 2)}° · ${fmt(px, 0)} px`;
  const qual = [
    ['spread about centroid', both(s.spreadPx)],
    ['sample-to-sample noise', both(s.noisePx)],
    ['95th percentile error', both(s.p95Px)],
    ['systematic offset', `${s.bias.x >= 0 ? 'right' : 'left'} ${fmt(Math.abs(s.bias.x), 0)} px, ` +
      `${s.bias.y >= 0 ? 'down' : 'up'} ${fmt(Math.abs(s.bias.y), 0)} px`],
    ['dropped frames', `${fmt(s.lossPct, 1)} %${s.dropped ? ` · ${s.dropped} target(s) lost` : ''}`],
    ['viewport', `${g.viewportW}×${g.viewportH} at ${fmt(g.pxPerCm, 1)} px/cm`],
  ];
  el('tbody', $('tblQual')).innerHTML =
    `<tr><th>measure</th><th>value</th></tr>` +
    qual.map(([k, v]) => `<tr><td>${k}</td><td><b>${v}</b></td></tr>`).join('');

  // per-target detail
  $('tblPoints').innerHTML =
    `<thead><tr><th>#</th><th>region</th><th>target</th><th>predicted</th>` +
    `<th>error</th><th>px</th><th>spread</th><th>n</th><th>loss</th></tr></thead><tbody>` +
    s.points.map((p, i) => `<tr>
      <td>${i + 1}</td><td>${p.region}</td>
      <td>${Math.round(p.target.x)}, ${Math.round(p.target.y)}</td>
      <td>${p.centroid ? `${Math.round(p.centroid.x)}, ${Math.round(p.centroid.y)}` : 'lost'}</td>
      <td class="${grade(p.errDeg)}"><b>${fmt(p.errDeg, 2)}°</b></td>
      <td>${fmt(p.errPx, 0)}</td><td>${fmt(p.spreadPx, 0)}</td>
      <td>${p.n}</td><td class="${p.lossPct > 25 ? 'bad' : ''}">${fmt(p.lossPct, 0)}%</td>
    </tr>`).join('') + '</tbody>';

  if (S.runs.length > 1) {
    $('runs').hidden = false;
    const first = S.runs[0].acc;
    $('runsList').innerHTML = S.runs.map((r, i) => {
      const d = r.acc - first;
      const drift = i === 0 ? 'baseline'
        : `${d >= 0 ? '+' : '−'}${fmt(Math.abs(d), 2)}° vs. run 1`;
      return `<li><i>run ${i + 1}</i><span>${r.at.toLocaleTimeString()}</span>` +
        `<b>${fmt(r.acc, 2)}°</b><i>${drift}</i></li>`;
    }).join('');
  }

  drawPlot(s);
}

function verdict(s) {
  if (S.synthetic) {
    return '<b style="color:var(--amber)">Simulated data.</b> This is what the report looks like ' +
      'filled in — the error was generated, not measured. Start the camera and run a real ' +
      'calibration for numbers that mean anything.';
  }
  const d = s.accuracyDeg;
  const target = Math.round(s.accuracyPx * 2);
  let head;
  if (d <= 1.5) head = 'Unusually tight for a webcam.';
  else if (d <= 3) head = 'Typical of a well-lit, still-headed WebGazer session.';
  else if (d <= 5) head = `In line with the ~4° the WebGazer paper reports.`;
  else head = 'Too coarse to localise gaze.';

  const use = d <= 1.5
    ? 'Fine enough to tell which paragraph you are reading.'
    : d <= 3
      ? 'Good for regions — which quadrant, which column, which card — not for anything small.'
      : d <= 5
        ? 'Only coarse regions are trustworthy. Treat it as a rough attention signal, not a pointer.'
        : 'Check that your face is evenly lit and your head has not moved since calibration, then recalibrate.';

  const skew = Math.hypot(s.bias.x, s.bias.y) > s.accuracyPx * 0.7
    ? ` The error is mostly a constant offset (${s.bias.x >= 0 ? 'right' : 'left'} and ` +
      `${s.bias.y >= 0 ? 'down' : 'up'}), which usually means your head shifted after calibration.`
    : '';

  return `${head} A prediction lands about <b>${fmt(s.accuracyPx, 0)} px</b> (<b>${fmt(d, 2)}°</b>) from ` +
    `where you were actually looking, so the smallest thing you could reliably point at is roughly ` +
    `<b>${target}×${target} px</b>. ${use}${skew}`;
}

/* ── field plot ────────────────────────────────────────────────────────── */

function drawPlot(s) {
  const cv = $('plot');
  const dpr = devicePixelRatio || 1;
  const cssW = cv.clientWidth || 900;
  const g = s.geom;
  const GUTTER = 26;                            // strip below the screen for the legend
  const screenH = Math.round(cssW * (g.viewportH / g.viewportW));
  const cssH = screenH + GUTTER;
  cv.style.height = `${cssH}px`;
  cv.width = Math.round(cssW * dpr);
  cv.height = Math.round(cssH * dpr);
  const c = cv.getContext('2d');
  c.setTransform(dpr, 0, 0, dpr, 0, 0);

  const k = cssW / g.viewportW;                 // viewport px -> plot px
  const X = (v) => v * k, Y = (v) => v * k;

  c.clearRect(0, 0, cssW, cssH);

  // thirds graticule — the regions the table breaks error down by
  c.strokeStyle = '#1f2c39';
  c.lineWidth = 1;
  for (let i = 1; i < 3; i++) {
    c.beginPath();
    c.moveTo(Math.round(cssW * i / 3) + .5, 0); c.lineTo(Math.round(cssW * i / 3) + .5, screenH);
    c.moveTo(0, Math.round(screenH * i / 3) + .5); c.lineTo(cssW, Math.round(screenH * i / 3) + .5);
    c.stroke();
  }

  // the screen edge, so it is obvious which misses fell off it
  c.strokeStyle = '#2b3b4b';
  c.beginPath();
  c.moveTo(0, screenH + .5); c.lineTo(cssW, screenH + .5);
  c.stroke();

  c.save();
  c.beginPath();
  c.rect(0, 0, cssW, screenH);                  // keep the legend gutter clean
  c.clip();

  for (const p of s.points) {
    const tx = X(p.target.x), ty = Y(p.target.y);

    // sample cloud
    c.fillStyle = 'rgba(213,232,90,.5)';
    for (const sm of p.samples) {
      c.beginPath(); c.arc(X(sm.x), Y(sm.y), 1.7, 0, Math.PI * 2); c.fill();
    }

    // error ring — radius is the mean miss distance at this target
    if (Number.isFinite(p.errPx) && p.centroid) {
      c.strokeStyle = 'rgba(232,113,140,.42)';
      c.setLineDash([3, 4]);
      c.lineWidth = 1;
      c.beginPath(); c.arc(tx, ty, Math.max(1, X(p.errPx)), 0, Math.PI * 2); c.stroke();
      c.setLineDash([]);

      // offset vector, target -> where it actually predicted
      c.strokeStyle = '#e8718c';
      c.lineWidth = 1.4;
      c.beginPath(); c.moveTo(tx, ty); c.lineTo(X(p.centroid.x), Y(p.centroid.y)); c.stroke();
      c.fillStyle = '#e8718c';
      c.beginPath(); c.arc(X(p.centroid.x), Y(p.centroid.y), 2.6, 0, Math.PI * 2); c.fill();
    }

    // the truth: a hairline cross at the target
    c.strokeStyle = p.n ? '#eaf1f8' : '#e8718c';
    c.lineWidth = 1.1;
    c.beginPath();
    c.moveTo(tx - 6, ty); c.lineTo(tx + 6, ty);
    c.moveTo(tx, ty - 6); c.lineTo(tx, ty + 6);
    c.stroke();
  }

  c.restore();

  // scale bar in the gutter: one degree of visual angle, so the plot reads in real units
  const onePx = degToPx(1, g.pxPerCm, g.distanceCm);
  if (Number.isFinite(onePx)) {
    const bar = X(onePx), x0 = 2, y0 = screenH + GUTTER / 2 + 1;
    c.strokeStyle = '#7a8ca0';
    c.lineWidth = 1;
    c.beginPath();
    c.moveTo(x0, y0); c.lineTo(x0 + bar, y0);
    c.moveTo(x0, y0 - 4); c.lineTo(x0, y0 + 4);
    c.moveTo(x0 + bar, y0 - 4); c.lineTo(x0 + bar, y0 + 4);
    c.stroke();
    c.fillStyle = '#7a8ca0';
    c.font = '10px "IBM Plex Mono", monospace';
    c.textBaseline = 'middle';
    c.fillText(`1° of visual angle = ${Math.round(onePx)} px`, x0 + bar + 8, y0);
  }
}

/* ── tile hit test ─────────────────────────────────────────────────────── */

async function tileTest() {
  const cols = 3, rows = 3;
  const box = $('chamberGrid');
  box.hidden = false;
  box.style.gridTemplateColumns = `repeat(${cols}, 1fr)`;
  box.style.gridTemplateRows = `repeat(${rows}, 1fr)`;
  box.innerHTML = Array.from({ length: cols * rows }, () => '<div class="tile"></div>').join('');
  const tiles = [...box.children];

  const order = shuffle([...tiles.keys()]);
  const trials = [];
  openChamber('grid',
    'A tile lights up: look at it and hold. This measures how often the prediction lands in the right tile — the number that decides how big a gaze-controlled button has to be.',
    'space · skip tile');

  for (const cell of order) {
    if (S.abort) break;
    S.skip = false;
    tiles.forEach((t) => t.classList.remove('tile--lit'));
    tiles[cell].classList.add('tile--lit');
    await wait(reduceMotion ? 700 : SETTLE_MS);
    if (S.abort) break;
    openWindow();
    await wait(DWELL_MS);
    const got = closeWindow();
    trials.push({ cell, samples: got.samples });
  }

  const grid = { w: innerWidth, h: innerHeight, cols, rows };
  closeChamber();
  if (!trials.length) { view('report'); return; }

  S.gridResult = scoreGrid(trials, grid);
  showTileResult();
  view('report');
}

function showTileResult() {
  const r = S.gridResult;
  const pct = (v) => `${(v * 100).toFixed(0)}%`;
  const cls = r.hitRate >= .8 ? '' : r.hitRate >= .55 ? 'mid' : 'bad';
  const line = document.createElement('section');
  line.className = 'tbl tbl--wide';
  line.id = 'tileOut';
  line.innerHTML = `<h3 class="block__h">Tile hit test · ${r.grid.cols}×${r.grid.rows}</h3>
    <table><tbody>
      <tr><td>samples in the correct tile</td><td class="${cls}"><b>${pct(r.hitRate)}</b></td>
          <td>tile is ${Math.round(r.tilePx.w)}×${Math.round(r.tilePx.h)} px</td></tr>
      <tr><td>tiles acquired (majority of samples on target)</td>
          <td><b>${r.trialsAcquired}/${r.trials}</b></td>
          <td>${r.hitRate >= .8 ? 'a 3×3 layout is selectable' : 'too tight for 3×3 — use larger regions'}</td></tr>
    </tbody></table>`;
  $('tileOut')?.remove();
  $('tblPoints').closest('.tbl').after(line);
}

/* ── live view ─────────────────────────────────────────────────────────── */

function liveView() {
  openChamber('live',
    'Free look. The yellow crosshair is the prediction; the blue ring is your measured error, so the true target is somewhere inside it. Warmth builds up where you dwell.',
    'h · clear the heat map');
  heatArmed = true;
  clearHeat();
}

/* ── exports ───────────────────────────────────────────────────────────── */

function download(name, text, type) {
  const url = URL.createObjectURL(new Blob([text], { type }));
  const a = Object.assign(document.createElement('a'), { href: url, download: name });
  a.click();
  URL.revokeObjectURL(url);
}

const stampName = () =>
  `${S.synthetic ? 'synthetic' : 'gaze'}-run-${new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19)}`;

function exportCsv() {
  if (S.summary) download(`${stampName()}.csv`, toCsv(S.summary), 'text/csv');
}

function exportJson() {
  if (!S.summary) return;
  const s = S.summary;
  const payload = {
    recordedAt: new Date().toISOString(),
    library: 'webgazer 3.3.0',
    ...(S.synthetic ? { synthetic: true, note: 'Simulated data. Not a measurement.' } : {}),
    settings: {
      regression: $('inReg').value,
      kalmanFilter: $('inKalman').checked,
      strayClicksIgnored: $('inLock').checked,
      calibrationPoints: Number($('inCalib').value),
      validationPoints: Number($('inValid').value),
      settleMs: SETTLE_MS, dwellMs: DWELL_MS,
    },
    geometry: s.geom,
    accuracy: { deg: s.accuracyDeg, px: s.accuracyPx, worstDeg: s.worstDeg, p95Px: s.p95Px },
    precision: { spreadPx: s.spreadPx, spreadDeg: s.spreadDeg, noiseS2SPx: s.noisePx },
    signal: { hz: s.hz, lossPct: s.lossPct, targetsLost: s.dropped },
    bias: s.bias,
    regions: s.regions,
    tileTest: S.gridResult && {
      grid: S.gridResult.grid, hitRate: S.gridResult.hitRate,
      trialsAcquired: S.gridResult.trialsAcquired, trials: S.gridResult.trials,
    },
    points: s.points.map((p) => ({
      target: p.target, predicted: p.centroid, errPx: p.errPx, errDeg: p.errDeg,
      spreadPx: p.spreadPx, noisePx: p.noisePx, samples: p.n, lossPct: p.lossPct,
      hz: p.hz, region: p.region,
    })),
    runs: S.runs.map((r) => ({ at: r.at.toISOString(), accuracyDeg: r.acc, accuracyPx: r.px, hz: r.hz })),
  };
  download(`gaze-run-${stampName()}.json`, JSON.stringify(payload, null, 2), 'application/json');
}

/* ── synthetic preview ─────────────────────────────────────────────────── */

/**
 * Fills the report from simulated data so the output can be inspected without a
 * camera. Error grows toward the corners, which is how WebGazer really behaves.
 * Labelled as synthetic throughout — it measures nothing.
 */
function syntheticRun() {
  const g = geometry();
  const points = shuffle(spread(VALID[13])).map((target) => {
    const rx = (target.x / g.viewportW - .5) * 2;
    const ry = (target.y / g.viewportH - .5) * 2;
    const edge = Math.hypot(rx, ry);
    const bx = 18 + 62 * rx * edge, by = 12 + 54 * ry * edge;
    const jitter = 24 + 32 * edge;
    const n = 17 + Math.round(Math.random() * 7);
    return {
      target,
      samples: Array.from({ length: n }, (_, i) => ({
        x: target.x + bx + (Math.random() - .5) * jitter,
        y: target.y + by + (Math.random() - .5) * jitter,
        t: i * 38,
      })),
      nulls: Math.random() < .3 ? 1 + Math.round(Math.random() * 3) : 0,
      windowMs: DWELL_MS,
    };
  });

  S.synthetic = true;
  S.summary = summarize(points, g);
  S.runs = [{ at: new Date(), acc: S.summary.accuracyDeg, px: S.summary.accuracyPx, hz: S.summary.hz }];
  phase('camera', 'done', 'simulated');
  phase('calibrate', 'done', '13 pts');
  phase('measure', 'done', '13 pts');
  phase('report', 'done', `${fmt(S.summary.accuracyDeg, 1)}°`);
  pill('warn', 'synthetic data');
  view('report');
  renderReport();
}

/* ── wiring ────────────────────────────────────────────────────────────── */

function syncCalibButton() {
  const n = $('inCalib').value;
  el('.btn__k', $('btnCalib')).textContent = `${n} points`;
}

/** Reflect what is actually available, so nothing looks clickable while it is inert. */
function syncControls() {
  for (const [id, ok] of [
    ['btnCalib', S.running],
    ['btnRecal', S.running],
    ['btnRemeasure', S.running && S.calibrated],
    ['btnFree', S.running && S.calibrated],
    ['btnFreeEarly', S.running && S.calibrated],
    ['btnGrid', S.running && S.calibrated],
  ]) {
    const b = $(id);
    b.disabled = !ok;
    b.title = ok ? '' : S.running ? 'Calibrate first' : 'Start the camera first';
  }
  for (const id of ['btnCsv', 'btnJson']) $(id).disabled = !S.summary;
}

/** Nothing that reads predictions is worth running before the model is trained. */
function guard(fn, needsCalibration = false) {
  return async () => {
    if (!S.running) { pill('warn', 'start the camera first'); return; }
    if (needsCalibration && !S.calibrated) { pill('warn', 'calibrate first'); return; }
    try { await fn(); } catch (err) { console.error(err); closeChamber(); pill('error', 'run failed'); }
  };
}

$('btnStart').addEventListener('click', startCamera);
$('btnCalib').addEventListener('click', guard(calibrate));
$('btnRecal').addEventListener('click', guard(calibrate));
$('btnRemeasure').addEventListener('click', guard(validate, true));
$('btnFree').addEventListener('click', guard(liveView, true));
$('btnFreeEarly').addEventListener('click', guard(liveView, true));
$('btnGrid').addEventListener('click', guard(tileTest, true));
$('btnCsv').addEventListener('click', exportCsv);
$('btnJson').addEventListener('click', exportJson);

$('btnFreeEarly').textContent = 'Live view';

for (const id of ['inDist', 'inDiag']) $(id).addEventListener('input', paintGeometry);
$('inCalib').addEventListener('change', () => {
  syncCalibButton();
  if (!S.calibrated) phase('calibrate', undefined, `0/${$('inCalib').value}`);
});
$('inValid').addEventListener('change', () => {
  if (!S.summary) phase('measure', undefined, `0/${$('inValid').value}`);
});
$('inKalman').addEventListener('change', () => S.running && webgazer.applyKalmanFilter($('inKalman').checked));
$('inLock').addEventListener('change', () => S.running && applyClickLock());
$('inReg').addEventListener('change', () => {
  if (!S.running) return;
  webgazer.setRegression($('inReg').value);
  S.calibrated = false;
  phase('calibrate', 'pending', `0/${$('inCalib').value}`);
  pill('warn', 'recalibrate needed');
});

addEventListener('keydown', (e) => {
  if (chamber.hidden) return;
  if (e.key === 'Escape') {
    S.abort = true; S.skip = true;
    trainTo = null; collector.on = false;
    closeChamber();
    view(S.summary ? 'report' : 'camera');
  } else if (e.key === ' ') {
    e.preventDefault();
    S.skip = true;
  } else if (e.key === 'h' && S.mode === 'live') {
    clearHeat();
  }
});

addEventListener('resize', () => {
  paintGeometry();
  sizeVideo();
  if (S.summary && !el('.view--report').hidden) drawPlot(S.summary);
});
paintGeometry();
syncCalibButton();
syncControls();
frame();

if (new URLSearchParams(location.search).has('demo')) syntheticRun();

if (!isSecureContext) {
  pill('error', 'insecure origin');
  $('btnStart').disabled = true;
  notice('bad', cameraError({}));
}
