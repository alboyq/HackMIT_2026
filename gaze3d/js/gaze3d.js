/**
 * gaze3d bench — drives the Python/GPU gaze server through calibration,
 * validation, a head-motion robustness test and a tile hit test, and reports
 * the error in pixels and degrees of visual angle.
 *
 * Coordinates: the server works in *screen* CSS px (the physical panel). The
 * page converts to viewport px with the window offset; fullscreen runs make the
 * two coincide.
 */
import { summarize, scoreGrid, toCsv, cssPxPerCm, pxToDeg, degToPx, cellOf, centroid, dist, spreadPoints } from './metrics.js';

const $ = (id) => document.getElementById(id);
const el = (sel, root = document) => root.querySelector(sel);
const wait = (ms) => new Promise((r) => setTimeout(r, ms));
const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
const fmt = (v, d = 2) => (Number.isFinite(v) ? v.toFixed(d) : '—');
const reduceMotion = matchMedia('(prefers-reduced-motion: reduce)').matches;

/* ── state ─────────────────────────────────────────────────────────────── */

const S = {
  running: false, calibrated: false, mode: 'idle',
  gaze: null, raw: null, frame: null,
  stamps: [], trail: [], runs: [], summary: null, gridResult: null, robust: null,
  calibReport: null, abort: false, skip: false, lastSeq: -1, hits: 0, misses: 0,
};
const collector = { on: false, samples: [], nulls: 0 };

/* ── server link ───────────────────────────────────────────────────────── */

let ws = null, wsReady = null;
const pending = new Map();
let msgId = 0;

function connect() {
  if (wsReady) return wsReady;
  wsReady = new Promise((resolve, reject) => {
    ws = new WebSocket(`ws://${location.host}/ws`);
    ws.onopen = () => resolve();
    ws.onerror = (e) => reject(new Error('server not reachable — is `./run.sh` running?'));
    ws.onclose = () => {
      wsReady = null; S.running = false;
      pill('error', 'server link lost');
      for (const p of pending.values()) p.reject(new Error('socket closed'));
      pending.clear();
    };
    ws.onmessage = (ev) => {
      const m = JSON.parse(ev.data);
      if (m.type === 'reply') {
        const p = pending.get(m.id);
        if (p) { pending.delete(m.id); m.ok ? p.resolve(m) : p.reject(new Error(m.error || 'command failed')); }
      } else if (m.type === 'frame') onFrame(m);
    };
  });
  return wsReady;
}

function send(cmd, extra = {}) {
  return new Promise((resolve, reject) => {
    const id = ++msgId;
    pending.set(id, { resolve, reject });
    ws.send(JSON.stringify({ cmd, id, ...extra }));
  });
}

/* ── geometry ──────────────────────────────────────────────────────────── */

function guessDiagonal() {
  const w = screen.width * devicePixelRatio, h = screen.height * devicePixelRatio;
  if (w === 3024 && h === 1964) return 14.2;   // MacBook Pro 14
  if (w === 3456 && h === 2234) return 16.2;   // MacBook Pro 16
  if (w === 2560 && h === 1664) return 13.6;   // MacBook Air 13 (M2+)
  if (w === 2880 && h === 1864) return 15.3;   // MacBook Air 15
  if (w === 2560 && h === 1600) return 13.3;
  return null;
}

/** Viewing distance is measured, not typed: latest calibrated face distance, else 55 cm. */
let measuredDistCm = null;
function geometry(distanceCm = measuredDistCm || 55) {
  const diagonalIn = Number($('inDiag').value) || 14.2;
  const pxPerCm = cssPxPerCm({ diagonalIn, screenWpx: screen.width, screenHpx: screen.height, dpr: devicePixelRatio || 1 });
  return { distanceCm, distanceMeasured: measuredDistCm != null, diagonalIn, pxPerCm, viewportW: innerWidth, viewportH: innerHeight, dpr: devicePixelRatio || 1,
    screenW: screen.width, screenH: screen.height, pitchMm: 10 / pxPerCm, camY: Number($('inCamY').value) || 0 };
}

/** viewport -> screen CSS px offset of this window. Zero in fullscreen. */
function winOffset() {
  if (document.fullscreenElement) return { x: 0, y: 0 };
  return { x: window.screenX, y: window.screenY + (window.outerHeight - window.innerHeight) };
}
const toScreen = (p) => { const o = winOffset(); return { x: p.x + o.x, y: p.y + o.y }; };
const toViewport = (p) => { const o = winOffset(); return { x: p.x - o.x, y: p.y - o.y }; };

async function pushGeometry() {
  const g = geometry();
  await send('geometry', { width_px: g.screenW, height_px: g.screenH, pitch_mm: g.pitchMm, cam_offset_px: [0, g.camY] });
}

function paintGeometry() {
  const g = geometry();
  const onePx = degToPx(1, g.pxPerCm, g.distanceCm);
  $('roNative').textContent = `${Math.round(screen.width * g.dpr)}×${Math.round(screen.height * g.dpr)}`;
  $('roPxCm').innerHTML = `${fmt(g.pxPerCm, 1)}<small>px/cm</small>`;
  $('roDegPx').innerHTML = `${fmt(onePx, 1)}<small>px</small>`;
  const EYE_X = 44, SCREEN_X = 400, MID = 100, RUN = SCREEN_X - EYE_X;
  const halfW = g.screenW / 2 / g.pxPerCm;
  const halfAngle = Math.atan(halfW / g.distanceCm);
  const half = clamp(Math.tan(halfAngle) * RUN, 8, 88);
  $('geoWedge').setAttribute('points', `${EYE_X},${MID} ${SCREEN_X},${MID - half} ${SCREEN_X},${MID + half}`);
  $('geoScreen').setAttribute('y1', MID - half);
  $('geoScreen').setAttribute('y2', MID + half);
  $('geoDist').textContent = `${Math.round(g.distanceCm)} cm${g.distanceMeasured ? '' : ' (assumed)'}`;
  $('geoSpan').textContent = `${fmt(halfAngle * 2 * 180 / Math.PI, 0)}°`;
}

/* ── rail ──────────────────────────────────────────────────────────────── */

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
function pill(state, text) { $('pill').dataset.state = state; $('pillText').textContent = text; }
function notice(kind, html) {
  const n = $('startNote');
  n.className = kind ? `notice notice--${kind}` : 'fineprint';
  n.innerHTML = html;
}

/* ── frames from the server ────────────────────────────────────────────── */

function onFrame(m) {
  S.frame = m;
  if (m.status && m.status !== S.status) {
    S.status = m.status;
    $('roState').textContent = m.status;
  }
  if (m.models && !S.modelsShown) {
    S.modelsShown = true;
    $('modelsBlock').hidden = false;
    $('mastSub').textContent = `${m.models.length} nets · ${m.device || 'gpu'}`;
  }
  if (m.ms && m.models) {
    $('modelsList').innerHTML = m.models.map((n) => `<li>${n}<span>${fmt(m.ms[n], 1)} ms</span></li>`).join('');
  }
  if (m.disk && !S.calibrated) {
    const b = $('btnRecover');
    b.hidden = !(m.disk.n > 0);
    if (m.disk.n > 0) el('.btn__k', b).textContent = `${m.disk.n} frames · ${m.disk.points} targets`;
  } else if (S.calibrated) {
    $('btnRecover').hidden = true;
  }
  if (m.seq === undefined || m.seq === S.lastSeq) return;
  S.lastSeq = m.seq;
  const t = performance.now();
  const useSmooth = $('inSmooth').checked;
  const p = useSmooth ? m.gaze : m.raw;

  if (!m.face || !p) {
    S.misses++;
    if (collector.on) collector.nulls++;
    S.gaze = null;
    return;
  }
  S.hits++;
  if (m.head?.dist) {
    measuredDistCm = m.head.dist / 10;
    if (!S.geoPaintedAt || t - S.geoPaintedAt > 1000) { S.geoPaintedAt = t; paintGeometry(); }
  }
  const v = toViewport({ x: p[0], y: p[1] });
  const rv = m.raw ? toViewport({ x: m.raw[0], y: m.raw[1] }) : null;
  S.gaze = { x: v.x, y: v.y, t, fix: m.fixating, blink: m.blink };
  S.raw = rv;
  S.stamps.push(t); if (S.stamps.length > 40) S.stamps.shift();
  if (collector.on) collector.samples.push({ x: v.x, y: v.y, t, head: m.head, raw: rv });
  S.trail.push({ x: v.x, y: v.y, t }); if (S.trail.length > 24) S.trail.shift();
}

function sampleRate() {
  if (S.stamps.length < 4) return NaN;
  const span = S.stamps[S.stamps.length - 1] - S.stamps[0];
  return span > 0 ? ((S.stamps.length - 1) / span) * 1000 : NaN;
}
function openWindow() { collector.on = true; collector.samples = []; collector.nulls = 0; }
function closeWindow() { collector.on = false; return { samples: collector.samples.slice(), nulls: collector.nulls }; }

/* ── camera ────────────────────────────────────────────────────────────── */

async function startCamera() {
  const btn = $('btnStart');
  btn.disabled = true;
  btn.textContent = 'Connecting…';
  try {
    await connect();
    pill('warn', 'loading models');
    btn.textContent = 'Loading models on GPU…';
    notice('wait', '<b>Loading the gaze networks onto the GPU.</b> ~3 GB of weights; 10–20 s the first time. ' +
      'macOS will ask for camera permission for the terminal that runs the server — allow it.');
    await pushGeometry();
    await send('start');
    S.running = true;
    $('camslot').innerHTML = `<img src="/preview.mjpg?${Date.now()}" alt="camera preview">`;
    $('cropslot').hidden = false;
    $('cropImg').src = `/crop.mjpg?${Date.now()}`;
    pill('live', 'tracking');
    phase('camera', 'done', 'live');
    phase('calibrate', 'pending', '0');
    notice(null, 'Nothing leaves this machine — inference runs in the local server process.');
    view('camera');
    syncControls();
  } catch (err) {
    console.error(err);
    btn.disabled = false;
    btn.textContent = 'Start camera';
    pill('error', 'server error');
    notice('bad', `<b>Could not start.</b> ${err.message}. Run <code>./run.sh</code> in the project and reload.`);
  }
}

/* ── live readout + marker ─────────────────────────────────────────────── */

const layer = $('gazelayer');
const lctx = layer.getContext('2d');
const heat = $('heat');
const hctx = heat.getContext('2d');
let heatArmed = false;

function fitCanvas(canvas, ctx) {
  const dpr = devicePixelRatio || 1;
  const w = canvas.clientWidth || innerWidth, h = canvas.clientHeight || innerHeight;
  if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
    canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr);
  }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { w, h };
}

function frame() {
  requestAnimationFrame(frame);
  const m = S.frame;
  const hz = sampleRate();
  $('roX').textContent = S.gaze ? Math.round(S.gaze.x) : '—';
  $('roY').textContent = S.gaze ? Math.round(S.gaze.y) : '—';
  $('roHz').innerHTML = `${fmt(hz, 1)}<small>Hz</small>`;
  if (m) {
    $('roGpu').innerHTML = `${fmt(m.ms?.gpu, 0)}<small>ms</small>`;
    $('roHead').textContent = m.head ? `${fmt(m.head.yaw, 0)}° · ${fmt(m.head.pitch, 0)}°` : '—';
    $('roDist').innerHTML = `${fmt(m.head ? m.head.dist / 10 : NaN, 0)}<small>cm</small>`;
    $('roAgree').innerHTML = `${fmt(m.agreement_deg, 1)}<small>°</small>`;
    if (m.head?.focal) $('roFocal').innerHTML = `${fmt(m.head.focal, 0)}<small>px</small>`;
    if (m.head?.dist) $('roDistCal').innerHTML = `${fmt(m.head.dist / 10, 0)}<small>cm</small>`;
    $('roState').textContent = !m.face ? 'no face' : m.blink ? 'blink' : m.armed ? `sampling ${m.armed_count}` : m.fixating ? 'fixation' : 'moving';
  }

  const showMarker = S.mode === 'live' || S.mode === 'grid';
  layer.classList.toggle('gazelayer--on', showMarker);
  if (!showMarker) return;
  const { w, h } = fitCanvas(layer, lctx);
  lctx.clearRect(0, 0, w, h);
  if (!S.gaze) return;
  const now = performance.now();
  const fresh = S.trail.filter((p) => now - p.t < 500);
  lctx.lineCap = 'round';
  for (let i = 1; i < fresh.length; i++) {
    const a = (i / fresh.length) * 0.5;
    lctx.strokeStyle = `rgba(213,232,90,${a.toFixed(3)})`;
    lctx.lineWidth = 1 + (i / fresh.length) * 3;
    lctx.beginPath(); lctx.moveTo(fresh[i - 1].x, fresh[i - 1].y); lctx.lineTo(fresh[i].x, fresh[i].y); lctx.stroke();
  }
  const { x, y } = S.gaze;
  const r = S.summary && Number.isFinite(S.summary.accuracyPx) ? clamp(S.summary.accuracyPx, 12, 260) : 34;
  lctx.strokeStyle = 'rgba(76,125,255,.5)'; lctx.lineWidth = 1; lctx.setLineDash([4, 5]);
  lctx.beginPath(); lctx.arc(x, y, r, 0, Math.PI * 2); lctx.stroke(); lctx.setLineDash([]);
  if (S.raw) { lctx.fillStyle = 'rgba(76,125,255,.6)'; lctx.beginPath(); lctx.arc(S.raw.x, S.raw.y, 2.5, 0, Math.PI * 2); lctx.fill(); }
  lctx.strokeStyle = S.gaze.blink ? 'rgba(232,113,140,.9)' : 'rgba(213,232,90,.9)'; lctx.lineWidth = 1.2;
  lctx.beginPath(); lctx.moveTo(x - 11, y); lctx.lineTo(x + 11, y); lctx.moveTo(x, y - 11); lctx.lineTo(x, y + 11); lctx.stroke();
  lctx.fillStyle = '#d5e85a'; lctx.beginPath(); lctx.arc(x, y, 3.2, 0, Math.PI * 2); lctx.fill();
  if (heatArmed) paintHeat(x, y);
}
function paintHeat(x, y) {
  fitCanvas(heat, hctx);
  const g = hctx.createRadialGradient(x, y, 0, x, y, 46);
  g.addColorStop(0, 'rgba(213,232,90,.055)'); g.addColorStop(1, 'rgba(213,232,90,0)');
  hctx.fillStyle = g; hctx.beginPath(); hctx.arc(x, y, 46, 0, Math.PI * 2); hctx.fill();
}
function clearHeat() { const { w, h } = fitCanvas(heat, hctx); hctx.clearRect(0, 0, w, h); }

/* ── chamber ───────────────────────────────────────────────────────────── */

const chamber = $('chamber');
const targetEl = $('target');
const ringEl = $('targetRing');

async function openChamber(mode, hud, keys = 'space · skip point') {
  S.mode = mode; S.abort = false;
  chamber.hidden = false;
  $('chamberHud').textContent = hud;
  $('chamberKeys').textContent = keys;
  if ($('inFull').checked && !document.fullscreenElement) {
    try { await chamber.requestFullscreen({ navigationUI: 'hide' }); await wait(350); } catch (e) { console.warn('fullscreen refused', e); }
  }
}
function closeChamber({ keepFullscreen = false } = {}) {
  chamber.hidden = true; targetEl.hidden = true; $('motion').hidden = true;
  $('chamberGrid').hidden = true; $('chamberGrid').textContent = '';
  heatArmed = false; clearHeat(); S.mode = 'idle';
  if (!keepFullscreen && document.fullscreenElement) {
    S.fsExiting = true;                       // our own exit, not a user abort
    document.exitFullscreen().catch(() => { S.fsExiting = false; });
  }
}
function placeTarget(p) { targetEl.style.left = `${p.x}px`; targetEl.style.top = `${p.y}px`; targetEl.hidden = false; }
/** Put the motion cue right next to the target (below it, or above near the bottom edge). */
function placeMotionCue(p, text) {
  const m = $('motion');
  const below = p.y < innerHeight * 0.8;
  m.classList.toggle('motion--above', !below);
  m.style.left = `${p.x}px`;
  m.style.top = `${p.y + (below ? 64 : -64)}px`;
  $('motionCue').textContent = text;
  $('motionBar').style.setProperty('--p', '0%');
  m.hidden = false;
}

function contractRing(ms, from = 50, to = 13) {
  return new Promise((resolve) => {
    if (reduceMotion) { ringEl.setAttribute('r', to); setTimeout(resolve, ms); return; }
    const t0 = performance.now();
    const step = () => {
      const k = clamp((performance.now() - t0) / ms, 0, 1);
      ringEl.setAttribute('r', (from + (to - from) * (1 - (1 - k) ** 3)).toFixed(1));
      if (k < 1 && !S.abort && !S.skip) requestAnimationFrame(step); else resolve();
    };
    requestAnimationFrame(step);
  });
}

const SETTLE_MS = 900, DWELL_MS = 1400, GAP_MS = 130, DYN_MS = 6000;

/** Static fixation points. `arm` sends the target to the server for sample collection. */
async function runPoints(points, { arm = false, dwell = DWELL_MS, onPoint, progress, pointOffset = 0 }) {
  for (let i = 0; i < points.length; i++) {
    if (S.abort) return false;
    S.skip = false;
    const p = points[i];
    targetEl.classList.remove('target--armed', 'target--dyn');
    placeTarget(p);
    progress?.(i, points.length);
    await contractRing(SETTLE_MS);
    if (S.abort) return false;
    targetEl.classList.add('target--armed');
    openWindow();
    if (arm) await send('arm', { point: pointOffset + i, target: [toScreen(p).x, toScreen(p).y], kind: 'static' });
    await wait(dwell);
    let n = 0;
    if (arm) n = (await send('disarm')).n;
    const got = closeWindow();
    if (S.abort) return false;
    onPoint?.(p, got, i, n);
    targetEl.hidden = true;
    await wait(GAP_MS);
  }
  progress?.(points.length, points.length);
  return true;
}

const MOTION_CUES = [
  { kind: 'yaw', text: 'turn your head slowly left … and right' },
  { kind: 'pitch', text: 'tilt your head up … and down' },
  { kind: 'depth', text: 'lean in closer … then sit back' },
  { kind: 'lateral', text: 'slide sideways in your seat, both ways' },
  { kind: 'roll', text: 'roll your head gently, ear toward shoulder' },
];

/** Build the motion trial list: `per` random spread locations for every cue, shuffled. */
function motionTrials(per, statics) {
  const trials = [];
  let chosen = [];
  for (const cue of MOTION_CUES) {
    const pts = spreadPoints(per, { w: innerWidth, h: innerHeight, margin: 0.12, existing: chosen });
    chosen = chosen.concat(pts);
    for (const p of pts) trials.push({ ...cue, p });
  }
  return shuffle(trials);
}

/** Head-motion points: keep fixating while moving the head through a cue. */
async function runMotionPoints(trials, { pointOffset = 0, progress }) {
  for (let i = 0; i < trials.length; i++) {
    if (S.abort) return false;
    S.skip = false;
    const { p, text, kind } = trials[i];
    targetEl.classList.remove('target--armed');
    targetEl.classList.add('target--dyn');
    placeTarget(p);
    progress?.(i, trials.length);
    placeMotionCue(p, text);
    await contractRing(SETTLE_MS);
    if (S.abort) return false;
    targetEl.classList.add('target--armed');
    await send('arm', { point: pointOffset + i, target: [toScreen(p).x, toScreen(p).y], kind: `dynamic:${kind}` });
    const t0 = performance.now();
    while (performance.now() - t0 < DYN_MS && !S.abort && !S.skip) {
      $('motionBar').style.setProperty('--p', `${((performance.now() - t0) / DYN_MS * 100).toFixed(0)}%`);
      await wait(50);
    }
    await send('disarm');
    $('motion').hidden = true;
    targetEl.hidden = true;
    if (S.abort) return false;
    await wait(GAP_MS);
  }
  targetEl.classList.remove('target--dyn');
  progress?.(points.length, points.length);
  return true;
}

/* ── layouts ───────────────────────────────────────────────────────────── */

const spread = (fracs) => {
  const w = innerWidth, h = innerHeight;
  return fracs.map(([fx, fy]) => ({ x: Math.round(clamp(fx * w, 44, w - 44)), y: Math.round(clamp(fy * h, 44, h - 44)) }));
};
const GRID3 = (a, b, c) => [[a, a], [b, a], [c, a], [a, b], [b, b], [c, b], [a, c], [b, c], [c, c]];
const CALIB = {
  5: [[.5, .5], [.12, .12], [.88, .12], [.12, .88], [.88, .88]],
  9: GRID3(.08, .5, .92),
  13: [...GRID3(.08, .5, .92), [.3, .3], [.7, .3], [.3, .7], [.7, .7]],
};
/** Head-motion phase: locations per movement type. Each type is shown at several random,
 *  well-separated places so pose and gaze angle are not confounded, interleaved randomly. */
const MOTION_PER_CUE = { 5: 1, 9: 2, 13: 3 };
const VALID = {
  5: [[.5, .5], [.25, .25], [.75, .25], [.25, .75], [.75, .75]],
  9: GRID3(.25, .5, .75),
  13: [...GRID3(.25, .5, .75), [.04, .04], [.96, .04], [.04, .96], [.96, .96]],
};
const shuffle = (a) => a.map((v) => [Math.random(), v]).sort((x, y) => x[0] - y[0]).map(([, v]) => v);

/* ── calibration ───────────────────────────────────────────────────────── */

function protocol() {
  const v = $('inCalib').value;
  return { n: Number(v.replace('m', '')), motion: v.endsWith('m') };
}

async function calibrate() {
  const { n, motion } = protocol();
  const nMotion = motion ? MOTION_CUES.length * MOTION_PER_CUE[n] : 0;
  const total = n + nMotion;
  await send('clear');
  await pushGeometry();
  S.calibrated = false; S.summary = null; S.gridResult = null; S.robust = null; S.runs = []; S.calibReport = null;
  $('tileOut')?.remove(); $('runs').hidden = true;
  syncControls();
  phase('calibrate', 'active', `0/${total}`);
  phase('measure', 'locked', '—'); phase('report', 'locked', '—');

  await openChamber('calibrate', 'Look at the centre of each target. When the ring locks and turns yellow, hold your gaze there.');

  const pts = shuffle(spread(CALIB[n]));
  let ok = await runPoints(pts, {
    arm: true,
    progress: (i) => { phase('calibrate', 'active', `${i}/${total}`); $('chamberHud').textContent = i < n ? `Point ${i + 1} of ${total} — look at the centre and hold.` : 'Now with head movement.'; },
  });
  if (ok && motion) {
    const mp = motionTrials(MOTION_PER_CUE[n], pts);
    ok = await runMotionPoints(mp, {
      pointOffset: n,
      progress: (i) => { phase('calibrate', 'active', `${n + i}/${total}`); $('chamberHud').textContent = i < mp.length ? `Point ${n + i + 1} of ${total} — keep your eyes on the target while you move your head.` : 'Fitting…'; },
    });
  }
  if (!ok) { closeChamber(); phase('calibrate', 'pending', '0'); view('camera'); return; }

  $('chamberHud').textContent = 'Fitting the mapping on the GPU host…';
  targetEl.hidden = true;
  let report;
  try {
    report = (await send('fit')).report;
  } catch (e) {
    closeChamber(); pill('error', 'fit failed'); console.error(e); phase('calibrate', 'pending', '0'); view('camera'); return;
  }
  closeChamber({ keepFullscreen: true });     // validation follows immediately in the same fullscreen session
  S.calibReport = report;
  S.calibrated = true;
  showCalibSummary(report);
  S.hits = S.misses = 0;
  phase('calibrate', 'done', `${total} pts`);
  phase('measure', 'pending', `0/${$('inValid').value}`);
  syncControls();
  pill('live', `calibrated · cv ${fmt(pxToDeg(report.cv_err_px, geometry().pxPerCm, geometry().distanceCm), 1)}°`);
  await validate();
}

/* ── validation ────────────────────────────────────────────────────────── */

async function validate(label = null) {
  const n = Number($('inValid').value);
  const pts = shuffle(spread(VALID[n]));
  const collected = [];
  phase('measure', 'active', `0/${n}`);
  await openChamber('validate', 'Nothing is learned now — the error at each target is being recorded. No marker is shown.');
  await send('reset_filter');
  const ok = await runPoints(pts, {
    onPoint: (p, got) => collected.push({ target: p, samples: got.samples, nulls: got.nulls, windowMs: DWELL_MS }),
    progress: (i) => { phase('measure', 'active', `${i}/${n}`); $('chamberHud').textContent = `Target ${Math.min(i + 1, n)} of ${n} — recording.`; },
  });
  closeChamber();
  if (!ok) { phase('measure', 'pending', `0/${n}`); view(S.summary ? 'report' : 'camera'); return; }
  const hp = meanHead(collected);
  S.summary = summarize(collected, geometry(hp ? hp.z / 10 : undefined));
  S.summary.headPose = hp;
  S.runs.push({ at: new Date(), acc: S.summary.accuracyDeg, px: S.summary.accuracyPx, hz: S.summary.hz, head: S.summary.headPose, label });
  phase('measure', 'done', `${n} pts`);
  phase('report', 'done', `${fmt(S.summary.accuracyDeg, 1)}°`);
  syncControls();
  view('report');
  renderReport();
}

function meanHead(collected) {
  const hs = collected.flatMap((c) => c.samples.map((s) => s.head)).filter(Boolean);
  if (!hs.length) return null;
  const m = (k) => hs.reduce((a, h) => a + h[k], 0) / hs.length;
  return { yaw: m('yaw'), pitch: m('pitch'), x: m('x'), y: m('y'), z: m('dist') };
}

/* ── head-motion robustness test ───────────────────────────────────────── */

async function robustTest() {
  const plan = shuffle(MOTION_CUES.flatMap((c) => [c, c]));          // every movement type twice
  const pts = spreadPoints(plan.length, { w: innerWidth, h: innerHeight, margin: 0.12 });
  const cues = plan.map((c) => c.text);
  const trials = [];
  await openChamber('robust', 'Keep your eyes on the target while moving your head as the cue says. This measures how well the mapping holds up off the calibration pose.');
  await send('reset_filter');
  for (let i = 0; i < pts.length; i++) {
    if (S.abort) break;
    S.skip = false;
    targetEl.classList.add('target--dyn');
    placeTarget(pts[i]);
    placeMotionCue(pts[i], cues[i]);
    $('chamberHud').textContent = `Target ${i + 1} of ${pts.length} — eyes on the target, head moving.`;
    await contractRing(SETTLE_MS);
    if (S.abort) break;
    openWindow();
    const t0 = performance.now();
    while (performance.now() - t0 < DYN_MS && !S.abort && !S.skip) {
      $('motionBar').style.setProperty('--p', `${((performance.now() - t0) / DYN_MS * 100).toFixed(0)}%`);
      await wait(50);
    }
    const got = closeWindow();
    trials.push({ target: pts[i], cue: plan[i].kind, samples: got.samples, nulls: got.nulls, windowMs: DYN_MS });
    $('motion').hidden = true; targetEl.hidden = true;
    await wait(GAP_MS);
  }
  targetEl.classList.remove('target--dyn');
  closeChamber();
  if (!trials.length) { view('report'); return; }
  S.robust = scoreRobust(trials, geometry());
  view('report');
  renderReport();
}

/** Per-sample error binned by head displacement from the calibration-time pose. */
function scoreRobust(trials, g) {
  const all = [];
  for (const t of trials) for (const s of t.samples) {
    if (!s.head) continue;
    all.push({ err: dist(s, t.target), yaw: s.head.yaw, pitch: s.head.pitch, z: s.head.dist, x: s.head.x, cue: t.cue });
  }
  if (!all.length) return null;
  const ref = S.runs[0]?.head || { yaw: 0, pitch: 0, z: 550, x: 0 };
  const bins = [['< 5°', 0, 5], ['5–12°', 5, 12], ['12–20°', 12, 20], ['> 20°', 20, 1e9]].map(([label, lo, hi]) => {
    const xs = all.filter((s) => { const d = Math.hypot(s.yaw - ref.yaw, s.pitch - ref.pitch); return d >= lo && d < hi; });
    const mean = xs.length ? xs.reduce((a, s) => a + s.err, 0) / xs.length : NaN;
    return { label, n: xs.length, errPx: mean, errDeg: pxToDeg(mean, g.pxPerCm, g.distanceCm) };
  });
  const depth = [['closer than −8 cm', -1e9, -80], ['within ±8 cm', -80, 80], ['farther than +8 cm', 80, 1e9]].map(([label, lo, hi]) => {
    const xs = all.filter((s) => s.z - ref.z >= lo && s.z - ref.z < hi);
    const mean = xs.length ? xs.reduce((a, s) => a + s.err, 0) / xs.length : NaN;
    return { label, n: xs.length, errPx: mean, errDeg: pxToDeg(mean, g.pxPerCm, g.distanceCm) };
  });
  const byCue = MOTION_CUES.map((c) => {
    const xs = all.filter((s) => s.cue === c.kind);
    const m = xs.length ? xs.reduce((a, s) => a + s.err, 0) / xs.length : NaN;
    return { label: c.kind, n: xs.length, errPx: m, errDeg: pxToDeg(m, g.pxPerCm, g.distanceCm) };
  });
  const mean = all.reduce((a, s) => a + s.err, 0) / all.length;
  const p95 = all.map((s) => s.err).sort((a, b) => a - b)[Math.floor(0.95 * (all.length - 1))];
  return { n: all.length, errPx: mean, errDeg: pxToDeg(mean, g.pxPerCm, g.distanceCm), p95Px: p95, p95Deg: pxToDeg(p95, g.pxPerCm, g.distanceCm),
    bins, depth, byCue, ref, trials: trials.map((t) => ({ target: t.target, n: t.samples.length, errPx: t.samples.length ? t.samples.reduce((a, s) => a + dist(s, t.target), 0) / t.samples.length : NaN })) };
}

/* ── report ────────────────────────────────────────────────────────────── */

const grade = (deg) => (deg <= 1.5 ? '' : deg <= 3 ? 'mid' : 'bad');

function renderReport() {
  const s = S.summary;
  const g = s.geom;
  $('repStamp').textContent = S.runs[S.runs.length - 1].at.toLocaleTimeString();
  $('vAcc').textContent = fmt(s.accuracyDeg, 2);
  $('vAccPx').textContent = fmt(s.accuracyPx, 0);
  $('vWorst').textContent = fmt(s.worstDeg, 2);
  $('vHz').textContent = fmt(s.hz, 1);
  $('vRead').innerHTML = verdict(s);

  const order = ['top left', 'top centre', 'top right', 'middle left', 'centre', 'middle right', 'bottom left', 'bottom centre', 'bottom right'];
  const rows = s.regions.slice().sort((a, b) => order.indexOf(a.region) - order.indexOf(b.region));
  el('tbody', $('tblRegion')).innerHTML = `<tr><th>region</th><th>error</th><th>px</th><th>n</th></tr>` +
    rows.map((r) => `<tr><td>${r.region}</td><td class="${grade(r.errDeg)}"><b>${fmt(r.errDeg, 2)}°</b></td><td>${fmt(r.errPx, 0)}</td><td>${r.n}</td></tr>`).join('');

  const both = (px) => `${fmt(pxToDeg(px, g.pxPerCm, g.distanceCm), 2)}° · ${fmt(px, 0)} px`;
  const f = S.frame || {};
  const qual = [
    ['spread about centroid', both(s.spreadPx)],
    ['sample-to-sample noise', both(s.noisePx)],
    ['95th percentile error', both(s.p95Px)],
    ['systematic offset', `${s.bias.x >= 0 ? 'right' : 'left'} ${fmt(Math.abs(s.bias.x), 0)} px, ${s.bias.y >= 0 ? 'down' : 'up'} ${fmt(Math.abs(s.bias.y), 0)} px`],
    ['dropped frames', `${fmt(s.lossPct, 1)} %${s.dropped ? ` · ${s.dropped} target(s) lost` : ''}`],
    ['head pose during run', s.headPose ? `yaw ${fmt(s.headPose.yaw, 0)}° pitch ${fmt(s.headPose.pitch, 0)}° at ${fmt(s.headPose.z / 10, 0)} cm (measured)` : '—'],
    ['gpu time per frame', `${fmt(f.ms?.gpu, 0)} ms · ${fmt(f.ms?.face, 0)} ms landmarks`],
    ['viewport', `${g.viewportW}×${g.viewportH} at ${fmt(g.pxPerCm, 1)} px/cm`],
  ];
  el('tbody', $('tblQual')).innerHTML = `<tr><th>measure</th><th>value</th></tr>` + qual.map(([k, v]) => `<tr><td>${k}</td><td><b>${v}</b></td></tr>`).join('');

  $('tblPoints').innerHTML = `<thead><tr><th>#</th><th>region</th><th>target</th><th>predicted</th><th>error</th><th>px</th><th>spread</th><th>n</th><th>loss</th></tr></thead><tbody>` +
    s.points.map((p, i) => `<tr><td>${i + 1}</td><td>${p.region}</td><td>${Math.round(p.target.x)}, ${Math.round(p.target.y)}</td>
      <td>${p.centroid ? `${Math.round(p.centroid.x)}, ${Math.round(p.centroid.y)}` : 'lost'}</td>
      <td class="${grade(p.errDeg)}"><b>${fmt(p.errDeg, 2)}°</b></td><td>${fmt(p.errPx, 0)}</td><td>${fmt(p.spreadPx, 0)}</td>
      <td>${p.n}</td><td class="${p.lossPct > 25 ? 'bad' : ''}">${fmt(p.lossPct, 0)}%</td></tr>`).join('') + '</tbody>';

  renderCalib(g);
  renderRobust(g);

  if (S.runs.length > 1) {
    $('runs').hidden = false;
    const first = S.runs[0].acc;
    $('runsList').innerHTML = S.runs.map((r, i) => {
      const d = r.acc - first;
      const drift = i === 0 ? 'baseline' : `${d >= 0 ? '+' : '−'}${fmt(Math.abs(d), 2)}° vs. run 1`;
      const hp = r.head ? ` · yaw ${fmt(r.head.yaw, 0)}° pitch ${fmt(r.head.pitch, 0)}° ${fmt(r.head.z / 10, 0)} cm` : '';
      return `<li><i>run ${i + 1}</i><span>${r.at.toLocaleTimeString()}${hp}</span><b>${fmt(r.acc, 2)}°</b><i>${drift}</i></li>`;
    }).join('');
  }
  drawPlot(s);
}

function showCalibSummary(r) {
  const g = geometry();
  const deg = (px) => fmt(pxToDeg(px, g.pxPerCm, g.distanceCm), 2);
  $('calibBlock').hidden = false;
  $('calibList').innerHTML = [
    ['pipeline', r.chosen], ['held-out error', `${deg(r.cv_err_px)}° · ${fmt(r.cv_err_px, 0)} px`],
    ['fit error', `${deg(r.fit_err_px)}°`], ['gain / depth scale', `×${fmt(Math.exp(r.params.log_depth), 2)}`],
    ['kappa', `${fmt(r.params.kappa[0] * 57.3, 1)}° / ${fmt(r.params.kappa[1] * 57.3, 1)}°`],
    ['frames · targets', `${r.n_samples} · ${r.n_points}`],
  ].map(([k, v]) => `<li>${k}<span>${v}</span></li>`).join('');
}

function renderCalib(g) {
  const r = S.calibReport;
  const box = $('calibOut');
  if (!r) { box.hidden = true; return; }
  box.hidden = false;
  const deg = (px) => fmt(pxToDeg(px, g.pxPerCm, g.distanceCm), 2);
  const p = r.params;
  $('calibNote').innerHTML = `${r.n_samples} frames over ${r.n_points} targets. Chosen by leave-one-target-out: <b>${r.chosen}</b> ` +
    `(held-out ${deg(r.cv_err_px)}° · fit ${deg(r.fit_err_px)}°). Kappa ${fmt(p.kappa[0] * 57.3, 1)}°/${fmt(p.kappa[1] * 57.3, 1)}° in the ${p.kappa_mode} frame, ` +
    `screen tilt ${fmt(Math.hypot(...p.omega) * 57.3, 1)}°, screen origin shift ${fmt(Math.hypot(p.t[0] - (r.t_prior?.[0] ?? p.t[0]), 0), 0)} mm, gain/depth scale ×${fmt(Math.exp(p.log_depth), 2)} (absorbs the nets' angular under-swing; distance readouts use raw PnP). Fit ${fmt(r.fit_ms / 1000, 1)} s.`;
  $('tblCalib').innerHTML = `<thead><tr><th>pipeline</th><th>held-out error</th><th>px</th><th>fit error</th></tr></thead><tbody>` +
    r.candidates.slice(0, 8).map((c) => `<tr><td>${c.name}</td><td class="${grade(pxToDeg(c.cv_err_px, g.pxPerCm, g.distanceCm))}"><b>${deg(c.cv_err_px)}°</b></td><td>${fmt(c.cv_err_px, 0)}</td><td>${deg(c.fit_err_px)}°</td></tr>`).join('') + '</tbody>';
}

function renderRobust(g) {
  $('robustOut')?.remove();
  const r = S.robust;
  if (!r) return;
  const sec = document.createElement('section');
  sec.className = 'tbl tbl--wide'; sec.id = 'robustOut';
  const row = (b) => `<tr><td>${b.label}</td><td class="${grade(b.errDeg)}"><b>${fmt(b.errDeg, 2)}°</b></td><td>${fmt(b.errPx, 0)}</td><td>${b.n}</td></tr>`;
  sec.innerHTML = `<h3 class="block__h">Head-motion test</h3>
    <p class="block__note">${r.n} frames while moving the head with eyes fixed on a target. Mean error <b>${fmt(r.errDeg, 2)}°</b> (${fmt(r.errPx, 0)} px), 95th percentile <b>${fmt(r.p95Deg, 2)}°</b>. Binned by head rotation away from the pose held during the first measurement (yaw ${fmt(r.ref.yaw, 0)}°, pitch ${fmt(r.ref.pitch, 0)}°, ${fmt(r.ref.z / 10, 0)} cm).</p>
    <div class="tbl__scroll"><table><thead><tr><th>head rotation</th><th>error</th><th>px</th><th>n</th></tr></thead><tbody>${r.bins.map(row).join('')}</tbody>
    <thead><tr><th>distance change</th><th>error</th><th>px</th><th>n</th></tr></thead><tbody>${r.depth.map(row).join('')}</tbody>
    <thead><tr><th>movement type</th><th>error</th><th>px</th><th>n</th></tr></thead><tbody>${r.byCue.map(row).join('')}</tbody></table></div>`;
  $('calibOut').after(sec);
}

function verdict(s) {
  const d = s.accuracyDeg;
  const target = Math.round(s.accuracyPx * 2);
  let head;
  if (d <= 1.0) head = 'Research-tracker territory for a webcam.';
  else if (d <= 2.0) head = 'Excellent for a webcam — around the best published calibrated webcam results.';
  else if (d <= 3) head = 'Typical of a good appearance-based tracker after calibration.';
  else if (d <= 5) head = 'Coarse. Check the lighting, distance setting and that calibration fixations were steady.';
  else head = 'Something is off — recalibrate, and check that Center Stage is disabled.';
  const use = d <= 1.5 ? 'Fine enough to tell which line you are reading.'
    : d <= 3 ? 'Good for regions — which card, which column — not for anything small.'
      : 'Only coarse regions are trustworthy.';
  const skew = Math.hypot(s.bias.x, s.bias.y) > s.accuracyPx * 0.7
    ? ` The error is mostly a constant offset (${s.bias.x >= 0 ? 'right' : 'left'} and ${s.bias.y >= 0 ? 'down' : 'up'}); with the 3D mapping that usually points at the screen geometry inputs rather than head movement.`
    : '';
  return `${head} A prediction lands about <b>${fmt(s.accuracyPx, 0)} px</b> (<b>${fmt(d, 2)}°</b>) from where you were looking, so the smallest thing you could reliably point at is roughly <b>${target}×${target} px</b>. ${use}${skew}`;
}

function drawPlot(s) {
  const cv = $('plot');
  const dpr = devicePixelRatio || 1;
  const cssW = cv.clientWidth || 900;
  const g = s.geom;
  const GUTTER = 26;
  const screenH = Math.round(cssW * (g.viewportH / g.viewportW));
  const cssH = screenH + GUTTER;
  cv.style.height = `${cssH}px`; cv.width = Math.round(cssW * dpr); cv.height = Math.round(cssH * dpr);
  const c = cv.getContext('2d'); c.setTransform(dpr, 0, 0, dpr, 0, 0);
  const k = cssW / g.viewportW;
  const X = (v) => v * k, Y = (v) => v * k;
  c.clearRect(0, 0, cssW, cssH);
  c.strokeStyle = '#1f2c39'; c.lineWidth = 1;
  for (let i = 1; i < 3; i++) {
    c.beginPath(); c.moveTo(Math.round(cssW * i / 3) + .5, 0); c.lineTo(Math.round(cssW * i / 3) + .5, screenH);
    c.moveTo(0, Math.round(screenH * i / 3) + .5); c.lineTo(cssW, Math.round(screenH * i / 3) + .5); c.stroke();
  }
  c.strokeStyle = '#2b3b4b'; c.beginPath(); c.moveTo(0, screenH + .5); c.lineTo(cssW, screenH + .5); c.stroke();
  c.save(); c.beginPath(); c.rect(0, 0, cssW, screenH); c.clip();
  for (const p of s.points) {
    const tx = X(p.target.x), ty = Y(p.target.y);
    c.fillStyle = 'rgba(213,232,90,.5)';
    for (const sm of p.samples) { c.beginPath(); c.arc(X(sm.x), Y(sm.y), 1.7, 0, Math.PI * 2); c.fill(); }
    if (Number.isFinite(p.errPx) && p.centroid) {
      c.strokeStyle = 'rgba(232,113,140,.42)'; c.setLineDash([3, 4]); c.lineWidth = 1;
      c.beginPath(); c.arc(tx, ty, Math.max(1, X(p.errPx)), 0, Math.PI * 2); c.stroke(); c.setLineDash([]);
      c.strokeStyle = '#e8718c'; c.lineWidth = 1.4;
      c.beginPath(); c.moveTo(tx, ty); c.lineTo(X(p.centroid.x), Y(p.centroid.y)); c.stroke();
      c.fillStyle = '#e8718c'; c.beginPath(); c.arc(X(p.centroid.x), Y(p.centroid.y), 2.6, 0, Math.PI * 2); c.fill();
    }
    c.strokeStyle = p.n ? '#eaf1f8' : '#e8718c'; c.lineWidth = 1.1;
    c.beginPath(); c.moveTo(tx - 6, ty); c.lineTo(tx + 6, ty); c.moveTo(tx, ty - 6); c.lineTo(tx, ty + 6); c.stroke();
  }
  c.restore();
  const onePx = degToPx(1, g.pxPerCm, g.distanceCm);
  if (Number.isFinite(onePx)) {
    const bar = X(onePx), x0 = 2, y0 = screenH + GUTTER / 2 + 1;
    c.strokeStyle = '#7a8ca0'; c.lineWidth = 1;
    c.beginPath(); c.moveTo(x0, y0); c.lineTo(x0 + bar, y0); c.moveTo(x0, y0 - 4); c.lineTo(x0, y0 + 4); c.moveTo(x0 + bar, y0 - 4); c.lineTo(x0 + bar, y0 + 4); c.stroke();
    c.fillStyle = '#7a8ca0'; c.font = '10px "IBM Plex Mono", monospace'; c.textBaseline = 'middle';
    c.fillText(`1° of visual angle = ${Math.round(onePx)} px`, x0 + bar + 8, y0);
  }
}

/* ── tile hit test ─────────────────────────────────────────────────────── */

async function tileTest() {
  const cols = 4, rows = 3;
  const box = $('chamberGrid');
  box.hidden = false;
  box.style.gridTemplateColumns = `repeat(${cols}, 1fr)`;
  box.style.gridTemplateRows = `repeat(${rows}, 1fr)`;
  box.innerHTML = Array.from({ length: cols * rows }, () => '<div class="tile"></div>').join('');
  const tiles = [...box.children];
  const order = shuffle([...tiles.keys()]);
  const trials = [];
  await openChamber('grid', 'A tile lights up: look at it and hold. Measures how often the prediction lands in the right tile.', 'space · skip tile');
  for (const cell of order) {
    if (S.abort) break;
    S.skip = false;
    tiles.forEach((t) => t.classList.remove('tile--lit'));
    tiles[cell].classList.add('tile--lit');
    await wait(reduceMotion ? 700 : SETTLE_MS);
    if (S.abort) break;
    openWindow(); await wait(DWELL_MS); const got = closeWindow();
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
  line.className = 'tbl tbl--wide'; line.id = 'tileOut';
  line.innerHTML = `<h3 class="block__h">Tile hit test · ${r.grid.cols}×${r.grid.rows}</h3><table><tbody>
    <tr><td>samples in the correct tile</td><td class="${cls}"><b>${pct(r.hitRate)}</b></td><td>tile is ${Math.round(r.tilePx.w)}×${Math.round(r.tilePx.h)} px</td></tr>
    <tr><td>tiles acquired (majority on target)</td><td><b>${r.trialsAcquired}/${r.trials}</b></td><td>${r.hitRate >= .8 ? `a ${r.grid.cols}×${r.grid.rows} layout is selectable` : 'too tight — use larger regions'}</td></tr>
  </tbody></table>`;
  $('tileOut')?.remove();
  $('tblPoints').closest('.tbl').after(line);
}

/* ── live view ─────────────────────────────────────────────────────────── */

async function liveView() {
  await openChamber('live', 'Free look. Yellow crosshair = smoothed prediction, blue dot = raw frame estimate, blue ring = your measured error. Move your head — the mapping should hold.', 'h · clear the heat map');
  await send('reset_filter');
  heatArmed = true; clearHeat();
}

/* ── exports ───────────────────────────────────────────────────────────── */

function download(name, text, type) {
  const url = URL.createObjectURL(new Blob([text], { type }));
  const a = Object.assign(document.createElement('a'), { href: url, download: name });
  a.click(); URL.revokeObjectURL(url);
}
const stampName = () => `gaze3d-run-${new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19)}`;
function exportCsv() { if (S.summary) download(`${stampName()}.csv`, toCsv(S.summary), 'text/csv'); }
function exportJson() {
  if (!S.summary) return;
  const s = S.summary;
  download(`${stampName()}.json`, JSON.stringify({
    recordedAt: new Date().toISOString(), tracker: 'gaze3d', models: S.frame?.models, camera: S.frame?.cam,
    settings: { calibration: $('inCalib').value, validationPoints: Number($('inValid').value), smoothing: $('inSmooth').checked, settleMs: SETTLE_MS, dwellMs: DWELL_MS, motionMs: DYN_MS },
    geometry: s.geom, calibration: S.calibReport,
    accuracy: { deg: s.accuracyDeg, px: s.accuracyPx, worstDeg: s.worstDeg, p95Px: s.p95Px },
    precision: { spreadPx: s.spreadPx, spreadDeg: s.spreadDeg, noiseS2SPx: s.noisePx },
    signal: { hz: s.hz, lossPct: s.lossPct, targetsLost: s.dropped }, bias: s.bias, regions: s.regions, headPose: s.headPose,
    robustness: S.robust,
    tileTest: S.gridResult && { grid: S.gridResult.grid, hitRate: S.gridResult.hitRate, trialsAcquired: S.gridResult.trialsAcquired, trials: S.gridResult.trials },
    points: s.points.map((p) => ({ target: p.target, predicted: p.centroid, errPx: p.errPx, errDeg: p.errDeg, spreadPx: p.spreadPx, noisePx: p.noisePx, samples: p.n, lossPct: p.lossPct, hz: p.hz, region: p.region,
      raw: p.samples.map((sm) => ({ x: sm.x, y: sm.y, t: sm.t, rawX: sm.raw?.x, rawY: sm.raw?.y, head: sm.head })) })),
    runs: S.runs.map((r) => ({ at: r.at.toISOString(), accuracyDeg: r.acc, accuracyPx: r.px, hz: r.hz, head: r.head })),
  }, null, 2), 'application/json');
}

/* ── wiring ────────────────────────────────────────────────────────────── */

function syncCalibButton() {
  const { n, motion } = protocol();
  el('.btn__k', $('btnCalib')).textContent = `${n} points${motion ? ` + ${MOTION_CUES.length * MOTION_PER_CUE[n]} moving` : ''}`;
}
function syncControls() {
  for (const [id, ok] of [['btnCalib', S.running], ['btnRecal', S.running], ['btnRemeasure', S.running && S.calibrated],
    ['btnFree', S.running && S.calibrated], ['btnFreeEarly', S.running], ['btnGrid', S.running && S.calibrated],
    ['btnRobust', S.running && S.calibrated], ['btnSave', S.running && S.calibrated], ['btnLoad', S.running]]) {
    const b = $(id); b.disabled = !ok; b.title = ok ? '' : S.running ? 'Calibrate first' : 'Start the camera first';
  }
  for (const id of ['btnCsv', 'btnJson']) $(id).disabled = !S.summary;
}
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
$('btnRemeasure').addEventListener('click', guard(() => validate('re-measure'), true));
$('btnRobust').addEventListener('click', guard(robustTest, true));
$('btnFree').addEventListener('click', guard(liveView, true));
$('btnFreeEarly').addEventListener('click', guard(liveView));
$('btnRecover').addEventListener('click', guard(async () => {
  pill('warn', 'refitting saved samples…');
  const report = (await send('recover')).report;
  S.calibReport = report;
  S.calibrated = true;
  showCalibSummary(report);
  $('btnRecover').hidden = true;
  phase('calibrate', 'done', `${report.n_points} pts`);
  phase('measure', 'pending', `0/${$('inValid').value}`);
  syncControls();
  pill('live', `recovered · held-out ${fmt(pxToDeg(report.cv_err_px, geometry().pxPerCm, geometry().distanceCm), 1)}°`);
  await validate('after recovery');
}));
$('btnGrid').addEventListener('click', guard(tileTest, true));
$('btnCsv').addEventListener('click', exportCsv);
$('btnJson').addEventListener('click', exportJson);
$('btnSave').addEventListener('click', guard(async () => { const r = await send('save'); pill('live', 'profile saved'); console.log(r.path); }, true));
$('btnLoad').addEventListener('click', guard(async () => {
  const r = await send('load'); S.calibReport = r.report; S.calibrated = true; showCalibSummary(r.report);
  phase('calibrate', 'done', 'profile'); phase('measure', 'pending', `0/${$('inValid').value}`); syncControls(); pill('live', 'profile loaded');
}));
$('btnFreeEarly').textContent = 'Live view';

for (const id of ['inDiag', 'inCamY']) $(id).addEventListener('input', () => { paintGeometry(); if (S.running) pushGeometry().catch(() => {}); });
$('inCalib').addEventListener('change', syncCalibButton);
$('inSmooth').addEventListener('change', () => S.running && send('reset_filter'));

addEventListener('keydown', (e) => {
  if (chamber.hidden) return;
  if (e.key === 'Escape') {
    S.abort = true; S.skip = true; collector.on = false;
    if (S.running) send('disarm').catch(() => {});
    closeChamber(); view(S.summary ? 'report' : 'camera');
  } else if (e.key === ' ') { e.preventDefault(); S.skip = true; }
  else if (e.key === 'h' && S.mode === 'live') clearHeat();
});
document.addEventListener('fullscreenchange', () => {
  if (document.fullscreenElement) return;
  if (S.fsExiting) { S.fsExiting = false; return; }        // we asked for this exit
  if (!chamber.hidden && S.mode !== 'idle') { S.abort = true; S.skip = true; }   // user pressed esc
});
addEventListener('resize', () => { paintGeometry(); if (S.summary && !el('.view--report').hidden) drawPlot(S.summary); });

const guess = guessDiagonal();
if (guess) $('inDiag').value = guess;
if (!guess) $('inCamY').value = -40;
paintGeometry();
syncCalibButton();
syncControls();
frame();
