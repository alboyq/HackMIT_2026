// Calibration target layout, ported from gaze3d's own bench page
// (alboyq/HackMIT_2026 js/gaze3d.js) rather than invented fresh, since the
// entire point of switching to the remote gaze3d pipeline is its measured
// accuracy - which was measured using *this* protocol (9 static points, plus
// head-motion trials across 5 movement types), not a smaller one.
//
// Units: fractions of the viewport (0..1), matching how this app's
// useRemoteIrisGaze treats the browser viewport as the calibration screen
// (see that file's geometry-push comment). gaze3d.js works in physical
// screen px and separately corrects for window offset; this app assumes a
// fullscreen-ish kiosk window, so viewport px double as "screen" px directly.

export interface CalibTarget {
  x: number
  y: number
  label: string
}

const MARGIN_PX = 44

function clamp(v: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, v))
}

/** 5x5 grid for robust calibration - 25 points covering full screen area.
 * Extra density at edges/corners where tracking tends to underestimate. */
const GRID_FRACS = [0.06, 0.28, 0.5, 0.72, 0.94]

/** Corner positions for extra passes */
const CORNER_FRACS = [
  { x: 0.06, y: 0.06 },  // TL
  { x: 0.94, y: 0.06 },  // TR
  { x: 0.06, y: 0.94 },  // BL
  { x: 0.94, y: 0.94 },  // BR
]

export function staticTargets(viewportW: number, viewportH: number): CalibTarget[] {
  const pts: CalibTarget[] = []
  let i = 0
  
  // Main 5x5 grid
  for (const fy of GRID_FRACS) {
    for (const fx of GRID_FRACS) {
      i++
      pts.push({
        x: Math.round(clamp(fx * viewportW, MARGIN_PX, viewportW - MARGIN_PX)),
        y: Math.round(clamp(fy * viewportH, MARGIN_PX, viewportH - MARGIN_PX)),
        label: `POINT ${i}`,
      })
    }
  }
  
  // Extra corner passes (4 more) - corners need more data
  for (const c of CORNER_FRACS) {
    i++
    pts.push({
      x: Math.round(clamp(c.x * viewportW, MARGIN_PX, viewportW - MARGIN_PX)),
      y: Math.round(clamp(c.y * viewportH, MARGIN_PX, viewportH - MARGIN_PX)),
      label: `CORNER ${i - 25}`,
    })
  }
  
  return pts  // 29 points total
}

export interface MotionCue {
  kind: string
  text: string
}

/** js/gaze3d.js MOTION_CUES - the five head-movement types the calibration's
 * pose-diversity trials cover (HANDOFF.md section 2: pose diversity during
 * calibration is what makes the fit survive head movement afterward). */
export const MOTION_CUES: MotionCue[] = [
  { kind: 'yaw', text: 'turn your head slowly left … and right' },
  { kind: 'pitch', text: 'tilt your head up … and down' },
  { kind: 'depth', text: 'lean in closer … then sit back' },
  { kind: 'lateral', text: 'slide sideways in your seat, both ways' },
  { kind: 'roll', text: 'roll your head gently, ear toward shoulder' },
]

/** Port of js/metrics.js::spreadPoints: `n` random, mutually well-separated
 * locations inside a margin, avoiding `existing` points too. */
export function spreadPoints(
  n: number,
  w: number,
  h: number,
  margin = 0.12,
  existing: CalibTarget[] = [],
  candidates = 40,
): CalibTarget[] {
  const out: CalibTarget[] = []
  const all = existing.map((p) => ({ x: p.x, y: p.y }))
  const x0 = w * margin,
    y0 = h * margin,
    x1 = w * (1 - margin),
    y1 = h * (1 - margin)
  for (let i = 0; i < n; i++) {
    let best: { x: number; y: number } | null = null
    let bestD = -1
    for (let c = 0; c < candidates; c++) {
      const p = { x: x0 + Math.random() * (x1 - x0), y: y0 + Math.random() * (y1 - y0) }
      const d = all.length ? Math.min(...all.map((q) => Math.hypot(p.x - q.x, p.y - q.y))) : Infinity
      if (d > bestD) {
        bestD = d
        best = p
      }
    }
    const chosen = { x: Math.round(best!.x), y: Math.round(best!.y) }
    out.push(chosen)
    all.push(chosen)
  }
  return out
}

export function shuffle<T>(arr: readonly T[]): T[] {
  return arr
    .map((v) => [Math.random(), v] as const)
    .sort((a, b) => a[0] - b[0])
    .map(([, v]) => v)
}

// --- calibration quality bucketing --------------------------------------
// The old in-browser calibrator (removed - see the project handoff notes)
// reported a normalized -1..1 RMS residual with fixed thresholds. The remote
// gaze3d server instead reports leave-one-target-out cross-validated error in
// screen px (Calibrator.fit's cv_err_px - gaze3d/gaze3d/calibration.py), which
// is a more meaningful number (it's the actual held-out on-screen error) but
// isn't bounded the same way, so these thresholds are a fresh, approximate
// bucketing in px rather than a straight port of the old ones. gaze3d's own
// measured session sat at ~75px/1.3deg (MEASUREMENTS.md) - "good" here is
// deliberately generous relative to that since this app's targets are much
// bigger than a 13-point research validation grid.
export type CalibrationQuality = 'good' | 'fair' | 'poor'

export function calibQuality(cvErrPx: number | null | undefined): CalibrationQuality | null {
  if (cvErrPx == null || !Number.isFinite(cvErrPx)) return null
  if (cvErrPx < 90) return 'good'
  if (cvErrPx < 180) return 'fair'
  return 'poor'
}
