// Per-eye geometry from MediaPipe FaceLandmarker's 478-point face mesh (the
// last 10 points, 468-477, are the iris refinement landmarks that only show
// up when the mesh isn't restricted to the base 468).
//
// Index choices below are cross-checked against MediaPipe's own published
// face mesh landmark map *and* against the eye model in a measured
// (1.3 deg mean error) webcam gaze system's source
// (alboyq/HackMIT_2026, gaze3d/gaze3d/face.py: R_EYE / L_EYE dicts) rather
// than guessed:
//   outer = temporal (outside) eye corner, inner = nasal (toward the nose)
//   corner, iris = iris center, up/down = eyelid landmarks used for the
//   vertical axis and for aperture (blink) detection.
//
// This module deliberately does NOT do what that reference project does
// (PnP + a 3D eyeball sphere + ray-plane intersection) - see calibration.ts
// for why a flat, per-person-fitted 2D regression is the right scope here
// instead. What it keeps from that approach is the *feature*: iris position
// expressed relative to the eye's own corners/lids, independent of head
// position or camera distance, which is what makes it sensitive to eye
// movement alone rather than requiring a head turn.

export interface EyeLandmarkSet {
  outer: number
  inner: number
  iris: number
  up: number
  down: number
}

export const RIGHT_EYE: EyeLandmarkSet = { outer: 33, inner: 133, iris: 468, up: 159, down: 145 }
export const LEFT_EYE: EyeLandmarkSet = { outer: 263, inner: 362, iris: 473, up: 386, down: 374 }

/** Minimal structural shape so callers don't have to import the mediapipe type. */
export interface LandmarkPoint {
  x: number
  y: number
  z?: number
}

export interface EyeOffset {
  /** -1..1ish: 0 = iris centered between the corners, sign is LIVE-TEST TUNABLE
   * (see combineEyes doc) same spirit as gazeZones.ts's YAW_SIGN/PITCH_SIGN. */
  dx: number
  dy: number
  /** Eyelid gap / eye width. Drops sharply on a blink. */
  openness: number
}

export interface CombinedEyeSample extends EyeOffset {
  blinking: boolean
}

/** Below this eyelid-aperture ratio we treat the eye as closed/blinking and
 * the caller should not trust dx/dy or feed it into calibration. */
export const BLINK_OPENNESS_THRESHOLD = 0.14

function px(landmarks: LandmarkPoint[], i: number, w: number, h: number): [number, number] {
  const p = landmarks[i]
  return [p.x * w, p.y * h]
}

/**
 * Iris center position relative to that eye's corners/lids, in pixel space
 * (not raw normalized 0..1 landmark coords) so the horizontal and vertical
 * axes aren't skewed by the video's width/height having different scales.
 */
export function computeEyeOffset(
  landmarks: LandmarkPoint[],
  eye: EyeLandmarkSet,
  videoW: number,
  videoH: number,
): EyeOffset {
  const [ox, oy] = px(landmarks, eye.outer, videoW, videoH)
  const [ix, iy] = px(landmarks, eye.inner, videoW, videoH)
  const [ux, uy] = px(landmarks, eye.up, videoW, videoH)
  const [lx, ly] = px(landmarks, eye.down, videoW, videoH)
  const [gx, gy] = px(landmarks, eye.iris, videoW, videoH)

  // Horizontal: project the iris center onto the outer->inner axis. t=0 at
  // the outer corner, t=1 at the inner corner.
  const hx = ix - ox
  const hy = iy - oy
  const hLenSq = hx * hx + hy * hy || 1e-6
  const t = ((gx - ox) * hx + (gy - oy) * hy) / hLenSq

  // Vertical: project onto the top-lid->bottom-lid axis. s=0 at the top lid,
  // s=1 at the bottom lid.
  const vx = lx - ux
  const vy = ly - uy
  const vLenSq = vx * vx + vy * vy || 1e-6
  const s = ((gx - ux) * vx + (gy - uy) * vy) / vLenSq

  const eyeWidth = Math.hypot(hx, hy) || 1e-6
  const eyeHeight = Math.hypot(vx, vy)

  return {
    dx: (t - 0.5) * 2,
    dy: (s - 0.5) * 2,
    openness: eyeHeight / eyeWidth,
  }
}

/**
 * Averages both eyes into one horizontal/vertical offset. Because outer/inner
 * are labeled per-eye (temporal/nasal), a real conjugate eye movement (both
 * eyes rotating the same real-world direction) produces the same-sign dx on
 * both eyes, so a plain average is meaningful rather than canceling out.
 *
 * Sign convention (LIVE-TEST TUNABLE, same spirit as gazeZones.ts):
 * with the raw, unmirrored camera buffer MediaPipe actually sees, looking
 * toward the camera's +X (the wearer's left, since the preview is CSS
 * mirrored) moves the iris toward each eye's inner/nasal corner, i.e. t -> 1,
 * so dx increases. If left/right come out inverted on a live camera, flip
 * the sign where dx feeds the calibration features, not here.
 */
export function combineEyes(landmarks: LandmarkPoint[], videoW: number, videoH: number): CombinedEyeSample {
  const r = computeEyeOffset(landmarks, RIGHT_EYE, videoW, videoH)
  const l = computeEyeOffset(landmarks, LEFT_EYE, videoW, videoH)
  const openness = Math.min(r.openness, l.openness)
  return {
    dx: (r.dx + l.dx) / 2,
    dy: (r.dy + l.dy) / 2,
    openness,
    blinking: openness < BLINK_OPENNESS_THRESHOLD,
  }
}
