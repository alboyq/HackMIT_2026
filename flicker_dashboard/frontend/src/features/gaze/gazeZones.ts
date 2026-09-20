// Map a smoothed HeadPose to one of four screen quadrants (or none).
//
// Product is coarse head-pose aiming, not iris tracking: the wearer turns
// toward a 2×2 tile and opens their jaw to confirm. Zones match TARGET 01–04
// in left-to-right, top-to-bottom order as the stimulus / scene grid:
//
//   0 TL = TARGET 01     1 TR = TARGET 02
//   2 BL = TARGET 03     3 BR = TARGET 04
//
// Sign convention (LIVE-TEST TUNABLE)
// -----------------------------------
// The <video> preview is CSS-mirrored (`scaleX(-1)`) so the wearer sees a
// selfie. MediaPipe FaceLandmarker does NOT see that CSS transform — it
// reads the raw, unmirrored camera buffer. Signs below are chosen so that
// **looking at the LEFT side of the screen highlights LEFT tiles** (and
// looking UP highlights the TOP row), given that unmirrored-buffer
// assumption. Flip YAW_SIGN or PITCH_SIGN after a live webcam test if
// left/right or up/down come out inverted (common if someone later mirrors
// the video element itself before `detectForVideo`).
//
// Geometry we assume from `decomposeHeadPose` on the raw frame
// (camera +X = right edge of the unmirrored image, +Y = up):
//   • User looks LEFT (screen-left tiles) → nose points to +X of the raw
//     image (user's left = camera's right) → raw yawDeg > 0.
//   • User looks UP → nose toward +Y → raw pitchDeg < 0
//     (`atan2(-fy, …)` with Y-up).
// YAW_SIGN = -1 and PITCH_SIGN = -1 then satisfy the formulas below.

import type { HeadPose } from './headPose'

/** Flip to `+1` if looking left highlights the RIGHT tiles. */
export const YAW_SIGN: 1 | -1 = -1

/** Flip to `+1` if looking up highlights the BOTTOM tiles. */
export const PITCH_SIGN: 1 | -1 = -1

/** Half-width of the center dead zone in yaw (degrees). */
export const YAW_DEAD_DEG = 2

/** Half-height of the center dead zone in pitch (degrees). */
export const PITCH_DEAD_DEG = 1.8

/** Once a zone is held, yaw/pitch must pass this far into the other half to switch. */
export const SWITCH_DEG = 1.4

export const ZONE_TL = 0
export const ZONE_TR = 1
export const ZONE_BL = 2
export const ZONE_BR = 3

export type GazeZoneIndex = 0 | 1 | 2 | 3

export const ZONE_LABELS: readonly [string, string, string, string] = [
  'TARGET 01',
  'TARGET 02',
  'TARGET 03',
  'TARGET 04',
]

export interface GazeZoneHit {
  zone: GazeZoneIndex | null
  col: 0 | 1 | null
  row: 0 | 1 | null
}

/**
 * Head pose → 2×2 quadrant. Inside the center dead zone (`|yaw|` and
 * `|pitch|` both small) we return `zone: null` so micro-movements around
 * rest don't jitter the highlight between tiles.
 *
 * Outside the dead zone:
 *   col = yawSign  * yawDeg  < 0 ? 0 : 1   // left / right
 *   row = pitchSign * pitchDeg > 0 ? 0 : 1  // up = top row
 */
export function poseToZone(pose: HeadPose, lastZone: GazeZoneIndex | null = null): GazeZoneIndex | null {
  return classifyPose(pose, lastZone).zone
}

export function classifyPose(pose: HeadPose, lastZone: GazeZoneIndex | null = null): GazeZoneHit {
  if (Math.abs(pose.yawDeg) < YAW_DEAD_DEG && Math.abs(pose.pitchDeg) < PITCH_DEAD_DEG) {
    return { zone: null, col: null, row: null }
  }

  const yaw = YAW_SIGN * pose.yawDeg
  const pitch = PITCH_SIGN * pose.pitchDeg
  let col: 0 | 1 = yaw < 0 ? 0 : 1
  let row: 0 | 1 = pitch > 0 ? 0 : 1

  if (lastZone !== null) {
    const lastCol = (lastZone % 2) as 0 | 1
    const lastRow = Math.floor(lastZone / 2) as 0 | 1
    if (lastCol !== col && Math.abs(yaw) < SWITCH_DEG) col = lastCol
    if (lastRow !== row && Math.abs(pitch) < SWITCH_DEG) row = lastRow
  }

  const zone = (row * 2 + col) as GazeZoneIndex
  return { zone, col, row }
}
