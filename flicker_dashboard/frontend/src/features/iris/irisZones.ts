// Maps the calibrated continuous gaze estimate (nx, ny in roughly -1..1
// normalized screen space) to one of the 4 object quadrants, with a small
// dead zone and hysteresis - same shape as features/gaze/gazeZones.ts's
// poseToZone, just driven by a continuous calibrated point instead of raw
// head yaw/pitch. Zone numbering matches it too, so TARGET 01-04 line up:
//
//   0 TL = TARGET 01     1 TR = TARGET 02
//   2 BL = TARGET 03     3 BR = TARGET 04

export type IrisZoneIndex = 0 | 1 | 2 | 3

/** Half-width/height of the center dead zone, in normalized -1..1 units.
 * Deliberately smaller than gazeZones' angular dead zone - the whole point
 * of iris tracking is to respond to subtler movement than a head turn. */
export const ZONE_DEAD = 0.16

/** Once a zone is held, the point must cross back this far past center to
 * switch away from it, so it doesn't chatter near the boundary. */
export const SWITCH_MARGIN = 0.1

export function positionToZone(nx: number, ny: number, lastZone: IrisZoneIndex | null = null): IrisZoneIndex | null {
  if (Math.abs(nx) < ZONE_DEAD && Math.abs(ny) < ZONE_DEAD) return null

  let col: 0 | 1 = nx < 0 ? 0 : 1
  let row: 0 | 1 = ny < 0 ? 0 : 1

  if (lastZone !== null) {
    const lastCol = (lastZone % 2) as 0 | 1
    const lastRow = Math.floor(lastZone / 2) as 0 | 1
    if (lastCol !== col && Math.abs(nx) < SWITCH_MARGIN) col = lastCol
    if (lastRow !== row && Math.abs(ny) < SWITCH_MARGIN) row = lastRow
  }

  return (row * 2 + col) as IrisZoneIndex
}
