import { useCallback, useEffect, useRef, useState, type RefObject } from 'react'
import { useIrisFaceTracking, type IrisFrame } from './useIrisFaceTracking'
import type { UseRemoteIrisGazeResult } from './useRemoteIrisGaze'
import { JawOpenDetector } from '../gaze/jawDetector'
import { positionToZone, type IrisZoneIndex } from './irisZones'

/** Dwell time to lock in selection - stare at target for this long to confirm.
 * No jaw-open gesture needed. */
export const IRIS_DWELL_MS = 6000  // 6 seconds to lock

export interface UseIrisGazeResult {
  faceDetected: boolean
  /** Local MediaPipe frame (landmarks + jaw openness), still run client-side
   * purely for the confirm gesture and the live landmark overlay - see
   * useIrisFaceTracking's doc. The gaze estimate itself no longer comes from
   * this; it comes from `remote` (useRemoteIrisGaze). */
  frame: IrisFrame | null
  /** Calibrated continuous gaze estimate from the remote gaze3d ensemble,
   * normalized -1..1ish screen space. Null until the server has a fitted
   * calibration and a trustworthy live sample (face visible, not a blink). */
  gaze: { nx: number; ny: number } | null
  zoneIndex: number | null
  dwelling: boolean
  dwellMs: number
  /** True for a short pulse on the frame a jaw-open confirm is detected. */
  confirmJustFired: boolean
  reset: () => void
}

/**
 * Combines the remote gaze3d pipeline's calibrated point with the local
 * jaw-open confirm gesture into the same zone+dwell+confirm state machine
 * /gaze uses (features/gaze/useGazeSelection) - just driven by a
 * server-computed continuous point instead of raw head-pose quadrant
 * classification, and now by a *remote* GPU ensemble instead of the previous
 * in-browser linear-regression approximation.
 */
export function useIrisGaze(
  active: boolean,
  videoRef: RefObject<HTMLVideoElement | null>,
  stream: MediaStream | null,
  remote: UseRemoteIrisGazeResult,
  lockHeld = false,
): UseIrisGazeResult {
  // Local landmarker, kept only for jawOpen (confirm gesture) and the live
  // landmark overlay - the gaze coordinate itself comes from `remote` now.
  const { frame, faceDetected: localFaceDetected } = useIrisFaceTracking(active, videoRef, stream)

  const jawRef = useRef<JawOpenDetector | null>(null)
  if (jawRef.current === null) jawRef.current = new JawOpenDetector()

  const lastZoneRef = useRef<IrisZoneIndex | null>(null)
  const [zoneIndex, setZoneIndex] = useState<number | null>(null)
  const [dwelling, setDwelling] = useState(false)
  const [dwellMs, setDwellMs] = useState(0)
  const [confirmJustFired, setConfirmJustFired] = useState(false)

  const dwellRef = useRef<{ zone: number; startedAt: number } | null>(null)
  const confirmClearRef = useRef<number | null>(null)

  const clearDwell = useCallback(() => {
    dwellRef.current = null
    lastZoneRef.current = null
    setZoneIndex(null)
    setDwelling(false)
    setDwellMs(0)
  }, [])

  useEffect(() => {
    if (!active || !remote.calibrated || lockHeld) {
      jawRef.current?.disarm()
      clearDwell()
      return
    }
    if (!remote.faceDetected || remote.blink || !remote.gaze) {
      // No face, a blink, or a momentary tracking gap: freeze the last good
      // zone/dwell instead of snapping the reticle to a garbage prediction,
      // and don't let a blink mid-dwell cost the whole hold.
      jawRef.current?.tick(performance.now())
      return
    }

    const { nx, ny } = remote.gaze
    const zone = positionToZone(nx, ny, lastZoneRef.current)
    lastZoneRef.current = zone
    const now = performance.now()

    if (zone === null) {
      jawRef.current?.disarm()
      jawRef.current?.tick(now)
      dwellRef.current = null
      setZoneIndex(null)
      setDwelling(false)
      setDwellMs(0)
    } else {
      if (dwellRef.current?.zone !== zone) {
        jawRef.current?.disarm()
        dwellRef.current = { zone, startedAt: now }
      }
      const elapsed = now - dwellRef.current.startedAt
      const isDwelling = elapsed >= IRIS_DWELL_MS
      setZoneIndex(zone)
      setDwellMs(elapsed)
      setDwelling(isDwelling)
      // Auto-confirm when dwell completes - no jaw gesture needed
      if (isDwelling && !confirmJustFired) {
        setConfirmJustFired(true)
        if (confirmClearRef.current !== null) window.clearTimeout(confirmClearRef.current)
        confirmClearRef.current = window.setTimeout(() => {
          setConfirmJustFired(false)
          confirmClearRef.current = null
        }, 280)
      }
    }
    // remote.gaze/faceDetected/blink change on every server frame push, which is
    // exactly what should drive this loop - same shape as the previous
    // per-video-frame effect, just fed by the remote pipeline's cadence now.
  }, [active, remote.gaze, remote.faceDetected, remote.blink, remote.calibrated, lockHeld, clearDwell, confirmJustFired])

  useEffect(() => {
    return () => {
      if (confirmClearRef.current !== null) window.clearTimeout(confirmClearRef.current)
    }
  }, [])

  const reset = useCallback(() => {
    jawRef.current?.reset()
    dwellRef.current = null
    lastZoneRef.current = null
    setZoneIndex(null)
    setDwelling(false)
    setDwellMs(0)
    setConfirmJustFired(false)
    if (confirmClearRef.current !== null) {
      window.clearTimeout(confirmClearRef.current)
      confirmClearRef.current = null
    }
  }, [])

  return {
    faceDetected: remote.faceDetected || localFaceDetected,
    frame,
    gaze: remote.gaze,
    zoneIndex,
    dwelling,
    dwellMs,
    confirmJustFired,
    reset,
  }
}
