import { useCallback, useEffect, useRef, useState, type RefObject } from 'react'
import { useFaceLandmarker, type GazeStatus } from './useFaceLandmarker'
import type { HeadPose } from './headPose'
import { JawOpenDetector } from './jawDetector'
import { poseToZone, type GazeZoneIndex } from './gazeZones'

/** Same non-null zone must be held this long before `dwelling` flips true. */
export const DWELL_MS = 550

export interface UseGazeSelectionResult {
  status: GazeStatus
  error: string | null
  pose: HeadPose | null
  faceDetected: boolean
  jawOpen: number
  zoneIndex: number | null
  dwelling: boolean
  dwellMs: number
  /** True for a short pulse on the frame a jaw-open confirm is detected. */
  confirmJustFired: boolean
  reset: () => void
}

export function useGazeSelection(
  active: boolean,
  videoRef: RefObject<HTMLVideoElement | null>,
  stream: MediaStream | null,
  lockHeld = false,
): UseGazeSelectionResult {
  const { status, error, pose, faceDetected, jawOpen } = useFaceLandmarker(active, videoRef, stream)

  const jawRef = useRef<JawOpenDetector | null>(null)
  if (jawRef.current === null) jawRef.current = new JawOpenDetector()

  const lastZoneRef = useRef<GazeZoneIndex | null>(null)
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
    if (!active || !pose || !faceDetected || lockHeld) {
      jawRef.current?.disarm()
      clearDwell()
      return
    }

    const zone = poseToZone(pose, lastZoneRef.current)
    lastZoneRef.current = zone
    const now = performance.now()
    let elapsed = 0
    let isDwelling = false

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
      elapsed = now - dwellRef.current.startedAt
      isDwelling = elapsed >= DWELL_MS
      setZoneIndex(zone)
      setDwellMs(elapsed)
      setDwelling(isDwelling)
      // Sample while looking at the tile (including the dwell) so a closed
      // mouth can arm. Only fire after the head has settled.
      const jawFired = jawRef.current!.update(jawOpen, now)
      if (isDwelling && jawFired) {
        setConfirmJustFired(true)
        if (confirmClearRef.current !== null) window.clearTimeout(confirmClearRef.current)
        confirmClearRef.current = window.setTimeout(() => {
          setConfirmJustFired(false)
          confirmClearRef.current = null
        }, 280)
      }
    }
  }, [active, pose, faceDetected, jawOpen, clearDwell, lockHeld])

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
    status,
    error,
    pose,
    faceDetected,
    jawOpen,
    zoneIndex,
    dwelling,
    dwellMs,
    confirmJustFired,
    reset,
  }
}
