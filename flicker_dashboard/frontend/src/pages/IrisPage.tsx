import { useEffect, useRef, useState } from 'react'
import { useObjectCrops } from '../features/gaze/useObjectCrops'
import { useObjectNames, displayNameFor, checkVisionStatus } from '../features/vision/useObjectNames'
import SquareObjectTile from '../components/SquareObjectTile'
import SpokenCaption from '../components/SpokenCaption'
import { HOLD_AFTER_MS } from '../lib/fetchMission'
import { announceAbort, announceDelivered, announceLock, speak, silence } from '../lib/speech'
import type { SceneObject } from '../hooks/useHubSocket'
import { armStageLabel, useArmDelivery } from '../hooks/useArmDelivery'
import { useIrisGaze } from '../features/iris/useIrisGaze'
import { useRemoteIrisGaze } from '../features/iris/useRemoteIrisGaze'
import { describeCameraError, requestCameraStream } from '../features/iris/useIrisFaceTracking'
import IrisCalibration from '../features/iris/IrisCalibration'
import IrisConsole from '../features/iris/IrisConsole'

interface Snapshot {
  snapshot_id: string
  image_path: string
  objects: SceneObject[]
}

const EMPTY_OBJECTS: SceneObject[] = []
const POS: Array<'tl' | 'tr' | 'bl' | 'br'> = ['tl', 'tr', 'bl', 'br']

type Phase = 'gate' | 'calibrating' | 'tracking'

export default function IrisPage() {
  const [snapshot, setSnapshot] = useState<Snapshot | null>(null)
  const [confirmed, setConfirmed] = useState<number | null>(null)
  const [aiStatus, setAiStatus] = useState<string | null>(null)
  const [phase, setPhase] = useState<Phase>('gate')
  const [stream, setStream] = useState<MediaStream | null>(null)
  const [camError, setCamError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)

  const videoRef = useRef<HTMLVideoElement>(null)
  const crops = useObjectCrops(snapshot?.image_path ?? null, snapshot?.objects ?? EMPTY_OBJECTS)
  const { names, ready } = useObjectNames(crops)
  const deliveredRef = useRef(false)
  const lockedRef = useRef(false)
  const arm = useArmDelivery()

  const fetching = confirmed !== null
  // Remote gaze3d ensemble on GPU hardware, reached through SSH tunnel
  const remote = useRemoteIrisGaze(!!stream, videoRef, stream)
  const iris = useIrisGaze(!!stream, videoRef, stream, remote, fetching)

  useEffect(() => {
    checkVisionStatus().then((s) => setAiStatus(s.provider || 'none'))
  }, [])

  useEffect(() => {
    silence()
    return () => silence()
  }, [])

  useEffect(() => {
    fetch('/api/snapshot/latest')
      .then((r) => r.json())
      .then((data) => {
        if (data?.objects?.length) setSnapshot(data)
      })
      .catch(() => {})
  }, [])

  useEffect(() => {
    if (confirmed === null) deliveredRef.current = false
  }, [confirmed])

  // Waits for the REAL arm.status broadcasts, not a fixed-duration animation.
  useEffect(() => {
    if (confirmed === null || deliveredRef.current) return
    if (arm.phase === 'done') {
      deliveredRef.current = true
      const crop = crops[confirmed]
      announceDelivered(displayNameFor(crop?.id ?? confirmed + 1, names[confirmed]))
      const t = window.setTimeout(() => {
        lockedRef.current = false
        setConfirmed(null)
        arm.reset()
      }, HOLD_AFTER_MS)
      return () => window.clearTimeout(t)
    }
    if (arm.phase === 'failed') {
      deliveredRef.current = true
      speak(arm.message ? `Arm trouble: ${arm.message}` : 'The arm ran into trouble.', { interrupt: true })
      const t = window.setTimeout(() => {
        lockedRef.current = false
        setConfirmed(null)
        arm.reset()
      }, HOLD_AFTER_MS)
      return () => window.clearTimeout(t)
    }
  }, [arm.phase, arm.message, confirmed, crops, names])

  function commit(z: number) {
    if (lockedRef.current) return
    lockedRef.current = true
    setConfirmed(z)
    const crop = crops[z]
    const label = displayNameFor(crop?.id ?? z + 1, names[z])
    announceLock(label)
    arm.trigger(crop?.id ?? z + 1, label)
  }

  function abort() {
    const wasLocked = lockedRef.current
    lockedRef.current = false
    if (wasLocked) announceAbort()
    setConfirmed(null)
    arm.reset()
  }

  // Fire the object commit on the rising edge of the jaw-open confirm pulse
  // - the same gesture/trigger /gaze uses (via GazeHud's own rising-edge
  // effect), just wired up at the page level here instead of inside the HUD
  // component.
  const prevConfirmRef = useRef(false)
  useEffect(() => {
    const rising = iris.confirmJustFired && !prevConfirmRef.current
    prevConfirmRef.current = iris.confirmJustFired
    if (!rising || fetching || iris.zoneIndex === null) return
    commit(iris.zoneIndex)
    // eslint-disable-next-line react-hooks/exhaustive-deps -- commit/crops/names read via closure intentionally, same pattern as GazeHud
  }, [iris.confirmJustFired, fetching, iris.zoneIndex])

  async function allowCamera() {
    setBusy(true)
    setCamError(null)
    try {
      const s = await requestCameraStream()
      setStream(s)
      // Don't force calibration - wait for remote to report calibration status
      // If server has saved calibration, it will auto-load and we skip to tracking
      setPhase('calibrating') // Start here, but useEffect below will skip to tracking if calibrated
    } catch (err) {
      setCamError(describeCameraError(err))
    } finally {
      setBusy(false)
    }
  }

  // Skip calibration if server already has a loaded calibration
  useEffect(() => {
    if (phase === 'calibrating' && remote.calibrated) {
      setPhase('tracking')
    }
  }, [phase, remote.calibrated])

  function stopCamera() {
    stream?.getTracks().forEach((t) => t.stop())
    setStream(null)
    setCalibrator(null)
    setPhase('gate')
    iris.reset()
    const v = videoRef.current
    if (v) v.srcObject = null
  }

  function onCalibrationDone(_report: unknown) {
    // Calibration state is managed server-side via remote.calibReport;
    // we just move to tracking phase
    setPhase('tracking')
  }

  async function recalibrate() {
    try {
      await remote.clear()
    } catch {
      // non-fatal - just proceed to recalibrate
    }
    iris.reset()
    setPhase('calibrating')
  }

  useEffect(() => {
    return () => {
      stream?.getTracks().forEach((t) => t.stop())
    }
  }, [stream])

  const slots = [0, 1, 2, 3].map((i) => crops[i] ?? null)
  const loading = Boolean(snapshot && crops.length === 0)
  const lockedId = confirmed !== null ? (slots[confirmed]?.id ?? confirmed + 1) : null
  const hudLabel =
    fetching && lockedId !== null
      ? `${armStageLabel(arm.stage)}  ${displayNameFor(lockedId, names[confirmed!]).toUpperCase()}`
      : null

  const reticleLeft = iris.gaze ? `${((Math.max(-1, Math.min(1, iris.gaze.nx)) + 1) / 2) * 100}%` : '50%'
  const reticleTop = iris.gaze ? `${((Math.max(-1, Math.min(1, iris.gaze.ny)) + 1) / 2) * 100}%` : '50%'

  return (
    <div className="gaze-stage iris-arena relative h-full min-h-0 w-full overflow-hidden">
      {POS.map((pos, i) => (
        <div
          key={pos}
          className="gaze-slot"
          style={{
            gridArea: { tl: '1 / 1', tr: '1 / 3', bl: '3 / 1', br: '3 / 3' }[pos],
            placeItems: { tl: 'start start', tr: 'start end', bl: 'end start', br: 'end end' }[pos],
          }}
        >
          <SquareObjectTile
            crop={slots[i]}
            index={i}
            name={names[i]}
            looking={iris.zoneIndex === i}
            locked={confirmed === i}
            fetching={fetching}
            loading={loading}
            phase={confirmed === i ? armStageLabel(arm.stage) : null}
            fetchPct={confirmed === i ? arm.pct : 0}
            showId={false}
            nameSize="sm"
          />
        </div>
      ))}

      <div className="iris-console-slot">
        <IrisConsole
          videoRef={videoRef}
          stream={stream}
          status={iris.status}
          error={camError || iris.error}
          faceDetected={iris.faceDetected}
          frame={iris.frame}
          gaze={iris.gaze}
          dwelling={iris.dwelling}
          dwellMs={iris.dwellMs}
          quality={remote.calibReport ? (remote.calibReport.cv_err_px < 30 ? 'good' : remote.calibReport.cv_err_px < 60 ? 'fair' : 'poor') : null}
          residual={remote.calibReport?.cv_err_px ?? null}
          lockHeld={fetching}
          fetchLabel={hudLabel}
          fetchPct={arm.pct}
          busy={busy}
          onAllowCamera={allowCamera}
          onStop={stopCamera}
          onRecalibrate={recalibrate}
          onAbort={abort}
        />
      </div>

      {phase === 'tracking' && iris.gaze && !fetching && (
        <div className="iris-reticle" style={{ left: reticleLeft, top: reticleTop }}>
          <svg viewBox="0 0 28 28" width="28" height="28">
            <circle cx="14" cy="14" r="10" fill="none" stroke="var(--accent-2)" strokeWidth="1.5" opacity="0.5" />
            <path d="M14 2v6M14 20v6M2 14h6M20 14h6" stroke="var(--accent-2)" strokeWidth="1.5" opacity="0.85" />
            <circle cx="14" cy="14" r="2.5" fill="var(--accent-2)" style={{ filter: 'drop-shadow(0 0 5px var(--accent-2))' }} />
          </svg>
        </div>
      )}

      {phase === 'calibrating' && stream && (
        <IrisCalibration remote={remote} onDone={onCalibrationDone} onCancel={stopCamera} />
      )}

      <div className="pointer-events-none absolute left-1/2 top-3 z-20 -translate-x-1/2 flex gap-3 font-mono text-[10px] tracking-widest">
        {!ready && crops.length > 0 && <span className="text-[var(--text-dim)]">NAMING…</span>}
        {aiStatus === 'none' && ready && (
          <span className="rounded bg-[var(--danger)]/20 px-1.5 py-0.5 text-[var(--danger)]">NO API KEY</span>
        )}
      </div>

      <SpokenCaption />
    </div>
  )
}
