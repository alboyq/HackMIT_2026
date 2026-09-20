import { useEffect, useRef, useState } from 'react'
import { DWELL_MS, useGazeSelection } from './useGazeSelection'
import { describeCameraError, requestCameraStream, type GazeStatus } from './useFaceLandmarker'
import { DEFAULT_JAW_CONFIG } from './jawDetector'

export interface GazeHudProps {
  onConfirm?: (zoneIndex: number) => void
  onZone?: (zoneIndex: number | null) => void
  lockHeld?: boolean
  fetchLabel?: string | null
  fetchPct?: number
  onAbort?: () => void
}

const STATUS_LABEL: Record<GazeStatus, string> = {
  idle: 'idle',
  'loading-model': 'loading',
  'requesting-camera': 'camera',
  running: 'live',
  error: 'error',
}

const CAM = 236
const RING_R = 112
const RING_C = 2 * Math.PI * RING_R

function playLockChime() {
  try {
    const ctx = new AudioContext()
    const now = ctx.currentTime
    const osc = ctx.createOscillator()
    const gain = ctx.createGain()
    osc.type = 'sine'
    osc.frequency.setValueAtTime(523, now)
    osc.frequency.setValueAtTime(784, now + 0.08)
    gain.gain.setValueAtTime(0.07, now)
    gain.gain.exponentialRampToValueAtTime(0.001, now + 0.32)
    osc.connect(gain)
    gain.connect(ctx.destination)
    osc.start(now)
    osc.stop(now + 0.34)
  } catch {
    // autoplay / closed context — visual lock is enough
  }
}

export default function GazeHud({
  onConfirm,
  onZone,
  lockHeld = false,
  fetchLabel = null,
  fetchPct = 0,
  onAbort,
}: GazeHudProps) {
  const videoRef = useRef<HTMLVideoElement>(null)
  const [stream, setStream] = useState<MediaStream | null>(null)
  const [camError, setCamError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)

  const { status, error, faceDetected, jawOpen, zoneIndex, dwelling, dwellMs, confirmJustFired, reset } =
    useGazeSelection(!!stream, videoRef, stream, lockHeld)

  useEffect(() => {
    return () => {
      stream?.getTracks().forEach((t) => t.stop())
    }
  }, [stream])

  const onConfirmRef = useRef(onConfirm)
  onConfirmRef.current = onConfirm
  const onZoneRef = useRef(onZone)
  onZoneRef.current = onZone
  const zoneIndexRef = useRef(zoneIndex)
  zoneIndexRef.current = zoneIndex

  useEffect(() => {
    onZoneRef.current?.(zoneIndex)
  }, [zoneIndex])

  const [flash, setFlash] = useState(false)
  const prevConfirm = useRef(false)
  useEffect(() => {
    const rising = confirmJustFired && !prevConfirm.current
    prevConfirm.current = confirmJustFired
    const z = zoneIndexRef.current
    if (!rising || lockHeld || z === null) return
    setFlash(true)
    playLockChime()
    onConfirmRef.current?.(z)
    const t = window.setTimeout(() => setFlash(false), 700)
    return () => window.clearTimeout(t)
  }, [confirmJustFired, lockHeld])

  async function allowCamera() {
    setBusy(true)
    setCamError(null)
    try {
      const s = await requestCameraStream()
      setStream(s)
      const v = videoRef.current
      if (v) {
        v.srcObject = s
        v.muted = true
        v.playsInline = true
        await v.play().catch(() => {})
      }
    } catch (err) {
      setCamError(describeCameraError(err))
    } finally {
      setBusy(false)
    }
  }

  function stopCamera() {
    stream?.getTracks().forEach((t) => t.stop())
    setStream(null)
    reset()
    const v = videoRef.current
    if (v) v.srcObject = null
  }

  const combinedError = camError || error
  const jawPct = Math.min(1, jawOpen / DEFAULT_JAW_CONFIG.openThreshold)
  const dwellPct = zoneIndex !== null ? Math.min(1, dwellMs / DWELL_MS) : 0
  const cx = CAM / 2

  return (
    <div className="relative" style={{ width: CAM, height: CAM }}>
      <svg viewBox={`0 0 ${CAM} ${CAM}`} className="absolute inset-0 h-full w-full">
        <circle cx={cx} cy={cx} r={cx - 2} fill="rgba(5,6,12,0.92)" stroke="rgba(139,155,255,0.25)" strokeWidth="1" />
        <circle cx={cx} cy={cx} r={RING_R} fill="none" stroke="rgba(61,255,200,0.2)" strokeWidth="2" />
        <circle
          cx={cx}
          cy={cx}
          r={RING_R}
          fill="none"
          stroke="var(--accent)"
          strokeWidth="4"
          strokeLinecap="round"
          strokeDasharray={RING_C}
          strokeDashoffset={RING_C * (1 - (lockHeld ? fetchPct : jawPct))}
          transform={`rotate(-90 ${cx} ${cx})`}
          style={{ filter: 'drop-shadow(0 0 8px var(--accent-glow))' }}
        />
      </svg>
      <div className="absolute overflow-hidden rounded-full border border-white/15 bg-black" style={{ inset: 18 }}>
        <video
          ref={videoRef}
          muted
          playsInline
          autoPlay
          className="h-full w-full object-cover"
          style={{ transform: 'scaleX(-1)' }}
        />
        {!stream && (
          <div className="absolute inset-0 flex items-center justify-center bg-[var(--bg-1)]">
            <button
              type="button"
              onClick={allowCamera}
              disabled={busy}
              className="rounded-full bg-[var(--accent)] px-5 py-2.5 font-mono text-[11px] font-bold tracking-widest text-white hover:brightness-110 disabled:opacity-50"
            >
              {busy ? 'ASKING…' : 'ALLOW CAMERA'}
            </button>
          </div>
        )}
        {stream && (
          <div className="absolute inset-x-0 top-0 flex items-center justify-center gap-2 bg-black/55 py-1.5 font-mono text-[9px] tracking-widest">
            <span style={{ color: status === 'running' ? 'var(--accent)' : 'var(--text-dim)' }}>
              {STATUS_LABEL[status]}
            </span>
            <span style={{ color: faceDetected ? 'var(--accent)' : 'var(--text-dim)' }}>
              {faceDetected ? 'FACE' : 'NO FACE'}
            </span>
            {lockHeld ? (
              <button type="button" onClick={onAbort} className="text-[var(--danger)] hover:brightness-110">
                {fetchPct >= 1 ? 'NEW PICK' : 'ABORT'}
              </button>
            ) : (
              <button type="button" onClick={stopCamera} className="text-[var(--text-dim)] hover:text-[var(--text-bright)]">
                STOP
              </button>
            )}
          </div>
        )}
        {lockHeld && fetchLabel && (
          <div className="absolute inset-x-0 bottom-0 bg-black/70 py-2 text-center font-mono text-[10px] tracking-widest text-[var(--accent)]">
            {fetchLabel}
          </div>
        )}
        {!lockHeld && stream && (
          <div className="absolute inset-x-6 bottom-2 h-0.5 overflow-hidden rounded-full bg-white/10">
            <div
              className="h-full bg-[var(--accent)]"
              style={{ width: `${dwellPct * 100}%`, opacity: dwelling ? 1 : 0.65 }}
            />
          </div>
        )}
      </div>
      {flash && <div className="pointer-events-none absolute rounded-full bg-[var(--accent)]/30" style={{ inset: 18 }} />}
      {combinedError && (
        <p className="absolute inset-x-4 bottom-8 text-center font-mono text-[10px] leading-snug text-[var(--danger)]">
          {combinedError}
        </p>
      )}
    </div>
  )
}
