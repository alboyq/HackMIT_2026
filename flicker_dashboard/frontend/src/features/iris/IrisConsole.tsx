import { useEffect, useRef, type RefObject } from 'react'
import type { IrisFrame, IrisTrackerStatus } from './useIrisFaceTracking'
import type { CalibrationQuality } from './calibration'
import { LEFT_EYE, RIGHT_EYE } from './irisLandmarks'
import { IRIS_DWELL_MS } from './useIrisGaze'

// The instrument panel for /iris: camera + a live landmark overlay (drawn
// from the raw 478-point mesh, not just a plain video feed like /gaze's
// circular HUD) plus numeric readouts. Loosely modeled on gaze3d's own
// "Live signal" readout block (gaze3d.html: a <dl> of gaze x/y, head
// yaw/pitch, distance, ensemble spread, state) - scaled down to what this
// app's simpler pipeline actually measures.

export interface IrisConsoleProps {
  videoRef: RefObject<HTMLVideoElement | null>
  stream: MediaStream | null
  status: IrisTrackerStatus
  error: string | null
  faceDetected: boolean
  frame: IrisFrame | null
  gaze: { nx: number; ny: number } | null
  dwelling: boolean
  dwellMs: number
  quality: CalibrationQuality | null
  residual: number | null
  lockHeld: boolean
  fetchLabel: string | null
  fetchPct: number
  busy: boolean
  onAllowCamera: () => void
  onStop: () => void
  onRecalibrate: () => void
  onAbort: () => void
}

const STATUS_LABEL: Record<IrisTrackerStatus, string> = {
  idle: 'idle',
  'loading-model': 'loading',
  'requesting-camera': 'camera',
  running: 'live',
  error: 'error',
}

const QUALITY_COLOR: Record<CalibrationQuality, string> = {
  good: 'var(--good)',
  fair: 'var(--accent-2)',
  poor: 'var(--danger)',
}

const OVERLAY_EYE_POINTS = [RIGHT_EYE, LEFT_EYE]

function fmt(n: number, digits = 2): string {
  return (n >= 0 ? '+' : '') + n.toFixed(digits)
}

function LandmarkOverlay({ frame, videoRef }: { frame: IrisFrame | null; videoRef: RefObject<HTMLVideoElement | null> }) {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    const video = videoRef.current
    if (!canvas || !video || !video.videoWidth) return
    if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
      canvas.width = video.videoWidth
      canvas.height = video.videoHeight
    }
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    ctx.clearRect(0, 0, canvas.width, canvas.height)
    if (!frame) return

    const w = canvas.width
    const h = canvas.height
    for (const eye of OVERLAY_EYE_POINTS) {
      const outer = frame.landmarks[eye.outer]
      const inner = frame.landmarks[eye.inner]
      const up = frame.landmarks[eye.up]
      const down = frame.landmarks[eye.down]
      const iris = frame.landmarks[eye.iris]

      ctx.strokeStyle = 'rgba(139, 155, 255, 0.55)'
      ctx.lineWidth = Math.max(1, w * 0.0035)
      ctx.beginPath()
      ctx.moveTo(outer.x * w, outer.y * h)
      ctx.lineTo(inner.x * w, inner.y * h)
      ctx.stroke()
      ctx.beginPath()
      ctx.moveTo(up.x * w, up.y * h)
      ctx.lineTo(down.x * w, down.y * h)
      ctx.stroke()

      ctx.fillStyle = frame.eye.blinking ? '#fb7185' : '#3dffc8'
      ctx.beginPath()
      ctx.arc(iris.x * w, iris.y * h, Math.max(2, w * 0.01), 0, Math.PI * 2)
      ctx.fill()

      ctx.fillStyle = 'rgba(197, 203, 224, 0.85)'
      for (const p of [outer, inner, up, down]) {
        ctx.beginPath()
        ctx.arc(p.x * w, p.y * h, Math.max(1.2, w * 0.004), 0, Math.PI * 2)
        ctx.fill()
      }
    }
  }, [frame, videoRef])

  return <canvas ref={canvasRef} className="pointer-events-none absolute inset-0 h-full w-full" style={{ transform: 'scaleX(-1)' }} />
}

export default function IrisConsole({
  videoRef,
  stream,
  status,
  error,
  faceDetected,
  frame,
  gaze,
  dwelling,
  dwellMs,
  quality,
  residual,
  lockHeld,
  fetchLabel,
  fetchPct,
  busy,
  onAllowCamera,
  onStop,
  onRecalibrate,
  onAbort,
}: IrisConsoleProps) {
  const dwellPct = Math.min(1, dwellMs / IRIS_DWELL_MS)

  return (
    <div className="iris-console flex flex-col">
      <div className="flex items-center gap-1.5 px-3 pt-3 font-mono text-[9px] tracking-widest">
        <span style={{ color: status === 'running' ? 'var(--accent)' : 'var(--text-dim)' }}>{STATUS_LABEL[status]}</span>
        <span className="text-[var(--text-dim)]">·</span>
        <span style={{ color: faceDetected ? 'var(--accent)' : 'var(--text-dim)' }}>{faceDetected ? 'FACE' : 'NO FACE'}</span>
        {frame?.eye.blinking && (
          <>
            <span className="text-[var(--text-dim)]">·</span>
            <span style={{ color: 'var(--danger)' }}>BLINK</span>
          </>
        )}
        {quality && (
          <span
            className="ml-auto rounded px-1.5 py-0.5"
            style={{ background: `${QUALITY_COLOR[quality]}22`, color: QUALITY_COLOR[quality] }}
          >
            CAL {quality.toUpperCase()}
          </span>
        )}
      </div>

      <div className="relative mx-3 mt-2 aspect-[4/3] overflow-hidden rounded-xl border border-white/10 bg-black">
        <video ref={videoRef} muted playsInline autoPlay className="h-full w-full object-cover" style={{ transform: 'scaleX(-1)' }} />
        <LandmarkOverlay frame={frame} videoRef={videoRef} />

        {!stream && (
          <div className="absolute inset-0 flex items-center justify-center bg-[var(--bg-1)]">
            <button
              type="button"
              onClick={onAllowCamera}
              disabled={busy}
              className="rounded-full bg-[var(--accent)] px-4 py-2 font-mono text-[10px] font-bold tracking-widest text-black hover:brightness-110 disabled:opacity-50"
            >
              {busy ? 'ASKING…' : 'ALLOW CAMERA'}
            </button>
          </div>
        )}

        {lockHeld && fetchLabel && (
          <div className="absolute inset-x-0 bottom-0 bg-black/70 py-1.5 text-center font-mono text-[9px] tracking-widest text-[var(--accent)]">
            {fetchLabel}
          </div>
        )}
      </div>

      {!lockHeld && stream && (
        <div className="mx-3 mt-2 h-0.5 overflow-hidden rounded-full bg-white/10">
          <div className="h-full bg-[var(--accent)]" style={{ width: `${dwellPct * 100}%`, opacity: dwelling ? 1 : 0.6 }} />
        </div>
      )}

      <dl className="iris-readout mx-3 mt-3">
        <dt>gaze x·y</dt>
        <dd>{gaze ? `${fmt(gaze.nx)} , ${fmt(gaze.ny)}` : '—'}</dd>
        <dt>iris Δx·Δy</dt>
        <dd>{frame ? `${fmt(frame.eye.dx)} , ${fmt(frame.eye.dy)}` : '—'}</dd>
        <dt>head yaw·pitch</dt>
        <dd>{frame ? `${frame.pose.yawDeg.toFixed(1)}° , ${frame.pose.pitchDeg.toFixed(1)}°` : '—'}</dd>
        <dt>fit residual</dt>
        <dd>{residual !== null ? residual.toFixed(3) : '—'}</dd>
      </dl>

      {error && <p className="mx-3 mt-2 font-mono text-[9px] leading-snug text-[var(--danger)]">{error}</p>}

      <div className="mt-auto flex gap-1.5 px-3 pb-3 pt-3">
        {stream && !lockHeld && (
          <button
            type="button"
            onClick={onRecalibrate}
            className="flex-1 rounded-full border border-[var(--border-bright)] py-1.5 font-mono text-[9px] tracking-widest text-[var(--text)] hover:text-[var(--text-bright)]"
          >
            RECALIBRATE
          </button>
        )}
        {stream && !lockHeld && (
          <button
            type="button"
            onClick={onStop}
            className="flex-1 rounded-full border border-[var(--border-bright)] py-1.5 font-mono text-[9px] tracking-widest text-[var(--text-dim)] hover:text-[var(--text-bright)]"
          >
            STOP
          </button>
        )}
        {lockHeld && (
          <button
            type="button"
            onClick={onAbort}
            className="flex-1 rounded-full border border-[var(--danger)] py-1.5 font-mono text-[9px] tracking-widest text-[var(--danger)]"
          >
            {fetchPct >= 1 ? 'NEW PICK' : 'ABORT'}
          </button>
        )}
      </div>
    </div>
  )
}
