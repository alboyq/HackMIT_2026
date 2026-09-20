import { useEffect, useRef, useState } from 'react'
import type { UseRemoteIrisGazeResult } from './useRemoteIrisGaze'
import type { CalibReport } from './remoteGaze'
import { staticTargets, spreadPoints, shuffle, MOTION_CUES } from './calibPoints'

// Calibration chamber, now driving gaze3d's own server-side protocol
// (arm -> hold -> disarm per target, then a single fit()) instead of
// collecting samples locally and fitting a small linear model in the browser
// - see the project handoff notes: that local approximation calibrated
// poorly, and the entire point of the switch to the remote ensemble is its
// measured accuracy, obtained with exactly this 9-static + 5-motion-cue
// protocol (ported from alboyq/HackMIT_2026 js/gaze3d.js's `calibrate()`).
// The visual chamber (ring target, HUD strings) is kept close to the
// previous version's look; what changed is that samples now live
// server-side and this component's job is just to tell the server where to
// look and for how long.

const SETTLE_MS = 600
// Longer dwell time so user can do multiple head rotations/positions per point
const DWELL_MS = 10000  // 10 seconds per static point
// Long motion trials for thorough head movement coverage
const MOTION_MS = 10000  // 10 seconds per motion trial
const GAP_MS = 100
const CONNECT_TIMEOUT_MS = 15000

const wait = (ms: number) => new Promise<void>((r) => setTimeout(r, ms))

export interface IrisCalibrationProps {
  remote: UseRemoteIrisGazeResult
  onDone: (report: CalibReport) => void
  onCancel: () => void
}

interface Step {
  kind: 'static' | 'motion'
  x: number
  y: number
  label: string
  cueText?: string
}

type Phase = 'waiting' | 'settle' | 'hold' | 'fitting' | 'error'

export default function IrisCalibration({ remote, onDone, onCancel }: IrisCalibrationProps) {
  const [stepIndex, setStepIndex] = useState(0)
  const [totalSteps, setTotalSteps] = useState(0)
  const [phase, setPhase] = useState<Phase>('waiting')
  const [progress, setProgress] = useState(0)
  const [target, setTarget] = useState<Step | null>(null)
  const [errorMsg, setErrorMsg] = useState<string | null>(null)

  const cancelledRef = useRef(false)
  const remoteRef = useRef(remote)
  remoteRef.current = remote

  useEffect(() => {
    cancelledRef.current = false
    let raf = 0

    function ring(ms: number, onTick: (p: number) => void): Promise<void> {
      const t0 = performance.now()
      return new Promise((resolve) => {
        const step = () => {
          if (cancelledRef.current) return resolve()
          const k = Math.min(1, (performance.now() - t0) / ms)
          onTick(k)
          if (k < 1) raf = requestAnimationFrame(step)
          else resolve()
        }
        raf = requestAnimationFrame(step)
      })
    }

    async function run() {
      const t0 = performance.now()
      while (
        !cancelledRef.current &&
        remoteRef.current.status !== 'tracking' &&
        performance.now() - t0 < CONNECT_TIMEOUT_MS
      ) {
        await wait(150)
      }
      if (cancelledRef.current) return
      if (remoteRef.current.status !== 'tracking') {
        setPhase('error')
        setErrorMsg(remoteRef.current.error || 'Timed out waiting for the remote gaze server to start tracking.')
        return
      }

      try {
        await remoteRef.current.clear()
      } catch {
        // non-fatal - clear() failing just means we calibrate on top of
        // whatever samples the server already had, which fit() still handles
      }

      const statics = shuffle(staticTargets(innerWidth, innerHeight))
      const motionSpots = spreadPoints(MOTION_CUES.length, innerWidth, innerHeight, 0.12, statics)
      const steps: Step[] = [
        ...statics.map((p, i) => ({ kind: 'static' as const, x: p.x, y: p.y, label: `POINT ${i + 1}` })),
        ...MOTION_CUES.map((cue, i) => ({
          kind: 'motion' as const,
          x: motionSpots[i].x,
          y: motionSpots[i].y,
          label: cue.kind.toUpperCase(),
          cueText: cue.text,
        })),
      ]
      setTotalSteps(steps.length)

      for (let i = 0; i < steps.length; i++) {
        if (cancelledRef.current) return
        const step = steps[i]
        setStepIndex(i)
        setTarget(step)
        setPhase('settle')
        setProgress(0)
        await ring(SETTLE_MS, () => {})
        if (cancelledRef.current) return

        setPhase('hold')
        const holdMs = step.kind === 'static' ? DWELL_MS : MOTION_MS
        try {
          await remoteRef.current.arm(
            i,
            [step.x, step.y],
            step.kind === 'static' ? 'static' : `dynamic:${step.label.toLowerCase()}`,
          )
        } catch (err) {
          setPhase('error')
          setErrorMsg(err instanceof Error ? err.message : 'Could not arm the calibration point.')
          return
        }
        await ring(holdMs, (p) => setProgress(p))
        if (cancelledRef.current) return
        try {
          await remoteRef.current.disarm()
        } catch {
          // best-effort - fit() below just sees fewer samples for this point
        }
        await wait(GAP_MS)
      }
      if (cancelledRef.current) return

      setPhase('fitting')
      setTarget(null)
      try {
        const report = await remoteRef.current.fit()
        if (cancelledRef.current) return
        onDone(report)
      } catch (err) {
        setPhase('error')
        setErrorMsg(err instanceof Error ? err.message : 'Calibration fit failed.')
      }
    }

    run()
    return () => {
      cancelledRef.current = true
      if (raf) cancelAnimationFrame(raf)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps -- runs its own one-shot protocol; onDone/onCancel read via closure intentionally, same pattern as the previous version
  }, [])

  const ringR = 44
  const ringC = 2 * Math.PI * ringR
  const leftPct = target ? (target.x / innerWidth) * 100 : 50
  const topPct = target ? (target.y / innerHeight) * 100 : 50
  const cueBelow = topPct < 70

  return (
    <div className="iris-chamber hud-grid absolute inset-0 z-30 flex flex-col">
      <div className="flex items-center justify-between px-6 pt-5">
        <div>
          <p className="font-mono text-[10px] tracking-[0.28em] text-[var(--accent)]">PHASE 02 · CALIBRATE</p>
          <h2 className="font-display mt-1 text-lg tracking-[0.06em] text-[var(--text-bright)]">
            {phase === 'fitting'
              ? 'FITTING ON THE GPU HOST…'
              : phase === 'error'
                ? 'CALIBRATION ERROR'
                : target?.kind === 'motion'
                  ? 'KEEP YOUR EYES ON THE TARGET'
                  : 'EYES ON TARGET · MOVE YOUR HEAD'}
          </h2>
        </div>
        <button
          type="button"
          onClick={onCancel}
          className="rounded-full border border-[var(--border-bright)] px-4 py-1.5 font-mono text-[10px] tracking-widest text-[var(--text-dim)] hover:text-[var(--text-bright)]"
        >
          CANCEL
        </button>
      </div>

      <div className="relative min-h-0 flex-1">
        {target && phase !== 'fitting' && (
          <div
            className="iris-target absolute -translate-x-1/2 -translate-y-1/2"
            style={{ left: `${leftPct}%`, top: `${topPct}%` }}
          >
            <svg viewBox="0 0 120 120" width={96} height={96}>
              <circle cx="60" cy="60" r={ringR} fill="none" stroke="rgba(61,255,200,0.18)" strokeWidth="2" />
              <circle
                cx="60"
                cy="60"
                r={ringR}
                fill="none"
                stroke="var(--accent)"
                strokeWidth="4"
                strokeLinecap="round"
                strokeDasharray={ringC}
                strokeDashoffset={ringC * (1 - (phase === 'hold' ? progress : 0))}
                transform="rotate(-90 60 60)"
                style={{ filter: 'drop-shadow(0 0 8px var(--accent-glow))', transition: 'stroke-dashoffset 60ms linear' }}
              />
              <path d="M60 30v20M60 70v20M30 60h20M70 60h20" stroke="var(--accent-2)" strokeWidth="2" opacity="0.7" />
              <circle cx="60" cy="60" r="5" fill={phase === 'hold' ? 'var(--accent)' : 'var(--accent-2)'} />
            </svg>
          </div>
        )}

        {target?.kind === 'motion' && phase === 'hold' && (
          <p
            className="pointer-events-none absolute -translate-x-1/2 text-center font-mono text-[11px] tracking-widest text-[var(--accent-2)]"
            style={{ left: `${leftPct}%`, top: `${topPct}%`, marginTop: cueBelow ? '64px' : '-64px' }}
          >
            {target.cueText}
          </p>
        )}

        {phase === 'waiting' && (
          <p className="pointer-events-none absolute inset-x-0 top-1/2 -translate-y-1/2 text-center font-mono text-xs tracking-widest text-[var(--text-dim)]">
            CONNECTING TO REMOTE GAZE SERVER…
          </p>
        )}
        {phase === 'hold' && !remote.faceDetected && (
          <p className="pointer-events-none absolute inset-x-0 top-1/2 -translate-y-1/2 text-center font-mono text-xs tracking-widest text-[var(--danger)]">
            NO FACE DETECTED — CENTER YOUR FACE IN FRAME
          </p>
        )}
        {phase === 'error' && (
          <p className="pointer-events-none absolute inset-x-0 top-1/2 -translate-y-1/2 px-10 text-center font-mono text-xs leading-relaxed tracking-widest text-[var(--danger)]">
            {errorMsg}
          </p>
        )}
      </div>

      <div className="px-6 pb-6 text-center">
        <p className="font-mono text-[10px] tracking-widest text-[var(--text-dim)]">
          {phase === 'fitting'
            ? 'Cross-validating candidate models on the GPU host…'
            : totalSteps > 0
              ? `POINT ${stepIndex + 1} / ${totalSteps} · ${target?.label ?? ''}${phase === 'settle' ? ' · SETTLING…' : ''}`
              : ''}
        </p>
        {phase === 'hold' && target?.kind === 'static' && (
          <p className="mt-1 font-mono text-[9px] tracking-widest text-[var(--accent-2)]">
            Keep eyes on target · rotate head left/right, up/down, lean in/out
          </p>
        )}
      </div>
    </div>
  )
}
