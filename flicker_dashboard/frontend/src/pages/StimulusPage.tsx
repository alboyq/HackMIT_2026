import { useEffect, useRef, useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import SnapshotFlickerGrid from '../components/SnapshotFlickerGrid'
import type { SceneObject } from '../hooks/useHubSocket'
import { useObjectCrops } from '../features/gaze/useObjectCrops'
import { useObjectNames } from '../features/vision/useObjectNames'

// S6 (M0): one fixed test-pattern box at a chosen frequency, plus a live
// readout of measured refresh rate and dropped frames — use this to verify
// timing before trusting anything else. S1/S2/S8 (M1): the real stimulus,
// each segmented object flickering independently via its own mask.

const FREQ_OPTIONS = [8.57, 10, 12, 15]
const DROP_DEVIATION_FRAC = 0.25 // log/count any frame interval off nominal by more than this (S6)
const READOUT_INTERVAL_MS = 250

interface Snapshot {
  snapshot_id: string
  image_path: string
  objects: SceneObject[]
}

const EMPTY_OBJECTS: SceneObject[] = []

export default function StimulusPage() {
  const [mode, setMode] = useState<'test' | 'snapshot'>('snapshot')
  const [snapshot, setSnapshot] = useState<Snapshot | null>(null)
  const [frequencyHz, setFrequencyHz] = useState(10)
  const [running, setRunning] = useState(false)
  const [warningAcked, setWarningAcked] = useState(false)
  const [readout, setReadout] = useState({
    measuredHz: 0,
    droppedFrames: 0,
    totalFrames: 0,
    elapsedS: 0,
  })

  // Real object names via cloud vision, same as the dashboard (falls back to
  // "TARGET NN" per-object if the vision API is unconfigured/unreachable).
  const crops = useObjectCrops(snapshot?.image_path ?? null, snapshot?.objects ?? EMPTY_OBJECTS)
  const { names: visionNames } = useObjectNames(crops)
  const namesById: Record<number, string> = {}
  crops.forEach((c, i) => {
    if (visionNames[i]) namesById[c.id] = visionNames[i]!.toUpperCase()
  })

  const canvasRef = useRef<HTMLCanvasElement>(null)
  const rafRef = useRef<number | null>(null)
  const runningRef = useRef(false)
  const frequencyRef = useRef(frequencyHz)
  const startTimeRef = useRef<number | null>(null)
  const lastFrameTimeRef = useRef<number | null>(null)
  const intervalsRef = useRef<number[]>([])
  const droppedRef = useRef(0)
  const totalRef = useRef(0)
  const lastReadoutPushRef = useRef(0)

  useEffect(() => {
    frequencyRef.current = frequencyHz
  }, [frequencyHz])

  useEffect(() => {
    runningRef.current = running
  }, [running])

  // Esc is a hard kill switch (S7) — always listening, not gated on focus.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') stop()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])

  useEffect(() => {
    return () => {
      if (rafRef.current !== null) cancelAnimationFrame(rafRef.current)
    }
  }, [])

  useEffect(() => {
    fetch('/api/snapshot/latest')
      .then((r) => r.json())
      .then((data) => {
        if (data?.objects?.length) setSnapshot(data)
      })
      .catch(() => {})
  }, [])

  function resetCounters() {
    startTimeRef.current = null
    lastFrameTimeRef.current = null
    intervalsRef.current = []
    droppedRef.current = 0
    totalRef.current = 0
    lastReadoutPushRef.current = 0
    setReadout({ measuredHz: 0, droppedFrames: 0, totalFrames: 0, elapsedS: 0 })
  }

  function start() {
    if (!warningAcked) return
    setRunning(true)
    runningRef.current = true
    if (mode === 'test') {
      resetCounters()
      rafRef.current = requestAnimationFrame(tick)
    }
  }

  function stop() {
    setRunning(false)
    runningRef.current = false
    if (rafRef.current !== null) {
      cancelAnimationFrame(rafRef.current)
      rafRef.current = null
    }
    const canvas = canvasRef.current
    const ctx = canvas?.getContext('2d')
    if (canvas && ctx && mode === 'test') {
      ctx.fillStyle = '#000'
      ctx.fillRect(0, 0, canvas.width, canvas.height)
    }
  }

  function tick(timestamp: DOMHighResTimeStamp) {
    if (!runningRef.current) return

    if (startTimeRef.current === null) startTimeRef.current = timestamp
    if (lastFrameTimeRef.current !== null) {
      const dt = timestamp - lastFrameTimeRef.current
      intervalsRef.current.push(dt)
      if (intervalsRef.current.length > 120) intervalsRef.current.shift()

      // Nominal interval from the rolling median, so this self-calibrates to
      // whatever the real refresh rate is rather than assuming 60 Hz.
      const sorted = [...intervalsRef.current].sort((a, b) => a - b)
      const nominal = sorted[Math.floor(sorted.length / 2)] || dt
      if (Math.abs(dt - nominal) / nominal > DROP_DEVIATION_FRAC) {
        droppedRef.current += 1
      }
    }
    lastFrameTimeRef.current = timestamp
    totalRef.current += 1

    // Timing model (PRD section "Timing model"): luminance from presentation
    // time, not a frame counter, so a dropped frame doesn't slip the phase.
    const t = (timestamp - startTimeRef.current) / 1000
    const L = 0.5 * (1 + Math.sin(2 * Math.PI * frequencyRef.current * t))

    const canvas = canvasRef.current
    const ctx = canvas?.getContext('2d')
    if (canvas && ctx) {
      ctx.fillStyle = '#000'
      ctx.fillRect(0, 0, canvas.width, canvas.height)
      const gray = Math.round(L * 255)
      ctx.fillStyle = `rgb(${gray},${gray},${gray})`
      const size = Math.min(canvas.width, canvas.height) * 0.4
      ctx.fillRect((canvas.width - size) / 2, (canvas.height - size) / 2, size, size)
    }

    if (timestamp - lastReadoutPushRef.current > READOUT_INTERVAL_MS) {
      lastReadoutPushRef.current = timestamp
      const intervals = intervalsRef.current
      const meanInterval = intervals.length
        ? intervals.reduce((a, b) => a + b, 0) / intervals.length
        : 0
      setReadout({
        measuredHz: meanInterval > 0 ? 1000 / meanInterval : 0,
        droppedFrames: droppedRef.current,
        totalFrames: totalRef.current,
        elapsedS: (timestamp - startTimeRef.current) / 1000,
      })
    }

    rafRef.current = requestAnimationFrame(tick)
  }

  const dropRate = readout.totalFrames > 0 ? readout.droppedFrames / readout.totalFrames : 0

  return (
    <div className="flex h-full min-h-0 w-full flex-col gap-4 bg-[var(--bg-0)] p-5">
      {!warningAcked && (
        <div className="fixed inset-0 z-10 flex items-center justify-center bg-black/95 backdrop-blur-sm">
          <div className="corner-brackets panel relative max-w-md p-7" style={{ borderColor: 'var(--danger)' }}>
            <div className="mb-3 flex items-center gap-2">
              <span className="h-2 w-2 animate-pulse-glow rounded-full bg-[var(--danger)]" />
              <h2 className="font-mono text-xs font-bold tracking-widest text-[var(--danger)]">
                PHOTOSENSITIVITY WARNING
              </h2>
            </div>
            <p className="mb-5 text-sm leading-relaxed text-[var(--text)]">
              This screen flickers at {FREQ_OPTIONS.join(', ')} Hz. Flicker in this range can
              trigger seizures in people with photosensitive epilepsy. Press{' '}
              <kbd className="rounded-[2px] border border-[var(--border-bright)] bg-[var(--bg-2)] px-1.5 py-0.5 font-mono text-xs">
                Esc
              </kbd>{' '}
              at any time to stop immediately.
            </p>
            <button
              className="w-full rounded-md bg-[var(--accent)] py-2.5 font-mono text-xs font-bold tracking-widest text-white transition hover:brightness-110"
              onClick={() => setWarningAcked(true)}
            >
              I UNDERSTAND, CONTINUE
            </button>
          </div>
        </div>
      )}

      <div className="flex shrink-0 items-center justify-center">
        <div className="panel flex gap-1 p-1 font-mono text-[11px] font-bold tracking-widest">
          {(['snapshot', 'test'] as const).map((m) => (
            <button
              key={m}
              className="relative rounded-md px-4 py-1.5"
              onClick={() => {
                stop()
                setMode(m)
              }}
            >
              {mode === m && (
                <motion.span
                  layoutId="stimulus-mode-pill"
                  className="absolute inset-0 rounded-md bg-[var(--accent-dim)]"
                  transition={{ type: 'spring', stiffness: 400, damping: 32 }}
                />
              )}
              <span className={`relative ${mode === m ? 'text-[var(--accent)]' : 'text-[var(--text-dim)]'}`}>
                {m === 'snapshot' ? 'LIVE SNAPSHOT' : 'TEST PATTERN'}
              </span>
            </button>
          ))}
        </div>
      </div>

      <div className="flex min-h-0 min-w-0 flex-1 items-center justify-center">
        <AnimatePresence mode="wait">
          {mode === 'test' ? (
            <motion.div
              key="test"
              initial={{ opacity: 0, scale: 0.98 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.98 }}
              transition={{ duration: 0.2 }}
              className="corner-brackets relative"
            >
              <canvas ref={canvasRef} width={800} height={800} className="max-h-full max-w-full rounded bg-black" />
              <span className="absolute left-3 top-3 font-mono text-[10px] tracking-widest text-[var(--text-dim)]">
                TEST PATTERN
              </span>
            </motion.div>
          ) : snapshot ? (
            <motion.div
              key="snapshot"
              initial={{ opacity: 0, scale: 0.98 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.98 }}
              transition={{ duration: 0.2 }}
              className="h-full min-h-0 w-full min-w-0"
            >
              <SnapshotFlickerGrid
                imagePath={snapshot.image_path}
                objects={snapshot.objects}
                running={running}
                maxTilePx={560}
                names={namesById}
              />
            </motion.div>
          ) : (
            <motion.div
              key="empty"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="panel flex h-64 w-96 items-center justify-center font-mono text-xs text-[var(--text-dim)]"
            >
              NO SNAPSHOT YET — run export_snapshot.py first
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      <div className="mx-auto flex shrink-0 flex-wrap items-center justify-center gap-5 font-mono text-xs text-[var(--text)]">
        <div className="panel flex flex-wrap items-center gap-5 px-5 py-3">
        {mode === 'test' && (
          <label className="flex items-center gap-2 tracking-widest text-[var(--text-dim)]">
            FREQ
            <select
              className="rounded border border-[var(--border-bright)] bg-[var(--bg-2)] px-2 py-1 text-[var(--text-bright)]"
              value={frequencyHz}
              disabled={running}
              onChange={(e) => setFrequencyHz(Number(e.target.value))}
            >
              {FREQ_OPTIONS.map((f) => (
                <option key={f} value={f}>
                  {f} Hz
                </option>
              ))}
            </select>
          </label>
        )}

        {!running ? (
          <button
            className="rounded-md bg-[var(--good)] px-4 py-1.5 font-bold tracking-widest text-white transition hover:brightness-110 disabled:opacity-30"
            onClick={start}
            disabled={!warningAcked || (mode === 'snapshot' && !snapshot)}
          >
            START
          </button>
        ) : (
          <button
            className="rounded-md bg-[var(--danger)] px-4 py-1.5 font-bold tracking-widest text-white transition hover:brightness-110"
            onClick={stop}
          >
            STOP (ESC)
          </button>
        )}

        <span className="h-4 w-px bg-[var(--border-bright)]" />

        {mode === 'test' ? (
          <>
            <span className="text-[var(--text-dim)]">
              TARGET <b className="text-[var(--text-bright)]">{frequencyHz.toFixed(2)}Hz</b>
            </span>
            <span className="text-[var(--text-dim)]">
              MEASURED <b className="text-[var(--text-bright)]">{readout.measuredHz.toFixed(2)}Hz</b>
            </span>
            <span className="text-[var(--text-dim)]">
              DROPPED{' '}
              <b className={dropRate > 0.05 ? 'text-[var(--danger)]' : 'text-[var(--text-bright)]'}>
                {readout.droppedFrames}/{readout.totalFrames} ({(dropRate * 100).toFixed(1)}%)
              </b>
            </span>
            <span className="text-[var(--text-dim)]">
              ELAPSED <b className="text-[var(--text-bright)]">{readout.elapsedS.toFixed(1)}s</b>
            </span>
          </>
        ) : (
          snapshot && (
            <span className="text-[var(--text-dim)]">
              {snapshot.objects.length} OBJECTS:{' '}
              <b className="text-[var(--text-bright)]">
                {snapshot.objects.map((o) => `T${String(o.id).padStart(2, '0')}@${o.frequency_hz}Hz`).join('  ')}
              </b>
            </span>
          )
        )}
        </div>
      </div>
    </div>
  )
}
