import { useEffect, useRef, useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { useHubSocket, type HubMessage, type SceneObject } from '../hooks/useHubSocket'
import SnapshotFlickerGrid from '../components/SnapshotFlickerGrid'
import AnimatedNumber from '../components/AnimatedNumber'
import ConfidenceTrace from '../components/ConfidenceTrace'
import { colorForIndex } from '../lib/colors'
import { useObjectCrops } from '../features/gaze/useObjectCrops'
import { useObjectNames } from '../features/vision/useObjectNames'

// M2 skeleton (Flicker_and_Dashboard_PRD.md section 6/9): scene view (D1),
// decoder confidence (D5), pipeline stage (D6), event log (D10) against the
// mock server. Not built yet, pending direction on the PRD's open questions
// (section 12): EEG traces (D2), head map (D3), SSVEP spectrum (D4), robot
// twin (D7), rover map (D8), ErrP indicator (D9), metrics strip (D11),
// operator controls / e-stop (D12).

const MODE = 'MOCK' // TODO: real mode switch (Live/Replay/Scripted) per section 6 "Modes"
const STAGES = ['IDLE', 'SNAPSHOT', 'FLICKER', 'SELECT', 'PLAN', 'EXECUTE', 'DONE']

// How many decoder.scores ticks the confidence trace keeps on screen. The
// mock feed (and the real decoder's window_s) ticks ~every 100ms, so 50
// samples is a rolling ~5s window — deliberately matched to how often a
// winner actually changes, so the trace always shows one full run-up.
const HISTORY_LEN = 50

interface LogEntry {
  id: number
  t: string
  text: string
  kind: 'info' | 'lock' | 'warn' | 'lead'
}

const LOG_KIND_STYLE: Record<LogEntry['kind'], { color: string; glyph: string; bold?: boolean }> = {
  info: { color: 'var(--text)', glyph: '›' },
  lead: { color: 'var(--teal)', glyph: '◆' },
  lock: { color: 'var(--accent)', glyph: '✓', bold: true },
  warn: { color: 'var(--danger)', glyph: '⚠' },
}

export default function DashboardPage() {
  const [objects, setObjects] = useState<SceneObject[]>([])
  const [imagePath, setImagePath] = useState<string | null>(null)
  const [scores, setScores] = useState<Record<string, number>>({})
  const [stage, setStage] = useState('IDLE')
  const [log, setLog] = useState<LogEntry[]>([])
  const [warningAcked, setWarningAcked] = useState(false)
  const nextLogId = useRef(0)

  // Rolling per-object score history (D5 confidence trace) + winner-change
  // drama (flash + log entry) + real elapsed-time telemetry (D6 stepper).
  // All local to this component and consumed only by the three owned panels.
  const [scoreHistory, setScoreHistory] = useState<Record<string, number[]>>({})
  const [winnerFlash, setWinnerFlash] = useState(0)
  const prevWinnerRef = useRef<string | undefined>(undefined)
  const stageEnteredAtRef = useRef<number>(Date.now())
  const prevStageRef = useRef('IDLE')
  const runStartRef = useRef<number | null>(null)
  const [stageDurations, setStageDurations] = useState<Record<string, number>>({})
  const [, setClockTick] = useState(0)
  const objectsRef = useRef(objects)
  objectsRef.current = objects

  // Real object names via cloud vision (falls back to "target NN" per-object
  // if the vision API is unconfigured/unreachable — see useObjectNames). Kept
  // in a ref too so the WS message handler below always reads the latest
  // names without needing to be redeclared per the objectsRef pattern above.
  const crops = useObjectCrops(imagePath, objects)
  const { names: visionNames } = useObjectNames(crops)
  const namesById: Record<number, string> = {}
  crops.forEach((c, i) => {
    if (visionNames[i]) namesById[c.id] = visionNames[i]!.toUpperCase()
  })
  const namesByIdRef = useRef(namesById)
  namesByIdRef.current = namesById

  function pushLog(text: string, kind: LogEntry['kind'] = 'info') {
    const t = new Date().toLocaleTimeString()
    setLog((prev) => [{ id: nextLogId.current++, t, text, kind }, ...prev].slice(0, 50))
  }

  function handleMessage(msg: HubMessage) {
    switch (msg.type) {
      case 'scene.snapshot':
        setObjects(msg.objects)
        if (typeof msg.image_path === 'string') setImagePath(msg.image_path)
        pushLog(`Snapshot ${msg.snapshot_id}: ${msg.objects.length} objects`)
        break
      case 'decoder.scores':
        setScores(msg.scores)
        break
      case 'decoder.selected': {
        const name = namesByIdRef.current[msg.id] || `TARGET ${String(msg.id).padStart(2, '0')}`
        pushLog(`Target locked: ${name} (${msg.confidence.toFixed(2)})`, 'lock')
        // Mock hub emits decoder.selected every ~5s. Do not speak that —
        // spoken lock is only for a real gaze / voice confirm.
        break
      }
      case 'robot.state':
        setStage(msg.stage)
        break
      default:
        break
    }
  }

  const { connected } = useHubSocket(handleMessage)

  const [flickerKilled, setFlickerKilled] = useState(false)
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setFlickerKilled(true)
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])

  const winnerId = Object.entries(scores).sort((a, b) => b[1] - a[1])[0]?.[0]
  const winnerObj = objects.find((o) => String(o.id) === winnerId)
  const winnerLabel = winnerObj ? namesById[winnerObj.id] || `TARGET ${String(winnerObj.id).padStart(2, '0')}` : undefined
  const stageIdx = STAGES.indexOf(stage)
  const flickerRunning = warningAcked && !flickerKilled && stage === 'FLICKER'

  // D5: append each tick's scores into a rolling per-object history so the
  // confidence trace can show a trajectory, not just an instant.
  useEffect(() => {
    setScoreHistory((prev) => {
      const next: Record<string, number[]> = { ...prev }
      for (const o of objects) {
        const key = String(o.id)
        const arr = next[key] ? [...next[key], scores[key] ?? 0] : [scores[key] ?? 0]
        next[key] = arr.length > HISTORY_LEN ? arr.slice(arr.length - HISTORY_LEN) : arr
      }
      return next
    })
  }, [scores, objects])

  // D5 + D10: the moment the provisional leader flips is the single most
  // important instant in the dashboard (PRD narrative: the decision forming
  // live) — mark it with a panel flash and a distinct log entry, not just a
  // color swap.
  useEffect(() => {
    if (!winnerId) return
    if (prevWinnerRef.current !== undefined && prevWinnerRef.current !== winnerId) {
      setWinnerFlash((k) => k + 1)
      pushLog(`New leader: ${winnerLabel ?? `object ${winnerId}`}`, 'lead')
    }
    prevWinnerRef.current = winnerId
  }, [winnerId, winnerLabel])

  // D6: real elapsed-time-per-stage telemetry — record how long the previous
  // stage actually ran for, and when the current run started, from real
  // Date.now() deltas driven by the actual robot.state messages.
  useEffect(() => {
    if (prevStageRef.current !== stage) {
      const now = Date.now()
      const finishedStage = prevStageRef.current
      setStageDurations((d) => ({ ...d, [finishedStage]: now - stageEnteredAtRef.current }))
      stageEnteredAtRef.current = now
      prevStageRef.current = stage
      if (stage !== 'IDLE' && runStartRef.current === null) runStartRef.current = now
      if (stage === 'IDLE') runStartRef.current = null
    }
  }, [stage])

  // D6: tick a render every 200ms purely so the live "elapsed" readouts
  // below count up in real time instead of only updating on new messages.
  useEffect(() => {
    const id = window.setInterval(() => setClockTick((t) => t + 1), 200)
    return () => window.clearInterval(id)
  }, [])

  const currentStageElapsedMs = Date.now() - stageEnteredAtRef.current
  const totalElapsedMs = runStartRef.current !== null ? Date.now() - runStartRef.current : 0

  return (
    <div className="hud-grid flex h-full min-w-0 w-full flex-col gap-3 overflow-x-hidden bg-[var(--bg-0)] p-4" style={{ fontFamily: 'var(--font-sans)' }}>
      {!warningAcked && (
        <div className="fixed inset-0 z-20 flex items-center justify-center bg-black/95 backdrop-blur-sm">
          <div className="corner-brackets panel relative max-w-md p-7" style={{ borderColor: 'var(--danger)' }}>
            <div className="mb-3 flex items-center gap-2">
              <span className="h-2 w-2 animate-pulse-glow rounded-full bg-[var(--danger)]" />
              <h2 className="font-mono text-xs font-bold tracking-widest text-[var(--danger)]">
                PHOTOSENSITIVITY WARNING
              </h2>
            </div>
            <p className="mb-5 text-sm leading-relaxed text-[var(--text)]">
              The scene view on this dashboard flickers at 8.57&ndash;15 Hz while a target is being
              selected. Flicker in this range can trigger seizures in people with photosensitive
              epilepsy. Press{' '}
              <kbd className="rounded-[2px] border border-[var(--border-bright)] bg-[var(--bg-2)] px-1.5 py-0.5 font-mono text-xs">
                Esc
              </kbd>{' '}
              at any time to kill it.
            </p>
            <button
              className="w-full rounded-[2px] bg-[var(--accent)] py-2.5 font-mono text-xs font-bold tracking-widest text-white transition hover:brightness-110"
              onClick={() => setWarningAcked(true)}
            >
              I UNDERSTAND, CONTINUE
            </button>
          </div>
        </div>
      )}

      {/* Status bar */}
      <header className="flex min-w-0 shrink-0 flex-wrap items-center justify-between gap-2 border-b border-[var(--border)] pb-3">
        <h2 className="font-mono text-[10px] tracking-widest text-[var(--text-dim)]">SCENE / DECODER / PIPELINE</h2>
        <div className="flex min-w-0 flex-wrap items-center gap-3">
          <span className="rounded-[2px] border border-[var(--border)] px-2.5 py-1 font-mono text-[11px] font-semibold tracking-widest text-[var(--text-dim)]">
            MODE: {MODE}
          </span>
          <span
            className={`flex items-center gap-1.5 rounded-[2px] border px-2.5 py-1 font-mono text-[11px] tracking-widest ${
              connected
                ? 'border-[var(--accent)] bg-[var(--accent-dim)] text-[var(--accent)]'
                : 'border-[var(--danger)] text-[var(--danger)]'
            }`}
          >
            <span
              className={`h-1.5 w-1.5 rounded-full ${connected ? 'animate-pulse-glow bg-[var(--accent)]' : 'bg-[var(--danger)]'}`}
            />
            {connected ? 'LIVE' : 'RECONNECTING'}
          </span>
          {flickerKilled ? (
            <button
              className="rounded-[2px] border border-[var(--danger)] px-2.5 py-1 font-mono text-[11px] font-bold tracking-widest text-[var(--danger)]"
              onClick={() => setFlickerKilled(false)}
            >
              FLICKER KILLED &mdash; RESUME
            </button>
          ) : (
            <span className="font-mono text-[11px] tracking-widest text-[var(--text-dim)]">ESC = KILL FLICKER</span>
          )}
        </div>
      </header>

      <div className="grid min-h-0 min-w-0 flex-1 grid-cols-[minmax(0,1fr)_minmax(280px,340px)] gap-3">
        <div className="flex min-h-0 min-w-0 flex-col gap-3">
          {/* D1: scene view (hero) — each object separated into its own tile,
              same treatment as /stimulus, so this reflects what the wearer
              actually sees rather than a single annotated photo. */}
          <div className="corner-brackets panel relative flex min-h-0 min-w-0 flex-1 flex-col overflow-hidden p-4">
            <div className="mb-3 flex shrink-0 items-center justify-between font-mono text-[10px] tracking-widest text-[var(--text-dim)]">
              <span>SCENE VIEW</span>
              <span className="tabular-nums">
                {objects.length} OBJECT{objects.length === 1 ? '' : 'S'}
              </span>
            </div>
            {imagePath ? (
              <div className="min-h-0 min-w-0 flex-1">
                <SnapshotFlickerGrid
                  imagePath={imagePath}
                  objects={objects}
                  running={flickerRunning}
                  winnerId={winnerId}
                  maxTilePx={360}
                  names={namesById}
                />
              </div>
            ) : (
              <div className="flex h-40 items-center justify-center font-mono text-sm text-[var(--text-dim)]">
                NO SNAPSHOT
              </div>
            )}
          </div>

          {/* D6: pipeline stepper */}
          <div className="panel flex min-w-0 flex-col gap-2 px-3 py-2.5">
            <div className="flex min-w-0 flex-wrap items-start justify-between">
              {STAGES.map((s, i) => {
                const isDone = i < stageIdx
                const isActive = i === stageIdx
                const durationMs = stageDurations[s]
                return (
                  <div key={s} className="flex min-w-0 flex-1 items-center last:flex-none">
                    <div className="relative flex min-w-0 flex-col items-center gap-0.5 px-1 py-0.5">
                      {isActive && (
                        <motion.span
                          layoutId="stage-pill"
                          className="absolute inset-0 rounded-md bg-[var(--accent-dim)]"
                          transition={{ type: 'spring', stiffness: 350, damping: 30 }}
                        />
                      )}
                      <motion.div
                        className="relative h-2 w-2 rounded-full border"
                        animate={{
                          borderColor: isActive ? 'var(--accent)' : isDone ? 'var(--teal)' : 'var(--border-bright)',
                          backgroundColor: i <= stageIdx ? (isActive ? 'var(--accent)' : 'var(--teal)') : 'transparent',
                          scale: isActive ? 1.15 : 1,
                        }}
                      />
                      <span
                        className={`relative max-w-full truncate font-mono text-[9px] tracking-wider ${
                          isActive ? 'text-[var(--accent)]' : isDone ? 'text-[var(--teal)]' : 'text-[var(--text-dim)]'
                        }`}
                      >
                        {s}
                      </span>
                      <span className="relative font-mono text-[9px] tabular-nums text-[var(--text-dim)]">
                        {isActive
                          ? `${(currentStageElapsedMs / 1000).toFixed(1)}s`
                          : durationMs !== undefined
                            ? `${(durationMs / 1000).toFixed(1)}s`
                            : '—'}
                      </span>
                    </div>
                    {i < STAGES.length - 1 && (
                      <div className="relative mx-1 h-px min-w-2 flex-1 overflow-hidden bg-[var(--border)]">
                        <motion.div
                          className="absolute inset-y-0 left-0 bg-[var(--teal)]"
                          animate={{ width: i < stageIdx ? '100%' : '0%' }}
                          transition={{ duration: 0.3 }}
                        />
                      </div>
                    )}
                  </div>
                )
              })}
            </div>
            <div className="flex min-w-0 items-center justify-between gap-2 border-t border-[var(--border)] pt-1.5 font-mono text-[10px] tracking-widest text-[var(--text-dim)]">
              <span className="min-w-0 truncate">
                RUN ELAPSED <span className="tabular-nums text-[var(--text-bright)]">{(totalElapsedMs / 1000).toFixed(1)}s</span>
              </span>
              <span className="shrink-0">
                STAGE <span className="text-[var(--accent)]">{stage}</span>
                {' · '}
                <span className="tabular-nums text-[var(--text-bright)]">{(currentStageElapsedMs / 1000).toFixed(1)}s</span>
              </span>
            </div>
          </div>
        </div>

        <div className="flex min-h-0 min-w-0 flex-col gap-3 overflow-x-hidden overflow-y-auto">
          {/* D5: decoder confidence */}
          <div className="panel relative min-w-0 overflow-hidden p-4">
            <h3 className="mb-3 flex items-center justify-between font-mono text-[10px] tracking-widest text-[var(--text-dim)]">
              <span>DECODER CONFIDENCE</span>
              <span className="tabular-nums">LAST {(HISTORY_LEN / 10).toFixed(0)}S</span>
            </h3>
            {objects.length === 0 && <p className="font-mono text-xs text-[var(--text-dim)]">NO DATA</p>}

            {/* Lock-in flash: the instant a new candidate takes the lead is
                the actual narrative payload of this panel, so it gets a
                distinct visual event, not just a re-colored bar. */}
            <AnimatePresence>
              {winnerFlash > 0 && (
                <motion.div
                  key={winnerFlash}
                  className="pointer-events-none absolute inset-0 z-10"
                  initial={{ opacity: 0.55 }}
                  animate={{ opacity: 0 }}
                  exit={{ opacity: 0 }}
                  transition={{ duration: 0.7, ease: 'easeOut' }}
                  style={{ background: 'radial-gradient(circle at 15% 10%, var(--accent-glow), transparent 65%)' }}
                />
              )}
            </AnimatePresence>

            {winnerLabel && (
              <motion.div
                key={winnerFlash}
                initial={{ scale: 0.96 }}
                animate={{ scale: 1 }}
                transition={{ type: 'spring', stiffness: 320, damping: 15 }}
                className="relative mb-3 flex items-baseline gap-3 border-b border-[var(--border)] pb-3"
              >
                <span className="flex items-baseline font-mono text-4xl font-bold tabular-nums text-[var(--accent)]">
                  <AnimatedNumber value={Math.round((scores[winnerId!] ?? 0) * 100)} />
                  <span className="text-lg text-[var(--text-dim)]">%</span>
                </span>
                <div className="flex flex-col">
                  <span className="font-mono text-[10px] tracking-widest text-[var(--text-dim)]">LEADING</span>
                  <AnimatePresence mode="wait">
                    <motion.span
                      key={winnerLabel}
                      initial={{ opacity: 0, y: -6 }}
                      animate={{ opacity: 1, y: 0 }}
                      exit={{ opacity: 0, y: 6 }}
                      transition={{ duration: 0.18 }}
                      className="font-mono text-sm font-bold text-[var(--text-bright)]"
                    >
                      {winnerLabel.toUpperCase()}
                    </motion.span>
                  </AnimatePresence>
                </div>
              </motion.div>
            )}

            {/* The trace is the actual "wow": it shows the winner *climbing*
                out of the pack over the last few seconds, not a static bar. */}
            <div className="relative mb-3 rounded border border-[var(--border)] bg-[var(--bg-1)]/60 px-1 py-1">
              <ConfidenceTrace
                height={104}
                series={objects.map((o, i) => {
                  const key = String(o.id)
                  return {
                    id: key,
                    color: key === winnerId ? 'var(--accent)' : colorForIndex(i),
                    values: scoreHistory[key] ?? [],
                    isLeader: key === winnerId,
                  }
                })}
              />
            </div>

            <div className="flex min-w-0 flex-col gap-1.5">
              {objects.map((o, i) => {
                const key = String(o.id)
                const score = scores[key] ?? 0
                const hist = scoreHistory[key] ?? []
                const prevScore = hist.length > 1 ? hist[hist.length - 2] : score
                const delta = score - prevScore
                const isTarget = key === winnerId
                const color = isTarget ? 'var(--accent)' : colorForIndex(i)
                const trend = delta > 0.02 ? 'up' : delta < -0.02 ? 'down' : 'flat'
                return (
                  <motion.div
                    key={o.id}
                    animate={{ backgroundColor: isTarget ? 'var(--accent-dim)' : 'rgba(0,0,0,0)' }}
                    className="grid min-w-0 grid-cols-[8px_minmax(0,1fr)_3.25rem_minmax(2rem,1fr)_14px_2.75rem] items-center gap-2 rounded px-1.5 py-1"
                  >
                    <span
                      className="h-2 w-2 shrink-0 rounded-full"
                      style={{ background: color, boxShadow: isTarget ? '0 0 6px var(--accent-glow)' : undefined }}
                    />
                    <span className="min-w-0 truncate font-mono text-xs text-[var(--text)]">
                      {namesById[o.id] || `TARGET ${String(o.id).padStart(2, '0')}`}
                    </span>
                    <span className="font-mono text-[9px] tabular-nums text-[var(--text-dim)]">
                      {o.frequency_hz.toFixed(2)}Hz
                    </span>
                    <div className="relative h-1.5 min-w-0 overflow-hidden rounded-sm bg-[var(--bg-2)]">
                      <motion.div
                        className="h-full rounded-sm"
                        animate={{ width: `${score * 100}%` }}
                        transition={{ type: 'spring', stiffness: 160, damping: 22 }}
                        style={{ background: color }}
                      />
                    </div>
                    <span
                      className={`text-center text-xs ${
                        trend === 'up' ? 'text-[var(--good)]' : trend === 'down' ? 'text-[var(--danger)]' : 'text-[var(--text-dim)]'
                      }`}
                    >
                      {trend === 'up' ? '▲' : trend === 'down' ? '▼' : '·'}
                    </span>
                    <span
                      className="text-right font-mono text-xs font-bold tabular-nums"
                      style={{ color: isTarget ? 'var(--accent)' : 'var(--text-bright)' }}
                    >
                      {score.toFixed(2)}
                    </span>
                  </motion.div>
                )
              })}
            </div>
          </div>

          {/* D10: event log */}
          <div className="panel flex min-h-0 min-w-0 flex-1 flex-col overflow-hidden p-4">
            <h3 className="mb-2 font-mono text-[10px] tracking-widest text-[var(--text-dim)]">
              EVENT LOG
            </h3>
            <ul className="min-w-0 flex-1 space-y-1.5 overflow-y-auto overflow-x-hidden font-mono text-[11px] leading-relaxed">
              {log.length === 0 && <li className="text-[var(--text-dim)]">NO EVENTS</li>}
              <AnimatePresence initial={false}>
                {log.map((entry) => {
                  const style = LOG_KIND_STYLE[entry.kind]
                  return (
                    <motion.li
                      key={entry.id}
                      initial={{ opacity: 0, x: -8, height: 0 }}
                      animate={{ opacity: 1, x: 0, height: 'auto' }}
                      exit={{ opacity: 0, height: 0 }}
                      transition={{ duration: 0.2 }}
                      className="flex min-w-0 items-baseline gap-2"
                    >
                      <span className="shrink-0 tabular-nums text-[var(--text-dim)]">{entry.t}</span>
                      <span className="shrink-0" style={{ color: style.color }}>{style.glyph}</span>
                      <span
                        className="min-w-0 truncate"
                        style={{ color: style.color, fontWeight: style.bold ? 700 : 400 }}
                      >
                        {entry.text}
                      </span>
                    </motion.li>
                  )
                })}
              </AnimatePresence>
            </ul>
          </div>
        </div>
      </div>
    </div>
  )
}
