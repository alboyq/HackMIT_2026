import { useEffect, useRef, useState } from 'react'
import { useObjectCrops } from '../features/gaze/useObjectCrops'
import { useObjectNames, displayNameFor, checkVisionStatus } from '../features/vision/useObjectNames'
import SquareObjectTile from '../components/SquareObjectTile'
import SpokenCaption from '../components/SpokenCaption'
import { HOLD_AFTER_MS } from '../lib/fetchMission'
import {
  announceAbort,
  announceDelivered,
  announceHelp,
  announceLock,
  announceScene,
  announceStatus,
  createRecognizer,
  lastSpokenLine,
  speechRecognitionAvailable,
  speak,
} from '../lib/speech'
import { matchVoiceCommand } from '../lib/voiceMatch'
import type { SceneObject } from '../hooks/useHubSocket'
import { armStageLabel, useArmDelivery } from '../hooks/useArmDelivery'

interface Snapshot {
  snapshot_id: string
  image_path: string
  objects: SceneObject[]
}

const EMPTY_OBJECTS: SceneObject[] = []

export default function VoicePage() {
  const [snapshot, setSnapshot] = useState<Snapshot | null>(null)
  const [confirmed, setConfirmed] = useState<number | null>(null)
  const [listening, setListening] = useState(false)
  const [transcript, setTranscript] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [aiStatus, setAiStatus] = useState<string | null>(null)
  const crops = useObjectCrops(snapshot?.image_path ?? null, snapshot?.objects ?? EMPTY_OBJECTS)
  const { names, ready } = useObjectNames(crops)
  const deliveredRef = useRef(false)
  const arm = useArmDelivery()

  useEffect(() => {
    checkVisionStatus().then((s) => setAiStatus(s.provider || 'none'))
  }, [])
  const briefedRef = useRef(false)
  const lockRef = useRef<number | null>(null)
  lockRef.current = confirmed
  const namesRef = useRef(names)
  namesRef.current = names
  const cropsRef = useRef(crops)
  cropsRef.current = crops
  const armRef = useRef(arm)
  armRef.current = arm
  const stopRef = useRef<(() => void) | null>(null)

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
        setConfirmed(null)
        arm.reset()
        setTranscript('')
      }, HOLD_AFTER_MS)
      return () => window.clearTimeout(t)
    }
    if (arm.phase === 'failed') {
      deliveredRef.current = true
      speak(arm.message ? `Arm trouble: ${arm.message}` : 'The arm ran into trouble.', { interrupt: true })
      const t = window.setTimeout(() => {
        setConfirmed(null)
        arm.reset()
        setTranscript('')
      }, HOLD_AFTER_MS)
      return () => window.clearTimeout(t)
    }
  }, [arm.phase, arm.message, confirmed, crops, names])

  useEffect(() => {
    if (!ready || names.length === 0 || briefedRef.current) return
    briefedRef.current = true
    announceScene(names.filter((n): n is string => Boolean(n)))
  }, [ready, names])

  function commit(z: number) {
    // Don't allow new commits while something is locked/fetching
    if (lockRef.current !== null) {
      // But if delivery is complete, allow new order
      if (armRef.current.phase === 'done' || armRef.current.phase === 'failed') {
        setConfirmed(null)
        armRef.current.reset()
        deliveredRef.current = false
      } else {
        return
      }
    }
    const crop = cropsRef.current[z]
    const label = displayNameFor(crop?.id ?? z + 1, namesRef.current[z])
    announceLock(label)
    setConfirmed(z)
    armRef.current.trigger(crop?.id ?? z + 1, label)
  }

  function abort() {
    if (lockRef.current !== null) announceAbort()
    setConfirmed(null)
    arm.reset()
  }

  function onFinal(text: string) {
    const hit = matchVoiceCommand(text, namesRef.current)
    if (!hit) {
      if (/\b(get|bring|fetch|want|select|lock|target|give)\b/.test(text.toLowerCase())) {
        speak(`I heard ${text}, but I don't know that item.`)
      }
      return
    }
    if (hit.kind === 'abort') {
      abort()
      return
    }
    if (hit.kind === 'scene') {
      announceScene(namesRef.current.filter((n): n is string => Boolean(n)))
      return
    }
    if (hit.kind === 'repeat') {
      const prev = lastSpokenLine()
      if (prev) speak(prev, { interrupt: true })
      return
    }
    if (hit.kind === 'status') {
      const z = lockRef.current
      const crop = z !== null ? cropsRef.current[z] : null
      announceStatus(
        z !== null ? displayNameFor(crop?.id ?? z + 1, namesRef.current[z]) : null,
        z !== null ? armStageLabel(armRef.current.stage) : '',
      )
      return
    }
    if (hit.kind === 'help') {
      announceHelp()
      return
    }
    commit(hit.index)
  }

  function startListening() {
    setError(null)
    const rec = createRecognizer({
      onTranscript: (text, isFinal) => {
        setTranscript(text)
        if (isFinal) onFinal(text)
      },
      onError: (message) => {
        setError(message)
        setListening(false)
      },
    })
    if (!rec) {
      setError('Voice needs Chrome or Edge (Web Speech API).')
      return
    }
    try {
      rec.start()
      setListening(true)
      stopRef.current = rec.stop
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Could not start the microphone.')
    }
  }

  function stopListening() {
    stopRef.current?.()
    stopRef.current = null
    setListening(false)
  }

  useEffect(() => () => stopRef.current?.(), [])

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return
      if (e.key === 'Escape') {
        abort()
        return
      }
      const n = Number(e.key)
      if (n >= 1 && n <= 4) commit(n - 1)
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])

  const slots = [0, 1, 2, 3].map((i) => crops[i] ?? null)
  const loading = Boolean(snapshot && crops.length === 0)
  const fetching = confirmed !== null
  const supported = speechRecognitionAvailable()

  return (
    <div className="gaze-stage relative flex h-full min-h-0 w-full flex-col overflow-hidden">
      <div className="grid min-h-0 flex-1 grid-cols-2 grid-rows-2 gap-6 p-6 pb-3">
        {[0, 1, 2, 3].map((i) => (
          <div key={i} className="gaze-slot h-full w-full">
            <SquareObjectTile
              crop={slots[i]}
              index={i}
              name={names[i]}
              looking={false}
              locked={confirmed === i}
              fetching={fetching}
              loading={loading}
              fill
              onPick={commit}
              phase={confirmed === i ? armStageLabel(arm.stage) : null}
              fetchPct={confirmed === i ? arm.pct : 0}
            />
          </div>
        ))}
      </div>

      <div className="relative z-10 mx-auto mb-4 flex w-[min(640px,92vw)] flex-col items-center gap-2 rounded-2xl border border-[var(--border)] bg-[var(--bg-1)]/80 px-5 py-3 backdrop-blur-md">
        <div className="flex items-center gap-3">
          <span
            className="h-2.5 w-2.5 rounded-full"
            style={{
              background: listening ? 'var(--accent)' : 'var(--border-bright)',
              boxShadow: listening ? '0 0 12px var(--accent-glow)' : undefined,
            }}
          />
          <span className="font-mono text-[10px] tracking-widest text-[var(--text-dim)]">
            {listening ? 'LISTENING' : 'MIC OFF'}
            {ready ? '' : ' · NAMING…'}
            {fetching ? ` · ${armStageLabel(arm.stage)}` : ''}
          </span>
          {aiStatus === 'none' && ready && (
            <span className="rounded bg-[var(--danger)]/20 px-1.5 py-0.5 font-mono text-[9px] tracking-widest text-[var(--danger)]">
              NO API KEY
            </span>
          )}
        </div>
        <p className="min-h-6 text-center font-mono text-sm text-[var(--text-bright)]">
          {transcript ||
            (supported
              ? 'Say “get me the can”, “target two”, “what do you see”, or “abort”.'
              : 'Chrome or Edge required for voice.')}
        </p>
        {error && <p className="text-center font-mono text-[11px] text-[var(--danger)]">{error}</p>}
        <div className="flex gap-2">
          {!listening ? (
            <button
              type="button"
              onClick={startListening}
              disabled={!supported}
              className="rounded-full bg-[var(--accent)] px-5 py-2 font-mono text-[11px] font-bold tracking-widest text-white disabled:opacity-40"
            >
              START LISTENING
            </button>
          ) : (
            <button
              type="button"
              onClick={stopListening}
              className="rounded-full border border-[var(--border-bright)] px-5 py-2 font-mono text-[11px] tracking-widest text-[var(--text)]"
            >
              MUTE
            </button>
          )}
          {fetching && (
            <button
              type="button"
              onClick={abort}
              className="rounded-full border border-[var(--danger)] px-5 py-2 font-mono text-[11px] tracking-widest text-[var(--danger)]"
            >
              {arm.phase === 'done' || arm.phase === 'failed' ? 'NEW PICK' : 'ABORT'}
            </button>
          )}
        </div>
      </div>
      <SpokenCaption className="!bottom-36" />
    </div>
  )
}
