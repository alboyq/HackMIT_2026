import { useEffect, useRef, useState } from 'react'
import GazeHud from '../features/gaze/GazeTracker'
import { useObjectCrops } from '../features/gaze/useObjectCrops'
import { useObjectNames, displayNameFor, checkVisionStatus } from '../features/vision/useObjectNames'
import SquareObjectTile from '../components/SquareObjectTile'
import SpokenCaption from '../components/SpokenCaption'
import { HOLD_AFTER_MS } from '../lib/fetchMission'
import { announceAbort, announceDelivered, announceLock, speak, silence } from '../lib/speech'
import type { SceneObject } from '../hooks/useHubSocket'
import { armStageLabel, useArmDelivery } from '../hooks/useArmDelivery'

interface Snapshot {
  snapshot_id: string
  image_path: string
  objects: SceneObject[]
}

const EMPTY_OBJECTS: SceneObject[] = []
const POS: Array<'tl' | 'tr' | 'bl' | 'br'> = ['tl', 'tr', 'bl', 'br']

export default function GazePage() {
  const [snapshot, setSnapshot] = useState<Snapshot | null>(null)
  const [zone, setZone] = useState<number | null>(null)
  const [confirmed, setConfirmed] = useState<number | null>(null)
  const [aiStatus, setAiStatus] = useState<string | null>(null)
  const crops = useObjectCrops(snapshot?.image_path ?? null, snapshot?.objects ?? EMPTY_OBJECTS)
  const { names, ready } = useObjectNames(crops)
  const deliveredRef = useRef(false)
  const lockedRef = useRef(false)
  const arm = useArmDelivery()

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

  // Waits for the REAL arm.status broadcasts (the arm executor running on the
  // physical arm over SSH) -- not a fixed-duration animation. `DONE` and
  // `FAILED` both unlock; a failure gets its own brief spoken notice instead
  // of "delivered".
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

  const slots = [0, 1, 2, 3].map((i) => crops[i] ?? null)
  const loading = Boolean(snapshot && crops.length === 0)
  const fetching = confirmed !== null
  const lockedId = confirmed !== null ? (slots[confirmed]?.id ?? confirmed + 1) : null
  const hudLabel =
    fetching && lockedId !== null
      ? `${armStageLabel(arm.stage)}  ${displayNameFor(lockedId, names[confirmed!]).toUpperCase()}`
      : null

  return (
    <div className="gaze-stage gaze-arena relative h-full min-h-0 w-full overflow-hidden">
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
            looking={zone === i}
            locked={confirmed === i}
            fetching={fetching}
            loading={loading}
            phase={confirmed === i ? armStageLabel(arm.stage) : null}
            fetchPct={confirmed === i ? arm.pct : 0}
            showId={false}
          />
        </div>
      ))}
      <div className="gaze-arena-hud">
        <GazeHud
          lockHeld={fetching}
          fetchLabel={hudLabel}
          fetchPct={arm.pct}
          onZone={setZone}
          onConfirm={commit}
          onAbort={abort}
        />
      </div>
      <div className="pointer-events-none absolute left-1/2 top-3 z-20 -translate-x-1/2 flex gap-3 font-mono text-[10px] tracking-widest">
        {!ready && crops.length > 0 && (
          <span className="text-[var(--text-dim)]">NAMING…</span>
        )}
        {aiStatus === 'none' && ready && (
          <span className="rounded bg-[var(--danger)]/20 px-1.5 py-0.5 text-[var(--danger)]">
            NO API KEY
          </span>
        )}
      </div>
      <SpokenCaption />
    </div>
  )
}
