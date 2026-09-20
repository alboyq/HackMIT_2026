import { useEffect, useRef, useState } from 'react'
import GazeHud from '../features/gaze/GazeTracker'
import { useObjectCrops } from '../features/gaze/useObjectCrops'
import { useObjectNames, displayNameFor, checkVisionStatus } from '../features/vision/useObjectNames'
import SquareObjectTile from '../components/SquareObjectTile'
import SpokenCaption from '../components/SpokenCaption'
import { FETCH_MS, HOLD_AFTER_MS, fetchPhaseLabel } from '../lib/fetchMission'
import { announceAbort, announceDelivered, announceLock, silence } from '../lib/speech'
import type { SceneObject } from '../hooks/useHubSocket'

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
  const [fetchPct, setFetchPct] = useState(0)
  const [aiStatus, setAiStatus] = useState<string | null>(null)
  const crops = useObjectCrops(snapshot?.image_path ?? null, snapshot?.objects ?? EMPTY_OBJECTS)
  const { names, ready, provider } = useObjectNames(crops)
  const deliveredRef = useRef(false)
  const lockedRef = useRef(false)

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
    if (confirmed === null) {
      setFetchPct(0)
      deliveredRef.current = false
      return
    }
    const t0 = Date.now()
    setFetchPct(0)
    const id = window.setInterval(() => {
      setFetchPct(Math.min(1, (Date.now() - t0) / FETCH_MS))
    }, 50)
    return () => window.clearInterval(id)
  }, [confirmed])

  useEffect(() => {
    if (confirmed === null || fetchPct < 1 || deliveredRef.current) return
    deliveredRef.current = true
    const crop = crops[confirmed]
    announceDelivered(displayNameFor(crop?.id ?? confirmed + 1, names[confirmed]))
    const t = window.setTimeout(() => {
      lockedRef.current = false
      setConfirmed(null)
      setFetchPct(0)
    }, HOLD_AFTER_MS)
    return () => window.clearTimeout(t)
  }, [fetchPct, confirmed, crops, names])

  function commit(z: number) {
    if (lockedRef.current) return
    lockedRef.current = true
    setConfirmed(z)
    const crop = crops[z]
    announceLock(displayNameFor(crop?.id ?? z + 1, names[z]))
  }

  function abort() {
    const wasLocked = lockedRef.current
    lockedRef.current = false
    if (wasLocked) announceAbort()
    setConfirmed(null)
    setFetchPct(0)
  }

  const slots = [0, 1, 2, 3].map((i) => crops[i] ?? null)
  const loading = Boolean(snapshot && crops.length === 0)
  const fetching = confirmed !== null
  const lockedId = confirmed !== null ? (slots[confirmed]?.id ?? confirmed + 1) : null
  const hudLabel =
    fetching && lockedId !== null
      ? `${fetchPhaseLabel(fetchPct)}  ${displayNameFor(lockedId, names[confirmed!]).toUpperCase()}`
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
            phase={confirmed === i ? fetchPhaseLabel(fetchPct) : null}
            fetchPct={confirmed === i ? fetchPct : 0}
          />
        </div>
      ))}
      <div className="gaze-arena-hud">
        <GazeHud
          lockHeld={fetching}
          fetchLabel={hudLabel}
          fetchPct={fetchPct}
          onZone={setZone}
          onConfirm={commit}
          onAbort={abort}
        />
      </div>
      <div className="pointer-events-none absolute left-1/2 top-3 z-20 -translate-x-1/2 flex gap-3 font-mono text-[10px] tracking-widest">
        {!ready && crops.length > 0 && (
          <span className="text-[var(--text-dim)]">NAMING…</span>
        )}
        {provider && (
          <span className="rounded bg-[var(--accent)]/20 px-1.5 py-0.5 text-[var(--accent)]">
            {provider.toUpperCase()}
          </span>
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
