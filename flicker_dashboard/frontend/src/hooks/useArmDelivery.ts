import { useCallback, useRef, useState } from 'react'
import { useHubSocket, type HubMessage } from './useHubSocket'

export type ArmPhase = 'idle' | 'running' | 'done' | 'failed'

// Rough progress mapping from replay.py's own stage names (see
// arm_teach_replay/replay.py::build_stages) to a 0..1 fraction, purely for
// the existing progress-bar UI -- the real gate for "waiting till the arm
// finishes" is `phase`, not this number.
const STAGE_PCT: Record<string, number> = {
  QUEUED: 0.03,
  home: 0.08,
  hover: 0.22,
  grasp: 0.42,
  lift: 0.58,
  place: 0.8,
  release: 0.92,
  DONE: 1,
}

const STAGE_LABEL: Record<string, string> = {
  QUEUED: 'LOCKED',
  home: 'RETURNING',
  hover: 'REACHING',
  grasp: 'GRASPING',
  lift: 'LIFTING',
  place: 'DELIVERING',
  release: 'RELEASING',
  DONE: 'IN HAND',
  FAILED: 'ARM FAILED',
  BUSY: 'ARM BUSY',
}

export function armStageLabel(stage: string | null): string {
  if (stage === null) return 'LOCKED'
  return STAGE_LABEL[stage] ?? 'WORKING'
}

/** Sends `arm.target` when a selection locks in, and turns the backend's
 * `arm.status` broadcasts (real progress from arm_teach_replay/replay.py
 * running on the physical arm) into simple state a page can render and gate
 * its "waiting for the arm" UI on. One in-flight delivery at a time, matching
 * the backend's own single-arm lock. */
export function useArmDelivery() {
  const [stage, setStage] = useState<string | null>(null)
  const [phase, setPhase] = useState<ArmPhase>('idle')
  const [message, setMessage] = useState<string | undefined>()
  const activeId = useRef<number | null>(null)

  const { send, connected } = useHubSocket((msg: HubMessage) => {
    if (msg.type !== 'arm.status') return
    if (activeId.current !== null && msg.object_id !== activeId.current) return
    setStage(msg.stage)
    setMessage(msg.message)
    if (msg.stage === 'DONE') setPhase('done')
    else if (msg.stage === 'FAILED' || msg.stage === 'BUSY') setPhase('failed')
    else setPhase('running')
  })

  const trigger = useCallback(
    (objectId: number, label?: string) => {
      activeId.current = objectId
      setPhase('running')
      setStage('QUEUED')
      setMessage(undefined)
      send({ type: 'arm.target', object_id: objectId, label })
    },
    [send],
  )

  const reset = useCallback(() => {
    activeId.current = null
    setPhase('idle')
    setStage(null)
    setMessage(undefined)
  }, [])

  const pct = stage !== null ? (STAGE_PCT[stage] ?? (phase === 'done' ? 1 : 0.5)) : 0

  return { trigger, reset, stage, phase, pct, message, connected }
}
