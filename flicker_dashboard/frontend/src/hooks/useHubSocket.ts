import { useEffect, useRef, useState } from 'react'

// Section 7 message contract. Extend as real producers (decoder/robot/EEG) land.
export type HubMessage =
  | { type: 'scene.snapshot'; snapshot_id: string; objects: SceneObject[]; [k: string]: unknown }
  | { type: 'stim.started'; snapshot_id: string; t_ms: number; measured_refresh_hz: number; table: Record<string, number> }
  | { type: 'stim.stopped'; [k: string]: unknown }
  | { type: 'decoder.scores'; window_s: number; scores: Record<string, number>; state: string }
  | { type: 'decoder.selected'; id: number; confidence: number }
  | { type: 'robot.state'; stage: string; joints: Record<string, number>; gripper: number }
  | { type: 'eeg.chunk'; fs: number; channels: string[]; data: number[][] }
  | { type: 'robot.command'; action: string }

export interface SceneObject {
  id: number
  label: string
  score: number
  frequency_hz: number
  bbox_px: [number, number, number, number]
  centroid_px: [number, number]
  area_px: number
  mask_url: string
}

const WS_URL = `ws://${window.location.hostname}:8000/ws`
const RECONNECT_DELAY_MS = 1000

/** Connects to the backend hub, auto-reconnects on drop (NFR: resilience). */
export function useHubSocket(onMessage: (msg: HubMessage) => void) {
  const [connected, setConnected] = useState(false)
  const onMessageRef = useRef(onMessage)
  onMessageRef.current = onMessage

  useEffect(() => {
    let ws: WebSocket | null = null
    let reconnectTimer: number | null = null
    let cancelled = false

    function connect() {
      ws = new WebSocket(WS_URL)
      ws.onopen = () => setConnected(true)
      ws.onclose = () => {
        setConnected(false)
        if (!cancelled) reconnectTimer = window.setTimeout(connect, RECONNECT_DELAY_MS)
      }
      ws.onerror = () => ws?.close()
      ws.onmessage = (event) => {
        try {
          onMessageRef.current(JSON.parse(event.data))
        } catch {
          // ignore malformed frames
        }
      }
    }
    connect()

    return () => {
      cancelled = true
      if (reconnectTimer !== null) clearTimeout(reconnectTimer)
      ws?.close()
    }
  }, [])

  return { connected }
}
