// Thin client for gaze3d's own WebSocket protocol
// (alboyq/HackMIT_2026 gaze3d/gaze3d/server.py::ws), talking to a copy of that
// server running on remote GPU hardware and reached through a local SSH port
// forward (see the project's ops notes for the exact tunnel command) - NOT
// gaze3d's own bundled page (js/gaze3d.js). This module reimplements just
// enough of that page's connect()/send() request-id bookkeeping to drive the
// same command surface from this app's /iris UI instead.
//
// Protocol recap (gaze3d/gaze3d/server.py, unchanged by the remote-frame work):
//   - JSON text both ways: {cmd, id, ...} in: either {type:'reply', id, ok, ...}
//     (a direct answer to that command) or an unprompted, continuous
//     {type:'frame', ...state} push of Pipeline.snapshot() (server.py::pump()),
//     sent every time a new frame has been processed.
//   - Binary WS messages ONE-WAY into the server: a single JPEG-encoded frame.
//     This is the addition that lets a server started with `--source push`
//     accept frames from here instead of opening a local camera (see
//     gaze3d/gaze3d/pushcam.py). No reply is sent per pushed frame - the next
//     'frame' state message (with an incremented `seq`) is the proof that a
//     specific pushed frame was received, decoded and run through the full
//     ensemble + calibration + filter chain server-side.

export interface Gaze3dHead {
  pitch: number
  yaw: number
  roll: number
  x: number
  y: number
  z: number
  dist: number
  dist_cal: number
  depth_scale: number
  focal: number
}

export interface Gaze3dState {
  type: 'frame'
  seq?: number
  t?: number
  server_t?: number
  face?: boolean
  fps?: number
  gaze?: [number, number] | null
  raw?: [number, number] | null
  fixating?: boolean
  blink?: boolean
  head?: Gaze3dHead
  gaze_norm?: { pitch: number; yaw: number }
  agreement_deg?: number
  reproj?: number
  armed?: boolean
  armed_count?: number
  n_samples?: number
  calibrated?: boolean
  calib_report?: CalibReport
  status?: string
  models?: string[]
  device?: string
  ms?: Record<string, number>
  disk?: { n: number; path: string | null; points: number; at?: number }
  running?: boolean
  [k: string]: unknown
}

export interface CalibCandidate {
  name: string
  cv_err_px: number
  fit_err_px: number
}

export interface CalibReport {
  n_samples: number
  n_points: number
  chosen: string
  cv_err_px: number
  fit_err_px: number
  candidates: CalibCandidate[]
  fit_ms?: number
  saved_to?: string
  [k: string]: unknown
}

interface Pending {
  resolve: (v: Record<string, unknown>) => void
  reject: (e: Error) => void
}

/** Thrown/rejected when a command is sent while the socket isn't open, or the
 * server replies with ok:false. */
export class RemoteGazeError extends Error {}

export class RemoteGazeClient {
  private ws: WebSocket | null = null
  private pending = new Map<number, Pending>()
  private msgId = 0
  private stateListeners = new Set<(s: Gaze3dState) => void>()
  private closeListeners = new Set<() => void>()

  get isOpen(): boolean {
    return this.ws !== null && this.ws.readyState === WebSocket.OPEN
  }

  connect(url: string): Promise<void> {
    return new Promise((resolve, reject) => {
      let settled = false
      const ws = new WebSocket(url)
      this.ws = ws
      ws.onopen = () => {
        settled = true
        resolve()
      }
      ws.onerror = () => {
        if (!settled) {
          settled = true
          reject(new RemoteGazeError(`Could not reach the remote gaze server at ${url}.`))
        }
      }
      ws.onclose = () => {
        for (const p of this.pending.values()) p.reject(new RemoteGazeError('gaze3d socket closed'))
        this.pending.clear()
        for (const cb of this.closeListeners) cb()
        if (!settled) {
          settled = true
          reject(new RemoteGazeError(`gaze3d server closed the connection (${url}).`))
        }
      }
      ws.onmessage = (ev) => {
        if (typeof ev.data !== 'string') return // this protocol never sends binary server->client
        let m: Record<string, unknown>
        try {
          m = JSON.parse(ev.data)
        } catch {
          return
        }
        if (m.type === 'reply') {
          const p = this.pending.get(m.id as number)
          if (!p) return
          this.pending.delete(m.id as number)
          if (m.ok) p.resolve(m)
          else p.reject(new RemoteGazeError((m.error as string) || 'gaze3d command failed'))
        } else if (m.type === 'frame') {
          for (const cb of this.stateListeners) cb(m as Gaze3dState)
        }
      }
    })
  }

  /** Subscribe to every 'frame' state push. Returns an unsubscribe fn. */
  onState(cb: (s: Gaze3dState) => void): () => void {
    this.stateListeners.add(cb)
    return () => this.stateListeners.delete(cb)
  }

  onClose(cb: () => void): () => void {
    this.closeListeners.add(cb)
    return () => this.closeListeners.delete(cb)
  }

  send(cmd: string, extra: Record<string, unknown> = {}): Promise<Record<string, unknown>> {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      return Promise.reject(new RemoteGazeError('gaze3d socket is not open'))
    }
    const id = ++this.msgId
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject })
      this.ws!.send(JSON.stringify({ cmd, id, ...extra }))
    })
  }

  /** Fire-and-forget: push one JPEG-encoded frame. No reply is expected for
   * this - see the module doc for why the next 'frame' state push is the
   * real round-trip proof instead. Returns false (drops the frame) if the
   * socket isn't open, e.g. a frame captured just after a disconnect. */
  pushFrame(jpeg: Blob): boolean {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) return false
    this.ws.send(jpeg)
    return true
  }

  close() {
    this.ws?.close()
    this.ws = null
  }
}
