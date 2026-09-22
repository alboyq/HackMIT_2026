import { useEffect, useRef, useState, type RefObject } from 'react'
import { RemoteGazeClient, type CalibReport, type Gaze3dState } from './remoteGaze'

// Drives the /iris page's gaze estimate from the remote gaze3d ensemble
// instead of computing it in-browser (see calibration.ts's removal note /
// the project handoff notes: the previous 5-point linear-regression
// approximation calibrated poorly and is replaced here with gaze3d's real
// PnP + XGaze-normalization + 4-model-ensemble + geometric-calibration
// pipeline, running on remote GPU hardware and reached through a local SSH
// tunnel). This hook owns: the WebSocket connection, the video->JPEG capture
// loop that feeds it, and turning the server's continuous state push into a
// normalized on-screen gaze point. Calibration itself (arm/disarm/fit) is
// driven by the caller (IrisCalibration.tsx) through the returned command
// helpers, while this hook is already connected and pushing frames.

const GAZE3D_WS_URL = `ws://${window.location.hostname}:8012/ws`  // local server
const CAPTURE_W = 640
const CAPTURE_H = 480
// ~12.5 fps: gaze3d's own ensemble tops out well under that anyway (HANDOFF.md:
// ~8Hz for the full ViT-H ensemble), and a slower push rate keeps the SSH
// tunnel light, per the project's operating constraints on the shared GPU box.
const PUSH_INTERVAL_MS = 30  // ~33 FPS for faster tracking
// Edge gain: gaze underestimates at screen borders, so we amplify distance from center
const EDGE_GAIN = 1.4  // 1.0 = no change, 1.4 = 40% boost at edges
const JPEG_QUALITY = 0.5  // Lower quality for faster transfer
// Assumed panel diagonal for the px-per-cm estimate below - gaze3d.js hits the
// same wall (HANDOFF.md section 6: no browser API reports true display DPI)
// and falls back to an assumed 14.2" diagonal; this only feeds pitch_mm; the
// calibration's own depth-scale term absorbs most of the resulting error.
const ASSUMED_DIAGONAL_IN = 14.0

export type RemoteGazeStatus =
  | 'idle'
  | 'connecting'
  | 'loading-models'
  | 'waiting-for-frames'
  | 'tracking'
  | 'error'

export interface RemoteGazePoint {
  /** Screen/viewport px, straight from the server. */
  xPx: number
  yPx: number
  /** Normalized to roughly -1..1 over the viewport, for the zone/reticle logic
   * the rest of /iris already works in. */
  nx: number
  ny: number
}

export interface UseRemoteIrisGazeResult {
  status: RemoteGazeStatus
  error: string | null
  faceDetected: boolean
  blink: boolean
  fixating: boolean
  gaze: RemoteGazePoint | null
  calibrated: boolean
  calibReport: CalibReport | null
  models: string[] | null
  device: string | null
  agreementDeg: number | null
  gpuMs: number | null
  fps: number | null
  headDeg: { yaw: number; pitch: number } | null
  arm: (point: number, targetPx: [number, number], kind: string) => Promise<void>
  disarm: () => Promise<number>
  fit: () => Promise<CalibReport>
  clear: () => Promise<void>
}

export function useRemoteIrisGaze(
  active: boolean,
  videoRef: RefObject<HTMLVideoElement | null>,
  stream: MediaStream | null,
): UseRemoteIrisGazeResult {
  const [status, setStatus] = useState<RemoteGazeStatus>('idle')
  const [error, setError] = useState<string | null>(null)
  const [state, setState] = useState<Gaze3dState | null>(null)
  const clientRef = useRef<RemoteGazeClient | null>(null)

  useEffect(() => {
    if (!active || !stream) {
      setStatus('idle')
      setState(null)
      return
    }

    let cancelled = false
    const client = new RemoteGazeClient()
    clientRef.current = client
    const canvas = document.createElement('canvas')
    canvas.width = CAPTURE_W
    canvas.height = CAPTURE_H
    const ctx = canvas.getContext('2d')
    let pushTimer: number | null = null
    let capturing = false

    const offState = client.onState((s) => {
      if (cancelled) return
      setState(s)
      if (s.status === 'loading models') setStatus('loading-models')
      else if (s.status === 'waiting for pushed frames') setStatus('waiting-for-frames')
      else if (s.running || s.status === 'tracking') setStatus('tracking')
    })
    const offClose = client.onClose(() => {
      if (cancelled) return
      setStatus('error')
      setError((prev) => prev ?? 'Lost connection to the remote gaze server.')
    })

    function startPushLoop() {
      pushTimer = window.setInterval(() => {
        const video = videoRef.current
        if (capturing || cancelled || !video || !ctx || video.readyState < 2) return
        capturing = true
        ctx.drawImage(video, 0, 0, CAPTURE_W, CAPTURE_H)
        canvas.toBlob(
          (blob) => {
            capturing = false
            if (blob && !cancelled) client.pushFrame(blob)
          },
          'image/jpeg',
          JPEG_QUALITY,
        )
      }, PUSH_INTERVAL_MS)
    }

    async function run() {
      try {
        setStatus('connecting')
        setError(null)
        await client.connect(GAZE3D_WS_URL)
        if (cancelled) return
        // Best-effort screen geometry: treats the browser viewport as the
        // whole physical screen (reasonable for a fullscreen kiosk-style
        // demo - see the module doc) and derives px-per-cm from an assumed
        // panel diagonal, same fallback gaze3d's own page uses. cam_offset_px
        // is deliberately omitted so the server keeps its own built-in
        // default (Pipeline's ScreenGeometry default of (0, 18)) rather than
        // this guessing a value - HANDOFF.md is explicit that a wrong sign
        // here doesn't degrade gracefully, it silently breaks the geometry.
        const devDiagPx = Math.hypot(screen.width * devicePixelRatio, screen.height * devicePixelRatio)
        const pxPerCm = devDiagPx / ASSUMED_DIAGONAL_IN / devicePixelRatio / 2.54
        const pitchMm = 10 / pxPerCm
        await client.send('geometry', { width_px: innerWidth, height_px: innerHeight, pitch_mm: pitchMm })
        if (cancelled) return
        startPushLoop()
        // pushcam.py's PushCamera.start() never blocks waiting for a frame
        // (see its docstring), so this ordering isn't load-bearing - it just
        // avoids a moment of "waiting for pushed frames" status on a fast
        // connection by giving the push loop a couple of frames' head start.
        await new Promise((r) => setTimeout(r, 150))
        if (cancelled) return
        await client.send('start', { width: CAPTURE_W, height: CAPTURE_H })
      } catch (err) {
        if (cancelled) return
        setStatus('error')
        setError(err instanceof Error ? err.message : 'Could not reach the remote gaze server.')
      }
    }
    run()

    return () => {
      cancelled = true
      offState()
      offClose()
      if (pushTimer !== null) window.clearInterval(pushTimer)
      client.send('stop').catch(() => {})
      client.close()
      clientRef.current = null
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps -- videoRef identity is stable
  }, [active, stream])

  const face = !!state?.face
  const p = state?.gaze ?? state?.raw ?? null
  const gaze: RemoteGazePoint | null = (() => {
    if (!face || !p) return null
    // Raw normalized coords (-1 to 1)
    const rawNx = (p[0] / Math.max(1, innerWidth)) * 2 - 1
    const rawNy = (p[1] / Math.max(1, innerHeight)) * 2 - 1
    // Apply edge gain
    const nx = Math.max(-1.35, Math.min(1.35, rawNx * EDGE_GAIN))
    const ny = Math.max(-1.35, Math.min(1.35, rawNy * EDGE_GAIN))
    return { xPx: p[0], yPx: p[1], nx, ny }
  })()

  return {
    status,
    error,
    faceDetected: face,
    blink: !!state?.blink,
    fixating: !!state?.fixating,
    gaze,
    calibrated: !!state?.calibrated,
    calibReport: state?.calib_report ?? null,
    models: state?.models ?? null,
    device: state?.device ?? null,
    agreementDeg: state?.agreement_deg ?? null,
    gpuMs: state?.ms?.gpu ?? null,
    fps: state?.fps ?? null,
    headDeg: state?.head ? { yaw: state.head.yaw, pitch: state.head.pitch } : null,
    arm: async (point, targetPx, kind) => {
      if (!clientRef.current) throw new Error('gaze3d client not connected')
      await clientRef.current.send('arm', { point, target: targetPx, kind })
    },
    disarm: async () => {
      if (!clientRef.current) return 0
      const res = await clientRef.current.send('disarm')
      return (res.n as number) ?? 0
    },
    fit: async () => {
      if (!clientRef.current) throw new Error('gaze3d client not connected')
      const res = await clientRef.current.send('fit')
      return res.report as CalibReport
    },
    clear: async () => {
      await clientRef.current?.send('clear')
    },
  }
}
