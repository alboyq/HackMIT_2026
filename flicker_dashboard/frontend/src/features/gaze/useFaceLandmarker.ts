import { useEffect, useState, type RefObject } from 'react'
import { FaceLandmarker, FilesetResolver } from '@mediapipe/tasks-vision'
import { decomposeHeadPose, EmaSmoother, type HeadPose } from './headPose'
import { mouthOpenness } from './jawDetector'

// Pinned to the installed npm package version (see package.json) so the WASM
// runtime fetched from jsdelivr at runtime always matches the JS bindings
// Vite bundled — an unpinned "@latest" tag risks a runtime/bindings version
// skew if MediaPipe ships a breaking wasm update after this was built.
const TASKS_VISION_VERSION = '1.0.1'
const WASM_BASE_URL = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${TASKS_VISION_VERSION}/wasm`
// Google's standard hosted FaceLandmarker asset (float16 variant — smaller
// download, plenty of precision for coarse head pose). No self-hosting
// needed for a demo; this is the same URL MediaPipe's own docs point at.
const MODEL_URL =
  'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task'

// Run detection every Nth animation frame rather than every one. At a
// typical 30fps camera feed under a 60fps rAF loop this still yields
// ~15-30 pose updates/sec — plenty for coarse zone selection and nod timing
// — while keeping CPU headroom. This rAF loop is entirely independent of the
// dashboard's own flicker-rendering canvas loop (SnapshotFlickerGrid), so
// neither competes with the other for the same frame budget, but it's still
// real per-frame WASM inference work, so we don't run it flat-out.
const DETECT_EVERY_N_FRAMES = 2

// Smoothing on the raw per-frame yaw/pitch estimate. Higher alpha = snappier
// but jitterier; lower = smoother but laggier. 0.35 tracks a real head turn
// within a couple of frames while still killing single-frame landmark noise.
const POSE_SMOOTHING_ALPHA = 0.55

export type GazeStatus = 'idle' | 'loading-model' | 'requesting-camera' | 'running' | 'error'

export interface UseFaceLandmarkerResult {
  status: GazeStatus
  error: string | null
  pose: HeadPose | null
  faceDetected: boolean
  /** 0..1 combined jaw/mouth openness for confirm. */
  jawOpen: number
}

let landmarkerPromise: Promise<FaceLandmarker> | null = null
let landmarkerKey: string | null = null
const LANDMARKER_KEY = 'v2-blendshapes'

async function createLandmarker(delegate: 'GPU' | 'CPU'): Promise<FaceLandmarker> {
  const fileset = await FilesetResolver.forVisionTasks(WASM_BASE_URL)
  return FaceLandmarker.createFromOptions(fileset, {
    baseOptions: { modelAssetPath: MODEL_URL, delegate },
    runningMode: 'VIDEO',
    numFaces: 1,
    outputFaceBlendshapes: true,
    outputFacialTransformationMatrixes: true,
  })
}

/** Lazily creates (once) and caches the FaceLandmarker instance, trying the GPU delegate first and falling back to CPU on unsupported hardware/browsers. */
function getFaceLandmarker(): Promise<FaceLandmarker> {
  if (landmarkerKey !== LANDMARKER_KEY) {
    landmarkerPromise = null
    landmarkerKey = LANDMARKER_KEY
  }
  if (landmarkerPromise) return landmarkerPromise
  landmarkerPromise = (async () => {
    try {
      return await createLandmarker('GPU')
    } catch (gpuErr) {
      console.warn('[gaze] GPU delegate unavailable, falling back to CPU delegate.', gpuErr)
      try {
        return await createLandmarker('CPU')
      } catch (cpuErr) {
        landmarkerPromise = null // let a future retry start fresh instead of permanently caching a rejected promise
        throw cpuErr
      }
    }
  })()
  return landmarkerPromise
}

export function describeCameraError(err: unknown): string {
  if (err instanceof DOMException) {
    switch (err.name) {
      case 'NotAllowedError':
      case 'PermissionDeniedError':
        return 'Camera access was denied. Click the lock icon in the address bar, allow camera, then try again.'
      case 'NotFoundError':
      case 'DevicesNotFoundError':
        return 'No camera was found on this device.'
      case 'NotReadableError':
      case 'TrackStartError':
        return 'The camera is already in use by another application.'
      case 'OverconstrainedError':
        return 'The camera does not support the requested video settings.'
      default:
        return `Camera error: ${err.message || err.name}`
    }
  }
  if (err instanceof Error) return err.message
  return 'An unknown error occurred while starting gaze tracking.'
}

/**
 * Must be called from a click/tap handler so the browser shows a permission
 * prompt. Do NOT call this from an effect after an await — Chrome treats that
 * as a lost user gesture and often never asks.
 */
export async function requestCameraStream(): Promise<MediaStream> {
  return navigator.mediaDevices.getUserMedia({
    video: { facingMode: 'user' },
    audio: false,
  })
}

/**
 * Runs FaceLandmarker on an already-open camera stream. The parent owns
 * getUserMedia (must happen in a click handler) and stopping tracks.
 */
export function useFaceLandmarker(
  active: boolean,
  videoRef: RefObject<HTMLVideoElement | null>,
  stream: MediaStream | null,
): UseFaceLandmarkerResult {
  const [status, setStatus] = useState<GazeStatus>('idle')
  const [error, setError] = useState<string | null>(null)
  const [pose, setPose] = useState<HeadPose | null>(null)
  const [faceDetected, setFaceDetected] = useState(false)

  const [jawOpen, setJawOpen] = useState(0)

  useEffect(() => {
    if (!active || !stream) {
      setStatus('idle')
      setFaceDetected(false)
      setPose(null)
      setJawOpen(0)
      return
    }

    let cancelled = false
    const rafHolder: { id: number | null } = { id: null }
    const yawSmoother = new EmaSmoother(POSE_SMOOTHING_ALPHA)
    const pitchSmoother = new EmaSmoother(POSE_SMOOTHING_ALPHA)
    let frameCounter = 0
    let lastVideoTime = -1

    async function start() {
      try {
        setError(null)
        const video = videoRef.current
        if (!video) throw new Error('Gaze preview <video> element is not mounted yet.')
        video.srcObject = stream
        video.muted = true
        video.playsInline = true
        await video.play()
        if (cancelled) return

        setStatus('loading-model')
        const landmarker = await getFaceLandmarker()
        if (cancelled) return

        setStatus('running')

        function tick() {
          if (cancelled) return
          const v = videoRef.current
          frameCounter++
          if (v && v.readyState >= 2 && frameCounter % DETECT_EVERY_N_FRAMES === 0 && v.currentTime !== lastVideoTime) {
            lastVideoTime = v.currentTime
            const result = landmarker.detectForVideo(v, performance.now())
            const matrix = result.facialTransformationMatrixes[0]
            if (matrix) {
              const raw = decomposeHeadPose(matrix.data)
              setPose({
                yawDeg: yawSmoother.next(raw.yawDeg),
                pitchDeg: pitchSmoother.next(raw.pitchDeg),
                rollDeg: raw.rollDeg,
              })
              const cats = result.faceBlendshapes[0]?.categories ?? []
              const scores: Record<string, number> = {}
              for (const c of cats) scores[c.categoryName] = c.score
              setJawOpen(mouthOpenness(scores))
              setFaceDetected(true)
            } else {
              setFaceDetected(false)
              setJawOpen(0)
            }
          }
          rafHolder.id = requestAnimationFrame(tick)
        }
        rafHolder.id = requestAnimationFrame(tick)
      } catch (err) {
        if (cancelled) return
        setStatus('error')
        setError(describeCameraError(err))
      }
    }

    start()

    return () => {
      cancelled = true
      if (rafHolder.id !== null) cancelAnimationFrame(rafHolder.id)
      setFaceDetected(false)
      setPose(null)
      setJawOpen(0)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps -- videoRef identity is stable
  }, [active, stream])

  return { status, error, pose, faceDetected, jawOpen }
}
