import { useEffect, useState, type RefObject } from 'react'
import { FaceLandmarker, FilesetResolver, type NormalizedLandmark } from '@mediapipe/tasks-vision'
import { decomposeHeadPose, EmaSmoother, type HeadPose } from '../gaze/headPose'
import { mouthOpenness } from '../gaze/jawDetector'
import { combineEyes, type CombinedEyeSample } from './irisLandmarks'

// Parallel to features/gaze/useFaceLandmarker.ts, not a fork of its logic.
// That hook only ever surfaces the facial transformation matrix + jaw
// blendshape score - all /gaze needs. This mode additionally needs the raw
// 478 landmarks for the iris/eye-corner geometry, so rather than growing a
// second, iris-only return field onto a hook three other pages depend on, it
// gets its own detectForVideo loop with its own cached FaceLandmarker
// instance. Camera-permission handling and the model/WASM URLs are the same
// as /gaze on purpose - re-exported below instead of duplicated.

export { describeCameraError, requestCameraStream } from '../gaze/useFaceLandmarker'

const TASKS_VISION_VERSION = '1.0.1'
const WASM_BASE_URL = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${TASKS_VISION_VERSION}/wasm`
const MODEL_URL =
  'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task'

const DETECT_EVERY_N_FRAMES = 2
const POSE_SMOOTHING_ALPHA = 0.55
// Iris offsets are noisier per-frame than head pose - a couple pixels of
// landmark jitter is a much larger fraction of eye width than of the whole
// face - so smooth them a bit harder.
const IRIS_SMOOTHING_ALPHA = 0.4

export type IrisTrackerStatus = 'idle' | 'loading-model' | 'requesting-camera' | 'running' | 'error'

export interface IrisFrame {
  pose: HeadPose
  eye: CombinedEyeSample
  jawOpen: number
  /** Raw 478 landmarks for this frame, for the live overlay. Normalized 0..1
   * image coordinates, as MediaPipe returns them. */
  landmarks: NormalizedLandmark[]
}

export interface UseIrisFaceTrackingResult {
  status: IrisTrackerStatus
  error: string | null
  frame: IrisFrame | null
  faceDetected: boolean
}

let landmarkerPromise: Promise<FaceLandmarker> | null = null

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

/** Lazily creates (once) and caches its own FaceLandmarker instance, GPU
 * delegate first with a CPU fallback - independent of /gaze's cached
 * instance so the two pages never fight over one landmarker's config. */
function getFaceLandmarker(): Promise<FaceLandmarker> {
  if (landmarkerPromise) return landmarkerPromise
  landmarkerPromise = (async () => {
    try {
      return await createLandmarker('GPU')
    } catch (gpuErr) {
      console.warn('[iris] GPU delegate unavailable, falling back to CPU delegate.', gpuErr)
      try {
        return await createLandmarker('CPU')
      } catch (cpuErr) {
        landmarkerPromise = null
        throw cpuErr
      }
    }
  })()
  return landmarkerPromise
}

/**
 * Runs FaceLandmarker on an already-open camera stream and derives head
 * pose + per-eye iris offset + jaw openness every detection. The parent owns
 * getUserMedia (must happen in a click handler) and stopping tracks - same
 * contract as useFaceLandmarker.
 */
export function useIrisFaceTracking(
  active: boolean,
  videoRef: RefObject<HTMLVideoElement | null>,
  stream: MediaStream | null,
): UseIrisFaceTrackingResult {
  const [status, setStatus] = useState<IrisTrackerStatus>('idle')
  const [error, setError] = useState<string | null>(null)
  const [frame, setFrame] = useState<IrisFrame | null>(null)
  const [faceDetected, setFaceDetected] = useState(false)

  useEffect(() => {
    if (!active || !stream) {
      setStatus('idle')
      setFaceDetected(false)
      setFrame(null)
      return
    }

    let cancelled = false
    const rafHolder: { id: number | null } = { id: null }
    const yawSmoother = new EmaSmoother(POSE_SMOOTHING_ALPHA)
    const pitchSmoother = new EmaSmoother(POSE_SMOOTHING_ALPHA)
    const irisXSmoother = new EmaSmoother(IRIS_SMOOTHING_ALPHA)
    const irisYSmoother = new EmaSmoother(IRIS_SMOOTHING_ALPHA)
    let frameCounter = 0
    let lastVideoTime = -1

    async function start() {
      try {
        setError(null)
        const video = videoRef.current
        if (!video) throw new Error('Iris preview <video> element is not mounted yet.')
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
            const landmarks = result.faceLandmarks[0]
            if (matrix && landmarks && landmarks.length >= 478) {
              const rawPose = decomposeHeadPose(matrix.data)
              const pose: HeadPose = {
                yawDeg: yawSmoother.next(rawPose.yawDeg),
                pitchDeg: pitchSmoother.next(rawPose.pitchDeg),
                rollDeg: rawPose.rollDeg,
              }
              const rawEye = combineEyes(landmarks, v.videoWidth, v.videoHeight)
              const eye: CombinedEyeSample = {
                dx: irisXSmoother.next(rawEye.dx),
                dy: irisYSmoother.next(rawEye.dy),
                openness: rawEye.openness,
                blinking: rawEye.blinking,
              }
              const cats = result.faceBlendshapes[0]?.categories ?? []
              const scores: Record<string, number> = {}
              for (const c of cats) scores[c.categoryName] = c.score
              setFrame({ pose, eye, jawOpen: mouthOpenness(scores), landmarks })
              setFaceDetected(true)
            } else {
              setFaceDetected(false)
              setFrame(null)
            }
          }
          rafHolder.id = requestAnimationFrame(tick)
        }
        rafHolder.id = requestAnimationFrame(tick)
      } catch (err) {
        if (cancelled) return
        setStatus('error')
        setError(err instanceof Error ? err.message : 'An unknown error occurred while starting iris tracking.')
      }
    }

    start()

    return () => {
      cancelled = true
      if (rafHolder.id !== null) cancelAnimationFrame(rafHolder.id)
      setFaceDetected(false)
      setFrame(null)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps -- videoRef identity is stable
  }, [active, stream])

  return { status, error, frame, faceDetected }
}
