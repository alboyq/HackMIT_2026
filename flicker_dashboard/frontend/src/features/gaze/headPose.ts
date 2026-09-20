// Pure math: turn a MediaPipe FaceLandmarker "facial transformation matrix"
// into a head yaw/pitch/roll estimate, plus a tiny exponential smoother
// reused for every noisy per-frame scalar in this feature (yaw, pitch, the
// nod baseline, ...).
//
// Background for whoever edits this: FaceLandmarker's
// `facialTransformationMatrixes[i]` is a 4x4 model matrix (rotation +
// translation) that maps MediaPipe's canonical face mesh onto the detected
// face, in the same column-major flattening MediaPipe's own face-effect
// samples feed straight into `THREE.Matrix4.fromArray(...)` — i.e. element
// (row r, col c) lives at `data[c * 4 + r]`.
//
// Rather than decomposing full Euler angles (order-ambiguous and easy to get
// subtly wrong without a live face to test against), we take the simpler,
// robust route: transform the face's local +Z axis (straight out of the
// face) into camera space using the rotation sub-matrix. That gives a single
// "where is the nose pointing" direction vector, and yaw/pitch fall out of
// it with plain trig — no rotation-order assumptions needed.

export interface HeadPose {
  /** Left/right head turn, degrees. Sign convention: see GazeTracker's YAW_SIGN. */
  yawDeg: number
  /** Up/down head tilt, degrees. Sign convention: see GazeTracker's PITCH_SIGN. */
  pitchDeg: number
  /** Head tilt (ear-to-shoulder), degrees. Computed but not currently used for zone/nod logic. */
  rollDeg: number
}

const RAD2DEG = 180 / Math.PI

/**
 * @param matrixData Flattened 4x4 column-major matrix, i.e.
 *   `FaceLandmarkerResult.facialTransformationMatrixes[0].data`.
 */
export function decomposeHeadPose(matrixData: number[]): HeadPose {
  // Column 2 of the rotation sub-matrix = local +Z axis transformed into
  // camera space = the direction the face is pointing.
  const fx = matrixData[8]
  const fy = matrixData[9]
  const fz = matrixData[10]

  const yawDeg = Math.atan2(fx, fz) * RAD2DEG
  const pitchDeg = Math.atan2(-fy, Math.hypot(fx, fz)) * RAD2DEG

  // Roll from how the local +X (right) axis has tilted in the camera's
  // image plane. Approximate (ignores yaw/pitch coupling) but roll isn't
  // load-bearing for this feature — it's exposed only for future use /
  // debugging.
  const rx = matrixData[0]
  const ry = matrixData[1]
  const rollDeg = Math.atan2(ry, rx) * RAD2DEG

  return { yawDeg, pitchDeg, rollDeg }
}

/** Simple exponential-moving-average smoother for one noisy scalar signal. */
export class EmaSmoother {
  private value: number | null = null
  private readonly alpha: number
  constructor(alpha: number) {
    this.alpha = alpha
  }

  next(sample: number): number {
    this.value = this.value === null ? sample : this.value + this.alpha * (sample - this.value)
    return this.value
  }

  reset() {
    this.value = null
  }
}
