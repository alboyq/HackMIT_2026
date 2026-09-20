// Per-person calibration: fits a small linear map from (head pose + iris
// offset) to normalized screen position.
//
// The reference project (alboyq/HackMIT_2026 gaze3d) gets its 1.3 deg
// accuracy from a much heavier pipeline: a GPU ensemble of pretrained
// appearance-based gaze networks, a full PnP head pose, a per-eye 3D eyeball
// model, and a ray intersected with a physically-measured screen plane,
// refined by 9 static + 10 head-motion calibration trials. Standing that up
// (its own ~2.7GB-weight FastAPI service) is explicitly out of scope here.
//
// What's kept is the *concept*, scaled to fit a lightweight, in-browser demo:
// head pose gives the coarse direction (as /gaze already does), the iris
// offset refines it within that, and a short per-person calibration fits
// away whatever the crude geometry gets wrong - camera position, face shape,
// resting eye position, etc. With only ~5 calibration points, a low-order
// (plain linear, ridge-regularized) fit is the honest choice: anything higher
// order would just memorize noise.

export interface GazeFeatures {
  yawDeg: number
  pitchDeg: number
  irisDx: number
  irisDy: number
}

/** 5 targets: four corners plus center. Enough points to fit a 5-parameter
 * linear model (bias + yaw + pitch + irisDx + irisDy) without being
 * under-determined, while staying well short of gaze3d's 9+10-point protocol. */
export const CALIBRATION_POINTS: ReadonlyArray<{ nx: number; ny: number; label: string }> = [
  { nx: -0.82, ny: -0.78, label: 'TOP LEFT' },
  { nx: 0.82, ny: -0.78, label: 'TOP RIGHT' },
  { nx: 0, ny: 0, label: 'CENTER' },
  { nx: -0.82, ny: 0.78, label: 'BOTTOM LEFT' },
  { nx: 0.82, ny: 0.78, label: 'BOTTOM RIGHT' },
]

const FEATURE_DIM = 5
const RIDGE_LAMBDA = 0.35
// Degree scale used only to bring yaw/pitch into roughly the same numeric
// range as the ~[-1,1] iris offsets, for a better-conditioned fit - not a
// physical constant.
const POSE_SCALE_DEG = 30

function featureVector(f: GazeFeatures): number[] {
  return [1, f.yawDeg / POSE_SCALE_DEG, f.pitchDeg / POSE_SCALE_DEG, f.irisDx, f.irisDy]
}

function dot(a: number[], b: number[]): number {
  let s = 0
  for (let i = 0; i < a.length; i++) s += a[i] * b[i]
  return s
}

function clamp(v: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, v))
}

/** Gauss-Jordan elimination with partial pivoting. `A` is square; returns
 * null if it's singular (shouldn't happen once the ridge term is added). */
function solveLinearSystem(A: number[][], b: number[]): number[] | null {
  const n = A.length
  const M = A.map((row, i) => [...row, b[i]])
  for (let col = 0; col < n; col++) {
    let pivotRow = col
    for (let r = col + 1; r < n; r++) {
      if (Math.abs(M[r][col]) > Math.abs(M[pivotRow][col])) pivotRow = r
    }
    if (Math.abs(M[pivotRow][col]) < 1e-9) return null
    if (pivotRow !== col) {
      const tmp = M[col]
      M[col] = M[pivotRow]
      M[pivotRow] = tmp
    }
    const pivotVal = M[col][col]
    for (let c = col; c <= n; c++) M[col][c] /= pivotVal
    for (let r = 0; r < n; r++) {
      if (r === col) continue
      const factor = M[r][col]
      if (factor === 0) continue
      for (let c = col; c <= n; c++) M[r][c] -= factor * M[col][c]
    }
  }
  return M.map((row) => row[n])
}

/** Ridge-regularized least squares: minimizes ||Xw - y||^2 + lambda*||w||^2
 * via the normal equations (X^T X + lambda I) w = X^T y. */
function ridgeFit(X: number[][], y: number[]): number[] | null {
  const k = FEATURE_DIM
  const XtX: number[][] = Array.from({ length: k }, () => new Array(k).fill(0))
  const Xty: number[] = new Array(k).fill(0)
  for (let i = 0; i < X.length; i++) {
    for (let a = 0; a < k; a++) {
      Xty[a] += X[i][a] * y[i]
      for (let b = 0; b < k; b++) XtX[a][b] += X[i][a] * X[i][b]
    }
  }
  for (let a = 0; a < k; a++) XtX[a][a] += RIDGE_LAMBDA
  return solveLinearSystem(XtX, Xty)
}

export type CalibrationQuality = 'good' | 'fair' | 'poor'

/** In-memory only (per session, per the task's constraints) - nothing here
 * is persisted server-side or to localStorage. */
export class GazeCalibrator {
  private samplesX: number[][] = []
  private samplesNx: number[] = []
  private samplesNy: number[] = []
  private weightsX: number[] | null = null
  private weightsY: number[] | null = null
  private trainResidual: number | null = null

  addSample(features: GazeFeatures, targetNx: number, targetNy: number) {
    this.samplesX.push(featureVector(features))
    this.samplesNx.push(targetNx)
    this.samplesNy.push(targetNy)
  }

  get sampleCount(): number {
    return this.samplesX.length
  }

  get calibrated(): boolean {
    return this.weightsX !== null && this.weightsY !== null
  }

  /** RMS fit error on the training points themselves, in the same -1..1
   * normalized units as screen position. Not a held-out validation score
   * (gaze3d's own separate Measure phase is that, and is out of scope for a
   * 5-point calibration) - just a rough "did this converge sanely" readout. */
  get residual(): number | null {
    return this.trainResidual
  }

  get quality(): CalibrationQuality | null {
    if (this.trainResidual === null) return null
    if (this.trainResidual < 0.12) return 'good'
    if (this.trainResidual < 0.25) return 'fair'
    return 'poor'
  }

  fit(): boolean {
    if (this.samplesX.length < FEATURE_DIM) return false
    const wx = ridgeFit(this.samplesX, this.samplesNx)
    const wy = ridgeFit(this.samplesX, this.samplesNy)
    if (!wx || !wy) return false
    this.weightsX = wx
    this.weightsY = wy
    let sq = 0
    for (let i = 0; i < this.samplesX.length; i++) {
      const px = dot(wx, this.samplesX[i])
      const py = dot(wy, this.samplesX[i])
      sq += (px - this.samplesNx[i]) ** 2 + (py - this.samplesNy[i]) ** 2
    }
    this.trainResidual = Math.sqrt(sq / this.samplesX.length)
    return true
  }

  predict(features: GazeFeatures): { nx: number; ny: number } | null {
    if (!this.weightsX || !this.weightsY) return null
    const fv = featureVector(features)
    return {
      nx: clamp(dot(this.weightsX, fv), -1.35, 1.35),
      ny: clamp(dot(this.weightsY, fv), -1.35, 1.35),
    }
  }

  reset() {
    this.samplesX = []
    this.samplesNx = []
    this.samplesNy = []
    this.weightsX = null
    this.weightsY = null
    this.trainResidual = null
  }
}
