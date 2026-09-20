// Detects a deliberate jaw drop / "ah" as confirm.
//
// The gesture must happen on the tile you are already looking at:
// close mouth (arm) → drop jaw (fire). Head-turn blendshape spikes and
// a mouth that was already open while looking at the camera do not count.

export interface JawOpenConfig {
  openThreshold: number
  closedThreshold: number
  closedHoldMs: number
  warmupMs: number
  cooldownMs: number
}

export const DEFAULT_JAW_CONFIG: JawOpenConfig = {
  openThreshold: 0.52,
  closedThreshold: 0.2,
  closedHoldMs: 320,
  warmupMs: 700,
  cooldownMs: 1600,
}

export class JawOpenDetector {
  private armed = false
  private closedSinceMs: number | null = null
  private firstSampleAt: number | null = null
  private cooldownUntilMs = 0
  private readonly config: JawOpenConfig

  constructor(config: JawOpenConfig = DEFAULT_JAW_CONFIG) {
    this.config = config
  }

  update(jawOpen: number, tMs: number): boolean {
    if (this.firstSampleAt === null) this.firstSampleAt = tMs

    if (jawOpen < this.config.closedThreshold) {
      if (this.closedSinceMs === null) this.closedSinceMs = tMs
      if (tMs - this.closedSinceMs >= this.config.closedHoldMs) this.armed = true
    } else {
      this.closedSinceMs = null
    }

    if (tMs - this.firstSampleAt < this.config.warmupMs) return false
    if (tMs < this.cooldownUntilMs) return false

    if (this.armed && jawOpen >= this.config.openThreshold) {
      this.armed = false
      this.cooldownUntilMs = tMs + this.config.cooldownMs
      return true
    }
    return false
  }

  /** Advance warmup without using mouth pose — e.g. looking at the camera. */
  tick(tMs: number) {
    if (this.firstSampleAt === null) this.firstSampleAt = tMs
  }

  /** Drop a pending arm without restarting warmup — used when gaze leaves a tile. */
  disarm() {
    this.armed = false
    this.closedSinceMs = null
  }

  reset() {
    this.armed = false
    this.closedSinceMs = null
    this.firstSampleAt = null
    this.cooldownUntilMs = 0
  }
}

/** Jaw drop only — smiles/puckers were false-triggering confirm. */
export function mouthOpenness(scores: Record<string, number>): number {
  return scores.jawOpen ?? 0
}
