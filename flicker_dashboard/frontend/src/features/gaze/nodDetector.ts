// Detects a deliberate vertical head "nod" (a down/up or up/down yes-gesture)
// from a stream of pitch-angle samples, without triggering on ordinary head
// micro-movement, reading text, or drift.
//
// Design: rather than a fixed pitch threshold (which breaks the moment
// someone's neutral head angle differs from whatever the constants assume —
// camera height, posture, webcam angle all vary a lot across laptops), we
// track a slow-moving BASELINE (a heavily-smoothed EMA of pitch) as "neutral
// right now" and look for a fast deviation away from and back past that
// baseline, both within a bounded window. That combination — amplitude *and*
// speed *and* reversal — is what makes it read as an intentional nod rather
// than someone slowly leaning to read a label.
//
// Deliberately direction-agnostic: it records whichever way the pitch first
// moves (up or down — we don't hard-code which sign of `pitchDeg` means
// "chin down", since that depends on the head-pose matrix decomposition
// convention in headPose.ts and isn't worth risking a guess on without a
// live face to test against) and then waits for the *reverse* swing past the
// threshold. A real nod is a down-up (or up-down) oscillation either way, so
// this loses nothing.

export interface NodDetectorConfig {
  /** How far from baseline (degrees) the initiating half of a nod must reach. */
  downThresholdDeg: number
  /** How far back past baseline, in the opposite direction, completes the nod. */
  upThresholdDeg: number
  /** Max time (ms) allowed from the initiating swing to the completing one. Too slow = not a deliberate nod. */
  maxNodDurationMs: number
  /** After a confirmed nod, ignore new nod attempts for this long (ms) so one physical nod can't multi-fire. */
  cooldownMs: number
  /** Smoothing factor for the slow "neutral pitch" baseline. Small = slow to drift, so a fast nod doesn't get absorbed into it before detection fires. */
  baselineAlpha: number
}

// Tuned to be "a real, on-purpose nod" rather than incidental movement.
// These are the first constants to tweak while testing live: if nods aren't
// registering, lower the threshold degrees or raise maxNodDurationMs; if
// it's over-triggering on normal movement, raise the thresholds.
export const DEFAULT_NOD_CONFIG: NodDetectorConfig = {
  downThresholdDeg: 8,
  upThresholdDeg: 6,
  maxNodDurationMs: 900,
  cooldownMs: 1000,
  baselineAlpha: 0.02,
}

type Phase = 'neutral' | 'moved'

/**
 * Stateful nod detector. Feed it every smoothed pitch sample as it arrives;
 * `update()` returns true on exactly the frame a nod is confirmed.
 */
export class NodDetector {
  private baseline: number | null = null
  private phase: Phase = 'neutral'
  private moveSign: 1 | -1 = 1
  private movedAtMs = 0
  private cooldownUntilMs = 0

  private readonly config: NodDetectorConfig
  constructor(config: NodDetectorConfig = DEFAULT_NOD_CONFIG) {
    this.config = config
  }

  update(pitchDeg: number, tMs: number): boolean {
    if (this.baseline === null) {
      this.baseline = pitchDeg
      return false
    }
    // Slow baseline tracks "neutral" so this keeps working regardless of a
    // person's resting head angle / camera placement.
    this.baseline += this.config.baselineAlpha * (pitchDeg - this.baseline)
    const deviation = pitchDeg - this.baseline

    if (tMs < this.cooldownUntilMs) return false

    if (this.phase === 'neutral') {
      if (deviation <= -this.config.downThresholdDeg) {
        this.phase = 'moved'
        this.moveSign = -1
        this.movedAtMs = tMs
      } else if (deviation >= this.config.downThresholdDeg) {
        this.phase = 'moved'
        this.moveSign = 1
        this.movedAtMs = tMs
      }
      return false
    }

    // phase === 'moved': waiting for the reverse swing that completes the nod.
    const elapsed = tMs - this.movedAtMs
    if (elapsed > this.config.maxNodDurationMs) {
      // Took too long — treat as drift, not a nod. Re-arm from here.
      this.phase = 'neutral'
      return false
    }
    const reversed =
      this.moveSign === -1 ? deviation >= this.config.upThresholdDeg : deviation <= -this.config.upThresholdDeg
    if (reversed) {
      this.phase = 'neutral'
      this.cooldownUntilMs = tMs + this.config.cooldownMs
      return true
    }
    return false
  }

  /** True while the initiating half of a nod has been seen and we're waiting on the reversal — useful for a live "keep going" hint in the UI. */
  get isMidNod(): boolean {
    return this.phase === 'moved'
  }

  reset() {
    this.baseline = null
    this.phase = 'neutral'
    this.movedAtMs = 0
    this.cooldownUntilMs = 0
  }
}
