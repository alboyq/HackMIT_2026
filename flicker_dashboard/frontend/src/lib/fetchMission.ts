const FETCH_MS = 8000
const HOLD_AFTER_MS = 2200

const FETCH_STEPS = [
  { at: 0, label: 'LOCKED' },
  { at: 0.12, label: 'DISPATCH' },
  { at: 0.38, label: 'EN ROUTE' },
  { at: 0.72, label: 'DELIVERING' },
  { at: 1, label: 'IN HAND' },
] as const

export function fetchPhaseLabel(pct: number): string {
  let label: string = FETCH_STEPS[0].label
  for (const step of FETCH_STEPS) {
    if (pct >= step.at) label = step.label
  }
  return label
}

export { FETCH_MS, HOLD_AFTER_MS }
