/** Map a spoken phrase to a 2×2 slot or a meta command. */

export type VoiceCommand =
  | { kind: 'slot'; index: number }
  | { kind: 'abort' }
  | { kind: 'scene' }
  | { kind: 'repeat' }
  | { kind: 'status' }
  | { kind: 'help' }
  | null

export function matchVoiceCommand(utterance: string, names: Array<string | null>): VoiceCommand {
  const t = utterance.toLowerCase().replace(/[^a-z0-9\s]/g, ' ').replace(/\s+/g, ' ').trim()
  if (!t) return null
  if (/\b(abort|cancel|stop|never mind|nevermind|new pick|release)\b/.test(t)) return { kind: 'abort' }
  if (/\b(what do you see|what can you see|what.s (there|in front)|list (the )?(items|objects)|describe (the )?scene)\b/.test(t)) {
    return { kind: 'scene' }
  }
  if (/\b(repeat|say (that|it) again|what did you say)\b/.test(t)) return { kind: 'repeat' }
  if (/\b(status|where is it|what.s happening|progress)\b/.test(t)) return { kind: 'status' }
  if (/\b(help|what can i say|commands)\b/.test(t)) return { kind: 'help' }

  const indexHits: [RegExp, number][] = [
    [/\btarget\s*(1|one|01)\b|\bfirst\b|\btop left\b|\bnorth\s*west\b|\bupper left\b/, 0],
    [/\btarget\s*(2|two|02)\b|\bsecond\b|\btop right\b|\bnorth\s*east\b|\bupper right\b/, 1],
    [/\btarget\s*(3|three|03)\b|\bthird\b|\bbottom left\b|\bsouth\s*west\b|\blower left\b/, 2],
    [/\btarget\s*(4|four|04)\b|\bfourth\b|\bbottom right\b|\bsouth\s*east\b|\blower right\b/, 3],
  ]
  for (const [re, i] of indexHits) {
    if (re.test(t)) return { kind: 'slot', index: i }
  }

  let best: { i: number; score: number } | null = null
  for (let i = 0; i < names.length; i++) {
    const name = names[i]
    if (!name) continue
    const score = nameScore(t, name)
    if (score > 0 && (!best || score > best.score)) best = { i, score }
  }
  return best ? { kind: 'slot', index: best.i } : null
}

function nameScore(utterance: string, name: string): number {
  const n = name.toLowerCase()
  if (utterance.includes(n)) return n.length + 8
  const tokens = n.split(/\s+/).filter((w) => w.length >= 4)
  let hit = 0
  for (const tok of tokens) {
    if (utterance.includes(tok)) hit += tok.length
  }
  if (hit > 0) return hit
  const last = n.split(/\s+/).pop()
  if (last && last.length >= 3 && new RegExp(`\\b(the|a|an|my|that)\\s+${last}\\b`).test(utterance)) {
    return last.length + 2
  }
  return 0
}
