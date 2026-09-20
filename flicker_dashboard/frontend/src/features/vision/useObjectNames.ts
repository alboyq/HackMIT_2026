import { useEffect, useState } from 'react'
import type { ObjectCrop } from '../gaze/useObjectCrops'

/** Status of the vision API */
export interface VisionStatus {
  provider: string | null
  error: string | null
}

let cachedStatus: VisionStatus | null = null

export async function checkVisionStatus(): Promise<VisionStatus> {
  if (cachedStatus) return cachedStatus
  try {
    const res = await fetch('/api/ai/status')
    if (!res.ok) throw new Error(`${res.status}`)
    const data = await res.json()
    cachedStatus = { provider: data.vision || null, error: null }
  } catch {
    cachedStatus = { provider: null, error: 'hub offline' }
  }
  return cachedStatus
}

function uniquify(names: string[]): string[] {
  const seen = new Map<string, number>()
  return names.map((n) => {
    const c = (seen.get(n) ?? 0) + 1
    seen.set(n, c)
    return c === 1 ? n : `${n} ${c}`
  })
}

export function displayNameFor(id: number, clipName?: string | null): string {
  if (clipName && clipName.trim()) return clipName
  return `target ${String(id).padStart(2, '0')}`
}

/** Names for each crop via cloud vision API. Falls back to Target 0N if unconfigured. */
export function useObjectNames(crops: ObjectCrop[]): {
  names: Array<string | null>
  ready: boolean
  provider: string | null
} {
  const [names, setNames] = useState<Array<string | null>>([])
  const [ready, setReady] = useState(false)
  const [provider, setProvider] = useState<string | null>(null)

  useEffect(() => {
    if (crops.length === 0) {
      setNames([])
      setReady(false)
      setProvider(null)
      return
    }
    let cancelled = false
    setReady(false)
    ;(async () => {
      // Convert canvases to base64
      const images = crops.map((c) => c.canvas.toDataURL('image/jpeg', 0.85))
      try {
        const res = await fetch('/api/vision/name', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ images }),
        })
        if (cancelled) return
        if (!res.ok) {
          console.warn('[vision] API error', res.status)
          setNames(crops.map((c) => `target ${String(c.id).padStart(2, '0')}`))
          setReady(true)
          setProvider(null)
          return
        }
        const data = await res.json()
        if (cancelled) return
        const rawNames: string[] = data.names || []
        // Fill missing with fallback
        const filled = crops.map((c, i) => rawNames[i] || `target ${String(c.id).padStart(2, '0')}`)
        setNames(uniquify(filled))
        setProvider(data.provider || null)
        setReady(true)
      } catch (err) {
        if (cancelled) return
        console.warn('[vision] fetch failed', err)
        setNames(crops.map((c) => `target ${String(c.id).padStart(2, '0')}`))
        setReady(true)
        setProvider(null)
      }
    })()
    return () => {
      cancelled = true
    }
  }, [crops])

  return { names, ready, provider }
}
