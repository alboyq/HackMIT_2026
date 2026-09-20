import { useEffect, useRef } from 'react'
import { colorForIndex } from '../lib/colors'

const CORNERS = ['NW', 'NE', 'SW', 'SE'] as const

function CropCanvas({ source, cover }: { source: HTMLCanvasElement; cover: boolean }) {
  const ref = useRef<HTMLCanvasElement>(null)
  useEffect(() => {
    const el = ref.current
    if (!el) return
    el.width = source.width
    el.height = source.height
    el.getContext('2d')!.drawImage(source, 0, 0)
  }, [source])
  return <canvas ref={ref} className={`h-full w-full ${cover ? 'object-cover' : 'object-contain'}`} />
}

export default function SquareObjectTile({
  crop,
  index,
  name,
  looking,
  locked,
  fetching,
  loading,
  fill = false,
  onPick,
  phase = null,
  fetchPct = 0,
}: {
  crop: { id: number; canvas: HTMLCanvasElement } | null
  index: number
  name?: string | null
  looking: boolean
  locked: boolean
  fetching: boolean
  loading: boolean
  fill?: boolean
  onPick?: (index: number) => void
  phase?: string | null
  fetchPct?: number
}) {
  const color = colorForIndex(index)
  const dim = fetching && !locked
  const title = name?.trim() ? name : `TARGET ${String(crop?.id ?? index + 1).padStart(2, '0')}`
  const clickable = Boolean(onPick) && !fetching && Boolean(crop)
  return (
    <div
      role={clickable ? 'button' : undefined}
      tabIndex={clickable ? 0 : undefined}
      onClick={() => {
        if (clickable) onPick?.(index)
      }}
      onKeyDown={(e) => {
        if (!clickable) return
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault()
          onPick?.(index)
        }
      }}
      className={`gaze-square relative overflow-hidden transition-opacity duration-300${fill ? ' fill' : ''}${
        clickable ? ' cursor-pointer' : ''
      }`}
      style={{
        borderColor: locked ? 'var(--accent)' : looking && !fetching ? 'var(--accent)' : 'var(--border)',
        boxShadow: locked ? '0 0 36px var(--accent-glow)' : looking ? '0 0 22px var(--accent-glow)' : undefined,
        opacity: dim ? 0.35 : 1,
      }}
    >
      <span className="gaze-frame-tl" />
      <span className="gaze-frame-tr" />
      <span className="gaze-frame-bl" />
      <span className="gaze-frame-br" />
      <span
        className="pointer-events-none absolute left-3 top-3 z-10 font-mono text-[10px] tracking-[0.22em]"
        style={{ color: locked || looking ? 'var(--accent)' : 'var(--text-dim)' }}
      >
        {CORNERS[index]}
      </span>
      <div className="absolute inset-0 bg-[var(--bg-0)]">
        {crop ? (
          <CropCanvas source={crop.canvas} cover />
        ) : (
          <div className="flex h-full items-center justify-center font-mono text-xs text-[var(--text-dim)]">
            {loading ? 'LOADING…' : 'NO SNAPSHOT'}
          </div>
        )}
      </div>
      <div className="pointer-events-none absolute inset-x-0 bottom-0 bg-gradient-to-t from-black/90 via-black/60 to-transparent px-4 pb-4 pt-12">
        <div className="flex items-end justify-between gap-2">
          <div className="min-w-0 flex-1">
            <div className="font-display text-xl font-semibold leading-tight tracking-wide text-white drop-shadow-lg sm:text-2xl">
              {title}
            </div>
            <div className="mt-1 flex items-center gap-2 font-mono text-[9px] tracking-widest text-white/60">
              <span className="h-1.5 w-1.5 shrink-0 rotate-45" style={{ background: color }} />
              <span className="shrink-0">#{String(crop?.id ?? index + 1).padStart(2, '0')}</span>
            </div>
          </div>
          {(locked || (looking && !fetching)) && (
            <span className="shrink-0 rounded bg-[var(--accent)]/90 px-2 py-1 font-mono text-[10px] font-bold tracking-widest text-black shadow-lg">
              {locked ? phase || 'LOCKED' : 'LOOKING'}
            </span>
          )}
        </div>
        {locked && (
          <div className="mt-3 h-1 overflow-hidden rounded-full bg-white/20">
            <div
              className="h-full bg-[var(--accent)] transition-[width] duration-100"
              style={{ width: `${Math.round(fetchPct * 100)}%` }}
            />
          </div>
        )}
      </div>
    </div>
  )
}
