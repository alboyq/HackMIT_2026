import { useCallback, useEffect, useRef, useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import type { SceneObject } from '../hooks/useHubSocket'
import {
  buildAlphaStencil,
  buildCheckerboardPair,
  cropCanvas,
  invertStencil,
  presentationLuminance,
} from '../lib/flickerRender'
import { colorForIndex } from '../lib/colors'
import { BorderBeam } from './BorderBeam'

// The real stimulus (S1/S2/S8): each object gets its own padded tile,
// generously separated from the others so neighbouring flicker targets can't
// visually interfere. Used for both /stimulus (the wearer's actual view) and
// /dashboard's scene view (same treatment, in a smaller size).
//
// Checked against the literature (see chat): SSVEP amplitude is maximal in
// central/foveal vision and falls off sharply in the periphery. An earlier
// version flickered only the tile's background and kept the object fully
// static so it stayed identifiable — but that puts the actual flicker away
// from the fixation point (the object, which is what the wearer looks at),
// weakening the signal this whole product depends on. The fix isn't to move
// the flicker off the object; it's to flicker directly on it at reduced
// contrast — amplitude only drops sharply below roughly 0.13 contrast, so
// there's plenty of room to keep the object recognizable while still driving
// a real response. So: full-contrast checkerboard in the padding (this is
// what S8's "pad small masks" is actually for — more effective stimulus
// area), translucent checkerboard directly over the object (stays at the
// fixation point, stays legible). No bounding-box rectangle is drawn over
// the image — targets are marked by their own silhouette plus HUD framing
// chrome around the tile, never an annotation rectangle on the photo.

const PAD_FRAC = 0.3
const BG_MAX_OPACITY = 1.0
const FG_MAX_OPACITY = 0.28 // lower contrast over the object itself — legibility over stimulus purity, still well above where amplitude actually drops off (~0.13)
const GAP_PX = 20
const LABEL_BAND_PX = 28

interface Props {
  imagePath: string
  objects: SceneObject[]
  running: boolean
  winnerId?: string
  /** Upper bound on tile size in px. Tiles otherwise fill the container. */
  maxTilePx?: number
  /** Vision-identified name per object id (from /api/vision/name). Falls back to "TARGET NN" when absent/unnamed. */
  names?: Record<number, string>
}

interface Tile {
  id: number
  label: string
  frequencyHz: number
  crop: HTMLCanvasElement
  bgCheckerA: HTMLCanvasElement
  bgCheckerB: HTMLCanvasElement
  fgCheckerA: HTMLCanvasElement
  fgCheckerB: HTMLCanvasElement
  w: number
  h: number
  canvasRef: { current: HTMLCanvasElement | null }
}

/**
 * Measures a container and returns its content box size, live. Uses a
 * callback ref (rather than a plain useRef + mount-only effect) because the
 * measured node isn't always present from the component's first render —
 * the grid container below only mounts once tiles exist — so the observer
 * needs to (re)attach whenever the node itself changes, not just once.
 */
function useContainerSize<T extends HTMLElement>() {
  const [node, setNode] = useState<T | null>(null)
  const ref = useCallback((el: T | null) => setNode(el), [])
  const [size, setSize] = useState({ width: 0, height: 0 })

  useEffect(() => {
    if (!node) return
    const observer = new ResizeObserver((entries) => {
      const box = entries[0]?.contentBoxSize?.[0]
      if (box) {
        setSize({ width: box.inlineSize, height: box.blockSize })
      } else {
        setSize({ width: node.clientWidth, height: node.clientHeight })
      }
    })
    observer.observe(node)
    return () => observer.disconnect()
  }, [node])

  return [ref, size] as const
}

// Width-driven layout that hugs content height (no height-stretch-and-center).
// 3 and 4 objects always wrap to 2 columns so the dashboard stays a 2×2 (or
// 2+1) instead of one overflowing horizontal row. Other counts still pick
// the fewest rows that keep tiles above MIN_TILE_PX.
const HARD_FLOOR_PX = 96

function tileSizeForCols(width: number, cols: number, maxTilePx: number) {
  const bounded = (width - GAP_PX * (cols - 1)) / cols
  return Math.max(HARD_FLOOR_PX, Math.min(maxTilePx, bounded > 0 ? bounded : maxTilePx))
}

function computeGridLayout(n: number, containerWidth: number, containerHeight: number, maxTilePx: number) {
  if (n === 0) return { cols: 1, tileSize: maxTilePx }
  const viewportCap = typeof window !== 'undefined' ? window.innerWidth - 48 : Number.POSITIVE_INFINITY
  const fallbackCols = n === 3 || n === 4 ? 2 : Math.max(1, n)
  const rawWidth =
    containerWidth > 0 ? containerWidth : maxTilePx * fallbackCols + GAP_PX * Math.max(0, fallbackCols - 1)
  const width = Math.min(rawWidth, viewportCap)

  const cols = n === 3 || n === 4 ? 2 : n
  const rows = Math.ceil(n / cols)
  const sizeW = tileSizeForCols(width, cols, maxTilePx)
  let tileSize = sizeW
  if (containerHeight > 0) {
    const sizeH = (containerHeight - GAP_PX * (rows - 1) - LABEL_BAND_PX * rows) / rows
    tileSize = Math.max(HARD_FLOOR_PX, Math.min(sizeW, sizeH, maxTilePx))
  }
  return { cols, tileSize }
}

const LOCK_CORNERS = [
  { top: 2, left: 2, rotate: 0 },
  { top: 2, right: 2, rotate: 90 },
  { bottom: 2, right: 2, rotate: 180 },
  { bottom: 2, left: 2, rotate: 270 },
] as const

const IDLE_CORNERS = [
  { top: 4, left: 4, rotate: 0 },
  { top: 4, right: 4, rotate: 90 },
  { bottom: 4, right: 4, rotate: 180 },
  { bottom: 4, left: 4, rotate: 270 },
] as const

export default function SnapshotFlickerGrid({ imagePath, objects, running, winnerId, maxTilePx = 320, names }: Props) {
  const [tiles, setTiles] = useState<Tile[]>([])
  const rafRef = useRef<number | null>(null)
  const startRef = useRef<number | null>(null)
  const [containerRef, containerSize] = useContainerSize<HTMLDivElement>()

  useEffect(() => {
    let cancelled = false
    const baseImg = new Image()
    baseImg.src = `/${imagePath}`

    const maskLoaders = objects.map(
      (o) =>
        new Promise<[SceneObject, HTMLImageElement]>((resolve) => {
          const img = new Image()
          img.onload = () => resolve([o, img])
          img.src = o.mask_url
        }),
    )

    Promise.all([
      new Promise<void>((resolve) => {
        baseImg.onload = () => resolve()
      }),
      ...maskLoaders,
    ]).then((results) => {
      if (cancelled) return
      const built: Tile[] = []
      for (const r of results) {
        if (!Array.isArray(r)) continue
        const [obj, maskImg] = r
        const [x0, y0, x1, y1] = obj.bbox_px
        const w = x1 - x0
        const h = y1 - y0
        const pad = w * PAD_FRAC + h * PAD_FRAC
        const sx = Math.max(0, x0 - pad)
        const sy = Math.max(0, y0 - pad)
        const sw = Math.min(baseImg.naturalWidth - sx, w + pad * 2)
        const sh = Math.min(baseImg.naturalHeight - sy, h + pad * 2)

        const crop = cropCanvas(baseImg, sx, sy, sw, sh)
        const fullStencil = buildAlphaStencil(maskImg)
        const stencilCrop = cropCanvas(fullStencil, sx, sy, sw, sh)
        const bgStencil = invertStencil(stencilCrop)

        const [bgCheckerA, bgCheckerB] = buildCheckerboardPair(bgStencil, 10)
        const [fgCheckerA, fgCheckerB] = buildCheckerboardPair(stencilCrop, 10)

        built.push({
          id: obj.id,
          label: obj.label,
          frequencyHz: obj.frequency_hz,
          crop,
          bgCheckerA,
          bgCheckerB,
          fgCheckerA,
          fgCheckerB,
          w: sw,
          h: sh,
          canvasRef: { current: null },
        })
      }
      setTiles(built)
    })

    return () => {
      cancelled = true
    }
  }, [imagePath, objects])

  useEffect(() => {
    if (tiles.length === 0) return

    function tick(ts: number) {
      if (startRef.current === null) startRef.current = ts
      const t = (ts - startRef.current) / 1000

      for (const tile of tiles) {
        const canvas = tile.canvasRef.current
        if (!canvas) continue
        if (canvas.width !== tile.w) canvas.width = tile.w
        if (canvas.height !== tile.h) canvas.height = tile.h
        const ctx = canvas.getContext('2d')!

        ctx.drawImage(tile.crop, 0, 0)

        if (running) {
          const L = presentationLuminance(tile.frequencyHz, t)

          ctx.globalAlpha = L * BG_MAX_OPACITY
          ctx.drawImage(tile.bgCheckerA, 0, 0)
          ctx.globalAlpha = (1 - L) * BG_MAX_OPACITY
          ctx.drawImage(tile.bgCheckerB, 0, 0)

          ctx.globalAlpha = L * FG_MAX_OPACITY
          ctx.drawImage(tile.fgCheckerA, 0, 0)
          ctx.globalAlpha = (1 - L) * FG_MAX_OPACITY
          ctx.drawImage(tile.fgCheckerB, 0, 0)

          ctx.globalAlpha = 1
        }
      }
      rafRef.current = requestAnimationFrame(tick)
    }

    startRef.current = null
    rafRef.current = requestAnimationFrame(tick)
    return () => {
      if (rafRef.current !== null) cancelAnimationFrame(rafRef.current)
    }
  }, [tiles, running])

  const { cols, tileSize } = computeGridLayout(
    tiles.length,
    containerSize.width,
    containerSize.height,
    maxTilePx,
  )

  if (tiles.length === 0) {
    // Rendered as a sibling of the measured container below, never inside
    // it: on a page whose layout doesn't otherwise pin our width (like
    // /stimulus, which just centers content rather than stretching it), the
    // measured div's own width is only as wide as whatever's inside it. If
    // this placeholder text lived in there, that transient narrow width
    // would get "measured", land in state, and then feed straight into
    // computeGridLayout the instant real tiles show up — permanently
    // wedging the grid at a stale, tiny size (verified: this actually
    // happened, collapsing every tile to a single ~120px-wide column).
    // Keeping the container empty (and thus reporting 0) while loading
    // means the real first layout pass correctly falls back to the
    // optimistic width estimate above (2-col for 3–4 objects) instead of
    // trusting that stale reading.
    return <div className="py-6 font-mono text-xs text-[var(--text-dim)]">LOADING SNAPSHOT…</div>
  }

  return (
    <div ref={containerRef} className="flex h-full min-h-0 min-w-0 w-full items-center justify-center">
      <div
        className="grid min-w-0"
        style={{
          gridTemplateColumns: `repeat(${cols}, ${tileSize}px)`,
          gap: GAP_PX,
        }}
      >
        {tiles.map((tile, i) => {
            const isTarget = winnerId !== undefined && String(tile.id) === winnerId
            const color = colorForIndex(i)
            return (
              <motion.div
                key={tile.id}
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: i * 0.04, duration: 0.22 }}
                className="flex min-w-0 flex-col items-center gap-1.5"
                style={{ width: tileSize }}
              >
                <div
                  className="panel relative"
                  style={{
                    width: tileSize,
                    height: tileSize,
                    borderColor: isTarget ? 'var(--accent)' : 'var(--border)',
                    boxShadow: isTarget ? '0 0 18px var(--accent-glow)' : undefined,
                  }}
                >
                  {isTarget && <BorderBeam size={70} duration={3} colorFrom="var(--accent)" colorTo="transparent" />}

                  <canvas
                    ref={tile.canvasRef}
                    className="absolute inset-2 h-[calc(100%-16px)] w-[calc(100%-16px)] object-contain"
                  />

                  {!isTarget && (
                    <div className="pointer-events-none absolute inset-0">
                      {IDLE_CORNERS.map(({ rotate, ...pos }, ci) => (
                        <svg
                          key={ci}
                          width="10"
                          height="10"
                          viewBox="0 0 10 10"
                          className="absolute opacity-70"
                          style={{ ...pos, transform: `rotate(${rotate}deg)` }}
                        >
                          <path d="M9 1H3a2 2 0 0 0-2 2v6" fill="none" stroke={color} strokeWidth="1.5" />
                        </svg>
                      ))}
                    </div>
                  )}

                  <AnimatePresence>
                    {isTarget && (
                      <>
                        {LOCK_CORNERS.map(({ rotate, ...pos }, ci) => (
                          <motion.svg
                            key={ci}
                            width="18"
                            height="18"
                            viewBox="0 0 18 18"
                            className="absolute"
                            style={{ ...pos, transform: `rotate(${rotate}deg)` }}
                            initial={{ opacity: 0, scale: 1.8 }}
                            animate={{ opacity: 1, scale: 1 }}
                            exit={{ opacity: 0, scale: 1.8 }}
                            transition={{ type: 'spring', stiffness: 320, damping: 20 }}
                          >
                            <path d="M17 1H5a4 4 0 0 0-4 4v12" fill="none" stroke="var(--accent)" strokeWidth="2" />
                          </motion.svg>
                        ))}
                        <motion.div
                          initial={{ opacity: 0 }}
                          animate={{ opacity: 1 }}
                          className="absolute right-1.5 top-1.5 flex items-center gap-1 rounded-sm bg-black/70 px-1.5 py-0.5"
                        >
                          <span className="h-1.5 w-1.5 animate-pulse-glow rounded-full bg-[var(--accent)]" />
                          <span className="font-mono text-[9px] font-bold tracking-widest text-[var(--accent)]">LOCK</span>
                        </motion.div>
                      </>
                    )}
                  </AnimatePresence>
                </div>

                <span className="flex min-w-0 max-w-full items-center gap-1.5 font-mono text-xs font-bold tracking-widest text-[var(--text-bright)]">
                  <span className="h-1.5 w-1.5 shrink-0 rotate-45" style={{ background: isTarget ? 'var(--accent)' : color }} />
                  <span className="min-w-0 truncate">
                    {(names?.[tile.id] || `TARGET ${String(tile.id).padStart(2, '0')}`).toUpperCase()}
                  </span>
                  <span className="shrink-0 font-normal tabular-nums text-[var(--text-dim)]">{tile.frequencyHz.toFixed(2)}Hz</span>
                </span>
              </motion.div>
            )
          })}
        </div>
    </div>
  )
}
