import { useEffect, useState } from 'react'
import type { SceneObject } from '../../hooks/useHubSocket'
import { cropCanvas } from '../../lib/flickerRender'

const PAD_FRAC = 0.18

export interface ObjectCrop {
  id: number
  frequencyHz: number
  canvas: HTMLCanvasElement
}

/** Loads per-object photo crops (no flicker overlay) for the gaze arena. */
export function useObjectCrops(imagePath: string | null, objects: SceneObject[]): ObjectCrop[] {
  const [crops, setCrops] = useState<ObjectCrop[]>([])

  useEffect(() => {
    if (!imagePath || objects.length === 0) {
      setCrops([])
      return
    }
    let cancelled = false
    const baseImg = new Image()
    baseImg.src = `/${imagePath}`

    Promise.all([
      new Promise<void>((resolve, reject) => {
        baseImg.onload = () => resolve()
        baseImg.onerror = () => reject(new Error('Failed to load snapshot image'))
      }),
    ])
      .then(() => {
        if (cancelled) return
        const built: ObjectCrop[] = objects.map((obj) => {
          const [x0, y0, x1, y1] = obj.bbox_px
          const imgW = baseImg.naturalWidth
          const imgH = baseImg.naturalHeight
          const bw = Math.max(1, x1 - x0)
          const bh = Math.max(1, y1 - y0)
          const pad = Math.max(bw, bh) * PAD_FRAC
          const side = Math.min(imgW, imgH, Math.max(bw, bh) + pad * 2)
          let sx = (x0 + x1) / 2 - side / 2
          let sy = (y0 + y1) / 2 - side / 2
          sx = Math.min(Math.max(0, sx), Math.max(0, imgW - side))
          sy = Math.min(Math.max(0, sy), Math.max(0, imgH - side))
          return {
            id: obj.id,
            frequencyHz: obj.frequency_hz,
            canvas: cropCanvas(baseImg, sx, sy, side, side),
          }
        })
        setCrops(built)
      })
      .catch(() => {
        if (!cancelled) setCrops([])
      })

    return () => {
      cancelled = true
    }
  }, [imagePath, objects])

  return crops
}
