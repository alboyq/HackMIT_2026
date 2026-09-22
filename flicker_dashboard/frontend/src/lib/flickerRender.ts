// Shared canvas helpers for the real per-object SSVEP stimulus.
//
// Two literature checks (per user request, see chat):
// 1. Checkerboard-pattern stimuli are the established SSVEP convention over
//    flat flicker (Won et al., "Optimizing spatial properties of a new
//    checkerboard-like visual stimulus for user-friendly SSVEP-based BCIs",
//    PubMed 34544060; PMC12349513).
// 2. SSVEP amplitude is maximal in central/foveal vision and falls off
//    sharply in the periphery — so the flicker has to be AT the fixation
//    point (the object itself), not just around it. Contrast can be reduced
//    a lot without losing signal (amplitude only drops sharply below ~0.13
//    contrast), which is the room used here: full-contrast checkerboard in
//    the padding (extra effective stimulus area, per S8), translucent
//    checkerboard directly over the object (stays foveal, stays legible).
// No bounding-box rectangle is drawn — the checkerboard region itself marks
// the target.

/**
 * export_snapshot.py writes masks as plain grayscale L-mode PNGs (opaque
 * everywhere, white=object/black=background) — the common CV convention.
 * Canvas compositing needs real alpha, so this converts the grayscale value
 * into the alpha channel once per mask, producing a proper RGBA stencil.
 */
export function buildAlphaStencil(maskImg: HTMLImageElement): HTMLCanvasElement {
  const w = maskImg.naturalWidth
  const h = maskImg.naturalHeight
  const tmp = document.createElement('canvas')
  tmp.width = w
  tmp.height = h
  const tctx = tmp.getContext('2d')!
  tctx.drawImage(maskImg, 0, 0)
  const imgData = tctx.getImageData(0, 0, w, h)
  const px = imgData.data
  for (let i = 0; i < px.length; i += 4) {
    const gray = px[i] // R channel; R=G=B for a grayscale PNG
    px[i] = 255
    px[i + 1] = 255
    px[i + 2] = 255
    px[i + 3] = gray
  }
  tctx.putImageData(imgData, 0, 0)
  return tmp
}

/** Inverts a stencil's alpha (object-shaped hole -> everywhere-except-object). */
export function invertStencil(stencil: HTMLCanvasElement): HTMLCanvasElement {
  const out = document.createElement('canvas')
  out.width = stencil.width
  out.height = stencil.height
  const ctx = out.getContext('2d')!
  ctx.fillStyle = '#fff'
  ctx.fillRect(0, 0, out.width, out.height)
  ctx.globalCompositeOperation = 'destination-out'
  ctx.drawImage(stencil, 0, 0)
  ctx.globalCompositeOperation = 'source-over'
  return out
}

/** Crops a source canvas/image into a new canvas of the given rect. */
export function cropCanvas(
  source: CanvasImageSource,
  sx: number,
  sy: number,
  sw: number,
  sh: number,
): HTMLCanvasElement {
  const out = document.createElement('canvas')
  out.width = Math.max(1, Math.round(sw))
  out.height = Math.max(1, Math.round(sh))
  out.getContext('2d')!.drawImage(source, sx, sy, sw, sh, 0, 0, out.width, out.height)
  return out
}

/**
 * Builds a contrast-reversing checkerboard pair, both clipped to `stencil`'s
 * alpha (so only the object's own mask shape is affected). Crossfading
 * between the two each frame (phaseA at alpha=L, phaseB at alpha=1-L)
 * produces a real checkerboard flicker that still follows the PRD's
 * presentation-time sine luminance model.
 */
export function buildCheckerboardPair(
  stencil: HTMLCanvasElement,
  cellsAcrossShortSide = 8,
): [HTMLCanvasElement, HTMLCanvasElement] {
  const w = stencil.width
  const h = stencil.height
  const cell = Math.max(4, Math.round(Math.min(w, h) / cellsAcrossShortSide))

  function phase(invert: boolean): HTMLCanvasElement {
    const c = document.createElement('canvas')
    c.width = w
    c.height = h
    const ctx = c.getContext('2d')!
    for (let y = 0; y < h; y += cell) {
      for (let x = 0; x < w; x += cell) {
        const even = (Math.floor(x / cell) + Math.floor(y / cell)) % 2 === 0
        const isWhite = invert ? !even : even
        ctx.fillStyle = isWhite ? '#fff' : '#000'
        ctx.fillRect(x, y, cell, cell)
      }
    }
    // Clip to the mask shape using the stencil's alpha channel.
    ctx.globalCompositeOperation = 'destination-in'
    ctx.drawImage(stencil, 0, 0)
    ctx.globalCompositeOperation = 'source-over'
    return c
  }

  return [phase(false), phase(true)]
}

export function presentationLuminance(frequencyHz: number, tSeconds: number): number {
  return 0.5 * (1 + Math.sin(2 * Math.PI * frequencyHz * tSeconds))
}
