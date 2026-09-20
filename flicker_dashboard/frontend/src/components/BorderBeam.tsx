// Adapted from Magic UI's BorderBeam (magicui.design/docs/components/border-beam,
// apps/www/registry/magicui/border-beam.tsx) for framer-motion instead of motion/react.
import { motion, type MotionStyle, type Transition } from 'framer-motion'

interface BorderBeamProps {
  size?: number
  duration?: number
  delay?: number
  colorFrom?: string
  colorTo?: string
  transition?: Transition
  className?: string
  style?: React.CSSProperties
  reverse?: boolean
  initialOffset?: number
  borderWidth?: number
  /** Opacity of the traveling beam itself — handy for a dimmer "idle" variant layered under a brighter "locked" one. */
  opacity?: number
}

export function BorderBeam({
  className,
  size = 60,
  delay = 0,
  duration = 4,
  colorFrom = 'var(--accent)',
  colorTo = 'transparent',
  transition,
  style,
  reverse = false,
  initialOffset = 0,
  borderWidth = 1.5,
  opacity = 1,
}: BorderBeamProps) {
  return (
    <div
      className="pointer-events-none absolute inset-0 rounded-[inherit] border-transparent [mask-clip:padding-box,border-box] [mask-composite:intersect] [mask-image:linear-gradient(transparent,transparent),linear-gradient(#000,#000)]"
      style={{ borderWidth } as React.CSSProperties}
    >
      <motion.div
        className={`absolute aspect-square bg-gradient-to-l from-[var(--beam-from)] via-[var(--beam-to)] to-transparent ${className ?? ''}`}
        style={
          {
            width: size,
            offsetPath: `rect(0 auto auto 0 round ${size}px)`,
            '--beam-from': colorFrom,
            '--beam-to': colorTo,
            opacity,
            ...style,
          } as MotionStyle
        }
        initial={{ offsetDistance: `${initialOffset}%` }}
        animate={{
          offsetDistance: reverse
            ? [`${100 - initialOffset}%`, `${-initialOffset}%`]
            : [`${initialOffset}%`, `${100 + initialOffset}%`],
        }}
        transition={{ repeat: Infinity, ease: 'linear', duration, delay: -delay, ...transition }}
      />
    </div>
  )
}
