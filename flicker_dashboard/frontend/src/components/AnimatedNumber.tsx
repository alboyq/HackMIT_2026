import { useEffect } from 'react'
import { motion, useSpring, useTransform } from 'framer-motion'

export default function AnimatedNumber({ value, className }: { value: number; className?: string }) {
  const spring = useSpring(value, { stiffness: 120, damping: 20 })
  const rounded = useTransform(spring, (v) => Math.round(v))

  useEffect(() => {
    spring.set(value)
  }, [spring, value])

  return <motion.span className={className}>{rounded}</motion.span>
}
