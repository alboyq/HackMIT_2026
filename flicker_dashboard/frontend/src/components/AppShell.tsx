import { useEffect, useState, type ReactNode } from 'react'
import { Link, useLocation } from 'react-router-dom'
import { motion, AnimatePresence } from 'framer-motion'

function useClock() {
  const [now, setNow] = useState(new Date())
  useEffect(() => {
    const id = setInterval(() => setNow(new Date()), 1000)
    return () => clearInterval(id)
  }, [])
  return now
}

const SIGNAL_BARS = [0.4, 0.85, 0.55, 1, 0.5]

export default function AppShell({ children }: { children: ReactNode }) {
  const location = useLocation()
  const now = useClock()
  const utc = now.toISOString().slice(11, 19)

  const links = [
    { to: '/stimulus', label: 'STIMULUS' },
    { to: '/dashboard', label: 'DASHBOARD' },
    { to: '/gaze', label: 'GAZE' },
    { to: '/iris', label: 'IRIS' },
    { to: '/voice', label: 'VOICE' },
  ]

  return (
    <div className="flex h-full min-w-0 w-full flex-col overflow-x-hidden">
      <nav className="relative flex min-w-0 shrink-0 items-center justify-between border-b border-[var(--border)] bg-[var(--bg-1)]/80 px-5 py-2.5 backdrop-blur-md">
        <Link to="/" className="flex min-w-0 items-center gap-3">
          <div className="signal-bars" aria-hidden="true">
            {SIGNAL_BARS.map((h, i) => (
              <span key={i} className="signal-bar" style={{ height: `${Math.round(h * 16)}px` }} />
            ))}
          </div>
          <span className="font-display text-sm font-semibold tracking-[0.22em] text-[var(--text-bright)]">
            WITHIN <span className="text-[var(--accent)]">REACH</span>
          </span>
        </Link>
        <div className="flex min-w-0 items-center gap-1 font-mono text-[11px] tracking-widest">
          {links.map((l) => {
            const active = location.pathname === l.to
            return (
              <Link key={l.to} to={l.to} className="relative px-3 py-1">
                {active && (
                  <motion.span
                    layoutId="nav-pill"
                    className="absolute inset-0 border border-[var(--accent)] bg-[var(--accent-dim)]"
                    style={{ borderRadius: 8 }}
                    transition={{ type: 'spring', stiffness: 500, damping: 34 }}
                  />
                )}
                <span
                  className={`relative flex items-center gap-1.5 ${
                    active ? 'text-[var(--accent)]' : 'text-[var(--text-dim)] hover:text-[var(--text)]'
                  }`}
                >
                  <span
                    className="h-1.5 w-1.5 rounded-full"
                    style={{
                      background: active ? 'var(--accent)' : 'var(--border-bright)',
                      boxShadow: active ? '0 0 6px var(--accent-glow)' : 'none',
                    }}
                  />
                  {l.label}
                </span>
              </Link>
            )
          })}
          <span className="ml-3 flex items-baseline gap-1.5 border-l border-[var(--border)] pl-3 text-[var(--text-dim)]">
            <span className="font-display text-[9px] tracking-[0.15em]">UTC</span>
            <AnimatePresence mode="popLayout" initial={false}>
              <motion.span
                key={utc}
                initial={{ opacity: 0, y: -4 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: 4 }}
                transition={{ duration: 0.15 }}
                className="inline-block tabular-nums text-[var(--text-bright)]"
              >
                {utc}
              </motion.span>
            </AnimatePresence>
          </span>
        </div>
      </nav>
      <AnimatePresence mode="wait">
        <motion.div
          key={location.pathname}
          className="min-h-0 min-w-0 flex-1 overflow-x-hidden"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.18, ease: 'easeOut' }}
        >
          {children}
        </motion.div>
      </AnimatePresence>
    </div>
  )
}
