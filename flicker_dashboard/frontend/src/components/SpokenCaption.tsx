import { useEffect, useState } from 'react'
import { subscribeSpeech } from '../lib/speech'

export default function SpokenCaption({ className = '' }: { className?: string }) {
  const [line, setLine] = useState<string | null>(null)
  useEffect(() => subscribeSpeech(setLine), [])
  if (!line) return null
  return (
    <div className={`pointer-events-none absolute inset-x-0 bottom-3 z-30 flex justify-center px-4 ${className}`}>
      <p className="max-w-[42rem] rounded-full border border-[var(--accent)]/40 bg-black/75 px-5 py-2 text-center font-mono text-[12px] leading-snug tracking-wide text-[var(--text-bright)] shadow-[0_0_24px_var(--accent-glow)]">
        {line}
      </p>
    </div>
  )
}
