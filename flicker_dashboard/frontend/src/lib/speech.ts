/** Spoken lock / delivery lines. Uses backend TTS (ElevenLabs/OpenAI) when available,
 * falls back to Web Speech Synthesis. */

let lastText = ''
let lastAt = 0
let speakingUntil = 0
const listeners = new Set<(line: string | null) => void>()
let useBackendTts: boolean | null = null  // null = not checked yet
let audioContext: AudioContext | null = null

function notify(line: string | null) {
  for (const cb of listeners) cb(line)
}

export function subscribeSpeech(cb: (line: string | null) => void): () => void {
  listeners.add(cb)
  return () => {
    listeners.delete(cb)
  }
}

export function lastSpokenLine(): string {
  return lastText
}

export function isSpeechBusy(): boolean {
  if (typeof window === 'undefined') return false
  return Boolean(window.speechSynthesis?.speaking) || Date.now() < speakingUntil
}

export function silence() {
  if (typeof window === 'undefined' || !window.speechSynthesis) return
  window.speechSynthesis.cancel()
  lastText = ''
  lastAt = 0
  speakingUntil = Date.now()
  notify(null)
}

async function checkBackendTts(): Promise<boolean> {
  if (useBackendTts !== null) return useBackendTts
  try {
    const res = await fetch('/api/ai/status')
    const data = await res.json()
    useBackendTts = !!data.tts && data.tts !== 'browser'
  } catch {
    useBackendTts = false
  }
  return useBackendTts
}

async function speakWithBackend(text: string): Promise<boolean> {
  try {
    const res = await fetch('/api/voice/speak', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text })
    })
    if (!res.ok) return false
    
    const audioData = await res.arrayBuffer()
    if (!audioContext) {
      audioContext = new AudioContext()
    }
    const audioBuffer = await audioContext.decodeAudioData(audioData)
    const source = audioContext.createBufferSource()
    source.buffer = audioBuffer
    source.connect(audioContext.destination)
    source.onended = () => {
      speakingUntil = Date.now() + 200
      notify(null)
    }
    source.start()
    return true
  } catch {
    return false
  }
}

function speakWithBrowser(text: string, interrupt: boolean) {
  if (!window.speechSynthesis) return
  if (interrupt !== false) window.speechSynthesis.cancel()
  const u = new SpeechSynthesisUtterance(text)
  u.rate = 1.02
  u.pitch = 1
  u.lang = 'en-US'
  u.onend = () => {
    speakingUntil = Date.now() + 200
    notify(null)
  }
  u.onerror = () => {
    speakingUntil = Date.now() + 100
    notify(null)
  }
  window.speechSynthesis.speak(u)
}

export function speak(text: string, opts?: { interrupt?: boolean }) {
  if (typeof window === 'undefined') return
  const now = Date.now()
  if (!opts?.interrupt && text === lastText && now - lastAt < 6000) return
  lastText = text
  lastAt = now
  notify(text)
  
  // Try backend TTS first (ElevenLabs/OpenAI), fall back to browser
  checkBackendTts().then(async (hasBackend) => {
    if (hasBackend) {
      const ok = await speakWithBackend(text)
      if (ok) return
    }
    // Fallback to browser speech
    speakWithBrowser(text, opts?.interrupt !== false)
  })
}

export function announceLock(objectName: string) {
  const name = objectName.replace(/^target\s+/i, '')
  speak(`You've locked in the ${name}. It is being delivered to you.`, { interrupt: false })
}

export function announceDelivered(objectName: string) {
  const name = objectName.replace(/^target\s+/i, '')
  speak(`The ${name} is in your reach.`)
}

export function announceAbort() {
  speak('Fetch cancelled. You can choose again.')
}

export function announceScene(names: string[]) {
  const clean = names.map((n) => n.replace(/^target\s+/i, '')).filter(Boolean)
  if (clean.length === 0) return
  if (clean.length === 1) {
    speak(`I can see a ${clean[0]}. Say get me the ${clean[0]} to lock it.`)
    return
  }
  const head = clean.slice(0, -1).join(', ')
  const last = clean[clean.length - 1]
  speak(`I can see ${clean.length} items: ${head}, and a ${last}. Say what you want.`)
}

export function announceHelp() {
  speak('Say get me the object, or target one through four. Say what do you see, or abort to cancel.')
}

export function announceStatus(objectName: string | null, phase: string) {
  if (!objectName) {
    speak('Nothing is locked yet. Say an object when you are ready.')
    return
  }
  const name = objectName.replace(/^target\s+/i, '')
  speak(`Currently ${phase.toLowerCase()} the ${name}.`)
}

type Recog = {
  continuous: boolean
  interimResults: boolean
  lang: string
  start: () => void
  stop: () => void
  abort: () => void
  onresult: ((ev: { resultIndex: number; results: ArrayLike<{ isFinal: boolean; 0: { transcript: string } }> }) => void) | null
  onerror: ((ev: { error: string }) => void) | null
  onend: (() => void) | null
}

function getRecognitionCtor(): (new () => Recog) | null {
  const w = window as unknown as { SpeechRecognition?: new () => Recog; webkitSpeechRecognition?: new () => Recog }
  return w.SpeechRecognition ?? w.webkitSpeechRecognition ?? null
}

export function speechRecognitionAvailable(): boolean {
  return getRecognitionCtor() !== null
}

export function createRecognizer(handlers: {
  onTranscript: (text: string, isFinal: boolean) => void
  onError: (message: string) => void
}): { start: () => void; stop: () => void } | null {
  const Ctor = getRecognitionCtor()
  if (!Ctor) return null
  const rec = new Ctor()
  rec.continuous = true
  rec.interimResults = true
  rec.lang = 'en-US'
  let stopped = false
  let restartTimeout: number | null = null
  
  rec.onresult = (ev) => {
    if (isSpeechBusy()) return
    let interim = ''
    let finalText = ''
    for (let i = ev.resultIndex; i < ev.results.length; i++) {
      const piece = ev.results[i][0].transcript
      if (ev.results[i].isFinal) finalText += piece
      else interim += piece
    }
    if (finalText) handlers.onTranscript(finalText.trim(), true)
    // Only show interim if it has real words (3+ chars), ignore noise
    else if (interim.trim().length >= 3) handlers.onTranscript(interim.trim(), false)
  }
  
  rec.onerror = (ev) => {
    // Silently ignore common non-critical errors
    if (ev.error === 'no-speech' || ev.error === 'aborted' || ev.error === 'network') return
    if (ev.error === 'not-allowed') {
      handlers.onError('Microphone permission was denied.')
    }
    // Don't show other errors, just let it restart
  }
  
  rec.onend = () => {
    if (stopped) return
    // Delay restart to prevent rapid cycling
    restartTimeout = window.setTimeout(() => {
      if (stopped) return
      try {
        rec.start()
      } catch {
        // already started or other error, ignore
      }
    }, 300)
  }
  
  return {
    start: () => {
      stopped = false
      rec.start()
    },
    stop: () => {
      stopped = true
      if (restartTimeout) window.clearTimeout(restartTimeout)
      rec.onend = null
      try { rec.stop() } catch { /* ignore */ }
      try { rec.abort() } catch { /* ignore */ }
    },
  }
}

if (typeof document !== 'undefined') {
  document.addEventListener('visibilitychange', () => {
    if (document.hidden) silence()
  })
}
