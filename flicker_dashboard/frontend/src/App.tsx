import { BrowserRouter, Routes, Route, Link } from 'react-router-dom'
import AppShell from './components/AppShell'
import StimulusPage from './pages/StimulusPage'
import DashboardPage from './pages/DashboardPage'
import GazePage from './pages/GazePage'
import VoicePage from './pages/VoicePage'
import IrisPage from './pages/IrisPage'

function Home() {
  const cards = [
    {
      to: '/stimulus',
      kicker: 'Cortex',
      title: 'Stimulus',
      body: 'Four objects flicker at SSVEP frequencies. Look at one — your visual cortex is the click.',
    },
    {
      to: '/dashboard',
      kicker: 'Operator',
      title: 'Dashboard',
      body: 'Live decoder confidence, pipeline stage, and the scene the wearer sees.',
    },
    {
      to: '/gaze',
      kicker: 'Head',
      title: 'Gaze',
      body: 'Can’t keep a steady stare? Turn toward a square and drop your jaw to lock.',
    },
    {
      to: '/voice',
      kicker: 'Speech',
      title: 'Voice',
      body: 'Can’t move? Say the object. We name it from the photo and speak the delivery back.',
    },
    {
      to: '/iris',
      kicker: 'Eye',
      title: 'Iris',
      body: 'Sharper than a head turn. Calibrate once, then just flick your eyes and drop your jaw to lock.',
    },
  ]

  return (
    <div className="hud-grid flex h-full flex-col items-center justify-center gap-10 px-6">
      <div className="text-center">
        <p className="font-mono text-[11px] tracking-[0.28em] text-[var(--accent)]">BRAIN → VOICE → ROBOT</p>
        <h1 className="font-display mt-3 text-5xl tracking-[0.08em] text-[var(--text-bright)]">SSVEP CONSOLE</h1>
        <p className="mx-auto mt-4 max-w-lg text-sm leading-relaxed text-[var(--text)]">
          Three ways in: visual cortex, head pose, or speech. Same lock. Same robot. Same spoken confirmation.
        </p>
      </div>
      <div className="grid w-full max-w-5xl gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {cards.map((c) => (
          <Link key={c.to} to={c.to} className="panel group p-5 transition hover:border-[var(--accent)]">
            <p className="font-mono text-[10px] tracking-widest text-[var(--accent)]">{c.kicker}</p>
            <h2 className="font-display mt-2 text-xl tracking-wider text-[var(--text-bright)]">{c.title}</h2>
            <p className="mt-2 text-sm leading-relaxed text-[var(--text-dim)] group-hover:text-[var(--text)]">{c.body}</p>
          </Link>
        ))}
      </div>
    </div>
  )
}

export default function App() {
  return (
    <BrowserRouter>
      <AppShell>
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/stimulus" element={<StimulusPage />} />
          <Route path="/dashboard" element={<DashboardPage />} />
          <Route path="/gaze" element={<GazePage />} />
          <Route path="/voice" element={<VoicePage />} />
          <Route path="/iris" element={<IrisPage />} />
        </Routes>
      </AppShell>
    </BrowserRouter>
  )
}
