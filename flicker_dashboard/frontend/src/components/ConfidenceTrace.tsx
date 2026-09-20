// Live multi-line trace of decoder confidence over the last few seconds, per
// candidate object. This is the panel's actual "wow" surface: a bar shows a
// number, this shows a number *becoming* a decision — a candidate visibly
// climbing away from the pack (or getting overtaken) as the decoder converges.
// Pure SVG, redrawn on every score tick (~100ms) like an oscilloscope trace,
// so what's on screen is exactly the real `scores` history — nothing
// fabricated or smoothed into a lie.

export interface ConfidenceSeries {
  id: string
  color: string
  values: number[] // 0..1, oldest first
  isLeader: boolean
}

const WIDTH = 400
const PAD = 6

function pathFor(values: number[], height: number) {
  if (values.length === 0) return ''
  const n = values.length
  const usableW = WIDTH - PAD * 2
  const usableH = height - PAD * 2
  return values
    .map((v, i) => {
      const x = PAD + (n === 1 ? usableW : (i / (n - 1)) * usableW)
      const y = PAD + (1 - Math.min(1, Math.max(0, v))) * usableH
      return `${i === 0 ? 'M' : 'L'}${x.toFixed(1)},${y.toFixed(1)}`
    })
    .join(' ')
}

function areaFor(values: number[], height: number) {
  if (values.length === 0) return ''
  const line = pathFor(values, height)
  const usableW = WIDTH - PAD * 2
  const baseY = height - PAD
  return `${line} L${(PAD + usableW).toFixed(1)},${baseY} L${PAD},${baseY} Z`
}

export default function ConfidenceTrace({
  series,
  height = 108,
}: {
  series: ConfidenceSeries[]
  height?: number
}) {
  const usableH = height - PAD * 2

  return (
    <svg viewBox={`0 0 ${WIDTH} ${height}`} className="block w-full" preserveAspectRatio="none" style={{ height }}>
      {[0.25, 0.5, 0.75].map((f) => (
        <line
          key={f}
          x1={PAD}
          x2={WIDTH - PAD}
          y1={PAD + f * usableH}
          y2={PAD + f * usableH}
          stroke="var(--border)"
          strokeWidth={1}
          strokeDasharray={f === 0.5 ? '3 3' : undefined}
        />
      ))}

      {series
        .filter((s) => s.isLeader)
        .map((s) => (
          <path key={`${s.id}-area`} d={areaFor(s.values, height)} fill={s.color} opacity={0.14} stroke="none" />
        ))}

      {series
        .filter((s) => !s.isLeader)
        .map((s) => (
          <path
            key={s.id}
            d={pathFor(s.values, height)}
            fill="none"
            stroke={s.color}
            strokeWidth={1.5}
            strokeLinecap="round"
            strokeLinejoin="round"
            opacity={0.5}
          />
        ))}

      {series
        .filter((s) => s.isLeader)
        .map((s) => (
          <path
            key={s.id}
            d={pathFor(s.values, height)}
            fill="none"
            stroke={s.color}
            strokeWidth={2.5}
            strokeLinecap="round"
            strokeLinejoin="round"
            style={{ filter: `drop-shadow(0 0 5px ${s.color})` }}
          />
        ))}

      {series
        .filter((s) => s.isLeader && s.values.length > 0)
        .map((s) => {
          const last = s.values[s.values.length - 1] ?? 0
          const x = WIDTH - PAD
          const y = PAD + (1 - Math.min(1, Math.max(0, last))) * usableH
          return (
            <circle
              key={`${s.id}-dot`}
              cx={x}
              cy={y}
              r={3.5}
              fill={s.color}
              style={{ filter: `drop-shadow(0 0 6px ${s.color})` }}
            />
          )
        })}
    </svg>
  )
}
