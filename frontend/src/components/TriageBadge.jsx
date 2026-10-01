const LEVELS = {
  self_care: { label: 'Self-care', cls: 'bg-green-100 text-green-800 ring-green-600/30', dot: 'bg-green-600' },
  gp_appointment: { label: 'See a GP', cls: 'bg-blue-100 text-blue-800 ring-blue-600/30', dot: 'bg-blue-600' },
  urgent_care: { label: 'Urgent care', cls: 'bg-amber-100 text-amber-900 ring-amber-600/30', dot: 'bg-amber-500' },
  emergency: { label: 'Emergency', cls: 'bg-red-100 text-red-800 ring-red-600/30', dot: 'bg-red-600' },
}

export default function TriageBadge({ level }) {
  const cfg = LEVELS[level]
  if (!cfg) return null
  return (
    <span className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-0.5 text-xs font-semibold ring-1 ring-inset ${cfg.cls}`}>
      <span className={`h-1.5 w-1.5 rounded-full ${cfg.dot}`} />
      {cfg.label}
    </span>
  )
}
