import { LANGUAGES } from '../i18n.js'

export default function LanguageSelect({ value, onChange, className = '' }) {
  return (
    <label className={`flex items-center gap-1.5 ${className}`}>
      <span className="sr-only">Language</span>
      <span aria-hidden="true" className="hidden text-slate-500 sm:inline">
        🌐
      </span>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="rounded-lg border border-slate-300 bg-white px-2 py-1.5 text-sm outline-none focus:border-teal-600 focus:ring-2 focus:ring-teal-600/20"
      >
        {LANGUAGES.map((l) => (
          <option key={l.code} value={l.code}>
            {l.label}
          </option>
        ))}
      </select>
    </label>
  )
}
