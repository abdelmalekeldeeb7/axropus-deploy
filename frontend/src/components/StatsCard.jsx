export default function StatsCard({ label, value, subtitle = "", color = "" }) {
  return (
    <div className="stats-card">
      <div className="stats-label">{label}</div>
      <div className={`stats-value ${color}`.trim()}>{value}</div>
      {subtitle ? <div className="stats-subtitle">{subtitle}</div> : null}
    </div>
  );
}
