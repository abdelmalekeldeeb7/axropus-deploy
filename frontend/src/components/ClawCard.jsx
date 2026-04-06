const STATUS_STYLES = {
  active: { bg: "rgba(76,175,80,0.15)", color: "#4CAF50", label: "Active" },
  draft: { bg: "rgba(255,255,255,0.08)", color: "#888", label: "Draft" },
  paused: { bg: "rgba(234,179,8,0.15)", color: "#EAB308", label: "Paused" },
};

const CHANNEL_ICONS = {
  slack: "💬",
  api: "🔌",
  web: "🌐",
  discord: "🎮",
  email: "📧",
};

export default function ClawCard({ claw }) {
  const st = STATUS_STYLES[claw.status] || STATUS_STYLES.draft;

  return (
    <div
      className="flex flex-col rounded-2xl border p-5 transition-all duration-200"
      style={{
        background: "rgba(255,255,255,0.03)",
        borderColor: "rgba(255,255,255,0.08)",
      }}
      onMouseEnter={(e) => {
        e.currentTarget.style.transform = "translateY(-2px)";
        e.currentTarget.style.borderColor = "rgba(232,93,58,0.25)";
      }}
      onMouseLeave={(e) => {
        e.currentTarget.style.transform = "translateY(0)";
        e.currentTarget.style.borderColor = "rgba(255,255,255,0.08)";
      }}
    >
      {/* Header */}
      <div className="flex items-start justify-between mb-3">
        <div>
          <h3 className="text-white text-sm font-semibold">{claw.name}</h3>
          <p className="text-xs mt-0.5" style={{ color: "#888" }}>{claw.model}</p>
        </div>
        <span
          className="text-[10px] font-bold uppercase px-2.5 py-1 rounded-lg"
          style={{ background: st.bg, color: st.color }}
        >
          {st.label}
        </span>
      </div>

      {/* Channels */}
      <div className="flex items-center gap-1.5 mb-3">
        {(claw.channels || []).map((ch) => (
          <span
            key={ch}
            className="text-xs px-2 py-0.5 rounded-md"
            style={{ background: "rgba(255,255,255,0.06)", color: "#aaa" }}
            title={ch}
          >
            {CHANNEL_ICONS[ch] || "📡"} {ch}
          </span>
        ))}
      </div>

      {/* Skills */}
      <div className="text-xs mb-4" style={{ color: "#888" }}>
        {claw.skillsCount} skill{claw.skillsCount !== 1 ? "s" : ""} configured
      </div>

      {/* Metrics */}
      <div
        className="flex items-center justify-between pt-3 gap-4"
        style={{ borderTop: "1px solid rgba(255,255,255,0.06)" }}
      >
        <div className="flex flex-col items-center flex-1">
          <span className="text-[10px] uppercase" style={{ color: "#666" }}>Avg Steps</span>
          <span className="text-xs font-semibold text-white" style={{ fontFamily: "'JetBrains Mono', monospace" }}>
            {claw.avgSteps ?? "—"}
          </span>
        </div>
        <div className="flex flex-col items-center flex-1">
          <span className="text-[10px] uppercase" style={{ color: "#666" }}>Reuse</span>
          <span className="text-xs font-semibold" style={{ color: "#4CAF50", fontFamily: "'JetBrains Mono', monospace" }}>
            {claw.amfReuseRate != null ? `${(claw.amfReuseRate * 100).toFixed(0)}%` : "—"}
          </span>
        </div>
        <div className="flex flex-col items-center flex-1">
          <span className="text-[10px] uppercase" style={{ color: "#666" }}>TPS</span>
          <span className="text-xs font-semibold" style={{ color: "#00B4D8", fontFamily: "'JetBrains Mono', monospace" }}>
            {claw.throughput ?? "—"}
          </span>
        </div>
      </div>
    </div>
  );
}
