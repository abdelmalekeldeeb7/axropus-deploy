export default function AMFMetricsPanel({ metrics = {} }) {
  const rows = [
    { label: "Hit Rate", key: "hit_rate", format: (v) => `${(v * 100).toFixed(1)}%`, color: "#4CAF50" },
    { label: "Tokens Saved", key: "tokens_saved", format: (v) => Number(v).toLocaleString(), color: "#E85D3A" },
    { label: "Prefill Skip", key: "prefill_skip", format: (v) => `${v}x`, color: "#FF8C42" },
    { label: "P50 Latency", key: "p50_latency", format: (v) => `${v}ms`, color: "#00B4D8" },
    { label: "P99 Latency", key: "p99_latency", format: (v) => `${v}ms`, color: "#00B4D8" },
    { label: "Cost Saved/hr", key: "cost_saved_hr", format: (v) => `$${Number(v).toFixed(2)}`, color: "#4CAF50" },
    { label: "VRAM Pool", key: "vram_pool_usage", format: (v) => `${(v * 100).toFixed(0)}%`, color: "#7C5CFC" },
    { label: "Quant Mode", key: "quant_mode", format: (v) => v || "—", color: "#888" },
  ];

  return (
    <div
      className="flex flex-col gap-0 w-[280px] min-h-0 rounded-2xl border overflow-hidden"
      style={{
        background: "rgba(255,255,255,0.03)",
        borderColor: "rgba(255,255,255,0.08)",
      }}
    >
      <div
        className="px-4 py-3 text-xs font-semibold uppercase tracking-wider"
        style={{ color: "#888", borderBottom: "1px solid rgba(255,255,255,0.08)" }}
      >
        AMF Metrics
      </div>
      <div className="flex flex-col">
        {rows.map(({ label, key, format, color }) => (
          <div
            key={key}
            className="flex items-center justify-between px-4 py-2.5"
            style={{ borderBottom: "1px solid rgba(255,255,255,0.04)" }}
          >
            <span className="text-xs" style={{ color: "#888" }}>
              {label}
            </span>
            <span
              className="text-sm font-medium"
              style={{ color, fontFamily: "'JetBrains Mono', monospace" }}
            >
              {metrics[key] != null ? format(metrics[key]) : "—"}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}
