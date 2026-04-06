export default function SavingsCard({ totalSavings = 0, withoutAxropus = 0, withAxropus = 0, savingsRate = 0 }) {
  return (
    <div
      className="relative rounded-2xl border overflow-hidden p-6"
      style={{
        background: "linear-gradient(135deg, rgba(232,93,58,0.12), rgba(255,140,66,0.06), rgba(124,92,252,0.06))",
        borderColor: "rgba(232,93,58,0.2)",
      }}
    >
      {/* Glow effect */}
      <div
        className="absolute top-0 right-0 w-48 h-48 rounded-full opacity-20 blur-3xl"
        style={{ background: "radial-gradient(circle, #E85D3A, transparent)" }}
      />

      <div className="relative z-10">
        <div className="text-xs font-semibold uppercase tracking-wider mb-2" style={{ color: "#E85D3A" }}>
          Total Agentic Savings
        </div>
        <div
          className="text-4xl font-bold text-white mb-6"
          style={{ fontFamily: "'JetBrains Mono', monospace" }}
        >
          ${totalSavings.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
        </div>

        <div className="flex items-center gap-8">
          <div>
            <div className="text-[10px] uppercase tracking-wider mb-1" style={{ color: "#666" }}>
              Without Axropus
            </div>
            <div
              className="text-sm line-through"
              style={{ color: "#FF5252", fontFamily: "'JetBrains Mono', monospace" }}
            >
              ${withoutAxropus.toLocaleString(undefined, { minimumFractionDigits: 2 })}
            </div>
          </div>
          <div>
            <div className="text-[10px] uppercase tracking-wider mb-1" style={{ color: "#666" }}>
              With Axropus
            </div>
            <div
              className="text-sm font-semibold"
              style={{ color: "#4CAF50", fontFamily: "'JetBrains Mono', monospace" }}
            >
              ${withAxropus.toLocaleString(undefined, { minimumFractionDigits: 2 })}
            </div>
          </div>
          <div>
            <div className="text-[10px] uppercase tracking-wider mb-1" style={{ color: "#666" }}>
              Savings Rate
            </div>
            <div
              className="text-sm font-bold"
              style={{ color: "#FF8C42", fontFamily: "'JetBrains Mono', monospace" }}
            >
              {savingsRate.toFixed(1)}%
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
