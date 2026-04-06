const TAG_COLORS = {
  POPULAR: { bg: "rgba(232,93,58,0.15)", text: "#E85D3A" },
  REASONING: { bg: "rgba(124,92,252,0.15)", text: "#7C5CFC" },
  MOE: { bg: "rgba(0,180,216,0.15)", text: "#00B4D8" },
  FRONTIER: { bg: "rgba(255,140,66,0.15)", text: "#FF8C42" },
  FAST: { bg: "rgba(76,175,80,0.15)", text: "#4CAF50" },
  MULTILINGUAL: { bg: "rgba(0,180,216,0.15)", text: "#00B4D8" },
  EFFICIENT: { bg: "rgba(118,185,0,0.15)", text: "#76B900" },
};

export default function ModelCard({ model, onDeploy }) {
  const ctx = model.contextWindow >= 1000 ? `${(model.contextWindow / 1000).toFixed(0)}K` : model.contextWindow;

  return (
    <div
      className="relative flex flex-col rounded-2xl border p-5 transition-all duration-200 cursor-pointer group"
      style={{
        background: "rgba(255,255,255,0.03)",
        borderColor: "rgba(255,255,255,0.08)",
      }}
      onMouseEnter={(e) => {
        e.currentTarget.style.transform = "translateY(-2px)";
        e.currentTarget.style.borderColor = `${model.familyColor}44`;
        e.currentTarget.style.boxShadow = `0 8px 32px ${model.familyColor}11`;
      }}
      onMouseLeave={(e) => {
        e.currentTarget.style.transform = "translateY(0)";
        e.currentTarget.style.borderColor = "rgba(255,255,255,0.08)";
        e.currentTarget.style.boxShadow = "none";
      }}
    >
      {/* Family badge top-right */}
      <div className="absolute top-4 right-4">
        <span
          className="text-[10px] font-bold uppercase px-2 py-0.5 rounded-md"
          style={{ background: `${model.familyColor}22`, color: model.familyColor }}
        >
          {model.family}
        </span>
      </div>

      {/* Icon */}
      <div
        className="w-9 h-9 rounded-lg flex items-center justify-center text-sm font-bold mb-3"
        style={{
          background: `linear-gradient(135deg, ${model.familyColor}33, ${model.familyColor}11)`,
          color: model.familyColor,
        }}
      >
        {model.family[0]}
      </div>

      {/* Name */}
      <h3 className="text-white text-sm font-semibold mb-1 pr-16 leading-tight">{model.name}</h3>

      {/* Params + Context */}
      <div className="flex items-center gap-3 mt-1 mb-3">
        <span className="text-xs" style={{ color: "#888", fontFamily: "'JetBrains Mono', monospace" }}>
          {model.params}
        </span>
        <span className="text-xs" style={{ color: "#555" }}>|</span>
        <span className="text-xs" style={{ color: "#888", fontFamily: "'JetBrains Mono', monospace" }}>
          {ctx} ctx
        </span>
        <span className="text-xs" style={{ color: "#555" }}>|</span>
        <span className="text-xs" style={{ color: "#888", fontFamily: "'JetBrains Mono', monospace" }}>
          AMF {model.amfTier}
        </span>
      </div>

      {/* Tags */}
      <div className="flex flex-wrap gap-1.5 mb-4">
        {model.tags.map((tag) => {
          const tc = TAG_COLORS[tag] || { bg: "rgba(255,255,255,0.08)", text: "#888" };
          return (
            <span
              key={tag}
              className="text-[10px] font-semibold uppercase px-2 py-0.5 rounded-md"
              style={{ background: tc.bg, color: tc.text }}
            >
              {tag}
            </span>
          );
        })}
        {model.supportsNemoClaw && (
          <span
            className="text-[10px] font-semibold uppercase px-2 py-0.5 rounded-md"
            style={{ background: "rgba(118,185,0,0.15)", color: "#76B900" }}
          >
            NemoClaw
          </span>
        )}
      </div>

      {/* Cost */}
      <div className="flex items-center justify-between mt-auto pt-3" style={{ borderTop: "1px solid rgba(255,255,255,0.06)" }}>
        <div>
          <span className="text-xs" style={{ color: "#666" }}>Cost/1K tokens</span>
          <div className="text-sm font-semibold text-white" style={{ fontFamily: "'JetBrains Mono', monospace" }}>
            ${model.costPer1kTokens.toFixed(4)}
          </div>
        </div>
        <button
          onClick={() => onDeploy?.(model)}
          className="px-4 py-2 text-xs font-semibold rounded-lg text-white transition-all duration-200 hover:opacity-90"
          style={{
            background: "linear-gradient(135deg, #E85D3A, #FF8C42)",
          }}
        >
          Deploy
        </button>
      </div>
    </div>
  );
}
