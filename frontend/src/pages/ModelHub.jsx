import { useState } from "react";
import Sidebar from "../components/Sidebar";
import ModelCard from "../components/ModelCard";
import { MODEL_CATALOG } from "../data/models";

const FAMILIES = ["All", "Llama", "DeepSeek", "NVIDIA", "Qwen", "Mistral", "Gemma"];

const QUANT_OPTIONS = ["FP16", "FP8", "AWQ-4bit", "GPTQ-4bit", "GGUF-Q5"];
const COMPUTE_TARGETS = ["A100 80GB", "H100 80GB", "L40S 48GB", "RTX 4090 24GB", "Auto"];

export default function ModelHub() {
  const [search, setSearch] = useState("");
  const [family, setFamily] = useState("All");
  const [deployModel, setDeployModel] = useState(null);
  const [quantMode, setQuantMode] = useState("");
  const [vram, setVram] = useState("");
  const [computeTarget, setComputeTarget] = useState("Auto");

  const filtered = MODEL_CATALOG.filter((m) => {
    const matchFamily = family === "All" || m.family === family;
    const matchSearch =
      !search ||
      m.name.toLowerCase().includes(search.toLowerCase()) ||
      m.tags.some((t) => t.toLowerCase().includes(search.toLowerCase()));
    return matchFamily && matchSearch;
  });

  const openDeploy = (model) => {
    setDeployModel(model);
    setQuantMode(model.defaultQuant);
    setVram(String(model.minVramGb));
    setComputeTarget("Auto");
  };

  return (
    <div className="flex min-h-screen" style={{ background: "#0A0A0C", color: "#fff" }}>
      <Sidebar />
      <main className="flex-1 ml-[72px] p-8 max-w-[1400px]">
        {/* Banner */}
        <div
          className="rounded-2xl border p-5 mb-8 flex items-center justify-between"
          style={{
            background: "linear-gradient(135deg, rgba(118,185,0,0.08), rgba(232,93,58,0.08))",
            borderColor: "rgba(118,185,0,0.2)",
          }}
        >
          <div>
            <span className="text-sm font-semibold" style={{ color: "#76B900" }}>
              OpenClaw + NemoClaw Compatible
            </span>
            <span className="text-sm ml-2" style={{ color: "#888" }}>
              — Run agents{" "}
              <span style={{ color: "#FF8C42", fontFamily: "'JetBrains Mono', monospace", fontWeight: 700 }}>
                147x
              </span>{" "}
              faster
            </span>
          </div>
          <div
            className="px-3 py-1 rounded-lg text-xs font-semibold"
            style={{ background: "rgba(118,185,0,0.15)", color: "#76B900" }}
          >
            AMF Enabled
          </div>
        </div>

        {/* Header */}
        <div className="flex items-center justify-between mb-6">
          <h1 className="text-2xl font-bold" style={{ fontFamily: "'Instrument Sans', sans-serif" }}>
            Model Hub
          </h1>
          <span className="text-xs" style={{ color: "#666", fontFamily: "'JetBrains Mono', monospace" }}>
            {filtered.length} model{filtered.length !== 1 ? "s" : ""}
          </span>
        </div>

        {/* Search + Filters */}
        <div className="flex flex-col gap-4 mb-8">
          <input
            type="text"
            placeholder="Search models, tags..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="w-full px-4 py-3 rounded-xl text-sm outline-none transition-colors duration-200"
            style={{
              background: "rgba(255,255,255,0.04)",
              border: "1px solid rgba(255,255,255,0.08)",
              color: "#fff",
            }}
            onFocus={(e) => (e.target.style.borderColor = "rgba(232,93,58,0.4)")}
            onBlur={(e) => (e.target.style.borderColor = "rgba(255,255,255,0.08)")}
          />
          <div className="flex flex-wrap gap-2">
            {FAMILIES.map((f) => (
              <button
                key={f}
                onClick={() => setFamily(f)}
                className="px-4 py-2 rounded-lg text-xs font-semibold transition-all duration-200"
                style={{
                  background: family === f ? "linear-gradient(135deg, #E85D3A, #FF8C42)" : "rgba(255,255,255,0.04)",
                  color: family === f ? "#fff" : "#888",
                  border: family === f ? "none" : "1px solid rgba(255,255,255,0.08)",
                }}
              >
                {f}
              </button>
            ))}
          </div>
        </div>

        {/* Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {filtered.map((m) => (
            <ModelCard key={m.id} model={m} onDeploy={openDeploy} />
          ))}
        </div>

        {filtered.length === 0 && (
          <div className="text-center py-20" style={{ color: "#555" }}>
            No models match your search.
          </div>
        )}

        {/* Deploy Modal */}
        {deployModel && (
          <div
            className="fixed inset-0 z-[100] flex items-center justify-center"
            style={{ background: "rgba(0,0,0,0.7)", backdropFilter: "blur(8px)" }}
            onClick={() => setDeployModel(null)}
          >
            <div
              className="rounded-2xl border p-8 w-full max-w-md"
              style={{
                background: "#111114",
                borderColor: "rgba(255,255,255,0.1)",
              }}
              onClick={(e) => e.stopPropagation()}
            >
              <h2 className="text-lg font-bold text-white mb-1">Deploy {deployModel.name}</h2>
              <p className="text-xs mb-6" style={{ color: "#888" }}>
                Configure quantization, VRAM, and compute target
              </p>

              <div className="flex flex-col gap-5">
                {/* Quant Mode */}
                <div>
                  <label className="text-xs font-semibold mb-2 block" style={{ color: "#aaa" }}>
                    Quantization Mode
                  </label>
                  <div className="flex flex-wrap gap-2">
                    {QUANT_OPTIONS.map((q) => (
                      <button
                        key={q}
                        onClick={() => setQuantMode(q)}
                        className="px-3 py-1.5 rounded-lg text-xs font-medium transition-all duration-150"
                        style={{
                          background: quantMode === q ? "rgba(232,93,58,0.2)" : "rgba(255,255,255,0.04)",
                          color: quantMode === q ? "#E85D3A" : "#888",
                          border: quantMode === q ? "1px solid rgba(232,93,58,0.4)" : "1px solid rgba(255,255,255,0.08)",
                        }}
                      >
                        {q}
                      </button>
                    ))}
                  </div>
                </div>

                {/* VRAM */}
                <div>
                  <label className="text-xs font-semibold mb-2 block" style={{ color: "#aaa" }}>
                    VRAM Allocation (GB)
                  </label>
                  <input
                    type="number"
                    value={vram}
                    onChange={(e) => setVram(e.target.value)}
                    className="w-full px-4 py-2.5 rounded-xl text-sm outline-none"
                    style={{
                      background: "rgba(255,255,255,0.04)",
                      border: "1px solid rgba(255,255,255,0.08)",
                      color: "#fff",
                      fontFamily: "'JetBrains Mono', monospace",
                    }}
                  />
                </div>

                {/* Compute Target */}
                <div>
                  <label className="text-xs font-semibold mb-2 block" style={{ color: "#aaa" }}>
                    Compute Target
                  </label>
                  <div className="flex flex-wrap gap-2">
                    {COMPUTE_TARGETS.map((c) => (
                      <button
                        key={c}
                        onClick={() => setComputeTarget(c)}
                        className="px-3 py-1.5 rounded-lg text-xs font-medium transition-all duration-150"
                        style={{
                          background: computeTarget === c ? "rgba(118,185,0,0.2)" : "rgba(255,255,255,0.04)",
                          color: computeTarget === c ? "#76B900" : "#888",
                          border: computeTarget === c ? "1px solid rgba(118,185,0,0.4)" : "1px solid rgba(255,255,255,0.08)",
                        }}
                      >
                        {c}
                      </button>
                    ))}
                  </div>
                </div>
              </div>

              <div className="flex items-center gap-3 mt-8">
                <button
                  onClick={() => setDeployModel(null)}
                  className="flex-1 px-4 py-3 rounded-xl text-sm font-semibold transition-all duration-200"
                  style={{
                    background: "rgba(255,255,255,0.04)",
                    border: "1px solid rgba(255,255,255,0.08)",
                    color: "#888",
                  }}
                >
                  Cancel
                </button>
                <button
                  onClick={() => {
                    // In a real app, trigger deployment
                    setDeployModel(null);
                  }}
                  className="flex-1 px-4 py-3 rounded-xl text-sm font-bold text-white transition-all duration-200 hover:opacity-90"
                  style={{ background: "linear-gradient(135deg, #E85D3A, #FF8C42)" }}
                >
                  Deploy Now
                </button>
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
