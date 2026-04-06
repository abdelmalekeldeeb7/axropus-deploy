import { useState } from "react";
import Sidebar from "../components/Sidebar";
import ClawCard from "../components/ClawCard";
import SavingsCard from "../components/SavingsCard";

const SAMPLE_CLAWS = [
  {
    id: "claw-1",
    name: "Support Triage Agent",
    model: "Llama 3.3 70B Instruct",
    channels: ["slack", "web"],
    skillsCount: 8,
    avgSteps: 4.2,
    amfReuseRate: 0.89,
    throughput: 342,
    status: "active",
  },
  {
    id: "claw-2",
    name: "Code Review Bot",
    model: "DeepSeek R1",
    channels: ["api", "discord"],
    skillsCount: 12,
    avgSteps: 7.1,
    amfReuseRate: 0.93,
    throughput: 218,
    status: "active",
  },
  {
    id: "claw-3",
    name: "Data Pipeline Orchestrator",
    model: "Qwen 3 72B",
    channels: ["api"],
    skillsCount: 15,
    avgSteps: 11.4,
    amfReuseRate: 0.85,
    throughput: 156,
    status: "active",
  },
  {
    id: "claw-4",
    name: "Customer Onboarding",
    model: "Mistral 8x22B",
    channels: ["web", "email"],
    skillsCount: 6,
    avgSteps: 3.8,
    amfReuseRate: 0.78,
    throughput: 410,
    status: "paused",
  },
  {
    id: "claw-5",
    name: "Security Audit Agent",
    model: "Nemotron 70B",
    channels: ["api"],
    skillsCount: 20,
    avgSteps: 15.2,
    amfReuseRate: 0.91,
    throughput: 98,
    status: "draft",
  },
  {
    id: "claw-6",
    name: "Meeting Summarizer",
    model: "Llama 3.3 8B Instruct",
    channels: ["slack", "web"],
    skillsCount: 3,
    avgSteps: 2.1,
    amfReuseRate: 0.72,
    throughput: 580,
    status: "active",
  },
];

const TABS = ["All", "Active", "Draft", "Paused"];

export default function Claws() {
  const [tab, setTab] = useState("All");
  const [claws] = useState(SAMPLE_CLAWS);

  const filtered = tab === "All" ? claws : claws.filter((c) => c.status === tab.toLowerCase());

  return (
    <div className="flex min-h-screen" style={{ background: "#0A0A0C", color: "#fff" }}>
      <Sidebar />
      <main className="flex-1 ml-[72px] p-8 max-w-[1400px]">
        {/* Header */}
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center gap-3">
            <span className="text-2xl">&#x1F99E;</span>
            <h1 className="text-2xl font-bold" style={{ fontFamily: "'Instrument Sans', sans-serif" }}>
              Claws
            </h1>
          </div>
          <button
            className="px-5 py-2.5 rounded-xl text-sm font-semibold text-white transition-all duration-200 hover:opacity-90"
            style={{ background: "linear-gradient(135deg, #E85D3A, #FF8C42)" }}
          >
            + New Claw
          </button>
        </div>

        {/* Savings Banner */}
        <div className="mb-8">
          <SavingsCard
            totalSavings={12847.32}
            withoutAxropus={34210.50}
            withAxropus={21363.18}
            savingsRate={62.4}
          />
        </div>

        {/* Tabs */}
        <div className="flex gap-2 mb-6">
          {TABS.map((t) => (
            <button
              key={t}
              onClick={() => setTab(t)}
              className="px-4 py-2 rounded-lg text-xs font-semibold transition-all duration-200"
              style={{
                background: tab === t ? "linear-gradient(135deg, #E85D3A, #FF8C42)" : "rgba(255,255,255,0.04)",
                color: tab === t ? "#fff" : "#888",
                border: tab === t ? "none" : "1px solid rgba(255,255,255,0.08)",
              }}
            >
              {t}
              <span className="ml-1.5 text-[10px] opacity-70">
                {t === "All"
                  ? claws.length
                  : claws.filter((c) => c.status === t.toLowerCase()).length}
              </span>
            </button>
          ))}
        </div>

        {/* Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {filtered.map((claw) => (
            <ClawCard key={claw.id} claw={claw} />
          ))}
        </div>

        {filtered.length === 0 && (
          <div className="text-center py-20" style={{ color: "#555" }}>
            No claws match the current filter.
          </div>
        )}
      </main>
    </div>
  );
}
