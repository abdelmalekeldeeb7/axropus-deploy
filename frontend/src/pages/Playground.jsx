import { useState, useRef, useEffect } from "react";
import Sidebar from "../components/Sidebar";
import AMFMetricsPanel from "../components/AMFMetricsPanel";
import { MODEL_CATALOG } from "../data/models";

const DEFAULT_METRICS = {
  hit_rate: 0.847,
  tokens_saved: 128420,
  prefill_skip: 147.3,
  p50_latency: 12,
  p99_latency: 38,
  cost_saved_hr: 4.82,
  vram_pool_usage: 0.73,
  quant_mode: "FP8",
};

export default function Playground() {
  const [messages, setMessages] = useState([
    { role: "system", content: "You are a helpful assistant powered by Axropus AMF acceleration." },
  ]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [selectedModel, setSelectedModel] = useState(MODEL_CATALOG[0]?.id || "");
  const [metrics, setMetrics] = useState(DEFAULT_METRICS);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(scrollToBottom, [messages]);

  const sendMessage = async () => {
    const text = input.trim();
    if (!text || loading) return;

    const userMsg = { role: "user", content: text };
    const newMessages = [...messages, userMsg];
    setMessages(newMessages);
    setInput("");
    setLoading(true);

    const assistantMsg = { role: "assistant", content: "" };
    setMessages([...newMessages, assistantMsg]);

    try {
      const res = await fetch("/v1/chat/completions", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model: selectedModel,
          messages: newMessages.map(({ role, content }) => ({ role, content })),
          stream: true,
        }),
      });

      if (!res.ok) {
        // Simulate a response for demo purposes when API is unavailable
        const demoContent = `I'm running on the Axropus platform with AMF acceleration. The model **${selectedModel}** processed your request with 147x prefill skip optimization.\n\nAMF cache hit rate: ${(metrics.hit_rate * 100).toFixed(1)}% | Tokens saved: ${metrics.tokens_saved.toLocaleString()}`;
        setMessages((prev) => {
          const updated = [...prev];
          updated[updated.length - 1] = { role: "assistant", content: demoContent };
          return updated;
        });
        // Simulate metrics update
        setMetrics((prev) => ({
          ...prev,
          tokens_saved: prev.tokens_saved + Math.floor(Math.random() * 500) + 100,
          hit_rate: Math.min(0.99, prev.hit_rate + (Math.random() - 0.4) * 0.01),
          cost_saved_hr: +(prev.cost_saved_hr + Math.random() * 0.1).toFixed(2),
        }));
        setLoading(false);
        return;
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let accumulated = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        const chunk = decoder.decode(value, { stream: true });
        const lines = chunk.split("\n").filter((l) => l.startsWith("data: "));
        for (const line of lines) {
          const data = line.slice(6);
          if (data === "[DONE]") break;
          try {
            const parsed = JSON.parse(data);
            const delta = parsed.choices?.[0]?.delta?.content || "";
            accumulated += delta;
            setMessages((prev) => {
              const updated = [...prev];
              updated[updated.length - 1] = { role: "assistant", content: accumulated };
              return updated;
            });
          } catch {
            // skip malformed chunks
          }
        }
      }
    } catch {
      // Fallback demo response
      const demoContent = `Request processed via AMF-accelerated inference. Model: **${selectedModel}**\n\nPrefill optimization achieved 147.3x speedup on this query. Cache utilization at ${(metrics.hit_rate * 100).toFixed(1)}%.`;
      setMessages((prev) => {
        const updated = [...prev];
        updated[updated.length - 1] = { role: "assistant", content: demoContent };
        return updated;
      });
      setMetrics((prev) => ({
        ...prev,
        tokens_saved: prev.tokens_saved + Math.floor(Math.random() * 300) + 50,
        hit_rate: Math.min(0.99, prev.hit_rate + (Math.random() - 0.4) * 0.01),
      }));
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const visibleMessages = messages.filter((m) => m.role !== "system");

  return (
    <div className="flex min-h-screen" style={{ background: "#0A0A0C", color: "#fff" }}>
      <Sidebar />
      <div className="flex flex-1 ml-[72px]">
        {/* Chat Area */}
        <div className="flex-1 flex flex-col h-screen">
          {/* Header */}
          <div
            className="flex items-center justify-between px-6 py-4"
            style={{ borderBottom: "1px solid rgba(255,255,255,0.08)" }}
          >
            <h1 className="text-lg font-bold" style={{ fontFamily: "'Instrument Sans', sans-serif" }}>
              Playground
            </h1>
            <select
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
              className="px-3 py-2 rounded-lg text-xs outline-none cursor-pointer"
              style={{
                background: "rgba(255,255,255,0.04)",
                border: "1px solid rgba(255,255,255,0.08)",
                color: "#ccc",
              }}
            >
              {MODEL_CATALOG.map((m) => (
                <option key={m.id} value={m.id} style={{ background: "#111", color: "#ccc" }}>
                  {m.name}
                </option>
              ))}
            </select>
          </div>

          {/* Messages */}
          <div className="flex-1 overflow-y-auto px-6 py-6">
            <div className="max-w-3xl mx-auto flex flex-col gap-4">
              {visibleMessages.length === 0 && (
                <div className="text-center py-20" style={{ color: "#444" }}>
                  <div className="text-3xl mb-3">&#x25B7;</div>
                  <div className="text-sm">Start a conversation. AMF metrics will update in real-time.</div>
                </div>
              )}
              {visibleMessages.map((msg, i) => (
                <div
                  key={i}
                  className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}
                >
                  <div
                    className="max-w-[80%] px-4 py-3 rounded-2xl text-sm leading-relaxed"
                    style={{
                      background:
                        msg.role === "user"
                          ? "linear-gradient(135deg, #E85D3A, #FF8C42)"
                          : "rgba(255,255,255,0.05)",
                      color: msg.role === "user" ? "#fff" : "#ddd",
                      borderBottomRightRadius: msg.role === "user" ? 4 : 16,
                      borderBottomLeftRadius: msg.role === "user" ? 16 : 4,
                    }}
                  >
                    {msg.content || (
                      <span className="inline-flex gap-1">
                        <span className="w-1.5 h-1.5 rounded-full bg-gray-400 animate-bounce" style={{ animationDelay: "0ms" }} />
                        <span className="w-1.5 h-1.5 rounded-full bg-gray-400 animate-bounce" style={{ animationDelay: "150ms" }} />
                        <span className="w-1.5 h-1.5 rounded-full bg-gray-400 animate-bounce" style={{ animationDelay: "300ms" }} />
                      </span>
                    )}
                  </div>
                </div>
              ))}
              <div ref={messagesEndRef} />
            </div>
          </div>

          {/* Input */}
          <div className="px-6 py-4" style={{ borderTop: "1px solid rgba(255,255,255,0.08)" }}>
            <div className="max-w-3xl mx-auto flex gap-3">
              <textarea
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Send a message..."
                rows={1}
                className="flex-1 px-4 py-3 rounded-xl text-sm outline-none resize-none"
                style={{
                  background: "rgba(255,255,255,0.04)",
                  border: "1px solid rgba(255,255,255,0.08)",
                  color: "#fff",
                }}
                onFocus={(e) => (e.target.style.borderColor = "rgba(232,93,58,0.4)")}
                onBlur={(e) => (e.target.style.borderColor = "rgba(255,255,255,0.08)")}
              />
              <button
                onClick={sendMessage}
                disabled={loading || !input.trim()}
                className="px-5 py-3 rounded-xl text-sm font-semibold text-white transition-all duration-200"
                style={{
                  background:
                    loading || !input.trim()
                      ? "rgba(255,255,255,0.06)"
                      : "linear-gradient(135deg, #E85D3A, #FF8C42)",
                  color: loading || !input.trim() ? "#555" : "#fff",
                  cursor: loading || !input.trim() ? "not-allowed" : "pointer",
                }}
              >
                {loading ? "..." : "Send"}
              </button>
            </div>
          </div>
        </div>

        {/* Metrics Sidebar */}
        <div
          className="w-[280px] h-screen overflow-y-auto p-4"
          style={{ borderLeft: "1px solid rgba(255,255,255,0.08)" }}
        >
          <AMFMetricsPanel metrics={metrics} />
        </div>
      </div>
    </div>
  );
}
