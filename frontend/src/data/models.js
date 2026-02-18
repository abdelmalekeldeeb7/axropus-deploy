export const MODEL_OPTIONS = {
  Llama: ["7B", "8B", "13B", "70B", "405B"],
  Mistral: ["7B", "8x7B", "8x22B"],
  Qwen: ["7B", "14B", "72B"],
  Gemma: ["2B", "9B", "27B"],
  DeepSeek: ["7B", "67B", "236B"],
};

export function inferDraftModel(family, size) {
  const f = String(family || "").toLowerCase();
  const s = String(size || "").toUpperCase();
  if (f === "llama") return "Llama 3.2 1B Instruct";
  if (f === "mistral") return "Mistral 7B Instruct";
  if (f === "qwen") return "Qwen2.5 1.5B Instruct";
  if (f === "gemma") return "Gemma 2B Instruct";
  if (f === "deepseek") return s.includes("236") ? "DeepSeek 16B Draft" : "DeepSeek 7B Draft";
  return "Auto-selected draft model";
}
