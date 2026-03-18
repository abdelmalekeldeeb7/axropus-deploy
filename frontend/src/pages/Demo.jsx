import { useEffect, useMemo, useRef, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import api from "../api";
import "../styles/demo.css";

const CONTEXT_SCALING = [
  { context: "4K", savings_pct: 34.6, confidence: "Measured" },
  { context: "32K", savings_pct: 80.0, confidence: "Estimated" },
  { context: "128K", savings_pct: 95.3, confidence: "Measured" },
  { context: "512K", savings_pct: 98.0, confidence: "Projected" },
  { context: "1M", savings_pct: 99.0, confidence: "Projected" },
];

const CALC_CONTEXT_OPTIONS = [
  { label: "4K", factor: 0.346 },
  { label: "32K", factor: 0.80 },
  { label: "128K", factor: 0.953 },
  { label: "512K", factor: 0.98 },
  { label: "1M+", factor: 0.99 },
];

const COLD_COLOR = "#ef4444";
const WARM_COLOR = "#22c55e";

function fmtMs(value) {
  return `${Number(value || 0).toLocaleString(undefined, { maximumFractionDigits: 1 })} ms`;
}

function fmtPct(value, digits = 1) {
  return `${Number(value || 0).toFixed(digits)}%`;
}

function fmtSeconds(value) {
  return `${Number(value || 0).toFixed(1)}s`;
}

function fmtUSD(value) {
  if (value >= 1_000_000) return `$${(value / 1_000_000).toFixed(2)}M`;
  if (value >= 1_000) return `$${(value / 1_000).toFixed(1)}K`;
  return `$${value.toFixed(0)}`;
}

function interpolateSavings(rows, ratio) {
  if (!rows.length) return 0;
  const sorted = [...rows].sort((a, b) => Number(a.warm_hit_ratio) - Number(b.warm_hit_ratio));
  if (ratio <= Number(sorted[0].warm_hit_ratio)) return Number(sorted[0].savings_pct || 0);
  if (ratio >= Number(sorted[sorted.length - 1].warm_hit_ratio)) return Number(sorted[sorted.length - 1].savings_pct || 0);
  for (let i = 1; i < sorted.length; i += 1) {
    const prev = sorted[i - 1];
    const next = sorted[i];
    const x0 = Number(prev.warm_hit_ratio);
    const x1 = Number(next.warm_hit_ratio);
    if (ratio >= x0 && ratio <= x1) {
      const y0 = Number(prev.savings_pct || 0);
      const y1 = Number(next.savings_pct || 0);
      const t = (ratio - x0) / (x1 - x0);
      return y0 + (y1 - y0) * t;
    }
  }
  return 0;
}

// Animates a number from 0 to target over duration ms
function useCountUp(target, duration = 1400, enabled = true) {
  const [value, setValue] = useState(0);
  useEffect(() => {
    if (!enabled || target === 0) {
      setValue(target);
      return;
    }
    let start = null;
    let raf;
    const step = (ts) => {
      if (!start) start = ts;
      const progress = Math.min((ts - start) / duration, 1);
      // ease-out cubic
      const eased = 1 - Math.pow(1 - progress, 3);
      setValue(target * eased);
      if (progress < 1) raf = requestAnimationFrame(step);
    };
    raf = requestAnimationFrame(step);
    return () => cancelAnimationFrame(raf);
  }, [target, duration, enabled]);
  return value;
}

// Reveals element when it enters the viewport
function useReveal() {
  const ref = useRef(null);
  const [revealed, setRevealed] = useState(false);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setRevealed(true);
          observer.disconnect();
        }
      },
      { threshold: 0.08 }
    );
    observer.observe(el);
    return () => observer.disconnect();
  }, []);
  return [ref, revealed];
}

function BenchmarkTooltip({ active, payload }) {
  if (!active || !payload || !payload.length) return null;
  const row = payload[0].payload;
  return (
    <div className="demo-tooltip">
      <div className="demo-tooltip-title">Run {row.run_index}</div>
      <div>Decision: <strong>{row.decision}</strong></div>
      <div>Skip Ratio: <strong>{Number(row.skip_ratio || 0).toFixed(3)}</strong></div>
      <div>Total: <strong>{fmtMs(row.total_ms)}</strong></div>
      <div>Prefill: <strong>{fmtMs(row.prefill_ms)}</strong></div>
      <div>Decode: <strong>{fmtMs(row.decode_ms)}</strong></div>
    </div>
  );
}

function MetricCard({ label, value, sub, highlight }) {
  const [ref, revealed] = useReveal();
  return (
    <article ref={ref} className={`demo-card metric-card${revealed ? " revealed" : " reveal"}`}>
      <p className="metric-label">{label}</p>
      <p className={`metric-value${highlight ? " metric-highlight" : ""}`}>{value}</p>
      <p className="metric-sub">{sub}</p>
    </article>
  );
}

function Section({ children, className = "" }) {
  const [ref, revealed] = useReveal();
  return (
    <section ref={ref} className={`${className}${revealed ? " revealed" : " reveal"}`}>
      {children}
    </section>
  );
}

export default function Demo() {
  const [summary, setSummary] = useState(null);
  const [runs, setRuns] = useState([]);
  const [mixed, setMixed] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [warmHitRatioPct, setWarmHitRatioPct] = useState(90);
  const [timerProgress, setTimerProgress] = useState(0);
  const [dataReady, setDataReady] = useState(false);

  // GPU Calculator state
  const [calcGpus, setCalcGpus] = useState(100);
  const [calcCostPerHr, setCalcCostPerHr] = useState(3.50);
  const [calcUtilization, setCalcUtilization] = useState(80);

  // Nebius Fleet Calculator state — full fleet (H100+H200+B200+L40S), blended $2.22/hr
  // H100 95K@$2.00 + H200 15K@$2.30 + B200 7K@$5.50 + L40S 8K@$1.82 = 125K GPUs, $202.6M/mo
  const NEBIUS_BLENDED_HR = 2.217; // blended $/GPU/hr across all types
  const [nebiusGpus, setNebiusGpus] = useState(125000);
  const [nebiusPrefillPct, setNebiusPrefillPct] = useState(65);
  const [nebiusContextIdx, setNebiusContextIdx] = useState(1); // 0=32K,1=128K,2=1M
  const NEBIUS_CONTEXTS = [
    { label: "32K",  savings: 0.85 },
    { label: "128K", savings: 0.953 },
    { label: "1M",   savings: 0.993 },
  ];
  const [calcContextIdx, setCalcContextIdx] = useState(2); // default 128K

  useEffect(() => {
    let mounted = true;
    const load = async () => {
      try {
        const [summaryData, runsData, mixedData] = await Promise.all([
          api.getBenchmarkSummary(),
          api.getBenchmarkRuns(),
          api.getBenchmarkMixed(),
        ]);
        if (!mounted) return;
        setSummary(summaryData);
        setRuns(Array.isArray(runsData?.rows) ? runsData.rows : []);
        setMixed(Array.isArray(mixedData?.rows) ? mixedData.rows : []);
        setDataReady(true);
      } catch (err) {
        if (mounted) setError(err.message || "Failed to load benchmark data");
      } finally {
        if (mounted) setLoading(false);
      }
    };
    load();
    return () => { mounted = false; };
  }, []);

  useEffect(() => {
    let raf = null;
    const cycleMs = 6000;
    const start = performance.now();
    const tick = (now) => {
      const elapsed = (now - start) % cycleMs;
      setTimerProgress(elapsed / cycleMs);
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => { if (raf) cancelAnimationFrame(raf); };
  }, []);

  const longSummary = summary?.long || {};
  const shortSummary = summary?.short || {};

  // Nebius fleet savings calculation — all GPU types, blended rate
  const nebiusSavings = useMemo(() => {
    const ctx = NEBIUS_CONTEXTS[nebiusContextIdx];
    const monthlyFleet  = nebiusGpus * NEBIUS_BLENDED_HR * 730;
    const prefillCost   = monthlyFleet * (nebiusPrefillPct / 100);
    const monthlySaved  = prefillCost * ctx.savings;
    const annualSaved   = monthlySaved * 12;
    const paybackMonths = 500_000_000 / monthlySaved;
    return { monthlyFleet, prefillCost, monthlySaved, annualSaved, paybackMonths, ctx };
  }, [nebiusGpus, nebiusPrefillPct, nebiusContextIdx]);

  const warmRows = useMemo(
    () => runs.filter((row) => String(row.decision).toLowerCase() === "hit"),
    [runs]
  );

  const warmPrefillAvg = useMemo(() => {
    if (!warmRows.length) return 0;
    const total = warmRows.reduce((acc, row) => acc + Number(row.prefill_ms || 0), 0);
    return total / warmRows.length;
  }, [warmRows]);

  const coldMs = Number(longSummary.cold_total_ms || 0);
  const warmMs = Number(longSummary.warm_total_ms_avg || 0);
  const coldSeconds = coldMs / 1000;
  const warmSeconds = warmMs / 1000;
  const animatedCold = coldSeconds * timerProgress;
  const animatedWarm = warmSeconds * timerProgress;
  const speedupRatio = warmMs > 0 ? coldMs / warmMs : 0;

  const mixedLongRows = useMemo(
    () => mixed.filter((row) => String(row.workload).toLowerCase().startsWith("long")),
    [mixed]
  );
  const mixedShortRows = useMemo(
    () => mixed.filter((row) => String(row.workload).toLowerCase().startsWith("short")),
    [mixed]
  );

  const hitRatio = warmHitRatioPct / 100;
  const mixedLongSavings = interpolateSavings(mixedLongRows, hitRatio);
  const mixedShortSavings = interpolateSavings(mixedShortRows, hitRatio);

  // GPU Calculator
  const calcFactor = CALC_CONTEXT_OPTIONS[calcContextIdx].factor;
  const baselineMonthly = calcGpus * calcCostPerHr * (calcUtilization / 100) * 730;
  const grossSavings = baselineMonthly * calcFactor;
  const axropusFee = grossSavings * 0.30;
  const netBenefit = grossSavings * 0.70;

  // Animated counter values (only after data is loaded)
  const animE2e = useCountUp(Number(longSummary.e2e_savings_pct || 0), 1400, dataReady);
  const animHitRate = useCountUp(Number(longSummary.hit_rate || 0) * 100, 1400, dataReady);
  const animHits = useCountUp(Number(longSummary.hits || 0), 1200, dataReady);
  const animRuns = Number(longSummary.runs || 0);
  const animShort = useCountUp(Number(shortSummary.e2e_savings_pct || 0), 1400, dataReady);

  return (
    <div className="demo-page">
      <div className="demo-shell">
        <header className="demo-header">
          <div>
            <p className="demo-kicker">Axropus AMF · Live Benchmark Dashboard</p>
            <h1>95.3% Compute Reduction at 128K Context</h1>
            <p className="demo-subtitle">
              817-run H100 validation · 99.88% cache hit rate · 0.0ms prefill on warm runs
            </p>
          </div>
          <div className="demo-badge-group">
            <span className="demo-badge live-badge">
              <span className="live-dot" />
              Live Data
            </span>
            <span className="demo-badge">Public /demo</span>
          </div>
        </header>

        {error ? <div className="demo-error">{error}</div> : null}
        {loading ? <div className="demo-loading">Loading benchmark data…</div> : null}

        {!loading && !error ? (
          <>
            {/* ── Metrics Row ── */}
            <div className="demo-grid metrics-grid">
              <MetricCard
                label="End-to-End Reduction"
                value={`${animE2e.toFixed(1)}%`}
                sub="128K measured benchmark"
                highlight
              />
              <MetricCard
                label="Cache Hit Rate"
                value={`${animHitRate.toFixed(2)}%`}
                sub="Replay workload"
              />
              <MetricCard
                label="Warm Hits"
                value={`${Math.round(animHits)}/${animRuns}`}
                sub={`Cold misses: ${Number(longSummary.misses || 0)}`}
              />
              <MetricCard
                label="Warm Prefill"
                value={fmtMs(warmPrefillAvg)}
                sub="100% prefix elimination on warm runs"
              />
              <MetricCard
                label="Short Context Floor"
                value={`${animShort.toFixed(1)}%`}
                sub="4K measured"
              />
            </div>

            {/* ── Live Benchmark Viewer ── */}
            <Section className="demo-card chart-card">
              <div className="card-head">
                <h2>Live Benchmark Viewer</h2>
                <p>Run 1 cold miss (red) vs runs 2–817 warm hits (green) · 120K prompt / 128 output tokens</p>
              </div>
              <div className="chart-wrap large">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={runs} margin={{ top: 10, right: 14, left: 12, bottom: 8 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
                    <XAxis
                      dataKey="run_index"
                      stroke="#9ca3af"
                      tickFormatter={(v) => (v === 1 || v % 100 === 0 ? String(v) : "")}
                    />
                    <YAxis stroke="#9ca3af" tickFormatter={(v) => `${Math.round(Number(v) / 1000)}k`} />
                    <Tooltip content={<BenchmarkTooltip />} cursor={{ fill: "rgba(255,255,255,0.05)" }} />
                    <ReferenceLine y={warmMs} stroke="#22c55e" strokeDasharray="4 4" label={{ value: "warm avg", fill: "#22c55e", fontSize: 11 }} />
                    <Bar dataKey="total_ms" isAnimationActive={false}>
                      {runs.map((row, idx) => (
                        <Cell
                          key={`bar-${idx}`}
                          fill={String(row.decision).toLowerCase() === "miss" ? COLD_COLOR : WARM_COLOR}
                        />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </Section>

            {/* ── Before / After ── */}
            <Section className="demo-card">
              <div className="card-head">
                <h2>Before / After</h2>
                <p>Cold run latency vs AMF warm replay latency at 128K context</p>
              </div>
              <div className="speedup-banner">
                <span className="speedup-number">{speedupRatio > 0 ? `${speedupRatio.toFixed(0)}×` : "—"}</span>
                <span className="speedup-label">faster with AMF</span>
              </div>
              <div className="compare-grid">
                <div className="compare-card cold">
                  <p className="compare-label">Without AMF (Cold)</p>
                  <p className="compare-time">{fmtSeconds(animatedCold)}</p>
                  <p className="compare-sub">Full: {fmtSeconds(coldSeconds)} · 4+ minutes of prefill</p>
                  <div className="time-bar cold-bar"><span style={{ width: `${timerProgress * 100}%` }} /></div>
                </div>
                <div className="compare-card warm">
                  <p className="compare-label">With AMF (Warm Hit)</p>
                  <p className="compare-time">{fmtSeconds(animatedWarm)}</p>
                  <p className="compare-sub">Full: {fmtSeconds(warmSeconds)} · 0.0ms prefill</p>
                  <div className="time-bar warm-bar"><span style={{ width: `${timerProgress * 100}%` }} /></div>
                </div>
              </div>
            </Section>

            {/* ── Context Scaling + Mixed Workload ── */}
            <div className="demo-grid two-col">
              <Section className="demo-card">
                <div className="card-head">
                  <h2>Context Scaling</h2>
                  <p>Savings increase with context length — longer = more prefill = more saved</p>
                </div>
                <div className="chart-wrap medium">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={CONTEXT_SCALING} margin={{ top: 8, right: 8, left: 8, bottom: 8 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
                      <XAxis dataKey="context" stroke="#9ca3af" />
                      <YAxis stroke="#9ca3af" domain={[0, 100]} tickFormatter={(v) => `${v}%`} />
                      <Tooltip
                        formatter={(v, _, payload) => [`${Number(v).toFixed(1)}%`, payload?.payload?.confidence || ""]}
                      />
                      <Line type="monotone" dataKey="savings_pct" stroke="#60a5fa" strokeWidth={3} dot={{ r: 5, fill: "#60a5fa" }} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
                <div className="legend-row">
                  {CONTEXT_SCALING.map((point) => (
                    <span key={point.context} className={`legend-pill ${point.confidence.toLowerCase()}`}>
                      {point.context}: {fmtPct(point.savings_pct)} ({point.confidence})
                    </span>
                  ))}
                </div>
              </Section>

              <Section className="demo-card">
                <div className="card-head">
                  <h2>Mixed Workload Calculator</h2>
                  <p>Adjust warm-hit ratio and see blended savings</p>
                </div>
                <label className="slider-label" htmlFor="warm-hit-ratio">
                  Warm-Hit Ratio: <strong>{warmHitRatioPct}%</strong>
                </label>
                <input
                  id="warm-hit-ratio"
                  type="range"
                  min="50"
                  max="95"
                  step="1"
                  value={warmHitRatioPct}
                  onChange={(e) => setWarmHitRatioPct(Number(e.target.value))}
                  className="ratio-slider"
                />
                <div className="mixed-grid">
                  <div className="mixed-card long">
                    <p className="mixed-title">Long Context (128K)</p>
                    <p className="mixed-value">{fmtPct(mixedLongSavings)}</p>
                  </div>
                  <div className="mixed-card short">
                    <p className="mixed-title">Short Context (4K)</p>
                    <p className="mixed-value">{fmtPct(mixedShortSavings)}</p>
                  </div>
                </div>
              </Section>
            </div>

            {/* ── GPU Savings Calculator ── */}
            <Section className="demo-card calc-card">
              <div className="card-head">
                <h2>GPU Savings Calculator</h2>
                <p>Estimate your monthly compute savings with AMF · Axropus takes 30% of savings generated</p>
              </div>
              <div className="calc-grid">
                <div className="calc-inputs">
                  <div className="calc-field">
                    <label className="slider-label" htmlFor="calc-gpus">
                      GPU Count: <strong>{calcGpus}</strong>
                    </label>
                    <input
                      id="calc-gpus"
                      type="range"
                      min="1"
                      max="1000"
                      step="1"
                      value={calcGpus}
                      onChange={(e) => setCalcGpus(Number(e.target.value))}
                      className="ratio-slider"
                    />
                  </div>
                  <div className="calc-field">
                    <label className="slider-label" htmlFor="calc-cost">
                      Cost per GPU/hr: <strong>${calcCostPerHr.toFixed(2)}</strong>
                    </label>
                    <input
                      id="calc-cost"
                      type="range"
                      min="1"
                      max="10"
                      step="0.10"
                      value={calcCostPerHr}
                      onChange={(e) => setCalcCostPerHr(Number(e.target.value))}
                      className="ratio-slider"
                    />
                    <p className="calc-hint">H100: ~$3.50/hr · A100: ~$2.50/hr · B200: ~$6.00/hr</p>
                  </div>
                  <div className="calc-field">
                    <label className="slider-label" htmlFor="calc-util">
                      GPU Utilization: <strong>{calcUtilization}%</strong>
                    </label>
                    <input
                      id="calc-util"
                      type="range"
                      min="10"
                      max="100"
                      step="5"
                      value={calcUtilization}
                      onChange={(e) => setCalcUtilization(Number(e.target.value))}
                      className="ratio-slider"
                    />
                  </div>
                  <div className="calc-field">
                    <label className="slider-label">Context Length:</label>
                    <div className="context-pills">
                      {CALC_CONTEXT_OPTIONS.map((opt, i) => (
                        <button
                          key={opt.label}
                          type="button"
                          className={`context-pill${calcContextIdx === i ? " active" : ""}`}
                          onClick={() => setCalcContextIdx(i)}
                        >
                          {opt.label}
                          <span className="context-pill-sub">{(opt.factor * 100).toFixed(0)}% saved</span>
                        </button>
                      ))}
                    </div>
                  </div>
                </div>

                <div className="calc-output">
                  <div className="calc-row baseline">
                    <p className="calc-row-label">Baseline Monthly Cost</p>
                    <p className="calc-row-value">{fmtUSD(baselineMonthly)}</p>
                  </div>
                  <div className="calc-row savings">
                    <p className="calc-row-label">Gross Savings with AMF</p>
                    <p className="calc-row-value green-val">{fmtUSD(grossSavings)}</p>
                    <p className="calc-row-pct">{(calcFactor * 100).toFixed(0)}% reduction</p>
                  </div>
                  <div className="calc-divider" />
                  <div className="calc-row fee">
                    <p className="calc-row-label">Axropus Fee (30% of savings)</p>
                    <p className="calc-row-value dim-val">{fmtUSD(axropusFee)}</p>
                  </div>
                  <div className="calc-row net">
                    <p className="calc-row-label">Your Net Benefit (70%)</p>
                    <p className="calc-row-value green-val large">{fmtUSD(netBenefit)}</p>
                    <p className="calc-row-sub">per month · zero infra changes</p>
                  </div>
                </div>
              </div>
            </Section>

            {/* ── NemoClaw / 1M Token Section ── */}
            <Section className="demo-card nemo-section">
              <div className="nemo-header">
                <div className="nemo-tag">NVIDIA NemoClaw · Agent Workloads</div>
                <h2>The 1M Token Opportunity</h2>
                <p>
                  NemoClaw enterprise agents run at 1M token context — maximum prefix redundancy, maximum AMF impact.
                  Every agent fleet deployment multiplies the savings.
                </p>
              </div>
              <div className="nemo-stats">
                <div className="nemo-stat">
                  <p className="nemo-stat-value">~99%</p>
                  <p className="nemo-stat-label">Compute saved at 1M context</p>
                  <p className="nemo-stat-sub">Projected (extrapolated from 128K curve)</p>
                </div>
                <div className="nemo-stat">
                  <p className="nemo-stat-value">20×</p>
                  <p className="nemo-stat-label">GPU capacity multiplier</p>
                  <p className="nemo-stat-sub">On agent prefill-heavy workloads</p>
                </div>
                <div className="nemo-stat">
                  <p className="nemo-stat-value">$0</p>
                  <p className="nemo-stat-label">Infrastructure changes required</p>
                  <p className="nemo-stat-sub">Drop-in CUDA layer · no model changes</p>
                </div>
                <div className="nemo-stat">
                  <p className="nemo-stat-value">Any</p>
                  <p className="nemo-stat-label">Model size · 8B to 1T+</p>
                  <p className="nemo-stat-sub">Larger models = larger absolute savings</p>
                </div>
              </div>
              <div className="nemo-flow">
                <div className="nemo-flow-step">
                  <div className="flow-icon">📥</div>
                  <p>Agent Request<br /><span>1M token context</span></p>
                </div>
                <div className="nemo-flow-arrow">→</div>
                <div className="nemo-flow-step highlight-step">
                  <div className="flow-icon">⚡</div>
                  <p>AMF Cache Hit<br /><span>0.0ms prefill</span></p>
                </div>
                <div className="nemo-flow-arrow">→</div>
                <div className="nemo-flow-step">
                  <div className="flow-icon">🚀</div>
                  <p>Response<br /><span>Decode only</span></p>
                </div>
              </div>
            </Section>

            {/* ── Nebius Fleet Calculator ── */}
            <Section className="demo-card nebius-calc-section">
              <div className="nebius-calc-header">
                <div className="nebius-tag">Nebius AI Cloud · Real Numbers · Public Data</div>
                <h2>What Korith Saves at Fleet Scale</h2>
                <p>
                  Nebius runs 125,000 GPUs (H100 · H200 · B200 · L40S) — $202.6M/month.
                  NemoClaw agents run at 32K–128K+ token context.
                  Prefill dominates. Korith eliminates it.
                </p>
              </div>

              <div className="nebius-sliders">
                <div className="nebius-slider-row">
                  <label>GPU Fleet Size (all types): <strong>{nebiusGpus.toLocaleString()} GPUs</strong></label>
                  <input type="range" min="10000" max="300000" step="5000"
                    value={nebiusGpus} onChange={e => setNebiusGpus(Number(e.target.value))} />
                  <div className="nebius-slider-hints">
                    <span>10K</span><span>Nebius today: 125K (H100+H200+B200+L40S) →</span><span>300K</span>
                  </div>
                </div>

                <div className="nebius-slider-row">
                  <label>Prefill % of GPU Time: <strong>{nebiusPrefillPct}%</strong></label>
                  <input type="range" min="30" max="90" step="5"
                    value={nebiusPrefillPct} onChange={e => setNebiusPrefillPct(Number(e.target.value))} />
                  <div className="nebius-slider-hints">
                    <span>30%</span><span>Agentic workloads: 65% →</span><span>90%</span>
                  </div>
                </div>

                <div className="nebius-ctx-select">
                  <label>Agent Context Window:</label>
                  <div className="nebius-ctx-buttons">
                    {NEBIUS_CONTEXTS.map((c, i) => (
                      <button key={c.label}
                        className={`nebius-ctx-btn${nebiusContextIdx === i ? " active" : ""}`}
                        onClick={() => setNebiusContextIdx(i)}>
                        {c.label}
                        <span>{(c.savings * 100).toFixed(1)}% saved</span>
                      </button>
                    ))}
                  </div>
                </div>
              </div>

              <div className="nebius-result">
                <div className="nebius-result-main">
                  <p className="nebius-result-label">Korith saves Nebius</p>
                  <p className="nebius-result-value">
                    ${(nebiusSavings.monthlySaved / 1e6).toFixed(0)}M
                    <span>/month</span>
                  </p>
                  <p className="nebius-result-annual">
                    ${(nebiusSavings.annualSaved / 1e9).toFixed(2)}B/year
                  </p>
                </div>
                <div className="nebius-result-breakdown">
                  <div className="nebius-breakdown-row">
                    <span>Fleet GPU cost</span>
                    <strong>${(nebiusSavings.monthlyFleet / 1e6).toFixed(0)}M/mo</strong>
                  </div>
                  <div className="nebius-breakdown-row">
                    <span>Prefill ({nebiusPrefillPct}% of fleet)</span>
                    <strong>${(nebiusSavings.prefillCost / 1e6).toFixed(0)}M/mo</strong>
                  </div>
                  <div className="nebius-breakdown-row highlight">
                    <span>Korith eliminates ({(nebiusSavings.ctx.savings * 100).toFixed(1)}%)</span>
                    <strong>${(nebiusSavings.monthlySaved / 1e6).toFixed(0)}M/mo</strong>
                  </div>
                  <div className="nebius-breakdown-row">
                    <span>Payback at $500M acquisition</span>
                    <strong>{nebiusSavings.paybackMonths.toFixed(1)} months</strong>
                  </div>
                </div>
              </div>

              <p className="nebius-sources">
                Sources: nebius.com/newsroom (H100: 95K, H200: 15K, B200: 7K, L40S: 8K),
                nebius.com/prices ($2.00/hr H100 · $2.30 H200 · $5.50 B200 · $1.82 L40S committed),
                Nebius Q4 2025 earnings ($529.8M FY2025), NVIDIA $2B investment March 11 2026.
                Blended fleet rate: $2.22/GPU/hr. Korith savings from measured benchmarks (128K, H100).
              </p>
            </Section>
          </>
        ) : null}
      </div>
    </div>
  );
}
