import { useEffect, useMemo, useState } from "react";
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

export default function Demo() {
  const [summary, setSummary] = useState(null);
  const [runs, setRuns] = useState([]);
  const [mixed, setMixed] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [warmHitRatioPct, setWarmHitRatioPct] = useState(90);
  const [timerProgress, setTimerProgress] = useState(0);

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
      } catch (err) {
        if (mounted) setError(err.message || "Failed to load benchmark data");
      } finally {
        if (mounted) setLoading(false);
      }
    };
    load();
    return () => {
      mounted = false;
    };
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
    return () => {
      if (raf) cancelAnimationFrame(raf);
    };
  }, []);

  const longSummary = summary?.long || {};
  const shortSummary = summary?.short || {};

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

  return (
    <div className="demo-page">
      <div className="demo-shell">
        <header className="demo-header">
          <div>
            <p className="demo-kicker">Axropus AMF Demo</p>
            <h1>Benchmark Dashboard</h1>
            <p className="demo-subtitle">Instant value view for long-context inference acceleration.</p>
          </div>
          <div className="demo-badge">Public /demo</div>
        </header>

        {error ? <div className="demo-error">{error}</div> : null}
        {loading ? <div className="demo-loading">Loading benchmark data...</div> : null}

        {!loading && !error ? (
          <>
            <section className="demo-grid metrics-grid">
              <article className="demo-card metric-card">
                <p className="metric-label">End-to-End Reduction</p>
                <p className="metric-value">{fmtPct(longSummary.e2e_savings_pct, 1)}</p>
                <p className="metric-sub">128K measured benchmark</p>
              </article>
              <article className="demo-card metric-card">
                <p className="metric-label">Cache Hit Rate</p>
                <p className="metric-value">{fmtPct(Number(longSummary.hit_rate || 0) * 100, 2)}</p>
                <p className="metric-sub">Replay workload</p>
              </article>
              <article className="demo-card metric-card">
                <p className="metric-label">Warm Hits</p>
                <p className="metric-value">{Number(longSummary.hits || 0)}/{Number(longSummary.runs || 0)}</p>
                <p className="metric-sub">Cold misses: {Number(longSummary.misses || 0)}</p>
              </article>
              <article className="demo-card metric-card">
                <p className="metric-label">Warm Prefill</p>
                <p className="metric-value">{fmtMs(warmPrefillAvg)}</p>
                <p className="metric-sub">100% prefix elimination on warm runs</p>
              </article>
              <article className="demo-card metric-card">
                <p className="metric-label">Short Context Floor</p>
                <p className="metric-value">{fmtPct(shortSummary.e2e_savings_pct, 1)}</p>
                <p className="metric-sub">4K measured</p>
              </article>
            </section>

            <section className="demo-card chart-card">
              <div className="card-head">
                <h2>Live Benchmark Viewer</h2>
                <p>Run 1 cold miss vs runs 2-817 warm hits</p>
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
                    <ReferenceLine y={warmMs} stroke="#22c55e" strokeDasharray="4 4" />
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
            </section>

            <section className="demo-grid two-col">
              <article className="demo-card">
                <div className="card-head">
                  <h2>Context Scaling</h2>
                  <p>Savings increase with context length</p>
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
                      <Line type="monotone" dataKey="savings_pct" stroke="#60a5fa" strokeWidth={3} dot={{ r: 4 }} />
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
              </article>

              <article className="demo-card">
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
              </article>
            </section>

            <section className="demo-card">
              <div className="card-head">
                <h2>Before / After</h2>
                <p>Cold run latency vs warm replay latency</p>
              </div>
              <div className="compare-grid">
                <div className="compare-card cold">
                  <p className="compare-label">Cold Run</p>
                  <p className="compare-time">{fmtSeconds(animatedCold)}</p>
                  <p className="compare-sub">Full duration: {fmtSeconds(coldSeconds)} (4+ minutes)</p>
                  <div className="time-bar"><span style={{ width: `${timerProgress * 100}%` }} /></div>
                </div>
                <div className="compare-card warm">
                  <p className="compare-label">Warm Run</p>
                  <p className="compare-time">{fmtSeconds(animatedWarm)}</p>
                  <p className="compare-sub">Full duration: {fmtSeconds(warmSeconds)}</p>
                  <div className="time-bar"><span style={{ width: `${timerProgress * 100}%` }} /></div>
                </div>
              </div>
            </section>
          </>
        ) : null}
      </div>
    </div>
  );
}
