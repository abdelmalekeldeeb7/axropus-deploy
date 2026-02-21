import { useEffect, useMemo, useState } from "react";
import { Activity, Copy, ShieldCheck, Sparkles } from "lucide-react";
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import api from "../api";
import Sidebar from "../components/Sidebar";
import StatsCard from "../components/StatsCard";
import StatusBadge from "../components/StatusBadge";
import { useAuth } from "../context/AuthContext";
import "../styles/dashboard.css";

function pct(v) {
  return `${Math.round(Number(v || 0) * 100)}%`;
}

function maskKey(key) {
  if (!key) return "-";
  if (key.length < 10) return key;
  return `${key.slice(0, 3)}-****${key.slice(-4)}`;
}

export default function Dashboard() {
  const { lastDeployment } = useAuth();
  const [dashboard, setDashboard] = useState(null);
  const [keys, setKeys] = useState([]);
  const [error, setError] = useState("");
  const [lastUpdated, setLastUpdated] = useState(null);

  const load = async () => {
    try {
      const [d, k] = await Promise.all([api.getDashboard(), api.getKeys()]);
      setDashboard(d);
      setKeys(k || []);
      setLastUpdated(new Date());
      setError("");
    } catch (err) {
      setError(err.message || "Failed to load dashboard");
    }
  };

  useEffect(() => {
    load();
    const t = setInterval(load, 30000);
    return () => clearInterval(t);
  }, []);

  const primaryKey = keys[0]?.key || "";
  const deployment = useMemo(() => lastDeployment || {}, [lastDeployment]);
  const chartRows = dashboard?.daily_breakdown || [];

  const copyKey = async () => {
    if (!primaryKey) return;
    try {
      await navigator.clipboard.writeText(primaryKey);
    } catch {
      // ignore
    }
  };

  return (
    <div className="dashboard-layout">
      <Sidebar
        status={dashboard?.status || "inactive"}
        runtime={deployment.runtime || "-"}
        model={deployment.model_family ? `${deployment.model_family} ${deployment.model_size}` : "-"}
      />

      <main className="dashboard-main">
        <div className="dashboard-main-shell">
          <div className="dashboard-top reveal r1">
            <div>
              <h1>Dashboard</h1>
              <p className="dashboard-subtitle">Live savings and runtime health across your Axropus deployment.</p>
            </div>
            <div className={`dashboard-live ${dashboard?.status === "active" ? "active" : ""}`}>
              <span className="live-dot" />
              <span>{dashboard?.status === "active" ? "Metrics Live" : "Waiting for Metrics"}</span>
            </div>
          </div>

          {error ? <p className="auth-error reveal r1">{error}</p> : null}

          <div className="highlights-row reveal r2">
            <div className="highlight-chip">
              <Activity size={16} />
              <div>
                <span className="highlight-label">Tokens Processed</span>
                <strong>{Number(dashboard?.total_tokens_processed || 0).toLocaleString()}</strong>
              </div>
            </div>
            <div className="highlight-chip">
              <Sparkles size={16} />
              <div>
                <span className="highlight-label">Prefix Skipped</span>
                <strong>{Number(dashboard?.total_prefix_skipped || 0).toLocaleString()}</strong>
              </div>
            </div>
            <div className="highlight-chip">
              <ShieldCheck size={16} />
              <div>
                <span className="highlight-label">Decode Accelerated</span>
                <strong>{Number(dashboard?.total_decode_accelerated || 0).toLocaleString()}</strong>
              </div>
            </div>
          </div>

          <div className="stats-row reveal r3">
          <StatsCard
            label="Compute Saved"
            value={pct(dashboard?.current_compute_saved_pct)}
            color={Number(dashboard?.current_compute_saved_pct || 0) > 0.5 ? "green" : ""}
          />
          <StatsCard label="Tokens Today" value={Number(dashboard?.tokens_today || 0).toLocaleString()} />
          <StatsCard
            label="Throughput"
            value={`${Number(dashboard?.current_effective_tps || 0).toFixed(1)} TPS`}
            subtitle={`vs ${Number(dashboard?.current_baseline_tps || 0).toFixed(1)} baseline`}
          />
          <StatsCard
            label="Monthly Savings"
            value={`$${Number(dashboard?.estimated_monthly_savings_usd || 0).toFixed(2)}`}
            color="green"
          />
          </div>

          <div className="charts-row reveal r4">
            <div className="chart-card">
            <h3>Daily Savings</h3>
            {chartRows.length ? (
              <ResponsiveContainer width="100%" height={260}>
                <AreaChart data={chartRows}>
                  <CartesianGrid stroke="#EDEBE6" strokeDasharray="3 3" />
                  <XAxis dataKey="date" />
                  <YAxis tickFormatter={(v) => `${Math.round(Number(v) * 100)}%`} />
                  <Tooltip formatter={(v) => `${(Number(v) * 100).toFixed(1)}%`} />
                  <Area
                    type="monotone"
                    dataKey="savings_pct"
                    stroke="#1D4ED8"
                    fill="#DBEAFE"
                    strokeWidth={2.2}
                    isAnimationActive
                    animationDuration={900}
                  />
                </AreaChart>
              </ResponsiveContainer>
            ) : (
              <p className="empty-chart">No metrics yet. Savings trend appears once SDK metrics arrive.</p>
            )}
            </div>

            <div className="chart-card">
            <h3>Tokens Processed</h3>
            {chartRows.length ? (
              <ResponsiveContainer width="100%" height={260}>
                <BarChart data={chartRows}>
                  <CartesianGrid stroke="#EDEBE6" strokeDasharray="3 3" />
                  <XAxis dataKey="date" />
                  <YAxis />
                  <Tooltip formatter={(v) => Number(v).toLocaleString()} />
                  <Bar dataKey="tokens" fill="#16A34A" radius={[6, 6, 0, 0]} isAnimationActive animationDuration={900} />
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <p className="empty-chart">No token history yet.</p>
            )}
            </div>
          </div>

          <div className="status-row reveal r5">
            <StatusBadge
              status={dashboard?.status === "active" ? "active" : "inactive"}
              label="AMF"
              detail={`Hit ${pct(dashboard?.current_amf_hit_rate)}`}
            />
            <StatusBadge
              status={dashboard?.status === "active" ? "active" : "inactive"}
              label="Spec V2"
              detail={`Acceptance ${pct(dashboard?.current_spec_acceptance_rate)}`}
            />
            <StatusBadge status="enforced" label="Zero-Data" detail="Enforced" />
          </div>

          <div className="deploy-info reveal r6">
            <h3>Deployment Info</h3>
            <div className="deploy-grid">
              <div>Runtime: {deployment.runtime || "-"}</div>
              <div>Model: {deployment.model_family ? `${deployment.model_family} ${deployment.model_size}` : "-"}</div>
              <div>Deployed: {deployment.deployed_at || "-"}</div>
              <div className="api-key-row">
                API Key: {maskKey(primaryKey)}
                {primaryKey ? (
                  <button type="button" onClick={copyKey} title="Copy API key">
                    <Copy size={14} />
                  </button>
                ) : null}
              </div>
            </div>
          </div>

          <div className="updated-at">
            Last updated: {lastUpdated ? lastUpdated.toLocaleTimeString() : "-"}
          </div>
        </div>
      </main>
    </div>
  );
}
