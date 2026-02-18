import { useEffect, useMemo, useState } from "react";
import { Copy } from "lucide-react";
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
        <h1>Dashboard</h1>
        {error ? <p className="auth-error">{error}</p> : null}

        <div className="stats-row">
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

        <div className="charts-row">
          <div className="chart-card">
            <h3>Daily Savings</h3>
            {chartRows.length ? (
              <ResponsiveContainer width="100%" height={260}>
                <AreaChart data={chartRows}>
                  <CartesianGrid stroke="#EDEBE6" strokeDasharray="3 3" />
                  <XAxis dataKey="date" />
                  <YAxis tickFormatter={(v) => `${Math.round(Number(v) * 100)}%`} />
                  <Tooltip formatter={(v) => `${(Number(v) * 100).toFixed(1)}%`} />
                  <Area type="monotone" dataKey="savings_pct" stroke="#1D4ED8" fill="#DBEAFE" />
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
                  <Bar dataKey="tokens" fill="#16A34A" />
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <p className="empty-chart">No token history yet.</p>
            )}
          </div>
        </div>

        <div className="status-row">
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

        <div className="deploy-info">
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

        <div className="updated-at">Last updated: {lastUpdated ? lastUpdated.toLocaleTimeString() : "-"}</div>
      </main>
    </div>
  );
}
