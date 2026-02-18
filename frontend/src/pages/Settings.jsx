import { useEffect, useMemo, useState } from "react";
import Sidebar from "../components/Sidebar";
import api from "../api";
import "../styles/billing.css";

function formatExpiry(value) {
  if (!value) return "Never";
  const d = new Date(value);
  return Number.isNaN(d.getTime()) ? String(value) : d.toLocaleString();
}

function maskKey(value) {
  if (!value) return "-";
  if (value.length < 12) return value;
  return `${value.slice(0, 6)}...${value.slice(-6)}`;
}

export default function Settings() {
  const [keys, setKeys] = useState([]);
  const [tier, setTier] = useState("standard");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [message, setMessage] = useState("");

  const hasActiveKey = useMemo(
    () => keys.some((row) => String(row.status).toLowerCase() !== "revoked"),
    [keys]
  );

  const loadKeys = async () => {
    try {
      const rows = await api.getKeys();
      setKeys(Array.isArray(rows) ? rows : []);
      setError("");
    } catch (err) {
      setError(err.message || "Failed to load API keys");
    }
  };

  useEffect(() => {
    loadKeys();
  }, []);

  const onGenerate = async () => {
    setLoading(true);
    setError("");
    setMessage("");
    try {
      const created = await api.generateKey(tier);
      setMessage(`Generated ${created.tier} key successfully.`);
      await loadKeys();
    } catch (err) {
      setError(err.message || "Failed to generate key");
    } finally {
      setLoading(false);
    }
  };

  const onRevoke = async (id) => {
    setLoading(true);
    setError("");
    setMessage("");
    try {
      await api.revokeKey(id);
      setMessage("API key revoked.");
      await loadKeys();
    } catch (err) {
      setError(err.message || "Failed to revoke key");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="billing-layout">
      <Sidebar status={hasActiveKey ? "active" : "inactive"} />
      <main className="billing-main">
        <h1>Settings</h1>
        <section className="period-card">
          <h3>API Keys</h3>
          <p>Generate and revoke keys used by Axropus SDK agents.</p>

          <div className="settings-actions">
            <select value={tier} onChange={(e) => setTier(e.target.value)} disabled={loading}>
              <option value="trial">Trial (7 days)</option>
              <option value="standard">Standard</option>
              <option value="enterprise">Enterprise</option>
            </select>
            <button type="button" onClick={onGenerate} disabled={loading}>
              {loading ? "Working..." : "Generate New Key"}
            </button>
          </div>

          {error ? <p className="auth-error">{error}</p> : null}
          {message ? <p className="settings-success">{message}</p> : null}
        </section>

        <section className="invoice-card">
          <h3>Key Inventory</h3>
          {!keys.length ? (
            <p>No API keys found.</p>
          ) : (
            <table>
              <thead>
                <tr>
                  <th>Key</th>
                  <th>Status</th>
                  <th>Tier</th>
                  <th>Expires</th>
                  <th>Action</th>
                </tr>
              </thead>
              <tbody>
                {keys.map((key) => {
                  const revoked = String(key.status).toLowerCase() === "revoked";
                  return (
                    <tr key={key.id}>
                      <td>
                        <code>{maskKey(key.key)}</code>
                      </td>
                      <td>{key.status}</td>
                      <td>{key.tier}</td>
                      <td>{formatExpiry(key.expires_at)}</td>
                      <td>
                        <button
                          type="button"
                          className="danger-btn"
                          disabled={loading || revoked}
                          onClick={() => onRevoke(key.id)}
                        >
                          {revoked ? "Revoked" : "Revoke"}
                        </button>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          )}
        </section>

        <section className="payment-placeholder">
          Settings panel coming soon: deployment policies, notification hooks, and billing contacts.
        </section>
      </main>
    </div>
  );
}
