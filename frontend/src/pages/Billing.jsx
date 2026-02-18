import { useEffect, useState } from "react";
import Sidebar from "../components/Sidebar";
import "../styles/billing.css";
import api from "../api";

function centsToDollars(cents) {
  return (Number(cents || 0) / 100).toFixed(2);
}

export default function Billing() {
  const [summary, setSummary] = useState(null);
  const [invoices, setInvoices] = useState([]);
  const [error, setError] = useState("");

  useEffect(() => {
    let mounted = true;
    Promise.all([api.getBillingSummary(), api.getInvoices()])
      .then(([s, i]) => {
        if (!mounted) return;
        setSummary(s);
        setInvoices(i || []);
      })
      .catch((err) => {
        if (mounted) setError(err.message || "Failed to load billing");
      });
    return () => {
      mounted = false;
    };
  }, []);

  return (
    <div className="billing-layout">
      <Sidebar />
      <main className="billing-main">
        <h1>Billing</h1>
        {error ? <p className="auth-error">{error}</p> : null}

        <section className="period-card">
          <h3>Current Billing Period</h3>
          <p>
            {summary?.current_period?.start || "-"} — {summary?.current_period?.end || "-"}
          </p>
          <div className="period-grid">
            <div>Tokens: {Number(summary?.total_tokens || 0).toLocaleString()}</div>
            <div>Amount: ${centsToDollars(summary?.amount_cents)}</div>
            <div>{summary?.pricing || "$0.10 per million tokens"}</div>
          </div>
        </section>

        <section className="roi-card">
          <h3>Your ROI</h3>
          <p>{summary?.roi || "For every $1 you pay Axropus, you save $0"}</p>
        </section>

        <section className="invoice-card">
          <h3>Invoice History</h3>
          {!invoices.length ? (
            <p>No invoices yet. Your first invoice will be generated at the end of your billing period.</p>
          ) : (
            <table>
              <thead>
                <tr>
                  <th>Period</th>
                  <th>Tokens</th>
                  <th>Amount</th>
                  <th>Status</th>
                </tr>
              </thead>
              <tbody>
                {invoices.map((invoice) => (
                  <tr key={invoice.id}>
                    <td>
                      {invoice.period_start || "-"} — {invoice.period_end || "-"}
                    </td>
                    <td>{Number(invoice.total_tokens || 0).toLocaleString()}</td>
                    <td>${centsToDollars(invoice.amount_cents)}</td>
                    <td>
                      <span className={`badge ${String(invoice.status || "pending").toLowerCase()}`}>{invoice.status}</span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </section>

        <section className="payment-placeholder">
          Payment method coming soon. Contact pilot@axropus.com for billing questions.
        </section>
      </main>
    </div>
  );
}
