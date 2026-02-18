export default function StatusBadge({ status = "inactive", label, detail = "" }) {
  const normalized = String(status).toLowerCase();
  const isGood = normalized === "active" || normalized === "enforced" || normalized === "success";
  return (
    <div className="status-card">
      <div className={`status-pill ${isGood ? "ok" : "bad"}`}>{label || status}</div>
      {detail ? <div className="status-detail">{detail}</div> : null}
    </div>
  );
}
