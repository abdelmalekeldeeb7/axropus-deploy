import { useEffect, useMemo, useState } from "react";
import { Link, useLocation, useNavigate, useParams } from "react-router-dom";
import Terminal from "../components/Terminal";
import api from "../api";
import { useAuth } from "../context/AuthContext";
import "../styles/deploy.css";

const STEP_NAMES = [
  "Connect",
  "Python",
  "Install",
  "Runtime",
  "Model",
  "Draft",
  "Configure",
  "AMF",
  "Spec V2",
  "Verify",
];

export default function Deploy() {
  const { id } = useParams();
  const navigate = useNavigate();
  const location = useLocation();
  const { setLastDeployment } = useAuth();

  const [lines, setLines] = useState([]);
  const [progress, setProgress] = useState(0);
  const [status, setStatus] = useState("running");
  const [showDone, setShowDone] = useState(false);

  const deploymentMeta = location.state?.deployment || null;

  const currentStep = useMemo(() => {
    if (!lines.length) return 0;
    return Number(lines[lines.length - 1]?.step || 0);
  }, [lines]);

  const stepDone = useMemo(() => {
    const done = new Set();
    lines.forEach((line) => {
      if (line.status === "success") done.add(Number(line.step));
    });
    return done;
  }, [lines]);

  const handleUpdate = (entry) => {
    setLines((prev) => [...prev, entry]);
    setProgress((prev) => Math.max(prev, Number(entry.progress || 0)));
    if (entry.status === "error") {
      setStatus("error");
    }
    if (String(entry.message || "").includes("DEPLOYMENT COMPLETE")) {
      setStatus("success");
      setLastDeployment((prev) => ({ ...(prev || deploymentMeta || {}), id: Number(id), status: "active" }));
      setTimeout(() => setShowDone(true), 2000);
    }
  };

  const handleClose = async () => {
    try {
      const data = await api.deployStatus(id);
      if (data?.status === "active") {
        setStatus("success");
        setLastDeployment((prev) => ({ ...(prev || deploymentMeta || {}), id: Number(id), status: "active", deployed_at: data.deployed_at }));
        setShowDone(true);
      } else if (data?.status === "failed") {
        setStatus("error");
      }
    } catch {
      // no-op
    }
  };

  return (
    <div className="deploy-page">
      <div className="deploy-progress-top">
        <div style={{ width: `${Math.round(progress * 100)}%` }} />
      </div>

      <div className="deploy-shell">
        <h1>Deploying to {deploymentMeta?.host || "cluster"}...</h1>

        <Terminal deploymentId={id} onUpdate={handleUpdate} onClose={handleClose} />

        <div className="step-pills">
          {STEP_NAMES.map((name, idx) => {
            const step = idx + 1;
            const done = stepDone.has(step);
            const active = !done && step === currentStep;
            return (
              <span key={name} className={`step-pill ${done ? "done" : ""} ${active ? "active" : ""}`}>
                {name}
              </span>
            );
          })}
        </div>

        {status === "success" && showDone ? (
          <button type="button" className="done-btn" onClick={() => navigate("/dashboard")}>
            Go to Dashboard →
          </button>
        ) : null}

        {status === "error" ? (
          <div className="error-actions">
            <button type="button" onClick={() => navigate("/configure")}>
              Retry
            </button>
            <Link to="/configure">Back to Configure</Link>
          </div>
        ) : null}
      </div>
    </div>
  );
}
