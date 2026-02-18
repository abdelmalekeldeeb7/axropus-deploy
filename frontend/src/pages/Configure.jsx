import { useEffect, useMemo, useState } from "react";
import { Lock, CheckCircle2 } from "lucide-react";
import { useNavigate } from "react-router-dom";
import api from "../api";
import { MODEL_OPTIONS, inferDraftModel } from "../data/models";
import { useAuth } from "../context/AuthContext";
import "../styles/configure.css";

const runtimeCards = [
  { id: "vllm", title: "vLLM", desc: "Most popular open-source inference runtime" },
  { id: "sglang", title: "SGLang", desc: "High-performance serving with RadixAttention" },
  { id: "trtllm", title: "TensorRT-LLM", desc: "NVIDIA's optimized inference runtime" },
];

export default function Configure() {
  const navigate = useNavigate();
  const { setLastDeployment } = useAuth();

  const [step, setStep] = useState(1);
  const [runtime, setRuntime] = useState("");
  const [family, setFamily] = useState("Llama");
  const [size, setSize] = useState("8B");
  const [authMethod, setAuthMethod] = useState("key");
  const [apiKeys, setApiKeys] = useState([]);
  const [apiKeyId, setApiKeyId] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const [cluster, setCluster] = useState({
    host: "",
    port: 22,
    username: "ubuntu",
    ssh_key: "",
    password: "",
    python_path: "/usr/bin/python3",
  });

  useEffect(() => {
    let mounted = true;
    api
      .getKeys()
      .then((rows) => {
        if (!mounted) return;
        setApiKeys(rows || []);
        if (rows?.length) setApiKeyId(String(rows[0].id));
      })
      .catch((err) => {
        if (mounted) setError(err.message || "Failed to load API keys");
      });
    return () => {
      mounted = false;
    };
  }, []);

  useEffect(() => {
    const sizes = MODEL_OPTIONS[family] || [];
    if (!sizes.includes(size)) {
      setSize(sizes[0] || "");
    }
  }, [family, size]);

  const draftModel = useMemo(() => inferDraftModel(family, size), [family, size]);

  const deploy = async () => {
    setLoading(true);
    setError("");
    try {
      const payload = {
        host: cluster.host,
        port: Number(cluster.port || 22),
        username: cluster.username || "ubuntu",
        auth_method: authMethod,
        ssh_key: authMethod === "key" ? cluster.ssh_key : undefined,
        password: authMethod === "password" ? cluster.password : undefined,
        python_path: cluster.python_path || undefined,
        api_key_id: Number(apiKeyId),
        runtime,
        model_family: family.toLowerCase(),
        model_size: size,
      };

      const res = await api.deploy(payload);
      const meta = {
        id: res.deployment_id,
        runtime,
        model_family: family,
        model_size: size,
        status: "pending",
      };
      setLastDeployment(meta);
      navigate(`/deploy/${res.deployment_id}`, { state: { deployment: meta } });
    } catch (err) {
      setError(err.message || "Deployment failed to start");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="wizard-page">
      <div className="wizard-card">
        <div className="wizard-header">
          <div className={`wizard-step ${step >= 1 ? "current" : ""}`}>① Select Runtime</div>
          <div className={`wizard-step ${step >= 2 ? "current" : ""}`}>② Select Model</div>
          <div className={`wizard-step ${step >= 3 ? "current" : ""}`}>③ Connect & Deploy</div>
        </div>

        {step === 1 ? (
          <section>
            <h2>Step 1 — Select Runtime</h2>
            <div className="runtime-grid">
              {runtimeCards.map((card) => (
                <button
                  key={card.id}
                  type="button"
                  className={`runtime-card ${runtime === card.id ? "selected" : ""}`}
                  onClick={() => setRuntime(card.id)}
                >
                  <h3>{card.title}</h3>
                  <p>{card.desc}</p>
                </button>
              ))}
            </div>
            <div className="wizard-actions">
              <button type="button" className="primary" disabled={!runtime} onClick={() => setStep(2)}>
                Next
              </button>
            </div>
          </section>
        ) : null}

        {step === 2 ? (
          <section>
            <h2>Step 2 — Select Model</h2>
            <div className="model-grid">
              <label>
                Model Family
                <select value={family} onChange={(e) => setFamily(e.target.value)}>
                  {Object.keys(MODEL_OPTIONS).map((f) => (
                    <option key={f} value={f}>
                      {f}
                    </option>
                  ))}
                </select>
              </label>

              <label>
                Model Size
                <select value={size} onChange={(e) => setSize(e.target.value)}>
                  {(MODEL_OPTIONS[family] || []).map((s) => (
                    <option key={s} value={s}>
                      {s}
                    </option>
                  ))}
                </select>
              </label>
            </div>

            <div className="draft-model">
              <CheckCircle2 size={16} /> Draft model for Spec V2: {draftModel}
            </div>

            <div className="wizard-actions split">
              <button type="button" className="secondary" onClick={() => setStep(1)}>
                Back
              </button>
              <button type="button" className="primary" onClick={() => setStep(3)}>
                Next
              </button>
            </div>
          </section>
        ) : null}

        {step === 3 ? (
          <section>
            <h2>Step 3 — Connect Cluster</h2>
            <div className="cluster-grid">
              <label>
                Host
                <input
                  placeholder="10.0.1.50 or gpu-cluster.company.com"
                  value={cluster.host}
                  onChange={(e) => setCluster((prev) => ({ ...prev, host: e.target.value }))}
                />
              </label>

              <label>
                Port
                <input
                  type="number"
                  value={cluster.port}
                  onChange={(e) => setCluster((prev) => ({ ...prev, port: Number(e.target.value || 22) }))}
                />
              </label>

              <label>
                Username
                <input
                  value={cluster.username}
                  onChange={(e) => setCluster((prev) => ({ ...prev, username: e.target.value }))}
                />
              </label>

              <label className="wide">
                API Key
                <select value={apiKeyId} onChange={(e) => setApiKeyId(e.target.value)}>
                  <option value="">Select key</option>
                  {apiKeys.map((key) => (
                    <option key={key.id} value={key.id}>
                      {key.key} ({key.tier})
                    </option>
                  ))}
                </select>
              </label>
            </div>

            <div className="auth-toggle">
              <button type="button" className={authMethod === "key" ? "selected" : ""} onClick={() => setAuthMethod("key")}>SSH Key</button>
              <button type="button" className={authMethod === "password" ? "selected" : ""} onClick={() => setAuthMethod("password")}>Password</button>
            </div>

            {authMethod === "key" ? (
              <label className="wide">
                SSH Key
                <textarea
                  rows={6}
                  value={cluster.ssh_key}
                  onChange={(e) => setCluster((prev) => ({ ...prev, ssh_key: e.target.value }))}
                />
              </label>
            ) : (
              <label className="wide">
                Password
                <input
                  type="password"
                  value={cluster.password}
                  onChange={(e) => setCluster((prev) => ({ ...prev, password: e.target.value }))}
                />
              </label>
            )}

            <p className="security-note">
              <Lock size={14} /> Your credentials are used only for this deployment session. They are never stored, logged,
              or transmitted after installation.
            </p>

            {error ? <div className="auth-error">{error}</div> : null}

            <div className="wizard-actions split">
              <button type="button" className="secondary" onClick={() => setStep(2)}>
                Back
              </button>
              <button
                type="button"
                className="primary deploy"
                onClick={deploy}
                disabled={loading || !runtime || !cluster.host || !apiKeyId || (authMethod === "key" ? !cluster.ssh_key : !cluster.password)}
              >
                {loading ? "Deploying..." : "Deploy"}
              </button>
            </div>
          </section>
        ) : null}
      </div>
    </div>
  );
}
