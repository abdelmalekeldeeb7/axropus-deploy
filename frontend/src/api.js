const API_BASE = String(import.meta.env.VITE_API_URL || "").trim().replace(/\/+$/, "");
const WS_BASE_DEFAULT = import.meta.env.DEV ? "ws://localhost:8000" : "wss://api.axropus.com";

class ApiClient {
  constructor() {
    this.token = null;
    this.onUnauthorized = null;
  }

  setToken(token) {
    this.token = token;
  }

  clearToken() {
    this.token = null;
  }

  setUnauthorizedHandler(handler) {
    this.onUnauthorized = typeof handler === "function" ? handler : null;
  }

  async request(method, path, body = null) {
    const headers = { "Content-Type": "application/json" };
    if (this.token) {
      headers.Authorization = `Bearer ${this.token}`;
    }

    const opts = { method, headers };
    if (body !== null) {
      opts.body = JSON.stringify(body);
    }

    const res = await fetch(`${API_BASE}${path}`, opts);

    let data = null;
    const text = await res.text();
    if (text) {
      try {
        data = JSON.parse(text);
      } catch {
        data = { detail: text };
      }
    }

    if (res.status === 401 && this.token) {
      this.clearToken();
      if (this.onUnauthorized) {
        this.onUnauthorized();
      } else if (typeof window !== "undefined") {
        window.location.href = "/login";
      }
      throw new Error((data && data.detail) || "Unauthorized");
    }

    if (!res.ok) {
      throw new Error((data && data.detail) || "Request failed");
    }

    return data;
  }

  signup(email, password, company_name) {
    return this.request("POST", "/api/auth/signup", { email, password, company_name });
  }

  login(email, password) {
    return this.request("POST", "/api/auth/login", { email, password });
  }

  me() {
    return this.request("GET", "/api/auth/me");
  }

  getKeys() {
    return this.request("GET", "/api/keys");
  }

  generateKey(tier = "standard") {
    return this.request("POST", "/api/keys/generate", { tier });
  }

  revokeKey(keyId) {
    return this.request("DELETE", `/api/keys/${keyId}`);
  }

  deploy(config) {
    return this.request("POST", "/api/deploy", config);
  }

  deployStatus(deploymentId) {
    return this.request("GET", `/api/deploy/status/${deploymentId}`);
  }

  getDashboard() {
    return this.request("GET", "/api/dashboard");
  }

  getBillingSummary() {
    return this.request("GET", "/api/billing/summary");
  }

  getInvoices() {
    return this.request("GET", "/api/billing/invoices");
  }

  getDeployStreamUrl(deploymentId) {
    const wsBase = API_BASE ? API_BASE.replace(/^http/i, "ws") : WS_BASE_DEFAULT;
    return `${wsBase}/api/deploy/stream/${deploymentId}`;
  }
}

export const api = new ApiClient();
export default api;
