import { createContext, useContext, useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import api from "../api";

const AuthContext = createContext(null);

// ── Dev-mode auth bypass ────────────────────────────────────────────────────
//
// When running locally (`npm run dev`) we skip the login gate by injecting
// a fake session. This lets you hit every Protected page (Dashboard, Hub,
// Playground, Claws, Billing, Settings) without having to sign up or wire
// the backend up first.
//
// To DISABLE the bypass (e.g. to actually test the login flow locally),
// set VITE_DEV_BYPASS_AUTH=false in frontend/.env.local.
//
// The bypass is ALWAYS off in production builds (`import.meta.env.PROD`),
// so deploying to Vercel / Railway stays secure.
const DEV_BYPASS_AUTH =
  import.meta.env.DEV &&
  String(import.meta.env.VITE_DEV_BYPASS_AUTH || "true").toLowerCase() !== "false";

const DEV_FAKE_TOKEN = "dev-bypass-token";
const DEV_FAKE_USER = {
  id: "dev",
  email: "dev@localhost",
  company_name: "Axropus Dev",
  tier: "enterprise",
  created_at: new Date().toISOString(),
};

export function AuthProvider({ children }) {
  const navigate = useNavigate();
  const [token, setToken] = useState(DEV_BYPASS_AUTH ? DEV_FAKE_TOKEN : null);
  const [user, setUser] = useState(DEV_BYPASS_AUTH ? DEV_FAKE_USER : null);
  const [loading, setLoading] = useState(false);
  const [lastDeployment, setLastDeployment] = useState(null);

  useEffect(() => {
    api.setToken(token);
  }, [token]);

  useEffect(() => {
    // In dev-bypass mode, don't let a backend 401 kick us back to /login.
    // The fake token is never going to validate against a real backend.
    if (DEV_BYPASS_AUTH) {
      return undefined;
    }
    api.setUnauthorizedHandler(() => {
      setToken(null);
      setUser(null);
      setLastDeployment(null);
      navigate("/login", { replace: true });
    });
    return () => api.setUnauthorizedHandler(null);
  }, [navigate]);

  useEffect(() => {
    // Skip the /api/auth/me validation call when in dev-bypass mode.
    // The fake token won't pass backend auth, and the failure would nuke
    // the session and redirect us to /login.
    if (DEV_BYPASS_AUTH) {
      return undefined;
    }
    let cancelled = false;
    if (!token) return undefined;

    const validateSession = async () => {
      try {
        const me = await api.me();
        if (!cancelled) {
          setUser(me);
        }
      } catch {
        if (!cancelled) {
          setToken(null);
          setUser(null);
          setLastDeployment(null);
        }
      }
    };

    validateSession();
    return () => {
      cancelled = true;
    };
  }, [token]);

  const logout = () => {
    setToken(null);
    setUser(null);
    setLastDeployment(null);
    api.clearToken();
    navigate("/login", { replace: true });
  };

  const attachSession = async (nextToken) => {
    api.setToken(nextToken);
    setToken(nextToken);
    const me = await api.me();
    setUser(me);
    return me;
  };

  const signup = async ({ email, password, company_name }) => {
    setLoading(true);
    try {
      const data = await api.signup(email, password, company_name);
      await attachSession(data.token);
      return data;
    } finally {
      setLoading(false);
    }
  };

  const login = async ({ email, password }) => {
    setLoading(true);
    try {
      const data = await api.login(email, password);
      await attachSession(data.token);
      return data;
    } finally {
      setLoading(false);
    }
  };

  const value = useMemo(
    () => ({
      token,
      user,
      loading,
      signup,
      login,
      logout,
      isAuthenticated: Boolean(token),
      lastDeployment,
      setLastDeployment,
    }),
    [token, user, loading, lastDeployment]
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  const ctx = useContext(AuthContext);
  if (!ctx) {
    throw new Error("useAuth must be used inside AuthProvider");
  }
  return ctx;
}
