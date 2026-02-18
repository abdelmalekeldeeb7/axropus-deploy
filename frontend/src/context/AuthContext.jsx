import { createContext, useContext, useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import api from "../api";

const AuthContext = createContext(null);

export function AuthProvider({ children }) {
  const navigate = useNavigate();
  const [token, setToken] = useState(null);
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(false);
  const [lastDeployment, setLastDeployment] = useState(null);

  useEffect(() => {
    api.setToken(token);
  }, [token]);

  useEffect(() => {
    api.setUnauthorizedHandler(() => {
      setToken(null);
      setUser(null);
      setLastDeployment(null);
      navigate("/login", { replace: true });
    });
    return () => api.setUnauthorizedHandler(null);
  }, [navigate]);

  useEffect(() => {
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
