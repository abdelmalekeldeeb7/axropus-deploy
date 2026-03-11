import { Navigate, Route, Routes } from "react-router-dom";
import ProtectedRoute from "./components/ProtectedRoute";
import { useAuth } from "./context/AuthContext";
import Billing from "./pages/Billing";
import Configure from "./pages/Configure";
import Dashboard from "./pages/Dashboard";
import Demo from "./pages/Demo";
import Deploy from "./pages/Deploy";
import Login from "./pages/Login";
import Settings from "./pages/Settings";
import Signup from "./pages/Signup";

function PublicOnlyRoute({ children }) {
  const { isAuthenticated } = useAuth();
  if (isAuthenticated) {
    return <Navigate to="/dashboard" replace />;
  }
  return children;
}

export default function App() {
  const { isAuthenticated } = useAuth();

  return (
    <Routes>
      <Route
        path="/signup"
        element={
          <PublicOnlyRoute>
            <Signup />
          </PublicOnlyRoute>
        }
      />
      <Route
        path="/login"
        element={
          <PublicOnlyRoute>
            <Login />
          </PublicOnlyRoute>
        }
      />
      <Route path="/demo" element={<Demo />} />

      <Route
        path="/configure"
        element={
          <ProtectedRoute>
            <Configure />
          </ProtectedRoute>
        }
      />
      <Route
        path="/deploy/:id"
        element={
          <ProtectedRoute>
            <Deploy />
          </ProtectedRoute>
        }
      />
      <Route
        path="/dashboard"
        element={
          <ProtectedRoute>
            <Dashboard />
          </ProtectedRoute>
        }
      />
      <Route
        path="/billing"
        element={
          <ProtectedRoute>
            <Billing />
          </ProtectedRoute>
        }
      />
      <Route
        path="/settings"
        element={
          <ProtectedRoute>
            <Settings />
          </ProtectedRoute>
        }
      />
      <Route path="/" element={<Navigate to={isAuthenticated ? "/dashboard" : "/signup"} replace />} />
      <Route path="*" element={<Navigate to={isAuthenticated ? "/dashboard" : "/signup"} replace />} />
    </Routes>
  );
}
