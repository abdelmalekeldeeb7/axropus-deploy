import { NavLink } from "react-router-dom";
import { BarChart3, CreditCard, Settings } from "lucide-react";

const links = [
  { to: "/dashboard", label: "Dashboard", icon: BarChart3 },
  { to: "/billing", label: "Billing", icon: CreditCard },
  { to: "/settings", label: "Settings", icon: Settings },
];

export default function Sidebar({ status = "inactive", runtime = "-", model = "-" }) {
  const active = String(status).toLowerCase() === "active";

  return (
    <aside className="sidebar">
      <div>
        <div className="sidebar-logo">Axropus</div>
        <nav className="sidebar-nav">
          {links.map(({ to, label, icon: Icon }) => (
            <NavLink key={to} to={to} className={({ isActive }) => `sidebar-link ${isActive ? "active" : ""}`}>
              <Icon size={16} />
              <span>{label}</span>
            </NavLink>
          ))}
        </nav>
      </div>

      <div className="sidebar-footer">
        <div className="system-status">
          <span className={`dot ${active ? "green" : "red"}`} />
          <span>{active ? "System Active" : "Inactive"}</span>
        </div>
        <div className="meta-line">{runtime} · {model}</div>
      </div>
    </aside>
  );
}
