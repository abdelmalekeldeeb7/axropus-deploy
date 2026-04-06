import { NavLink } from "react-router-dom";

const navItems = [
  { to: "/hub", icon: "\u2B21", label: "Models" },
  { to: "/playground", icon: "\u25B7", label: "Chat" },
  { to: "/claws", icon: "\uD83E\uDD9E", label: "Claws" },
  { to: "/dashboard", icon: "\u25D0", label: "Metrics" },
  { to: "/billing", icon: "\u25C9", label: "Billing" },
];

export default function Sidebar() {
  return (
    <aside
      className="fixed left-0 top-0 h-screen flex flex-col items-center py-5 z-50"
      style={{
        width: 72,
        background: "rgba(255,255,255,0.02)",
        borderRight: "1px solid rgba(255,255,255,0.08)",
      }}
    >
      {/* Logo */}
      <div
        className="w-10 h-10 rounded-xl flex items-center justify-center text-lg font-bold mb-8"
        style={{
          background: "linear-gradient(135deg, #E85D3A, #FF8C42)",
          color: "#fff",
        }}
      >
        A
      </div>

      {/* Nav */}
      <nav className="flex flex-col items-center gap-1 flex-1">
        {navItems.map(({ to, icon, label }) => (
          <NavLink
            key={to}
            to={to}
            className="no-underline"
            style={{ textDecoration: "none" }}
          >
            {({ isActive }) => (
              <div
                className="flex flex-col items-center justify-center rounded-xl transition-all duration-200"
                style={{
                  width: 52,
                  height: 52,
                  background: isActive ? "rgba(232,93,58,0.15)" : "transparent",
                  color: isActive ? "#E85D3A" : "#666",
                }}
                onMouseEnter={(e) => {
                  if (!isActive) e.currentTarget.style.background = "rgba(255,255,255,0.06)";
                }}
                onMouseLeave={(e) => {
                  if (!isActive) e.currentTarget.style.background = "transparent";
                }}
              >
                <span className="text-lg leading-none">{icon}</span>
                <span className="text-[9px] mt-1 font-medium">{label}</span>
              </div>
            )}
          </NavLink>
        ))}
      </nav>

      {/* Bottom */}
      <div className="flex flex-col items-center gap-3">
        <NavLink
          to="/settings"
          className="no-underline"
          style={{ textDecoration: "none" }}
        >
          {({ isActive }) => (
            <div
              className="flex items-center justify-center rounded-xl transition-all duration-200"
              style={{
                width: 40,
                height: 40,
                background: isActive ? "rgba(232,93,58,0.15)" : "transparent",
                color: isActive ? "#E85D3A" : "#666",
                fontSize: 18,
              }}
            >
              &#x2699;
            </div>
          )}
        </NavLink>
        <div
          className="w-8 h-8 rounded-full flex items-center justify-center text-xs font-bold"
          style={{
            background: "linear-gradient(135deg, #E85D3A, #FF8C42)",
            color: "#fff",
          }}
        >
          U
        </div>
      </div>
    </aside>
  );
}
