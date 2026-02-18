/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx,ts,tsx}"],
  theme: {
    extend: {
      colors: {
        "near-black": "#1A1917",
        "off-white": "#F7F6F3",
        cream: "#EDEBE6",
        accent: "#1D4ED8",
        green: "#16A34A",
        red: "#DC2626",
        "mid-gray": "#9B978F",
      },
      fontFamily: {
        heading: ["'Instrument Serif'", "serif"],
        body: ["'DM Sans'", "sans-serif"],
        mono: ["'JetBrains Mono'", "monospace"],
      },
      boxShadow: {
        card: "0 10px 30px rgba(26, 25, 23, 0.08)",
      },
    },
  },
  plugins: [],
};
