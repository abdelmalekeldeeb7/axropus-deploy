import { useEffect, useMemo, useState } from "react";

function parseDisplayValue(rawValue) {
  const text = String(rawValue ?? "-").trim();
  const match = text.match(/^([^\d\-+]*)([-+]?\d[\d,]*(?:\.\d+)?)(.*)$/);
  if (!match) {
    return { numeric: false, text };
  }

  const numericText = match[2].replace(/,/g, "");
  const target = Number(numericText);
  if (!Number.isFinite(target)) {
    return { numeric: false, text };
  }

  const decimals = (numericText.split(".")[1] || "").length;
  return {
    numeric: true,
    prefix: match[1],
    suffix: match[3],
    grouped: match[2].includes(","),
    decimals,
    target,
  };
}

function formatAnimated(value, meta) {
  const text = Number(value).toLocaleString(undefined, {
    useGrouping: meta.grouped || Math.abs(value) >= 1000,
    minimumFractionDigits: meta.decimals,
    maximumFractionDigits: meta.decimals,
  });
  return `${meta.prefix}${text}${meta.suffix}`;
}

export default function StatsCard({ label, value, subtitle = "", color = "" }) {
  const parsedValue = useMemo(() => parseDisplayValue(value), [value]);
  const [displayValue, setDisplayValue] = useState(parsedValue.numeric ? formatAnimated(0, parsedValue) : String(value ?? "-"));

  useEffect(() => {
    if (!parsedValue.numeric) {
      setDisplayValue(String(value ?? "-"));
      return undefined;
    }

    const target = parsedValue.target;
    const duration = 820;
    let rafId = 0;
    const startAt = performance.now();

    const step = (now) => {
      const progress = Math.min(1, (now - startAt) / duration);
      const eased = 1 - (1 - progress) ** 3;
      setDisplayValue(formatAnimated(target * eased, parsedValue));
      if (progress < 1) {
        rafId = requestAnimationFrame(step);
      }
    };

    rafId = requestAnimationFrame(step);
    return () => cancelAnimationFrame(rafId);
  }, [parsedValue, value]);

  return (
    <div className="stats-card">
      <div className="stats-label">{label}</div>
      <div className={`stats-value ${color}`.trim()}>{displayValue}</div>
      {subtitle ? <div className="stats-subtitle">{subtitle}</div> : null}
    </div>
  );
}
