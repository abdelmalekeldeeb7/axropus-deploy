import { useEffect, useMemo, useRef, useState } from "react";
import api from "../api";

function classifyLine(message, status) {
  const text = String(message || "");
  if (status === "error" || text.includes("✗") || text.toLowerCase().includes("failed")) return "line-error";
  if (text.includes("✓")) return "line-ok";
  return "line-normal";
}

export default function Terminal({ deploymentId, onUpdate, onClose }) {
  const [lines, setLines] = useState([]);
  const [progress, setProgress] = useState(0);
  const [wsError, setWsError] = useState("");
  const [mode, setMode] = useState("ws");
  const viewRef = useRef(null);
  const pollTimerRef = useRef(null);
  const closedRef = useRef(false);
  const nextPollIndexRef = useRef(0);

  const lastStep = useMemo(() => {
    if (!lines.length) return 0;
    return Number(lines[lines.length - 1]?.step || 0);
  }, [lines]);

  useEffect(() => {
    if (!deploymentId) return undefined;

    closedRef.current = false;
    nextPollIndexRef.current = 0;
    setLines([]);
    setProgress(0);
    setWsError("");
    setMode("ws");

    const pushMessage = (data) => {
      setLines((prev) => [...prev, data]);
      setProgress((prev) => Math.max(prev, Math.max(0, Math.min(1, Number(data.progress || 0)))));
      onUpdate?.(data);
    };

    const stopPolling = () => {
      if (pollTimerRef.current) {
        clearInterval(pollTimerRef.current);
        pollTimerRef.current = null;
      }
    };

    const startPollingFallback = () => {
      if (closedRef.current || pollTimerRef.current) return;
      setMode("poll");
      setWsError("WebSocket unavailable. Using status polling.");

      const poll = async () => {
        try {
          const status = await api.deployStatus(deploymentId);
          const logs = Array.isArray(status?.logs) ? status.logs : [];
          if (logs.length > nextPollIndexRef.current) {
            const next = logs.slice(nextPollIndexRef.current);
            next.forEach((entry) => pushMessage(entry));
            nextPollIndexRef.current = logs.length;
          }
          if (status?.status === "active" || status?.status === "failed") {
            stopPolling();
            onClose?.();
          }
        } catch (err) {
          setWsError(err?.message || "Polling failed");
        }
      };

      poll();
      pollTimerRef.current = setInterval(poll, 2000);
    };

    let ws;
    try {
      ws = new WebSocket(api.getDeployStreamUrl(deploymentId));
    } catch {
      startPollingFallback();
      return () => {
        closedRef.current = true;
        stopPolling();
      };
    }

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        pushMessage(data);
      } catch {
        // Ignore malformed lines.
      }
    };

    ws.onerror = () => {
      startPollingFallback();
    };

    ws.onclose = () => {
      if (!closedRef.current) {
        startPollingFallback();
      }
      onClose?.();
    };

    return () => {
      closedRef.current = true;
      stopPolling();
      try {
        ws.close();
      } catch {
        // ignore close race
      }
    };
  }, [deploymentId, onClose, onUpdate]);

  useEffect(() => {
    if (!viewRef.current) return;
    viewRef.current.scrollTop = viewRef.current.scrollHeight;
  }, [lines]);

  return (
    <div className="terminal-wrap">
      <div className="terminal-topbar">
        <span className="dot red" />
        <span className="dot yellow" />
        <span className="dot green" />
      </div>
      <div className="terminal-progress">
        <div style={{ width: `${Math.round(progress * 100)}%` }} />
      </div>
      <div className="terminal-view" ref={viewRef}>
        {lines.map((line, idx) => (
          <div key={`${idx}-${line.step}-${line.status}`} className={classifyLine(line.message, line.status)}>
            <span className="prefix">[AXROPUS]</span> {String(line.message || "").replace(/^\[AXROPUS\]\s*/, "")}
          </div>
        ))}
        {!lines.length ? <div className="line-normal">Waiting for deployment logs...</div> : null}
        <span className="terminal-cursor">▌</span>
      </div>
      <div className="terminal-meta">
        <span>Step {lastStep || "-"}/10</span>
        <span>{mode === "ws" ? "live stream" : "poll fallback"}</span>
        {wsError ? <span className="line-error">{wsError}</span> : null}
      </div>
    </div>
  );
}
