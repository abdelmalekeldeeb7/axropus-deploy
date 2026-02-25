from __future__ import annotations

from pathlib import Path
from typing import Dict


class ArtifactStore:
    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def init_job(self, job_id: str, org_id: str = "default") -> Dict[str, Path]:
        job_dir = self.base_dir / org_id / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        return {
            "job_dir": job_dir,
            "metrics": job_dir / "metrics.json",
            "events": job_dir / "events.jsonl",
            "output": job_dir / "output.txt",
            "log": job_dir / "run.log",
            "mf_snapshot": job_dir / "mf_snapshot.json",
            "engine_metrics": job_dir / "engine_metrics.json",
            "engine_events": job_dir / "engine_events.jsonl",
        }

    def append_event(self, events_path: Path, event: dict) -> None:
        import json
        with events_path.open("a", encoding="utf-8") as f:
            if isinstance(event, str):
                f.write(event + "\n")
            else:
                f.write(json.dumps(event, ensure_ascii=False) + "\n")
