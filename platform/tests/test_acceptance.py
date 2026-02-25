import json
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from platform.artifacts.store import ArtifactStore
from platform.ledger import db as ledger
from platform.runtime.router import Router
from platform.runtime.worker import Worker


class FakeAdapter:
    backend_id = "fake"
    backend_version = "v1"

    def __init__(self, kv_replay: bool, decision: str, prompt_hash: str, job_id: str):
        self.kv_replay = kv_replay
        self.decision = decision
        self.prompt_hash = prompt_hash
        self.job_id = job_id

    def get_fingerprint(self):
        return {
            "model_hash": "fake-model",
            "tokenizer_hash": "fake-tokenizer",
            "backend_id": self.backend_id,
            "backend_version": self.backend_version,
        }

    def get_capabilities(self):
        class Caps:
            kv_replay = self.kv_replay
            deterministic_seeding = True
            streaming = False
            batch_prefill = False
            verify_tokens = False
        return Caps()

    def tokenize(self, prompt: str) -> int:
        return max(1, len(prompt.split()))

    def run_baseline(self, prompt: str, max_tokens: int, deterministic_cfg: dict, policy: dict, artifacts: dict, mf_snapshot_in=None):
        output = f"output for {self.prompt_hash}"
        Path(artifacts["output"]).write_text(output, encoding="utf-8")
        Path(artifacts["log"]).write_text("[FAKE]\n", encoding="utf-8")

        events = []
        if self.kv_replay:
            if self.decision == "hit":
                events.append({"type": "AMF_HIT", "ts": "now", "payload": {"job_id": self.job_id}})
            else:
                events.append({"type": "AMF_MISS", "ts": "now", "payload": {"job_id": self.job_id}})
            events.append({"type": "MF_APPLY", "ts": "now", "payload": {"job_id": self.job_id}})
        if artifacts.get("engine_events"):
            with Path(artifacts["engine_events"]).open("w", encoding="utf-8") as f:
                for evt in events:
                    f.write(json.dumps(evt) + "\n")

        amf = {
            "supported": self.kv_replay,
            "decision": self.decision if self.kv_replay else "unavailable",
            "prefix_len": 100 if self.decision == "hit" else 0,
            "skipped_tokens": 100 if self.decision == "hit" else 0,
            "skip_ratio": 0.5 if self.decision == "hit" else 0.0,
            "restore_ms": 5.0 if self.decision == "hit" else 0.0,
            "baseline_prefix_ms": 25.0 if self.decision == "hit" else 0.0,
            "saved_ms": 20.0 if self.decision == "hit" else 0.0,
            "roi": 4.0 if self.decision == "hit" else 0.0,
        }
        mf = {
            "supported": self.kv_replay,
            "min_admit_roi": 1.2,
            "eviction_pressure": 0.8,
            "replay_disable_mask": 0,
            "cooldown_ms": 0,
            "snapshot_id": self.job_id,
        }
        perf = {
            "tokens_out": 10,
            "total_ms": 50.0,
            "prefill_ms": 20.0,
            "decode_ms": 30.0,
            "avg_tps": 200.0,
        }
        return {
            "exit_code": 0,
            "total_ms": 50.0,
            "engine_metrics": {"amf": amf, "mf": mf, "perf": perf},
            "engine_events_path": artifacts["engine_events"],
        }


class TestRegistry:
    def __init__(self, adapter_factory):
        self.adapter_factory = adapter_factory

    def get_adapter(self, jobspec):
        return self.adapter_factory(jobspec)


def _prompt_hash(obj: dict) -> str:
    import hashlib
    prompt = obj.get("prompt") or obj.get("prompt_template") or ""
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def wait_status(db_path: Path, job_id: str, timeout_s: float = 5.0):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        rec = ledger.get_job(db_path, job_id)
        if rec and rec["status"] in ("SUCCEEDED", "FAILED"):
            return rec
        time.sleep(0.05)
    return None


class PlatformAcceptanceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        base = Path(self.tmp.name)
        self.db_path = base / "ledger.sqlite"
        self.artifacts = ArtifactStore(base / "artifacts")
        ledger.init_db(self.db_path)

    def tearDown(self):
        self.tmp.cleanup()

    def _build_runtime(self, adapter_factory):
        restore_store = type("RestoreStore", (), {"set": lambda *_: None, "get": lambda *_: None})()
        registry = TestRegistry(adapter_factory)
        worker = Worker(
            worker_id="worker-0",
            gpu_id=0,
            db_path=self.db_path,
            artifacts=self.artifacts,
            adapter_factory=registry.get_adapter,
            restore_store=restore_store,
        )
        worker.start()
        router = Router(
            db_path=self.db_path,
            artifacts=self.artifacts,
            workers=[worker],
            adapter_registry=registry,
            restore_store=restore_store,
        )
        return router, worker

    def test_cold_and_warm_runs(self):
        seen = set()

        def factory(job):
            prompt_hash = job.get("prompt_hash") or _prompt_hash(job)
            if "job_id" in job:
                decision = "hit" if prompt_hash in seen else "miss"
                seen.add(prompt_hash)
            else:
                decision = "hit" if prompt_hash in seen else "miss"
            return FakeAdapter(
                kv_replay=True,
                decision=decision,
                prompt_hash=prompt_hash,
                job_id=job.get("job_id", "unknown"),
            )

        router, worker = self._build_runtime(factory)
        jobspec = {
            "schema_version": "korith.jobspec.v1",
            "backend_id": "korith_local",
            "model": {"model_id": "fake", "model_path": "/tmp/fake"},
            "prompt": "hello world",
            "deterministic_cfg": {"seed": 1, "n_ctx": 1024, "n_batch": 64, "max_tokens": 16},
            "policy": {"allow_amf_reuse": True, "allow_spec": False},
        }
        job_id = router.submit(jobspec)
        rec = wait_status(self.db_path, job_id)
        self.assertIsNotNone(rec)
        run = ledger.get_latest_run(self.db_path, job_id)
        self.assertTrue(Path(run["metrics_path"]).exists())
        metrics = json.loads(Path(run["metrics_path"]).read_text(encoding="utf-8"))
        self.assertEqual(metrics["amf"]["decision"], "miss")
        self.assertGreaterEqual(metrics["scheduling"]["queue_latency_ms"], 0.0)
        self.assertEqual(metrics["scheduling"]["lane"], "MISS")
        events = Path(run["events_path"]).read_text(encoding="utf-8")
        self.assertIn("AMF_MISS", events)

        job_id2 = router.submit(jobspec)
        rec2 = wait_status(self.db_path, job_id2)
        self.assertIsNotNone(rec2)
        run2 = ledger.get_latest_run(self.db_path, job_id2)
        metrics2 = json.loads(Path(run2["metrics_path"]).read_text(encoding="utf-8"))
        self.assertEqual(metrics2["amf"]["decision"], "hit")
        self.assertGreater(metrics2["amf"]["skipped_tokens"], 0)
        self.assertGreater(metrics2["amf"]["roi"], 1.0)
        self.assertEqual(metrics2["scheduling"]["lane"], "HIT")
        events2 = Path(run2["events_path"]).read_text(encoding="utf-8")
        self.assertIn("AMF_HIT", events2)
        worker.stop()

    def test_backend_without_kv_replay(self):
        def factory(job):
            prompt_hash = job.get("prompt_hash") or _prompt_hash(job)
            return FakeAdapter(
                kv_replay=False,
                decision="unavailable",
                prompt_hash=prompt_hash,
                job_id=job.get("job_id", "unknown"),
            )

        router, worker = self._build_runtime(factory)
        jobspec = {
            "schema_version": "korith.jobspec.v1",
            "backend_id": "openai_compatible",
            "model": {"model_id": "fake", "endpoint": "http://localhost"},
            "prompt": "baseline only",
            "deterministic_cfg": {"seed": 1, "n_ctx": 1024, "n_batch": 64, "max_tokens": 8},
            "policy": {"allow_amf_reuse": True, "allow_spec": False},
        }
        job_id = router.submit(jobspec)
        rec = wait_status(self.db_path, job_id)
        self.assertIsNotNone(rec)
        run = ledger.get_latest_run(self.db_path, job_id)
        metrics = json.loads(Path(run["metrics_path"]).read_text(encoding="utf-8"))
        self.assertFalse(metrics["amf"]["supported"])
        self.assertEqual(metrics["amf"]["decision"], "unavailable")
        events = Path(run["events_path"]).read_text(encoding="utf-8")
        self.assertIn("AMF_BLOCK", events)
        worker.stop()

    def test_restore_mismatch_fail_closed(self):
        def factory(job):
            prompt_hash = job.get("prompt_hash") or _prompt_hash(job)
            return FakeAdapter(
                kv_replay=True,
                decision="miss",
                prompt_hash=prompt_hash,
                job_id=job.get("job_id", "unknown"),
            )

        router, worker = self._build_runtime(factory)
        jobspec = {
            "schema_version": "korith.jobspec.v1",
            "backend_id": "korith_local",
            "model": {"model_id": "fake", "model_path": "/tmp/fake"},
            "prompt": "restore test",
            "deterministic_cfg": {"seed": 1, "n_ctx": 1024, "n_batch": 64, "max_tokens": 8},
            "policy": {"allow_amf_reuse": True, "allow_spec": False},
        }
        job_id = router.submit(jobspec)
        rec = wait_status(self.db_path, job_id)
        self.assertIsNotNone(rec)

        ledger.insert_snapshot(
            self.db_path,
            snapshot_id="snap-mismatch",
            job_id=job_id,
            fingerprint_hash="mismatch",
            snapshot_path=str(self.artifacts.init_job(job_id)["mf_snapshot"]),
            created_at="now",
        )
        res = router.restore(job_id)
        self.assertFalse(res["restored"])
        run = ledger.get_latest_run(self.db_path, job_id)
        if run:
            events = Path(run["events_path"]).read_text(encoding="utf-8")
            self.assertIn("MF_RESTORE_FAILED", events)
        worker.stop()


if __name__ == "__main__":
    unittest.main()
