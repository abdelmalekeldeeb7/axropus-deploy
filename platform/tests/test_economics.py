import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from platform.ledger import db as ledger
from platform.economics.model import estimate_savings
from platform.economics.report import generate_report


class EconomicsReportTests(unittest.TestCase):
    def test_estimate_savings_uses_canonical_components(self):
        metrics = {
            "amf": {"saved_ms": 10.0},
            "spec": {"saved_ms": 30.0},
            "kernels": {"ms_saved": 12.0, "kernels_applied": True, "comparable": True},
            "savings": {
                "prefill_saved_ms": 10.0,
                "spec_saved_ms": 20.0,
                "kernels_saved_ms": 5.0,
                "total_saved_ms": 35.0,
            },
            "perf": {"total_ms": 100.0},
        }
        out = estimate_savings(metrics, gpu_hourly_cost=3.6)
        self.assertEqual(float(out["saved_ms"]), 35.0)
        self.assertEqual(float(out["prefill_saved_ms"]), 10.0)
        self.assertEqual(float(out["spec_saved_ms"]), 20.0)
        self.assertEqual(float(out["kernels_saved_ms"]), 5.0)

    def test_estimate_savings_fallback_zeroes_unapplied_kernel_credit(self):
        metrics = {
            "amf": {"saved_ms": 10.0},
            "spec": {"saved_ms": 25.0},
            "kernels": {"ms_saved": 7.0, "kernels_applied": False, "comparable": True},
            "perf": {"total_ms": 100.0},
        }
        out = estimate_savings(metrics, gpu_hourly_cost=3.6)
        self.assertEqual(float(out["saved_ms"]), 35.0)
        self.assertEqual(float(out["kernels_saved_ms"]), 0.0)

    def test_generate_report_outputs(self):
        with TemporaryDirectory() as tmp:
            base = Path(tmp)
            db_path = base / "ledger.sqlite"
            artifacts = base / "artifacts"
            job_dir = artifacts / "job-1"
            job_dir.mkdir(parents=True, exist_ok=True)

            ledger.init_db(db_path)
            ledger.insert_job(
                db_path=db_path,
                job_id="job-1",
                created_at="2026-01-01T00:00:00Z",
                jobspec={
                    "schema_version": "korith.jobspec.v1",
                    "workload": {"type": "ticket_triage"},
                },
                fingerprint={},
                prompt_hash="p",
                fingerprint_hash="f",
                status="SUCCEEDED",
            )

            metrics = {
                "ids": {"job_id": "job-1", "finished_at": "2026-01-01T00:01:00Z"},
                "scheduling": {"gpu_id": 0},
                "amf": {"saved_ms": 120.0, "restore_ms": 10.0, "skipped_tokens": 64},
                "perf": {"total_ms": 300.0, "tokens_out": 20},
            }
            metrics_path = job_dir / "metrics.json"
            metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
            ledger.insert_run(
                db_path=db_path,
                run_id="run-1",
                job_id="job-1",
                started_at="2026-01-01T00:00:00Z",
                finished_at="2026-01-01T00:01:00Z",
                exit_code=0,
                metrics_path=str(metrics_path),
                events_path=str(job_dir / "events.jsonl"),
                output_path=str(job_dir / "output.txt"),
                log_path=str(job_dir / "run.log"),
            )

            out = base / "savings_report.json"
            generate_report(db_path, out, 2.5)

            self.assertTrue(out.exists())
            self.assertTrue((base / "savings_report.csv").exists())
            self.assertTrue((base / "executive_summary.pdf").exists())
            data = json.loads(out.read_text(encoding="utf-8"))
            self.assertIn("summary", data)
            self.assertIn("per_gpu_savings", data)


if __name__ == "__main__":
    unittest.main()
