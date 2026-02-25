import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from platform.ledger import db as ledger
from platform.economics.targets import generate_target_report


class EconomicsTargetReportTests(unittest.TestCase):
    def test_target_report_computes_decode_gap(self):
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
                    "owner": {"org_id": "pilot_a"},
                },
                fingerprint={},
                prompt_hash="p",
                fingerprint_hash="f",
                status="SUCCEEDED",
                org_id="pilot_a",
            )

            metrics = {
                "ids": {"job_id": "job-1", "finished_at": "2026-01-01T00:01:00Z", "org_id": "pilot_a"},
                "scheduling": {"gpu_id": 0, "queue_latency_ms": 5.0},
                "amf": {"saved_ms": 40.0, "restore_ms": 5.0, "skipped_tokens": 64},
                "spec": {"saved_ms": 10.0},
                "kernels": {"ms_saved": 5.0, "kernels_applied": True, "comparable": True},
                "perf": {"total_ms": 200.0, "prefill_ms": 20.0, "decode_ms": 160.0, "tokens_out": 20},
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
                org_id="pilot_a",
            )

            out = base / "target_report.json"
            report = generate_target_report(
                db_path=db_path,
                out_path=out,
                targets_csv="0.5,0.6,0.7",
                org_id="pilot_a",
                limit=100,
            )
            self.assertTrue(out.exists())
            self.assertEqual(report["summary"]["jobs_analyzed"], 1)
            self.assertGreater(report["summary"]["current_savings_pct"], 0.0)
            self.assertEqual(len(report["targets"]), 3)
            # Higher target should require at least as much decode cut.
            t50 = report["targets"][0]["required_decode_cut_pct"]
            t70 = report["targets"][2]["required_decode_cut_pct"]
            self.assertGreaterEqual(t70, t50)

    def test_target_report_handles_no_metrics(self):
        with TemporaryDirectory() as tmp:
            base = Path(tmp)
            db_path = base / "ledger.sqlite"
            ledger.init_db(db_path)
            out = base / "target_report.json"
            report = generate_target_report(
                db_path=db_path,
                out_path=out,
                targets_csv="0.5",
                org_id=None,
                limit=10,
            )
            self.assertTrue(out.exists())
            self.assertEqual(report["summary"]["jobs_analyzed"], 0)
            self.assertEqual(report["summary"]["error"], "no_usable_metrics")

    def test_target_report_prefers_canonical_savings_components(self):
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
                    "owner": {"org_id": "pilot_a"},
                },
                fingerprint={},
                prompt_hash="p",
                fingerprint_hash="f",
                status="SUCCEEDED",
                org_id="pilot_a",
            )

            metrics = {
                "ids": {"job_id": "job-1", "finished_at": "2026-01-01T00:01:00Z", "org_id": "pilot_a"},
                "scheduling": {"gpu_id": 0, "queue_latency_ms": 5.0},
                "amf": {"saved_ms": 40.0},
                # Legacy raw fields would imply decode_saved=35.0 if naively summed.
                "spec": {"saved_ms": 30.0},
                "kernels": {"ms_saved": 5.0, "kernels_applied": True, "comparable": True},
                "savings": {
                    "prefill_saved_ms": 40.0,
                    "spec_saved_ms": 10.0,
                    "kernels_saved_ms": 5.0,
                    "total_saved_ms": 55.0,
                },
                "perf": {"total_ms": 200.0, "prefill_ms": 20.0, "decode_ms": 160.0, "tokens_out": 20},
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
                org_id="pilot_a",
            )

            out = base / "target_report.json"
            report = generate_target_report(
                db_path=db_path,
                out_path=out,
                targets_csv="0.5",
                org_id="pilot_a",
                limit=100,
            )
            self.assertEqual(report["summary"]["current_total_saved_ms"], 55.0)


if __name__ == "__main__":
    unittest.main()
