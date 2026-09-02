"""
Phase 3 tests for benchmark_manager.py. HTTP calls and GPU polling are
mocked -- no real deployment or nvidia-smi needed. Focus is on the
aggregation math (throughput, percentiles, GPU min/max/avg) and the
async-job lifecycle, since those are the parts most likely to have a
silent correctness bug.
"""
from __future__ import annotations

import json
import sys
import tempfile
import time
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, "..")
from backend import benchmark_manager as bm
from backend.settings import AppSettings


def _fake_response(completion_tokens=50, prompt_tokens=10):
    body = json.dumps({
        "choices": [{"message": {"content": "hi"}}],
        "usage": {"completion_tokens": completion_tokens, "prompt_tokens": prompt_tokens},
    }).encode("utf-8")
    resp = MagicMock()
    resp.read.return_value = body
    resp.__enter__ = lambda s: resp
    resp.__exit__ = lambda s, *a: False
    return resp


class PercentileTests(unittest.TestCase):
    def test_empty_list_returns_none_not_zero(self):
        self.assertIsNone(bm._percentile([], 0.5))

    def test_single_value(self):
        self.assertEqual(bm._percentile([5.0], 0.5), 5.0)
        self.assertEqual(bm._percentile([5.0], 0.95), 5.0)

    def test_p50_of_sorted_range(self):
        vals = [float(i) for i in range(1, 11)]
        p50 = bm._percentile(vals, 0.5)
        self.assertAlmostEqual(p50, 5.5, places=1)

    def test_p95_is_near_high_end(self):
        vals = [float(i) for i in range(1, 101)]
        p95 = bm._percentile(vals, 0.95)
        self.assertGreater(p95, 90)


class OneRequestTests(unittest.TestCase):
    def test_successful_request_reports_latency_and_tokens(self):
        with patch.object(bm, "urlopen", return_value=_fake_response(completion_tokens=42)):
            result = bm._one_request("http://x:8000", "model", "prompt", 100, 30)
        self.assertIsNone(result["error"])
        self.assertEqual(result["completion_tokens"], 42)
        self.assertGreaterEqual(result["latency_s"], 0)

    def test_failed_request_reports_error_not_raise(self):
        with patch.object(bm, "urlopen", side_effect=TimeoutError("timed out")):
            result = bm._one_request("http://x:8000", "model", "prompt", 100, 30)
        self.assertIsNotNone(result["error"])
        self.assertIsNone(result["completion_tokens"])


class RunBenchmarkAggregationTests(unittest.TestCase):
    def test_throughput_matches_tokens_over_wall_time(self):
        with patch.object(bm, "urlopen", return_value=_fake_response(completion_tokens=100)), \
             patch.object(bm.gpu_monitor, "poll", return_value={"gpus": []}):
            report = bm.run_benchmark(
                host="127.0.0.1", port=8000, served_model_name="m",
                request_count=5, concurrency=5, max_tokens=50,
            )
        self.assertEqual(report["requests_ok"], 5)
        self.assertEqual(report["requests_failed"], 0)
        self.assertEqual(report["total_completion_tokens"], 500)
        expected_throughput = 500 / report["wall_time_s"]
        self.assertAlmostEqual(report["throughput_tokens_per_s"], expected_throughput, places=3)

    def test_mixed_success_and_failure_counts_correctly(self):
        call_count = {"n": 0}

        def flaky_urlopen(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] % 2 == 0:
                raise ConnectionError("refused")
            return _fake_response(completion_tokens=10)

        with patch.object(bm, "urlopen", side_effect=flaky_urlopen), \
             patch.object(bm.gpu_monitor, "poll", return_value={"gpus": []}):
            report = bm.run_benchmark(
                host="127.0.0.1", port=8000, served_model_name="m",
                request_count=10, concurrency=2, max_tokens=50,
            )
        self.assertEqual(report["requests_ok"] + report["requests_failed"], 10)
        self.assertEqual(len(report["errors"]), min(report["requests_failed"], 10))

    def test_zero_successful_requests_does_not_crash_on_throughput(self):
        with patch.object(bm, "urlopen", side_effect=ConnectionError("down")), \
             patch.object(bm.gpu_monitor, "poll", return_value={"gpus": []}):
            report = bm.run_benchmark(
                host="127.0.0.1", port=8000, served_model_name="m",
                request_count=3, concurrency=3, max_tokens=50,
            )
        self.assertEqual(report["requests_ok"], 0)
        self.assertEqual(report["total_completion_tokens"], 0)
        self.assertIsNotNone(report["throughput_tokens_per_s"])
        self.assertEqual(report["latency_s"]["p50"], None)

    def test_gpu_samples_captured_and_aggregated(self):
        samples = [
            {"gpus": [{"utilization_gpu": "50", "memory_used": "3000"}]},
            {"gpus": [{"utilization_gpu": "80", "memory_used": "4000"}]},
        ]
        with patch.object(bm, "urlopen", return_value=_fake_response()), \
             patch.object(bm.gpu_monitor, "poll", side_effect=samples * 5):
            report = bm.run_benchmark(
                host="127.0.0.1", port=8000, served_model_name="m",
                request_count=2, concurrency=1, max_tokens=10,
            )
        self.assertGreaterEqual(report["gpu"]["samples_captured"], 0)
        util = report["gpu"]["utilization_gpu_pct"]
        if util["min"] is not None:
            self.assertLessEqual(util["min"], util["max"])

    def test_result_always_reports_config_used(self):
        with patch.object(bm, "urlopen", return_value=_fake_response()), \
             patch.object(bm.gpu_monitor, "poll", return_value={"gpus": []}):
            report = bm.run_benchmark(
                host="127.0.0.1", port=9001, served_model_name="qwen",
                request_count=1, concurrency=1, max_tokens=77, prompt="custom prompt",
            )
        self.assertEqual(report["config"]["port"], 9001)
        self.assertEqual(report["config"]["served_model_name"], "qwen")
        self.assertEqual(report["config"]["max_tokens"], 77)
        self.assertEqual(report["config"]["prompt"], "custom prompt")


class PersistenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.settings = AppSettings(data_dir=self.tmp.name, model_dir=self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_save_and_get_round_trip(self):
        report = {"id": "abc123", "wall_time_s": 1.0}
        bm.save_result(self.settings, report)
        loaded = bm.get_result(self.settings, "abc123")
        self.assertEqual(loaded["id"], "abc123")

    def test_get_missing_raises_keyerror(self):
        with self.assertRaises(KeyError):
            bm.get_result(self.settings, "does-not-exist")

    def test_list_results_sorted_newest_first(self):
        bm.save_result(self.settings, {"id": "first", "wall_time_s": 1.0})
        time.sleep(0.01)
        bm.save_result(self.settings, {"id": "second", "wall_time_s": 1.0})
        results = bm.list_results(self.settings)
        ids = [r["id"] for r in results]
        self.assertEqual(ids[0], "second")

    def test_delete_removes_file(self):
        bm.save_result(self.settings, {"id": "to-delete", "wall_time_s": 1.0})
        bm.delete_result(self.settings, "to-delete")
        with self.assertRaises(KeyError):
            bm.get_result(self.settings, "to-delete")

    def test_delete_nonexistent_does_not_raise(self):
        bm.delete_result(self.settings, "never-existed")


class JobLifecycleTests(unittest.TestCase):
    def setUp(self):
        bm._current_job = None
        self.tmp = tempfile.TemporaryDirectory()
        self.settings = AppSettings(data_dir=self.tmp.name, model_dir=self.tmp.name)

    def tearDown(self):
        bm._current_job = None
        self.tmp.cleanup()

    def test_start_job_returns_immediately_with_started_status(self):
        with patch.object(bm, "run_benchmark", return_value={"id": "x", "wall_time_s": 1.0}), \
             patch.object(bm, "save_result"):
            result = bm.start_benchmark_job(self.settings, host="h", port=1, served_model_name="m")
        self.assertEqual(result["status"], "started")
        self.assertIn("id", result)

    def test_job_reaches_done_status_after_completion(self):
        with patch.object(bm, "run_benchmark", return_value={"id": "x", "wall_time_s": 1.0}), \
             patch.object(bm, "save_result"):
            bm.start_benchmark_job(self.settings, host="h", port=1, served_model_name="m")
            for _ in range(50):
                if bm.current_job() and bm.current_job()["status"] != "running":
                    break
                time.sleep(0.02)
        self.assertEqual(bm.current_job()["status"], "done")

    def test_second_job_rejected_while_one_running(self):
        block = MagicMock()
        block.side_effect = lambda **kw: time.sleep(0.3) or {"id": "x", "wall_time_s": 1.0}
        with patch.object(bm, "run_benchmark", block), patch.object(bm, "save_result"):
            bm.start_benchmark_job(self.settings, host="h", port=1, served_model_name="m")
            with self.assertRaises(RuntimeError):
                bm.start_benchmark_job(self.settings, host="h", port=1, served_model_name="m")

    def test_job_error_is_captured_not_raised(self):
        with patch.object(bm, "run_benchmark", side_effect=RuntimeError("boom")):
            bm.start_benchmark_job(self.settings, host="h", port=1, served_model_name="m")
            for _ in range(50):
                job = bm.current_job()
                if job and job["status"] != "running":
                    break
                time.sleep(0.02)
        job = bm.current_job()
        self.assertEqual(job["status"], "error")
        self.assertIn("boom", job["error"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
