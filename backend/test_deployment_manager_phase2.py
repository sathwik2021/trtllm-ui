"""
Phase 2 tests for deployment_manager.py: preflight() and the
kv_cache_dtype hardware-capability gate in build_command(). Mocked
docker/nvidia-smi, same convention as test_deployment_manager_phase1.py.
"""
from __future__ import annotations

import sys
import unittest
from unittest.mock import patch, MagicMock

sys.path.insert(0, "..")
from backend import deployment_manager as dm
from backend.deployment_manager import DeploymentConfig
from backend.settings import AppSettings


FAKE_MODEL = {
    "name": "Qwen2.5-1.5B-Instruct",
    "host_path": "/mnt/d/models/Qwen2.5-1.5B-Instruct",
    "container_path": "/models/Qwen2.5-1.5B-Instruct",
    "size_bytes": 3_000_000_000,
}

AMPERE_MANIFEST = {
    "parsed": {
        "gpu": {"count": 1},
        "kv_cache_dtype_options": ["auto", "int8"],
        "parallel_max": 1,
        "serve_flags": [],
    }
}

FAKE_GPU_OK = {"gpus": [{"index": 0, "utilization_gpu": "5", "memory_used": "1000", "memory_total": "6144",
                          "temperature_gpu": "50", "power_draw": "20"}]}


class BuildCommandCapabilityGateTests(unittest.TestCase):
    def setUp(self):
        dm._deployments.clear()
        self.settings = AppSettings(data_dir="/tmp/trtllm-ui-test", model_dir="/tmp/models")

    def test_unsupported_kv_cache_dtype_blocked_when_manifest_known(self):
        with patch.object(dm, "get_model", return_value=FAKE_MODEL), \
             patch.object(dm, "cached_manifest", return_value=AMPERE_MANIFEST), \
             patch.object(dm, "_port_free", return_value=True):
            with self.assertRaises(ValueError) as ctx:
                dm.build_command(self.settings, DeploymentConfig(
                    model_name="Qwen2.5-1.5B-Instruct", port=8000, kv_cache_dtype="fp8",
                ))
        self.assertIn("kv_cache_dtype", str(ctx.exception))

    def test_supported_kv_cache_dtype_allowed(self):
        with patch.object(dm, "get_model", return_value=FAKE_MODEL), \
             patch.object(dm, "cached_manifest", return_value=AMPERE_MANIFEST), \
             patch.object(dm, "_port_free", return_value=True):
            cmd, _ = dm.build_command(self.settings, DeploymentConfig(
                model_name="Qwen2.5-1.5B-Instruct", port=8000, kv_cache_dtype="int8",
            ))
        self.assertIn("int8", cmd)

    def test_no_manifest_yet_does_not_block_deploy(self):
        # Can't judge hardware support without a probe -- must not fail closed.
        with patch.object(dm, "get_model", return_value=FAKE_MODEL), \
             patch.object(dm, "cached_manifest", return_value=None), \
             patch.object(dm, "_port_free", return_value=True):
            cmd, _ = dm.build_command(self.settings, DeploymentConfig(
                model_name="Qwen2.5-1.5B-Instruct", port=8000, kv_cache_dtype="fp8",
            ))
        self.assertIn("fp8", cmd)


class PreflightTests(unittest.TestCase):
    def setUp(self):
        dm._deployments.clear()
        self.settings = AppSettings(data_dir="/tmp/trtllm-ui-test", model_dir="/tmp/models")

    def _run(self, config, docker_rc=0, gpu=FAKE_GPU_OK, manifest=AMPERE_MANIFEST, port_free=True):
        with patch.object(dm, "get_model", return_value=FAKE_MODEL), \
             patch.object(dm, "cached_manifest", return_value=manifest), \
             patch.object(dm.gpu_monitor, "poll", return_value=gpu), \
             patch.object(dm, "_port_free", return_value=port_free), \
             patch("subprocess.run", return_value=MagicMock(returncode=docker_rc, stdout="28.0.0", stderr="")):
            return dm.preflight(self.settings, config)

    def test_all_pass_is_feasible(self):
        report = self._run(DeploymentConfig(model_name="Qwen2.5-1.5B-Instruct", port=8000, kv_cache_dtype="int8"))
        self.assertTrue(report["feasible"])
        self.assertTrue(all(c["status"] != "fail" for c in report["checks"]))
        self.assertIsNotNone(report["vram_estimate"])
        self.assertTrue(report["vram_estimate"]["heuristic"])

    def test_missing_model_reported_not_raised(self):
        with patch.object(dm, "get_model", side_effect=KeyError("Model not found: nope")), \
             patch.object(dm.gpu_monitor, "poll", return_value=FAKE_GPU_OK), \
             patch.object(dm, "cached_manifest", return_value=AMPERE_MANIFEST), \
             patch.object(dm, "_port_free", return_value=True), \
             patch("subprocess.run", return_value=MagicMock(returncode=0, stdout="28.0.0", stderr="")):
            report = dm.preflight(self.settings, DeploymentConfig(model_name="nope", port=8000))
        self.assertFalse(report["feasible"])
        model_check = next(c for c in report["checks"] if c["name"] == "model")
        self.assertEqual(model_check["status"], "fail")
        # No VRAM estimate possible without a resolved model.
        self.assertIsNone(report["vram_estimate"])

    def test_docker_unreachable_fails_feasible(self):
        report = self._run(DeploymentConfig(model_name="Qwen2.5-1.5B-Instruct", port=8000), docker_rc=1)
        self.assertFalse(report["feasible"])
        docker_check = next(c for c in report["checks"] if c["name"] == "docker")
        self.assertEqual(docker_check["status"], "fail")

    def test_gpu_unreachable_fails_feasible_and_skips_vram(self):
        report = self._run(DeploymentConfig(model_name="Qwen2.5-1.5B-Instruct", port=8000), gpu={"error": "no GPU"})
        self.assertFalse(report["feasible"])
        self.assertIsNone(report["vram_estimate"])

    def test_no_manifest_warns_but_does_not_fail(self):
        report = self._run(DeploymentConfig(model_name="Qwen2.5-1.5B-Instruct", port=8000), manifest=None)
        cap_check = next(c for c in report["checks"] if c["name"] == "capability_manifest")
        self.assertEqual(cap_check["status"], "warn")
        # warn alone shouldn't sink feasibility (docker/gpu/model/port all still pass)
        self.assertTrue(report["feasible"])

    def test_unsupported_kv_cache_dtype_fails_feasible(self):
        report = self._run(DeploymentConfig(model_name="Qwen2.5-1.5B-Instruct", port=8000, kv_cache_dtype="fp8"))
        self.assertFalse(report["feasible"])
        kv_check = next(c for c in report["checks"] if c["name"] == "kv_cache_dtype")
        self.assertEqual(kv_check["status"], "fail")

    def test_excessive_context_flags_vram_as_warn_not_silent(self):
        # Tiny free VRAM (200MB) vs a large context request should trip the
        # vram check to "warn" (not "fail" -- it's a heuristic, not a hard gate).
        tight_gpu = {"gpus": [{"index": 0, "utilization_gpu": "5", "memory_used": "5900",
                                "memory_total": "6144", "temperature_gpu": "50", "power_draw": "20"}]}
        report = self._run(
            DeploymentConfig(model_name="Qwen2.5-1.5B-Instruct", port=8000, max_batch_size=8, max_seq_len=16384),
            gpu=tight_gpu,
        )
        vram_check = next(c for c in report["checks"] if c["name"] == "vram")
        self.assertEqual(vram_check["status"], "warn")

    def test_port_taken_fails(self):
        report = self._run(DeploymentConfig(model_name="Qwen2.5-1.5B-Instruct", port=8000), port_free=False)
        self.assertFalse(report["feasible"])
        port_check = next(c for c in report["checks"] if c["name"] == "port")
        self.assertEqual(port_check["status"], "fail")


if __name__ == "__main__":
    unittest.main(verbosity=2)
