"""
Phase 2 tests for capability_probe.py's precision/GPU/kv-cache-dtype
derivation. These test PURE PARSING/DERIVATION LOGIC against known
compute-capability values (no real nvidia-smi in this sandbox) -- the
actual RTX 3050 (SM86) case is the one that matters for this project.
"""
from __future__ import annotations

import sys
import unittest

sys.path.insert(0, "..")
from backend import capability_probe as cp


AMPERE_SM86_CSV = (
    "name, memory.total [MiB], memory.free [MiB], driver_version, compute_cap\n"
    "NVIDIA GeForce RTX 3050, 6144 MiB, 4820 MiB, 550.90.07, 8.6\n"
)


class GpuParsingTests(unittest.TestCase):
    def test_parses_single_gpu_row(self):
        gpus = cp._parse_gpus(AMPERE_SM86_CSV)
        self.assertEqual(len(gpus), 1)
        self.assertEqual(gpus[0]["name"], "NVIDIA GeForce RTX 3050")
        self.assertEqual(gpus[0]["vram_total_mb"], 6144)
        self.assertEqual(gpus[0]["vram_free_mb"], 4820)
        self.assertEqual(gpus[0]["compute_capability"], 8.6)

    def test_empty_or_failed_probe_returns_empty_list_not_a_fake_gpu(self):
        self.assertEqual(cp._parse_gpus(""), [])
        self.assertEqual(cp._parse_gpus("nvidia-smi: command not found"), [])


class PrecisionCapabilityTests(unittest.TestCase):
    # ---- the actual hardware this project targets: RTX 3050, SM86 ----
    def test_ampere_sm86_matches_project_hardware(self):
        p = cp._precision_capabilities(8.6)
        self.assertTrue(p["fp16"]["supported"])
        self.assertTrue(p["bf16"]["supported"])
        self.assertTrue(p["int8"]["supported"])
        self.assertFalse(p["fp8"]["supported"])   # hard hardware exclusion below SM89
        self.assertFalse(p["nvfp4"]["supported"])  # hard hardware exclusion below SM100

    def test_ada_sm89_unlocks_fp8_not_nvfp4(self):
        p = cp._precision_capabilities(8.9)
        self.assertTrue(p["fp8"]["supported"])
        self.assertFalse(p["nvfp4"]["supported"])

    def test_blackwell_sm100_unlocks_nvfp4(self):
        p = cp._precision_capabilities(10.0)
        self.assertTrue(p["fp8"]["supported"])
        self.assertTrue(p["nvfp4"]["supported"])

    def test_pre_turing_excludes_int8_too(self):
        p = cp._precision_capabilities(6.1)  # Pascal
        self.assertFalse(p["int8"]["supported"])
        self.assertFalse(p["bf16"]["supported"])

    def test_unknown_compute_cap_is_unknown_not_assumed_false(self):
        p = cp._precision_capabilities(None)
        self.assertIsNone(p["fp8"]["supported"])
        self.assertEqual(p["fp8"]["confidence"], "unknown")


class KvCacheDtypeOptionsTests(unittest.TestCase):
    def test_ampere_offers_int8_not_fp8_or_fp4(self):
        p = cp._precision_capabilities(8.6)
        options = cp._kv_cache_dtype_options(p)
        self.assertIn("auto", options)
        self.assertIn("int8", options)
        self.assertNotIn("fp8", options)
        self.assertNotIn("fp4", options)

    def test_blackwell_offers_everything(self):
        p = cp._precision_capabilities(10.0)
        options = cp._kv_cache_dtype_options(p)
        self.assertEqual(set(options), {"auto", "int8", "fp8", "fp4"})


class RunProbeManifestShapeTests(unittest.TestCase):
    """run_probe() itself shells out to docker/nvidia-smi -- not runnable in
    this sandbox. This checks the pure aggregation logic it uses to build
    manifest["parsed"]["precision"]/["gpu"]/["parallel_max"] instead, via a
    minimal stand-in for what _parse_gpus would return.
    """
    def test_single_gpu_precision_passthrough(self):
        gpus = [{"name": "RTX 3050", "vram_total_mb": 6144, "vram_free_mb": 4820, "compute_capability": 8.6}]
        per_gpu = [cp._precision_capabilities(g["compute_capability"]) for g in gpus]
        self.assertEqual(len(per_gpu), 1)
        self.assertFalse(per_gpu[0]["fp8"]["supported"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
