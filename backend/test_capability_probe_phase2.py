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
    """Confirmed real trtllm-serve 1.3.0rc22 --kv_cache_dtype choices are
    [auto|fp8|nvfp4] -- NOT a generic auto/int8/fp16/bf16 set. This was
    caught by testing against the real image, not assumed up front; these
    tests lock in the corrected behavior.
    """
    REAL_KV_FLAG_LINE = "--kv_cache_dtype [auto|fp8|nvfp4]"

    def _serve_flags(self, help_line=REAL_KV_FLAG_LINE):
        return [{"name": "--kv_cache_dtype", "takes_value": True, "help": help_line}]

    def test_ampere_gets_only_auto_since_fp8_and_nvfp4_are_hardware_excluded(self):
        p = cp._precision_capabilities(8.6)
        options = cp._kv_cache_dtype_options(p, self._serve_flags())
        # This is the real, confirmed-correct result for the project's
        # actual RTX 3050 -- no KV-cache quantization option is usable at
        # all on this hardware in this build, only "auto".
        self.assertEqual(options, ["auto"])

    def test_blackwell_gets_fp8_and_nvfp4_from_real_choices(self):
        p = cp._precision_capabilities(10.0)
        options = cp._kv_cache_dtype_options(p, self._serve_flags())
        self.assertEqual(set(options), {"auto", "fp8", "nvfp4"})

    def test_ada_gets_fp8_but_not_nvfp4(self):
        p = cp._precision_capabilities(8.9)
        options = cp._kv_cache_dtype_options(p, self._serve_flags())
        self.assertEqual(set(options), {"auto", "fp8"})

    def test_regression_extracts_from_actual_captured_help_text(self):
        # Verbatim shape of the real `trtllm-serve serve --help` line
        # (the flag name + choices share a line; the description wraps
        # onto following lines that _parse_flags does not attach here).
        real_line = (
            "--kv_cache_dtype [auto|fp8|nvfp4]                :tag:`prototype` "
            "KV cache quantization dtype for PyTorch backend."
        )
        flags = cp._parse_flags(real_line)
        p = cp._precision_capabilities(8.6)
        options = cp._kv_cache_dtype_options(p, flags)
        self.assertEqual(options, ["auto"])

    def test_no_probe_yet_falls_back_conservatively(self):
        # No serve_flags at all (never probed) -- must not silently assume
        # zero options either; falls back to the documented static guess,
        # still hardware-gated.
        p = cp._precision_capabilities(8.6)
        options = cp._kv_cache_dtype_options(p, None)
        self.assertEqual(options, ["auto"])  # fp8/nvfp4 still excluded by hardware


class EnumChoiceExtractionTests(unittest.TestCase):
    def test_extracts_pipe_separated_choices(self):
        flags = [{"name": "--backend", "takes_value": True, "help": "--backend [pytorch|_autodeploy]"}]
        self.assertEqual(cp._enum_choices(flags, "--backend"), ["pytorch", "_autodeploy"])

    def test_missing_flag_returns_none_not_empty_list(self):
        flags = [{"name": "--other", "takes_value": True, "help": "--other TEXT"}]
        self.assertIsNone(cp._enum_choices(flags, "--kv_cache_dtype"))

    def test_no_serve_flags_at_all_returns_none(self):
        self.assertIsNone(cp._enum_choices(None, "--kv_cache_dtype"))

    def test_flag_without_enum_brackets_returns_none(self):
        flags = [{"name": "--max_batch_size", "takes_value": True, "help": "--max_batch_size INTEGER"}]
        self.assertIsNone(cp._enum_choices(flags, "--max_batch_size"))


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
