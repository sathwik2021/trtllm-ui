"""
Phase 2 tests for vram_estimator.py. These check the arithmetic and the
detailed-vs-fallback method selection -- NOT real-world VRAM accuracy,
which the module itself is explicit about not guaranteeing (see its
docstring). A wrong number here would be a real bug; the module being
"only approximate" by design is not.
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, "..")
from backend import vram_estimator as ve


QWEN_1_5B_LIKE_CONFIG = {
    "num_hidden_layers": 28,
    "hidden_size": 1536,
    "num_attention_heads": 12,
    "num_key_value_heads": 2,  # GQA
}


class VramEstimatorTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.model_dir = Path(self.tmp.name)
        (self.model_dir / "config.json").write_text(json.dumps(QWEN_1_5B_LIKE_CONFIG))
        self.model = {
            "name": "test-model",
            "host_path": str(self.model_dir),
            "size_bytes": 3_000_000_000,  # ~3GB on disk, roughly fp16 1.5B params
        }

    def tearDown(self):
        self.tmp.cleanup()

    def test_uses_detailed_method_when_config_has_architecture_fields(self):
        result = ve.estimate(self.model, max_batch_size=1, max_seq_len=2048)
        self.assertEqual(result["method"], "detailed")
        self.assertTrue(result["heuristic"])

    def test_weights_estimate_applies_overhead_factor(self):
        result = ve.estimate(self.model, max_batch_size=1, max_seq_len=2048)
        expected_mb = (3_000_000_000 / (1024 * 1024)) * 1.2
        self.assertAlmostEqual(result["weights_mb"], expected_mb, places=1)

    def test_kv_cache_scales_with_batch_size(self):
        r1 = ve.estimate(self.model, max_batch_size=1, max_seq_len=2048)
        r4 = ve.estimate(self.model, max_batch_size=4, max_seq_len=2048)
        self.assertAlmostEqual(r4["kv_cache_mb"], r1["kv_cache_mb"] * 4, places=1)

    def test_kv_cache_scales_with_context_plus_output_length(self):
        short = ve.estimate(self.model, max_batch_size=1, max_seq_len=2048, max_output_tokens=0)
        long = ve.estimate(self.model, max_batch_size=1, max_seq_len=2048, max_output_tokens=2048)
        self.assertAlmostEqual(long["kv_cache_mb"], short["kv_cache_mb"] * 2, places=1)

    def test_fp8_kv_cache_dtype_halves_bytes_vs_fp16(self):
        fp16 = ve.estimate(self.model, max_batch_size=1, max_seq_len=2048, kv_cache_dtype="auto")
        int8 = ve.estimate(self.model, max_batch_size=1, max_seq_len=2048, kv_cache_dtype="int8")
        self.assertAlmostEqual(int8["kv_cache_mb"], fp16["kv_cache_mb"] / 2, places=1)

    def test_falls_back_when_config_missing_architecture_fields(self):
        (self.model_dir / "config.json").write_text(json.dumps({"model_type": "qwen2"}))
        result = ve.estimate(self.model, max_batch_size=1, max_seq_len=2048)
        self.assertEqual(result["method"], "fallback")
        self.assertTrue(any("falling back" in n for n in result["notes"]))

    def test_missing_size_bytes_flags_unreliable_not_silent(self):
        model = dict(self.model, size_bytes=0)
        result = ve.estimate(model, max_batch_size=1, max_seq_len=2048)
        self.assertEqual(result["weights_mb"], 0)
        self.assertTrue(any("unreliable" in n for n in result["notes"]))

    def test_result_always_labeled_heuristic(self):
        result = ve.estimate(self.model, max_batch_size=1, max_seq_len=2048)
        self.assertIs(result["heuristic"], True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
