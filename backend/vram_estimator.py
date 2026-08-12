from __future__ import annotations

"""
VRAM pre-flight heuristic. Deliberately conservative and explicitly labeled
as approximate everywhere it surfaces -- this is NOT a TensorRT-LLM memory
simulator. It exists to catch obviously-doomed configurations (e.g. 16k
context with a 4GB free-VRAM budget) before `docker run` is even attempted,
not to give an exact byte count. The real TRT-LLM runtime remains the final
authority; a config that passes this check can still OOM at load time, and
callers must present the result that way.
"""

import json
from pathlib import Path
from typing import Any

# bytes-per-element for the dtypes trtllm-serve's --kv_cache_dtype accepts.
# "auto" falls back to the model's own weight dtype (assumed fp16/bf16 == 2 bytes,
# since that's the confirmed baseline this app targets -- see model_manager torch_dtype).
_KV_DTYPE_BYTES = {
    "auto": 2.0,
    "fp16": 2.0,
    "bf16": 2.0,
    "int8": 1.0,
    "fp8": 1.0,
    "fp4": 0.5,
}

# Fixed overhead multiplier on top of raw weight size, covering CUDA context,
# activation buffers, framework bookkeeping. Not measured on this hardware --
# a round, documented guess, not a derived constant.
_WEIGHT_OVERHEAD_FACTOR = 1.2

# Fallback KV-cache-as-fraction-of-weights, used ONLY when the model's
# config.json doesn't expose enough architecture fields to compute KV cache
# directly. Coarser than the detailed path -- flagged as such in the result.
_KV_FALLBACK_FRACTION = 0.15


def _read_config(host_path: str) -> dict[str, Any]:
    try:
        return json.loads((Path(host_path) / "config.json").read_text(encoding="utf-8"))
    except Exception:
        return {}


def estimate(
    model: dict[str, Any],
    max_batch_size: int | None,
    max_seq_len: int | None,
    max_output_tokens: int | None = None,
    kv_cache_dtype: str | None = None,
) -> dict[str, Any]:
    """Rough VRAM estimate in MB for a deployment config against a given model.

    Returns a dict always containing `heuristic: True` and a `notes` list --
    callers (API + UI) must surface both, not just the numbers.
    """
    notes: list[str] = []
    batch = max(1, max_batch_size or 1)
    context_len = max(1, max_seq_len or 2048)
    output_len = max(0, max_output_tokens or 0)
    seq_positions = context_len + output_len

    size_bytes = model.get("size_bytes") or 0
    weights_mb = (size_bytes / (1024 * 1024)) * _WEIGHT_OVERHEAD_FACTOR
    if size_bytes == 0:
        notes.append("model size_bytes unavailable -- weights estimate is 0, treat result as unreliable")

    kv_bytes_per_elem = _KV_DTYPE_BYTES.get((kv_cache_dtype or "auto").lower(), 2.0)
    config = _read_config(model.get("host_path", ""))
    num_layers = config.get("num_hidden_layers")
    hidden_size = config.get("hidden_size")
    num_heads = config.get("num_attention_heads")
    num_kv_heads = config.get("num_key_value_heads") or num_heads

    method = "fallback"
    if num_layers and hidden_size and num_heads and num_kv_heads:
        head_dim = hidden_size / num_heads
        # 2x for K and V, per layer, per KV head, per position, per batch item.
        kv_bytes = 2 * num_layers * num_kv_heads * head_dim * kv_bytes_per_elem * seq_positions * batch
        kv_cache_mb = kv_bytes / (1024 * 1024)
        method = "detailed"
    else:
        notes.append(
            "config.json missing num_hidden_layers/hidden_size/num_attention_heads -- "
            "falling back to a coarse weights-fraction KV estimate, wider error margin than usual"
        )
        kv_cache_mb = weights_mb * _KV_FALLBACK_FRACTION * batch * (seq_positions / max(context_len, 1))

    total_mb = weights_mb + kv_cache_mb
    return {
        "heuristic": True,
        "method": method,
        "weights_mb": round(weights_mb, 1),
        "kv_cache_mb": round(kv_cache_mb, 1),
        "total_estimated_mb": round(total_mb, 1),
        "assumptions": {
            "batch": batch,
            "context_len": context_len,
            "output_len": output_len,
            "kv_cache_dtype": kv_cache_dtype or "auto",
            "overhead_factor": _WEIGHT_OVERHEAD_FACTOR,
        },
        "notes": notes,
    }
