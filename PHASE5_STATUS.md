# Phase 5 Status — Engine-Build Investigation

**Decision: Do not pursue engine-build for this deployment. Confirmed
"no," not skipped by default.**

This document exists because the reasoning behind that decision
previously only existed in a chat conversation, not as a checkable
artifact in this repo. It consolidates what was actually searched, what
was found, and which real evidence (with file paths) backs each claim.

## The question

Per the original build spec (`trtllm-ui-build-spec.md`), Phase 5's job
was: *"Is an actual TensorRT engine worth pursuing?"* — deciding whether
to test an alternate TensorRT-LLM image/release for real
`trtllm-build` engine-compilation support, as opposed to the confirmed
working path (`trtllm-serve` against the HF checkpoint directly, PyTorch
backend).

## What was checked, and how

### 1. Confirmed absence of `trtllm-build` in the current image

The current image (`nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc22`) was
diagnosed directly (Phase 0, before Phase 1): `trtllm-build` is not
present as a CLI tool in this container. Only `trtllm-serve` (with
`serve`, `disaggregated`, `disaggregated_mpi_worker`, `embeddings`,
`mm_embedding_serve` subcommands) exists. This was confirmed by running
the actual container and inspecting its `--help` output and installed
binaries, not inferred from documentation.

### 2. Hardware precision exclusions (Phase 2)

The GPU is an RTX 3050 Laptop, compute capability 8.6 (Ampere/SM86),
confirmed via `nvidia-smi --query-gpu=compute_cap`. FP8 and NVFP4 —
the two precisions engine-build traditionally unlocks the most value
from — require SM89 (Ada Lovelace/Hopper) and SM100 (Blackwell)
respectively. Both are hardware-excluded on this GPU regardless of
software version. See `backend/capability_probe.py`'s
`_precision_capabilities()` and its test suite
(`backend/test_capability_probe_phase2.py`).

### 3. Confirmed already-active optimizations that don't need engine-build

CUDA graphs (batch sizes captured up to 128) and paged KV cache with
block reuse are both active by default in the PyTorch-backend runtime
path already in use — confirmed directly from a real deployment's
`LLM Args:` log dump, not assumed from documentation. Two of
engine-build's traditional advantages are already present without it.

### 4. Compute-saturation evidence (Phase 3)

Manual benchmark runs via the app's Benchmark page (Phase 3), saved as
individual JSON result files under the app's configured data directory
(`<data_dir>\benchmarks\*.json` — 26 files independently confirmed to
exist via `dir` during this project, not narrated), showed:

- Concurrency 1: ~24 tok/s (single-stream ceiling, consistent with
  memory-bandwidth-bound decode)
- Concurrency 250: ~2400-2550 tok/s, GPU utilization 97-98%
- Concurrency 600: ~2470-2560 tok/s, GPU utilization ~98% (essentially
  flat vs. 250 — not still scaling)

These figures are **from the Phase 3 manual Benchmark-page runs
specifically** — a separate, later exercise
(`backend/build_sweep.py`'s automated config-variant sweep, Phase 4)
used a different fixed concurrency (100) for a different purpose
(comparing config flags cheaply across many variants, not finding the
concurrency ceiling) and its `sweep_results/*.json` files should not be
expected to reproduce these same numbers — they were never the same
measurement.

### 5. Config-tuning ceiling confirmed (Phase 4)

An 11-variant automated sweep (`backend/build_sweep.py`,
`backend/sweep_results/sweep_1787835447.json`) tested config-level
levers (memory fraction, chunked prefill, CUDA graph padding, attention
backend, scheduler policy, postprocessing workers, telemetry) with 4
repeats each. Result: every variant except one was statistically
indistinguishable from baseline (Mann-Whitney p > 0.05 in all 10 cases).
The one exception, `attn_backend=FLASHINFER`, was ~60% **slower** than
default — a confirmed regression, not an improvement. No config change
tested closed a meaningful gap versus the compute ceiling found in
Phase 3.

### 6. External search on current engine-build ecosystem direction

A live web search was run (`"TensorRT-LLM trtllm-build engine build
support 2026 release"`) followed by a direct fetch of
`https://github.com/NVIDIA/TensorRT-LLM/releases` to check current
release notes as a primary source, not just documentation pages that
may lag behind actual development focus.

Finding, stated at the confidence level it was actually confirmed at:
across 5 consecutive recent pre-releases (rc13 through rc17) fetched
directly from that primary source, `trtllm-build` and classic
engine-build received **zero mentions** in any Feature/API/Fix section
— all visible engineering effort was going into `trtllm-serve`, the
PyTorch backend, disaggregated serving, and the LLM API (the same path
already in use here). One specific claim from an initial search-engine
snippet — that TensorRT-backend support would be removed entirely in an
upcoming release — could **not** be independently verified against the
primary source during the direct fetch, and was explicitly retracted
rather than repeated as fact. The zero-mentions-across-5-releases
finding stands on its own regardless of that retraction.

## Reasoning

The two traditional justifications for chasing engine-build were both
checked directly and ruled out:

1. **Unused hardware capability** — ruled out. FP8/NVFP4 are
   hardware-excluded on this GPU; no image or engine-build path changes
   that.
2. **A real optimization gap to close** — ruled out. The system is
   compute-saturated (97-98% GPU utilization from concurrency 250
   onward), and a targeted config-tuning sweep found no lever that
   moved throughput beyond measured noise.

Per the original spec's own stated criterion (section 6): *"if
PyTorch-backend optimization already gets close to hardware limits,
engine-build's marginal value may not justify the added
image/maintenance burden."* Both of this project's own real
measurements (Phase 3's saturation data, Phase 4's sweep) demonstrate
that condition directly, not by assumption.

## Conclusion

**Phase 5 decision: do not pursue an alternate TensorRT-LLM image or
engine-build path.** This is a deliberate, evidence-backed "no," not an
oversight or a default skip.

**Phase 6 (engine lifecycle: versioned engine storage, build pipeline,
`current.json` pointer) is correctly not implemented as a direct
consequence of this decision** — the original spec explicitly scoped
Phase 6 as contingent on Phase 5 answering "yes." No engine-build code
exists anywhere in this repository (`backend/build_sweep.py` sweeps
*deployment configuration* of the existing `trtllm-serve` path only —
its filename containing "build" is unrelated to TensorRT engine-build
and should not be read as such).

## Revisiting this decision

This conclusion is specific to the current hardware (RTX 3050 6GB,
SM86), the current model (Qwen2.5-1.5B-Instruct), and the TensorRT-LLM
release tested (1.3.0rc22). It should be re-evaluated, not assumed to
still hold, if any of the following change:
- Different/newer GPU with FP8+ hardware support
- A materially larger model where engine-build's kernel-fusion benefits
  might matter more
- A future TensorRT-LLM release that reintroduces engine-build as an
  actively developed path (check release notes directly, the way this
  investigation did, rather than trusting cached documentation)
