"""
Phase 2 tests for deployment_manager.py: preflight() and the
kv_cache_dtype hardware-capability gate in build_command(). Mocked
docker/nvidia-smi, same convention as test_deployment_manager_phase1.py.
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
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
        # Confirmed real-world value (tested against actual RTX 3050 +
        # trtllm-serve 1.3.0rc22): the CLI's own enum is [auto|fp8|nvfp4],
        # and fp8/nvfp4 are BOTH hardware-excluded on Ampere, so "auto" is
        # the only usable choice. Do not add "int8" back -- it isn't a
        # real --kv_cache_dtype value in this build; see capability_probe.
        "kv_cache_dtype_options": ["auto"],
        "parallel_max": 1,
        "serve_flags": [{"name": "--kv_cache_dtype", "takes_value": True,
                          "help": "--kv_cache_dtype [auto|fp8|nvfp4]"}],
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
                model_name="Qwen2.5-1.5B-Instruct", port=8000, kv_cache_dtype="auto",
            ))
        self.assertIn("auto", cmd)

    def test_no_manifest_yet_does_not_block_deploy(self):
        # Can't judge hardware support without a probe -- must not fail closed.
        with patch.object(dm, "get_model", return_value=FAKE_MODEL), \
             patch.object(dm, "cached_manifest", return_value=None), \
             patch.object(dm, "_port_free", return_value=True):
            cmd, _ = dm.build_command(self.settings, DeploymentConfig(
                model_name="Qwen2.5-1.5B-Instruct", port=8000, kv_cache_dtype="fp8",
            ))


class ExtraLlmApiOptionsTests(unittest.TestCase):
    """extra_llm_api_options is the only path to nested trtllm-serve config
    (cuda_graph_config.enable_padding, scheduler_config policy, etc) not
    exposed as top-level CLI flags -- these tests check the file-write +
    mount + --config wiring, not whether trtllm-serve actually accepts
    JSON-as-YAML content (that part is genuinely unconfirmed, see the
    field's docstring).
    """
    def setUp(self):
        dm._deployments.clear()
        self.tmp = tempfile.TemporaryDirectory()
        self.settings = AppSettings(data_dir=self.tmp.name, model_dir=self.tmp.name)

    def tearDown(self):
        dm._deployments.clear()
        self.tmp.cleanup()

    def test_no_options_means_no_config_flag_or_mount(self):
        with patch.object(dm, "get_model", return_value=FAKE_MODEL), \
             patch.object(dm, "cached_manifest", return_value=None), \
             patch.object(dm, "_port_free", return_value=True):
            cmd, _ = dm.build_command(self.settings, DeploymentConfig(
                model_name="Qwen2.5-1.5B-Instruct", port=8000,
            ))
        self.assertNotIn("--config", cmd)

    def test_options_present_adds_config_flag_and_mount(self):
        with patch.object(dm, "get_model", return_value=FAKE_MODEL), \
             patch.object(dm, "cached_manifest", return_value=None), \
             patch.object(dm, "_port_free", return_value=True):
            cmd, _ = dm.build_command(self.settings, DeploymentConfig(
                model_name="Qwen2.5-1.5B-Instruct", port=8000,
                extra_llm_api_options={"cuda_graph_config": {"enable_padding": True}},
            ))
        self.assertIn("--config", cmd)
        self.assertIn("/trtllm_extra_config.yaml", cmd)
        mount_args = [a for a in cmd if a.endswith(":/trtllm_extra_config.yaml:ro")]
        self.assertEqual(len(mount_args), 1)

    def test_options_are_actually_written_to_host_file(self):
        with patch.object(dm, "get_model", return_value=FAKE_MODEL), \
             patch.object(dm, "cached_manifest", return_value=None), \
             patch.object(dm, "_port_free", return_value=True):
            cmd, _ = dm.build_command(self.settings, DeploymentConfig(
                model_name="Qwen2.5-1.5B-Instruct", port=8000,
                extra_llm_api_options={"scheduler_config": {"capacity_scheduler_policy": "MAX_UTILIZATION"}},
            ))
        mount_arg = next(a for a in cmd if a.endswith(":/trtllm_extra_config.yaml:ro"))
        host_path = mount_arg.split(":/trtllm_extra_config.yaml:ro")[0]
        written = json.loads(Path(host_path).read_text(encoding="utf-8"))
        self.assertEqual(written["scheduler_config"]["capacity_scheduler_policy"], "MAX_UTILIZATION")


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
        report = self._run(DeploymentConfig(model_name="Qwen2.5-1.5B-Instruct", port=8000, kv_cache_dtype="auto"))
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

    def test_reuse_of_own_running_deployment_does_not_false_flag_port(self):
        # Regression: preflight for a deployment that already owns this
        # port (e.g. re-running preflight on an already-running, reused
        # deployment) must not call the raw OS port check and see its own
        # bound port as a collision against itself.
        deployment_id = "qwen2-5-1-5b-instruct"
        dm._deployments[deployment_id] = {
            "deployment_id": deployment_id, "container_name": f"trtllm-ui-{deployment_id}",
            "config": {}, "port": 8000, "status": "running", "warnings": [], "command": [],
        }
        try:
            # _port_free would report False here (something IS listening on
            # 8000 -- this deployment itself) -- preflight must still pass.
            report = self._run(
                DeploymentConfig(model_name="Qwen2.5-1.5B-Instruct", port=8000), port_free=False,
            )
        finally:
            dm._deployments.clear()
        port_check = next(c for c in report["checks"] if c["name"] == "port")
        self.assertEqual(port_check["status"], "pass")
        self.assertIn("reused", port_check["detail"])
        self.assertTrue(report["feasible"])


class GetStatusStaleReasonTests(unittest.TestCase):
    """Regression: a slow-booting deployment (confirmed real-world case --
    this project's own Qwen 1.5B model took ~160s to fully start, past
    _watch()'s old 90s deadline) must not be stuck showing status=running
    with a permanently attached stale "startup timeout" reason once Docker
    and the real /v1/models endpoint both confirm it's actually fine.
    """
    def setUp(self):
        dm._deployments.clear()
        self.deployment_id = "qwen2-5-1-5b-instruct"
        dm._deployments[self.deployment_id] = {
            "deployment_id": self.deployment_id,
            "container_name": f"trtllm-ui-{self.deployment_id}",
            "config": {}, "port": 8000, "status": "error",
            "reason": "startup timeout — check logs", "warnings": [], "command": [],
        }

    def tearDown(self):
        dm._deployments.clear()

    def test_recovered_deployment_clears_stale_reason_and_reports_ready(self):
        fake_response = MagicMock()
        fake_response.status = 200
        fake_response.__enter__ = lambda s: fake_response
        fake_response.__exit__ = lambda s, *a: False
        with patch.object(dm, "_docker_inspect", return_value={"State": {"Status": "running"}}), \
             patch.object(dm, "urlopen", return_value=fake_response):
            record = dm.get_status(self.deployment_id)
        self.assertEqual(record["status"], "ready")
        self.assertNotIn("reason", record)

    def test_running_but_not_yet_serving_clears_stale_reason_reports_running(self):
        # Docker says running, but /v1/models isn't answering yet (still
        # loading) -- status should be "running" (not stuck on the old
        # error/timeout), and the stale reason should still be cleared
        # since the container itself did NOT time out or crash.
        with patch.object(dm, "_docker_inspect", return_value={"State": {"Status": "running"}}), \
             patch.object(dm, "urlopen", side_effect=Exception("connection refused")):
            record = dm.get_status(self.deployment_id)
        self.assertEqual(record["status"], "running")
        self.assertNotIn("reason", record)

    def test_actually_exited_container_keeps_its_own_fresh_reason(self):
        # Sanity check the fix doesn't blanket-clear reasons for genuinely
        # dead containers -- only the "running" branch should touch it.
        with patch.object(dm, "_docker_inspect", return_value={"State": {"Status": "exited"}}):
            record = dm.get_status(self.deployment_id)
        self.assertEqual(record["status"], "exited")
        # Stale reason from the old error state is left alone here; a
        # fresh create/watch cycle would overwrite it if this exit is new.


class RehydrationOnMissedReconcileTests(unittest.TestCase):
    """Regression: confirmed in real-world testing, not hypothetical --
    reconcile()'s finite startup retry window can be exceeded by a slow
    Docker cold boot, after which the container comes up seconds later
    but nothing in-memory ever knows about it for the rest of the
    process's life. get_status() and preflight's port-ownership check
    must self-heal via an on-demand Docker lookup rather than trusting
    the in-memory table as proof of non-existence.
    """
    def setUp(self):
        dm._deployments.clear()
        self.settings = AppSettings(data_dir="/tmp/trtllm-ui-test", model_dir="/tmp/models")

    def tearDown(self):
        dm._deployments.clear()

    def test_get_status_rehydrates_a_running_container_reconcile_missed(self):
        fake_inspect = {
            "State": {"Status": "running", "Running": True},
            "NetworkSettings": {"Ports": {"8000/tcp": [{"HostPort": "8000"}]}},
            "Config": {"Cmd": ["trtllm-serve", "serve", "/models/x"]},
        }
        with patch.object(dm, "_docker_inspect", return_value=fake_inspect), \
             patch.object(dm, "_extract_port", return_value=8000), \
             patch.object(dm, "_reconstruct_config", return_value={"reconstructed": True}), \
             patch.object(dm, "urlopen", side_effect=Exception("not answering yet")):
            record = dm.get_status("qwen2-5-1-5b-instruct")
        self.assertEqual(record["port"], 8000)
        self.assertIn("Rehydrated on-demand", record["warnings"][0])
        # Second call must not re-hit docker inspect for the rehydration
        # path -- it's now tracked in memory.
        self.assertIn("qwen2-5-1-5b-instruct", dm._deployments)

    def test_get_status_still_raises_for_a_container_that_truly_does_not_exist(self):
        with patch.object(dm, "_docker_inspect", return_value=None):
            with self.assertRaises(KeyError):
                dm.get_status("never-deployed")

    def test_preflight_self_heals_port_ownership_after_missed_reconcile(self):
        # No in-memory record at all (simulating a missed reconcile), but
        # Docker confirms this exact deployment already owns port 8000 --
        # preflight must pass, not hard-fail against its own port.
        fake_inspect = {
            "State": {"Status": "running", "Running": True},
            "NetworkSettings": {"Ports": {"8000/tcp": [{"HostPort": "8000"}]}},
            "Config": {"Cmd": ["trtllm-serve", "serve", "/models/x"]},
        }
        with patch.object(dm, "get_model", return_value=FAKE_MODEL), \
             patch.object(dm, "cached_manifest", return_value=AMPERE_MANIFEST), \
             patch.object(dm.gpu_monitor, "poll", return_value=FAKE_GPU_OK), \
             patch.object(dm, "_docker_inspect", return_value=fake_inspect), \
             patch.object(dm, "_extract_port", return_value=8000), \
             patch.object(dm, "_reconstruct_config", return_value={}), \
             patch.object(dm, "_port_free", return_value=False), \
             patch("subprocess.run", return_value=MagicMock(returncode=0, stdout="28.0.0", stderr="")):
            report = dm.preflight(self.settings, DeploymentConfig(
                model_name="Qwen2.5-1.5B-Instruct", port=8000,
            ))
        port_check = next(c for c in report["checks"] if c["name"] == "port")
        self.assertEqual(port_check["status"], "pass")
        self.assertIn("reused", port_check["detail"])


class EffectiveConfigServedModelNameTests(unittest.TestCase):
    """Regression: confirmed in real-world use, not hypothetical --
    build_sweep.py's deploy requests omit served_model_name (reasonably,
    relying on the server-side default), and every single one of the 3
    places that persist the deployment record used raw
    config.model_dump() -- leaving served_model_name as None in the
    STORED record even though build_command() already resolves it to
    model_name for the actual `docker run` argv. Downstream consumers
    that read the stored record's served_model_name (e.g. main.py's
    /api/benchmarks deployment_id resolution) then saw None and 400'd,
    even though the container was genuinely running fine. Reliably
    reproduced this exact failure across an entire 11-variant sweep.
    """
    def test_effective_config_resolves_served_model_name_when_omitted(self):
        config = DeploymentConfig(model_name="Qwen2.5-1.5B-Instruct", port=8000)
        self.assertIsNone(config.served_model_name)  # confirms the omission itself
        effective = dm._effective_config_dict(config)
        self.assertEqual(effective["served_model_name"], "Qwen2.5-1.5B-Instruct")

    def test_effective_config_preserves_explicit_served_model_name(self):
        config = DeploymentConfig(model_name="Qwen2.5-1.5B-Instruct", port=8000, served_model_name="custom-name")
        effective = dm._effective_config_dict(config)
        self.assertEqual(effective["served_model_name"], "custom-name")

    def test_create_container_persists_resolved_served_model_name(self):
        dm._deployments.clear()
        settings = AppSettings(data_dir="/tmp/trtllm-ui-test", model_dir="/tmp/models")
        config = DeploymentConfig(model_name="Qwen2.5-1.5B-Instruct", port=8000)
        with patch.object(dm, "get_model", return_value=FAKE_MODEL), \
             patch.object(dm, "cached_manifest", return_value=None), \
             patch.object(dm, "_port_free", return_value=True), \
             patch("subprocess.run", return_value=MagicMock(returncode=0, stdout="containerid", stderr="")), \
             patch.object(dm.threading, "Thread"):
            record = dm.create_container(settings, config)
        # This is the exact field main.py's /api/benchmarks reads via
        # record["config"]["served_model_name"] -- must never be None
        # for a deployment that's actually running.
        self.assertEqual(record["config"]["served_model_name"], "Qwen2.5-1.5B-Instruct")
        dm._deployments.clear()

    def test_reused_running_deployment_also_has_resolved_served_model_name(self):
        dm._deployments.clear()
        settings = AppSettings(data_dir="/tmp/trtllm-ui-test", model_dir="/tmp/models")
        config = DeploymentConfig(model_name="Qwen2.5-1.5B-Instruct", port=8000)
        fake_inspect = {
            "State": {"Status": "running", "Running": True},
            "NetworkSettings": {"Ports": {"8000/tcp": [{"HostPort": "8000"}]}},
            "Config": {"Cmd": ["trtllm-serve", "serve", "/models/x"]},
        }
        with patch.object(dm, "_docker_inspect", return_value=fake_inspect), \
             patch.object(dm, "_extract_port", return_value=8000):
            record = dm.start_deployment(settings, config)
        self.assertEqual(record["config"]["served_model_name"], "Qwen2.5-1.5B-Instruct")
        dm._deployments.clear()


class PublishHostTests(unittest.TestCase):
    """publish_host controls Docker's -p HOST_IP:port:port binding --
    separate from `host`, which is trtllm-serve's own --host flag inside
    the container. Default must stay loopback-only for every deployment
    that doesn't explicitly opt in, since this server has no
    authentication of its own.
    """
    def setUp(self):
        dm._deployments.clear()
        self.settings = AppSettings(data_dir="/tmp/trtllm-ui-test", model_dir="/tmp/models")

    def tearDown(self):
        dm._deployments.clear()

    def test_default_publish_host_is_loopback(self):
        config = DeploymentConfig(model_name="Qwen2.5-1.5B-Instruct", port=8000)
        self.assertEqual(config.publish_host, "127.0.0.1")

    def test_default_publish_host_generates_loopback_bound_port_flag(self):
        with patch.object(dm, "get_model", return_value=FAKE_MODEL), \
             patch.object(dm, "cached_manifest", return_value=None), \
             patch.object(dm, "_port_free", return_value=True):
            cmd, warnings = dm.build_command(self.settings, DeploymentConfig(
                model_name="Qwen2.5-1.5B-Instruct", port=8000,
            ))
        self.assertIn("127.0.0.1:8000:8000", cmd)
        self.assertFalse(any("publish_host" in w.lower() or "network" in w.lower() for w in warnings),
                          f"default loopback deployment should not trigger a network-exposure warning, got: {warnings}")

    def test_explicit_0000_generates_open_port_flag_and_warns(self):
        with patch.object(dm, "get_model", return_value=FAKE_MODEL), \
             patch.object(dm, "cached_manifest", return_value=None), \
             patch.object(dm, "_port_free", return_value=True):
            cmd, warnings = dm.build_command(self.settings, DeploymentConfig(
                model_name="Qwen2.5-1.5B-Instruct", port=8000, publish_host="0.0.0.0",
            ))
        self.assertIn("0.0.0.0:8000:8000", cmd)
        self.assertNotIn("127.0.0.1:8000:8000", cmd)
        self.assertTrue(any("no" in w.lower() and "auth" in w.lower() for w in warnings),
                         f"expected a no-authentication warning, got: {warnings}")

    def test_health_check_urls_stay_loopback_regardless_of_publish_host(self):
        # get_status()'s own liveness check must always use loopback --
        # the app itself runs on the same host regardless of what
        # publish_host exposes the deployment to externally.
        dm._deployments["qwen2-5-1-5b-instruct"] = {
            "deployment_id": "qwen2-5-1-5b-instruct", "container_name": "trtllm-ui-qwen2-5-1-5b-instruct",
            "config": {"publish_host": "0.0.0.0"}, "port": 8000, "status": "running", "warnings": [], "command": [],
        }
        fake_response = MagicMock()
        fake_response.status = 200
        fake_response.__enter__ = lambda s: fake_response
        fake_response.__exit__ = lambda s, *a: False
        with patch.object(dm, "_docker_inspect", return_value={"State": {"Status": "running"}}), \
             patch.object(dm, "urlopen", return_value=fake_response) as mock_urlopen:
            dm.get_status("qwen2-5-1-5b-instruct")
        called_url = mock_urlopen.call_args[0][0]
        self.assertTrue(called_url.startswith("http://127.0.0.1:"), f"expected loopback health-check URL, got: {called_url}")
        dm._deployments.clear()


if __name__ == "__main__":
    unittest.main(verbosity=2)
