"""
Phase 1 verification tests for deployment_manager.py.

These test the PURE LOGIC of create/start/reuse orchestration and port
recovery using mocked `docker`/`nvidia-smi` calls, since this sandbox has
no real Docker daemon or GPU. This is NOT a substitute for the real
reboot test on the target Windows/WSL2 machine -- see the reboot
checklist delivered alongside this repo. Passing these tests means
"TESTED" in isolation; it does not mean "CONFIRMED WORKING" end to end.
"""
from __future__ import annotations

import json
import sys
import types
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
}


def _inspect_payload(running: bool, port: int = 8000, cmd=None):
    return [{
        "State": {"Running": running, "Status": "running" if running else "exited"},
        "NetworkSettings": {"Ports": {f"{port}/tcp": [{"HostIp": "127.0.0.1", "HostPort": str(port)}]}},
        "Config": {"Image": "nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc22", "Cmd": cmd or [
            "trtllm-serve", "serve", "/models/Qwen2.5-1.5B-Instruct",
            "--backend", "pytorch", "--host", "0.0.0.0", "--port", str(port),
            "--served_model_name", "Qwen2.5-1.5B-Instruct",
        ]},
    }]


class Phase1Tests(unittest.TestCase):
    def setUp(self):
        dm._deployments.clear()
        self.settings = AppSettings(data_dir="/tmp/trtllm-ui-test", model_dir="/tmp/models")

    # ---- item 3: --restart unless-stopped, no --rm ----
    def test_restart_policy_present_and_no_rm(self):
        with patch.object(dm, "get_model", return_value=FAKE_MODEL), \
             patch.object(dm, "cached_manifest", return_value=None):
            cmd, _ = dm.build_command(self.settings, DeploymentConfig(model_name="Qwen2.5-1.5B-Instruct", port=8000))
        self.assertIn("--restart", cmd)
        self.assertEqual(cmd[cmd.index("--restart") + 1], "unless-stopped")
        self.assertNotIn("--rm", cmd)

    # ---- item 2: create path (no existing container) ----
    def test_start_deployment_creates_when_absent(self):
        with patch.object(dm, "get_model", return_value=FAKE_MODEL), \
             patch.object(dm, "cached_manifest", return_value=None), \
             patch.object(dm, "_docker_inspect", return_value=None), \
             patch("subprocess.run") as run, \
             patch.object(dm.threading, "Thread") as thread:
            run.return_value = MagicMock(returncode=0, stdout="containerid123", stderr="")
            record = dm.start_deployment(self.settings, DeploymentConfig(model_name="Qwen2.5-1.5B-Instruct", port=8000))
        self.assertEqual(record["status"], "starting")
        self.assertEqual(record["deployment_id"], "qwen2-5-1-5b-instruct")
        docker_run_call = run.call_args_list[0][0][0]
        self.assertEqual(docker_run_call[:3], ["docker", "run", "-d"])
        thread.assert_called_once()

    # ---- item 2: start path (exists, stopped) ----
    def test_start_deployment_starts_when_stopped(self):
        with patch.object(dm, "_docker_inspect", return_value=_inspect_payload(running=False)[0]), \
             patch("subprocess.run") as run, \
             patch.object(dm.threading, "Thread") as thread:
            run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            record = dm.start_deployment(self.settings, DeploymentConfig(model_name="Qwen2.5-1.5B-Instruct", port=8000))
        run.assert_called_once_with(["docker", "start", "trtllm-ui-qwen2-5-1-5b-instruct"],
                                     capture_output=True, text=True, check=False)
        self.assertEqual(record["status"], "starting")
        self.assertEqual(record["port"], 8000)

    # ---- item 2: reuse path (exists, already running) -- the actual bug ----
    def test_start_deployment_reuses_when_already_running(self):
        with patch.object(dm, "_docker_inspect", return_value=_inspect_payload(running=True)[0]), \
             patch("subprocess.run") as run:
            record = dm.start_deployment(self.settings, DeploymentConfig(model_name="Qwen2.5-1.5B-Instruct", port=8000))
        # Must NOT attempt docker run or docker start -- container is already up.
        run.assert_not_called()
        self.assertEqual(record["status"], "running")
        self.assertTrue(any("reused" in w.lower() for w in record["warnings"]))
        self.assertEqual(record["port"], 8000)

    # ---- item 1b: port recovery from docker inspect ----
    def test_extract_port_from_network_settings(self):
        info = _inspect_payload(running=True, port=8123)[0]
        self.assertEqual(dm._extract_port(info), 8123)

    def test_extract_port_fallback_to_cmd_flag(self):
        info = {"NetworkSettings": {"Ports": {}}, "Config": {"Cmd": ["trtllm-serve", "serve", "--port", "9001"]}}
        self.assertEqual(dm._extract_port(info), 9001)

    def test_extract_port_none_when_unavailable(self):
        self.assertIsNone(dm._extract_port(None))
        self.assertIsNone(dm._extract_port({"NetworkSettings": {"Ports": {}}, "Config": {"Cmd": []}}))

    # ---- item 1a/1b/1c: reconcile() port + config recovery, with retry ----
    def test_reconcile_recovers_port_and_config(self):
        ps_line = json.dumps({"Names": "trtllm-ui-qwen2-5-1-5b-instruct", "State": "running"})
        with patch("subprocess.run") as run:
            def side_effect(cmd, **kwargs):
                if cmd[:3] == ["docker", "ps", "-a"]:
                    return MagicMock(returncode=0, stdout=ps_line + "\n", stderr="")
                if cmd[:2] == ["docker", "inspect"]:
                    return MagicMock(returncode=0, stdout=json.dumps(_inspect_payload(running=True, port=8555)), stderr="")
                return MagicMock(returncode=1, stdout="", stderr="")
            run.side_effect = side_effect
            dm.reconcile()
        record = dm._deployments["qwen2-5-1-5b-instruct"]
        self.assertEqual(record["port"], 8555)  # the actual bug this fixes
        self.assertEqual(record["status"], "running")
        self.assertTrue(record["config"].get("reconstructed"))

    def test_reconcile_retries_on_docker_not_ready(self):
        # Simulates cold-boot: docker ps fails twice (daemon not up yet), succeeds on 3rd try.
        calls = {"n": 0}

        def side_effect(cmd, **kwargs):
            if cmd[:3] == ["docker", "ps", "-a"]:
                calls["n"] += 1
                if calls["n"] < 3:
                    return MagicMock(returncode=1, stdout="", stderr="Cannot connect to the Docker daemon")
                return MagicMock(returncode=0, stdout="", stderr="")
            return MagicMock(returncode=1, stdout="", stderr="")

        with patch("subprocess.run", side_effect=side_effect), patch.object(dm.time, "sleep") as sleep:
            dm.reconcile(retries=3, retry_delay=0.01)
        self.assertEqual(calls["n"], 3)
        self.assertEqual(sleep.call_count, 2)  # slept between attempt 1->2 and 2->3, not after success

    def test_reconcile_gives_up_after_retries_without_crashing(self):
        with patch("subprocess.run", return_value=MagicMock(returncode=1, stdout="", stderr="daemon down")), \
             patch.object(dm.time, "sleep"):
            dm.reconcile(retries=2, retry_delay=0.01)  # should not raise
        self.assertEqual(dm._deployments, {})

    # ---- port-collision check excludes the deployment's own stale record ----
    def test_build_command_port_check_excludes_self(self):
        dm._deployments["qwen2-5-1-5b-instruct"] = {"port": 8000}
        with patch.object(dm, "get_model", return_value=FAKE_MODEL), \
             patch.object(dm, "cached_manifest", return_value=None), \
             patch.object(dm, "_port_free", return_value=True):
            # Same deployment_id reusing its own port 8000 should NOT raise.
            cmd, _ = dm.build_command(self.settings, DeploymentConfig(model_name="Qwen2.5-1.5B-Instruct", port=8000))
        self.assertIn("8000", cmd)

    def test_build_command_port_check_still_blocks_other_deployments(self):
        dm._deployments["some-other-deployment"] = {"port": 8000}
        with patch.object(dm, "get_model", return_value=FAKE_MODEL), \
             patch.object(dm, "cached_manifest", return_value=None), \
             patch.object(dm, "_port_free", return_value=True):
            with self.assertRaises(ValueError):
                dm.build_command(self.settings, DeploymentConfig(model_name="Qwen2.5-1.5B-Instruct", port=8000))

    # ---- Windows cp1252 log_stream crash (found in live testing, not the original Phase 1 list) ----
    def test_log_stream_uses_explicit_utf8_encoding(self):
        # docker logs output is UTF-8. On Windows, Python's text-mode
        # default encoding is the system codepage (often cp1252), which
        # cannot decode bytes like 0x8f (from tqdm-style "▏" progress-bar
        # characters) and previously crashed the whole log stream with
        # UnicodeDecodeError on every reconnect from the browser's
        # EventSource. Assert Popen is called with an explicit encoding
        # that can't fail on arbitrary UTF-8 input.
        class FakePopen:
            def __init__(self, cmd, **kwargs):
                self.kwargs = kwargs
                self.stdout = iter(["Loading weights: 87%|\u258f\u258f\u258f\n"])
            def poll(self):
                return 0
            def terminate(self):
                pass

        captured = {}

        def fake_popen(cmd, **kwargs):
            captured.update(kwargs)
            return FakePopen(cmd, **kwargs)

        with patch.object(dm, "get_status", return_value={"container_name": "trtllm-ui-test"}), \
             patch("subprocess.Popen", fake_popen):
            lines = list(dm.log_stream("test"))
        self.assertEqual(captured.get("encoding"), "utf-8")
        self.assertEqual(captured.get("errors"), "replace")
        self.assertIn("\u258f", lines[0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
