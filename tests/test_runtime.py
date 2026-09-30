import errno
import os
from pathlib import Path
import signal
import subprocess
import sys
from types import SimpleNamespace

import pytest

from teemoe import runtime


def start(processes, code):
    process = subprocess.Popen([sys.executable, "-u", "-c", code], start_new_session=True,
                               stdout=subprocess.PIPE, text=True)
    processes.append(process)
    return process


def test_worker_failure_stops_running_peers():
    with pytest.raises(RuntimeError, match="status 3"):
        with runtime.worker_pool() as processes:
            sleeping = start(processes, "import time; time.sleep(60)")
            failed = start(processes, "raise SystemExit(3)")
            runtime.wait_workers(processes)
    assert failed.returncode == 3
    assert sleeping.poll() is not None


def test_worker_cleanup_on_launch_error():
    with pytest.raises(OSError, match="launch failed"):
        with runtime.worker_pool() as processes:
            sleeping = start(processes, "import time; time.sleep(60)")
            raise OSError("launch failed")
    assert sleeping.poll() is not None


def test_successful_workers_and_empty_pool():
    before = signal.getsignal(signal.SIGTERM)
    with runtime.worker_pool() as processes:
        runtime.wait_workers(processes)
        finished = start(processes, "pass")
        runtime.wait_workers(processes)
    assert finished.returncode == 0
    assert signal.getsignal(signal.SIGTERM) == before


def test_sigterm_cleans_nested_worker_sessions():
    code = """
import subprocess, sys
from teemoe.runtime import worker_pool, wait_workers
with worker_pool() as processes:
    child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'], start_new_session=True)
    processes.append(child)
    print(child.pid, flush=True)
    wait_workers(processes)
"""
    with runtime.worker_pool() as processes:
        parent = start(processes, code)
        grandchild_pid = int(parent.stdout.readline())
        parent.terminate()
        assert parent.wait(timeout=10) == 128 + signal.SIGTERM
        with pytest.raises(ProcessLookupError):
            os.kill(grandchild_pid, 0)


@pytest.mark.parametrize("launcher", ["generation", "gift", "forecasters"])
def test_inference_launchers_clean_up_failed_pools(launcher, monkeypatch, tmp_path):
    from teemoe import generation
    from teemoe.ensemble import forecasters
    from teemoe.eval import gift

    real_popen = subprocess.Popen
    children = []

    def launch(command, **kwargs):
        assert kwargs["start_new_session"] is True
        code = "import time; time.sleep(60)" if not children else "raise SystemExit(3)"
        process = real_popen([sys.executable, "-c", code], **kwargs)
        children.append(process)
        return process

    monkeypatch.setattr(subprocess, "Popen", launch)
    with pytest.raises(RuntimeError):
        if launcher == "generation":
            generation.generate_vllm([dict(prompt="test")] * 2, adapters=[], model="test", revision="test",
                                     devices="0,1")
        elif launcher == "gift":
            options = {key: None for key in ("checkpoint", "backend", "vllm_python", "environments", "granite",
                                            "fnf_root", "output", "gift_root", "gift_data")}
            gift.launch(SimpleNamespace(gpus=[0, 1], **options))
        else:
            forecasters.run_all("unused.jsonl", tmp_path / "output", devices=("cuda:0", "cuda:1"),
                                models=["core/chronos2", "core/toto2"])
    assert len(children) == 2
    assert all(p.poll() is not None for p in children)
    assert children[1].returncode == 3


@pytest.fixture
def scratch(monkeypatch, tmp_path):
    chosen = []
    real_temporary_directory = runtime.tempfile.TemporaryDirectory

    def temporary_directory(*, prefix, dir):
        chosen.append(Path(dir))
        return real_temporary_directory(prefix=prefix, dir=tmp_path)

    monkeypatch.setattr(runtime.tempfile, "gettempdir", lambda: str(tmp_path))
    monkeypatch.setattr(runtime.tempfile, "TemporaryDirectory", temporary_directory)
    monkeypatch.setattr(runtime.Path, "is_dir", lambda p: True)
    monkeypatch.setattr(runtime.os, "access", lambda *args: True)
    return tmp_path, chosen


def test_adapter_scratch_falls_back_from_small_shared_memory(scratch, monkeypatch):
    disk, chosen = scratch
    monkeypatch.setattr(runtime.shutil, "disk_usage", lambda p: SimpleNamespace(
        free=64 * 1024**2 if str(p) == "/dev/shm" else 4 * 1024**3))
    with runtime.adapter_scratch(1_600_000_000):
        assert chosen == [disk]


def test_adapter_scratch_prefers_sufficient_shared_memory(scratch, monkeypatch):
    _, chosen = scratch
    monkeypatch.setattr(runtime.shutil, "disk_usage", lambda p: SimpleNamespace(free=4 * 1024**3))
    with runtime.adapter_scratch(1_600_000_000):
        assert chosen == [Path("/dev/shm")]


def test_adapter_scratch_respects_explicit_directory(scratch, monkeypatch):
    disk, chosen = scratch
    monkeypatch.setattr(runtime.shutil, "disk_usage", lambda p: SimpleNamespace(free=4 * 1024**3))
    with runtime.adapter_scratch(1_600_000_000, str(disk)):
        assert chosen == [disk]


def test_adapter_scratch_reports_insufficient_space(scratch, monkeypatch):
    monkeypatch.setattr(runtime.shutil, "disk_usage", lambda p: SimpleNamespace(free=0))
    with pytest.raises(OSError, match="TEEMOE_LORA_SCRATCH") as error:
        runtime.adapter_scratch(1_600_000_000)
    assert error.value.errno == errno.ENOSPC
