"""Worker cleanup and temporary storage for inference."""

from __future__ import annotations

import errno
import os
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def worker_pool():
    """Track workers started with ``start_new_session=True`` and clean up on failure."""
    processes = []
    handle_term = (threading.current_thread() is threading.main_thread()
                   and signal.getsignal(signal.SIGTERM) == signal.SIG_DFL)

    def terminate(signum, frame):
        raise SystemExit(128 + signum)

    if handle_term:
        signal.signal(signal.SIGTERM, terminate)
    try:
        yield processes
    finally:
        pending = [p for p in processes if p.poll() != 0]
        for process in pending:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        deadline = time.monotonic() + 5
        for process in pending:
            try:
                process.wait(timeout=max(0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                pass
        for process in pending:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
        if handle_term:
            signal.signal(signal.SIGTERM, signal.SIG_DFL)


def wait_workers(processes) -> None:
    """Notice any failed worker without waiting for earlier workers to finish first."""
    while True:
        codes = [p.poll() for p in processes]
        for process, code in zip(processes, codes, strict=True):
            if code is not None and code != 0:
                raise RuntimeError(f"Worker exited with status {code}: {process.args}")
        if all(code == 0 for code in codes):
            return
        time.sleep(0.1)


def adapter_scratch(required_bytes: int, preferred: str | None = None):
    """Prefer shared memory when it fits; otherwise use the normal temporary directory."""
    needed = required_bytes + max(64 * 1024**2, required_bytes // 10)
    candidates = [Path(preferred)] if preferred else [Path("/dev/shm"), Path(tempfile.gettempdir())]
    for root in dict.fromkeys(candidates):
        try:
            if root.is_dir() and os.access(root, os.W_OK | os.X_OK) and shutil.disk_usage(root).free >= needed:
                return tempfile.TemporaryDirectory(prefix="teemoe-lora-", dir=root)
        except OSError:
            continue
    raise OSError(errno.ENOSPC, f"No writable adapter scratch directory has {needed} free bytes. "
                  "Set TEEMOE_LORA_SCRATCH to a directory with enough space.")
