import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


@pytest.mark.parametrize("fail_check", [False, True])
def test_setup_isolates_workers_and_stops_on_bad_dependencies(tmp_path, fail_check):
    root = Path(__file__).resolve().parents[1]
    (tmp_path / "scripts").mkdir()
    shutil.copyfile(root / "scripts/setup.sh", tmp_path / "scripts/setup.sh")
    commands = tmp_path / "commands.jsonl"
    binary = tmp_path / "bin"
    binary.mkdir()
    fake_uv = binary / "uv"
    fake_uv.write_text(f"#!{sys.executable}\n" + '''
import json, os, pathlib, sys
args = sys.argv[1:]
with open(os.environ["SETUP_TEST_LOG"], "a") as log:
    log.write(json.dumps(args) + "\\n")
if args[:2] == ["pip", "check"] and os.environ.get("SETUP_TEST_FAIL") == "1":
    raise SystemExit(9)
''')
    fake_uv.chmod(0o755)
    fake_git = binary / "git"
    fake_git.write_text("#!/bin/sh\nexit 0\n")
    fake_git.chmod(0o755)
    result = subprocess.run(["bash", "scripts/setup.sh", "inference"], cwd=tmp_path, capture_output=True,
                            env=dict(os.environ, PATH=str(binary) + os.pathsep + os.environ["PATH"],
                                     SETUP_TEST_LOG=str(commands), SETUP_TEST_FAIL=str(int(fail_check))))
    calls = [json.loads(line) for line in commands.read_text().splitlines()]
    if fail_check:
        assert result.returncode == 9
        assert not any(call[:2] == ["pip", "compile"] for call in calls)
        return
    assert result.returncode == 0, result.stderr.decode()
    checks = [call for call in calls if call[:2] == ["pip", "check"]]
    assert len(checks) == 6
    for kind in ("compile", "sync"):
        workers = [call for call in calls if call[:2] == ["pip", kind]]
        assert len(workers) == 5
        assert all("--no-config" in call for call in workers)
    assert all("--no-config" in call for call in checks[1:])
