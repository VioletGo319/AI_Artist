import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from artcurator.generator import resources, worker
from artcurator.generator.plan import make_plan


@pytest.mark.parametrize("initialized", [True, False])
def test_cleanup_does_not_initialize_cuda(monkeypatch, initialized):
    cuda = SimpleNamespace(is_initialized=lambda: initialized, empty_cache=Mock())
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda))
    resources.release_cuda_cache()
    assert cuda.empty_cache.call_count == int(initialized)


def test_generation_failure_preserves_saved_work_and_runs_cleanup(setup, monkeypatch):
    from datetime import datetime, timedelta, timezone
    env, store = setup
    plan, _ = make_plan(store, env, "2026-11-22", 2, "local")
    draws = 0

    def draw(prompt, seed, width, height):
        nonlocal draws
        draws += 1
        if draws > 1:
            raise RuntimeError("simulated CUDA failure")
        from artcurator.generator.backends import mock_image
        return mock_image(prompt, seed, width, height)

    cleanup = Mock()
    monkeypatch.setattr(worker, "renderer", lambda *args: draw)
    monkeypatch.setattr(worker, "release_cuda_cache", cleanup)
    with pytest.raises(RuntimeError, match="simulated CUDA"):
        worker.run(store, env, plan, datetime.now(timezone.utc) + timedelta(minutes=1))
    assert len(store.artworks(plan.id)) == 1
    assert store.batch()["state"] == "partial"
    cleanup.assert_called_once()


@pytest.mark.skipif(os.name != "posix", reason="GPU worker requires POSIX process groups")
@pytest.mark.parametrize("outcome", ["success", "failure", "timeout"])
def test_child_group_is_reaped_even_when_launcher_exits(tmp_path, outcome):
    # The descendant deliberately ignores TERM. Cleanup must also issue KILL,
    # including when its launcher has already returned successfully.
    child = tmp_path / "child.py"
    child.write_text('''import os, signal, sys, time
from pathlib import Path
signal.signal(signal.SIGTERM, signal.SIG_IGN)
Path(sys.argv[1]).write_text(str(os.getpid()))
time.sleep(30)
''')
    pidfile = tmp_path / "child.pid"
    launcher = tmp_path / "launcher.py"
    launcher.write_text('''import subprocess, sys, time
from pathlib import Path
subprocess.Popen([sys.executable, sys.argv[1], sys.argv[2]])
while not Path(sys.argv[2]).exists():
    time.sleep(.01)
if sys.argv[3] == "timeout":
    time.sleep(30)
sys.exit(7 if sys.argv[3] == "failure" else 0)
''')
    command = [sys.executable, str(launcher), str(child), str(pidfile), outcome]
    expected = {"failure": subprocess.CalledProcessError, "timeout": subprocess.TimeoutExpired}
    if outcome in expected:
        with pytest.raises(expected[outcome]):
            resources.run_isolated(command, cwd=tmp_path, env=os.environ.copy(), timeout=2, check=True)
    else:
        assert resources.run_isolated(command, cwd=tmp_path, env=os.environ.copy(), timeout=2, check=True) == 0
    pid = int(pidfile.read_text())
    for _ in range(40):
        result = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True)
        status = result.stdout.strip()
        # A zombie has exited and owns no GPU context; its reaping belongs to init.
        if not status or status.startswith("Z"):
            break
        time.sleep(.05)
    else:
        pytest.fail("GPU worker descendant survived process-group cleanup")
