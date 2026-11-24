import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

import pytest


@pytest.fixture
def container_adapter(tmp_path):
    (tmp_path / "config").mkdir()
    (tmp_path / "private").mkdir()
    wrapper = tmp_path / "config/hpc-python.sh"
    shutil.copy(Path(__file__).resolve().parents[1] / "config/hpc-python.sh", wrapper)
    runtime = tmp_path / "private/runtime.env"
    fake = tmp_path / "apptainer"
    fake.write_text(f"#!{sys.executable}\nimport json, sys\nprint(json.dumps(sys.argv[1:]))\n")
    fake.chmod(0o700)
    image = tmp_path / "python.sif"
    image.touch()
    runtime.write_text(f"HPC_APPTAINER={shlex.quote(str(fake))}\n"
                       f"HPC_CONTROL_IMAGE={shlex.quote(str(image))}\n"
                       "HPC_CONTROL_PYTHON=/control/bin/python\n"
                       f"HPC_GPU_IMAGE={shlex.quote(str(image))}\n"
                       "HPC_GPU_PYTHON=/gpu/bin/python\n")
    env = {k: v for k, v in os.environ.items() if not k.startswith("SLURM_")}
    env.pop("CUDA_VISIBLE_DEVICES", None)
    return wrapper, env


def test_control_container_never_enables_gpu(container_adapter):
    wrapper, env = container_adapter
    result = subprocess.run(["bash", str(wrapper), "-m", "artcurator", "status"], env=env,
                            capture_output=True, text=True, check=True)
    args = json.loads(result.stdout)
    assert "--nv" not in args and "CUDA_VISIBLE_DEVICES=" in args
    assert "--cleanenv" in args
    assert args[-4:] == ["/control/bin/python", "-m", "artcurator", "status"]


def test_model_check_uses_model_environment_without_gpu(container_adapter):
    wrapper, env = container_adapter
    env.update(SLURM_JOB_ID="123", CUDA_VISIBLE_DEVICES="2")
    result = subprocess.run(["bash", str(wrapper), "--model-check", "-m", "artcurator", "model-check"],
                            env=env, capture_output=True, text=True, check=True)
    args = json.loads(result.stdout)
    assert "--nv" not in args and "CUDA_VISIBLE_DEVICES=" in args
    assert "SLURM_JOB_ID=123" not in args
    assert args[-4:] == ["/gpu/bin/python", "-m", "artcurator", "model-check"]


def test_gpu_container_preserves_slurm_assignment(container_adapter):
    wrapper, env = container_adapter
    env.update(SLURM_JOB_ID="123", CUDA_VISIBLE_DEVICES="2", ART_ALLOCATION_START="1795438800")
    result = subprocess.run(["bash", str(wrapper), "-m", "artcurator", "hpc-worker"], env=env,
                            capture_output=True, text=True, check=True)
    args = json.loads(result.stdout)
    assert "--nv" in args and "CUDA_VISIBLE_DEVICES=2" in args and "SLURM_JOB_ID=123" in args
    assert "ART_ALLOCATION_START=1795438800" in args
    assert args[-4:] == ["/gpu/bin/python", "-m", "artcurator", "hpc-worker"]


def test_gpu_container_rejects_missing_allocation(container_adapter):
    wrapper, env = container_adapter
    env["SLURM_JOB_ID"] = "123"
    result = subprocess.run(["bash", str(wrapper), "--version"], env=env, capture_output=True, text=True)
    assert result.returncode != 0 and "must allocate a GPU" in result.stderr


def test_shared_unsquashfs_is_visible_with_minimal_slurm_path(container_adapter):
    wrapper, env = container_adapter
    root = wrapper.parent.parent
    helper = root / "shared-tools" / "unsquashfs"
    helper.parent.mkdir()
    helper.write_text("#!/bin/sh\nprintf 'shared helper ready\\n'\n")
    helper.chmod(0o700)
    with (root / "private/runtime.env").open("a") as file:
        file.write(f"HPC_UNSQUASHFS={shlex.quote(str(helper))}\n")
    (root / "apptainer").write_text(
        f"#!{sys.executable}\nimport subprocess\n"
        "print(subprocess.check_output(['unsquashfs', '-version'], text=True).strip())\n")
    env.update(PATH="/usr/bin:/bin", SLURM_JOB_ID="123", CUDA_VISIBLE_DEVICES="0")
    result = subprocess.run(["/bin/bash", str(wrapper), "--version"], env=env,
                            capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "shared helper ready"


def test_invalid_shared_unsquashfs_fails_before_launch(container_adapter):
    wrapper, env = container_adapter
    with (wrapper.parent.parent / "private/runtime.env").open("a") as file:
        file.write("HPC_UNSQUASHFS=missing-relative-tool\n")
    result = subprocess.run(["bash", str(wrapper), "--version"], env=env, capture_output=True, text=True)
    assert result.returncode == 2 and "HPC_UNSQUASHFS" in result.stderr
    assert not result.stdout
