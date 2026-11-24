import hashlib
import importlib.util
from pathlib import Path
import shutil
import subprocess

import pytest

spec = importlib.util.spec_from_file_location("stream_flux", Path(__file__).parents[1] / "config/stream-flux.py")
stream = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stream)


@pytest.mark.parametrize("name", ["../token", "/etc/passwd", "text_encoder/../../token", "x\ny"])
def test_model_transfer_rejects_unsafe_paths(name):
    with pytest.raises(ValueError):
        stream.safe_relative(name)


@pytest.mark.parametrize("lfs", [True, False])
def test_transfer_verifies_size_and_hash_before_use(tmp_path, lfs):
    if not shutil.which("sha256sum") or not shutil.which("sha1sum"):
        pytest.skip("GNU checksum tools are required on the Linux HPC/CI target")
    file = tmp_path / "model 'quoted'.bin"
    raw = b"model data for a complete transfer"
    file.write_bytes(raw)
    digest = hashlib.sha256(raw).hexdigest() if lfs else hashlib.sha1(
        f"blob {len(raw)}\0".encode() + raw).hexdigest()
    command = stream.verification(file, len(raw), digest, lfs)
    assert subprocess.run(["sh", "-c", command]).returncode == 0
    file.write_bytes(b"damaged" + raw)
    assert subprocess.run(["sh", "-c", command]).returncode != 0


def test_transfer_rejects_invalid_checksum():
    with pytest.raises(ValueError):
        stream.verification("model", 50, "malformed; anything", True)
