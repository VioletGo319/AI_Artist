import sys
from types import SimpleNamespace

import pytest

from artcurator.generator import backends


def fake_cuda(monkeypatch, name, memory, count=1):
    cuda = SimpleNamespace(is_available=lambda: True, device_count=lambda: count,
                           get_device_name=lambda index: name,
                           get_device_properties=lambda index: SimpleNamespace(total_memory=int(memory*1024**3)),
                           set_device=lambda index: None,
                           is_bf16_supported=lambda including_emulation=False: True)
    torch = SimpleNamespace(cuda=cuda)
    monkeypatch.setitem(sys.modules, 'torch', torch)
    monkeypatch.setattr(backends, 'find_spec', lambda name: object())
    return torch


@pytest.mark.parametrize('memory', [39.4, 79.2])
def test_full_a100_variants_are_accepted(monkeypatch, capsys, memory):
    torch = fake_cuda(monkeypatch, 'NVIDIA A100', memory)
    assert backends.gpu(only_a100=True) is torch
    assert f'memory={memory:.2f}' in capsys.readouterr().err


@pytest.mark.parametrize('name,memory,count', [
    ('NVIDIA A100', 39.4, 2), ('NVIDIA A40', 44.5, 1),
    ('NVIDIA A100 MIG 3g.40gb', 39.4, 1), ('NVIDIA A100', 19.5, 1),
])
def test_wrong_or_multiple_devices_are_rejected(monkeypatch, name, memory, count):
    fake_cuda(monkeypatch, name, memory, count)
    with pytest.raises(backends.NoGPU, match='found 2|received'):
        backends.gpu(only_a100=True)
