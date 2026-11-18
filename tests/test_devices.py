"""Choosing the device, with torch's view of the GPUs replaced, so that no GPU is needed."""

import pytest
import torch

from shelfvision.detector import ObjectDetector
from shelfvision.devices import resolve_device
from shelfvision.normals import NormalEstimator


def see_gpus(monkeypatch: pytest.MonkeyPatch, count: int) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: count > 0)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: count)


def test_auto_takes_the_gpu_only_when_torch_sees_one(monkeypatch: pytest.MonkeyPatch) -> None:
    see_gpus(monkeypatch, 0)
    assert resolve_device() == torch.device("cpu")
    see_gpus(monkeypatch, 1)
    assert resolve_device("auto") == torch.device("cuda")


def test_cpu_is_kept_even_with_a_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    see_gpus(monkeypatch, 1)
    assert resolve_device("cpu") == torch.device("cpu")


def test_a_missing_gpu_fails_when_asked_for(monkeypatch: pytest.MonkeyPatch) -> None:
    see_gpus(monkeypatch, 0)
    with pytest.raises(RuntimeError, match="CUDA build"):
        resolve_device("cuda")


def test_a_gpu_index_must_exist(monkeypatch: pytest.MonkeyPatch) -> None:
    see_gpus(monkeypatch, 1)
    assert resolve_device("cuda:0") == torch.device("cuda:0")
    with pytest.raises(RuntimeError, match="sees 1 GPUs"):
        resolve_device("cuda:1")


@pytest.mark.parametrize("name", ["mps", "meta"])
def test_other_devices_are_refused(name: str) -> None:
    with pytest.raises(ValueError, match="not supported"):
        resolve_device(name)


def test_the_models_take_the_device_before_loading_any_weights() -> None:
    assert ObjectDetector("cpu").device == torch.device("cpu")
    assert NormalEstimator("cpu").device == torch.device("cpu")
