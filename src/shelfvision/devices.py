"""Choose where the models run: the CPU, or an NVIDIA GPU through CUDA.

The environment decides which torch is installed: `pixi install` gives the CPU build and
`pixi run -e gpu ...` the CUDA build. The device only says which of the available ones to use.
"""

from typing import Final
import torch

AUTO: Final = "auto"
SUPPORTED: Final = ("cpu", "cuda")


def resolve_device(name: str = AUTO) -> torch.device:
    """Return the torch device for "auto", "cpu", "cuda" or "cuda:N"

    "auto" takes the GPU when torch sees one and the CPU otherwise. A GPU that is not there fails
    here, when the models are created, rather than in the middle of the first image.
    """
    if name == AUTO:  # GPU if there is one, CPU otherwise
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type not in SUPPORTED:
        raise ValueError(f"Device {name!r} is not supported; use {AUTO!r} or one of {SUPPORTED}")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            f"Device {name!r} needs an NVIDIA GPU and the CUDA build of torch: "
            "run the project with `pixi run -e gpu ...`, or use device='cpu'"
        )
    if device.index is not None and device.index >= torch.cuda.device_count():
        raise RuntimeError(f"Device {name!r} does not exist: torch sees {torch.cuda.device_count()} GPUs")
    return device
