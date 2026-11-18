"""Time the stages of a run and write a report that compares them with the README

Scene._timed records every orchestrator stage here, and timed() records any other block. At the end
of a run, write_report() overwrites docs/execution-time-report.md and .json in the project folder
with the date, the machine and the mean seconds of each stage next to the README's figure.
Writing the report never stops the program: a failure only gives a warning.
"""

import json
import os
import platform
import sys
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Any, Final

REPORT_NAME: Final = "execution-time-report"

# The figures the README gives for each stage, in seconds, and how they were measured
NOTEBOOK_RUN: Final = "notebook run of 2026-11-18, 8 torch threads, mean of the 7 photos"
SAM_RUN: Final = (
    "benchmark run beside others, 8 torch threads; the notebook runs SAM only with SEGMENT = True"
)
README_CLAIMS: Final[dict[str, tuple[float, str]]] = {
    "encode image (OWLv2)": (13.4, NOTEBOOK_RUN),
    "read barcodes": (104.0, NOTEBOOK_RUN),
    "estimate normals (MoGe)": (6.0, NOTEBOOK_RUN),
    "encode image (SAM)": (10.2, SAM_RUN),
    "segment objects (SAM)": (3.7, SAM_RUN),
    "all stages of one photo": (124.0, NOTEBOOK_RUN),
    "same queries again": (0.102, "notebook run of 2026-11-18, input_2.jpg, one measurement"),
}

_runs: dict[str, list[float]] = {}
_started = perf_counter()


def record(stage: str, seconds: float) -> None:
    """Add one run of a stage"""
    _runs.setdefault(stage, []).append(seconds)


@contextmanager
def timed(stage: str) -> Iterator[None]:
    """Record the seconds spent inside the block; also works as a decorator"""
    start = perf_counter()
    try:
        yield
    finally:
        record(stage, perf_counter() - start)


def reset() -> None:
    """Forget every run and start the total time again"""
    global _started
    _runs.clear()
    _started = perf_counter()


def project_root(start: Path | None = None) -> Path | None:
    """The first folder with a pyproject.toml, going up from this file, whatever the working folder"""
    here = (start or Path(__file__)).resolve()
    return next((folder for folder in (here, *here.parents) if (folder / "pyproject.toml").is_file()), None)


def _cpu_name() -> str:
    if sys.platform == "win32":
        import winreg

        key = r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key) as handle:
            return str(winreg.QueryValueEx(handle, "ProcessorNameString")[0]).strip()
    for line in Path("/proc/cpuinfo").read_text().splitlines():
        if line.startswith("model name"):
            return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"


def _ram_gib() -> float:
    if sys.platform == "win32":
        import ctypes

        class MemoryStatus(ctypes.Structure):
            sizes = ("total", "free", "page_total", "page_free", "virtual_total", "virtual_free", "extra")
            _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
                (name, ctypes.c_ulonglong) for name in sizes
            ]

        status = MemoryStatus()
        status.length = ctypes.sizeof(MemoryStatus)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
        return status.total / 2**30
    return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30


def _gpu() -> tuple[str, int | None]:
    """The GPU torch sees, or why none, and torch's CPU threads"""
    torch = sys.modules.get("torch")
    if torch is None:
        return "not checked (torch not loaded)", None
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none seen by torch"
    return gpu, torch.get_num_threads()


def _probe(read: Any) -> Any:
    """A value of the machine, or "unknown" if it cannot be read here"""
    try:
        return read()
    except Exception:
        return "unknown"


def machine() -> dict[str, Any]:
    """Operating system, CPU, RAM and GPU of the run"""
    probed = _probe(_gpu)
    gpu, threads = probed if probed != "unknown" else ("unknown", None)
    ram = _probe(_ram_gib)
    return {
        "os": platform.platform(),
        "cpu": _probe(_cpu_name),
        "logical_cores": os.cpu_count(),
        "torch_threads": threads,
        "ram_gib": round(ram, 1) if isinstance(ram, float) else ram,
        "gpu": gpu,
    }


def stage_rows() -> list[dict[str, Any]]:
    """One row per stage that ran or that the README gives a figure for"""
    rows = []
    for stage in [*_runs, *(name for name in README_CLAIMS if name not in _runs)]:
        runs = _runs.get(stage, [])
        claim, conditions = README_CLAIMS.get(stage, (None, None))
        seconds = mean(runs) if runs else None
        difference = 100 * (seconds - claim) / claim if seconds is not None and claim else None
        rows.append({
            "stage": stage,
            "seconds_mean": None if seconds is None else round(seconds, 3),
            "seconds_min": round(min(runs), 3) if runs else None,
            "seconds_max": round(max(runs), 3) if runs else None,
            "seconds_total": round(sum(runs), 3),
            "runs": len(runs),
            "readme_claim_seconds": claim,
            "readme_claim_conditions": conditions,
            "difference_percent": None if difference is None else round(difference, 1),
        })
    return rows


def build_report() -> dict[str, Any]:
    return {
        "date": datetime.now().astimezone().isoformat(timespec="seconds"),
        "total_seconds": round(perf_counter() - _started, 1),
        "machine": machine(),
        "stages": stage_rows(),
    }


def _number(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f}"


def to_markdown(report: dict[str, Any]) -> str:
    machine = report["machine"]
    lines = [
        "# Execution time report",
        "",
        "Rewritten by every run of the program (`shelfvision.timing.write_report`); do not edit by hand.",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| Date | {report['date']} |",
        f"| Total time | {report['total_seconds']:.1f} s, since `shelfvision.timing` was imported |",
        f"| OS | {machine['os']} |",
        f"| CPU | {machine['cpu']}, {machine['logical_cores']} logical cores, "
        f"{machine['torch_threads']} torch threads |",
        f"| RAM | {machine['ram_gib']} GiB |",
        f"| GPU | {machine['gpu']} |",
        "",
        "## Stages",
        "",
        "Time is the mean of the runs, one run per photo for the model stages. Difference is against the",
        "README's figure, measured under the conditions listed below the table.",
        "",
        "| Stage | Time (s) | Runs | README claim | Difference (%) |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in report["stages"]:
        claim = "-" if row["readme_claim_seconds"] is None else f"{row['readme_claim_seconds']} s"
        if row["difference_percent"] is not None:
            difference = f"{row['difference_percent']:+.1f}"
        else:
            difference = "not run" if row["runs"] == 0 else "-"
        lines.append(
            f"| {row['stage']} | {_number(row['seconds_mean'])} | {row['runs']} | {claim} | {difference} |"
        )
    lines += ["", "README conditions:", ""]
    lines += [f"- {row['stage']}: {row['readme_claim_conditions']}" for row in report["stages"]
              if row["readme_claim_conditions"]]
    return "\n".join(lines) + "\n"


def write_report(folder: Path | None = None) -> Path | None:
    """Overwrite the report in folder (docs/ of the project by default); return the .md, None on failure"""
    try:
        if folder is None:
            root = project_root()
            if root is None:
                raise FileNotFoundError("no pyproject.toml above shelfvision")
            folder = root / "docs"
        report = build_report()
        folder.mkdir(parents=True, exist_ok=True)
        markdown = folder / f"{REPORT_NAME}.md"
        markdown.write_text(to_markdown(report), encoding="utf-8")
        (folder / f"{REPORT_NAME}.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    except Exception as error:
        warnings.warn(f"The execution time report was not written: {error}", stacklevel=2)
        return None
    return markdown
