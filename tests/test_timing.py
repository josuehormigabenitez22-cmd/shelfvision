"""The execution time report, written to a temporary folder from hand-made stage times"""

import json
import warnings

import pytest

from shelfvision import timing


@pytest.fixture(autouse=True)
def fresh_runs():
    timing.reset()
    yield
    timing.reset()


CLAIM = timing.README_CLAIMS["read barcodes"][0]


def write(tmp_path):
    # 5 % and 15 % above the README: 10 % on average
    timing.record("read barcodes", 1.05 * CLAIM)
    timing.record("read barcodes", 1.15 * CLAIM)
    timing.record("load image", 0.08)
    with timing.timed("same queries again"):
        pass
    markdown = timing.write_report(tmp_path)
    assert markdown == tmp_path / "execution-time-report.md"
    data = json.loads((tmp_path / "execution-time-report.json").read_text(encoding="utf-8"))
    return markdown.read_text(encoding="utf-8"), data


def test_markdown_has_every_field(tmp_path):
    text, _ = write(tmp_path)
    for field in ("| Date |", "| Total time |", "| OS |", "| CPU |", "| RAM |", "| GPU |"):
        assert field in text
    for absent in ("Commit", "commit", "git"):
        assert absent not in text
    assert "| Stage | Time (s) | Runs | README claim | Difference (%) |" in text
    assert f"| read barcodes | {1.1 * CLAIM:.3f} | 2 | {CLAIM} s | +10.0 |" in text
    assert "| load image | 0.080 | 1 | - | - |" in text
    sam_claim = timing.README_CLAIMS["encode image (SAM)"][0]
    assert f"| encode image (SAM) | - | 0 | {sam_claim} s | not run |" in text


def test_json_has_the_same_data(tmp_path):
    _, report = write(tmp_path)
    assert set(report) == {"date", "total_seconds", "machine", "stages"}
    assert set(report["machine"]) == {"os", "cpu", "logical_cores", "torch_threads", "ram_gib", "gpu"}
    stages = {row["stage"]: row for row in report["stages"]}
    barcodes = stages["read barcodes"]
    assert (barcodes["runs"], barcodes["difference_percent"]) == (2, 10.0)
    assert barcodes["seconds_mean"] == pytest.approx(1.1 * CLAIM, abs=1e-3)
    assert barcodes["seconds_min"] == pytest.approx(1.05 * CLAIM, abs=1e-3)
    assert barcodes["seconds_max"] == pytest.approx(1.15 * CLAIM, abs=1e-3)
    assert stages["same queries again"]["runs"] == 1
    assert set(timing.README_CLAIMS) <= set(stages)


def test_date_is_iso_8601(tmp_path):
    from datetime import datetime

    _, report = write(tmp_path)
    assert datetime.fromisoformat(report["date"]).tzinfo is not None


def test_project_root_is_found_from_the_package_not_the_working_folder(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    root = timing.project_root()
    assert root is not None and (root / "pyproject.toml").is_file()
    assert (root / "src" / "shelfvision" / "timing.py").is_file()


def test_a_report_that_cannot_be_written_only_warns(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("not a folder")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert timing.write_report(blocker) is None
    assert any("execution time report was not written" in str(item.message) for item in caught)
