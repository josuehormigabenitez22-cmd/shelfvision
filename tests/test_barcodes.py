"""Tests of the barcode reader, on widths and lines created on the fly and with no model."""

import cv2
import numpy as np
import pytest

from shelfvision.barcodes import (
    FFT_WINDOW,
    MIN_QUIET_MODULES,
    MIN_RUNS,
    SYMBOL_PAIRS,
    WIDE_DARK_MODULES,
    best_read,
    line_to_runs,
    read_profile,
    read_runs,
    smooth,
    symbol_errors,
)
from shelfvision.code128 import SYMBOL_WIDTHS, CodeSet, encode, symbols_to_widths
from shelfvision.synthetic import DARK, LIGHT, render_profile

MODULE_PX = 3.0
START_B = 104
# Symbol 74 ("j") followed by a bar of 2 modules (" ") looks like the stop, and what comes
# before it is a valid code by itself: "A" with "B" as its checksum
HIDDEN_PREFIX = encode([(CodeSet.B, "ABj Box")])


def widths_of(symbols: tuple[int, ...]) -> np.ndarray:
    return np.array(symbols_to_widths(symbols), dtype=float) * MODULE_PX


def test_pairs_tell_all_the_symbols_apart():
    assert len({tuple(pairs) for pairs in SYMBOL_PAIRS.tolist()}) == len(SYMBOL_WIDTHS)


@pytest.mark.parametrize("module", [1.0, 2.7, 10.0])
def test_exact_widths_match_their_own_symbol_at_any_scale(module):
    for symbol, widths in enumerate(SYMBOL_WIDTHS):
        errors = symbol_errors(np.array(widths, dtype=float) * module)
        assert errors.argmin() == symbol
        assert errors[symbol] == pytest.approx(0.0)


@pytest.mark.parametrize("ink", [-1.2, 1.2])
def test_ink_does_not_change_the_distance(ink):
    for symbol, widths in enumerate(SYMBOL_WIDTHS):
        measured = np.array(widths, dtype=float) * MODULE_PX
        measured[::2] += ink
        measured[1::2] -= ink
        assert symbol_errors(measured)[symbol] == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("segments", "text"),
    [
        ([(CodeSet.B, "Box")], "Box"),
        ([(CodeSet.C, "1234")], "1234"),
        ([(CodeSet.A, "AB")], "AB"),
        ([(CodeSet.B, "X"), (CodeSet.C, "00"), (CodeSet.B, "Y")], "X00Y"),
    ],
)
def test_exact_widths_give_the_text(segments, text):
    read = read_runs(widths_of(encode(segments)))
    assert read is not None
    assert read.text == text
    assert read.error == pytest.approx(0.0)


def test_symbol_that_looks_like_the_stop_does_not_end_the_reading():
    # The prototype this reader replaces lost every code like this one
    assert read_runs(widths_of(encode([(CodeSet.B, "aj Box")]))).text == "aj Box"


def test_valid_code_hidden_at_the_start_is_not_read():
    assert read_runs(widths_of(HIDDEN_PREFIX)).text == "ABj Box"


def test_stop_needs_a_quiet_zone_after_it():
    widths = widths_of(encode([(CodeSet.B, "Box")]))
    tight = np.array([*widths, (MIN_QUIET_MODULES - 1) * MODULE_PX, MODULE_PX])
    wide = np.array([*widths, (MIN_QUIET_MODULES + 1) * MODULE_PX, MODULE_PX])
    assert read_runs(tight) is None
    assert read_runs(wide).text == "Box"


def test_narrow_margin_before_a_wide_dark_area_also_ends_the_code():
    widths = widths_of(encode([(CodeSet.B, "Box")]))
    margin = (MIN_QUIET_MODULES - 3) * MODULE_PX
    object_edge = np.array([*widths, margin, WIDE_DARK_MODULES * MODULE_PX])
    code_bar = np.array([*widths, margin, (WIDE_DARK_MODULES - 4) * MODULE_PX])
    assert read_runs(object_edge).text == "Box"
    assert read_runs(code_bar) is None


def test_code_is_found_after_other_bars():
    widths = widths_of(encode([(CodeSet.B, "Box")]))
    assert read_runs(np.array([MODULE_PX, 20 * MODULE_PX, *widths])).text == "Box"


@pytest.mark.parametrize(
    "runs",
    [
        np.array([]),
        np.full(10, MODULE_PX),
        np.full(60, MODULE_PX),
        widths_of(encode([(CodeSet.B, "Box")]))[:-7],
        widths_of((START_B, 34, 79, 88, 46)),
    ],
)
def test_runs_without_a_code_give_nothing(runs):
    assert read_runs(runs) is None


def line_of(text, **effects):
    return render_profile(symbols_to_widths(encode([(CodeSet.B, text)])), MODULE_PX, **effects)


def texts_of(profile):
    return {read.text for read in read_profile(profile)}


def test_flat_line_has_no_runs():
    assert line_to_runs(np.full(200, LIGHT), 100.0) is None


@pytest.mark.parametrize("first_pixel", [LIGHT, DARK])
def test_runs_begin_at_a_whole_bar(first_pixel):
    line = np.tile([LIGHT, LIGHT, DARK, DARK, DARK], MIN_RUNS + 2)
    runs = line_to_runs(np.array([first_pixel, *line]), 100.0)
    assert runs[:2].tolist() == [3.0, 2.0]


@pytest.mark.parametrize("module", [2.7, 3.5, 5.0])
@pytest.mark.parametrize("blur", [0.0, 0.7, 1.2])
@pytest.mark.parametrize("ink", [-0.5, 0.0, 0.5])
def test_reads_a_line_with_any_module_blur_and_ink(module, blur, ink):
    widths = symbols_to_widths(encode([(CodeSet.B, "Box-42")]))
    assert texts_of(render_profile(widths, module, blur, ink)) == {"Box-42"}


def test_reads_a_line_upside_down():
    assert texts_of(line_of("Box-42")[::-1]) == {"Box-42"}


def test_reads_upside_down_when_the_line_ends_inside_a_bar():
    # The last run is then a space, and a reversed reading has to skip it to begin with a bar
    assert texts_of(np.array([*line_of("Box-42")[::-1], DARK, DARK])) == {"Box-42"}


def test_reads_under_uneven_light():
    line = line_of("Box-42", blur_px=0.7)
    # A shadow across the line leaves no single gray level between dark and light
    assert texts_of(line * np.linspace(0.25, 1.0, line.size)) == {"Box-42"}


def test_reads_with_noise():
    assert texts_of(line_of("Box-42", blur_px=0.7, noise=3.0, rng=np.random.default_rng(0))) == {"Box-42"}


@pytest.mark.parametrize(
    "profile",
    [
        np.array([]),
        np.full(MIN_RUNS - 1, LIGHT),
        np.full(500, LIGHT),
        np.random.default_rng(0).normal(128.0, 40.0, 2000),
        line_of("Box-42")[: line_of("Box-42").size // 2],
    ],
)
def test_lines_without_a_whole_code_give_nothing(profile):
    assert texts_of(profile) == set()


@pytest.mark.parametrize(("module", "blur"), [(1.25, 0.0), (2.0, 1.4)])
def test_enlarging_and_sharpening_rescue_hard_lines(module, blur):
    # The first line needs edges between pixels, and the second also has its thin bars washed out
    widths = symbols_to_widths(encode([(CodeSet.B, "Box-42")]))
    assert texts_of(render_profile(widths, module, blur)) == {"Box-42"}


def test_best_read_of_nothing_is_nothing():
    assert best_read([]) is None


def test_best_read_prefers_the_right_value_over_a_wrong_one_that_passes_the_checksum():
    # With this blur and ink, one of the attempts on this line reads two wrong symbols
    widths = symbols_to_widths(encode([(CodeSet.B, "xHXCdrZwOR36k")]))
    line = render_profile(widths, 3.5, 1.6, 0.5, 3.0, np.random.default_rng(1))[::-1]
    readings = list(read_profile(line))
    assert {reading.text for reading in readings} == {"xHXCdrZwOR36k", "xHXCdyZw(R36k"}
    assert best_read(readings).text == "xHXCdrZwOR36k"


@pytest.mark.parametrize("length", [8000, 20800])
def test_the_fft_blur_matches_opencv_on_long_lines(length):
    line = np.random.default_rng(7).uniform(20, 230, length).astype(np.float32)
    window = (length // 12) | 1
    expected = cv2.GaussianBlur(line.reshape(-1, 1), (1, window), 0).ravel()
    assert window > FFT_WINDOW
    assert np.abs(smooth(line, window) - expected).max() < 1e-3
