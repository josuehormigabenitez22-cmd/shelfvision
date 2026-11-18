"""Tests of the Code128 tables and of the conversion of symbols to text."""

from itertools import permutations
from random import Random

import pytest

from shelfvision.code128 import (
    CHECKSUM_MODULUS,
    SET_CHARACTERS,
    STOP_WIDTHS,
    SYMBOL_WIDTHS,
    CodeSet,
    checksum,
    encode,
    symbols_to_text,
    symbols_to_widths,
)

START_A, START_B, START_C = 103, 104, 105
SHIFT, TO_C, FNC1 = 98, 99, 102
# Values computed by hand from the definition of the checksum, so that they do not depend on the code
BOX = (START_B, 34, 79, 88, 45)
DIGITS = (START_C, 12, 34, 82)
CONTROL = (START_A, 33, 74, 78)
# Example of the Code 128 article in Wikipedia: Start C, FNC1, 42 18 40 20 50, Code A, 0 and check symbol 92
PUBLIC_EXAMPLE = (START_C, FNC1, 42, 18, 40, 20, 50, 101, 16, 92)
CHARACTERS = {code_set: "".join(map(chr, codes)) for code_set, codes in SET_CHARACTERS.items()}
RANDOM_SEED = 128
RANDOM_CODES = 2000
MAX_RANDOM_SEGMENTS = 5


def with_checksum(*symbols: int) -> tuple[int, ...]:
    return (*symbols, checksum(symbols))


def test_every_symbol_has_eleven_modules_and_is_unique():
    assert len(SYMBOL_WIDTHS) == 106
    assert {sum(widths) for widths in SYMBOL_WIDTHS} == {11}
    assert len(set(SYMBOL_WIDTHS)) == len(SYMBOL_WIDTHS)
    assert sum(STOP_WIDTHS) == 13


def test_every_symbol_has_an_even_number_of_bar_modules():
    # Rule of the symbology: it catches a mistyped width in the table
    assert all(sum(widths[::2]) % 2 == 0 for widths in SYMBOL_WIDTHS)


@pytest.mark.parametrize(
    ("symbols", "text"),
    [
        (BOX, "Box"),
        (DIGITS, "1234"),
        (CONTROL, "A\n"),
        (PUBLIC_EXAMPLE, "42184020500"),
        (with_checksum(START_C, 0, 99), "0099"),
        (with_checksum(START_B, 95), "\x7f"),
        (with_checksum(START_A, 64, 95), "\x00\x1f"),
    ],
)
def test_each_code_set_gives_its_text(symbols, text):
    assert symbols_to_text(symbols) == text


@pytest.mark.parametrize(
    ("symbols", "text"),
    [
        (with_checksum(START_B, 34, TO_C, 12), "B12"),
        (with_checksum(START_C, 12, 100, 34), "12B"),
        (with_checksum(START_C, 12, 101, 74), "12\n"),
        (with_checksum(START_A, 74, 100, 79), "\no"),
        (with_checksum(START_B, 79, 101, 74), "o\n"),
    ],
)
def test_set_changes_last_until_the_next_change(symbols, text):
    assert symbols_to_text(symbols) == text


@pytest.mark.parametrize(
    ("symbols", "text"),
    [
        (with_checksum(START_B, 79, SHIFT, 74, 79), "o\no"),
        (with_checksum(START_A, 33, SHIFT, 79, 33), "AoA"),
        (with_checksum(START_C, SHIFT), "98"),
    ],
)
def test_shift_changes_only_the_next_symbol(symbols, text):
    assert symbols_to_text(symbols) == text


def test_function_symbols_add_no_text():
    assert symbols_to_text(with_checksum(START_C, FNC1, 12)) == "12"
    assert symbols_to_text(with_checksum(START_B, 34, 96, 97, 79)) == "Bo"


@pytest.mark.parametrize("wrong", sorted(set(range(CHECKSUM_MODULUS)) - {BOX[-1]}))
def test_every_altered_checksum_is_rejected(wrong):
    assert symbols_to_text((*BOX[:-1], wrong)) is None


@pytest.mark.parametrize("position", [1, 2, 3])
def test_every_altered_data_symbol_is_rejected(position):
    for symbol in set(range(CHECKSUM_MODULUS)) - {BOX[position]}:
        assert symbols_to_text((*BOX[:position], symbol, *BOX[position + 1 :])) is None


def test_swapped_neighbors_are_rejected():
    assert symbols_to_text((START_B, 79, 34, 88, BOX[-1])) is None


@pytest.mark.parametrize(
    "symbols",
    [
        (),
        (START_B,),
        with_checksum(START_B),
        with_checksum(34, 79),
        with_checksum(START_B, START_A, 34),
        with_checksum(START_B, 100, 34),
        with_checksum(START_A, 101, 34),
        with_checksum(START_B, SHIFT),
        with_checksum(START_B, SHIFT, TO_C, 12),
        with_checksum(START_B, TO_C),
    ],
)
def test_invalid_codes_are_rejected(symbols):
    assert symbols_to_text(symbols) is None


def random_segments(rng: Random) -> list[tuple[CodeSet, str]]:
    segments = []
    for _ in range(rng.randint(1, MAX_RANDOM_SEGMENTS)):
        code_set = rng.choice(list(CodeSet))
        if code_set == CodeSet.C:
            text = "".join(f"{rng.randrange(100):02d}" for _ in range(rng.randint(1, 8)))
        else:
            text = "".join(rng.choices(CHARACTERS[code_set], k=rng.randint(1, 12)))
        segments.append((code_set, text))
    return segments


@pytest.mark.parametrize("code_set", [CodeSet.A, CodeSet.B])
def test_round_trip_of_every_character_in_a_set(code_set):
    assert symbols_to_text(encode([(code_set, CHARACTERS[code_set])])) == CHARACTERS[code_set]


def test_round_trip_of_every_pair_in_set_c():
    digits = "".join(f"{pair:02d}" for pair in range(100))
    assert symbols_to_text(encode([(CodeSet.C, digits)])) == digits


@pytest.mark.parametrize(("first", "second"), list(permutations(CodeSet, 2)))
def test_round_trip_of_every_set_change(first, second):
    samples = {CodeSet.A: "A\n", CodeSet.B: "Bo", CodeSet.C: "1234"}
    segments = [(first, samples[first]), (second, samples[second]), (first, samples[first])]
    assert symbols_to_text(encode(segments)) == "".join(text for _, text in segments)


def test_round_trip_of_random_mixes():
    rng = Random(RANDOM_SEED)
    for _ in range(RANDOM_CODES):
        segments = random_segments(rng)
        assert symbols_to_text(encode(segments)) == "".join(text for _, text in segments)


def test_encoder_gives_the_symbols_of_public_examples():
    # Both ways of encoding "X00Y" in the Code 128 article in Wikipedia
    assert encode([(CodeSet.B, "X00Y")])[:-1] == (START_B, 56, 16, 16, 57)
    mixed = [(CodeSet.B, "X"), (CodeSet.C, "00"), (CodeSet.B, "Y")]
    assert encode(mixed)[:-1] == (START_B, 56, TO_C, 0, 100, 57)
    assert encode([(CodeSet.B, "Box")]) == BOX


@pytest.mark.parametrize(
    "segments",
    [
        [],
        [(CodeSet.B, "")],
        [(CodeSet.C, "123")],
        [(CodeSet.C, "12a4")],
        [(CodeSet.C, "١٢")],
        [(CodeSet.A, "a")],
        [(CodeSet.B, "\n")],
        [(CodeSet.B, "ñ")],
    ],
)
def test_text_that_does_not_fit_is_refused(segments):
    with pytest.raises(ValueError):
        encode(segments)


def test_widths_of_a_whole_code():
    widths = symbols_to_widths(BOX)
    assert widths[:6] == (2, 1, 1, 2, 1, 4)
    assert widths[-7:] == STOP_WIDTHS
    assert len(widths) == 6 * len(BOX) + 7
    assert sum(widths) == 11 * len(BOX) + 13
