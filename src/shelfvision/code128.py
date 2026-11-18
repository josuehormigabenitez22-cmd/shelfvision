"""Code128 symbology: symbol tables, checksum and conversion of symbols to text"""
from collections.abc import Sequence
from enum import StrEnum
from itertools import chain
from string import digits
from typing import Final


class CodeSet(StrEnum):
    A = "A"  # Uppercase and control characters
    B = "B"  # Uppercase and lowercase
    C = "C"  # Pairs of digits


# Bar and space widths of symbols 0 to 105, in modules. Every symbol starts with a bar
_PATTERNS: Final = (
    "212222", "222122", "222221", "121223", "121322", "131222", "122213", "122312", "132212", "221213",
    "221312", "231212", "112232", "122132", "122231", "113222", "123122", "123221", "223211", "221132",
    "221231", "213212", "223112", "312131", "311222", "321122", "321221", "312212", "322112", "322211",
    "212123", "212321", "232121", "111323", "131123", "131321", "112313", "132113", "132311", "211313",
    "231113", "231311", "112133", "112331", "132131", "113123", "113321", "133121", "313121", "211331",
    "231131", "213113", "213311", "213131", "311123", "311321", "331121", "312113", "312311", "332111",
    "314111", "221411", "431111", "111224", "111422", "121124", "121421", "141122", "141221", "112214",
    "112412", "122114", "122411", "142112", "142211", "241211", "221114", "413111", "241112", "134111",
    "111242", "121142", "121241", "114212", "124112", "124211", "411212", "421112", "421211", "212141",
    "214121", "412121", "111143", "111341", "131141", "114113", "114311", "411113", "411311", "113141",
    "114131", "311141", "411131", "211412", "211214", "211232",
)
SYMBOL_WIDTHS: Final = tuple(tuple(int(width) for width in pattern) for pattern in _PATTERNS)
# The stop symbol has a seventh width: the bar that closes the code
STOP_WIDTHS: Final = (2, 3, 3, 1, 1, 1, 2)

CHECKSUM_MODULUS: Final = 103
# A start symbol, at least one data symbol and the checksum
MIN_SYMBOLS: Final = 3
START_SETS: Final = {103: CodeSet.A, 104: CodeSet.B, 105: CodeSet.C}
SWITCHES: Final = {
    CodeSet.A: {99: CodeSet.C, 100: CodeSet.B},
    CodeSet.B: {99: CodeSet.C, 101: CodeSet.A},
    CodeSet.C: {100: CodeSet.B, 101: CodeSet.A},
}
# Changes the set of the next symbol only
SHIFT: Final = 98
SHIFTED_SETS: Final = {CodeSet.A: CodeSet.B, CodeSet.B: CodeSet.A}
# FNC1, FNC2 and FNC3 add no text. FNC4 is left out on purpose: it changes the meaning
# of the characters after it, so a code that uses it is rejected instead of misread
SILENT_SYMBOLS: Final = {CodeSet.A: {96, 97, 102}, CodeSet.B: {96, 97, 102}, CodeSet.C: {102}}
ASCII_OFFSET: Final = 32
# In set A, the symbols from here up to FIRST_FUNCTION_SYMBOL are the control characters
FIRST_CONTROL_SYMBOL: Final = 64
FIRST_FUNCTION_SYMBOL: Final = 96
PAIRS_IN_SET_C: Final = 100
# Character codes that each set can encode
SET_CHARACTERS: Final = {CodeSet.A: range(96), CodeSet.B: range(32, 128)}


def checksum(symbols: Sequence[int]) -> int:
    """Return the check symbol of a start symbol followed by data symbols"""
    # The start symbol counts once, the rest are weighted by their position (1, 2, 3...)
    weighted = sum(position * symbol for position, symbol in enumerate(symbols))
    return (symbols[0] + weighted) % CHECKSUM_MODULUS


def _character(symbol: int, code_set: CodeSet) -> str | None:
    """Return the text of a symbol in a code set, or None if the symbol holds no text"""
    if code_set == CodeSet.C:
        return f"{symbol:02d}" if symbol < PAIRS_IN_SET_C else None
    if symbol >= FIRST_FUNCTION_SYMBOL:
        return None
    if code_set == CodeSet.A and symbol >= FIRST_CONTROL_SYMBOL:
        return chr(symbol - FIRST_CONTROL_SYMBOL)
    return chr(symbol + ASCII_OFFSET)


def _data_to_text(data: Sequence[int], code_set: CodeSet) -> str | None:
    """Translate the data symbols, or return None if they break the rules of Code128"""
    characters: list[str] = []
    shifted = False
    # Go symbol by symbol, keeping track of the current set and of any shift
    for symbol in data:
        character = _character(symbol, SHIFTED_SETS[code_set] if shifted else code_set)
        if character is not None:
            characters.append(character)
            shifted = False
        elif shifted:
            return None  # A shift must be followed by a character
        elif symbol == SHIFT:
            shifted = True
        elif symbol in SWITCHES[code_set]:
            code_set = SWITCHES[code_set][symbol]
        elif symbol not in SILENT_SYMBOLS[code_set]:
            return None
    return None if shifted else "".join(characters)


def symbols_to_text(symbols: Sequence[int]) -> str | None:
    """Return the text of start, data and checksum symbols, or None if the code is not valid"""
    if len(symbols) < MIN_SYMBOLS or symbols[0] not in START_SETS or checksum(symbols[:-1]) != symbols[-1]:
        return None
    return _data_to_text(symbols[1:-1], START_SETS[symbols[0]]) or None


# The pipeline never encodes: what follows only serves to create labels for tests and benchmarks
def _text_to_data(text: str, code_set: CodeSet) -> list[int]:
    """Return the data symbols of a text in one code set, or raise ValueError if it does not fit """
    if code_set == CodeSet.C:
        if len(text) % 2 or not set(text) <= set(digits):
            raise ValueError(f"Set C needs an even number of digits, got: {text!r}")
        return [int(text[index : index + 2]) for index in range(0, len(text), 2)]
    if invalid := [character for character in text if ord(character) not in SET_CHARACTERS[code_set]]:
        raise ValueError(f"Set {code_set} cannot encode: {invalid!r}")
    # Set A wraps around: its control characters, below the offset, go after its other characters
    return [(ord(character) - ASCII_OFFSET) % FIRST_FUNCTION_SYMBOL for character in text]


def encode(segments: Sequence[tuple[CodeSet, str]]) -> tuple[int, ...]:
    """Return the start, data and checksum symbols of (code set, text) segments, changing set between them"""
    if not any(text for _, text in segments):
        raise ValueError("A code needs at least one character")
    current = segments[0][0]
    symbols = [next(symbol for symbol, code_set in START_SETS.items() if code_set == current)]
    for code_set, text in segments:
        if code_set != current:
            symbols.append(next(symbol for symbol, target in SWITCHES[current].items() if target == code_set))
            current = code_set
        symbols.extend(_text_to_data(text, code_set))
    return (*symbols, checksum(symbols))


def symbols_to_widths(symbols: Sequence[int]) -> tuple[int, ...]:
    """Return the bar and space widths of a whole code in modules, stop included"""
    return (*chain.from_iterable(SYMBOL_WIDTHS[symbol] for symbol in symbols), *STOP_WIDTHS)
