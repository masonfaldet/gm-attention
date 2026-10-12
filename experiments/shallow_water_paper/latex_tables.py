"""What the reproducibility bundles' ``tables.py`` generators share:
cell formats, the bold rule, centred labels, and a structural check."""

from __future__ import annotations

import re
from collections.abc import Callable

Pair = tuple[float, float] | list[float]

CHANNEL_SYMBOLS = {"geopotential": r"$\varphi$", "vorticity": r"$\zeta$", "divergence": r"$\delta$"}
CHANNELS = tuple(CHANNEL_SYMBOLS)


#: Scaled values at or above this are written as $m \times 10^{e}$.
SCIENTIFIC_FROM = 1e4


def number(value: float, digits: int) -> str:
    """A scaled value as fixed digits, or as ``$m\\times10^{e}$`` once it
    is too large for the column."""
    if abs(value) < SCIENTIFIC_FROM:
        return f"{value:.{digits}f}"
    mantissa, exponent = f"{value:.1e}".split("e")
    return rf"${mantissa}\times10^{{{int(exponent)}}}$"


def inline_cell(pair: Pair | None, scale: float, digits: int, bold: bool = False, seeds: int | None = None) -> str:
    """``mean $\\pm$ sd`` on one line; with ``seeds`` 1 the sd gives way to
    ``(1 seed)``, and ``pair`` None (a quantity the row cannot have) is a dash."""
    if pair is None:
        return "--"
    mean, sd = pair
    text = number(mean * scale, digits) + (" (1 seed)" if seeds == 1 else rf" $\pm$ {number(sd * scale, digits)}")
    return rf"{{\bfseries\boldmath {text}}}" if bold else text


def stacked_cell(pair: Pair | None, scale: float, digits: int, bold: bool = False, seeds: int | None = None) -> str:
    """The mean with ``$\\pm$ sd`` beneath it in small type, for tables
    with many columns. With ``seeds`` 1 the second line says so instead of
    a zero sd; ``pair`` None (a quantity the row cannot have) is a dash."""
    if pair is None:
        return "--"
    mean, sd = pair
    top = number(mean * scale, digits)
    if bold:
        top = rf"{{\bfseries {top}}}"
    below = "1 seed" if seeds == 1 else rf"$\pm${number(sd * scale, digits)}"
    return rf"\begin{{tabular}}[c]{{@{{}}c@{{}}}}{top}\\[-1pt]{{\scriptsize {below}}}\end{{tabular}}"


def centered(label: str) -> str:
    """A column label centred over its (right-aligned) numbers."""
    return rf"\multicolumn{{1}}{{c}}{{{label}}}"


def labels(*items: str) -> str:
    return " & ".join(centered(item) for item in items)


def winners(summary: dict, arms: list[str], pick: Callable[[dict], Pair]) -> set[str]:
    """The arm with the lowest mean (exact ties share the bold)."""
    best = min(pick(summary[a])[0] for a in arms)
    return {a for a in arms if pick(summary[a])[0] == best}


def check(tex: str) -> None:
    """Every data row has as many cells as its tabular's column spec, and
    environments and braces balance. Nested one-column tabulars (stacked
    cells) are removed before counting."""
    assert tex.count(r"\begin{table}") == tex.count(r"\end{table}")
    assert tex.count("{") == tex.count("}")
    flat = re.sub(r"\\begin\{tabular\}\[c\]\{@\{\}c@\{\}\}.*?\\end\{tabular\}", "CELL", tex, flags=re.DOTALL)
    pattern = (
        r"\\begin\{tabular\*?\}(?:\{\\textwidth\})?\{(?:@\{\\extracolsep\{\\fill\}\})?([lcr]+)(?:@\{\})?\}"
        r"(.*?)\\end\{tabular\*?\}"
    )
    for block in re.findall(pattern, flat, flags=re.DOTALL):
        spec, body = block
        for line in body.splitlines():
            if line.endswith(r"\\") and r"\multicolumn" not in line:
                assert line.count("&") + 1 == len(spec), (spec, line)
