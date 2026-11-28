"""NoMem's logo and budget glyphs, drawn with text only (no image files).

* ``WORDMARK``: the thin box-drawing "nomem" over a budget-bar underline,
  used in the README, the docs and ``nomem logo``.
* ``banner()``: the same mark with a few lines of status beside it, printed
  when ``nomem serve`` or ``nomem mcp`` starts.
* ``gauge()`` and ``budget_map()``: the token-budget readouts used by
  :meth:`nomem.ContextResult.explain`.

Colour is used only when asked for (``color=True``); the CLI asks when the
stream is a terminal and ``NO_COLOR`` is unset.
"""

from __future__ import annotations

import os
import sys
from typing import Optional, TextIO

_NO = ["┏┓╻┏━┓", "┃┗┫┃ ┃", "╹ ╹┗━┛"]
_MEM = ["┏┳┓┏━╸┏┳┓", "┃┃┃┣╸ ┃┃┃", "╹ ╹┗━╸╹ ╹"]
_BAR_FULL, _BAR_EMPTY = "▀" * 12, "▔" * 3
TAGLINE = "LLM-free memory for agents"

WORDMARK = "\n".join(a + b for a, b in zip(_NO, _MEM)) + f"\n{_BAR_FULL}{_BAR_EMPTY} {TAGLINE}"

TEAL, BRIGHT, AMBER, VIOLET, GREY = (
    "\033[38;5;43m", "\033[38;5;86m", "\033[38;5;214m", "\033[38;5;141m", "\033[38;5;240m")
DIM, BOLD, RESET = "\033[2m", "\033[1m", "\033[0m"


def use_color(stream: TextIO) -> bool:
    return "NO_COLOR" not in os.environ and hasattr(stream, "isatty") and stream.isatty()


def _paint(text: str, code: str, color: bool) -> str:
    return f"{code}{text}{RESET}" if color and text else text


def wordmark(color: bool = False) -> str:
    rows = [_paint(a, TEAL, color) + _paint(b, BOLD, color) for a, b in zip(_NO, _MEM)]
    rows.append(_paint(_BAR_FULL, TEAL, color) + _paint(_BAR_EMPTY, GREY, color)
                + " " + _paint(TAGLINE, DIM, color))
    return "\n".join(rows)


def banner(lines: list[str], version: str = "", color: bool = False) -> str:
    """The wordmark with up to four status lines beside it."""
    text = ([_paint(f"v{version}", DIM, color)] if version else []) + list(lines)
    if len(text) < 4:
        text.append(TAGLINE)
    text += [""] * (4 - len(text))
    marks = wordmark(color).split("\n")
    marks[3] = _paint(_BAR_FULL, TEAL, color) + _paint(_BAR_EMPTY, GREY, color)
    return "\n".join(f"{m}  {_paint(t, DIM, color) if i else t}".rstrip()
                     for i, (m, t) in enumerate(zip(marks, text[:4])))


def print_banner(lines: list[str], stream: Optional[TextIO] = None) -> None:
    """Print the startup banner to ``stream`` (stderr by default, so stdio MCP stays clean)."""
    from . import __version__

    stream = stream or sys.stderr
    print(banner(lines, __version__, use_color(stream)), file=stream, flush=True)


# -- budget readouts ---------------------------------------------------------

# (name, glyph without colour, colour)
SEGMENTS = [
    ("core", "◆", VIOLET),
    ("facts", "●", BRIGHT),
    ("retrieved", "■", TEAL),
    ("summary", "▲", AMBER),
    ("recent", "▪", "\033[38;5;252m"),
]
_FREE = "·"


def gauge(used: int, total: int, width: int = 30, color: bool = False) -> str:
    """``▕██████████████████░░░░▏ 286/300 tokens`` (the share of the budget used)."""
    total = max(total, 1)
    full = min(width, round(width * used / total))
    if used and not full:
        full = 1
    bar = _paint("█" * full, TEAL, color) + _paint("░" * (width - full), GREY, color)
    return f"▕{bar}▏ {used}/{total} tokens"


def _cells(parts: list[tuple[str, int]], total: int, n: int) -> list[str]:
    """Split ``n`` cells across the parts and the free space by token share
    (largest remainder); every non-empty part gets at least one cell."""
    used = sum(t for _, t in parts)
    allp = parts + [("free", max(total - used, 0))]
    whole = max(sum(t for _, t in allp), 1)
    exact = {name: n * t / whole for name, t in allp}
    alloc = {name: int(x) for name, x in exact.items()}
    for name, t in parts:
        if t and not alloc[name]:
            alloc[name] = 1
    while sum(alloc.values()) < n:
        name = max(exact, key=lambda k: exact[k] - alloc[k])
        alloc[name] += 1
    while sum(alloc.values()) > n:
        name = max(alloc, key=lambda k: alloc[k])
        alloc[name] -= 1
    out: list[str] = []
    for name, _ in allp:
        out += [name] * alloc[name]
    return out


def budget_map(usage: dict, total: int, width: int = 30, color: bool = False) -> str:
    """One row of cells, one per ``total/width`` tokens, coloured (or shaped) by tier,
    with a legend: ``◆◆●●●■■■■■■■■■■■■▪▪▪▪▪▪▪▪▪▪▪▪·`` ``◆ core 15  ● facts 28 ...``."""
    parts = [(name, int(usage.get(name, 0))) for name, _, _ in SEGMENTS]
    style = {name: (glyph, code) for name, glyph, code in SEGMENTS}
    row = ""
    for cell in _cells(parts, total, width):
        if cell == "free":
            row += _paint(_FREE, GREY, color)
        else:
            glyph, code = style[cell]
            row += _paint("■" if color else glyph, code, color)
    legend = "  ".join(
        f"{_paint('■' if color else style[name][0], style[name][1], color)} {name} {t}"
        for name, t in parts if t)
    free = max(total - sum(t for _, t in parts), 0)
    legend += f"  {_paint(_FREE, GREY, color)} free {free}"
    return f"{row}  {legend}"
