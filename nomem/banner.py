"""NoMem's logo, drawn with text only (no image files).

``WORDMARK`` is the large logo for READMEs and ``nomem logo``; ``chip()`` is
the compact mark printed when ``nomem serve`` or ``nomem mcp`` starts.
Colour is used only when the stream is a terminal and ``NO_COLOR`` is unset.
"""

from __future__ import annotations

import os
import sys
from typing import Optional, TextIO

_NO = [
    "█▀▀▄ ▄▀▀▄ ",
    "█  █ █  █ ",
    "▀  ▀  ▀▀  ",
]
_MEM = [
    "█▀▄▀▄ ▄▀▀▄ █▀▄▀▄",
    "█ █ █ █▀▀▀ █ █ █",
    "▀   ▀  ▀▀▀ ▀   ▀",
]
TAGLINE = "LLM-free memory for agents"

WORDMARK = "\n".join((a + b).rstrip() for a, b in zip(_NO, _MEM)) + "\n" + TAGLINE

_TEAL, _BRIGHT, _DIM, _BOLD, _RESET = "\033[38;5;43m", "\033[38;5;86m", "\033[2m", "\033[1m", "\033[0m"


def use_color(stream: TextIO) -> bool:
    return "NO_COLOR" not in os.environ and hasattr(stream, "isatty") and stream.isatty()


def wordmark(color: bool = False) -> str:
    if not color:
        return WORDMARK
    rows = [f"{_TEAL}{a}{_RESET}{_BOLD}{b}{_RESET}" for a, b in zip(_NO, _MEM)]
    return "\n".join(rows) + f"\n{_DIM}{TAGLINE}{_RESET}"


def chip(lines: list[str], title: str = "nomem", version: str = "", color: bool = False) -> str:
    """The compact chip mark with up to three lines of text beside it."""
    t, b, d, z = (_TEAL, _BRIGHT, _DIM, _RESET) if color else ("", "", "", "")
    bold = _BOLD if color else ""
    head = f"{bold}{title}{z}" + (f" {d}v{version}{z}" if version else "")
    text = [head] + list(lines) + ["", ""]
    rows = [
        f"{d} ╷╷╷╷╷{z}",
        f"{t}▐█▀▀▀█▌{z}  {text[0]}",
        f"{t}▐█ {b}◆{t} █▌{z}  {d}{text[1]}{z}",
        f"{t}▐█▄▄▄█▌{z}  {d}{text[2]}{z}",
        f"{d} ╵╵╵╵╵{z}",
    ]
    return "\n".join(r.rstrip() for r in rows)


def print_chip(lines: list[str], stream: Optional[TextIO] = None) -> None:
    """Print the chip banner to ``stream`` (stderr by default, so stdio MCP stays clean)."""
    from . import __version__

    stream = stream or sys.stderr
    print(chip(lines, version=__version__, color=use_color(stream)), file=stream, flush=True)
