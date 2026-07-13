"""Console output: a run header, an aligned per-epoch table, and a closing summary.

Column sets differ between learners and between fluid and non-fluid runs, so the table
is described declaratively as a list of (heading, width, format) and rendered from that.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

RULE = "─"
WIDTH = 78

# (heading, width, format spec, unit suffix)
Column = Tuple[str, int, str, str]


def header(title: str, rows: Sequence[Tuple[str, str]]) -> None:
    print()
    print(f"  {title}")
    print(f"  {RULE * (WIDTH - 2)}")
    for label, value in rows:
        print(f"  {label:<14}{value}")
    print(f"  {RULE * (WIDTH - 2)}")
    print()


def arch(input_dim: int, hidden: Sequence[int], num_classes: int, fluid_grid: Optional[int]) -> str:
    parts: List[str] = [str(input_dim)]
    if fluid_grid is not None:
        parts.append(f"[fluid {fluid_grid}x{fluid_grid}]")
    parts += [str(h) for h in hidden]
    parts.append(str(num_classes))
    return " -> ".join(parts)


def table_header(cols: Sequence[Column]) -> None:
    print("  " + "".join(f"{name:>{w}}" for name, w, _, _ in cols))
    print("  " + "".join(f"{RULE * (w - 2):>{w}}" for _, w, _, _ in cols))


def table_row(cols: Sequence[Column], values: Dict[str, object]) -> None:
    cells = []
    for name, w, fmt, unit in cols:
        v = values.get(name)
        cells.append(f"{'-':>{w}}" if v is None else f"{format(v, fmt) + unit:>{w}}")
    print("  " + "".join(cells))


def summary(rows: Sequence[Tuple[str, str]]) -> None:
    print(f"  {RULE * (WIDTH - 2)}")
    for label, value in rows:
        print(f"  {label:<14}{value}")
    print()
