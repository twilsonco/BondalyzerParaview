"""Read-only diagnostic: turn a saved params file's basin records into xyz loops.

Answers the question the ``maximum_basins`` / ``minimum_basins`` dictionaries
raise: they name the boundary (``saddle_id`` / ``branch_index`` / the far CP id)
and nothing else, and the list is grouped by BOUNDARY SADDLE, not walked around
the ring. This joins each arc to its ``arc_points`` in the connection table
(``min_connections_list`` for a maximum's basin, ``max_connections_list`` for a
minimum's) and chains the arcs into closed loops, so a basin becomes an ordered
polygon on the sphere.

Run with:
    uv run python gba_topology2/inspect_basin_loops.py Pd20k_atom1_params.json
    uv run python gba_topology2/inspect_basin_loops.py Pd20k_atom1_params.json --basin E-MAX_MC31
    uv run python gba_topology2/inspect_basin_loops.py Pd20k_atom1_params.json --basin E-MAX_MC31 --dump loop.json
    uv run python gba_topology2/inspect_basin_loops.py Pd20k_atom1_params.json --list
"""

import argparse
import json
import math
import os
import sys
from typing import Any

workspace_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)

from gba_topology2.src.basin_loop import (
    basin_boundary_loops,
    chain_boundary_arcs,
    endpoint_key_for,
)


def main() -> None:
    parser = argparse.ArgumentParser(prog="inspect_basin_loops.py")
    parser.add_argument("params_file", nargs="?", default="Pd20k_atom1_params.json")
    parser.add_argument(
        "--basin",
        type=str,
        default=None,
        help="Basin CP id, e.g. E-MAX_MC31. Default: every basin.",
    )
    parser.add_argument(
        "--polarity",
        choices=("maximum", "minimum"),
        default=None,
        help="Which basin table. Default: inferred from the id's polarity.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List the basins in the file and exit.",
    )
    parser.add_argument(
        "--dump",
        type=str,
        default=None,
        metavar="OUT.json",
        help="Write the loops as JSON: [[{id, polarity, closed, points}]].",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Per-basin lines only (no walk detail).",
    )
    args = parser.parse_args()

    try:
        with open(args.params_file, encoding="utf-8") as fh:
            payload: dict[str, Any] = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"cannot read {args.params_file}: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    derived: dict[str, Any] = payload.get("derived_metrics", payload)

    tables = {
        side: (derived.get(f"{side}_basins") or {}) for side in ("maximum", "minimum")
    }
    if args.list:
        for side, basins in tables.items():
            print(f"{side}: {len(basins)} basins")
            for bid in sorted(basins):
                row = basins[bid]
                print(
                    f"  {bid:16s} arcs={row.get('n_boundary_arcs', 0):3d} "
                    f"saddles={len(row.get('boundary_saddles') or [])} "
                    f"ends={len(row.get(f'boundary_{_end_name(side)}s') or [])}"
                )
        return

    targets: list[tuple[str, str]] = []
    if args.basin is not None:
        polarity = args.polarity or _guess_polarity(args.basin, tables)
        targets.append((args.basin, polarity))
    else:
        for side, basins in tables.items():
            if args.polarity is None or args.polarity == side:
                targets.extend((bid, side) for bid in sorted(basins))

    records: list[dict[str, Any]] = []
    for basin_id, polarity in targets:
        try:
            loops = basin_boundary_loops(derived, basin_id, polarity)
        except ValueError as exc:
            print(f"{basin_id}: SKIP {exc}")
            continue
        arcs = tables[polarity][basin_id]["boundary_arcs"]
        print(
            f"{basin_id} ({polarity}): {len(arcs)} boundary arcs -> "
            f"{len(loops)} loop(s)"
        )
        for li, pts in enumerate(loops):
            closed = len(pts) > 2 and _close_gap(pts) <= 1e-9
            print(
                f"  loop {li}: {len(pts)} pts  "
                f"closure_gap={_close_gap(pts):.2e}  closed={closed}"
            )
        if not args.quiet:
            for walk in _walks(arcs, polarity):
                for i, (s, b, e, fwd) in enumerate(walk):
                    tag = "saddle->end" if fwd else "end->saddle (REVERSED)"
                    print(f"   {i}: {s}[{b}] -> {e}   {tag}")
        records.append(
            {
                "basin_id": basin_id,
                "polarity": polarity,
                "loops": [[[list(p) for p in pts]] for pts in loops],
            }
        )

    if args.dump:
        try:
            with open(args.dump, "w", encoding="utf-8") as fh:
                json.dump(records, fh, indent=1)
        except OSError as exc:
            print(f"cannot write {args.dump}: {exc}", file=sys.stderr)
            raise SystemExit(1) from exc
        print(f"wrote {args.dump}")


def _end_name(polarity: str) -> str:
    """Field stem of the boundary endpoint set for a basin polarity."""
    return "min" if polarity == "maximum" else "max"


def _guess_polarity(basin_id: str, tables: dict[str, dict[str, Any]]) -> str:
    """Pick the table that contains ``basin_id``, defaulting on the id's shape."""
    for side in ("maximum", "minimum"):
        if basin_id in tables[side]:
            return side
    upper = basin_id.upper()
    if "MAX" in upper:
        return "maximum"
    return "minimum"


def _close_gap(pts: list[tuple[float, float, float]]) -> float:
    """Distance from a loop's last vertex back to its first."""
    if len(pts) < 2:
        return 0.0
    a, b = pts[0], pts[-1]
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    dz = a[2] - b[2]
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def _walks(
    arcs: list[dict[str, Any]], polarity: str
) -> list[list[tuple[str, int, str, bool]]]:
    """Re-expose the walk as readable (saddle, branch, endpoint, forward) tuples."""
    return [
        [(w.saddle_id, w.branch_index, w.endpoint_id, w.forward) for w in walk]
        for walk in chain_boundary_arcs(arcs, endpoint_key_for(polarity))
    ]


if __name__ == "__main__":
    main()
