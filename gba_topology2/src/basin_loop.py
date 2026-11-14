"""Chain a basin's boundary arcs into ordered xyz loops.

A basin record (``derived_metrics.maximum_basins`` / ``.minimum_basins`` in a
saved params file, the output of ``serialize_maximum_basins`` /
``serialize_minimum_basins``) names its boundary and nothing else: each
``boundary_arcs`` entry is ``{saddle_id, branch_index, min_id}`` (or
``max_id``). The polyline is in the connection table -- ``arc_points`` on the
row with the same ``(saddle_id, branch_index)``.

That join is the easy half. The order is the real work: ``boundary_arcs`` is
grouped by BOUNDARY SADDLE (the basin collectors sort the saddles and append
each saddle's arcs), so the list reads ``S1[b0], S1[b1], S2[b0], S2[b1], ...``
-- a per-saddle grouping, NOT a walk around the ring. Concatenating it draws
eight independent spurs, not one closed curve.

The ring is a bipartite graph whose nodes are the boundary saddles and the
boundary endpoints (minima, for a maximum's basin; maxima, for a minimum's).
Every arc runs saddle -> endpoint, and the v12 cycle rule
(:func:`gba_topology2.src.basin_boundary._boundary_cycle`) makes every node
even-degree: a basin's boundary is a closed separatrix cycle, so each maximum
on it is met by exactly two bounding saddles. Walk that graph and the arcs come
out in order. Each arc is stored saddle-first, so a step that leaves a node
that is an ARC ENDPOINT must traverse the next arc BACKWARDS -- the reverse of
its stored ``arc_points`` -- to stay continuous.

Degenerate shape handled: when the boundary is a pair of parallel branches
(ethene3's E-MIN_MC3 -> MAX_4864, twice from each of two saddles), the walk
closes early into a LENS -- the two branches of one saddle, out and back -- and
the remaining arcs start a second loop. The pairing is reported as loops, never
spliced into one open chain, and an arc that cannot close is reported as an
open walk instead of being dropped.

Run with:
    uv run python gba_topology2/inspect_basin_loops.py Pd20k_atom1_params.json
"""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

__all__ = [
    "POINT_TOL",
    "Point",
    "WalkedArc",
    "basin_boundary_loops",
    "chain_boundary_arcs",
    "endpoint_key_for",
    "table_key_for",
    "walk_to_points",
]

#: A position on the sphere, in the same world frame as ``arc_points``.
Point = tuple[float, float, float]

#: Two consecutive arc endpoints closer than this are ONE vertex. The arc
#: endpoints are the CP centres, so the true gap is 0; the slack is for the
#: raw (non-effective) CPs, whose arc tip sits ~2e-6 off the tabulated centre.
POINT_TOL = 1e-4

# Which id a boundary arc names as its far end, and which table carries its
# polyline, for each polarity of BASIN CENTRE. The duality of basin v11: a
# minimum is bounded by ascending saddle->maximum arcs, a maximum by descending
# saddle->minimum arcs.
_ENDPOINT_KEYS: dict[str, str] = {"minimum": "max_id", "maximum": "min_id"}
_TABLE_KEYS: dict[str, str] = {
    "minimum": "max_connections_list",
    "maximum": "min_connections_list",
}


@dataclass(frozen=True)
class WalkedArc:
    """One boundary arc with the direction the walk traversed it.

    Attributes:
        saddle_id: Source saddle's id, as stored on the arc.
        branch_index: Index of the branch at that saddle.
        endpoint_id: Far CP id (``min_id`` for a maximum's basin, ``max_id``
            for a minimum's).
        forward: True when the walk ran saddle -> endpoint, i.e. in the order
            ``arc_points`` is stored. False when it ran endpoint -> saddle and
            the polyline must be REVERSED.
    """

    saddle_id: str
    branch_index: int
    endpoint_id: str
    forward: bool

    @property
    def key(self) -> tuple[str, int]:
        """The connection-table key of this arc."""
        return (self.saddle_id, self.branch_index)


def endpoint_key_for(polarity: str) -> str:
    """Far-end field name on a boundary arc for a basin-centre polarity.

    Args:
        polarity: ``"minimum"`` or ``"maximum"``.

    Returns:
        ``"max_id"`` for a minimum's basin, ``"min_id"`` for a maximum's.

    Raises:
        ValueError: Unknown polarity.
    """
    try:
        return _ENDPOINT_KEYS[polarity]
    except KeyError as exc:
        raise ValueError(f"unknown basin polarity {polarity!r}") from exc


def table_key_for(polarity: str) -> str:
    """Connection-table field name carrying ``arc_points`` for a polarity.

    Args:
        polarity: ``"minimum"`` or ``"maximum"``.

    Returns:
        ``"max_connections_list"`` for a minimum's basin,
        ``"min_connections_list"`` for a maximum's.

    Raises:
        ValueError: Unknown polarity.
    """
    try:
        return _TABLE_KEYS[polarity]
    except KeyError as exc:
        raise ValueError(f"unknown basin polarity {polarity!r}") from exc


def chain_boundary_arcs(
    boundary_arcs: Sequence[Mapping[str, Any]],
    endpoint_key: str,
) -> list[list[WalkedArc]]:
    """Order a basin's boundary arcs into closed walks around the ring.

    Args:
        boundary_arcs: The ``boundary_arcs`` list of one serialized basin, in
            any order (the collected, per-saddle order is the norm).
        endpoint_key: ``"min_id"`` (maximum basin) or ``"max_id"`` (minimum
            basin) -- the field naming the arc's far CP.

    Returns:
        One list of :class:`WalkedArc` per closed loop, in traversal order,
        each arc carrying the direction it was walked. A normal basin is one
        loop; a parallel-branch boundary is several lenses. Arcs whose endpoint
        is missing (a stalled branch, which ``_boundary_cycle`` only keeps in
        the no-cycle fallback) come back as single-arc walks at the end, so the
        caller sees them instead of losing them.
    """
    arcs: list[tuple[str, int, str]] = []
    stalled: list[WalkedArc] = []
    for arc in boundary_arcs:
        saddle_id = str(arc.get("saddle_id", ""))
        branch_index = int(arc.get("branch_index", 0))
        far = arc.get(endpoint_key)
        if far is None:
            stalled.append(WalkedArc(saddle_id, branch_index, "", forward=True))
            continue
        arcs.append((saddle_id, branch_index, str(far)))

    at_saddle: dict[str, list[int]] = {}
    at_endpoint: dict[str, list[int]] = {}
    for i, (saddle_id, _branch, endpoint_id) in enumerate(arcs):
        at_saddle.setdefault(saddle_id, []).append(i)
        at_endpoint.setdefault(endpoint_id, []).append(i)

    used: set[int] = set()
    loops: list[list[WalkedArc]] = []

    while len(used) < len(arcs):
        start = next(i for i in range(len(arcs)) if i not in used)
        used.add(start)
        saddle_id, branch_index, endpoint_id = arcs[start]
        walk = [WalkedArc(saddle_id, branch_index, endpoint_id, forward=True)]
        # The walk opens at the arc's saddle and is closed when it gets back.
        home = ("saddle", saddle_id)
        node = ("endpoint", endpoint_id)
        while node != home:
            kind, nid = node
            pool = at_saddle if kind == "saddle" else at_endpoint
            cands = [i for i in pool.get(nid, []) if i not in used]
            if not cands:
                break
            nxt = cands[0]
            used.add(nxt)
            s2, b2, f2 = arcs[nxt]
            # Leaving a SADDLE continues outward (saddle -> endpoint, forward);
            # leaving an ENDPOINT climbs the other arc back to its saddle.
            forward = kind == "saddle"
            walk.append(WalkedArc(s2, b2, f2, forward=forward))
            node = ("endpoint", f2) if forward else ("saddle", s2)
        loops.append(walk)

    loops.extend([arc] for arc in stalled)
    return loops


def walk_to_points(
    walk: Sequence[WalkedArc],
    arc_points_by_key: Mapping[tuple[str, int], Sequence[Sequence[float]]],
    *,
    joint_tol: float = POINT_TOL,
) -> list[Point]:
    """Materialise one walk into a polyline of world-frame xyz points.

    Args:
        walk: Consecutive :class:`WalkedArc` from :func:`chain_boundary_arcs`.
        arc_points_by_key: ``{(saddle_id, branch_index): arc_points}``, the
            connection table indexed by the arc key.
        joint_tol: Distance under which a shared endpoint is collapsed to a
            single vertex.

    Returns:
        The ordered points. Consecutive arcs share their joint vertex, and a
        closed walk ends exactly where it began, so the last point duplicates
        the first -- drop it for a polygon ring, keep it for a closed polyline.

    Raises:
        KeyError: The walk names an arc the table does not have. The basin
            records the topology; the table owns the geometry, and a missing
            row means the two disagree. Silently skipping it would draw a
            boundary with a hole in it.
    """
    pts: list[Point] = []
    for arc in walk:
        try:
            raw = arc_points_by_key[arc.key]
        except KeyError as exc:
            raise KeyError(
                f"no arc_points for boundary arc {arc.saddle_id}[{arc.branch_index}]"
            ) from exc
        seg: list[Point] = [(float(p[0]), float(p[1]), float(p[2])) for p in raw]
        if not arc.forward:
            seg.reverse()
        if pts and seg and _dist(pts[-1], seg[0]) <= joint_tol:
            seg = seg[1:]
        pts.extend(seg)
    return pts


def basin_boundary_loops(
    derived_metrics: Mapping[str, Any],
    basin_id: str,
    polarity: str,
) -> list[list[Point]]:
    """Ordered xyz loops of one basin's boundary, straight from a params file.

    The whole recipe in one call: pick the basin out of ``maximum_basins`` /
    ``minimum_basins``, join its ``boundary_arcs`` against the connection table
    that carries ``arc_points``, and chain them around the ring.

    Args:
        derived_metrics: The ``derived_metrics`` block of a saved
            ``*_params.json`` (the top-level object of a ``diagnose_params``
            results file works too -- it holds the same keys).
        basin_id: The CP id naming the basin, e.g. ``"E-MAX_MC31"``.
        polarity: ``"maximum"`` or ``"minimum"`` -- which basin table to read.

    Returns:
        One point list per closed loop, in traversal order.

    Raises:
        ValueError: Unknown polarity, or the basin is not in that table (an
            id given under the wrong polarity lands here too).
        KeyError: A boundary arc has no row in the connection table.
    """
    table_name = table_key_for(polarity)
    basins = derived_metrics.get(f"{polarity}_basins") or {}
    basin = basins.get(basin_id)
    if basin is None:
        raise ValueError(
            f"basin {basin_id!r} is not in derived_metrics.{polarity}_basins "
            f"(that table has {len(basins)} basins)"
        )
    rows = derived_metrics.get(table_name) or []
    by_key: dict[tuple[str, int], Sequence[Sequence[float]]] = {}
    for row in rows:
        by_key[(str(row.get("saddle_id", "")), int(row.get("branch_index", 0)))] = (
            row.get("arc_points") or []
        )
    return [
        walk_to_points(walk, by_key)
        for walk in chain_boundary_arcs(
            basin.get("boundary_arcs") or [], endpoint_key_for(polarity)
        )
    ]


def _dist(a: Point, b: Point) -> float:
    """Euclidean distance between two world-frame points."""
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    dz = a[2] - b[2]
    return math.sqrt(dx * dx + dy * dy + dz * dz)
