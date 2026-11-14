"""Tests for chaining a basin's boundary arcs into ordered xyz loops.

A basin record names its boundary by ``(saddle_id, branch_index)`` and the far
CP id; the polyline lives in the connection table, and the arc list is grouped
by boundary saddle, NOT walked around the ring. These cover the join, the walk
order, the reversal of backward-traversed arcs, and the shapes that occur in
real snapshots: parallel branches (ethene3's hill MAX_4864, whose boundary
closes into two lenses rather than one ring), an open chain, and a stalled arc
with no endpoint.

The ``_MC31`` fixture is the 8-arc ring of ``E-MAX_MC31`` out of a real Pd20k
snapshot, relabelled: saddle S1/S2/S3/S4 for E-SAD_MC1/MC2/MC9/MC7, minima
M_a/M_b/M_c/M_d for MIN_2 / E-MIN_MC11 / MIN_6 / E-MIN_MC13.
"""

import itertools
import random
from collections.abc import Mapping, Sequence
from typing import Any

import pytest

from gba_topology2.src.basin_loop import (
    POINT_TOL,
    WalkedArc,
    basin_boundary_loops,
    chain_boundary_arcs,
    endpoint_key_for,
    table_key_for,
    walk_to_points,
)


def _arc(saddle: str, branch: int, min_id: str | None) -> dict[str, Any]:
    """A serialized boundary arc of a MAXIMUM's basin (far end = ``min_id``)."""
    return {"saddle_id": saddle, "branch_index": branch, "min_id": min_id}


def _arc_max(saddle: str, branch: int, max_id: str | None) -> dict[str, Any]:
    """A serialized boundary arc of a MINIMUM's basin (far end = ``max_id``)."""
    return {"saddle_id": saddle, "branch_index": branch, "max_id": max_id}


# Grouped by saddle, as the basin collectors write it, so a naive concatenation
# is eight spurs from four saddles, not a loop.
_MC31: list[dict[str, Any]] = [
    _arc("S1", 0, "M_a"),
    _arc("S1", 1, "M_b"),
    _arc("S2", 0, "M_b"),
    _arc("S2", 1, "M_c"),
    _arc("S3", 0, "M_a"),
    _arc("S3", 1, "M_d"),
    _arc("S4", 0, "M_d"),
    _arc("S4", 1, "M_c"),
]


def _nodes(walk: Sequence[WalkedArc]) -> list[str]:
    """The CP nodes a walk visits, start node first (n+1 names for n arcs)."""
    nodes = [walk[0].saddle_id]
    for arc in walk:
        nodes.append(arc.endpoint_id if arc.forward else arc.saddle_id)
    return nodes


def _edges(walk: Sequence[WalkedArc]) -> set[frozenset[str]]:
    """The node pairs a walk covers, ignoring the direction it walked them."""
    return {frozenset({a.saddle_id, a.endpoint_id}) for a in walk}


def _ladder_table(
    arcs: Sequence[Mapping[str, Any]],
    endpoint_key: str = "min_id",
) -> dict[tuple[str, int], Sequence[Sequence[float]]]:
    """Give every arc a 3-point polyline: saddle at index 0, endpoint at index 2.

    Each CP gets a distinct x, so a reversed arc is visible in the point order
    and a shared joint is testable.
    """
    saddles = sorted({str(a["saddle_id"]) for a in arcs})
    ends = sorted({str(a[endpoint_key]) for a in arcs if a.get(endpoint_key)})
    sx = {name: float(i + 1) for i, name in enumerate(saddles)}
    ex = {name: float(i + 10) for i, name in enumerate(ends)}
    table: dict[tuple[str, int], Sequence[Sequence[float]]] = {}
    for a in arcs:
        s = str(a["saddle_id"])
        e = str(a[endpoint_key])
        table[(s, int(a["branch_index"]))] = [
            (sx[s], 0.0, 0.0),
            (0.5 * (sx[s] + ex[e]), 1.0, 0.0),
            (ex[e], 0.0, 0.0),
        ]
    return table


# ---------------------------------------------------------------------------
# The polarity duality: which field names the far end, which table has the arc
# ---------------------------------------------------------------------------


def test_the_two_polarities_read_the_opposite_tables() -> None:
    """A minimum is bounded by ASCENT arcs; a maximum by DESCENT arcs (basin v11)."""
    assert endpoint_key_for("minimum") == "max_id"
    assert endpoint_key_for("maximum") == "min_id"
    assert table_key_for("minimum") == "max_connections_list"
    assert table_key_for("maximum") == "min_connections_list"


def test_an_unknown_polarity_is_refused() -> None:
    """A typo lands loudly, not as an empty boundary."""
    with pytest.raises(ValueError, match="unknown basin polarity"):
        endpoint_key_for("saddle")
    with pytest.raises(ValueError, match="unknown basin polarity"):
        table_key_for("saddle")


def test_walked_arc_key_is_the_table_key() -> None:
    """The walk's lookup key is (saddle, branch), never the far id."""
    assert WalkedArc("S1", 2, "M_a", forward=True).key == ("S1", 2)


# ---------------------------------------------------------------------------
# The walk
# ---------------------------------------------------------------------------


def test_the_ring_chains_into_one_closed_walk_in_geographic_order() -> None:
    """The per-saddle list becomes one ring, walked in the ring's own order.

    The node sequence is the assertion: it is the answer to "which arc do I
    read next", and it is NOT the input order.
    """
    loops = chain_boundary_arcs(_MC31, "min_id")
    assert len(loops) == 1
    walk = loops[0]
    assert len(walk) == len(_MC31)
    assert _nodes(walk) == ["S1", "M_a", "S3", "M_d", "S4", "M_c", "S2", "M_b", "S1"]
    # Direction alternates: out of a saddle, back down the next arc to its saddle.
    assert [a.forward for a in walk] == [True, False] * 4


def test_every_arc_is_used_exactly_once() -> None:
    """The walk covers the boundary, with no repeat and no drop."""
    walk = chain_boundary_arcs(_MC31, "min_id")[0]
    keys = [(a.saddle_id, a.branch_index) for a in walk]
    assert len(keys) == len(set(keys))
    assert set(keys) == {(str(a["saddle_id"]), int(a["branch_index"])) for a in _MC31}


def test_consecutive_arcs_share_the_node_the_walk_passed_through() -> None:
    """Continuity, stated on the arcs: each step starts where the last ended."""
    walk = chain_boundary_arcs(_MC31, "min_id")[0]
    for prev, nxt in itertools.pairwise(walk):
        here = prev.endpoint_id if prev.forward else prev.saddle_id
        there = nxt.saddle_id if nxt.forward else nxt.endpoint_id
        assert here == there, f"{prev} -> {nxt} is a jump"


def test_the_walk_is_independent_of_the_input_arc_order() -> None:
    """The loop is a property of the arc SET.

    The starting arc (and so the direction of travel) follows the input order,
    which is why the invariant is the node pairs and the closure, not the
    sequence. ``find_*_basins`` sorts boundary saddles to keep the record
    order-stable; the walk must not need that to be stable.
    """
    reference = chain_boundary_arcs(_MC31, "min_id")[0]
    for seed in range(20):
        shuffled = list(_MC31)
        random.Random(seed).shuffle(shuffled)
        loops = chain_boundary_arcs(shuffled, "min_id")
        assert len(loops) == 1, f"seed {seed} split the ring"
        walk = loops[0]
        assert _edges(walk) == _edges(reference)
        nodes = _nodes(walk)
        assert nodes[0] == nodes[-1], f"seed {seed} left the ring open"
        assert len(nodes) == len(_MC31) + 1


def test_two_saddles_sharing_one_far_cp_close_into_lenses() -> None:
    """ethene3's E-MIN_MC3: 2 saddles, 2 branches each, all into MAX_4864.

    The boundary is not one ring: each saddle's two parallel branches, out and
    back, is its own closed lens. Two loops of two arcs -- never one chain that
    splices across the hill.
    """
    arcs = [
        _arc_max("SA", 0, "F"),
        _arc_max("SA", 1, "F"),
        _arc_max("SB", 0, "F"),
        _arc_max("SB", 1, "F"),
    ]
    loops = chain_boundary_arcs(arcs, "max_id")
    assert [len(w) for w in loops] == [2, 2]
    for walk in loops:
        assert _nodes(walk)[0] == _nodes(walk)[-1]
        assert walk[0].forward and not walk[1].forward
        assert walk[0].saddle_id == walk[1].saddle_id
        assert walk[0].endpoint_id == walk[1].endpoint_id == "F"
    assert _edges(loops[0]) == {frozenset({"SA", "F"})}
    assert _edges(loops[1]) == {frozenset({"SB", "F"})}


def test_a_single_saddle_and_its_far_cp_is_one_lens() -> None:
    """The smallest boundary the rule can emit: 2 arcs, 1 closed loop."""
    loops = chain_boundary_arcs([_arc("S", 0, "M"), _arc("S", 1, "M")], "min_id")
    assert len(loops) == 1
    assert _nodes(loops[0]) == ["S", "M", "S"]


def test_an_open_chain_is_reported_open_and_a_single_arc_is_itself() -> None:
    """A 4-arc PATH (degrees 1,2,2,1) cannot close: report the chain, not a cut.

    This is the ``_boundary_cycle`` no-cycle fallback shape, plus the degenerate
    one-arc boundary and the empty basin.
    """
    path = [
        _arc("S1", 0, "M_a"),
        _arc("S2", 0, "M_a"),
        _arc("S2", 1, "M_b"),
        _arc("S3", 0, "M_b"),
    ]
    loops = chain_boundary_arcs(path, "min_id")
    assert [len(w) for w in loops] == [4]
    assert _nodes(loops[0]) == ["S1", "M_a", "S2", "M_b", "S3"]
    assert _nodes(loops[0])[0] != _nodes(loops[0])[-1]

    assert [len(w) for w in chain_boundary_arcs([_arc("S", 0, "M")], "min_id")] == [1]
    assert chain_boundary_arcs([], "min_id") == []


def test_a_stalled_arc_is_appended_as_its_own_walk_not_dropped() -> None:
    """``min_id`` None survives the walk, last, as a single-arc open walk.

    ``_boundary_cycle`` drops arcs with no target and keeps the raw candidate
    list only when nothing survives -- so a None endpoint reaches here on that
    path, and a boundary that loses an arc silently is a hole in the curve.
    """
    arcs = [_arc("S1", 0, None), _arc("S1", 1, "M_a"), _arc("S2", 0, "M_a")]
    loops = chain_boundary_arcs(arcs, "min_id")
    assert [len(w) for w in loops] == [2, 1]
    assert _nodes(loops[0]) == ["S1", "M_a", "S2"]
    stall = loops[-1][0]
    assert (stall.saddle_id, stall.branch_index, stall.endpoint_id) == ("S1", 0, "")


def test_a_repeated_arc_is_walked_once_per_input_row() -> None:
    """The walk keys arcs by (saddle, branch), so a duplicate row is a twin.

    ``min_connections_list`` can hold two rows with the same key, which the
    serializer's LAST-WINS index collapses. The walk uses each row of the basin
    record once; the geometry it reads is the table's surviving row.
    """
    arcs = [
        _arc("S1", 0, "M_a"),
        _arc("S1", 0, "M_a"),
        _arc("S2", 0, "M_a"),
        _arc("S2", 1, "M_b"),
    ]
    loops = chain_boundary_arcs(arcs, "min_id")
    assert sum(len(w) for w in loops) == 4


# ---------------------------------------------------------------------------
# Joining to arc_points
# ---------------------------------------------------------------------------


def test_backward_arcs_are_reversed_so_the_polyline_stays_continuous() -> None:
    """arc_points is stored saddle-first; a min->saddle step walks it backwards."""
    arcs = [_arc("S1", 0, "M_a"), _arc("S2", 0, "M_a")]
    table = _ladder_table(arcs)
    pts = walk_to_points(chain_boundary_arcs(arcs, "min_id")[0], table)
    # S1 -> M_a, then M_a -> S2 down the second arc in reverse.
    assert pts[0] == (1.0, 0.0, 0.0)  # S1
    assert pts[2] == (10.0, 0.0, 0.0)  # M_a, the joint
    assert pts[-1] == (2.0, 0.0, 0.0)  # S2, the reversed arc's stored start
    assert len(pts) == 5, "the joint vertex is counted once"


def test_the_ring_polyline_closes_on_its_first_point() -> None:
    """Closed walk -> last point equals the first, so it draws as a ring."""
    pts = walk_to_points(chain_boundary_arcs(_MC31, "min_id")[0], _ladder_table(_MC31))
    assert pts[0] == pts[-1]
    assert len(pts) == 2 * len(_MC31) + 1


def test_a_joint_inside_tolerance_is_merged_beyond_it_is_kept() -> None:
    """The dedupe threshold is CP-centre slack, not a licence to skip a gap."""
    walk = [WalkedArc("S1", 0, "M_a", True), WalkedArc("S2", 0, "M_a", False)]
    # S1 ends at x=1; arc S2->M_a is stored saddle-first, so walked backwards it
    # OPENS at M_a. Within tolerance, that opening point is the same vertex.
    tight: dict[tuple[str, int], Sequence[Sequence[float]]] = {
        ("S1", 0): [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)],
        ("S2", 0): [(2.0, 0.0, 0.0), (1.0 + POINT_TOL / 2, 0.0, 0.0)],
    }
    assert walk_to_points([walk[0]], tight) == [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)]
    assert walk_to_points(walk, tight) == [
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (2.0, 0.0, 0.0),
    ]
    # A joint wider than the tolerance is a real gap, and stays visible.
    wide = dict(tight)
    wide[("S2", 0)] = [(2.0, 0.0, 0.0), (1.0 + 10 * POINT_TOL, 0.0, 0.0)]
    assert len(walk_to_points(walk, wide)) == 4


def test_a_missing_arc_row_raises_rather_than_punching_a_hole() -> None:
    """The basin records topology, the table owns geometry; disagreement is loud."""
    arcs = [_arc("S1", 0, "M_a"), _arc("S2", 0, "M_a")]
    table = {("S1", 0): [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)]}
    walk = chain_boundary_arcs(arcs, "min_id")[0]
    with pytest.raises(KeyError, match=r"S2\[0\]"):
        walk_to_points(walk, table)


def test_an_empty_arc_row_contributes_no_points() -> None:
    """A verified arc can carry an empty polyline; the walk keeps going.

    It is reported as a gap by the caller's closure check, not faked.
    """
    table: dict[tuple[str, int], Sequence[Sequence[float]]] = {
        ("S1", 0): [],
        ("S2", 0): [(2.0, 0.0, 0.0), (3.0, 0.0, 0.0)],
    }
    arcs = [_arc("S1", 0, "M_a"), _arc("S2", 0, "M_a")]
    pts = walk_to_points(chain_boundary_arcs(arcs, "min_id")[0], table)
    assert pts == [(3.0, 0.0, 0.0), (2.0, 0.0, 0.0)]


# ---------------------------------------------------------------------------
# The one-call path over a params-shaped document
# ---------------------------------------------------------------------------


def _params_doc() -> dict[str, Any]:
    """A minimal ``*_params.json`` shape: one maximum basin + its descent table."""
    arcs = list(_MC31)
    return {
        "schema_version": 12,
        "derived_metrics": {
            "maximum_basins": {
                "E-MAX_MC31": {"boundary_arcs": arcs, "n_boundary_arcs": len(arcs)}
            },
            "minimum_basins": {
                "E-MIN_MC3": {"boundary_arcs": [], "n_boundary_arcs": 0}
            },
            "min_connections_list": [
                {
                    "saddle_id": a["saddle_id"],
                    "branch_index": a["branch_index"],
                    "min_id": a["min_id"],
                    "arc_points": _ladder_table(arcs)[
                        (str(a["saddle_id"]), int(a["branch_index"]))
                    ],
                }
                for a in arcs
            ],
            "max_connections_list": [],
        },
    }


def _derived() -> Mapping[str, Any]:
    """The ``derived_metrics`` block of the fixture document."""
    doc: Mapping[str, Any] = _params_doc()["derived_metrics"]
    return doc


def test_basin_boundary_loops_joins_chains_and_orders_in_one_call() -> None:
    """The documented recipe, end to end, against a params-shaped document."""
    loops = basin_boundary_loops(_derived(), "E-MAX_MC31", "maximum")
    assert len(loops) == 1
    assert len(loops[0]) == 2 * len(_MC31) + 1
    assert loops[0][0] == loops[0][-1]


def test_basin_boundary_loops_names_the_table_it_needs() -> None:
    """An id asked of the wrong basin table says so, with the table's size."""
    derived = _derived()
    with pytest.raises(ValueError, match="is not in derived_metrics.minimum_basins"):
        basin_boundary_loops(derived, "E-MAX_MC31", "minimum")
    with pytest.raises(ValueError, match="unknown basin polarity"):
        basin_boundary_loops(derived, "E-MAX_MC31", "saddle")


def test_basin_boundary_loops_reaches_the_table_for_the_points() -> None:
    """An empty connection table is a KeyError, not an empty loop."""
    doc = _params_doc()
    doc["derived_metrics"]["min_connections_list"] = []
    with pytest.raises(KeyError, match=r"S1\[0\]"):
        basin_boundary_loops(doc["derived_metrics"], "E-MAX_MC31", "maximum")


def test_basin_boundary_loops_reads_a_minimum_basin_uphill() -> None:
    """The same call works on the other polarity, through ``max_id``."""
    arcs = [
        _arc_max("S1", 0, "X"),
        _arc_max("S1", 1, "Y"),
        _arc_max("S2", 0, "X"),
        _arc_max("S2", 1, "Y"),
    ]
    table = _ladder_table(arcs, "max_id")
    doc: dict[str, Any] = {
        "minimum_basins": {"E-MIN_1": {"boundary_arcs": arcs}},
        "maximum_basins": {},
        "max_connections_list": [
            {
                "saddle_id": a["saddle_id"],
                "branch_index": a["branch_index"],
                "arc_points": table[(str(a["saddle_id"]), int(a["branch_index"]))],
            }
            for a in arcs
        ],
        "min_connections_list": [],
    }
    loops = basin_boundary_loops(doc, "E-MIN_1", "minimum")
    assert len(loops) == 1
    # 4 arcs x 3 points, one duplicate vertex dropped at each of the 3 internal
    # joints; the 4th (the closing joint) is kept as an exact repeat of the start.
    assert len(loops[0]) == 4 * 3 - 3
    assert loops[0][0] == loops[0][-1]
