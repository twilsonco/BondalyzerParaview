"""Tests for gba_topology2.src.basin_boundary module."""

import json
from dataclasses import asdict, replace
from typing import Any

import pytest

from gba_topology2.src.basin_boundary import (
    MaximumBasin,
    MinimumBasin,
    build_connection_graph,
    build_descent_graph,
    connection_from_state_dict,
    find_maximum_basins,
    find_minimum_basins,
    get_basin_arcs,
    get_max_basin_arcs,
    min_connection_from_state_dict,
    min_connection_to_state_dict,
    serialize_connection_graph,
    serialize_maximum_basins,
    serialize_minimum_basins,
)
from gba_topology2.src.basin_center import BasinCenter
from gba_topology2.src.max_connection import MaxConnection, MinConnection


def _make_connection(
    saddle_id: str,
    max_id: str | None,
    status: str = "connected",
    branch_index: int = 0,
    verified: bool = True,
    arc_points: tuple[tuple[float, float, float], ...] = (
        (0.0, 0.0, 1.0),
        (1.0, 0.0, 0.0),
    ),
) -> MaxConnection:
    """Helper to create a MaxConnection for testing."""
    return MaxConnection(
        saddle_id=saddle_id,
        branch_index=branch_index,
        max_id=max_id,
        status=status,
        direction_mode="ray",
        cross_track_deg=0.0,
        along_track_deg=1.0,
        separation_deg=10.0,
        mutual=True,
        arc_points=arc_points,
        saddle_value=0.5,
        max_value=1.0 if max_id else None,
        verified=verified,
    )


def _make_descent(
    saddle_id: str,
    min_id: str | None,
    status: str = "connected",
    branch_index: int = 0,
    verified: bool = True,
) -> MinConnection:
    """Helper to create a descent record (the basin v10 adjacency)."""
    return MinConnection(
        saddle_id=saddle_id,
        branch_index=branch_index,
        min_id=min_id,
        status=status,
        direction_mode="port_tangent",
        cross_track_deg=0.0,
        along_track_deg=1.0,
        separation_deg=10.0,
        arc_points=((0.0, 0.0, 1.0), (1.0, 0.0, 0.0)),
        saddle_value=0.5,
        min_value=0.1 if min_id else None,
        verified=verified,
    )


def _make_basin_center(
    center_id: str,
    polarity: str,
    position: tuple[float, float, float],
    value: float = 1.0,
) -> BasinCenter:
    """Helper to create a BasinCenter for testing."""
    return BasinCenter(
        center_id=center_id,
        polarity=polarity,
        position=position,
        value=value,
        is_effective=True,
        vertex_id=-1,
        origin_id=None,
        net_index=1,
        member_count=1,
    )


class TestBuildConnectionGraph:
    """Tests for build_connection_graph()."""

    def test_empty_connections(self) -> None:
        """Empty connection list yields empty graph."""
        graph = build_connection_graph([])
        assert graph == {}

    def test_single_connection(self) -> None:
        """Single connection creates one edge."""
        conn = _make_connection("SAD_1", "MAX_2")
        graph = build_connection_graph([conn])
        assert "SAD_1" in graph
        assert len(graph["SAD_1"]) == 1
        assert graph["SAD_1"][0] == conn

    def test_filters_by_status(self) -> None:
        """Only connected and verified arcs are included."""
        conn_connected = _make_connection("SAD_1", "MAX_2", status="connected")
        conn_occluded = _make_connection("SAD_1", "MAX_3", status="occluded")
        conn_unverified = _make_connection("SAD_1", "MAX_4", verified=False)
        graph = build_connection_graph([conn_connected, conn_occluded, conn_unverified])
        assert "SAD_1" in graph
        assert len(graph["SAD_1"]) == 1
        assert graph["SAD_1"][0] == conn_connected

    def test_multiple_edges_from_saddle(self) -> None:
        """Multiple connections from one saddle create multiple edges."""
        conn1 = _make_connection("SAD_1", "MAX_2", branch_index=0)
        conn2 = _make_connection("SAD_1", "MAX_3", branch_index=1)
        graph = build_connection_graph([conn1, conn2])
        assert "SAD_1" in graph
        assert len(graph["SAD_1"]) == 2
        assert conn1 in graph["SAD_1"]
        assert conn2 in graph["SAD_1"]

    def test_complex_graph(self) -> None:
        """Multiple saddles and maxima create a complex graph."""
        conn1 = _make_connection("SAD_1", "MAX_1")
        conn2 = _make_connection("SAD_1", "MAX_2")
        conn3 = _make_connection("SAD_2", "MAX_1")
        conn4 = _make_connection("SAD_2", "MAX_2")
        graph = build_connection_graph([conn1, conn2, conn3, conn4])
        assert len(graph) == 2
        assert len(graph["SAD_1"]) == 2
        assert len(graph["SAD_2"]) == 2


class TestMinimumBasinDataclass:
    """Tests for MinimumBasin dataclass."""

    def test_create_basin(self) -> None:
        """Create a MinimumBasin with arcs."""
        conn = _make_connection("SAD_1", "MAX_1")
        basin = MinimumBasin(
            minimum_id="MIN_1",
            boundary_arcs=[conn],
        )
        assert basin.minimum_id == "MIN_1"
        assert len(basin.boundary_arcs) == 1
        assert basin.boundary_arcs[0] == conn

    def test_basin_is_frozen(self) -> None:
        """MinimumBasin is immutable after creation."""
        conn = _make_connection("SAD_1", "MAX_1")
        basin = MinimumBasin(
            minimum_id="MIN_1",
            boundary_arcs=[conn],
        )
        with pytest.raises(AttributeError):
            basin.minimum_id = "MIN_2"  # type: ignore


class TestGetBasinArcs:
    """Tests for get_basin_arcs()."""

    def test_extract_arcs(self) -> None:
        """Extract arcs from a basin."""
        conn1 = _make_connection("SAD_1", "MAX_1", branch_index=0)
        conn2 = _make_connection("SAD_2", "MAX_1", branch_index=0)
        basin = MinimumBasin(minimum_id="MIN_1", boundary_arcs=[conn1, conn2])
        arcs = get_basin_arcs(basin)
        assert len(arcs) == 2
        assert arcs[0] == conn1
        assert arcs[1] == conn2

    def test_empty_basin(self) -> None:
        """Basin with no arcs returns empty list."""
        basin = MinimumBasin(minimum_id="MIN_1", boundary_arcs=[])
        arcs = get_basin_arcs(basin)
        assert arcs == []


class TestFindMinimumBasins:
    """Tests for find_minimum_basins()."""

    def test_empty_topology(self) -> None:
        """Empty connections and centers yield no basins."""
        result = find_minimum_basins([], [], [])
        assert result == {}

    def test_single_minimum_with_adjacent_saddles(self) -> None:
        """Saddles whose descent drains into a minimum bound its basin."""
        minimum = _make_basin_center("MIN_1", "minimum", (0.0, 0.0, 0.0), value=0.1)
        maximum = _make_basin_center("MAX_1", "maximum", (0.0, 0.0, 2.0))

        conn1 = _make_connection("SAD_1", "MAX_1")
        conn2 = _make_connection("SAD_2", "MAX_1")
        conn3 = _make_connection("SAD_3", "MAX_1")
        descents = [
            _make_descent("SAD_1", "MIN_1"),
            _make_descent("SAD_2", "MIN_1"),
            _make_descent("SAD_3", "MIN_1"),
        ]

        basin_centers = [minimum, maximum]
        result = find_minimum_basins([conn1, conn2, conn3], basin_centers, descents)

        # basin v10: adjacency is the descent table, so the membership is
        # asserted, not "isinstance" -- every arc of the draining saddles is
        # on the boundary.
        assert set(result) == {"MIN_1"}
        assert [a.saddle_id for a in result["MIN_1"].boundary_arcs] == [
            "SAD_1",
            "SAD_2",
            "SAD_3",
        ]

    def test_no_cycles(self) -> None:
        """Topology with no cycles (isolated minimum) yields no basins."""
        minimum = _make_basin_center("MIN_1", "minimum", (0.0, 0.0, 0.0))
        # No connections to any saddles
        result = find_minimum_basins([], [minimum], [])
        assert result == {}

    def test_disconnected_minimum(self) -> None:
        """A minimum nothing drains into has no basin."""
        minimum1 = _make_basin_center("MIN_1", "minimum", (0.0, 0.0, 0.0))
        minimum2 = _make_basin_center("MIN_2", "minimum", (10.0, 10.0, 10.0))
        maximum = _make_basin_center("MAX_1", "maximum", (0.0, 0.0, 2.0))

        conn = _make_connection("SAD_1", "MAX_1")
        basin_centers = [minimum1, minimum2, maximum]
        result = find_minimum_basins(
            [conn], basin_centers, [_make_descent("SAD_1", "MIN_1")]
        )

        # SAD_1 drains into MIN_1 only: MIN_2 has no boundary, even though the
        # saddle's arc sits spatially near both.
        assert set(result) == {"MIN_1"}
        assert [a.saddle_id for a in result["MIN_1"].boundary_arcs] == ["SAD_1"]

    def test_filters_non_connected_arcs(self) -> None:
        """Only connected arcs are included in basins."""
        minimum = _make_basin_center("MIN_1", "minimum", (0.0, 0.0, 0.0))
        maximum = _make_basin_center("MAX_1", "maximum", (0.0, 0.0, 2.0))

        conn_connected = _make_connection("SAD_1", "MAX_1", status="connected")
        conn_occluded = _make_connection("SAD_1", "MAX_2", status="occluded")

        basin_centers = [minimum, maximum]
        result = find_minimum_basins(
            [conn_connected, conn_occluded],
            basin_centers,
            [_make_descent("SAD_1", "MIN_1")],
        )

        # Graph is built with only connected/verified arcs
        assert set(result) == {"MIN_1"}
        assert [a.max_id for a in result["MIN_1"].boundary_arcs] == ["MAX_1"]

    def test_basin_structure(self) -> None:
        """Basin contains correct arc structure."""
        minimum = _make_basin_center("MIN_1", "minimum", (0.0, 0.0, 0.0))
        maximum = _make_basin_center("MAX_1", "maximum", (0.0, 0.0, 2.0))

        conn = _make_connection("SAD_1", "MAX_1")
        result = find_minimum_basins(
            [conn], [minimum, maximum], [_make_descent("SAD_1", "MIN_1")]
        )

        # Result is a dict mapping minimum_id to MinimumBasin
        assert isinstance(result, dict)
        for basin in result.values():
            assert isinstance(basin, MinimumBasin)
            assert all(isinstance(arc, MaxConnection) for arc in basin.boundary_arcs)


class TestEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_none_max_id(self) -> None:
        """Connection with None max_id is skipped (stalled climb)."""
        minimum = _make_basin_center("MIN_1", "minimum", (0.0, 0.0, 0.0))
        maximum = _make_basin_center("MAX_1", "maximum", (0.0, 0.0, 2.0))

        conn_stalled = _make_connection("SAD_1", None, status="stalled")
        conn_connected = _make_connection("SAD_1", "MAX_1")

        basin_centers = [minimum, maximum]
        result = find_minimum_basins(
            [conn_stalled, conn_connected],
            basin_centers,
            [_make_descent("SAD_1", "MIN_1")],
        )

        # Graph filters out None max_id connections
        assert set(result) == {"MIN_1"}
        assert [a.max_id for a in result["MIN_1"].boundary_arcs] == ["MAX_1"]

    def test_empty_cycle_list(self) -> None:
        """Empty basin list yields empty dict."""
        graph = build_connection_graph([])
        assert graph == {}

    def test_only_maxima_and_saddles(self) -> None:
        """No minima in basin_centers yields no basins."""
        maximum = _make_basin_center("MAX_1", "maximum", (0.0, 0.0, 2.0))

        conn = _make_connection("SAD_1", "MAX_1")
        result = find_minimum_basins([conn], [maximum], [])
        assert result == {}


class TestBasinAdjacencyV10:
    """basin v10: adjacency is DESCENT CONNECTIVITY, never spatial proximity.

    The v9 rule (k=3 nearest saddles by Euclidean distance) saturated: with
    few saddles every basin claimed every arc, and the advice pinned the fix
    as "raise k". These are the rules the replacement must honour.
    """

    def test_adjacency_beats_proximity(self) -> None:
        """The spatially nearest saddle is NOT the boundary; the draining one is."""
        minimum = _make_basin_center("MIN_1", "minimum", (0.0, 0.0, 0.0))
        maximum = _make_basin_center("MAX_1", "maximum", (0.0, 0.0, 2.0))
        near = _make_connection(
            "SAD_NEAR", "MAX_1", arc_points=((0.0, 0.0, 0.2), (1.0, 0.0, 0.0))
        )
        far = _make_connection(
            "SAD_FAR", "MAX_1", arc_points=((0.0, 0.0, 5.0), (1.0, 0.0, 0.0))
        )
        result = find_minimum_basins(
            [near, far], [minimum, maximum], [_make_descent("SAD_FAR", "MIN_1")]
        )
        assert [a.saddle_id for a in result["MIN_1"].boundary_arcs] == ["SAD_FAR"]

    def test_saturation_resolved(self) -> None:
        """Two minima, two saddles, one drain each: the basins are DISJOINT.

        Under proximity with k=3 and only 2 saddles, every basin claimed every
        arc of every saddle -- 2-of-2, the saturation the diagnose panel then
        reported as "raise k". The descent table separates them.
        """
        min_a = _make_basin_center("MIN_A", "minimum", (0.0, 0.0, 0.0))
        min_b = _make_basin_center("MIN_B", "minimum", (0.2, 0.0, 0.0))
        maximum = _make_basin_center("MAX_1", "maximum", (0.0, 0.0, 2.0))
        conn_a = _make_connection("SAD_1", "MAX_1")
        conn_b = _make_connection("SAD_2", "MAX_1")
        result = find_minimum_basins(
            [conn_a, conn_b],
            [min_a, min_b, maximum],
            [_make_descent("SAD_1", "MIN_A"), _make_descent("SAD_2", "MIN_B")],
        )
        assert [a.saddle_id for a in result["MIN_A"].boundary_arcs] == ["SAD_1"]
        assert [a.saddle_id for a in result["MIN_B"].boundary_arcs] == ["SAD_2"]

    def test_empty_descent_table_yields_no_basins(self) -> None:
        """No descent records = no adjacency known = no rings (no proximity fallback).

        The fallback IS what v10 removed: a basin boundary must come from the
        field, never from luck of the proximity list.
        """
        minimum = _make_basin_center("MIN_1", "minimum", (0.0, 0.0, 0.0))
        maximum = _make_basin_center("MAX_1", "maximum", (0.0, 0.0, 2.0))
        conn = _make_connection("SAD_1", "MAX_1")
        assert find_minimum_basins([conn], [minimum, maximum], []) == {}

    def test_unresolved_branches_are_not_adjacency(self) -> None:
        """Only status == "connected" descent records bound a basin."""
        minimum = _make_basin_center("MIN_1", "minimum", (0.0, 0.0, 0.0))
        maximum = _make_basin_center("MAX_1", "maximum", (0.0, 0.0, 2.0))
        conn = _make_connection("SAD_1", "MAX_1")
        result = find_minimum_basins(
            [conn],
            [minimum, maximum],
            [
                _make_descent("SAD_1", None, status="no_target"),
                _make_descent("SAD_1", "MIN_1", status="occluded"),
            ],
        )
        assert result == {}

    def test_multi_drain_saddle_bounds_every_basin_it_drains_into(self) -> None:
        """A saddle with two draining branches bounds BOTH basins (ethene3 case)."""
        min_a = _make_basin_center("MIN_A", "minimum", (0.0, 0.0, 0.0))
        min_b = _make_basin_center("MIN_B", "minimum", (5.0, 0.0, 0.0))
        maximum = _make_basin_center("MAX_1", "maximum", (0.0, 0.0, 2.0))
        conn = _make_connection("SAD_1", "MAX_1")
        result = find_minimum_basins(
            [conn],
            [min_a, min_b, maximum],
            [
                _make_descent("SAD_1", "MIN_A", branch_index=0),
                _make_descent("SAD_1", "MIN_B", branch_index=1),
            ],
        )
        assert set(result) == {"MIN_A", "MIN_B"}
        assert [a.saddle_id for a in result["MIN_A"].boundary_arcs] == ["SAD_1"]
        assert [a.saddle_id for a in result["MIN_B"].boundary_arcs] == ["SAD_1"]

    def test_arc_order_follows_sorted_saddles_not_input_order(self) -> None:
        """Membership depends on the record SET, never on input order."""
        minimum = _make_basin_center("MIN_1", "minimum", (0.0, 0.0, 0.0))
        maximum = _make_basin_center("MAX_1", "maximum", (0.0, 0.0, 2.0))
        c1 = _make_connection("SAD_1", "MAX_1")
        c2 = _make_connection("SAD_2", "MAX_1")
        descents = [_make_descent("SAD_2", "MIN_1"), _make_descent("SAD_1", "MIN_1")]
        for conns in ([c1, c2], [c2, c1]):
            result = find_minimum_basins(conns, [minimum, maximum], descents)
            assert [a.saddle_id for a in result["MIN_1"].boundary_arcs] == [
                "SAD_1",
                "SAD_2",
            ]
        for desc in (
            [_make_descent("SAD_2", "MIN_1"), _make_descent("SAD_1", "MIN_1")],
            [_make_descent("SAD_1", "MIN_1"), _make_descent("SAD_2", "MIN_1")],
        ):
            result = find_minimum_basins([c1, c2], [minimum, maximum], desc)
            assert [a.saddle_id for a in result["MIN_1"].boundary_arcs] == [
                "SAD_1",
                "SAD_2",
            ]


class TestMinConnectionStateDict:
    """The descent table crosses the params.json boundary (basin v10).

    The v9 bug class: a null Optional field fed float(None), the per-row
    except swallowed it, and the whole table vanished. These pin the round
    trip against the same failure.
    """

    def test_round_trip_preserves_every_field(self) -> None:
        c = _make_descent("SAD_1", "MIN_1")
        assert min_connection_from_state_dict(min_connection_to_state_dict(c)) == c

    def test_none_optional_fields_are_stored_values_not_corruption(self) -> None:
        c = _make_descent("SAD_1", None, status="no_target")
        row = min_connection_to_state_dict(c)
        assert row["min_id"] is None
        assert row["min_value"] is None
        back = min_connection_from_state_dict(row)
        assert back == c
        assert back.min_id is None

    def test_traced_tail_survives_the_round_trip(self) -> None:
        """stall_point/transited_ids/completed_leg_deg: the traced valley's tail."""
        c = replace(
            _make_descent("SAD_1", None, status="no_target"),
            method="gradient",
            stall_point=(0.1, 0.2, 0.3),
            transited_ids=("MAX_1", "MIN_2"),
            completed_leg_deg=4.5,
        )
        back = min_connection_from_state_dict(min_connection_to_state_dict(c))
        assert back == c
        assert back.stall_point == (0.1, 0.2, 0.3)
        assert back.transited_ids == ("MAX_1", "MIN_2")

    def test_row_is_json_safe(self) -> None:
        import json

        row = min_connection_to_state_dict(_make_descent("SAD_1", "MIN_1"))
        assert json.loads(json.dumps(row)) == row


class TestSerializeConnectionGraph:
    """Tests for serialize_connection_graph JSON-safe export."""

    def test_empty_graph(self) -> None:
        """Empty graph serializes to empty dict."""
        assert serialize_connection_graph({}) == {}

    def test_serializes_all_fields(self) -> None:
        """Each saddle maps to dicts with max_id/branch_index/status/verified."""
        # Built by hand: build_connection_graph filters stalled/unverified arcs
        # out, so the serializer's row shape is checked on a graph that keeps
        # them (the serializer itself must not filter).
        graph = {
            "SAD_1": [
                _make_connection("SAD_1", "MAX_1"),
                _make_connection(
                    "SAD_1", "MAX_2", branch_index=1, status="stalled", verified=False
                ),
            ]
        }
        out = serialize_connection_graph(graph)
        assert set(out) == {"SAD_1"}
        rows = {r["max_id"]: r for r in out["SAD_1"]}
        assert rows["MAX_1"] == {
            "max_id": "MAX_1",
            "branch_index": 0,
            "status": "connected",
            "verified": True,
        }
        assert rows["MAX_2"]["verified"] is False
        assert rows["MAX_2"]["status"] == "stalled"

    def test_json_round_trip(self) -> None:
        """Output is JSON-serializable and round-trips."""
        import json

        graph = build_connection_graph([_make_connection("SAD_1", "MAX_1")])
        out = serialize_connection_graph(graph)
        assert json.loads(json.dumps(out)) == out

    def test_graph_filters_unverified(self) -> None:
        """build_connection_graph keeps only connected+verified arcs."""
        graph = build_connection_graph(
            [
                _make_connection("SAD_1", "MAX_1"),
                _make_connection("SAD_1", "MAX_2", status="stalled", verified=False),
            ]
        )
        assert [c.max_id for c in graph["SAD_1"]] == ["MAX_1"]


class TestSerializeMinimumBasins:
    """Tests for serialize_minimum_basins JSON-safe export."""

    def test_empty_basins(self) -> None:
        """Empty basin dict serializes to empty dict."""
        assert serialize_minimum_basins({}) == {}

    def test_serializes_basin_fields(self) -> None:
        """Basin emits counts, ordered arcs, and sorted boundary sets."""
        basin = MinimumBasin(
            minimum_id="MIN_1",
            boundary_arcs=[
                _make_connection("SAD_2", "MAX_B", branch_index=1),
                _make_connection("SAD_1", "MAX_A"),
                _make_connection("SAD_2", "MAX_A", branch_index=0),
            ],
        )
        out = serialize_minimum_basins({"MIN_1": basin})
        b = out["MIN_1"]
        assert b["minimum_id"] == "MIN_1"
        assert b["n_boundary_arcs"] == 3
        # Order preserved from boundary_arcs.
        assert [a["saddle_id"] for a in b["boundary_arcs"]] == [
            "SAD_2",
            "SAD_1",
            "SAD_2",
        ]
        assert b["boundary_arcs"][0] == {
            "saddle_id": "SAD_2",
            "branch_index": 1,
            "max_id": "MAX_B",
        }
        # Deduplicated, sorted, JSON-safe (no sets).
        assert b["boundary_saddles"] == ["SAD_1", "SAD_2"]
        assert b["boundary_maxima"] == ["MAX_A", "MAX_B"]

    def test_stalled_arcs_drop_none_maxima(self) -> None:
        """max_id=None endpoints appear in arcs but not in boundary_maxima."""

        basin = MinimumBasin(
            minimum_id="MIN_1",
            boundary_arcs=[_make_connection("SAD_1", None, status="stalled")],
        )
        out = serialize_minimum_basins({"MIN_1": basin})
        assert out["MIN_1"]["boundary_arcs"][0]["max_id"] is None
        assert out["MIN_1"]["boundary_maxima"] == []

    def test_json_round_trip(self) -> None:
        """Output is JSON-serializable and round-trips."""
        import json

        basin = MinimumBasin(
            minimum_id="MIN_1", boundary_arcs=[_make_connection("SAD_1", "MAX_1")]
        )
        out = serialize_minimum_basins({"MIN_1": basin})
        assert json.loads(json.dumps(out)) == out


_ETHENE3_ARC_ROWS: list[dict[str, Any]] = [
    {
        "along_track_deg": 170.94861716936637,
        "arc_points": [
            (1.4904, 0.3435, -0.0003),
            (1.4917, 0.3426, -0.0101),
            (0.8491, -0.0002, -0.0075),
        ],
        "blocker_id": None,
        "blocker_polarity": None,
        "branch_index": 0,
        "completed_leg_deg": None,
        "cross_track_deg": None,
        "direction_mode": "port_tangent",
        "failure_gate": "cone",
        "max_id": "MAX_4864",
        "max_value": 0.5798894166946411,
        "method": "gradient",
        "mutual": False,
        "n_corners": 1,
        "n_nodes": 15,
        "n_runs": 0,
        "refine_max_shift_deg": 2.4404450377844404,
        "refined": True,
        "saddle_id": "E-SAD_MS3",
        "saddle_value": 0.0012506436323747039,
        "separation_deg": 123.63428869026909,
        "stall_point": None,
        "status": "connected",
        "total_turn_after_deg": 284.4426885335097,
        "total_turn_before_deg": 1189.4082906317901,
        "transited_ids": ["MS1", "MC12"],
        "verified": True,
    },
    {
        "along_track_deg": 167.61462268464416,
        "arc_points": [
            (1.4904, 0.3435, -0.0003),
            (1.4914, 0.3428, 0.0084),
            (0.8491, -0.0002, -0.0075),
        ],
        "blocker_id": None,
        "blocker_polarity": None,
        "branch_index": 1,
        "completed_leg_deg": None,
        "cross_track_deg": None,
        "direction_mode": "port_tangent",
        "failure_gate": "cone",
        "max_id": "MAX_4864",
        "max_value": 0.5798894166946411,
        "method": "gradient",
        "mutual": False,
        "n_corners": 0,
        "n_nodes": 14,
        "n_runs": 0,
        "refine_max_shift_deg": 2.2929171954928997,
        "refined": True,
        "saddle_id": "E-SAD_MS3",
        "saddle_value": 0.0012506436323747039,
        "separation_deg": 123.63428869026909,
        "stall_point": None,
        "status": "connected",
        "total_turn_after_deg": 145.11386433029045,
        "total_turn_before_deg": 873.2105168569078,
        "transited_ids": ["MC2", "MC12"],
        "verified": True,
    },
    {
        "along_track_deg": 170.66211670144577,
        "arc_points": [
            (1.4903, -0.3438, -0.0003),
            (1.4908, -0.3434, -0.0091),
            (0.8491, -0.0002, -0.0075),
        ],
        "blocker_id": None,
        "blocker_polarity": None,
        "branch_index": 0,
        "completed_leg_deg": None,
        "cross_track_deg": None,
        "direction_mode": "port_tangent",
        "failure_gate": "cone",
        "max_id": "MAX_4864",
        "max_value": 0.5798894166946411,
        "method": "gradient",
        "mutual": False,
        "n_corners": 2,
        "n_nodes": 16,
        "n_runs": 0,
        "refine_max_shift_deg": 2.6804817978444806,
        "refined": True,
        "saddle_id": "E-SAD_MS4",
        "saddle_value": 0.0012494416441768408,
        "separation_deg": 123.57676361564958,
        "stall_point": None,
        "status": "connected",
        "total_turn_after_deg": 381.1555452439809,
        "total_turn_before_deg": 1306.827600208038,
        "transited_ids": ["MS1", "MC12"],
        "verified": True,
    },
    {
        "along_track_deg": 163.9978769354392,
        "arc_points": [
            (1.4903, -0.3438, -0.0003),
            (1.4908, -0.3434, 0.0085),
            (0.8491, -0.0002, -0.0075),
        ],
        "blocker_id": None,
        "blocker_polarity": None,
        "branch_index": 1,
        "completed_leg_deg": None,
        "cross_track_deg": None,
        "direction_mode": "port_tangent",
        "failure_gate": "cone",
        "max_id": "MAX_4864",
        "max_value": 0.5798894166946411,
        "method": "gradient",
        "mutual": False,
        "n_corners": 1,
        "n_nodes": 14,
        "n_runs": 0,
        "refine_max_shift_deg": 2.5206387557526777,
        "refined": True,
        "saddle_id": "E-SAD_MS4",
        "saddle_value": 0.0012494416441768408,
        "separation_deg": 123.57676361564958,
        "stall_point": None,
        "status": "connected",
        "total_turn_after_deg": 164.5792706175673,
        "total_turn_before_deg": 832.1394066511576,
        "transited_ids": ["MC2", "MC12"],
        "verified": True,
    },
]


_ETHENE3_BASIN_CENTERS: list[dict[str, Any]] = [
    {
        "id": "MAX_4864",
        "polarity": "maximum",
        "position": [0.8472, -0.0002, 0.0143],
        "value": 0.579889,
        "is_effective": False,
        "vertex_id": 4864,
        "origin_id": None,
        "net_index": 1,
        "member_count": 1,
        "badge_color": "error",
    },
    {
        "id": "E-MIN_MC3",
        "polarity": "minimum",
        "position": [1.6745, -0.0001, -0.0003],
        "value": 0.000498,
        "is_effective": True,
        "vertex_id": -1,
        "origin_id": "MC3",
        "net_index": 1,
        "member_count": 5,
        "badge_color": "info",
    },
    {
        "id": "MIN_5070",
        "polarity": "minimum",
        "position": [1.0586, -0.3604, -0.0003],
        "value": 0.000254,
        "is_effective": False,
        "vertex_id": 5070,
        "origin_id": None,
        "net_index": 1,
        "member_count": 1,
        "badge_color": "info",
    },
    {
        "id": "MIN_5011",
        "polarity": "minimum",
        "position": [1.0586, 0.3604, -0.0003],
        "value": 0.000253,
        "is_effective": False,
        "vertex_id": 5011,
        "origin_id": None,
        "net_index": 1,
        "member_count": 1,
        "badge_color": "info",
    },
]


def _minimal_conn_row(**overrides: Any) -> dict[str, Any]:
    """A complete required-fields row in the viewer's state-dict shape."""
    row: dict[str, Any] = {
        "saddle_id": "SAD_1",
        "branch_index": 0,
        "max_id": "MAX_1",
        "status": "connected",
        "direction_mode": "ray",
        "cross_track_deg": 1.5,
        "along_track_deg": 10.0,
        "separation_deg": 12.0,
        "mutual": True,
        "arc_points": [[0.0, 0.0, 1.0], [0.7, 0.0, 0.7], [1.0, 0.0, 0.0]],
        "saddle_value": 0.5,
        "max_value": 1.0,
        "verified": True,
    }
    row.update(overrides)
    return row


class TestConnectionFromStateDict:
    """None-safe rehydration of MaxConnection from state/params dicts.

    The schema-v9 regression: rows saved with ``"cross_track_deg": null`` fed
    ``float(None)`` in the viewer's hand-written reconstruction, the per-row
    ``except`` swallowed the TypeError, and the ENTIRE arc table vanished —
    empty graph, no basin rings. The dataclass-driven rehydrator must keep
    legitimate nulls and never drop a well-formed row.
    """

    def test_round_trip_from_dataclass(self) -> None:
        """asdict(conn) rehydrates to an equal object."""
        conn = _make_connection("SAD_1", "MAX_1")
        assert connection_from_state_dict(asdict(conn)) == conn

    def test_null_optional_fields_pass_through(self) -> None:
        """A stored null on an Optional field stays None — it is data, not dirt.

        cross_track_deg is None for a branch whose geodesic ray hit nothing;
        coercing it to 0.0 would forge a perfect cross-track measurement.
        """
        row = _minimal_conn_row(
            cross_track_deg=None,
            stall_point=None,
            completed_leg_deg=None,
            blocker_id=None,
            blocker_polarity=None,
            failure_gate=None,
        )
        conn = connection_from_state_dict(row)
        assert conn.cross_track_deg is None
        assert conn.stall_point is None
        assert conn.completed_leg_deg is None
        assert conn.blocker_id is None
        assert conn.failure_gate is None

    def test_null_with_default_gets_default(self) -> None:
        """None on a defaulted non-Optional field falls back to the default."""
        conn = connection_from_state_dict(
            _minimal_conn_row(refine_max_shift_deg=None, n_corners=None, method=None)
        )
        assert conn.refine_max_shift_deg == 0.0
        assert conn.n_corners == 0
        assert conn.method == "geodesic"

    def test_missing_keys_use_defaults(self) -> None:
        """A v8-era row lacking the tail fields reconstructs with defaults."""
        row = _minimal_conn_row()
        conn = connection_from_state_dict(row)
        assert conn.refined is False
        assert conn.transited_ids == ()
        assert conn.n_nodes == 0

    def test_required_field_none_raises(self) -> None:
        """None for a required non-Optional field is malformed, not defaulted."""
        with pytest.raises(TypeError):
            connection_from_state_dict(_minimal_conn_row(saddle_id=None))
        with pytest.raises(TypeError):
            connection_from_state_dict(_minimal_conn_row(saddle_value=None))

    def test_uncoercible_value_raises(self) -> None:
        """A present value of the wrong shape raises, so the caller can log it."""
        with pytest.raises(ValueError):
            connection_from_state_dict(_minimal_conn_row(branch_index="x"))

    def test_ethene3_rows_rehydrate_and_form_basins(self) -> None:
        """The four real saved arcs (cross_track_deg null) form 3 basin rings.

        Fixture: ethene3 atom 1, the exact max_connections_list and
        basin_centers_list rows from ethene3_atom1_params.json (arc polylines
        trimmed to endpoints; endpoints preserved). This is the user's screen
        as a test: the same data that rendered nothing under the old code.
        """
        conn_objs = [connection_from_state_dict(row) for row in _ETHENE3_ARC_ROWS]
        assert len(conn_objs) == 4

        graph = build_connection_graph(conn_objs)
        assert set(graph) == {"E-SAD_MS3", "E-SAD_MS4"}
        assert sum(len(v) for v in graph.values()) == 4

        basin_centers = [
            BasinCenter(
                center_id=str(bc["id"]),
                polarity=str(bc["polarity"]),
                position=tuple(bc["position"]),
                value=float(bc["value"]),
                is_effective=bool(bc["is_effective"]),
                vertex_id=int(bc["vertex_id"]),
                origin_id=bc["origin_id"],
                net_index=int(bc["net_index"]),
                member_count=int(bc["member_count"]),
            )
            for bc in _ETHENE3_BASIN_CENTERS
        ]
        basins = find_minimum_basins(
            conn_objs,
            basin_centers,
            [
                _make_descent("E-SAD_MS3", "E-MIN_MC3", branch_index=0),
                _make_descent("E-SAD_MS3", "MIN_5011", branch_index=1),
                _make_descent("E-SAD_MS4", "E-MIN_MC3", branch_index=0),
                _make_descent("E-SAD_MS4", "MIN_5070", branch_index=1),
            ],
        )
        # v10 reads the adjacency off the field: each saddle drains into two
        # minima here (2 ascent arcs each), so E-MIN_MC3's ring is the 4 arcs
        # of both saddles, and each bare minimum's ring is the 2 arcs of the
        # single saddle that drains into it. v9 gave every basin all 4 and
        # every ring passed through both saddles -- the saturation the diagnose
        # panel misreported as "raise k".
        assert set(basins) == {"E-MIN_MC3", "MIN_5070", "MIN_5011"}
        assert len(basins["E-MIN_MC3"].boundary_arcs) == 4
        assert {a.saddle_id for a in basins["E-MIN_MC3"].boundary_arcs} == {
            "E-SAD_MS3",
            "E-SAD_MS4",
        }
        assert [a.saddle_id for a in basins["MIN_5011"].boundary_arcs] == [
            "E-SAD_MS3",
            "E-SAD_MS3",
        ]
        assert [a.saddle_id for a in basins["MIN_5070"].boundary_arcs] == [
            "E-SAD_MS4",
            "E-SAD_MS4",
        ]


# ---------------------------------------------------------------------------
# basin v11: the maximum side. Same rule, read from the other end.
# ---------------------------------------------------------------------------


class TestBuildDescentGraph:
    """Tests for build_descent_graph(), the descent mirror of the arc graph."""

    def test_empty(self) -> None:
        """No descent records, no graph."""
        assert build_descent_graph([]) == {}

    def test_filters_by_status_and_verified(self) -> None:
        """The same filter as build_connection_graph: connected AND verified.

        A branch the ring did not confirm bounds nothing on either side.
        """
        graph = build_descent_graph(
            [
                _make_descent("SAD_1", "MIN_1", branch_index=0),
                _make_descent("SAD_1", None, status="no_target", branch_index=1),
                _make_descent("SAD_1", "MIN_2", status="occluded", branch_index=2),
                _make_descent("SAD_2", "MIN_1", branch_index=0, verified=False),
            ]
        )
        assert set(graph) == {"SAD_1"}
        assert [c.branch_index for c in graph["SAD_1"]] == [0]

    def test_groups_branches_by_saddle(self) -> None:
        """Every connected branch of a saddle is in its list (input order)."""
        graph = build_descent_graph(
            [
                _make_descent("SAD_1", "MIN_2", branch_index=1),
                _make_descent("SAD_1", "MIN_1", branch_index=0),
                _make_descent("SAD_2", "MIN_1", branch_index=0),
            ]
        )
        assert set(graph) == {"SAD_1", "SAD_2"}
        assert [c.branch_index for c in graph["SAD_1"]] == [1, 0]


class TestMaximumBasinDataclass:
    """MaximumBasin is the ascent-side twin of MinimumBasin."""

    def test_create_basin(self) -> None:
        arcs = [_make_descent("SAD_1", "MIN_1")]
        basin = MaximumBasin(maximum_id="MAX_1", boundary_arcs=arcs)
        assert basin.maximum_id == "MAX_1"
        assert basin.boundary_arcs == arcs

    def test_basin_is_frozen(self) -> None:
        basin = MaximumBasin(maximum_id="MAX_1", boundary_arcs=[])
        with pytest.raises(AttributeError):
            basin.maximum_id = "MAX_2"  # type: ignore[misc]

    def test_get_max_basin_arcs(self) -> None:
        arcs = [_make_descent("SAD_1", "MIN_1"), _make_descent("SAD_2", "MIN_1")]
        assert get_max_basin_arcs(MaximumBasin("MAX_1", arcs)) == arcs
        assert get_max_basin_arcs(MaximumBasin("MAX_1", [])) == []


class TestFindMaximumBasins:
    """find_maximum_basins: a hill is bounded by the valleys around it."""

    def test_empty_topology(self) -> None:
        assert find_maximum_basins([], [], []) == {}

    def test_no_maxima_yields_no_basins(self) -> None:
        """Minima never get maximum basins (the polarity filter)."""
        minimum = _make_basin_center("MIN_1", "minimum", (0.0, 0.0, 0.0))
        result = find_maximum_basins(
            [_make_connection("SAD_1", None)],
            [minimum],
            [_make_descent("SAD_1", "MIN_1")],
        )
        assert result == {}

    def test_polarity_filter_excludes_a_named_minimum(self) -> None:
        """The filter, not the data, is what keeps minima out of this table.

        Adjacency that NAMES a minimum id (the degenerate list a mixed pool
        produces -- a raw center and its effective id crossing polarity) must
        still not yield a maximum basin. Without the polarity filter the
        minimum below gets a ring, and the screen draws a hill boundary around
        a valley.
        """
        minimum = _make_basin_center("MIN_1", "minimum", (0.0, 0.0, 0.0))
        maximum = _make_basin_center("MAX_1", "maximum", (0.0, 0.0, 2.0))
        result = find_maximum_basins(
            [_make_connection("SAD_1", "MIN_1")],
            [minimum, maximum],
            [_make_descent("SAD_1", "MIN_A")],
        )
        assert result == {}
        # And the mirrored statement is the one that DOES bind:
        assert set(
            find_minimum_basins(
                [_make_connection("SAD_1", "MAX_1")],
                [minimum, maximum],
                [_make_descent("SAD_1", "MIN_1")],
            )
        ) == {"MIN_1"}

    def test_single_maximum_with_climbing_saddles(self) -> None:
        """The ring is the DESCENT arcs of the saddles that CLIMB into it."""
        maximum = _make_basin_center("MAX_1", "maximum", (0.0, 0.0, 2.0))
        minimum_a = _make_basin_center("MIN_A", "minimum", (0.0, 0.0, 0.0))
        minimum_b = _make_basin_center("MIN_B", "minimum", (4.0, 0.0, 0.0))
        result = find_maximum_basins(
            [
                _make_connection("SAD_1", "MAX_1"),
                _make_connection("SAD_2", "MAX_1"),
            ],
            [maximum, minimum_a, minimum_b],
            [
                _make_descent("SAD_1", "MIN_A"),
                _make_descent("SAD_2", "MIN_B"),
            ],
        )
        assert set(result) == {"MAX_1"}
        assert [(a.saddle_id, a.min_id) for a in result["MAX_1"].boundary_arcs] == [
            ("SAD_1", "MIN_A"),
            ("SAD_2", "MIN_B"),
        ]

    def test_adjacency_is_ascent_not_descent(self) -> None:
        """A saddle that drains into a minimum's basin does not bound the HILL.

        The adjacency and the geometry come from opposite tables: the
        maximum's ring is the saddles whose ASCENT reached it, and each one
        contributes its DESCENT arcs. Feeding a saddle that only drains (no
        ascent record) bounds nothing.
        """
        maximum = _make_basin_center("MAX_1", "maximum", (0.0, 0.0, 2.0))
        minimum = _make_basin_center("MIN_1", "minimum", (0.0, 0.0, 0.0))
        result = find_maximum_basins(
            [_make_connection("SAD_1", "MAX_OTHER")],
            [maximum, minimum],
            [_make_descent("SAD_2", "MIN_1")],
        )
        assert result == {}

    def test_isolated_maximum_is_omitted(self) -> None:
        """No climbing saddle = open boundary = not a basin (v10 precedent)."""
        maximum = _make_basin_center("MAX_1", "maximum", (0.0, 0.0, 2.0))
        result = find_maximum_basins(
            [_make_connection("SAD_1", "MAX_2")],
            [maximum],
            [_make_descent("SAD_1", "MIN_1")],
        )
        assert result == {}

    def test_climbing_saddle_without_descents_is_omitted(self) -> None:
        """A saddle that climbs in but has no recorded drain contributes nothing;

        a maximum whose every climbing saddle has no descent branch has no
        boundary at all, so it is absent rather than an empty basin.
        """
        maximum = _make_basin_center("MAX_1", "maximum", (0.0, 0.0, 2.0))
        result = find_maximum_basins(
            [
                _make_connection("SAD_1", "MAX_1"),
                _make_connection("SAD_2", "MAX_1"),
            ],
            [maximum],
            [_make_descent("SAD_2", None, status="no_target")],
        )
        assert result == {}

    def test_unconnected_and_unverified_arcs_are_excluded(self) -> None:
        """Both filters apply on the geometry side, and both on adjacency."""
        maximum = _make_basin_center("MAX_1", "maximum", (0.0, 0.0, 2.0))
        result = find_maximum_basins(
            [
                _make_connection("SAD_1", "MAX_1"),
                _make_connection("SAD_OCCLUDED", "MAX_1", status="occluded"),
            ],
            [maximum],
            [
                _make_descent("SAD_1", "MIN_1", verified=False),
                _make_descent("SAD_OCCLUDED", "MIN_1"),
            ],
        )
        assert result == {}

    def test_multi_climb_saddle_bounds_every_maximum_it_reaches(self) -> None:
        """One saddle, two ascent branches, two hills: it rings BOTH.

        The dual of the v10 multi-drain case, and the ethene3 shape: a saddle
        climbing into two maxima is a pass between both their basins, so it is
        on both boundaries.
        """
        max_a = _make_basin_center("MAX_A", "maximum", (0.0, 0.0, 2.0))
        max_b = _make_basin_center("MAX_B", "maximum", (0.0, 4.0, 2.0))
        minimum = _make_basin_center("MIN_1", "minimum", (0.0, 0.0, 0.0))
        result = find_maximum_basins(
            [
                _make_connection("SAD_1", "MAX_A", branch_index=0),
                _make_connection("SAD_1", "MAX_B", branch_index=1),
            ],
            [max_a, max_b, minimum],
            [_make_descent("SAD_1", "MIN_1", branch_index=0)],
        )
        assert set(result) == {"MAX_A", "MAX_B"}
        assert [a.saddle_id for a in result["MAX_A"].boundary_arcs] == ["SAD_1"]
        assert [a.saddle_id for a in result["MAX_B"].boundary_arcs] == ["SAD_1"]

    def test_arc_order_follows_sorted_saddles_not_input_order(self) -> None:
        """Membership and order depend on the record SET, never on input order."""
        maximum = _make_basin_center("MAX_1", "maximum", (0.0, 0.0, 2.0))
        c1 = _make_connection("SAD_1", "MAX_1")
        c2 = _make_connection("SAD_2", "MAX_1")
        d1 = _make_descent("SAD_1", "MIN_1")
        d2 = _make_descent("SAD_2", "MIN_1")
        for asc in ([c1, c2], [c2, c1]):
            for desc in ([d1, d2], [d2, d1]):
                result = find_maximum_basins(asc, [maximum], desc)
                assert [a.saddle_id for a in result["MAX_1"].boundary_arcs] == [
                    "SAD_1",
                    "SAD_2",
                ]

    def test_branch_order_within_a_saddle_follows_the_table(self) -> None:
        """A saddle contributes ALL its descent branches, in table order."""
        maximum = _make_basin_center("MAX_1", "maximum", (0.0, 0.0, 2.0))
        result = find_maximum_basins(
            [_make_connection("SAD_1", "MAX_1")],
            [maximum],
            [
                _make_descent("SAD_1", "MIN_A", branch_index=0),
                _make_descent("SAD_1", "MIN_B", branch_index=1),
            ],
        )
        assert [
            (a.saddle_id, a.branch_index, a.min_id)
            for a in result["MAX_1"].boundary_arcs
        ] == [("SAD_1", 0, "MIN_A"), ("SAD_1", 1, "MIN_B")]

    def test_rehydrated_rows_produce_the_same_basin(self) -> None:
        """The ring survives the params.json round trip.

        The viewer draws basins off rehydrated dicts, never off live objects,
        so the deserialized table must give the identical boundary.
        """
        maximum = _make_basin_center("MAX_1", "maximum", (0.0, 0.0, 2.0))
        asc = [_make_connection("SAD_1", "MAX_1"), _make_connection("SAD_2", "MAX_1")]
        desc = [_make_descent("SAD_1", "MIN_1"), _make_descent("SAD_2", "MIN_2")]
        direct = find_maximum_basins(asc, [maximum], desc)
        via_rows = find_maximum_basins(
            [connection_from_state_dict(asdict(c)) for c in asc],
            [maximum],
            [min_connection_from_state_dict(asdict(d)) for d in desc],
        )
        assert [
            (a.saddle_id, a.branch_index, a.min_id)
            for a in via_rows["MAX_1"].boundary_arcs
        ] == [
            (a.saddle_id, a.branch_index, a.min_id)
            for a in direct["MAX_1"].boundary_arcs
        ]


def _flip_id(center_id: str) -> str:
    """MIN_x <-> MAX_x, for the negation test."""
    if "MIN" in center_id:
        return center_id.replace("MIN", "MAX", 1)
    return center_id.replace("MAX", "MIN", 1)


def _negate_ascent(ascent: list[MaxConnection]) -> list[MinConnection]:
    """The ascent table under a negated field is the descent table.

    Only the fields the boundary rule reads are carried across (saddle, branch,
    the endpoint id, status, verified); the arc geometry helpers use fixed
    defaults, so the two sides of the duality are compared on identity and
    order, not on polyline bytes.
    """
    return [
        _make_descent(
            c.saddle_id,
            _flip_id(c.max_id) if c.max_id else None,
            c.status,
            c.branch_index,
            c.verified,
        )
        for c in ascent
    ]


def _negate_descent(descents: list[MinConnection]) -> list[MaxConnection]:
    """The descent table under a negated field is the ascent table."""
    return [
        _make_connection(
            d.saddle_id,
            _flip_id(d.min_id) if d.min_id else None,
            d.status,
            d.branch_index,
            d.verified,
        )
        for d in descents
    ]


class TestRuleIsTheMorseDual:
    """Negate the field and the two rules must be ONE rule.

    This is the property the v11 plan rests on: the boundary of a maximum's
    basin is the boundary of a minimum's basin in the negated field. If the
    maximum side were derived from anything other than the same adjacency
    statement with the tables exchanged, this test is where it shows.
    """

    ROWS: tuple[tuple[str, str, str], ...] = (
        # (saddle, minimum it drains into, maximum it climbs into)
        ("SAD_1", "MIN_1", "MAX_1"),
        ("SAD_2", "MIN_1", "MAX_2"),
        ("SAD_3", "MIN_2", "MAX_1"),
        ("SAD_4", "MIN_2", "MAX_2"),
    )

    def _tables(
        self,
    ) -> tuple[list[MaxConnection], list[MinConnection], list[BasinCenter]]:
        ascent = [_make_connection(s, mx) for s, _mn, mx in self.ROWS]
        descent = [_make_descent(s, mn) for s, mn, _mx in self.ROWS]
        minima = [
            _make_basin_center(mn, "minimum", (i * 3.0, 0.0, 0.0))
            for i, mn in enumerate(sorted({mn for _s, mn, _x in self.ROWS}))
        ]
        maxima = [
            _make_basin_center(mx, "maximum", (i * 3.0, 0.0, 2.0))
            for i, mx in enumerate(sorted({mx for _s, _mn, mx in self.ROWS}))
        ]
        return ascent, descent, [*minima, *maxima]

    def test_duality(self) -> None:
        """Every minimum basin has an identical maximum basin in the negation."""
        ascent, descent, centers = self._tables()
        forward = find_minimum_basins(ascent, centers, descent)
        negated_centers = [
            _make_basin_center(
                _flip_id(c.center_id),
                "minimum" if c.polarity == "maximum" else "maximum",
                c.position,
                c.value,
            )
            for c in centers
        ]
        dual = find_maximum_basins(
            _negate_descent(descent),
            negated_centers,
            _negate_ascent(ascent),
        )
        assert dual, "the duality is vacuous on an empty result"
        # find_maximum_basins keys by the maximum's id, which under negation is
        # the flipped id of the minimum it mirrors.
        assert {_flip_id(k) for k in forward} == set(dual)
        for min_id, basin in forward.items():
            mirror = dual[_flip_id(min_id)]
            assert [a.saddle_id for a in mirror.boundary_arcs] == [
                a.saddle_id for a in basin.boundary_arcs
            ]

    def test_shared_tables_are_symmetric(self) -> None:
        """A saddle's drain and its climb are the same ring statement twice.

        On this fixture every minimum is ringed by two saddles and so is every
        maximum, with the same saddle ids -- the 2x2 grid is self-dual.
        """
        ascent, descent, centers = self._tables()
        forward = find_minimum_basins(ascent, centers, descent)
        dual = find_maximum_basins(ascent, centers, descent)
        assert {
            k: sorted(a.saddle_id for a in b.boundary_arcs) for k, b in forward.items()
        } == {
            "MIN_1": ["SAD_1", "SAD_2"],
            "MIN_2": ["SAD_3", "SAD_4"],
        }
        assert {
            k: sorted(a.saddle_id for a in b.boundary_arcs) for k, b in dual.items()
        } == {
            "MAX_1": ["SAD_1", "SAD_3"],
            "MAX_2": ["SAD_2", "SAD_4"],
        }


class TestSerializeMaximumBasins:
    """The maximum table for params.json and diagnose_params' results JSON."""

    def test_empty_basins(self) -> None:
        assert serialize_maximum_basins({}) == {}

    def test_serializes_basin_fields(self) -> None:
        basins = {
            "MAX_1": MaximumBasin(
                maximum_id="MAX_1",
                boundary_arcs=[
                    _make_descent("SAD_1", "MIN_A", branch_index=0),
                    _make_descent("SAD_2", "MIN_B", branch_index=1),
                ],
            )
        }
        out = serialize_maximum_basins(basins)
        assert out["MAX_1"]["maximum_id"] == "MAX_1"
        assert out["MAX_1"]["n_boundary_arcs"] == 2
        assert out["MAX_1"]["boundary_saddles"] == ["SAD_1", "SAD_2"]
        assert out["MAX_1"]["boundary_minima"] == ["MIN_A", "MIN_B"]
        assert out["MAX_1"]["boundary_arcs"] == [
            {"saddle_id": "SAD_1", "branch_index": 0, "min_id": "MIN_A"},
            {"saddle_id": "SAD_2", "branch_index": 1, "min_id": "MIN_B"},
        ]

    def test_stalled_arcs_drop_none_minima(self) -> None:
        """A None endpoint is not an id; the id set stays clean, the arc stays."""
        basins = {
            "MAX_1": MaximumBasin(
                maximum_id="MAX_1",
                boundary_arcs=[_make_descent("SAD_1", None, branch_index=0)],
            )
        }
        out = serialize_maximum_basins(basins)
        assert out["MAX_1"]["boundary_minima"] == []
        assert out["MAX_1"]["boundary_saddles"] == ["SAD_1"]
        assert out["MAX_1"]["boundary_arcs"] == [
            {"saddle_id": "SAD_1", "branch_index": 0, "min_id": None}
        ]

    def test_id_sets_are_sorted(self) -> None:
        """Determinism: the JSON diffs cleanly across runs."""
        basins = {
            "MAX_1": MaximumBasin(
                maximum_id="MAX_1",
                boundary_arcs=[
                    _make_descent("SAD_9", "MIN_Z", branch_index=0),
                    _make_descent("SAD_1", "MIN_A", branch_index=1),
                ],
            )
        }
        out = serialize_maximum_basins(basins)
        assert out["MAX_1"]["boundary_saddles"] == ["SAD_1", "SAD_9"]
        assert out["MAX_1"]["boundary_minima"] == ["MIN_A", "MIN_Z"]

    def test_json_round_trip(self) -> None:
        """The payload is JSON-safe, so the file is writable."""
        basins = {
            "MAX_1": MaximumBasin(
                maximum_id="MAX_1",
                boundary_arcs=[_make_descent("SAD_1", "MIN_A")],
            )
        }
        assert json.loads(json.dumps(serialize_maximum_basins(basins))) == (
            serialize_maximum_basins(basins)
        )


class TestEthene3MaximumBasin:
    """The real data: the hill MAX_4864, ringed by the two saddles that climb it.

    Fixture: the ethene3 rows above. Both saddles ascend into MAX_4864 (2
    branches each) and drain into the minima of the valley, so the maximum's
    boundary is the SAME four arcs that ring E-MIN_MC3 -- the pass between the
    hill and the valley is the valley's own rim. This is the v11 rule on the
    user's screen: descent arcs, drawn for the first time.
    """

    def test_max_4864_basin(self) -> None:
        conn_objs = [connection_from_state_dict(row) for row in _ETHENE3_ARC_ROWS]
        basin_centers = [
            BasinCenter(
                center_id=str(bc["id"]),
                polarity=str(bc["polarity"]),
                position=tuple(bc["position"]),
                value=float(bc["value"]),
                is_effective=bool(bc["is_effective"]),
                vertex_id=int(bc["vertex_id"]),
                origin_id=bc["origin_id"],
                net_index=int(bc["net_index"]),
                member_count=int(bc["member_count"]),
            )
            for bc in _ETHENE3_BASIN_CENTERS
        ]
        descents = [
            _make_descent("E-SAD_MS3", "E-MIN_MC3", branch_index=0),
            _make_descent("E-SAD_MS3", "MIN_5011", branch_index=1),
            _make_descent("E-SAD_MS4", "E-MIN_MC3", branch_index=0),
            _make_descent("E-SAD_MS4", "MIN_5070", branch_index=1),
        ]
        basins = find_maximum_basins(conn_objs, basin_centers, descents)
        assert set(basins) == {"MAX_4864"}
        ring = basins["MAX_4864"].boundary_arcs
        assert [(a.saddle_id, a.branch_index, a.min_id) for a in ring] == [
            ("E-SAD_MS3", 0, "E-MIN_MC3"),
            ("E-SAD_MS3", 1, "MIN_5011"),
            ("E-SAD_MS4", 0, "E-MIN_MC3"),
            ("E-SAD_MS4", 1, "MIN_5070"),
        ]
        # The ring covers every descent branch of every saddle that climbs the
        # hill: 2 saddles x 2 branches each, no branch left undrawn.
        minima = serialize_maximum_basins(basins)["MAX_4864"]["boundary_minima"]
        assert minima == ["E-MIN_MC3", "MIN_5011", "MIN_5070"]
        # E-SAD_MS3 bounds BOTH ends of the pass: it is in this hill's ring and
        # in the basin of each minimum it drains into.
        min_basins = find_minimum_basins(conn_objs, basin_centers, descents)
        assert (
            "E-SAD_MS3"
            in serialize_minimum_basins(min_basins)["E-MIN_MC3"]["boundary_saddles"]
        )
        assert (
            "E-SAD_MS3"
            in serialize_minimum_basins(min_basins)["MIN_5011"]["boundary_saddles"]
        )
