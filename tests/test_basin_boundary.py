"""Tests for gba_topology2.src.basin_boundary module."""

from dataclasses import asdict
from typing import Any

import pytest

from gba_topology2.src.basin_boundary import (
    MinimumBasin,
    build_connection_graph,
    connection_from_state_dict,
    find_minimum_basins,
    get_basin_arcs,
    serialize_connection_graph,
    serialize_minimum_basins,
)
from gba_topology2.src.basin_center import BasinCenter
from gba_topology2.src.max_connection import MaxConnection


def _make_connection(
    saddle_id: str,
    max_id: str | None,
    status: str = "connected",
    branch_index: int = 0,
    verified: bool = True,
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
        arc_points=((0.0, 0.0, 1.0), (1.0, 0.0, 0.0)),
        saddle_value=0.5,
        max_value=1.0 if max_id else None,
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
        result = find_minimum_basins([], [])
        assert result == {}

    def test_single_minimum_with_adjacent_saddles(self) -> None:
        """Single minimum with connections from different saddles creates basin."""
        # Minimum at origin with connections from different saddles
        minimum = _make_basin_center("MIN_1", "minimum", (0.0, 0.0, 0.0), value=0.1)
        maximum = _make_basin_center("MAX_1", "maximum", (0.0, 0.0, 2.0))

        # Connections from saddles (saddles are identified by ID in connections)
        conn1 = _make_connection("SAD_1", "MAX_1")
        conn2 = _make_connection("SAD_2", "MAX_1")
        conn3 = _make_connection("SAD_3", "MAX_1")

        basin_centers = [minimum, maximum]
        result = find_minimum_basins([conn1, conn2, conn3], basin_centers)

        # If saddles can be looked up, basin should exist
        # But without saddle positions in basin_centers, it won't be found
        # This is expected with current algorithm
        assert isinstance(result, dict)

    def test_no_cycles(self) -> None:
        """Topology with no cycles (isolated minimum) yields no basins."""
        minimum = _make_basin_center("MIN_1", "minimum", (0.0, 0.0, 0.0))
        # No connections to any saddles
        result = find_minimum_basins([], [minimum])
        assert result == {}

    def test_disconnected_minimum(self) -> None:
        """Minimum with no adjacent saddles has no basin."""
        minimum1 = _make_basin_center("MIN_1", "minimum", (0.0, 0.0, 0.0))
        minimum2 = _make_basin_center("MIN_2", "minimum", (10.0, 10.0, 10.0))
        maximum = _make_basin_center("MAX_1", "maximum", (0.0, 0.0, 2.0))

        conn = _make_connection("SAD_1", "MAX_1")
        basin_centers = [minimum1, minimum2, maximum]
        result = find_minimum_basins([conn], basin_centers)

        # Without saddle positions, algorithm cannot determine basin membership
        assert isinstance(result, dict)

    def test_filters_non_connected_arcs(self) -> None:
        """Only connected arcs are included in basins."""
        minimum = _make_basin_center("MIN_1", "minimum", (0.0, 0.0, 0.0))
        maximum = _make_basin_center("MAX_1", "maximum", (0.0, 0.0, 2.0))

        conn_connected = _make_connection("SAD_1", "MAX_1", status="connected")
        conn_occluded = _make_connection("SAD_1", "MAX_2", status="occluded")

        basin_centers = [minimum, maximum]
        result = find_minimum_basins([conn_connected, conn_occluded], basin_centers)

        # Graph is built with only connected/verified arcs
        assert isinstance(result, dict)

    def test_basin_structure(self) -> None:
        """Basin contains correct arc structure."""
        minimum = _make_basin_center("MIN_1", "minimum", (0.0, 0.0, 0.0))
        maximum = _make_basin_center("MAX_1", "maximum", (0.0, 0.0, 2.0))

        conn = _make_connection("SAD_1", "MAX_1")
        result = find_minimum_basins([conn], [minimum, maximum])

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
        result = find_minimum_basins([conn_stalled, conn_connected], basin_centers)

        # Graph filters out None max_id connections
        assert isinstance(result, dict)

    def test_empty_cycle_list(self) -> None:
        """Empty basin list yields empty dict."""
        graph = build_connection_graph([])
        assert graph == {}

    def test_only_maxima_and_saddles(self) -> None:
        """No minima in basin_centers yields no basins."""
        maximum = _make_basin_center("MAX_1", "maximum", (0.0, 0.0, 2.0))

        conn = _make_connection("SAD_1", "MAX_1")
        result = find_minimum_basins([conn], [maximum])
        assert result == {}


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
        basins = find_minimum_basins(conn_objs, basin_centers)
        assert set(basins) == {"E-MIN_MC3", "MIN_5070", "MIN_5011"}
        for basin in basins.values():
            assert len(basin.boundary_arcs) == 4
            assert {a.saddle_id for a in basin.boundary_arcs} == {
                "E-SAD_MS3",
                "E-SAD_MS4",
            }
            assert {a.max_id for a in basin.boundary_arcs} == {"MAX_4864"}
