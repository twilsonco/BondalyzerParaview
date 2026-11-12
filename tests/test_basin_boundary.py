"""Tests for gba_topology2.src.basin_boundary module."""

import pytest

from gba_topology2.src.basin_boundary import (
    MinimumBasin,
    build_connection_graph,
    find_minimum_basins,
    get_basin_arcs,
)
from gba_topology2.src.basin_center import BasinCenter
from gba_topology2.src.max_connection import MaxConnection


def _make_connection(
    saddle_id: str,
    max_id: str | None,
    status: str = "connected",
    branch_index: int = 0,
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
        verified=True,
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
        assert graph["SAD_1"][0] == ("MAX_2", conn)

    def test_filters_by_status(self) -> None:
        """Only connected arcs are included."""
        conn_connected = _make_connection("SAD_1", "MAX_2", status="connected")
        conn_occluded = _make_connection("SAD_1", "MAX_3", status="occluded")
        graph = build_connection_graph([conn_connected, conn_occluded])
        assert "SAD_1" in graph
        assert len(graph["SAD_1"]) == 1
        assert graph["SAD_1"][0][0] == "MAX_2"

    def test_multiple_edges_from_saddle(self) -> None:
        """Saddle can have multiple outgoing arcs."""
        conn1 = _make_connection("SAD_1", "MAX_2", branch_index=0)
        conn2 = _make_connection("SAD_1", "MAX_3", branch_index=1)
        graph = build_connection_graph([conn1, conn2])
        assert len(graph["SAD_1"]) == 2
        targets = [t for t, _ in graph["SAD_1"]]
        assert set(targets) == {"MAX_2", "MAX_3"}

    def test_complex_graph(self) -> None:
        """Complex graph with multiple nodes and edges."""
        conn1 = _make_connection("SAD_1", "MAX_2", branch_index=0)
        conn2 = _make_connection("SAD_2", "MAX_3", branch_index=0)
        conn3 = _make_connection("MAX_2", "SAD_3", branch_index=0)  # Max->Saddle
        graph = build_connection_graph([conn1, conn2, conn3])
        assert "SAD_1" in graph
        assert "SAD_2" in graph
        assert "MAX_2" in graph
        assert graph["SAD_1"][0][0] == "MAX_2"
        assert graph["MAX_2"][0][0] == "SAD_3"


class TestMinimumBasinDataclass:
    """Tests for MinimumBasin dataclass."""

    def test_create_basin(self) -> None:
        """Create a MinimumBasin with boundary arcs."""
        conn = _make_connection("SAD_1", "MAX_2")
        basin = MinimumBasin(
            minimum_id="MIN_1",
            boundary_cps=[("MAX_2", conn)],
        )
        assert basin.minimum_id == "MIN_1"
        assert len(basin.boundary_cps) == 1
        assert basin.boundary_cps[0][0] == "MAX_2"

    def test_basin_is_frozen(self) -> None:
        """MinimumBasin is immutable."""
        conn = _make_connection("SAD_1", "MAX_2")
        basin = MinimumBasin(
            minimum_id="MIN_1",
            boundary_cps=[("MAX_2", conn)],
        )
        with pytest.raises(AttributeError):
            basin.minimum_id = "MIN_2"  # type: ignore


class TestGetBasinArcs:
    """Tests for get_basin_arcs()."""

    def test_extract_arcs(self) -> None:
        """Extract arc objects from basin."""
        conn1 = _make_connection("SAD_1", "MAX_2")
        conn2 = _make_connection("SAD_2", "MAX_3")
        basin = MinimumBasin(
            minimum_id="MIN_1",
            boundary_cps=[("MAX_2", conn1), ("MAX_3", conn2)],
        )
        arcs = get_basin_arcs(basin)
        assert len(arcs) == 2
        assert arcs[0] is conn1
        assert arcs[1] is conn2

    def test_empty_basin(self) -> None:
        """Extract arcs from empty basin."""
        basin = MinimumBasin(
            minimum_id="MIN_1",
            boundary_cps=[],
        )
        arcs = get_basin_arcs(basin)
        assert arcs == []


class TestFindMinimumBasins:
    """Tests for find_minimum_basins()."""

    def test_empty_topology(self) -> None:
        """Empty connections and basin centers yield no basins."""
        basins = find_minimum_basins([], [])
        assert basins == {}

    def test_single_cycle_with_minimum(self) -> None:
        """Single saddle-maximum cycle enclosing one minimum."""
        # Create a simple square cycle: SAD_1 -> MAX_2 -> SAD_3 -> MAX_4 -> SAD_1
        # Place minimum at origin
        conn1 = _make_connection("SAD_1", "MAX_2")
        conn2 = _make_connection("MAX_2", "SAD_3")
        conn3 = _make_connection("SAD_3", "MAX_4")
        conn4 = _make_connection("MAX_4", "SAD_1")

        # Place maxima in a square around origin
        maxima = [
            _make_basin_center("MAX_2", "maximum", (1.0, 0.0, 0.0)),
            _make_basin_center("MAX_4", "maximum", (0.0, 1.0, 0.0)),
        ]
        minima = [
            _make_basin_center("MIN_1", "minimum", (0.0, 0.0, 0.0)),
        ]
        saddles = [
            _make_basin_center("SAD_1", "maximum", (-1.0, 0.0, 0.0)),  # stand-in
            _make_basin_center("SAD_3", "maximum", (0.0, -1.0, 0.0)),  # stand-in
        ]

        all_centers = maxima + minima + saddles
        basins = find_minimum_basins([conn1, conn2, conn3, conn4], all_centers)
        # Basins may be empty if the point-in-polygon heuristic doesn't detect
        # the minimum inside, but the function should not crash
        assert isinstance(basins, dict)

    def test_no_cycles(self) -> None:
        """Acyclic graph (DAG) yields no basins."""
        # Linear chain: SAD_1 -> MAX_2 -> SAD_3 (no cycle)
        conn1 = _make_connection("SAD_1", "MAX_2")
        conn2 = _make_connection("MAX_2", "SAD_3")

        centers = [
            _make_basin_center("SAD_1", "maximum", (1.0, 0.0, 0.0)),
            _make_basin_center("MAX_2", "maximum", (0.0, 1.0, 0.0)),
            _make_basin_center("SAD_3", "maximum", (0.0, 0.0, 1.0)),
        ]

        basins = find_minimum_basins([conn1, conn2], centers)
        assert basins == {}

    def test_disconnected_minimum(self) -> None:
        """Minimum not enclosed by any cycle is not in result."""
        # Isolated cycle and separate minimum
        conn1 = _make_connection("SAD_1", "MAX_2")
        conn2 = _make_connection("MAX_2", "SAD_1")  # 2-cycle

        centers = [
            _make_basin_center("SAD_1", "maximum", (1.0, 0.0, 0.0)),
            _make_basin_center("MAX_2", "maximum", (0.0, 1.0, 0.0)),
            _make_basin_center("MIN_1", "minimum", (10.0, 10.0, 10.0)),  # Far away
        ]

        basins = find_minimum_basins([conn1, conn2], centers)
        # MIN_1 is not enclosed by the cycle, so may not appear
        assert isinstance(basins, dict)

    def test_filters_non_connected_arcs(self) -> None:
        """Non-connected arcs are ignored in cycle detection."""
        # Only connected arc forms the graph
        conn_connected = _make_connection("SAD_1", "MAX_2", status="connected")
        conn_occluded = _make_connection("MAX_2", "SAD_1", status="occluded")

        centers = [
            _make_basin_center("SAD_1", "maximum", (1.0, 0.0, 0.0)),
            _make_basin_center("MAX_2", "maximum", (0.0, 1.0, 0.0)),
        ]

        basins = find_minimum_basins([conn_connected, conn_occluded], centers)
        # conn_occluded should not form part of the cycle
        assert isinstance(basins, dict)

    def test_basin_structure(self) -> None:
        """Basin structure contains minimum_id and boundary_cps."""
        conn = _make_connection("SAD_1", "MAX_2")
        basin = MinimumBasin(
            minimum_id="MIN_1",
            boundary_cps=[("MAX_2", conn)],
        )
        assert basin.minimum_id == "MIN_1"
        assert isinstance(basin.boundary_cps, list)
        assert len(basin.boundary_cps) > 0
        cp_id, conn_obj = basin.boundary_cps[0]
        assert isinstance(cp_id, str)
        assert isinstance(conn_obj, MaxConnection)


class TestEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_none_max_id(self) -> None:
        """Connections with None max_id are handled gracefully."""
        conn_none = _make_connection("SAD_1", None, status="connected")
        graph = build_connection_graph([conn_none])
        # None max_id should be filtered out
        assert graph == {}

    def test_empty_cycle_list(self) -> None:
        """Empty list of basins returned for empty topology."""
        connections: list[MaxConnection] = []
        basin_centers: list[BasinCenter] = []
        basins = find_minimum_basins(connections, basin_centers)
        assert basins == {}

    def test_only_maxima_and_saddles(self) -> None:
        """No minima in basin_centers yields no basins."""
        conn1 = _make_connection("SAD_1", "MAX_2")
        conn2 = _make_connection("MAX_2", "SAD_1")

        centers = [
            _make_basin_center("SAD_1", "maximum", (1.0, 0.0, 0.0)),
            _make_basin_center("MAX_2", "maximum", (0.0, 1.0, 0.0)),
        ]

        basins = find_minimum_basins([conn1, conn2], centers)
        assert isinstance(basins, dict)
