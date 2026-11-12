"""Basin Boundary Consolidation for gba_topology2.

Identifies and manages minimum basin boundaries by cycle detection on the
saddle-maximum connection graph. Each basin is a closed loop of saddle-maximum
arcs that enclose exactly one minimum (topological invariant).
"""

from dataclasses import dataclass

from gba_topology2.src.basin_center import BasinCenter
from gba_topology2.src.max_connection import MaxConnection

__all__ = [
    "MinimumBasin",
    "build_connection_graph",
    "find_minimum_basins",
    "get_basin_arcs",
]


@dataclass(frozen=True)
class MinimumBasin:
    """Represents one minimum's basin boundary.

    Attributes:
        minimum_id: Effective minimum CP ID (e.g., "E-MIN_MC3", "MIN_42").
        boundary_cps: Ordered list of (cp_id, MaxConnection) tuples forming the
            closed loop boundary. Order preserves the cyclic topology so arcs can
            be drawn in sequence. No duplicate endpoints between consecutive arcs.
    """

    minimum_id: str
    boundary_cps: list[tuple[str, MaxConnection]]


def build_connection_graph(
    connections: list[MaxConnection],
) -> dict[str, list[tuple[str, MaxConnection]]]:
    """Build adjacency dict from saddle-maximum connections.

    Maps each CP ID to all outgoing saddle-maximum arcs (edges leaving that CP).
    Only includes connected arcs; filters out "occluded" and "no_target" statuses.

    Args:
        connections: List of MaxConnection objects (typically from
            state.max_connections_list) that have status="connected".

    Returns:
        Dict[cp_id, [(target_cp_id, MaxConnection), ...]]. Maps each CP ID to
        the list of (target CP, connection object) tuples representing outgoing edges.
        Only connected arcs are included.
    """
    graph: dict[str, list[tuple[str, MaxConnection]]] = {}

    for conn in connections:
        # Only include connected arcs (saddle-maximum pairs with status="connected")
        if conn.status != "connected":
            continue

        # Saddle is the source, maximum is the target
        source = conn.saddle_id
        target = conn.max_id

        if target is None:
            # Should not happen if status="connected", but be defensive
            continue

        if source not in graph:
            graph[source] = []

        graph[source].append((target, conn))

    return graph


def _find_cycles_from_node(
    node: str,
    graph: dict[str, list[tuple[str, MaxConnection]]],
    all_cycles: list[list[tuple[str, MaxConnection]]],
    visited: set[str],
    rec_stack: set[str],
    path: list[tuple[str, MaxConnection]],
) -> None:
    """DFS helper to find all simple cycles starting from a node.

    A simple cycle is a path that returns to the starting node without
    repeating any intermediate nodes. This is used to identify closed loops
    of saddle-maximum arcs in the connection graph.

    Args:
        node: Current node in the DFS traversal.
        graph: Adjacency dict from build_connection_graph().
        all_cycles: Accumulator for all cycles found (mutated).
        visited: Set of all visited nodes (used to track which nodes are reachable).
        rec_stack: Set of nodes currently in the recursion stack (used to detect cycles).
        path: Current path of (cp_id, connection) tuples being explored.
    """
    visited.add(node)
    rec_stack.add(node)

    if node in graph:
        for target, conn in graph[node]:
            if target not in visited:
                # Continue DFS to unexplored nodes
                path.append((target, conn))
                _find_cycles_from_node(
                    target, graph, all_cycles, visited, rec_stack, path
                )
                path.pop()
            elif target in rec_stack and len(path) > 1:
                # Found a cycle: target is in the recursion stack and is not
                # the immediate parent (len(path) > 1 ensures we close a loop
                # of more than 2 nodes). Record the cycle from target onwards.
                start_idx = next(
                    (i for i, (cp, _) in enumerate(path) if cp == target),
                    -1,
                )
                if start_idx >= 0:
                    # Extract the cycle: from start_idx to end of path, plus the
                    # closing connection from path[-1] back to target
                    cycle_path = path[start_idx:] + [(target, conn)]
                    all_cycles.append(cycle_path)

    rec_stack.remove(node)


def _find_all_cycles(
    graph: dict[str, list[tuple[str, MaxConnection]]],
) -> list[list[tuple[str, MaxConnection]]]:
    """Find all simple cycles in the connection graph.

    Uses DFS to identify closed loops of saddle-maximum arcs. Each cycle is
    represented as an ordered list of (cp_id, MaxConnection) tuples.

    Args:
        graph: Adjacency dict from build_connection_graph().

    Returns:
        List of cycles, each cycle being a list of (cp_id, connection) tuples.
        Empty list if the graph is acyclic or empty.
    """
    all_cycles: list[list[tuple[str, MaxConnection]]] = []
    visited: set[str] = set()

    for start_node in graph:
        if start_node not in visited:
            _find_cycles_from_node(start_node, graph, all_cycles, visited, set(), [])

    # Deduplicate cycles (same cycle may be found from different starting points)
    # Two cycles are equal if they visit the same nodes in the same order (modulo rotation)
    unique_cycles: list[list[tuple[str, MaxConnection]]] = []
    seen_cycle_nodes: list[frozenset[str]] = []

    for cycle in all_cycles:
        cycle_nodes = frozenset(cp for cp, _ in cycle)
        if cycle_nodes not in seen_cycle_nodes:
            unique_cycles.append(cycle)
            seen_cycle_nodes.append(cycle_nodes)

    return unique_cycles


def _point_in_polygon_3d(
    point: tuple[float, float, float],
    polygon: list[tuple[float, float, float]],
) -> bool:
    """Determine if a 3D point is inside a spherical polygon (on the unit sphere).

    Uses a simplified ray-casting algorithm adapted for spherical geometry:
    project the point and polygon onto a reference plane, then apply 2D
    point-in-polygon test. This is a heuristic; exact inclusion on a sphere
    requires spherical excess calculations.

    Args:
        point: 3D world-frame coordinates of the test point.
        polygon: List of 3D world-frame coordinates forming the polygon vertices,
            in order around the loop.

    Returns:
        True if the point appears to be inside the polygon, False otherwise.
    """
    if len(polygon) < 3:
        return False

    # For a sphere, we project onto the XY plane as a heuristic
    # and apply 2D point-in-polygon ray casting
    px, py = point[0], point[1]

    # Ray casting: cast a ray to the right and count edge crossings
    inside = False
    j = len(polygon) - 1
    for i in range(len(polygon)):
        xi, yi = polygon[i][0], polygon[i][1]
        xj, yj = polygon[j][0], polygon[j][1]

        if (yi > py) != (yj > py) and px < (xj - xi) * (py - yi) / (yj - yi) + xi:
            inside = not inside

        j = i

    return inside


def _find_minimum_in_cycle(
    cycle: list[tuple[str, MaxConnection]],
    basin_centers_dict: dict[str, BasinCenter],
    all_basin_centers: list[BasinCenter],
) -> str | None:
    """Identify which minimum (if any) is enclosed by a cycle of saddle-maximum arcs.

    For each minimum in all_basin_centers, test whether its position is inside
    the cycle's closed loop. A minimum is inside if it is enclosed by the ordered
    set of arc positions.

    Args:
        cycle: List of (cp_id, MaxConnection) tuples forming a closed loop.
        basin_centers_dict: Dict mapping CP ID to BasinCenter for fast lookup.
        all_basin_centers: List of all basin centers (all minima and maxima).

    Returns:
        The minimum_id enclosed by this cycle, or None if no minimum is found.

    Raises:
        ValueError: If multiple minima are enclosed (topological error).
    """
    if not cycle or len(cycle) < 2:
        return None

    # Extract the polygon: ordered list of maxima positions (targets of arcs)
    polygon: list[tuple[float, float, float]] = []
    for _cp_id, conn in cycle:
        if conn.max_id is not None:
            target_bc = basin_centers_dict.get(conn.max_id)
            if target_bc is not None:
                polygon.append(target_bc.position)

    if len(polygon) < 3:
        return None

    # Test each minimum to see if it is inside this polygon
    enclosed_minima: list[str] = []
    for bc in all_basin_centers:
        if bc.polarity == "minimum" and _point_in_polygon_3d(bc.position, polygon):
            enclosed_minima.append(bc.center_id)

    if len(enclosed_minima) == 0:
        return None
    elif len(enclosed_minima) == 1:
        return enclosed_minima[0]
    else:
        # Multiple minima enclosed by one cycle is a topological error
        raise ValueError(
            f"Cycle encloses multiple minima: {enclosed_minima}. "
            "Topology may be malformed."
        )


def find_minimum_basins(
    connections: list[MaxConnection],
    basin_centers: list[BasinCenter],
) -> dict[str, MinimumBasin]:
    """Compute all minimum basin boundaries from saddle-maximum arcs.

    For each minimum, identifies the closed loop of saddle-maximum arcs
    that form its basin boundary by cycle detection on the connection graph.
    Exactly one minimum is enclosed by each cycle (topological invariant).

    Args:
        connections: List of MaxConnection objects with status="connected".
        basin_centers: List of all basin centers (raw and effective maxima/minima).

    Returns:
        Dict[minimum_id, MinimumBasin] keyed by effective minimum CP ID.
        Minima not enclosed by any cycle (e.g., outer boundary minima, or
        disconnected topologies) are omitted.

    Raises:
        ValueError: If topology is malformed (e.g., multiple minima in one cycle).
    """
    # Build the graph and find all cycles
    graph = build_connection_graph(connections)
    cycles = _find_all_cycles(graph)

    if not cycles:
        return {}

    # Create a dict for fast lookup by ID
    basin_centers_dict = {bc.center_id: bc for bc in basin_centers}

    # For each cycle, find the enclosed minimum and construct a MinimumBasin
    basins_dict: dict[str, MinimumBasin] = {}

    for cycle in cycles:
        minimum_id = _find_minimum_in_cycle(cycle, basin_centers_dict, basin_centers)

        if minimum_id is not None and minimum_id not in basins_dict:
            # Build the boundary_cps list: (cp_id, connection) for each arc in the cycle
            # The cycle includes the closing connection, so we include all of them
            boundary_cps: list[tuple[str, MaxConnection]] = []

            # Each cycle element is (cp_id, connection).
            # The cp_id is actually the target of that arc; we want the arc itself
            for cp_id, conn in cycle:
                boundary_cps.append((cp_id, conn))

            basins_dict[minimum_id] = MinimumBasin(
                minimum_id=minimum_id,
                boundary_cps=boundary_cps,
            )

    return basins_dict


def get_basin_arcs(basin: MinimumBasin) -> list[MaxConnection]:
    """Extract ordered MaxConnection objects from a basin.

    Args:
        basin: A MinimumBasin instance.

    Returns:
        Ordered list of MaxConnection objects forming the basin boundary.
    """
    return [conn for _cp_id, conn in basin.boundary_cps]
