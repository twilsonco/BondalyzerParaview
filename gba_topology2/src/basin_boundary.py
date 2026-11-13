"""Basin Boundary Consolidation for gba_topology2.

Identifies minimum basin boundaries by finding saddles that topologically bound
each minimum in the Morse-Smale complex. A minimum's basin boundary consists of
the connections from saddles that are spatially adjacent to the minimum.
"""

from dataclasses import dataclass
from typing import Any

from gba_topology2.src.basin_center import BasinCenter
from gba_topology2.src.max_connection import MaxConnection

__all__ = [
    "MinimumBasin",
    "build_connection_graph",
    "find_minimum_basins",
    "get_basin_arcs",
    "serialize_connection_graph",
    "serialize_minimum_basins",
]


@dataclass(frozen=True)
class MinimumBasin:
    """Represents one minimum's basin boundary.

    Attributes:
        minimum_id: Effective minimum CP ID (e.g., "E-MIN_MC3", "MIN_42").
        boundary_arcs: List of MaxConnection objects representing saddle-to-maximum
            connections emanating from saddles adjacent to this minimum. These arcs
            collectively form the topological boundary of the minimum's basin.
    """

    minimum_id: str
    boundary_arcs: list[MaxConnection]


def build_connection_graph(
    connections: list[MaxConnection],
) -> dict[str, list[MaxConnection]]:
    """Build adjacency dict from saddle-maximum connections.

    Maps each saddle ID to all its outgoing connections to maxima.
    Only includes connected arcs.

    Args:
        connections: List of MaxConnection objects with status="connected".

    Returns:
        Dict[saddle_id, [MaxConnection, ...]]. Maps each saddle ID to the list
        of connection objects from that saddle.
    """
    graph: dict[str, list[MaxConnection]] = {}

    for conn in connections:
        # Only include connected arcs
        if conn.status != "connected" or not conn.verified:
            continue

        saddle_id = conn.saddle_id

        if saddle_id not in graph:
            graph[saddle_id] = []

        graph[saddle_id].append(conn)

    return graph


def _spatial_distance(
    pos1: tuple[float, float, float], pos2: tuple[float, float, float]
) -> float:
    """Euclidean distance between two 3D points."""
    dx = pos1[0] - pos2[0]
    dy = pos1[1] - pos2[1]
    dz = pos1[2] - pos2[2]
    return float((dx * dx + dy * dy + dz * dz) ** 0.5)


def find_minimum_basins(
    connections: list[MaxConnection],
    basin_centers: list[BasinCenter],
) -> dict[str, MinimumBasin]:
    """Compute all minimum basin boundaries from saddle-maximum arcs.

    For each minimum, finds all saddles that are spatially adjacent to it
    (within the k-nearest neighbors by distance). The connections from those
    saddles form the basin boundary. Saddle positions are derived from the
    arc_points[0] (starting point) of each connection.

    Args:
        connections: List of MaxConnection objects with status="connected".
        basin_centers: List of all basin centers (raw and effective maxima/minima).

    Returns:
        Dict[minimum_id, MinimumBasin] keyed by effective minimum CP ID.
        Empty dict if no minima exist or have adjacent saddles.

    Raises:
        ValueError: If topology is malformed.
    """
    # Find all minima
    minima = [bc for bc in basin_centers if bc.polarity == "minimum"]

    if not minima:
        return {}

    # Build connection graph: saddle_id -> [MaxConnection, ...]
    graph = build_connection_graph(connections)

    if not graph:
        return {}

    # Derive saddle positions from arc_points[0] of each connection
    # Build a dict mapping saddle_id -> position (from first arc point)
    saddle_positions: dict[str, tuple[float, float, float]] = {}
    for conn in connections:
        if conn.status == "connected" and conn.verified and conn.arc_points:
            # arc_points[0] is the starting point (saddle position)
            saddle_pos = conn.arc_points[0]
            saddle_id = conn.saddle_id
            if saddle_id not in saddle_positions:
                saddle_positions[saddle_id] = saddle_pos

    # For each minimum, find adjacent saddles (k-nearest by spatial distance)
    basins_dict: dict[str, MinimumBasin] = {}

    for minimum in minima:
        # Find distances from this minimum to all saddles
        distances = []
        for saddle_id, pos in saddle_positions.items():
            dist = _spatial_distance(minimum.position, pos)
            distances.append((dist, saddle_id))

        if not distances:
            continue

        # Sort by distance and take the k nearest (k=3 for a 2-sphere minimum)
        # On a sphere, a generic minimum has 3 neighboring saddles
        distances.sort()
        k_nearest = min(3, len(distances))
        nearby_saddles = [saddle_id for _, saddle_id in distances[:k_nearest]]

        # Collect all connections from nearby saddles
        boundary_arcs = []
        for saddle_id in nearby_saddles:
            if saddle_id in graph:
                boundary_arcs.extend(graph[saddle_id])

        if boundary_arcs:
            basins_dict[minimum.center_id] = MinimumBasin(
                minimum_id=minimum.center_id,
                boundary_arcs=boundary_arcs,
            )

    return basins_dict


def get_basin_arcs(basin: MinimumBasin) -> list[MaxConnection]:
    """Extract ordered MaxConnection objects from a basin.

    Args:
        basin: A MinimumBasin instance.

    Returns:
        List of MaxConnection objects in the basin, ordered for rendering.
    """
    return basin.boundary_arcs


def serialize_connection_graph(
    graph: dict[str, list[MaxConnection]],
) -> dict[str, list[dict[str, Any]]]:
    """Convert a connection graph to a JSON-safe dict for parameter export.

    ONE implementation, deliberately: the viewer writes this into the saved
    params JSON and ``diagnose_params`` writes it into the results JSON, and a
    graph is only inspectable across the two files if both sides serialize it
    identically. The graph itself carries ONLY ``status="connected"`` and
    ``verified=True`` arcs (``build_connection_graph``'s filter); the
    ``status``/``verified`` fields are kept per row so the export is
    self-describing and survives a future change to that filter.

    Args:
        graph: Output of :func:`build_connection_graph`, mapping each
            saddle_id to its outgoing MaxConnection objects.

    Returns:
        Dict mapping saddle_id to a list of plain dicts with keys
        ``max_id``, ``branch_index``, ``status``, ``verified``.
    """
    return {
        saddle_id: [
            {
                "max_id": conn.max_id,
                "branch_index": conn.branch_index,
                "status": conn.status,
                "verified": conn.verified,
            }
            for conn in conns
        ]
        for saddle_id, conns in graph.items()
    }


def serialize_minimum_basins(
    basins: dict[str, MinimumBasin],
) -> dict[str, dict[str, Any]]:
    """Convert detected minimum basins to a JSON-safe dict for export.

    Companion of :func:`serialize_connection_graph`; same single-implementation
    rationale. Boundary saddle/maxima sets are emitted as SORTED lists so the
    JSON is deterministic and diffs across runs compare cleanly.

    Args:
        basins: Output of :func:`find_minimum_basins`, mapping each
            minimum_id to its MinimumBasin.

    Returns:
        Dict mapping minimum_id to a plain dict with keys ``minimum_id``,
        ``n_boundary_arcs``, ``boundary_arcs`` (list of
        ``{saddle_id, branch_index, max_id}`` in basin order),
        ``boundary_saddles`` (sorted str list) and ``boundary_maxima``
        (sorted str list, ``None`` endpoints dropped).
    """
    return {
        min_id: {
            "minimum_id": basin.minimum_id,
            "n_boundary_arcs": len(basin.boundary_arcs),
            "boundary_arcs": [
                {
                    "saddle_id": arc.saddle_id,
                    "branch_index": arc.branch_index,
                    "max_id": arc.max_id,
                }
                for arc in basin.boundary_arcs
            ],
            "boundary_saddles": sorted({arc.saddle_id for arc in basin.boundary_arcs}),
            "boundary_maxima": sorted(
                {arc.max_id for arc in basin.boundary_arcs if arc.max_id is not None}
            ),
        }
        for min_id, basin in basins.items()
    }
