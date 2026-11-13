"""Basin Boundary Consolidation for gba_topology2.

Identifies minimum basin boundaries by finding saddles that topologically bound
each minimum in the Morse-Smale complex. A minimum's basin boundary consists of
the connections from saddles that are spatially adjacent to the minimum.
"""

from collections.abc import Mapping
from dataclasses import MISSING, dataclass, fields
from types import UnionType
from typing import Any, Union, get_args, get_origin, get_type_hints

from gba_topology2.src.basin_center import BasinCenter
from gba_topology2.src.max_connection import MaxConnection

__all__ = [
    "MinimumBasin",
    "build_connection_graph",
    "connection_from_state_dict",
    "find_minimum_basins",
    "get_basin_arcs",
    "serialize_connection_graph",
    "serialize_minimum_basins",
]


def connection_from_state_dict(d: Mapping[str, Any]) -> MaxConnection:
    """Rehydrate a :class:`MaxConnection` from its state/params dict form.

    The viewer's basin pipeline receives ``state.max_connections_list`` rows
    (JSON-safe dicts written by ``max_to_dict``) and must rebuild the typed
    objects. Hand-written per-field coercion is what broke in schema v9:
    a row carrying ``"cross_track_deg": null`` fed ``float(None)``, the
    per-row ``except`` swallowed the ``TypeError``, and the whole arc table
    vanished from the basin computation. This function is driven by the
    dataclass itself, so it cannot drift from it, and it honours the field
    contract: **None is a legitimate stored value for an Optional field**
    (``cross_track_deg: None`` means "nothing was hit") and is passed
    through unchanged; None/missing for a field with a default gets the
    default; None/missing for a required non-Optional field raises.

    Args:
        d: A serialized connection row (keys as in ``max_connections_list``).

    Returns:
        The reconstructed :class:`MaxConnection`.

    Raises:
        TypeError: A required non-Optional field is None or missing.
        ValueError: A present value cannot be coerced to the field's type
            (e.g. ``branch_index="x"``).
    """
    hints = get_type_hints(MaxConnection)
    kwargs: dict[str, Any] = {}
    for fld in fields(MaxConnection):
        raw: Any = d.get(fld.name)
        ann = hints[fld.name]
        args = get_args(ann)
        # Only ``X | None`` unions are treated as Optional; a plain
        # tuple[...] annotation keeps its OUTER tuple as the base type —
        # get_args(tuple[tuple[float, float, float], ...]) is
        # (tuple[float, float, float], Ellipsis), so naively taking args[0]
        # would coerce arc_points as if it were a flat stall_point.
        is_optional = get_origin(ann) in (UnionType, Union)
        base = next(a for a in args if a is not type(None)) if is_optional else ann

        if raw is None:
            if fld.default is not MISSING:
                kwargs[fld.name] = fld.default
                continue
            if fld.default_factory is not MISSING:
                kwargs[fld.name] = fld.default_factory()
                continue
            if is_optional:
                kwargs[fld.name] = None
                continue
            raise TypeError(f"connection row is missing required field {fld.name!r}")

        origin = get_origin(base)
        if origin is tuple:
            elem_args = get_args(base)
            elem = elem_args[0] if elem_args else Any
            if get_origin(elem) is tuple:  # arc_points: tuple[tuple[float,...], ...]
                kwargs[fld.name] = tuple(tuple(float(c) for c in pt) for pt in raw)
            elif elem is str:
                kwargs[fld.name] = tuple(str(x) for x in raw)
            elif elem is float:
                kwargs[fld.name] = tuple(float(x) for x in raw)
            elif elem is int:
                kwargs[fld.name] = tuple(int(x) for x in raw)
            else:  # pragma: no cover - no other tuple shape in MaxConnection
                kwargs[fld.name] = tuple(raw)
        elif base is bool:
            kwargs[fld.name] = bool(raw)
        elif base is int:
            kwargs[fld.name] = int(raw)
        elif base is float:
            kwargs[fld.name] = float(raw)
        elif base is str:
            kwargs[fld.name] = str(raw)
        else:
            kwargs[fld.name] = raw
    return MaxConnection(**kwargs)


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
