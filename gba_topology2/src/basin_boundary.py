"""Basin Boundary Consolidation for gba_topology2.

Identifies minimum basin boundaries by DESCENT CONNECTIVITY: a minimum's basin
is bounded by the saddle->maximum arcs at the saddles whose descent branches
run into that minimum (basin v10). The v9 rule took the k=3 spatially nearest
saddles instead, and it saturated -- with few saddles every basin claimed the
same arcs and the same "raise k" advice, a real structure reported as a
parameter problem. The graph statement -- which saddles drain here -- is what
the boundary of a basin means.
"""

from collections.abc import Mapping, Sequence
from dataclasses import MISSING, dataclass, fields
from types import UnionType
from typing import Any, Union, get_args, get_origin, get_type_hints, overload

from gba_topology2.src.basin_center import BasinCenter
from gba_topology2.src.max_connection import MaxConnection, MinConnection

__all__ = [
    "MinimumBasin",
    "build_connection_graph",
    "connection_from_state_dict",
    "find_minimum_basins",
    "get_basin_arcs",
    "min_connection_from_state_dict",
    "min_connection_to_state_dict",
    "serialize_connection_graph",
    "serialize_minimum_basins",
]


@overload
def connection_from_state_dict(
    d: Mapping[str, Any], cls: type[MaxConnection] = ...
) -> MaxConnection: ...
@overload
def connection_from_state_dict(
    d: Mapping[str, Any], cls: type[MinConnection]
) -> MinConnection: ...
def connection_from_state_dict(
    d: Mapping[str, Any], cls: type[Any] = MaxConnection
) -> Any:
    """Rehydrate a connection record from its state/params dict form.

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
        cls: The dataclass to rebuild -- :class:`MaxConnection` (default) or
            :class:`MinConnection`. The descent rows are the basin adjacency
            (basin v10), so they cross the same file boundary and get the
            same None-safe treatment.

    Returns:
        The reconstructed connection record.

    Raises:
        TypeError: A required non-Optional field is None or missing.
        ValueError: A present value cannot be coerced to the field's type
            (e.g. ``branch_index="x"``).
    """
    hints = get_type_hints(cls)
    kwargs: dict[str, Any] = {}
    for fld in fields(cls):
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
            else:  # pragma: no cover - no other tuple shape in the records
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
    return cls(**kwargs)


def min_connection_to_state_dict(c: MinConnection) -> dict[str, Any]:
    """JSON-safe row for a descent record, the basin adjacency (basin v10).

    Mirrors the ascent rows written by ``state.max_connections_list``: plain
    scalars, ``None`` preserved for Optional fields (it is a legitimate stored
    value -- ``min_id: None`` means "this branch drained nowhere"), tuples as
    lists, and ``arc_points`` stored exactly as on the ascent side. That is
    the v9 lesson applied: a saved observation must carry everything the
    render reads, and in v10 the descent rows are what the selection ring
    reads.

    Args:
        c: One descent record.

    Returns:
        A dict with one key per :class:`MinConnection` field.
    """
    return {
        "saddle_id": c.saddle_id,
        "branch_index": c.branch_index,
        "min_id": c.min_id,
        "status": c.status,
        "direction_mode": c.direction_mode,
        "cross_track_deg": c.cross_track_deg,
        "along_track_deg": c.along_track_deg,
        "separation_deg": c.separation_deg,
        "arc_points": [list(p) for p in c.arc_points],
        "saddle_value": c.saddle_value,
        "min_value": c.min_value,
        "verified": c.verified,
        "blocker_id": c.blocker_id,
        "blocker_polarity": c.blocker_polarity,
        "method": c.method,
        "failure_gate": c.failure_gate,
        "stall_point": list(c.stall_point) if c.stall_point else None,
        "transited_ids": list(c.transited_ids),
        "completed_leg_deg": c.completed_leg_deg,
    }


def min_connection_from_state_dict(d: Mapping[str, Any]) -> MinConnection:
    """Rehydrate a descent record with the ascent's None-safe rehydrator."""
    return connection_from_state_dict(d, MinConnection)


@dataclass(frozen=True)
class MinimumBasin:
    """Represents one minimum's basin boundary.

    Attributes:
        minimum_id: Effective minimum CP ID (e.g., "E-MIN_MC3", "MIN_42").
        boundary_arcs: List of MaxConnection objects representing saddle-to-maximum
            connections emanating from the saddles whose descent branches drain
            into this minimum (basin v10 adjacency -- connectivity off the
            field, not spatial proximity). These arcs collectively form the
            topological boundary of the minimum's basin.
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


def find_minimum_basins(
    connections: list[MaxConnection],
    basin_centers: list[BasinCenter],
    descent_connections: Sequence[MinConnection],
) -> dict[str, MinimumBasin]:
    """Compute all minimum basin boundaries from the connection topology.

    basin v10: a minimum's basin is bounded by the saddle->maximum arcs at the
    saddles whose DESCENT branches run into that minimum. A saddle's stable
    manifold drains into the minima it bounds -- that is the statement. The v9
    rule took the k=3 spatially nearest saddles by Euclidean distance instead,
    and it saturated: on a 2-saddle system every basin claimed the same 3 arcs
    of 3 and every ring passed through both saddles, so a real structure read
    as "k too small; raise to 4-8". Adjacency is now measured off the field
    (the descent connector: cone capture on minima, disc occlusion, gradient
    fall), never off proximity. When the descent table is empty no basin has a
    boundary -- there is no proximity fallback, because proximity is exactly
    what this rule replaced.

    Args:
        connections: The ascent table (all records; the graph filters to
            connected, verified arcs via :func:`build_connection_graph`).
        basin_centers: List of all basin centers (raw and effective maxima
            /minima); only the minima get basins.
        descent_connections: The descent table from
            :func:`max_connection.connect_seed_to_minima`, raw (pre-refine;
            refine is an ascent-side concept). A saddle is adjacent to a
            minimum when one of its descent records has
            ``status == "connected"`` and ``min_id`` equal to that minimum's
            ``center_id``.

    Returns:
        Dict[minimum_id, MinimumBasin] keyed by minimum CP ID. A minimum with
        no draining saddle is absent.

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

    # The adjacency, off the field: minimum_id -> the saddle_ids that drain
    # into it. Sorted by saddle id so a basin's arc list depends only on the
    # record SET, never on the order the caller assembled it.
    drains_into: dict[str, set[str]] = {}
    for desc in descent_connections:
        if desc.status == "connected" and desc.min_id is not None:
            drains_into.setdefault(desc.min_id, set()).add(desc.saddle_id)

    basins_dict: dict[str, MinimumBasin] = {}

    for minimum in minima:
        adjacent = drains_into.get(minimum.center_id, set())

        # Collect all ascent arcs from the draining saddles.
        boundary_arcs: list[MaxConnection] = []
        for saddle_id in sorted(adjacent):
            boundary_arcs.extend(graph.get(saddle_id, []))

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
