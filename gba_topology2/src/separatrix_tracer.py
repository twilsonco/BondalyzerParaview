"""Cluster-Aware Separatrix Tracer for gba_topology2 (Basin Locator Step 3-lite).

Traces gradient-flow paths on the triangulated atom sphere from a saddle's
separatrix ports to the basins they terminate in, so the Morse-Smale adjacency
used by ``basin_validator`` is built from actual flow topology instead of a
one-shot tangent-direction heuristic.

Why naive gradient following is not enough
------------------------------------------
The condensed fields carry numerical noise at the mesh scale, which produces
spurious critical points. Two failure modes follow:

1. *Transit noise*: a monotone min -> max slope can be peppered with spurious
   extrema. A streamline dies in the first spurious dip long before the real
   target. The micro-cluster reduction already proves such groups are not CPs
   (net index 0, annihilated). The tracer therefore treats an annihilated
   cluster's disk as a **pass-through zone**: entering it is not termination.

2. *Cluster targets*: the real destination is often a multi-CP cluster whose
   members include spurious extrema on its flanks. The tracer treats the
   **whole cluster's disk** (the reduction unit's excision ring), not any
   single member vertex, as the absorption target: entering the disk
   terminates the path and assigns it to the cluster's effective CP.

Zone model
----------
Every zone is a spherical cap: a unit direction (centred atom frame) plus a
cosine radius. Kinds:

- ``absorb_max`` / ``absorb_min`` / ``absorb_sad``: entering the cap ends the
  walk and yields the zone label as the endpoint.
- ``passthrough``: entering is ignored (annihilated noise clusters).
- ``exclude``: the launching saddle's own cap; the walk starts inside it, so
  it must not count as a target.

The walk is a greedy steepest-ascent/descent on the mesh 1-skeleton. Because
each step strictly increases (decreases) the evaluated field, the walk cannot
cycle. Termination confidence:

- ``"absorbed"``: entered a matching absorption cap (high confidence).
- ``"raw_extremum"``: reached a local extremum of the raw data with no cap
  (medium confidence; the target CP exists but zone geometry missed it).
- ``"stalled"`` / ``"max_steps"``: no admissible move, or step budget spent
  (low confidence; caller should fall back to the tangent heuristic).

Frame contract: all positions (mesh vertices, zone centers, start points) are
in the **centred atom frame** (nucleus at origin). Callers map world-frame
inputs with ``sphere_frame.to_local``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

__all__ = [
    "TraceOutcome",
    "TraceZone",
    "ZoneKind",
    "build_walk_adjacency",
    "make_zones",
    "trace_separatrix",
]

ZoneKind = str  # "absorb_max" | "absorb_min" | "absorb_sad" | "passthrough" | "exclude"

# Which zone kinds stop a walk of the given direction.
_ABSORB_BY_DIRECTION: dict[str, frozenset[str]] = {
    "ascent": frozenset({"absorb_max", "absorb_sad"}),
    "descent": frozenset({"absorb_min", "absorb_sad"}),
}


@dataclass(frozen=True)
class TraceZone:
    """One spherical-cap zone on the atom sphere (centred frame).

    Attributes:
        label: Identifier of the CP the zone represents (basin-center id
            space, e.g. ``"MAX_142"``, ``"E-MAX_MC28"``, ``"SAD_7"``).
        kind: Zone behaviour on entry; see ``ZoneKind``.
        center_unit: (3,) unit direction of the cap centre.
        cos_radius: Cosine of the cap's angular radius. A point with unit
            direction ``p`` is inside when ``dot(p, center_unit) >= cos_radius``.
    """

    label: str
    kind: ZoneKind
    center_unit: tuple[float, float, float]
    cos_radius: float

    def contains(self, point: npt.ArrayLike) -> bool:
        """True when a sphere point lies inside this cap."""
        p = np.asarray(point, dtype=np.float64)
        n = float(np.linalg.norm(p))
        if n <= 0.0:
            return False
        return bool(
            np.dot(p / n, np.asarray(self.center_unit, dtype=np.float64))
            >= self.cos_radius
        )


@dataclass(frozen=True)
class TraceOutcome:
    """Result of one traced separatrix branch.

    Attributes:
        direction: ``"ascent"`` (ridge, towards maxima) or ``"descent"``.
        start_vertex: Mesh vertex the walk launched from.
        target_label: Zone label the walk terminated in, or None.
        target_kind: Kind of that zone, or None when not absorbed.
        status: ``"absorbed"`` | ``"raw_extremum"`` | ``"stalled"`` |
            ``"max_steps"``.
        n_steps: Number of mesh hops taken.
        path: Vertex ids visited, start first (kept for debugging/inspection).
    """

    direction: str
    start_vertex: int
    target_label: str | None
    target_kind: str | None
    status: str
    n_steps: int
    path: list[int] = field(default_factory=list)

    @property
    def confidence(self) -> str:
        """Coarse confidence class for downstream ranking."""
        return {"absorbed": "high", "raw_extremum": "medium"}.get(self.status, "low")


def _unit(pts: npt.NDArray[np.float64], v: int) -> npt.NDArray[np.float64]:
    """Unit direction of vertex ``v``."""
    p = np.asarray(pts[v], dtype=np.float64)
    n = float(np.linalg.norm(p))
    return np.asarray(p / n if n > 0.0 else p, dtype=np.float64)


def trace_separatrix(
    start_vertex: int,
    direction: str,
    pts: npt.NDArray[np.float64],
    f_vals: npt.NDArray[np.float64],
    neighbors: list[list[int]],
    zones: list[TraceZone],
    max_steps: int = 512,
) -> TraceOutcome:
    """Walk the mesh 1-skeleton along the gradient until a zone absorbs it.

    Args:
        start_vertex: Mesh vertex to launch from (a separatrix port seed,
            already offset from the saddle centre).
        direction: ``"ascent"`` (follow increasing f) or ``"descent"``.
        pts: (N, 3) centred-frame vertex coordinates.
        f_vals: (N,) field values.
        neighbors: Per-vertex adjacency (mesh 1-skeleton).
        zones: Cap zones consulted after every hop.
        max_steps: Hop budget per branch.

    Returns:
        TraceOutcome describing the termination.
    """
    if direction not in ("ascent", "descent"):
        raise ValueError(f"direction must be 'ascent' or 'descent', got {direction!r}")
    absorb_kinds = _ABSORB_BY_DIRECTION[direction]
    sign = 1.0 if direction == "ascent" else -1.0

    # Zones are indexed once; matching is a dot product per zone per hop,
    # and zone counts are small (tens), so a linear scan is fine.
    zc = np.asarray([z.center_unit for z in zones], dtype=np.float64).reshape(-1, 3)
    zr = np.asarray([z.cos_radius for z in zones], dtype=np.float64).reshape(-1)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    punit = pts / np.where(norms > 0.0, norms, 1.0)
    # Characteristic angular edge length of the mesh, used as the clearance
    # margin when escaping a noise disk so the resume vertex is outside it.
    edge_cos: list[float] = []
    for v, nbrs in enumerate(neighbors):
        for nb in nbrs[:2]:
            edge_cos.append(float(np.dot(punit[v], punit[int(nb)])))
    step_pitch = (
        math.acos(max(-1.0, min(1.0, float(np.median(edge_cos))))) if edge_cos else 0.05
    )

    def zone_hit(v: int) -> int | None:
        """Index of the first zone whose cap contains vertex v, or None."""
        cosines = zc @ _unit(pts, v)
        inside = np.nonzero(cosines >= zr)[0]
        for idx in inside:
            # passthrough and exclude zones are "entered" but never terminate;
            # return them so the caller can record, not stop.
            if zones[int(idx)].kind in absorb_kinds:
                return int(idx)
        return None

    def any_zone(v: int) -> list[int]:
        cosines = zc @ _unit(pts, v)
        return [int(i) for i in np.nonzero(cosines >= zr)[0]]

    path: list[int] = [int(start_vertex)]
    cur = int(start_vertex)

    # Consult zones at the start vertex itself: a port seed placed inside the
    # target disk (tiny saddles) should terminate immediately.
    hit = zone_hit(cur)
    if hit is not None:
        return TraceOutcome(
            direction=direction,
            start_vertex=cur,
            target_label=zones[hit].label,
            target_kind=zones[hit].kind,
            status="absorbed",
            n_steps=0,
            path=path,
        )

    def passthrough_hits(v: int) -> list[int]:
        cosines = zc @ _unit(pts, v)
        return [
            int(i)
            for i in np.nonzero(cosines >= zr)[0]
            if zones[i].kind == "passthrough"
        ]

    def tunnel_exit(zone_idx: int, entry_vertex: int, margin_rad: float) -> int | None:
        """Escape a noise disk by continuing the great circle through it.

        A pass-through zone is a spurious sink of the noisy field: every
        boundary vertex on its far side is itself a local extremum, so hopping
        to the boundary re-traps the walk. The separatrix, however, passes
        *through* the noise region. So the escape projects the incoming
        trajectory forward along the great circle through (entry -> centre)
        and resumes one radius-plus-margin beyond the centre.

        Args:
            zone_idx: Index of the pass-through zone entered.
            entry_vertex: First vertex of the walk that fell inside the zone.
            margin_rad: Extra angular distance to clear the zone's edge.

        Returns:
            Resume vertex id, or None when the geometry is degenerate.
        """
        z = zones[zone_idx]
        c = np.asarray(z.center_unit, dtype=np.float64)
        e = _unit(pts, entry_vertex)
        # Travel direction tangent to the sphere at the zone centre.
        d = c - e
        d = d - float(np.dot(d, c)) * c
        dn = float(np.linalg.norm(d))
        if dn < 1e-12:
            return None
        d = d / dn
        ang = math.acos(max(-1.0, min(1.0, z.cos_radius))) + margin_rad
        exit_pt = math.cos(ang) * c + math.sin(ang) * d
        en = float(np.linalg.norm(exit_pt))
        if en <= 0.0:
            return None
        return int(np.argmax(punit @ (exit_pt / en)))

    # zone idx -> vertex where the walk first entered it (reconstructs the
    # incoming trajectory for the tunnel), and the set of zones already left.
    zone_entry: dict[int, int] = {}
    zone_exited: set[int] = set()

    def in_banned(v: int) -> bool:
        """True when vertex v lies inside a pass-through zone we already left."""
        return any(zones[idx].contains(pts[v]) for idx in zone_exited)

    for _ in range(max_steps):
        cur_f = float(f_vals[cur])
        best_nb = -1
        best_f = cur_f
        for nb in neighbors[cur]:
            # Never re-enter a pass-through zone we have already tunneled out of:
            # its spurious extrema are noise, not a basin.
            if zone_exited and in_banned(nb):
                continue
            nf = float(f_vals[nb])
            # Strict improvement in the walk direction; tie-break on vertex id
            # for determinism (mirrors the SoS convention in morse_detector).
            if sign * nf > sign * best_f or (
                sign * nf == sign * best_f and best_nb >= 0 and nb < best_nb
            ):
                best_nb, best_f = int(nb), nf
        if best_nb < 0:
            # Local extremum of the raw data.
            hits = any_zone(cur)
            absorbed = [h for h in hits if zones[h].kind in absorb_kinds]
            if absorbed:
                z = zones[absorbed[0]]
                return TraceOutcome(
                    direction, cur, z.label, z.kind, "absorbed", len(path) - 1, path
                )
            # Noise-trapped inside a pass-through disk? Tunnel straight through
            # along the incoming great circle and resume beyond it.
            escaped = False
            for pt_idx in passthrough_hits(cur):
                if pt_idx in zone_exited:
                    continue
                entry_v = zone_entry.get(pt_idx, cur)
                radius = math.acos(max(-1.0, min(1.0, zones[pt_idx].cos_radius)))
                nxt = tunnel_exit(pt_idx, entry_v, max(radius, step_pitch))
                if nxt is not None and nxt != cur:
                    zone_exited.add(pt_idx)
                    cur = int(nxt)
                    path.append(cur)
                    escaped = True
                    break
            if escaped:
                continue
            return TraceOutcome(
                direction, cur, None, None, "raw_extremum", len(path) - 1, path
            )
        cur = best_nb
        path.append(cur)
        # Remember where the walk first entered each pass-through zone so the
        # tunnel can reconstruct the incoming trajectory.
        for pt_idx in passthrough_hits(cur):
            zone_entry.setdefault(pt_idx, cur)
        hit = zone_hit(cur)
        if hit is not None:
            z = zones[hit]
            return TraceOutcome(
                direction, cur, z.label, z.kind, "absorbed", len(path) - 1, path
            )

    return TraceOutcome(direction, cur, None, None, "max_steps", len(path) - 1, path)


def make_zones(
    *,
    visible_maxima: list[tuple[str, tuple[float, float, float]]],
    visible_minima: list[tuple[str, tuple[float, float, float]]],
    visible_saddles: list[tuple[str, tuple[float, float, float]]],
    absorption_caps: dict[str, float] | None = None,
    passthrough: list[tuple[str, tuple[float, float, float], float]] | None = None,
    default_radius_rad: float = 0.0,
) -> list[TraceZone]:
    """Assemble trace zones from visible CPs, cluster caps, and annihilations.

    Zone geometry is purely angular (a unit direction plus a cosine radius), so
    no mesh is needed; the tracer tests vertices against these caps itself.

    Args:
        visible_maxima: ``(label, position)`` for each visible maximum.
        visible_minima: ``(label, position)`` for each visible minimum.
        visible_saddles: ``(label, position)`` for each visible saddle.
        absorption_caps: Optional ``label -> angular radius (rad)`` overrides,
            typically the reduction unit's excision ring radius, so a whole
            cluster absorbs as one CP.
        passthrough: ``(label, position, radius_rad)`` for annihilated
            (net-index-0) clusters: noise regions that must not terminate.
        default_radius_rad: Cap radius for CPs without an override.

    Returns:
        Zone list in stable order (absorption first, then passthrough).
    """
    caps = absorption_caps or {}
    zones: list[TraceZone] = []

    def add(
        label: str,
        kind: ZoneKind,
        pos: tuple[float, float, float],
        radius_rad: float,
    ) -> None:
        p = np.asarray(pos, dtype=np.float64)
        n = float(np.linalg.norm(p))
        if n <= 0.0:
            return
        r = max(radius_rad, 1e-6)
        zones.append(
            TraceZone(
                label=label,
                kind=kind,
                center_unit=(float(p[0] / n), float(p[1] / n), float(p[2] / n)),
                cos_radius=math.cos(r),
            )
        )

    for label, pos in visible_maxima:
        add(label, "absorb_max", pos, caps.get(label, default_radius_rad))
    for label, pos in visible_minima:
        add(label, "absorb_min", pos, caps.get(label, default_radius_rad))
    for label, pos in visible_saddles:
        add(label, "absorb_sad", pos, caps.get(label, default_radius_rad))
    for label, pos, rad in passthrough or []:
        add(label, "passthrough", pos, rad)

    return zones


def build_walk_adjacency(
    triangles: npt.NDArray[np.integer[Any]], n_pts: int
) -> list[list[int]]:
    """Mesh 1-skeleton adjacency from triangle indices.

    Args:
        triangles: (M, 3) vertex indices.
        n_pts: Vertex count.

    Returns:
        Per-vertex neighbour lists (sorted, de-duplicated).
    """
    adj: list[set[int]] = [set() for _ in range(n_pts)]
    for tri in triangles:
        a, b, c = (int(tri[0]), int(tri[1]), int(tri[2]))
        adj[a].update((b, c))
        adj[b].update((a, c))
        adj[c].update((a, b))
    return [sorted(s) for s in adj]


def build_walk_adjacency_from_edges(
    edges: npt.NDArray[np.integer[Any]], n_pts: int
) -> list[list[int]]:
    """Mesh 1-skeleton adjacency from an (E, 2) edge array.

    Args:
        edges: (E, 2) vertex index pairs.
        n_pts: Vertex count.

    Returns:
        Per-vertex neighbour lists (sorted, de-duplicated).
    """
    adj: list[set[int]] = [set() for _ in range(n_pts)]
    for e in edges:
        a, b = int(e[0]), int(e[1])
        adj[a].add(b)
        adj[b].add(a)
    return [sorted(s) for s in adj]
