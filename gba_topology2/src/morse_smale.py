"""Morse-Smale Zone Construction for gba_topology2.

Builds the spherical-cap zone set the separatrix tracer needs from the plain
data structures the reduction stage produces. Shared by the interactive viewer
(``sphere_viewer``) and the headless harness (``diagnose_params``) so the two
cannot drift: the topology an operator sees on screen must be the topology the
harness reproduces from a saved snapshot.

Why a zone set is needed
------------------------
The condensed fields carry numerical noise at the mesh scale, which manufactures
spurious critical points. A naive gradient walk therefore terminates in the
wrong place in two ways: it dies on a spurious extremum inside a cluster that
the reduction proved is not a CP at all (net index 0, annihilated), and it dies
on a spurious flank extremum of a multi-CP cluster whose real destination is the
cluster's *effective* CP.

Zones encode the reduction's conclusions as boundary conditions on the walk:

- an absorption cap per visible CP, sized so a whole cluster absorbs as ONE CP
  (the effective CP's excision ring, not any single member vertex);
- a pass-through cap per annihilated cluster, which the walk crosses without
  terminating (and, when the noise disk traps it, tunnels straight through).

Labels live in basin-center / launch-seed id space so traced endpoints plug
straight into ``basin_validator.SaddleAdjacency``: raw CPs as ``MAX_<vid>`` /
``MIN_<vid>`` / ``SAD_<vid>``, effective CPs as their ``id_label``
(``E-MAX_MC28`` ...), annihilated units as ``ANNUL_<unit_id>``.

Frame contract: every position is in the **centred atom frame** (nucleus at the
origin). Effective CP dicts carry world-frame positions, so ``center`` is
required to map them back; pass the origin for meshes that are already centred.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from typing import Any, Protocol

from gba_topology2.src.morse_detector import CriticalPoint
from gba_topology2.src.separatrix_tracer import TraceZone, make_zones
from gba_topology2.src.sphere_frame import to_local

__all__ = [
    "ExcisionRadius",
    "UnitGeometry",
    "lone_cp_excision_radius_rad",
    "trace_zones_from_reduction",
]


class UnitGeometry(Protocol):
    """The subset of a reduction unit the zone builder reads.

    ``sphere_viewer.ReductionUnit`` satisfies this structurally. Declaring it
    here keeps the import one-directional: the viewer depends on this module,
    never the reverse.
    """

    @property
    def unit_id(self) -> str:
        """Display identifier (``MC12`` for a lone cluster, ``MS1`` bridged)."""
        ...

    @property
    def centroid(self) -> tuple[float, float, float]:
        """Unit centroid in the centred atom frame."""
        ...

    @property
    def as_cluster(self) -> ExcisionRadius:
        """Cluster view carrying the angular excision radius."""
        ...


class ExcisionRadius(Protocol):
    """The angular radius accessor ``UnitGeometry.as_cluster`` must provide."""

    @property
    def angular_radius_rad(self) -> float:
        """Angular radius of the unit's boundary ring, in radians."""
        ...


def lone_cp_excision_radius_rad(k_pitch: float, delta_theta_mesh_rad: float) -> float:
    """Excision radius for a lone critical point, matching the launch seeds.

    The boundary margin, or half the clustering threshold, whichever is larger
    -- the same rule ``build_launch_requests`` uses, so a raw CP's absorption cap
    and its excision ring agree.

    Args:
        k_pitch: Micro-clustering pitch multiplier.
        delta_theta_mesh_rad: Intrinsic angular mesh pitch.

    Returns:
        Radius in radians, always at least the 1.5 degree boundary margin.
    """
    return max(math.radians(1.5), k_pitch * delta_theta_mesh_rad / 2.0)


def trace_zones_from_reduction(
    *,
    maxima: Sequence[CriticalPoint],
    minima: Sequence[CriticalPoint],
    saddles: Sequence[CriticalPoint],
    effective_cps: Iterable[dict[str, Any]],
    units: Iterable[UnitGeometry],
    eff_ring_radius: dict[str, float],
    hidden_vertex_ids: set[int],
    annihilated_unit_ids: Iterable[str],
    center: tuple[float, float, float],
    delta_theta_mesh_rad: float,
    k_pitch: float,
) -> list[TraceZone]:
    """Assemble absorption and pass-through caps from a finished reduction.

    Args:
        maxima: Raw maximum critical points (centred frame).
        minima: Raw minimum critical points (centred frame).
        saddles: Raw saddle critical points (centred frame).
        effective_cps: Effective CP UI dicts from the reduction stage. Their
            ``position`` is **world frame** (the builder mapped it) and is the
            nudged centre when the nudge toggle is on.
        units: All reduction units; only the annihilated ones are read, for
            their centroid and angular radius.
        eff_ring_radius: Initial excision radius per effective CP, keyed by
            ``id_label``. Falls back to the lone-critical-point rule.
        hidden_vertex_ids: Raw vertices collapsed into an effective entity; they
            get no cap of their own.
        annihilated_unit_ids: Unit ids reduced to nothing (net index 0). These
            become pass-through caps.
        center: World-frame nucleus, used to map effective positions back to the
            centred frame the tracer walks in.
        delta_theta_mesh_rad: Intrinsic angular mesh pitch.
        k_pitch: Micro-clustering pitch multiplier, which sets the lone-CP cap
            radius.

    Returns:
        Zone list for ``separatrix_tracer`` / ``build_traced_adjacency``.
    """
    lone_a0 = lone_cp_excision_radius_rad(k_pitch, delta_theta_mesh_rad)

    max_list: list[tuple[str, tuple[float, float, float]]] = []
    min_list: list[tuple[str, tuple[float, float, float]]] = []
    sad_list: list[tuple[str, tuple[float, float, float]]] = []
    caps: dict[str, float] = {}

    for cp, prefix, bucket in (
        (maxima, "MAX", max_list),
        (minima, "MIN", min_list),
        (saddles, "SAD", sad_list),
    ):
        for point in cp:
            if point.vertex_id in hidden_vertex_ids:
                continue
            label = f"{prefix}_{point.vertex_id}"
            bucket.append((label, point.position))
            caps[label] = lone_a0

    for eff in effective_cps:
        label = str(eff["id_label"])
        pos = to_local(tuple(eff["position"]), center)
        kind = eff["type"]
        if kind == "E-MAX":
            max_list.append((label, pos))
        elif kind == "E-MIN":
            min_list.append((label, pos))
        elif kind == "E-SAD":
            sad_list.append((label, pos))
        else:
            continue
        caps[label] = eff_ring_radius.get(label, lone_a0)

    passthrough: list[tuple[str, tuple[float, float, float], float]] = []
    by_id = {u.unit_id: u for u in units}
    for uid in sorted(annihilated_unit_ids):
        unit = by_id.get(uid)
        if unit is None:
            continue
        passthrough.append(
            (
                f"ANNUL_{uid}",
                unit.centroid,
                max(unit.as_cluster.angular_radius_rad, lone_a0),
            )
        )

    return make_zones(
        visible_maxima=max_list,
        visible_minima=min_list,
        visible_saddles=sad_list,
        absorption_caps=caps,
        passthrough=passthrough,
    )
