"""Excision-Based Saddle Launch Seeds for gba_topology2 (Basin Locator Step 2).

Determines the directions in which separatrix branches leave a saddle and
produces the on-sphere seed positions from which Step 3 integrates them, by
profiling the field on the saddle's **excision boundary**.

Why excision, and not the cyclic link
-------------------------------------
``saddle_launch.extract_saddle_launch_seed`` reads the same information from a
mesh vertex's cyclic link. That method is parameter-free, interpolation-free and
uniform in fold order, and it is the right probe *at a mesh vertex*. But it needs
a ``vertex_id`` to walk the incident-triangle link graph, and an effective
critical point has none: its position is a mean of member directions, optionally
displaced by the harmonic nudge. It is not a mesh vertex, it has no link, and
nothing can be scanned.

The excision loop is defined for **any** point on the sphere, which is what makes
it the substrate that generalizes from raw critical points to effective ones --
the point of the whole reduction stage. The cost of that generality is that the
loop leaves the mesh, so the field on it must be interpolated, and the port
azimuths must be refined between samples. Both are handled here.

The method
----------
1. Profile ``f(phi)`` on the closed excision loop.
2. Ridge ports (local maxima of ``f|_loop``) are the ascending / unstable
   branches; valley ports (local minima) are the descending / stable branches.
   This is the same upper/lower sector decomposition the link scan performs, read
   at a radius instead of at the vertex.
3. Locate each port between samples by parabolic refinement, so the launch
   azimuth is not quantized to the ring's 7.5 deg sample pitch.
4. The launch direction at a port is the outward geodesic away from the ring
   centre: the separatrix arrives from the centre, crosses the loop, and continues
   out. The launch position is the port itself, which is outside the degenerate
   core by construction -- that is what excising buys.

Fold order is **measured** from the port count and audited against the index the
reduction stage reported, so a saddle whose observed fold order disagrees with its
index is visible in the viewer rather than silently traced with the wrong number
of branches.

Frame contract: all inputs and outputs are in the CENTRED atom frame, where the
nucleus is the origin and the outward normal of a surface point p is p / ||p||.
The viewer maps positions to the world frame with ``sphere_frame.to_global``
before drawing.
"""

import math
from typing import Any

import numpy as np
import numpy.typing as npt

from gba_topology2.src.excision_boundary import (
    KNearestFinder,
    profile_boundary_loop,
)
from gba_topology2.src.micro_cluster import BoundaryRingPoint, ring_frame
from gba_topology2.src.saddle_launch import LaunchBranch, SaddleLaunchSeed

__all__ = ["FAILURE_REASONS", "extract_excision_launch_seed"]

# Why a ring produced no branches. Surfaced in the viewer so a failure can be
# inspected rather than silently rendered as an empty scene.
FAILURE_REASONS = frozenset(
    {"empty_ring", "too_few_samples", "no_ports", "unbalanced", "below_min_k"}
)

# A balanced 1 + 1 signature is a ring around an extremum, not a saddle: the
# loop has one ridge and one valley because f rises and falls once around it.
_MIN_SADDLE_K = 2


def _unit(value: npt.NDArray[np.float64]) -> npt.NDArray[np.float64] | None:
    """Normalise ``value``, returning None for a (near) zero vector."""
    norm = float(np.linalg.norm(value))
    if norm <= 1e-15:
        return None
    return value / norm


def _outward_tangent(
    port_unit: npt.NDArray[np.float64], centre_unit: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64] | None:
    """Unit geodesic direction at a port pointing away from the ring centre.

    The separatrix runs centre -> port -> outward, so this is the direction a
    gradient path launched at the port should travel. It is tangent to the sphere
    at the port by construction: the component along the port's own radial is
    removed.
    """
    radial = float(np.dot(port_unit, centre_unit))
    # Negating the centre-ward projection leaves the component of the port's own
    # direction that is tangent to the sphere, which points away from the centre:
    # its dot with the centre is radial^2 - 1 = -sin^2(a) < 0, so travelling along
    # it increases angular distance from the centre.
    tangent = radial * port_unit - centre_unit
    return _unit(tangent)


def _crossing_azimuths(
    profile: npt.NDArray[np.float64],
    phis: npt.NDArray[np.float64],
    level: float,
) -> list[float]:
    """Azimuths where the loop profile crosses ``level``, by chord interpolation.

    These are the separatrix tangents themselves: the boundaries between
    ascending and descending sectors, and the direct analogue of the sign
    crossings the cyclic-link scan computes.
    """
    delta = profile - level
    signs = np.where(np.abs(delta) < 1e-15, 1.0, np.sign(delta))
    n = len(profile)
    out: list[float] = []
    for i in range(n):
        j = (i + 1) % n
        if signs[i] == signs[j]:
            continue
        di, dj = float(delta[i]), float(delta[j])
        denom = dj - di
        t = (-di / denom) if abs(denom) > 1e-300 else 0.5
        t = min(max(t, 0.0), 1.0)
        phi_i, phi_j = float(phis[i]), float(phis[j])
        if j == 0:
            # Wrap the last cell onto [phi_i, phi_i + 2*pi) before interpolating.
            phi_j += 2.0 * math.pi
        out.append((phi_i + t * (phi_j - phi_i)) % (2.0 * math.pi))
    out.sort()
    return out


def _sector_bounds(
    phi: float,
    crossings: list[float],
) -> tuple[float, float]:
    """Bounding crossing azimuths (counter-clockwise) around a port azimuth."""
    if not crossings:
        return phi, phi
    below = [c for c in crossings if c <= phi]
    above = [c for c in crossings if c > phi]
    lo = below[-1] if below else crossings[-1] - 2.0 * math.pi
    hi = above[0] if above else crossings[0] + 2.0 * math.pi
    return lo, hi


def _mean_profile_in(
    profile: npt.NDArray[np.float64],
    phis: npt.NDArray[np.float64],
    lo: float,
    hi: float,
) -> float:
    """Mean profile value over the azimuth window ``[lo, hi]`` (periodic)."""
    span = hi - lo
    if span <= 0.0:
        return float(profile[int(np.argmin(np.abs(phis - lo)))])
    rel = (phis - lo) % (2.0 * math.pi)
    mask = rel <= span
    if not mask.any():
        return float(profile[int(np.argmin(np.abs(phis - lo)))])
    return float(np.mean(profile[mask]))


def extract_excision_launch_seed(
    ring_points: list[BoundaryRingPoint],
    mesh_pts: npt.NDArray[np.float64],
    f_vals: npt.NDArray[np.float64],
    saddle_id: str,
    saddle_position: tuple[float, float, float],
    saddle_value: float,
    expected_k: int,
    vertex_id: int = -1,
    multiplicity: int = 0,
    find_k_nearest: KNearestFinder | None = None,
    smoothing_window: int = 3,
    min_prominence_frac: float = 0.05,
) -> SaddleLaunchSeed:
    """Extract the separatrix launch branches from a saddle's excision boundary.

    Args:
        ring_points: The closed excision loop enclosing the saddle, in the centred
            atom frame. Its radius is the excision radius; the caller owns choosing
            and expanding it (see ``excision_launch_seeds``).
        mesh_pts: (N, 3) vertex coordinates in the centred atom frame.
        f_vals: (N,) scalar field values.
        saddle_id: Identifier of the source saddle (e.g. ``"SAD_142"``).
        saddle_position: Saddle centre in the centred atom frame.
        saddle_value: Field value at the saddle, used as the crossing level.
        expected_k: Fold order the reduction stage reports for this saddle:
            ``multiplicity + 1`` for a raw saddle, ``1 - net_index`` for an
            effective one. Audited against the measured port count, never used to
            select ports.
        vertex_id: Mesh vertex index, or -1 for an effective CP with no vertex.
        multiplicity: Multiplicity reported by the Morse detector (0 for an
            effective CP, which carries a net index instead).
        find_k_nearest: Optional spatial index over ``mesh_pts``. Strongly
            recommended when profiling many rings: the profile is then sampled
            with the local IDW kernel, numerically identical to the global scan.
        smoothing_window: Moving-average window for the loop profile.
        min_prominence_frac: Port prominence threshold as a fraction of the loop
            profile span.

    Returns:
        A ``SaddleLaunchSeed`` with ascent (ridge) and descent (valley) branches.
        When the ring yields no usable signature the branch lists are empty and
        ``failure_reason`` names why, so the caller can expand the radius and
        retry, or report the saddle as unresolvable.
    """

    def failed(reason: str) -> SaddleLaunchSeed:
        return _failed(
            saddle_id, saddle_position, saddle_value, vertex_id, multiplicity, reason
        )

    n = len(ring_points)
    if n == 0:
        return failed("empty_ring")
    if n < 4:
        return failed("too_few_samples")

    result = profile_boundary_loop(
        ring_points=ring_points,
        mesh_pts=mesh_pts,
        f_vals=f_vals,
        smoothing_window=smoothing_window,
        min_prominence_frac=min_prominence_frac,
        refine_azimuth=True,
        find_k_nearest=find_k_nearest,
    )

    k_valleys = len(result.valley_ports)
    k_ridges = len(result.ridge_ports)

    if k_valleys == 0 and k_ridges == 0:
        return failed("no_ports")
    if k_valleys != k_ridges:
        return failed("unbalanced")
    if k_valleys < _MIN_SADDLE_K:
        # Balanced 1 + 1: a ring around an extremum, not a saddle.
        return failed("below_min_k")

    c_unit, t1, t2, ang_radius = ring_frame(ring_points)
    phis = np.array([rp.phi_rad for rp in ring_points], dtype=np.float64)
    profile = np.array(result.profile_values, dtype=np.float64)
    crossings = _crossing_azimuths(profile, phis, saddle_value)

    def _branches(ports: Any, kind: str) -> list[LaunchBranch]:
        out: list[LaunchBranch] = []
        for i, port in enumerate(sorted(ports, key=lambda p: p.phi_rad)):
            pos = np.array(port.position, dtype=np.float64)
            r = float(np.linalg.norm(pos))
            if r <= 1e-15:
                continue
            tangent = _outward_tangent(pos / r, c_unit)
            if tangent is None:
                continue
            lo, hi = _sector_bounds(port.phi_rad, crossings)
            out.append(
                LaunchBranch(
                    kind=kind,
                    sector_index=i,
                    tangent_direction=(
                        float(tangent[0]),
                        float(tangent[1]),
                        float(tangent[2]),
                    ),
                    # The port IS the launch point: it lies on the excision loop,
                    # outside the degenerate core by construction.
                    launch_position=(float(pos[0]), float(pos[1]), float(pos[2])),
                    sector_width_rad=hi - lo,
                    sector_centroid_value=_mean_profile_in(profile, phis, lo, hi),
                )
            )
        # Re-index after skipping any degenerate ports.
        return [
            LaunchBranch(
                kind=b.kind,
                sector_index=i,
                tangent_direction=b.tangent_direction,
                launch_position=b.launch_position,
                sector_width_rad=b.sector_width_rad,
                sector_centroid_value=b.sector_centroid_value,
            )
            for i, b in enumerate(out)
        ]

    ascent = _branches(result.ridge_ports, "ascent")
    descent = _branches(result.valley_ports, "descent")

    fold_order = len(ascent)
    return SaddleLaunchSeed(
        saddle_id=saddle_id,
        saddle_position=saddle_position,
        saddle_value=saddle_value,
        vertex_id=vertex_id,
        multiplicity=int(multiplicity),
        fold_order=fold_order,
        index_agrees=fold_order == int(expected_k),
        ascent_branches=ascent,
        descent_branches=descent,
        separatrix_directions=[
            _loop_direction(c_unit, t1, t2, ang_radius, phi) for phi in crossings
        ],
        ring_radius_rad=float(ang_radius),
        failure_reason=None,
    )


def _loop_direction(
    c_unit: npt.NDArray[np.float64],
    t1: npt.NDArray[np.float64],
    t2: npt.NDArray[np.float64],
    ang_radius: float,
    phi: float,
) -> tuple[float, float, float]:
    """Unit radial direction of the loop point at azimuth ``phi``.

    The link-scan path reports separatrix directions as unit tangents at the
    saddle, because its crossings live in the tangent plane. Here the crossings
    live on the loop, so the reported direction is the unit position of the
    crossing point -- the direction in which it sits from the nucleus, which is
    what a downstream integrator needs to seed a path there.
    """
    dir_vec = math.cos(ang_radius) * c_unit + math.sin(ang_radius) * (
        math.cos(phi) * t1 + math.sin(phi) * t2
    )
    u = _unit(dir_vec)
    if u is None:
        return (float(c_unit[0]), float(c_unit[1]), float(c_unit[2]))
    return (float(u[0]), float(u[1]), float(u[2]))


def _failed(
    saddle_id: str,
    saddle_position: tuple[float, float, float],
    saddle_value: float,
    vertex_id: int,
    multiplicity: int,
    reason: str,
) -> SaddleLaunchSeed:
    """An empty seed carrying the reason the ring produced no branches."""
    return SaddleLaunchSeed(
        saddle_id=saddle_id,
        saddle_position=saddle_position,
        saddle_value=saddle_value,
        vertex_id=vertex_id,
        multiplicity=int(multiplicity),
        fold_order=0,
        index_agrees=False,
        ring_radius_rad=0.0,
        failure_reason=reason,
    )
