"""Excision Boundary Subroutine for gba_topology2.

Profiles the scalar field f(phi) along closed 1D excision boundary loops on a triangulated
sphere, extracts boundary ports (valley and ridge ports), and independently verifies
the topological index formula:
    chi = 1 - k
where k is the number of valley ports (and ridge ports).

Subroutine contract per AGENTS.md:
- Pure algorithmic computation (NumPy, standard library).
- Standalone testable via simple unit tests.
- Interpolates field values along 1D boundary points using inverse distance weighting.
- Resolves:
    k = 2 -> 4 ports (2 valleys, 2 ridges), chi = -1 (simple saddle)
    k = 3 -> 6 ports (3 valleys, 3 ridges), chi = -2 (monkey saddle)
    k = 4 -> 8 ports (4 valleys, 4 ridges), chi = -3 (4-fold octupolar cross)
"""

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from gba_topology2.src.micro_cluster import BoundaryRingPoint, ring_frame

__all__ = [
    "BoundaryPort",
    "ExcisionBoundaryResult",
    "KNearestFinder",
    "extract_boundary_ports",
    "interpolate_scalar_at_points",
    "interpolate_scalar_at_points_local",
    "profile_boundary_loop",
]

# Returns the indices of the ``k`` mesh vertices nearest a query point. Injected
# rather than constructed here so this module keeps its pure NumPy/stdlib
# contract: production passes a vtkKdTreePointLocator wrapper from
# ``point_locator``, tests pass a brute-force lambda.
KNearestFinder = Callable[[Sequence[float], int], Sequence[int]]


@dataclass(frozen=True)
class BoundaryPort:
    """Represents a discrete port along an excision boundary loop."""

    port_id: str
    port_type: str  # "valley" (local 1D min on loop) or "ridge" (local 1D max on loop)
    phi_rad: float
    phi_deg: float
    position: tuple[float, float, float]
    scalar_value: float


@dataclass(frozen=True)
class ExcisionBoundaryResult:
    """Container for 1D excision boundary profiling and port analysis."""

    boundary_points: list[BoundaryRingPoint]
    profile_values: list[float]
    valley_ports: list[BoundaryPort]
    ridge_ports: list[BoundaryPort]
    fold_order: int  # k = number of valley ports
    implied_euler_index: int  # chi = 1 - k
    is_topologically_balanced: bool  # len(valley_ports) == len(ridge_ports)


def interpolate_scalar_at_points(
    query_pts: npt.NDArray[np.floating[Any]],
    mesh_pts: npt.NDArray[np.floating[Any]],
    f_vals: npt.NDArray[np.floating[Any]],
    k_neighbors: int = 4,
    power: float = 2.0,
) -> npt.NDArray[np.float64]:
    """Interpolate scalar field at query points on sphere using Inverse Distance Weighting (IDW).

    Args:
        query_pts: (M, 3) coordinates of sample points.
        mesh_pts: (N, 3) coordinates of mesh vertices.
        f_vals: (N,) scalar values at mesh vertices.
        k_neighbors: Number of nearest neighbors to average (default 4).
        power: IDW power parameter (default 2.0).

    Returns:
        (M,) interpolated scalar values.
    """
    q = np.asarray(query_pts, dtype=np.float64)
    m = np.asarray(mesh_pts, dtype=np.float64)
    f = np.asarray(f_vals, dtype=np.float64)

    # Normalize vectors to unit sphere for robust chord distance
    q_norm = np.linalg.norm(q, axis=1, keepdims=True)
    q_norm = np.where(q_norm < 1e-12, 1.0, q_norm)
    q_unit = q / q_norm

    m_norm = np.linalg.norm(m, axis=1, keepdims=True)
    m_norm = np.where(m_norm < 1e-12, 1.0, m_norm)
    m_unit = m / m_norm

    interp_vals = np.zeros(len(q), dtype=np.float64)

    # Batch compute dot products: (M, N)
    dots = np.dot(q_unit, m_unit.T)
    dots = np.clip(dots, -1.0, 1.0)
    # Angular distance: theta = arccos(dot)
    ang_dists = np.arccos(dots)

    for i in range(len(q)):
        row_dists = ang_dists[i]
        # Check for exact or near-exact match
        min_idx = int(np.argmin(row_dists))
        if row_dists[min_idx] < 1e-7:
            interp_vals[i] = f[min_idx]
            continue

        # Find k nearest neighbors. k is clamped to the mesh size so a request
        # for more neighbours than exist uses every vertex, matching the
        # spatially-indexed variant (argpartition rejects kth >= N outright).
        kk = min(k_neighbors, len(row_dists))
        k_idx = (
            np.argpartition(row_dists, kk - 1)[:kk]
            if kk < len(row_dists)
            else np.arange(len(row_dists))
        )
        d_k = row_dists[k_idx]
        weights = 1.0 / (d_k**power)
        sum_w = float(np.sum(weights))
        interp_vals[i] = float(np.sum(weights * f[k_idx]) / sum_w)

    return interp_vals


def interpolate_scalar_at_points_local(
    query_pts: npt.NDArray[np.floating[Any]],
    mesh_pts: npt.NDArray[np.floating[Any]],
    f_vals: npt.NDArray[np.floating[Any]],
    find_k_nearest: KNearestFinder,
    k_neighbors: int = 4,
    power: float = 2.0,
) -> npt.NDArray[np.float64]:
    """Interpolate at query points by IDW over a spatially-indexed neighbour set.

    Numerically identical to :func:`interpolate_scalar_at_points` -- the same
    angular kernel, the same ``k_neighbors``/``power``, and the same exact-vertex
    snap -- but it evaluates the kernel on the ``k`` candidates a spatial index
    returns per query instead of on all N mesh vertices. That removes the
    ``(M, N)`` ``arccos`` matrix, which costs ~1e6 transcendental evaluations per
    excision ring at 20k mesh nodes.

    Asking the index for exactly ``k`` nearest is sound here because the mesh
    vertices and the ring samples both lie on the sphere of radius R: chord
    distance is ``2 R sin(theta / 2)``, a strictly monotone function of the
    angular distance the kernel uses, so the k nearest by chord are the k nearest
    by angle. No search radius, widening, or global fallback is required.

    Args:
        query_pts: (M, 3) coordinates of sample points.
        mesh_pts: (N, 3) coordinates of mesh vertices.
        f_vals: (N,) scalar values at mesh vertices.
        find_k_nearest: Callable returning the indices of the ``k`` mesh vertices
            nearest to a query point, e.g. the wrapper from
            :func:`point_locator.make_kdtree_nearest_finder`.
        k_neighbors: Number of nearest neighbours to average (default 4).
        power: IDW power parameter (default 2.0).

    Returns:
        (M,) interpolated scalar values.

    Raises:
        ValueError: If ``k_neighbors`` is not positive.
    """
    if k_neighbors <= 0:
        raise ValueError(f"k_neighbors must be positive, got {k_neighbors}")

    q = np.atleast_2d(np.asarray(query_pts, dtype=np.float64))
    m = np.asarray(mesh_pts, dtype=np.float64)
    f = np.asarray(f_vals, dtype=np.float64)

    out = np.zeros(len(q), dtype=np.float64)
    if len(q) == 0:
        return out

    # Unit directions: the kernel is angular, matching the global implementation.
    m_norm = np.linalg.norm(m, axis=1, keepdims=True)
    m_norm = np.where(m_norm < 1e-12, 1.0, m_norm)
    m_unit = m / m_norm

    q_norm = np.linalg.norm(q, axis=1, keepdims=True)
    q_unit = np.where(q_norm < 1e-12, 1.0, q_norm)
    q_unit = q / q_unit

    # One index query per point (the locator is per-query), then the kernel is
    # evaluated on the whole (M, k) candidate block at once.
    cand = np.zeros((len(q), k_neighbors), dtype=np.int64)
    widths = np.full(len(q), k_neighbors, dtype=np.int64)
    for i in range(len(q)):
        found = list(find_k_nearest(q[i].tolist(), k_neighbors))
        widths[i] = min(len(found), k_neighbors)
        for j, v in enumerate(found[:k_neighbors]):
            cand[i, j] = int(v)

    # Distances for the padded tail are forced to +inf so they drop out of the
    # weighting, and a row with no candidates keeps its zero output.
    cols = np.arange(k_neighbors)[None, :]
    valid = cols < widths[:, None]

    dots = np.einsum("ij,ikj->ik", q_unit, m_unit[cand])
    d = np.where(valid, np.arccos(np.clip(dots, -1.0, 1.0)), np.inf)

    # Exact-vertex snap, per the reference kernel's 1e-7 angular tolerance.
    nearest_col = np.argmin(d, axis=1)
    nearest_d = d[np.arange(len(q)), nearest_col]
    snapped = valid.any(axis=1) & (nearest_d < 1e-7)

    with np.errstate(divide="ignore", invalid="ignore"):
        w = 1.0 / d**power
        w = np.where(valid, w, 0.0)
        sum_w = w.sum(axis=1)
        # A zero distance (exact vertex hit) makes this nan; those rows are
        # replaced by the snap value below.
        weighted = (w * f[cand]).sum(axis=1) / np.where(sum_w > 0, sum_w, 1.0)
    weighted = np.where(np.isfinite(weighted), weighted, 0.0)

    out[:] = np.where(snapped, f[cand[np.arange(len(q)), nearest_col]], weighted)
    out[~valid.any(axis=1)] = 0.0
    return out


def _refine_extremum(
    smooth_vals: npt.NDArray[np.float64],
    idx: int,
    n: int,
    dphi_step: float,
) -> float:
    """Locate a 1D extremum between samples by fitting a parabola to three points.

    Ring samples sit at a fixed angular pitch, so a port snapped to a sample index
    carries a quantization error of up to half that pitch -- 3.75 deg at 48 samples,
    which is coarser than the 1.6 deg mesh pitch the port is meant to resolve. A
    parabola through the extremum and its two periodic neighbours recovers the
    vertex to sub-sample precision for any smooth profile.

    Args:
        smooth_vals: Smoothed periodic profile.
        idx: Index of the accepted discrete extremum.
        n: Profile length, for periodic wrap.
        dphi_step: Angular spacing between samples, in radians.

    Returns:
        Azimuth offset from ``idx``, in radians, signed along increasing ``phi``.
        Clamped to one sample step, so a degenerate stencil can never move a port
        outside the cells that bracket it; zero when the stencil is flat.
    """
    y_m = float(smooth_vals[(idx - 1) % n])
    y_0 = float(smooth_vals[idx])
    y_p = float(smooth_vals[(idx + 1) % n])

    den = y_m - 2.0 * y_0 + y_p
    if abs(den) < 1e-300:
        # Flat three-point stencil: the parabola is degenerate, keep the sample.
        return 0.0

    dphi = 0.5 * dphi_step * (y_m - y_p) / den
    if not math.isfinite(dphi):
        return 0.0
    return max(-dphi_step, min(dphi_step, dphi))


def extract_boundary_ports(
    profile_values: list[float],
    ring_points: list[BoundaryRingPoint],
    smoothing_window: int = 3,
    min_prominence_frac: float = 0.05,
    refine_azimuth: bool = False,
) -> tuple[list[BoundaryPort], list[BoundaryPort]]:
    """Detect alternating local minima (valleys) and maxima (ridges) on a closed periodic 1D profile.

    Args:
        profile_values: Sampled scalar values around the 1D closed loop.
        ring_points: Corresponding BoundaryRingPoint objects.
        smoothing_window: Moving average window for noise reduction (odd int, default 3).
        min_prominence_frac: Minimum peak/valley prominence fraction of profile span (default 0.05).
        refine_azimuth: When True, locate each port between samples by parabolic
            interpolation and place it back on the loop at that sub-sample azimuth
            (default False, which snaps ports to sample indices).

    Returns:
        Tuple of (valley_ports, ridge_ports).
    """
    n = len(profile_values)
    if n < 4:
        return [], []

    vals = np.array(profile_values, dtype=np.float64)
    v_span = float(np.ptp(vals))
    min_prominence = max(v_span * min_prominence_frac, 1e-12)

    # Sub-sample placement needs the loop's frame, recovered once and reused.
    dphi_step = 2.0 * math.pi / float(n)
    frame: tuple[np.ndarray, np.ndarray, np.ndarray, float, float] | None = None
    if refine_azimuth:
        c_unit, t1, t2, ang_radius = ring_frame(ring_points)
        r_ring = float(np.linalg.norm(ring_points[0].position))
        frame = (c_unit, t1, t2, ang_radius, r_ring)

    def _placed(idx: int, dphi: float) -> tuple[float, tuple[float, float, float]]:
        """Return the refined azimuth and its on-loop position for sample ``idx``."""
        phi = ring_points[idx].phi_rad + dphi
        if frame is None:
            return phi, ring_points[idx].position
        c_unit, t1, t2, ang_radius, r_ring = frame
        cos_a, sin_a = math.cos(ang_radius), math.sin(ang_radius)
        vec = cos_a * c_unit + sin_a * (math.cos(phi) * t1 + math.sin(phi) * t2)
        norm = float(np.linalg.norm(vec))
        if norm > 1e-12:
            vec = vec / norm
        return phi, (
            float(vec[0] * r_ring),
            float(vec[1] * r_ring),
            float(vec[2] * r_ring),
        )

    # Periodic moving-average smoothing if requested
    if 1 < smoothing_window < n:
        half_w = smoothing_window // 2
        padded = np.concatenate([vals[-half_w:], vals, vals[:half_w]])
        kernel = np.ones(smoothing_window) / float(smoothing_window)
        smooth_vals = np.convolve(padded, kernel, mode="valid")
    else:
        smooth_vals = vals

    # Step 1: Find raw discrete local extrema on the 1D ring
    raw_min_indices: list[int] = []
    raw_max_indices: list[int] = []

    for i in range(n):
        prev_idx = (i - 1) % n
        next_idx = (i + 1) % n
        curr_v = float(smooth_vals[i])
        prev_v = float(smooth_vals[prev_idx])
        next_v = float(smooth_vals[next_idx])

        if curr_v < prev_v and curr_v <= next_v:
            raw_min_indices.append(i)
        elif curr_v > prev_v and curr_v >= next_v:
            raw_max_indices.append(i)

    # Step 2: Filter by true topological prominence (peak-to-valley amplitude)
    def _port(idx: int, kind: str, seq: int) -> BoundaryPort:
        """Build one port, refined to a sub-sample azimuth when requested.

        The prominence test stays on the discrete profile: refinement moves an
        accepted port between its bracketing samples, it must never change which
        ports are accepted.
        """
        dphi = 0.0
        if frame is not None:
            dphi = _refine_extremum(smooth_vals, idx, n, dphi_step)
        phi, pos = _placed(idx, dphi)
        return BoundaryPort(
            port_id=f"{'VPORT' if kind == 'valley' else 'RPORT'}_{seq}",
            port_type=kind,
            phi_rad=phi,
            phi_deg=float(math.degrees(phi)),
            position=pos,
            # The sampled value at the bracketing index: this routine sees only
            # the profile, so a refined azimuth does not imply a re-interpolated
            # scalar. Callers that need it re-interpolate.
            scalar_value=float(vals[idx]),
        )

    # For each raw minimum, compute height difference relative to adjacent raw maxima
    valleys: list[BoundaryPort] = []
    for idx in raw_min_indices:
        v_curr = float(smooth_vals[idx])
        if raw_max_indices:
            max_vals = [float(smooth_vals[m]) for m in raw_max_indices]
            peak_height = max(max_vals) - v_curr
            if peak_height >= min_prominence or v_span < 1e-12:
                valleys.append(_port(idx, "valley", len(valleys) + 1))
        else:
            valleys.append(_port(idx, "valley", len(valleys) + 1))

    # For each raw maximum, compute height above valleys
    ridges: list[BoundaryPort] = []
    for idx in raw_max_indices:
        v_curr = float(smooth_vals[idx])
        if raw_min_indices:
            min_vals = [float(smooth_vals[m]) for m in raw_min_indices]
            valley_depth = v_curr - min(min_vals)
            if valley_depth >= min_prominence or v_span < 1e-12:
                ridges.append(_port(idx, "ridge", len(ridges) + 1))
        else:
            ridges.append(_port(idx, "ridge", len(ridges) + 1))

    return valleys, ridges


def profile_boundary_loop(
    ring_points: list[BoundaryRingPoint],
    mesh_pts: npt.NDArray[np.floating[Any]],
    f_vals: npt.NDArray[np.floating[Any]],
    smoothing_window: int = 3,
    min_prominence_frac: float = 0.05,
    refine_azimuth: bool = False,
    find_k_nearest: KNearestFinder | None = None,
) -> ExcisionBoundaryResult:
    """Profile a 1D excision boundary loop, extract ports, and compute implied Euler index.

    Args:
        ring_points: List of BoundaryRingPoint sample points forming the closed loop.
        mesh_pts: (N, 3) sphere mesh vertices.
        f_vals: (N,) scalar field values.
        smoothing_window: Moving average smoothing window (default 3).
        min_prominence_frac: Prominence threshold fraction (default 0.05).
        refine_azimuth: When True, ports are located between ring samples by
            parabolic interpolation (default False, which snaps them to samples).
        find_k_nearest: Optional spatial index over ``mesh_pts`` (see
            :func:`point_locator.make_kdtree_nearest_finder`). When given, the
            profile is sampled with :func:`interpolate_scalar_at_points_local`,
            which is numerically identical to the global scan and far cheaper when
            many rings are profiled per frame.

    Returns:
        ExcisionBoundaryResult with profile values, valley ports, ridge ports,
        fold order k, and implied index chi = 1 - k.
    """
    if not ring_points:
        return ExcisionBoundaryResult(
            boundary_points=[],
            profile_values=[],
            valley_ports=[],
            ridge_ports=[],
            fold_order=0,
            implied_euler_index=1,
            is_topologically_balanced=True,
        )

    pts_arr = np.array([rp.position for rp in ring_points], dtype=np.float64)
    if find_k_nearest is not None:
        interp_vals = interpolate_scalar_at_points_local(
            query_pts=pts_arr,
            mesh_pts=mesh_pts,
            f_vals=f_vals,
            find_k_nearest=find_k_nearest,
        )
    else:
        interp_vals = interpolate_scalar_at_points(
            query_pts=pts_arr,
            mesh_pts=mesh_pts,
            f_vals=f_vals,
        )
    profile_list = [float(v) for v in interp_vals]

    valleys, ridges = extract_boundary_ports(
        profile_values=profile_list,
        ring_points=ring_points,
        smoothing_window=smoothing_window,
        min_prominence_frac=min_prominence_frac,
        refine_azimuth=refine_azimuth,
    )

    k_valleys = len(valleys)
    k_ridges = len(ridges)
    is_balanced = (k_valleys == k_ridges) and (k_valleys > 0)

    # Fold order is number of valley ports (or ridge ports)
    fold_order = k_valleys if is_balanced else max(k_valleys, k_ridges)
    implied_chi = 1 - fold_order if fold_order > 0 else 1

    return ExcisionBoundaryResult(
        boundary_points=ring_points,
        profile_values=profile_list,
        valley_ports=valleys,
        ridge_ports=ridges,
        fold_order=fold_order,
        implied_euler_index=implied_chi,
        is_topologically_balanced=is_balanced,
    )
