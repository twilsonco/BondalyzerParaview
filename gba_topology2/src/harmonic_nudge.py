"""Harmonic Nudge Subroutine for gba_topology2.

Refines and centers effective critical points (ECPs) and boundary ports using
symmetry and eccentricity energy minimization on the sphere:
- Extrema (k = 0): Minimizes circular level-curve variance / gradient asymmetry.
- Simple Saddles (k = 2): Minimizes 2-fold orthogonal (90 deg) cross energy.
- Monkey Saddles (k = 3): Minimizes 3-fold C_3v (60 deg alternating) angular energy.
- Octupolar Saddles (k = 4): Minimizes 4-fold C_4v (45 deg alternating) angular energy.

Prepares perfected, centered launch poles for gradient basin tracing.

Subroutine contract per AGENTS.md:
- Pure algorithmic computation (NumPy, standard library).
- Standalone testable via simple unit tests.
- Scaled relative to delta_theta_mesh_rad.
"""

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from gba_topology2.src.excision_boundary import (
    BoundaryPort,
    interpolate_scalar_at_points,
)
from gba_topology2.src.micro_cluster import BoundaryRingPoint, ring_frame

__all__ = [
    "NudgeResult",
    "compute_k_fold_symmetry_energy",
    "harmonic_nudge_critical_point",
    "perfect_boundary_ports",
    "reposition_ports_on_ring",
]


@dataclass(frozen=True)
class NudgeResult:
    """Container for harmonic nudge optimization results."""

    initial_position: tuple[float, float, float]
    optimized_position: tuple[float, float, float]
    displacement_ang_rad: float
    displacement_ang_deg: float
    initial_energy: float
    final_energy: float
    fold_order: int
    converged: bool
    iterations: int


def compute_k_fold_symmetry_energy(
    center_pos: tuple[float, float, float],
    mesh_pts: npt.NDArray[np.floating[Any]],
    f_vals: npt.NDArray[np.floating[Any]],
    fold_order: int,
    probe_radius_rad: float,
    num_samples: int = 72,
    sphere_radius: float = 1.0,
) -> float:
    """Compute the symmetry energy of a candidate center point on the sphere.

    For fold_order == 0 (Extrema):
        Measures the variance of the scalar field along a small circular level ring around center.
        Lower variance => more circular/symmetric extremum.

    For fold_order in {2, 3, 4} (Saddles):
        Samples field f(phi) around candidate center, computes Fourier component
        at harmonic 2k (e.g. 4th harmonic for k=2, 6th harmonic for k=3, 8th for k=4),
        and returns the energy penalty (asymmetry / phase distortion).

    Args:
        center_pos: (3,) candidate coordinates on sphere.
        mesh_pts: (N, 3) sphere mesh coordinates.
        f_vals: (N,) scalar field values.
        fold_order: Symmetry order k (0 for extremum, 2 for simple, 3 for monkey, 4 for octupole).
        probe_radius_rad: Angular radius of probe ring in radians.
        num_samples: Number of circumferential sample points (default 72).
        sphere_radius: Radius R of the sphere.

    Returns:
        Scalar symmetry energy (lower is more symmetric/perfected).
    """
    c = np.array(center_pos, dtype=np.float64)
    norm_c = float(np.linalg.norm(c))
    c_unit = (
        c / norm_c if norm_c > 1e-12 else np.array([0.0, 0.0, 1.0], dtype=np.float64)
    )

    # Tangent orthonormal basis t1, t2
    if abs(float(c_unit[2])) < 0.9:
        v_temp = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    else:
        v_temp = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    t1 = np.cross(c_unit, v_temp)
    t1 /= np.linalg.norm(t1)
    t2 = np.cross(c_unit, t1)
    t2 /= np.linalg.norm(t2)

    # Sample circular probe ring
    cos_r = math.cos(probe_radius_rad)
    sin_r = math.sin(probe_radius_rad)
    ring_pts = np.zeros((num_samples, 3), dtype=np.float64)
    phis = np.linspace(0.0, 2.0 * math.pi, num_samples, endpoint=False)

    for i, phi in enumerate(phis):
        dir_vec = cos_r * c_unit + sin_r * (math.cos(phi) * t1 + math.sin(phi) * t2)
        dir_vec /= np.linalg.norm(dir_vec)
        ring_pts[i] = dir_vec * sphere_radius

    ring_f = interpolate_scalar_at_points(ring_pts, mesh_pts, f_vals)

    if fold_order == 0:
        # Extrema: circular variance minimization
        return float(np.var(ring_f))

    # Saddles with k-fold symmetry:
    # A pure k-fold saddle has periodic profile f(phi) ~ cos(k*phi + phi_0).
    # Fourier decomposition: energy is penalty of non-k harmonics.
    fft_coeffs = np.fft.rfft(ring_f)
    powers = np.abs(fft_coeffs) ** 2

    # Target harmonic is k (e.g. harmonic 2 for k=2, harmonic 3 for k=3)
    target_k = fold_order
    if target_k < len(powers):
        target_power = powers[target_k]
        total_ac_power = float(np.sum(powers[1:]))
        if total_ac_power > 1e-12:
            # Fraction of power NOT in target harmonic (lower is more purely k-fold)
            energy = 1.0 - (float(target_power) / total_ac_power)
        else:
            energy = 0.0
    else:
        energy = float(np.var(ring_f))

    return max(0.0, energy)


def harmonic_nudge_critical_point(
    initial_pos: tuple[float, float, float],
    mesh_pts: npt.NDArray[np.floating[Any]],
    f_vals: npt.NDArray[np.floating[Any]],
    fold_order: int,
    delta_theta_mesh_rad: float,
    sphere_radius: float = 1.0,
    max_displacement_pitch: float = 1.5,
    max_iterations: int = 15,
    step_size_pitch: float = 0.25,
    probe_radius_pitch: float | None = None,
) -> NudgeResult:
    """Nudge and center an effective critical point by minimizing symmetry energy on sphere.

    Uses spherical gradient-free pattern search (stencil on the tangent plane).

    Args:
        initial_pos: (3,) initial coordinates of critical point.
        mesh_pts: (N, 3) sphere mesh coordinates.
        f_vals: (N,) scalar field values.
        fold_order: Symmetry fold order k (0 for extremum, 2 for simple, 3 for monkey, 4 for octupole).
        delta_theta_mesh_rad: Mesh pitch delta_theta in radians.
        sphere_radius: Sphere radius R.
        max_displacement_pitch: Maximum allowable displacement in pitch units (default 1.5).
        max_iterations: Maximum optimization iterations (default 15).
        step_size_pitch: Initial search step size in pitch units (default 0.25).
        probe_radius_pitch: Probe ring radius in pitch units. When ``None`` (default),
            ``1.5`` pitch is used, which suits single-CP entities. Constellations
            spanning several pitches should pass a radius clearing their footprint.

    Returns:
        NudgeResult dataclass with optimized position, displacement, and energies.
    """
    p_init = np.array(initial_pos, dtype=np.float64)
    norm_p = float(np.linalg.norm(p_init))
    p_init = (p_init / norm_p) * sphere_radius if norm_p > 1e-12 else p_init

    probe_pitch = 1.5 if probe_radius_pitch is None else float(probe_radius_pitch)
    probe_r = max(probe_pitch, 0.5) * delta_theta_mesh_rad
    max_disp_rad = max_displacement_pitch * delta_theta_mesh_rad
    step_rad = step_size_pitch * delta_theta_mesh_rad

    curr_p = np.copy(p_init)
    init_energy = compute_k_fold_symmetry_energy(
        center_pos=(float(curr_p[0]), float(curr_p[1]), float(curr_p[2])),
        mesh_pts=mesh_pts,
        f_vals=f_vals,
        fold_order=fold_order,
        probe_radius_rad=probe_r,
        sphere_radius=sphere_radius,
    )
    curr_energy = init_energy

    # Compass directions in tangent plane
    angles = [0.0, math.pi / 2.0, math.pi, 3.0 * math.pi / 2.0]
    converged = False
    it = 0

    for it in range(1, max_iterations + 1):
        c_u = curr_p / np.linalg.norm(curr_p)
        if abs(float(c_u[2])) < 0.9:
            v_t = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        else:
            v_t = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        t1 = np.cross(c_u, v_t)
        t1 /= np.linalg.norm(t1)
        t2 = np.cross(c_u, t1)
        t2 /= np.linalg.norm(t2)

        best_cand: np.ndarray | None = None
        best_cand_e = curr_energy

        for ang in angles:
            tan_step = math.cos(ang) * t1 + math.sin(ang) * t2
            cos_s = math.cos(step_rad)
            sin_s = math.sin(step_rad)
            cand_u = cos_s * c_u + sin_s * tan_step
            cand_u /= np.linalg.norm(cand_u)
            cand_p = cand_u * sphere_radius

            # Enforce max displacement clamp from initial position
            cos_disp = float(
                np.clip(np.dot(cand_u, p_init / np.linalg.norm(p_init)), -1.0, 1.0)
            )
            disp = math.acos(cos_disp)
            if disp > max_disp_rad:
                continue

            cand_e = compute_k_fold_symmetry_energy(
                center_pos=(float(cand_p[0]), float(cand_p[1]), float(cand_p[2])),
                mesh_pts=mesh_pts,
                f_vals=f_vals,
                fold_order=fold_order,
                probe_radius_rad=probe_r,
                sphere_radius=sphere_radius,
            )

            if cand_e < best_cand_e - 1e-6:
                best_cand_e = cand_e
                best_cand = cand_p

        if best_cand is not None:
            curr_p = best_cand
            curr_energy = best_cand_e
        else:
            # Reduce step size
            step_rad *= 0.5
            if step_rad < 0.05 * delta_theta_mesh_rad:
                converged = True
                break

    cos_final_disp = float(
        np.clip(
            np.dot(curr_p / np.linalg.norm(curr_p), p_init / np.linalg.norm(p_init)),
            -1.0,
            1.0,
        )
    )
    final_disp_rad = math.acos(cos_final_disp)

    return NudgeResult(
        initial_position=(float(p_init[0]), float(p_init[1]), float(p_init[2])),
        optimized_position=(
            float(curr_p[0]),
            float(curr_p[1]),
            float(curr_p[2]),
        ),
        displacement_ang_rad=final_disp_rad,
        displacement_ang_deg=float(math.degrees(final_disp_rad)),
        initial_energy=init_energy,
        final_energy=curr_energy,
        fold_order=fold_order,
        converged=converged or it >= max_iterations,
        iterations=it,
    )


def perfect_boundary_ports(
    ports: list[BoundaryPort],
    fold_order: int,
) -> list[BoundaryPort]:
    """Perfect the azimuthal angles of boundary ports toward ideal k-fold symmetry spacing.

    For k-fold symmetry, consecutive ports of the same type should ideally be spaced
    at delta_phi = 360 / k degrees apart.

    Args:
        ports: List of raw BoundaryPort objects of the same type (all valleys or all ridges).
        fold_order: Expected fold order k (e.g. 2, 3, or 4).

    Returns:
        List of BoundaryPort objects with perfected angles and updated positions.
    """
    if len(ports) != fold_order or fold_order < 2:
        return ports

    # Sort ports by azimuthal angle phi
    sorted_ports = sorted(ports, key=lambda p: p.phi_rad)
    ideal_step_rad = 2.0 * math.pi / float(fold_order)

    # Determine optimal global phase offset phi_0 minimizing least-squares angle deviation
    # phi_i = phi_0 + i * ideal_step
    phi_0_candidates = [
        (sorted_ports[i].phi_rad - i * ideal_step_rad) % (2.0 * math.pi)
        for i in range(fold_order)
    ]
    # Circular mean of candidate phase offsets
    sin_sum = sum(math.sin(p0) for p0 in phi_0_candidates)
    cos_sum = sum(math.cos(p0) for p0 in phi_0_candidates)
    mean_phi_0 = math.atan2(sin_sum, cos_sum) % (2.0 * math.pi)

    perfected: list[BoundaryPort] = []
    for i, p in enumerate(sorted_ports):
        perf_phi_rad = (mean_phi_0 + i * ideal_step_rad) % (2.0 * math.pi)
        perfected.append(
            BoundaryPort(
                port_id=f"{p.port_id}_PERF",
                port_type=p.port_type,
                phi_rad=perf_phi_rad,
                phi_deg=float(math.degrees(perf_phi_rad)),
                position=p.position,
                scalar_value=p.scalar_value,
            )
        )

    return perfected


def reposition_ports_on_ring(
    ports: list[BoundaryPort],
    ring_points: list[BoundaryRingPoint],
    sphere_radius: float = 1.0,
    f_vals: npt.NDArray[np.floating[Any]] | None = None,
    mesh_pts: npt.NDArray[np.floating[Any]] | None = None,
) -> list[BoundaryPort]:
    """Map ports carrying (possibly perfected) azimuthal angles onto 3D ring coordinates.

    ``perfect_boundary_ports`` only rewrites the azimuthal angle ``phi_rad``; this
    subroutine projects those angles back onto the actual excision loop so the
    perfected spacing is visible in 3D. Scalar values are re-interpolated from the
    mesh when ``f_vals`` and ``mesh_pts`` are supplied.

    Args:
        ports: Ports of a single type (all valleys or all ridges) for one loop.
        ring_points: The boundary ring samples that define the loop frame.
        sphere_radius: Radius R of the sphere.
        f_vals: Optional (N,) mesh scalar values for re-interpolating port values.
        mesh_pts: Optional (N, 3) mesh coordinates used with ``f_vals``.

    Returns:
        New BoundaryPort objects whose positions lie on the ring at ``phi_rad``.
    """
    if not ports:
        return ports

    c_unit, t1, t2, ang_radius = ring_frame(ring_points)
    cos_a = math.cos(ang_radius)
    sin_a = math.sin(ang_radius)

    repositioned: list[BoundaryPort] = []
    new_positions: list[tuple[float, float, float]] = []
    for p in ports:
        dir_vec = cos_a * c_unit + sin_a * (
            math.cos(p.phi_rad) * t1 + math.sin(p.phi_rad) * t2
        )
        dir_vec /= float(np.linalg.norm(dir_vec))
        pos = (dir_vec * sphere_radius).tolist()
        new_positions.append((float(pos[0]), float(pos[1]), float(pos[2])))

    new_values: list[float] = [p.scalar_value for p in ports]
    if f_vals is not None and mesh_pts is not None:
        q = np.array(new_positions, dtype=np.float64)
        vals = interpolate_scalar_at_points(q, mesh_pts, f_vals)
        new_values = [float(v) for v in vals]

    for p, pos, val in zip(ports, new_positions, new_values):
        repositioned.append(
            BoundaryPort(
                port_id=p.port_id,
                port_type=p.port_type,
                phi_rad=p.phi_rad,
                phi_deg=p.phi_deg,
                position=pos,
                scalar_value=val,
            )
        )
    return repositioned
