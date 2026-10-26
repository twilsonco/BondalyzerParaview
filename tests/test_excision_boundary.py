"""Unit tests for excision_boundary module."""

import math

import numpy as np
import pytest

from gba_topology2.src.excision_boundary import (
    BoundaryPort,
    _refine_extremum,
    extract_boundary_ports,
    interpolate_scalar_at_points,
    profile_boundary_loop,
)
from gba_topology2.src.micro_cluster import (
    BoundaryRingPoint,
    generate_boundary_ring,
    ring_frame,
)


def _generate_fibonacci_sphere(
    num_points: int = 1200, radius: float = 1.0
) -> np.ndarray:
    """Generate uniform points on sphere."""
    indices = np.arange(0, num_points, dtype=float) + 0.5
    phi = np.arccos(1 - 2 * indices / num_points)
    theta = math.pi * (1 + 5**0.5) * indices
    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    z = radius * np.cos(phi)
    return np.column_stack([x, y, z])


def test_2fold_simple_saddle_ports() -> None:
    """Test 2-fold simple saddle field f = x^2 - y^2: must yield 2 valleys, 2 ridges, chi = -1."""
    ring = generate_boundary_ring(
        centroid=(0.0, 0.0, 1.0),
        sphere_radius=1.0,
        angular_radius_rad=math.radians(15.0),
        num_samples=72,
    )

    # Around north pole, f(phi) ~ cos(2*phi): has 2 maxima (phi = 0, pi) and 2 minima (phi = pi/2, 3pi/2)
    # Generate profile directly:
    vals = [math.cos(2.0 * rp.phi_rad) for rp in ring]

    valleys, ridges = extract_boundary_ports(vals, ring)
    assert len(valleys) == 2
    assert len(ridges) == 2

    # Check angular spacing: valleys should be spaced ~180 deg apart
    val_diff = abs(valleys[1].phi_deg - valleys[0].phi_deg)
    assert pytest.approx(val_diff, abs=10.0) == 180.0

    # Implied index: chi = 1 - 2 = -1
    chi = 1 - len(valleys)
    assert chi == -1


def test_3fold_monkey_saddle_ports() -> None:
    """Test 3-fold monkey saddle field f ~ cos(3*phi): must yield 3 valleys, 3 ridges, chi = -2."""
    ring = generate_boundary_ring(
        centroid=(0.0, 0.0, 1.0),
        sphere_radius=1.0,
        angular_radius_rad=math.radians(20.0),
        num_samples=72,
    )

    # 3-fold profile
    vals = [math.cos(3.0 * rp.phi_rad) for rp in ring]

    valleys, ridges = extract_boundary_ports(vals, ring)
    assert len(valleys) == 3
    assert len(ridges) == 3

    chi = 1 - len(valleys)
    assert chi == -2


def test_4fold_octupolar_cross_ports() -> None:
    """Test 4-fold octupolar cross field f ~ cos(4*phi): must yield 4 valleys, 4 ridges, chi = -3."""
    ring = generate_boundary_ring(
        centroid=(0.0, 0.0, 1.0),
        sphere_radius=1.0,
        angular_radius_rad=math.radians(25.0),
        num_samples=96,
    )

    # 4-fold profile
    vals = [math.cos(4.0 * rp.phi_rad) for rp in ring]

    valleys, ridges = extract_boundary_ports(vals, ring)
    assert len(valleys) == 4
    assert len(ridges) == 4

    chi = 1 - len(valleys)
    assert chi == -3


def test_profile_boundary_loop_with_mesh_interpolation() -> None:
    """Test end-to-end boundary loop profiling with IDW mesh interpolation."""
    pts = _generate_fibonacci_sphere(num_points=1500, radius=1.0)
    # Quadrupole field f(x, y, z) = x^2 - y^2
    f_vals = pts[:, 0] ** 2 - pts[:, 1] ** 2

    # Ring around North pole (0, 0, 1)
    ring = generate_boundary_ring(
        centroid=(0.0, 0.0, 1.0),
        sphere_radius=1.0,
        angular_radius_rad=math.radians(18.0),
        num_samples=72,
    )

    result = profile_boundary_loop(
        ring_points=ring,
        mesh_pts=pts,
        f_vals=f_vals,
    )

    assert result.is_topologically_balanced
    assert result.fold_order == 2
    assert result.implied_euler_index == -1
    assert len(result.valley_ports) == 2
    assert len(result.ridge_ports) == 2


def test_interpolate_scalar_exact_match() -> None:
    """Test IDW interpolation when query point exactly matches a mesh vertex."""
    pts = np.array(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64
    )
    f_vals = np.array([10.0, 20.0, 30.0], dtype=np.float64)

    res = interpolate_scalar_at_points(
        query_pts=pts,
        mesh_pts=pts,
        f_vals=f_vals,
        k_neighbors=2,
    )

    assert np.allclose(res, f_vals, atol=1e-5)


def test_empty_boundary_loop() -> None:
    """Test empty ring input handles gracefully."""
    res = profile_boundary_loop([], np.zeros((3, 3)), np.zeros(3))
    assert res.fold_order == 0
    assert res.implied_euler_index == 1
    assert res.valley_ports == []
    assert res.ridge_ports == []


# ---------------------------------------------------------------------------
# Sub-sample port azimuth refinement (refine_azimuth=True)
#
# Ring samples sit at a fixed pitch (7.5 deg at 48 samples), so a port snapped to
# a sample index carries up to half that pitch of quantization error -- coarser
# than the 1.6 deg mesh pitch it is meant to resolve. Refinement fits a parabola
# through the extremum and its two periodic neighbours.
# ---------------------------------------------------------------------------


def _ring(num_samples: int = 48, ang_deg: float = 15.0) -> list[BoundaryRingPoint]:
    """Circular ring about the north pole."""
    """Circular ring about the north pole."""
    return generate_boundary_ring(
        centroid=(0.0, 0.0, 1.0),
        sphere_radius=1.0,
        angular_radius_rad=math.radians(ang_deg),
        num_samples=num_samples,
    )


def test_refinement_recovers_off_sample_ridge_phase() -> None:
    """A cos(3*phi) field with maxima between samples is recovered to sub-degree."""
    n = 48
    ring = _ring(n)
    # True ridges (maxima) at phi = phase + 2*pi*m/3, phase chosen off-grid.
    phase = 0.3
    vals = [math.cos(3.0 * (rp.phi_rad - phase)) for rp in ring]

    _, ridges = extract_boundary_ports(vals, ring, refine_azimuth=True)
    assert len(ridges) == 3

    step = 2.0 * math.pi / 3.0
    for r in ridges:
        delta = (r.phi_rad - phase) % step
        delta = min(delta, step - delta)
        assert delta < math.radians(0.05), f"ridge off true phase by {delta:.5f} rad"

    # Quantization alone would allow up to half a sample step (3.75 deg at 48).
    assert 2.0 * math.pi / n > math.radians(7.0)


def test_refinement_beats_snapped_ports() -> None:
    """Refined ports are strictly closer to the true extrema than snapped ports."""
    n = 48
    ring = _ring(n)
    phase = 0.3
    vals = [math.cos(3.0 * (rp.phi_rad - phase)) for rp in ring]

    _, snapped = extract_boundary_ports(vals, ring, refine_azimuth=False)
    _, refined = extract_boundary_ports(vals, ring, refine_azimuth=True)

    def max_err(ports: list[BoundaryPort]) -> float:
        worst = 0.0
        step = 2.0 * math.pi / 3.0
        for r in ports:
            d = (r.phi_rad - phase) % step
            worst = max(worst, min(d, step - d))
        return worst

    assert max_err(refined) < max_err(snapped) / 100.0


def test_refinement_preserves_port_counts_and_types() -> None:
    """Refinement moves accepted ports; it must never change which ports exist."""
    ring = _ring(48)
    for k in (2, 3, 4):
        vals = [math.cos(k * (rp.phi_rad - 0.37)) for rp in ring]
        sv, sr = extract_boundary_ports(vals, ring, refine_azimuth=False)
        vv, vr = extract_boundary_ports(vals, ring, refine_azimuth=True)
        assert len(sv) == len(vv) == k
        assert len(sr) == len(vr) == k
        assert all(p.port_type == "valley" for p in vv)
        assert all(p.port_type == "ridge" for p in vr)
        # Same ordering, so index i of one list refines index i of the other.
        for a, b in zip(sv, vv):
            assert abs(a.phi_rad - b.phi_rad) <= 2.0 * math.pi / 48.0 + 1e-12


def test_refined_ports_lie_on_the_ring() -> None:
    """A refined position is on the small circle, not on the chord between samples."""
    ring = _ring(48, ang_deg=15.0)
    vals = [math.cos(3.0 * (rp.phi_rad - 0.3)) for rp in ring]
    valleys, ridges = extract_boundary_ports(vals, ring, refine_azimuth=True)

    ang_radius = math.radians(15.0)
    c = np.array([0.0, 0.0, 1.0])
    for p in [*valleys, *ridges]:
        v = np.array(p.position)
        assert np.linalg.norm(v) == pytest.approx(1.0, abs=1e-12)
        # Constant angular distance from the ring centre.
        cosang = float(np.clip(np.dot(v / np.linalg.norm(v), c), -1.0, 1.0))
        assert math.acos(cosang) == pytest.approx(ang_radius, abs=1e-9)


def test_refined_azimuth_matches_position() -> None:
    """The reported phi and the reported position describe the same point."""
    ring = _ring(48, ang_deg=20.0)
    vals = [math.cos(2.0 * (rp.phi_rad - 0.9)) for rp in ring]
    _, ridges = extract_boundary_ports(vals, ring, refine_azimuth=True)

    c_unit, t1, t2, ang_radius = ring_frame(ring)
    for r in ridges:
        v = np.array(r.position)
        v_unit = v / np.linalg.norm(v)
        # Reconstruct the azimuth from the position in the ring's own frame.
        proj = v_unit - math.cos(ang_radius) * c_unit
        proj /= np.linalg.norm(proj)
        phi = math.atan2(float(np.dot(proj, t2)), float(np.dot(proj, t1)))
        # Signed angular difference, wrapped to [-pi, pi).
        assert abs((phi - r.phi_rad + math.pi) % (2.0 * math.pi) - math.pi) < 1e-9


def test_no_refinement_is_unchanged() -> None:
    """Default off: ports keep their sample azimuth, position and scalar value."""
    ring = _ring(48)
    vals = [math.cos(3.0 * (rp.phi_rad - 0.3)) for rp in ring]
    valleys, ridges = extract_boundary_ports(vals, ring)

    by_phi = {rp.phi_rad: rp.position for rp in ring}
    for p in [*valleys, *ridges]:
        assert p.phi_rad in by_phi
        assert p.position == by_phi[p.phi_rad]


def test_refinement_handles_flat_stencil() -> None:
    """A degenerate three-point stencil keeps the sample instead of dividing by ~0."""
    # Collinear neighbours: den = y_m - 2*y_0 + y_p = 0 with y_m != y_p, so the
    # parabola vertex is at infinity and the sample must be kept.
    smooth = np.array([0.0, 1.0, 2.0, 3.0, 2.0, 0.0])
    assert _refine_extremum(smooth, 3, len(smooth), 0.1) == 0.0


def test_refinement_clamps_to_bracketing_cells() -> None:
    """A near-degenerate stencil cannot push a port beyond one sample step."""
    # Near-collinear: den is tiny with a non-zero numerator, so the raw parabola
    # vertex sits hundreds of sample steps away. The clamp keeps one step.
    smooth = np.array([1.0, 0.5, 1e-9, 0.0, 0.0, 0.0])
    assert abs(_refine_extremum(smooth, 1, len(smooth), 0.1)) <= 0.1 + 1e-15


def test_profile_loop_accepts_refine_flag() -> None:
    """The flag is threaded through the end-to-end profiling entry point."""
    pts = _generate_fibonacci_sphere(num_points=1500, radius=1.0)
    f_vals = pts[:, 0] ** 2 - pts[:, 1] ** 2
    ring = _ring(72, ang_deg=18.0)

    res = profile_boundary_loop(ring, pts, f_vals, refine_azimuth=True)
    assert res.is_topologically_balanced
    assert res.fold_order == 2
    assert len(res.valley_ports) == 2
    assert len(res.ridge_ports) == 2

    snapped = profile_boundary_loop(ring, pts, f_vals, refine_azimuth=False)
    # Same topology, sub-sample placement.
    assert len(snapped.valley_ports) == len(res.valley_ports)
    assert any(
        a.phi_rad != b.phi_rad for a, b in zip(snapped.valley_ports, res.valley_ports)
    )
