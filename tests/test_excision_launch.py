"""Unit tests for the excision-based saddle launch seed subroutine.

The load-bearing cases are the higher-order saddles the reduction stage produces:
a monkey saddle (k = 3) and an octupolar cross (k = 4) must yield 3 + 3 and 4 + 4
branches from the excision loop, with launch positions on the sphere, tangents
perpendicular to the radial, and azimuths at the ideal 360/k spacing. The failure
taxonomy is pinned too, because the ring-expansion loop routes on it.
"""

import math

import numpy as np
import pytest

from gba_topology2.src.excision_launch import (
    FAILURE_REASONS,
    extract_excision_launch_seed,
)
from gba_topology2.src.micro_cluster import BoundaryRingPoint, generate_boundary_ring
from gba_topology2.src.saddle_launch import SaddleLaunchSeed


def _sphere(num_points: int = 6000, radius: float = 1.0) -> np.ndarray:
    """Uniform points on a sphere."""
    indices = np.arange(0, num_points, dtype=float) + 0.5
    phi = np.arccos(1 - 2 * indices / num_points)
    theta = math.pi * (1 + 5**0.5) * indices
    return radius * np.column_stack(
        [np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)]
    )


PTS = _sphere()
X, Y = PTS[:, 0], PTS[:, 1]

# Analytic fields about the north pole; each has its critical point at (0, 0, 1)
# and a critical value of 0.
SIMPLE = X**2 - Y**2  # k = 2
MONKEY = X**3 - 3.0 * X * Y**2  # k = 3
OCTUPOLE = X**4 - 6.0 * X**2 * Y**2 + Y**4  # k = 4
MAXIMUM = -(X**2 + Y**2)  # an extremum, not a saddle


def _seed(
    f_vals: np.ndarray,
    ang_deg: float = 15.0,
    expected_k: int = 2,
    num_samples: int = 48,
    saddle_value: float = 0.0,
    vertex_id: int = -1,
    multiplicity: int = 0,
    centre: tuple[float, float, float] = (0.0, 0.0, 1.0),
) -> SaddleLaunchSeed:
    """Profile a circular ring about the north pole and extract its seed."""
    ring = generate_boundary_ring(
        centroid=centre,
        sphere_radius=1.0,
        angular_radius_rad=math.radians(ang_deg),
        num_samples=num_samples,
    )
    return extract_excision_launch_seed(
        ring_points=ring,
        mesh_pts=PTS,
        f_vals=f_vals,
        saddle_id="SAD_TEST",
        saddle_position=centre,
        saddle_value=saddle_value,
        expected_k=expected_k,
        vertex_id=vertex_id,
        multiplicity=multiplicity,
    )


def _azimuths(branches: list) -> list[float]:
    """Azimuth of each branch's launch position, in the ring's own frame."""
    return [math.atan2(b.launch_position[1], b.launch_position[0]) for b in branches]


def test_simple_saddle_gives_two_and_two() -> None:
    """k = 2: two ridge (ascent) and two valley (descent) branches, 4 crossings."""
    seed = _seed(SIMPLE, ang_deg=15.0, expected_k=2, multiplicity=1)
    assert seed.failure_reason is None
    assert len(seed.ascent_branches) == 2
    assert len(seed.descent_branches) == 2
    assert seed.fold_order == 2
    assert len(seed.separatrix_directions) == 4
    assert all(b.kind == "ascent" for b in seed.ascent_branches)
    assert all(b.kind == "descent" for b in seed.descent_branches)


def test_monkey_saddle_gives_three_and_three() -> None:
    """k = 3: the degenerate-Hessian case, resolved by the loop's port count."""
    seed = _seed(MONKEY, ang_deg=20.0, expected_k=3, multiplicity=2)
    assert seed.failure_reason is None
    assert len(seed.ascent_branches) == 3
    assert len(seed.descent_branches) == 3
    assert seed.fold_order == 3
    assert len(seed.separatrix_directions) == 6


def test_octupolar_cross_gives_four_and_four() -> None:
    """k = 4: four ridge and four valley branches, 8 crossings."""
    seed = _seed(OCTUPOLE, ang_deg=20.0, expected_k=4, multiplicity=3)
    assert seed.failure_reason is None
    assert len(seed.ascent_branches) == 4
    assert len(seed.descent_branches) == 4
    assert len(seed.separatrix_directions) == 8


def test_branch_azimuths_are_equally_spaced() -> None:
    """A symmetric field's ridge azimuths sit at the ideal 360/k spacing."""
    for f_vals, k in ((SIMPLE, 2), (MONKEY, 3), (OCTUPOLE, 4)):
        seed = _seed(f_vals, ang_deg=20.0, expected_k=k)
        az = sorted(_azimuths(seed.ascent_branches))
        step = 2.0 * math.pi / k
        gaps = [(az[(i + 1) % k] - az[i]) % (2.0 * math.pi) for i in range(k)]
        for gap in gaps:
            assert gap == pytest.approx(step, abs=math.radians(6.0)), (
                f"k={k}: gap {math.degrees(gap):.2f} deg"
            )


def test_ascent_and_descent_interleave() -> None:
    """Ridge and valley sectors alternate around the loop, as discrete Morse requires."""
    seed = _seed(MONKEY, ang_deg=20.0, expected_k=3)
    a = sorted(_azimuths(seed.ascent_branches))
    d = sorted(_azimuths(seed.descent_branches))
    merged = sorted([(p, "a") for p in a] + [(p, "d") for p in d], key=lambda t: t[0])
    kinds = [k for _, k in merged]
    # A cyclic sequence of 3 a's and 3 d's that alternates has exactly 6 sign flips.
    flips = sum(1 for i in range(6) if kinds[i] != kinds[(i + 1) % 6])
    assert flips == 6


def test_launch_positions_lie_on_the_sphere() -> None:
    """Every launch point sits on the sphere of radius R, not inside it."""
    seed = _seed(MONKEY, ang_deg=20.0, expected_k=3)
    for b in seed.all_branches:
        r = math.sqrt(sum(c * c for c in b.launch_position))
        assert r == pytest.approx(1.0, abs=1e-9)


def test_launch_positions_lie_on_the_ring() -> None:
    """Launch points are on the excision loop: constant angular distance from centre."""
    ang = 20.0
    seed = _seed(MONKEY, ang_deg=ang, expected_k=3)
    c = np.array([0.0, 0.0, 1.0])
    for b in seed.all_branches:
        v = np.array(b.launch_position)
        cosang = float(np.clip(np.dot(v / np.linalg.norm(v), c), -1.0, 1.0))
        assert math.degrees(math.acos(cosang)) == pytest.approx(ang, abs=1e-6)


def test_tangent_directions_are_perpendicular_to_the_normal() -> None:
    """A launch tangent must lie in the tangent plane, or Step 3 leaves the sphere."""
    seed = _seed(MONKEY, ang_deg=20.0, expected_k=3)
    for b in seed.all_branches:
        p = np.array(b.launch_position)
        p = p / np.linalg.norm(p)
        t = np.array(b.tangent_direction)
        assert float(np.linalg.norm(t)) == pytest.approx(1.0, abs=1e-9)
        assert abs(float(np.dot(p, t))) < 1e-9


def test_tangent_points_away_from_the_ring_centre() -> None:
    """The separatrix runs centre -> port -> outward, so the tangent points out."""
    seed = _seed(MONKEY, ang_deg=20.0, expected_k=3)
    c = np.array([0.0, 0.0, 1.0])
    for b in seed.all_branches:
        p = np.array(b.launch_position)
        t = np.array(b.tangent_direction)
        # Moving from the port along the tangent increases distance from the axis.
        moved = p + 0.05 * t
        moved = moved / np.linalg.norm(moved)
        start = p / np.linalg.norm(p)
        assert float(np.dot(moved, c)) < float(np.dot(start, c))


def test_sector_index_is_dense_and_ordered() -> None:
    """Branches are numbered 0..k-1 counter-clockwise, with no gaps."""
    seed = _seed(OCTUPOLE, ang_deg=20.0, expected_k=4)
    for branches in (seed.ascent_branches, seed.descent_branches):
        assert [b.sector_index for b in branches] == list(range(len(branches)))


def test_ring_radius_is_reported() -> None:
    """The accepted radius is carried on the seed for the expansion loop's bookkeeping."""
    seed = _seed(MONKEY, ang_deg=20.0, expected_k=3)
    assert math.degrees(seed.ring_radius_rad) == pytest.approx(20.0, abs=1e-6)


def test_index_agrees_with_expected_k() -> None:
    """A ring whose port count matches the reduction's index is marked agreeing."""
    seed = _seed(MONKEY, ang_deg=20.0, expected_k=3)
    assert seed.index_agrees is True


def test_index_mismatch_is_reported_not_silently_fixed() -> None:
    """A disagreement is surfaced; the measured k is never overwritten by expected_k.

    This is the audit the reduction stage depends on: if a constellation reported
    chi = -2 (k = 3) but its loop shows 4 ridge ports, the seed keeps k = 4 and
    flags the disagreement, so the index is visible as suspect.
    """
    seed = _seed(MONKEY, ang_deg=20.0, expected_k=4)
    assert seed.fold_order == 3  # measured, not coerced to the expected 4
    assert seed.index_agrees is False


def test_extremum_ring_is_not_accepted_as_a_saddle() -> None:
    """A ring around a maximum has a flat profile: noise, not topology.

    The profile of an extremum's loop is nearly constant, so IDW noise alone
    produces many balanced "ports". The subroutine reports what it measured and
    marks the index as disagreeing; rejecting the ring is the expansion loop's
    job, which is why the gate lives there and not here.
    """
    seed = _seed(MAXIMUM, ang_deg=15.0, expected_k=2, saddle_value=-0.9)
    # Not a k = 2 saddle signature, whichever way the noise fell.
    assert seed.fold_order != 2 or seed.index_agrees is False
    if seed.failure_reason is None:
        assert seed.fold_order > 2  # noise-generated port count, clearly off


def test_constant_field_yields_no_ports() -> None:
    """A structureless loop has no extrema to launch from."""
    seed = _seed(np.zeros(len(PTS)), ang_deg=15.0, expected_k=2)
    assert seed.failure_reason == "no_ports"
    assert seed.ascent_branches == []
    assert seed.descent_branches == []
    assert seed.fold_order == 0
    assert seed.ring_radius_rad == 0.0


def test_empty_ring_fails_cleanly() -> None:
    """No loop means nothing to profile."""
    seed = extract_excision_launch_seed(
        ring_points=[],
        mesh_pts=PTS,
        f_vals=SIMPLE,
        saddle_id="SAD_1",
        saddle_position=(0.0, 0.0, 1.0),
        saddle_value=0.0,
        expected_k=2,
    )
    assert seed.failure_reason == "empty_ring"
    assert seed.index_agrees is False


def test_too_few_samples_fails_cleanly() -> None:
    """A ring with fewer than four samples cannot carry a periodic profile."""
    seed = _seed(SIMPLE, ang_deg=15.0, expected_k=2, num_samples=3)
    assert seed.failure_reason == "too_few_samples"


def test_failure_reasons_are_a_closed_set() -> None:
    """Every reason the expansion loop routes on is declared in the module."""
    seen: set[str] = set()

    empty = extract_excision_launch_seed([], PTS, SIMPLE, "S", (0.0, 0.0, 1.0), 0.0, 2)
    seen.add(str(empty.failure_reason))
    seen.add(
        str(_seed(SIMPLE, ang_deg=15.0, expected_k=2, num_samples=3).failure_reason)
    )
    seen.add(str(_seed(np.zeros(len(PTS)), ang_deg=15.0, expected_k=2).failure_reason))

    assert "" not in seen and "None" not in seen
    assert seen <= set(FAILURE_REASONS)
    # The reasons the expansion loop branches on must exist and be reachable by
    # construction, not only declared.
    assert "unbalanced" in FAILURE_REASONS
    assert "below_min_k" in FAILURE_REASONS
    assert "empty_ring" in seen
    assert "too_few_samples" in seen
    assert "no_ports" in seen


def test_unbalanced_ring_reports_unbalanced() -> None:
    """A loop with more valleys than ridges is flagged, never given a fabricated k.

    A quadrupole tilted by a dipole leaves one ridge below the prominence filter,
    so the loop reports 2 valleys against 1 ridge. Inventing a fold order from that
    would seed Step 3 with a branch count the field does not have.
    """
    f_tilt = X**2 - Y**2 + 1.0 * X
    seed = _seed(f_tilt, ang_deg=20.0, expected_k=2)
    assert seed.failure_reason == "unbalanced"
    assert seed.fold_order == 0
    assert seed.ascent_branches == []
    assert seed.descent_branches == []
    assert seed.ring_radius_rad == 0.0
    assert seed.index_agrees is False


def test_balanced_one_plus_one_is_below_min_k() -> None:
    """A balanced 1 + 1 loop is an extremum signature, not a saddle.

    Dominating a quadrupole with a dipole leaves one valley and one ridge: balanced,
    and not a separatrix structure. Accepting it would launch a single ascent and a
    single descent from a point that is not a saddle.
    """
    f_dom = X**2 - Y**2 + 3.0 * X
    seed = _seed(f_dom, ang_deg=20.0, expected_k=2)
    assert seed.failure_reason == "below_min_k"
    assert seed.fold_order == 0


def test_off_pole_centre_is_handled() -> None:
    """The loop frame is recovered from the ring, so any centre works."""
    # Rotate the quadrupole so its saddle sits at (1, 0, 0).
    ang = math.pi / 2.0
    rot = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, math.cos(ang), -math.sin(ang)],
            [0.0, math.sin(ang), math.cos(ang)],
        ]
    )
    p = PTS @ rot.T
    f = p[:, 1] ** 2 - p[:, 2] ** 2
    centre = (1.0, 0.0, 0.0)
    seed = _seed(
        f,
        ang_deg=18.0,
        expected_k=2,
        centre=centre,
    )
    assert seed.failure_reason is None
    assert seed.fold_order == 2
    for b in seed.all_branches:
        v = np.array(b.launch_position)
        cosang = float(np.clip(np.dot(v / np.linalg.norm(v), np.array(centre)), -1, 1))
        assert math.degrees(math.acos(cosang)) == pytest.approx(18.0, abs=1e-6)


def test_local_and_global_sampling_agree() -> None:
    """The spatial index changes cost, not the seed."""
    from vtkmodules.vtkCommonCore import vtkPoints
    from vtkmodules.vtkCommonDataModel import vtkPolyData

    from gba_topology2.src.point_locator import make_kdtree_nearest_finder

    ring = generate_boundary_ring((0.0, 0.0, 1.0), 1.0, math.radians(20.0), 48)
    vp = vtkPoints()
    for p in PTS:
        vp.InsertNextPoint(float(p[0]), float(p[1]), float(p[2]))
    poly = vtkPolyData()
    poly.SetPoints(vp)

    global_seed = extract_excision_launch_seed(
        ring, PTS, MONKEY, "S", (0.0, 0.0, 1.0), 0.0, 3
    )
    local_seed = extract_excision_launch_seed(
        ring,
        PTS,
        MONKEY,
        "S",
        (0.0, 0.0, 1.0),
        0.0,
        3,
        find_k_nearest=make_kdtree_nearest_finder(poly),
    )

    assert global_seed.fold_order == local_seed.fold_order == 3
    for a, b in zip(global_seed.all_branches, local_seed.all_branches):
        assert a.kind == b.kind
        for ca, cb in zip(a.launch_position, b.launch_position):
            assert ca == pytest.approx(cb, abs=1e-12)
    assert all(isinstance(r, BoundaryRingPoint) for r in ring)


def test_seed_carries_identity_fields() -> None:
    """The identifiers the viewer's stats and failure list need are preserved."""
    seed = _seed(MONKEY, ang_deg=20.0, expected_k=3, vertex_id=142, multiplicity=2)
    assert seed.saddle_id == "SAD_TEST"
    assert seed.vertex_id == 142
    assert seed.multiplicity == 2
    assert seed.saddle_value == 0.0
    assert seed.saddle_position == (0.0, 0.0, 1.0)
