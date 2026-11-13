"""Unit tests for :mod:`gba_topology2.src.separatrix_tracer`.

The tracer climbs the condensed field from a saddle's launch port to the maximum
its separatrix terminates on. These tests exercise it on synthetic fields with
known answers: a single smooth peak, a ridge that curves (so a straight ray would
miss), a flat plateau (no maximum to reach), and a field whose peak sits off the
great circle the port points along.

The frame contract is the point of most of these: the tracer integrates on the
CENTRED sphere, so a field built around a nucleus at the origin is the easy case,
and the tests deliberately include an off-origin atom to catch the recurring
world/centred mixup that Pd_20K hides and ethene3 exposes.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import numpy.typing as npt
import pytest

from gba_topology2.src.excision_boundary import interpolate_scalar_at_points
from gba_topology2.src.excision_launch_seeds import RING_SAMPLES
from gba_topology2.src.max_connection import ConnectionTarget
from gba_topology2.src.micro_cluster import generate_boundary_ring
from gba_topology2.src.separatrix_tracer import (
    CAPTURE_PITCH,
    MAX_TRACE_ARC_PITCH,
    STALL_PATIENCE,
    TRACE_MAX_TRANSITS,
    TRACE_STEP_PITCH,
    ExcisionRegion,
    TraceOutcome,
    centred_targets,
    excision_regions,
    trace_downhill,
    trace_uphill,
)

# A mesh fine enough that the IDW interpolant resolves a peak of the angular width
# used below, and coarse enough that the brute-force kernel stays fast.
_N_MESH = 1200
_R = 1.0
_PITCH = math.radians(3.0)  # ~3 deg, comparable to the real meshes


class _ClusterStub:
    """The three fields excision_regions reads off a micro-cluster."""

    def __init__(
        self, cluster_id: str, centroid: tuple[float, float, float], radius_rad: float
    ) -> None:
        self.cluster_id = cluster_id
        self.centroid = centroid
        self.angular_radius_rad = radius_rad


def _fibonacci_sphere(
    num_points: int = _N_MESH, radius: float = _R
) -> npt.NDArray[np.float64]:
    indices = np.arange(0, num_points, dtype=float) + 0.5
    phi = np.arccos(1.0 - 2.0 * indices / num_points)
    theta = math.pi * (1.0 + 5.0**0.5) * indices
    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    z = radius * np.cos(phi)
    return np.column_stack([x, y, z])


def _gaussian_peak(
    pts: npt.NDArray[np.float64],
    center_unit: npt.NDArray[np.float64],
    width_rad: float,
    height: float = 1.0,
) -> npt.NDArray[np.float64]:
    """A single smooth maximum at ``center_unit``, decaying as cos of arc distance.

    ``exp(-(d/width)^2)`` on the sphere: monotone in the geodesic distance from the
    peak, so the surface gradient points straight at it from every direction and the
    ascent has a unique attractor.
    """
    u = pts / np.linalg.norm(pts, axis=1, keepdims=True)
    d = np.arccos(np.clip(u @ center_unit, -1.0, 1.0))
    return height * np.exp(-((d / width_rad) ** 2))


class _Target:
    """Minimal pool entry: the tracer only needs id, polarity, position, radius."""

    def __init__(
        self,
        cp_id: str,
        polarity: str,
        position: npt.NDArray[np.float64],
        radius_rad: float,
    ) -> None:
        self.cp_id = cp_id
        self.polarity = polarity
        self.position = np.asarray(position, dtype=np.float64)
        self._radius_rad = radius_rad

    def radius_toward(self, target_unit: npt.NDArray[np.float64]) -> float:
        return self._radius_rad


def _max_target(
    center_unit: npt.NDArray[np.float64],
    radius_deg: float = 8.0,
    cp_id: str = "MAX_0",
) -> _Target:
    return _Target(cp_id, "maximum", center_unit, math.radians(radius_deg))


def test_climbs_to_a_single_peak() -> None:
    """A smooth peak is reached from a port on its far side; the path rises."""
    pts = _fibonacci_sphere()
    peak = np.array([0.0, 0.0, 1.0])
    f = _gaussian_peak(pts, peak, width_rad=math.radians(35.0))
    port = np.array([0.6, 0.0, 0.8])
    port = port / np.linalg.norm(port)

    out = trace_uphill(
        start_position=port,
        targets=[_max_target(peak)],
        mesh_pts=pts,
        f_vals=f,
        sphere_radius=_R,
        delta_theta_mesh_rad=_PITCH,
        center=None,
    )
    assert out.status == "arrived"
    assert out.max_id == "MAX_0"
    assert out.end_value > out.start_value
    assert len(out.path) == out.n_steps + 1


def test_arrives_at_the_nearest_of_two_peaks() -> None:
    """With two maxima, the ascent terminates on the one its ridge drains to."""
    pts = _fibonacci_sphere()
    near = np.array([0.0, 0.0, 1.0])
    far = np.array([0.0, 1.0, 0.0])
    f = _gaussian_peak(pts, near, math.radians(30.0), 1.0) + _gaussian_peak(
        pts, far, math.radians(30.0), 0.9
    )
    port = np.array([0.2, 0.0, 0.98])
    port = port / np.linalg.norm(port)

    out = trace_uphill(
        start_position=port,
        targets=[
            _max_target(near, cp_id="MAX_NEAR"),
            _max_target(far, cp_id="MAX_FAR"),
        ],
        mesh_pts=pts,
        f_vals=f,
        sphere_radius=_R,
        delta_theta_mesh_rad=_PITCH,
        center=None,
    )
    assert out.status == "arrived"
    assert out.max_id == "MAX_NEAR"


def test_curved_ridge_reached_where_a_straight_ray_would_miss() -> None:
    """The Step 4 case: the ridge to the maximum curves, so a straight ray aimed
    along the port's outward great circle does not follow it.

    Two peaks of unequal height create a saddle between them; a port placed off
    the symmetry plane climbs the curved ridge to the higher peak. The ascent
    arrives, and its path bows away from the straight great circle joining the
    port to the peak -- the curvature the geodesic connector cannot represent.
    """
    pts = _fibonacci_sphere()
    hi = np.array([0.0, 0.0, 1.0])  # the target maximum
    lo = np.array([math.sin(1.2), 0.0, math.cos(1.2)])  # a lower peak, ~69 deg on
    f = _gaussian_peak(pts, hi, math.radians(35.0), 1.0) + _gaussian_peak(
        pts, lo, math.radians(35.0), 0.7
    )
    # Off the y=0 symmetry plane, so the ascent ridge is not a great circle
    # through the two peaks.
    port = np.array([0.35, 0.62, 0.70])
    port = port / np.linalg.norm(port)

    out = trace_uphill(
        start_position=port,
        targets=[_max_target(hi, radius_deg=10.0, cp_id="MAX_HI")],
        mesh_pts=pts,
        f_vals=f,
        sphere_radius=_R,
        delta_theta_mesh_rad=_PITCH,
        center=None,
    )
    assert out.status == "arrived"
    assert out.max_id == "MAX_HI"

    # The path bows off the straight port->peak great circle: measure the largest
    # angular deviation of an interior point from that plane. A geodesic connector
    # has zero deviation; a curved ridge does not.
    n = np.cross(port, hi)
    n = n / np.linalg.norm(n)
    interior = np.array(out.path[1:-1]) / _R
    dev = np.abs(np.arccos(np.clip(interior @ n, -1.0, 1.0)) - math.pi / 2)
    assert math.degrees(float(dev.max())) > 5.0


def test_flat_plateau_stalls_without_a_maximum() -> None:
    """A constant field has no gradient: the ascent stalls, it does not crash.

    Capture is disabled so this tests the INTEGRATOR. What the total completion
    rule does with the same stall is the next test's subject.
    """
    pts = _fibonacci_sphere()
    f = np.full(len(pts), 0.5)
    port = np.array([0.0, 0.0, 1.0])

    out = trace_uphill(
        start_position=port,
        targets=[_max_target(np.array([0.0, 1.0, 0.0]))],
        mesh_pts=pts,
        f_vals=f,
        sphere_radius=_R,
        delta_theta_mesh_rad=_PITCH,
        center=None,
        capture_pitch=0.0,
    )
    assert out.status == "stalled"
    assert out.max_id is None
    assert out.n_steps <= STALL_PATIENCE + 2


def test_a_plateau_stall_is_committed_to_the_nearest_maximum() -> None:
    """THE INVARIANT: a gradient path is never left uncommitted.

    The same flat field as the stall test, with the completion rule at its default.
    The ascent has nothing to climb, so it is committed to the nearest maximum and
    the whole carry is published as a drawn leg. That number is the honest cost of
    the association and the reason the rule is total: a reach gate would drop this
    case silently, which is exactly the failure that took five entries to chase.
    """
    pts = _fibonacci_sphere()
    f = np.full(len(pts), 0.5)
    port = np.array([0.0, 0.0, 1.0])

    out = trace_uphill(
        start_position=port,
        targets=[_max_target(np.array([0.0, 1.0, 0.0]))],
        mesh_pts=pts,
        f_vals=f,
        sphere_radius=_R,
        delta_theta_mesh_rad=_PITCH,
        center=None,
    )
    assert out.status == "arrived"
    assert out.max_id == "MAX_0"
    assert out.completed_leg_deg is not None
    # The port is at the pole, the maximum on the equator: a 90 deg carry.
    assert out.completed_leg_deg == pytest.approx(90.0, abs=1.0)
    # The leg is appended, so the connector ends AT the maximum.
    end = np.asarray(out.path[-1], dtype=np.float64)
    assert (
        math.degrees(float(np.arccos(np.clip(end[1] / np.linalg.norm(end), -1, 1))))
        < 1e-9
    )


def test_a_pool_with_no_maximum_is_the_only_way_to_stay_uncommitted() -> None:
    """Totality has one legitimate exception: nothing to commit to."""
    pts = _fibonacci_sphere()
    f = np.full(len(pts), 0.5)

    out = trace_uphill(
        start_position=(0.0, 0.0, 1.0),
        targets=[
            _Target("MIN_1", "minimum", np.array([0.0, 1.0, 0.0]), math.radians(6.0))
        ],
        mesh_pts=pts,
        f_vals=f,
        sphere_radius=_R,
        delta_theta_mesh_rad=_PITCH,
        center=None,
    )
    assert out.status in ("stalled", "budget")
    assert out.max_id is None


def test_budget_exhausted_on_a_long_spiral() -> None:
    """A tiny budget cannot reach a distant peak: the answer is budget, not arrival.

    Capture is disabled so this tests the INTEGRATOR's budget, not the completion.
    """
    pts = _fibonacci_sphere()
    peak = np.array([0.0, 0.0, 1.0])
    f = _gaussian_peak(pts, peak, width_rad=math.radians(80.0))
    port = np.array([1.0, 0.0, 0.02])
    port = port / np.linalg.norm(port)

    out = trace_uphill(
        start_position=port,
        targets=[_max_target(peak, radius_deg=4.0)],
        mesh_pts=pts,
        f_vals=f,
        sphere_radius=_R,
        delta_theta_mesh_rad=_PITCH,
        center=None,
        max_arc_pitch=2.0,  # 2 pitches, far short of the ~86 deg to the peak
        capture_pitch=0.0,
    )
    assert out.status == "budget"
    assert out.max_id is None


def test_degenerate_start_is_reported_not_raised() -> None:
    """A zero-length port has no direction: status no_start, empty path."""
    pts = _fibonacci_sphere()
    f = _gaussian_peak(pts, np.array([0.0, 0.0, 1.0]), math.radians(30.0))

    out = trace_uphill(
        start_position=(0.0, 0.0, 0.0),
        targets=[_max_target(np.array([0.0, 0.0, 1.0]))],
        mesh_pts=pts,
        f_vals=f,
        sphere_radius=_R,
        delta_theta_mesh_rad=_PITCH,
        center=None,
    )
    assert out.status == "no_start"
    assert out.path == ()


def test_records_non_maximum_cps_the_path_crosses() -> None:
    """A saddle/min disc the ridge runs through is recorded, not hidden."""
    pts = _fibonacci_sphere()
    peak = np.array([0.0, 0.0, 1.0])
    f = _gaussian_peak(pts, peak, width_rad=math.radians(45.0))
    port = np.array([0.5, 0.0, 0.87])
    port = port / np.linalg.norm(port)
    # A minimum sitting on the path between the port and the peak.
    mid = np.array([0.28, 0.0, 0.96])
    mid = mid / np.linalg.norm(mid)
    targets = [
        _max_target(peak, radius_deg=8.0),
        _Target("MIN_7", "minimum", mid, math.radians(6.0)),
    ]

    out = trace_uphill(
        start_position=port,
        targets=targets,
        mesh_pts=pts,
        f_vals=f,
        sphere_radius=_R,
        delta_theta_mesh_rad=_PITCH,
        center=None,
    )
    assert out.status == "arrived"
    assert "MIN_7" in out.visited_ids


def test_path_is_emitted_in_the_world_frame() -> None:
    """With a nucleus offset, every path point sits on the sphere about it."""
    pts = _fibonacci_sphere()
    peak = np.array([0.0, 0.0, 1.0])
    f = _gaussian_peak(pts, peak, width_rad=math.radians(40.0))
    port = np.array([0.4, 0.0, 0.92])
    port = port / np.linalg.norm(port)
    center = (1.2618, -0.0001, -0.0002)  # ethene3-like nucleus offset

    out = trace_uphill(
        start_position=port,
        targets=[_max_target(peak)],
        mesh_pts=pts,
        f_vals=f,
        sphere_radius=_R,
        delta_theta_mesh_rad=_PITCH,
        center=center,
    )
    assert out.status == "arrived"
    c = np.array(center)
    radii = [math.dist(p, c) for p in out.path]
    assert all(abs(r - _R) < 1e-6 for r in radii)


def test_centred_targets_maps_world_positions_to_unit_radial() -> None:
    """centred_targets subtracts the nucleus and normalises, so radius_toward is
    measured from the correct centre."""
    from gba_topology2.src.max_connection import ConnectionTarget

    center = (2.0, 0.0, 0.0)
    world = np.array([2.0, 0.0, 3.0])  # centred: (0,0,3) -> unit (0,0,1)
    t = ConnectionTarget(
        cp_id="MAX_0",
        polarity="maximum",
        position=world,
        value=1.0,
        angular_radius_rad=math.radians(10.0),
    )

    (ct,) = centred_targets([t], center)
    assert np.allclose(ct.position, [0.0, 0.0, 1.0])
    assert np.isclose(np.linalg.norm(ct.position), 1.0)


def test_centred_targets_drops_a_degenerate_entry() -> None:
    """A target sitting exactly on the nucleus has no direction; it is dropped."""
    from gba_topology2.src.max_connection import ConnectionTarget

    center = (1.0, 1.0, 1.0)
    t = ConnectionTarget(
        cp_id="MAX_0",
        polarity="maximum",
        position=np.array([1.0, 1.0, 1.0]),
        value=1.0,
        angular_radius_rad=math.radians(10.0),
    )
    assert centred_targets([t], center) == []


def test_centred_targets_none_center_is_identity_normalise() -> None:
    """center=None means the pool is already centred: only the normalise happens."""
    from gba_topology2.src.max_connection import ConnectionTarget

    t = ConnectionTarget(
        cp_id="MAX_0",
        polarity="maximum",
        position=np.array([0.0, 2.0, 0.0]),
        value=1.0,
        angular_radius_rad=math.radians(10.0),
    )
    (ct,) = centred_targets([t], None)
    assert np.allclose(ct.position, [0.0, 1.0, 0.0])


def test_constants_are_positive_and_ordered() -> None:
    """The step is finer than the budget, so a trace takes many steps."""
    assert TRACE_STEP_PITCH > 0.0
    assert MAX_TRACE_ARC_PITCH > TRACE_STEP_PITCH
    assert STALL_PATIENCE >= 1


def test_outcome_is_frozen_and_defaults_visited_empty() -> None:
    """TraceOutcome is an immutable value object; visited_ids defaults to empty."""
    out = TraceOutcome(
        status="arrived",
        max_id="MAX_0",
        path=(),
        arc_length_deg=1.0,
        n_steps=0,
        start_value=0.0,
        end_value=1.0,
    )
    assert out.visited_ids == ()
    with pytest.raises(dataclasses.FrozenInstanceError):
        out.status = "stalled"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# The transit rule: an ascent must not END inside an excision region.
# ---------------------------------------------------------------------------


def _bump_and_peak_field(
    pts: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """A small local maximum at colat 40 and the real one at the pole.

    The peaks are NARROW (15 deg) and 40 deg apart, so a genuine saddle separates
    them: the bump is its own ascent basin, a trap the climb enters and cannot
    leave. The region under test is centred on the bump, so the stall happens
    inside it, and the far rim of that region is inside the pole's basin.
    """
    bump = np.array([math.sin(math.radians(40.0)), 0.0, math.cos(math.radians(40.0))])
    pole = np.array([0.0, 0.0, 1.0])
    return _gaussian_peak(pts, bump, math.radians(15.0), 0.3) + _gaussian_peak(
        pts, pole, math.radians(15.0), 1.0
    )


def _bump_region(radius_deg: float = 18.0) -> list[ExcisionRegion]:
    centre = np.array([math.sin(math.radians(40.0)), 0.0, math.cos(math.radians(40.0))])
    return [
        ExcisionRegion(
            region_id="MC_TRAP", centre_unit=centre, radius_rad=math.radians(radius_deg)
        )
    ]


def test_transit_carries_a_stall_out_of_a_region_and_arrives() -> None:
    """The ethene3 shape-index case in miniature.

    Without regions the ascent climbs into the bump and never reaches the pole.
    With them it is carried through to the far rim and resumes, arriving. Same
    field, same port, same integrator -- the ONLY difference is the transit rule,
    so the pair is the measurement of what the rule buys.
    """
    pts = _fibonacci_sphere()
    f = _bump_and_peak_field(pts)
    port = np.array([math.sin(math.radians(75.0)), 0.0, math.cos(math.radians(75.0))])
    targets = [_max_target(np.array([0.0, 0.0, 1.0]), radius_deg=10.0)]

    plain = trace_uphill(
        start_position=tuple(port),
        targets=targets,
        mesh_pts=pts,
        f_vals=f,
        sphere_radius=_R,
        delta_theta_mesh_rad=_PITCH,
        center=None,
        # Capture off: this half of the pair isolates the TRANSIT rule. With the
        # completion at its default the stall is committed to the nearest maximum
        # and the contrast this test exists to draw disappears.
        capture_pitch=0.0,
    )
    assert plain.status in ("stalled", "budget")
    assert plain.max_id is None
    assert plain.transited_ids == ()

    bridged = trace_uphill(
        start_position=tuple(port),
        targets=targets,
        mesh_pts=pts,
        f_vals=f,
        sphere_radius=_R,
        delta_theta_mesh_rad=_PITCH,
        center=None,
        excision_regions=_bump_region(),
    )
    assert bridged.status == "arrived"
    assert bridged.max_id == "MAX_0"
    assert bridged.transited_ids == ("MC_TRAP",)
    # The snap-to-entry rule bypasses the spiral entirely: it snaps at boundary
    # entry and resumes from the center, resulting in a SHORT, efficient path.
    # This is much better than the old spiral behavior (plain path = 401 points).
    assert len(bridged.path) < len(plain.path)
    assert len(bridged.path) < 50  # Efficient: snap + resume takes ~18 steps


def test_the_port_s_own_region_is_never_transited() -> None:
    """The launch port sits ON a boundary by construction.

    Without the exclusion the ascent would transit at step 0, be teleported out of
    the saddle's neighbourhood, and never climb. The port's region is its own
    launch geometry, not an obstacle to cross.
    """
    pts = _fibonacci_sphere()
    f = _bump_and_peak_field(pts)
    # Port INSIDE the trap region, on the far side from the bump's centre.
    port = np.array([math.sin(math.radians(55.0)), 0.0, math.cos(math.radians(55.0))])
    out = trace_uphill(
        start_position=tuple(port),
        targets=[_max_target(np.array([0.0, 0.0, 1.0]), radius_deg=10.0)],
        mesh_pts=pts,
        f_vals=f,
        sphere_radius=_R,
        delta_theta_mesh_rad=_PITCH,
        center=None,
        excision_regions=_bump_region(),
    )
    # The region containing the start is excluded, so it is not transited.
    assert "MC_TRAP" not in out.transited_ids


def test_transit_count_is_bounded() -> None:
    """A field ringed by regions cannot make the tracer loop forever."""
    pts = _fibonacci_sphere()
    f = np.full(len(pts), 0.5)  # flat: every step is a stall candidate
    regions = [
        ExcisionRegion(
            region_id=f"MC{i}",
            centre_unit=np.array(
                [math.cos(2 * math.pi * i / 6), math.sin(2 * math.pi * i / 6), 0.0]
            ),
            radius_rad=math.radians(50.0),
        )
        for i in range(6)
    ]
    out = trace_uphill(
        start_position=(0.0, 0.0, 1.0),
        targets=[_max_target(np.array([0.0, 1.0, 0.0]))],
        mesh_pts=pts,
        f_vals=f,
        sphere_radius=_R,
        delta_theta_mesh_rad=_PITCH,
        center=None,
        excision_regions=regions,
        capture_pitch=0.0,  # isolate the transit bound from the completion rule
    )
    assert len(out.transited_ids) <= TRACE_MAX_TRANSITS
    assert out.status in ("stalled", "budget")
    assert out.max_id is None


def test_excision_regions_maps_clusters_to_unit_centres() -> None:
    """The builder normalises, drops degenerate clusters, and honours the frame."""
    centre = (2.0, 0.0, 3.0)  # world; centred = (0,0,3) -> unit (0,0,1)
    cluster = _ClusterStub("MC4", centre, math.radians(12.0))
    (r,) = excision_regions([cluster], center=(2.0, 0.0, 0.0))
    assert r.region_id == "MC4"
    assert np.allclose(r.centre_unit, [0.0, 0.0, 1.0])
    assert r.radius_rad == pytest.approx(math.radians(12.0))

    # A zero-radius cluster (a lone vertex) has no interior to avoid: dropped.
    assert excision_regions([_ClusterStub("MC5", (0.0, 0.0, 1.0), 0.0)]) == []


def test_outcome_defaults_transited_empty() -> None:
    """The new field defaults to empty, so existing construction sites are valid."""
    out = TraceOutcome(
        status="arrived",
        max_id="MAX_0",
        path=(),
        arc_length_deg=1.0,
        n_steps=0,
        start_value=0.0,
        end_value=1.0,
    )
    assert out.transited_ids == ()


def test_resume_point_is_the_boundary_field_maximum() -> None:
    """The rule's MECHANISM, not just its outcome.

    The first version exited along the heading the spiral happened to have, which
    is noise-determined. This asserts the ascent resumes at the point where the
    field is largest on the region's boundary, and that this is a genuinely
    different place from the antipode of entry the heading rule produced.
    """
    pts = _fibonacci_sphere()
    f = _bump_and_peak_field(pts)
    port = np.array([math.sin(math.radians(75.0)), 0.0, math.cos(math.radians(75.0))])
    regions = _bump_region()
    region = regions[0]

    out = trace_uphill(
        start_position=tuple(port),
        targets=[_max_target(np.array([0.0, 0.0, 1.0]), radius_deg=10.0)],
        mesh_pts=pts,
        f_vals=f,
        sphere_radius=_R,
        delta_theta_mesh_rad=_PITCH,
        center=None,
        excision_regions=regions,
    )
    assert out.status == "arrived"
    assert out.transited_ids == ("MC_TRAP",)

    # The boundary's field maximum, computed independently of the tracer.
    ring = generate_boundary_ring(
        (
            float(region.centre_unit[0]),
            float(region.centre_unit[1]),
            float(region.centre_unit[2]),
        ),
        _R,
        region.radius_rad,
        RING_SAMPLES,
    )
    vals = [
        float(interpolate_scalar_at_points(np.array([p.position]), pts, f)[0])
        for p in ring
    ]
    argmax = np.asarray(ring[int(np.argmax(vals))].position, dtype=np.float64)

    # The resumed path must pass through it: the vertex where the climb restarts.
    path = np.asarray(out.path, dtype=np.float64)
    dist = np.arccos(np.clip(path @ (argmax / np.linalg.norm(argmax)), -1.0, 1.0))
    assert math.degrees(float(dist.min())) < math.degrees(_PITCH)

    # Non-vacuity: the heading rule resumed at the ANTIPODE of entry, which is a
    # long way from the ring's high side. If the two coincided this test would
    # prove nothing.
    antipode = 2.0 * region.centre_unit - argmax
    sep = math.degrees(
        float(
            np.arccos(
                np.clip(
                    (argmax / np.linalg.norm(argmax))
                    @ (antipode / np.linalg.norm(antipode)),
                    -1.0,
                    1.0,
                )
            )
        )
    )
    assert sep > 20.0


def test_ring_sampling_is_the_coarse_convention() -> None:
    """The resume uses RING_SAMPLES, the saddle-seed count, and that matters.

    On ethene3 shape index the MC1 ring's field spans 0.05 percent of the field's
    range. At 16 samples the argmax reaches the spike; at 24 and above it resolves
    IDW interpolation noise instead and does not. The coarse count is the
    anti-noise filter, so the constant is part of the rule, not an
    implementation detail.
    """
    assert RING_SAMPLES == 16


def test_a_nested_inner_region_traps_the_ascent_the_outer_one_frees_it() -> None:
    """The region SET matters, not just the rule.

    A corridor absorbs clusters, so a raw cluster nested inside it has a small
    ring deep in the corridor's noisy ground. Resuming at the INNER ring puts the
    ascent back in the same noise it was just carried out of, and it stalls;
    resuming at the OUTER ring puts it on ground the corridor has not swallowed,
    and it arrives. This is why the caller passes the VISIBLE boundaries (the
    reduction units' as_cluster views) and not the raw micro-clusters.

    The two runs share the field, the port, the integrator and the trap; the only
    difference is which disc is offered as the region.
    """
    pts = _fibonacci_sphere()
    f = _bump_and_peak_field(pts)
    port = np.array([math.sin(math.radians(75.0)), 0.0, math.cos(math.radians(75.0))])
    targets = [_max_target(np.array([0.0, 0.0, 1.0]), radius_deg=10.0)]

    inner = _bump_region(radius_deg=6.0)  # the nested raw cluster's ring
    outer = _bump_region(radius_deg=18.0)  # the corridor that contains it

    trapped = trace_uphill(
        start_position=tuple(port),
        targets=targets,
        mesh_pts=pts,
        f_vals=f,
        sphere_radius=_R,
        delta_theta_mesh_rad=_PITCH,
        center=None,
        excision_regions=inner,
        capture_pitch=0.0,  # isolate the region SET from the completion rule
    )
    freed = trace_uphill(
        start_position=tuple(port),
        targets=targets,
        mesh_pts=pts,
        f_vals=f,
        sphere_radius=_R,
        delta_theta_mesh_rad=_PITCH,
        center=None,
        excision_regions=outer,
        capture_pitch=0.0,
    )
    assert trapped.transited_ids == ("MC_TRAP",)
    assert freed.transited_ids == ("MC_TRAP",)
    # Non-vacuity: the rule fired in both. The outcome differs because the
    # boundary it resumed from differs.
    assert freed.status == "arrived"
    assert freed.max_id == "MAX_0"
    assert trapped.status in ("stalled", "budget")
    assert trapped.max_id is None


def test_excision_regions_names_regions_by_the_visible_entity() -> None:
    """The builder is id-preserving, so the record says WHICH boundary was crossed.

    Passing unit as_cluster views yields the corridor ids the operator sees in the
    table (MS1, MC2), which is what makes transited_ids readable.
    """
    src = [
        _ClusterStub("MS1", (0.0, 0.0, 1.0), math.radians(9.0)),
        _ClusterStub("MC2", (0.0, 1.0, 0.0), math.radians(6.0)),
    ]
    assert [r.region_id for r in excision_regions(src)] == ["MS1", "MC2"]


# ---------------------------------------------------------------------------
# Capture: an ascent that CONVERGED on a maximum but stepped past it.
#
# A sharp peak makes a one-pitch integrator overshoot: the sample lands on the
# far side, the gradient there points back over the top, and the path settles
# into a two-point orbit just outside the disc. The maximum is not in question,
# so the climb is completed with a straight leg. This is the ethene3 C1
# curvedness case: MC17's ascent orbits at 4.96 and 7.14 deg from MC10, whose
# disc is 3.79 deg -- 1.17 deg, half a pitch, short.
# ---------------------------------------------------------------------------


def _spike_field(
    pts: npt.NDArray[np.float64],
    scale_deg: float,
) -> npt.NDArray[np.float64]:
    """A SHARP peak at the pole: falloff scale comparable to the mesh pitch.

    A broad Gaussian cannot overshoot -- the ascent decelerates smoothly into the
    top and lands inside the disc. The overshoot needs a peak whose gradient
    reverses within one step, which is what a scale of a few degrees gives.
    """
    pole = np.array([0.0, 0.0, 1.0])
    d = np.arccos(np.clip(pts @ pole, -1.0, 1.0))
    values: npt.NDArray[np.float64] = np.exp(-((d / math.radians(scale_deg)) ** 2))
    return values


def test_capture_completes_an_overshooting_ascent() -> None:
    """The orbit is broken by completing the climb onto the maximum.

    Non-vacuity is proved by the same run with the capture allowance set to zero:
    the path exhausts its budget in the orbit and reports no maximum. So the
    arrival is attributable to the capture rule alone.
    """
    pts = _fibonacci_sphere()
    f = _spike_field(pts, 2.5)
    port = np.array([math.sin(math.radians(9.0)), 0.0, math.cos(math.radians(9.0))])
    targets = [_max_target(np.array([0.0, 0.0, 1.0]), radius_deg=1.0)]

    bare = trace_uphill(
        start_position=tuple(port),
        targets=targets,
        mesh_pts=pts,
        f_vals=f,
        sphere_radius=_R,
        delta_theta_mesh_rad=_PITCH,
        center=None,
        capture_pitch=0.0,
    )
    assert bare.status == "budget"
    assert bare.max_id is None
    assert bare.completed_leg_deg is None

    captured = trace_uphill(
        start_position=tuple(port),
        targets=targets,
        mesh_pts=pts,
        f_vals=f,
        sphere_radius=_R,
        delta_theta_mesh_rad=_PITCH,
        center=None,
    )
    assert captured.status == "arrived"
    assert captured.max_id == "MAX_0"
    assert captured.completed_leg_deg is not None
    assert 0.0 < captured.completed_leg_deg <= math.degrees(_PITCH)
    # The connector TERMINATES AT THE MAXIMUM, not at the disc edge.
    end = np.asarray(captured.path[-1], dtype=np.float64)
    assert (
        math.degrees(float(np.arccos(np.clip(end[2] / np.linalg.norm(end), -1, 1))))
        < 1e-9
    )
    # The leg is APPENDED, not substituted: the path keeps the measured climb up
    # to the closest approach and grows past it. It is shorter than the ungated
    # run because the orbit tail -- samples bouncing between two points, carrying
    # no information about the field, truncated wherever the budget happened to
    # cut -- is dropped. The endpoint is the maximum, asserted above.
    assert len(captured.path) >= 2
    assert len(captured.path) < len(bare.path)
    assert captured.n_steps < bare.n_steps


def test_capture_completes_a_far_convergence_and_publishes_the_carry() -> None:
    """Totality: the completion reaches a maximum FAR from the terminal sample.

    This replaces the test that asserted a distant maximum is NOT captured. That
    was the reach gate, and the gate is what failed: MC20 -> MC12 overshoots by
    1.80 mesh pitches against a 1.0 pitch limit, and MC20 is the MIRROR PARTNER of
    MC17, which the gate did capture. A symmetric field was being answered
    asymmetrically -- the same class of failure Entry 046 caught when a disc gate
    split the symmetric maxima MAX_7 / MAX_4.

    Under the total rule the far case is committed and the whole carry is published
    as a drawn leg, so the cost is visible instead of the case being dropped.
    """
    pts = _fibonacci_sphere()
    f = _spike_field(pts, 2.5)
    # Port on the far side of the sphere: the ascent cannot get near the pole.
    port = np.array([math.sin(math.radians(150.0)), 0.0, math.cos(math.radians(150.0))])
    out = trace_uphill(
        start_position=tuple(port),
        targets=[_max_target(np.array([0.0, 0.0, 1.0]), radius_deg=1.0)],
        mesh_pts=pts,
        f_vals=f,
        sphere_radius=_R,
        delta_theta_mesh_rad=_PITCH,
        center=None,
    )
    assert out.status == "arrived"
    assert out.max_id == "MAX_0"
    assert out.completed_leg_deg is not None
    # The carry is long, and that is the point: it is reported, not rejected.
    assert out.completed_leg_deg > 10.0 * math.degrees(_PITCH)


def test_capture_pitch_is_an_opt_in_limit_not_the_default() -> None:
    """The gate is now a parameter callers may set; the default is total."""
    assert CAPTURE_PITCH == 1.0  # the documented opt-in limit
    pts = _fibonacci_sphere()
    f = _spike_field(pts, 2.5)
    port = np.array([math.sin(math.radians(150.0)), 0.0, math.cos(math.radians(150.0))])
    # A caller who wants the old behaviour can ask for it explicitly.
    gated = trace_uphill(
        start_position=tuple(port),
        targets=[_max_target(np.array([0.0, 0.0, 1.0]), radius_deg=1.0)],
        mesh_pts=pts,
        f_vals=f,
        sphere_radius=_R,
        delta_theta_mesh_rad=_PITCH,
        center=None,
        capture_pitch=CAPTURE_PITCH,
    )
    assert gated.status in ("stalled", "budget")
    assert gated.max_id is None


def test_capture_is_independent_of_the_transit_rule() -> None:
    """Overshoot is a property of the STEP, not of the boundary geometry.

    Capture must run when no excision regions are supplied, so a pure Step 4
    ascent gets the same completion.
    """
    pts = _fibonacci_sphere()
    f = _spike_field(pts, 2.5)
    port = np.array([math.sin(math.radians(9.0)), 0.0, math.cos(math.radians(9.0))])
    out = trace_uphill(
        start_position=tuple(port),
        targets=[_max_target(np.array([0.0, 0.0, 1.0]), radius_deg=1.0)],
        mesh_pts=pts,
        f_vals=f,
        sphere_radius=_R,
        delta_theta_mesh_rad=_PITCH,
        center=None,
        excision_regions=(),
    )
    assert out.status == "arrived"
    assert out.completed_leg_deg is not None


def test_a_broad_peak_arrives_under_its_own_steps_and_is_not_completed() -> None:
    """Ordinary arrivals are untouched: the leg is None, the path is all measured.

    This is the no-op contract. Every branch on Pd_20K and every clean arrival on
    ethene3 takes this path, so the rule cannot disturb a working field.
    """
    pts = _fibonacci_sphere()
    f = _gaussian_peak(pts, np.array([0.0, 0.0, 1.0]), math.radians(35.0), 1.0)
    port = np.array([0.6, 0.0, 0.8])
    port = port / np.linalg.norm(port)
    out = trace_uphill(
        start_position=tuple(port),
        targets=[_max_target(np.array([0.0, 0.0, 1.0]), radius_deg=8.0)],
        mesh_pts=pts,
        f_vals=f,
        sphere_radius=_R,
        delta_theta_mesh_rad=_PITCH,
        center=None,
    )
    assert out.status == "arrived"
    assert out.completed_leg_deg is None


def test_capture_pitch_is_one_mesh_step() -> None:
    """The bound is the step size, because the step is what causes the overshoot."""
    assert CAPTURE_PITCH == 1.0


def test_snap_to_entry_detects_boundary_crossing() -> None:
    """When an arc crosses a region boundary, it snaps to the center.

    This is the new entry-detection behavior: instead of spiraling inside the
    noisy excision region until termination, the arc snaps immediately upon entry.
    """
    pts = _fibonacci_sphere()
    # A field with a smooth peak that is reachable, and an annihilated region
    # that sits in the way
    f = _bump_and_peak_field(pts)
    # Start the port close to the bump so the path enters it
    port = np.array([math.sin(math.radians(80.0)), 0.0, math.cos(math.radians(80.0))])
    regions = _bump_region()

    out = trace_uphill(
        start_position=tuple(port),
        targets=[_max_target(np.array([0.0, 0.0, 1.0]), radius_deg=10.0)],
        mesh_pts=pts,
        f_vals=f,
        sphere_radius=_R,
        delta_theta_mesh_rad=_PITCH,
        center=None,
        excision_regions=regions,
    )
    # Should arrive at the maximum, not stall inside the region
    assert out.status == "arrived"
    assert out.max_id == "MAX_0"
    assert out.transited_ids == ("MC_TRAP",)
    # The path should be longer than a direct path because of the snap bridge
    assert out.n_steps > 10


def test_snap_bridge_is_drawn_in_path() -> None:
    """The straight-line bridge from crossing to center is included in the path.

    This ensures the visualizer can draw the snap-to-center behavior, and it
    records which region was transited.
    """
    pts = _fibonacci_sphere()
    f = _bump_and_peak_field(pts)
    port = np.array([math.sin(math.radians(80.0)), 0.0, math.cos(math.radians(80.0))])
    regions = _bump_region()
    region = regions[0]

    out = trace_uphill(
        start_position=tuple(port),
        targets=[_max_target(np.array([0.0, 0.0, 1.0]), radius_deg=10.0)],
        mesh_pts=pts,
        f_vals=f,
        sphere_radius=_R,
        delta_theta_mesh_rad=_PITCH,
        center=None,
        excision_regions=regions,
    )
    assert out.status == "arrived"
    # The transited record should name the region that was entered
    assert len(out.transited_ids) > 0
    assert out.transited_ids[0] == "MC_TRAP"
    # The path should have points that lie near the center
    path = np.asarray(out.path, dtype=np.float64)
    center_world = region.centre_unit * _R
    min_dist_to_center = np.min(np.linalg.norm(path - center_world, axis=1))
    # The closest point should be within a few steps of the center
    assert min_dist_to_center < _PITCH * _R * 3.0


def test_snap_resumes_from_boundary_maximum() -> None:
    """After snapping to center, the climb resumes from the boundary maximum.

    This combines the snap-to-entry logic with the boundary-maximum resume logic:
    the arc should snap at entry and then continue climbing from the boundary
    field maximum of that center.
    """
    pts = _fibonacci_sphere()
    f = _bump_and_peak_field(pts)
    port = np.array([math.sin(math.radians(80.0)), 0.0, math.cos(math.radians(80.0))])
    regions = _bump_region()
    region = regions[0]

    out = trace_uphill(
        start_position=tuple(port),
        targets=[_max_target(np.array([0.0, 0.0, 1.0]), radius_deg=10.0)],
        mesh_pts=pts,
        f_vals=f,
        sphere_radius=_R,
        delta_theta_mesh_rad=_PITCH,
        center=None,
        excision_regions=regions,
    )
    assert out.status == "arrived"
    # The boundary's field maximum (computed independently)
    ring = generate_boundary_ring(
        (
            float(region.centre_unit[0]),
            float(region.centre_unit[1]),
            float(region.centre_unit[2]),
        ),
        _R,
        region.radius_rad,
        RING_SAMPLES,
    )
    vals = [
        float(interpolate_scalar_at_points(np.array([p.position]), pts, f)[0])
        for p in ring
    ]
    argmax = np.asarray(ring[int(np.argmax(vals))].position, dtype=np.float64)

    # The resumed path must pass through the boundary maximum
    path = np.asarray(out.path, dtype=np.float64)
    dist = np.arccos(np.clip(path @ (argmax / np.linalg.norm(argmax)), -1.0, 1.0))
    assert math.degrees(float(dist.min())) < math.degrees(_PITCH)


# ---------------------------------------------------------------------------
# trace_downhill: the descent as the ascent of the negated field (Unit A,
# basin v10 — the saddle->min connectors the basin association consumes).
# ---------------------------------------------------------------------------


def _min_target(
    center_unit: npt.NDArray[np.float64],
    radius_deg: float = 8.0,
    cp_id: str = "MIN_0",
    value: float = 0.0,
) -> ConnectionTarget:
    """A minimum pool entry at ``center_unit`` with an 8-deg excision disc."""
    return ConnectionTarget(
        cp_id=cp_id,
        polarity="minimum",
        position=np.asarray(center_unit, dtype=np.float64),
        value=value,
        angular_radius_rad=math.radians(radius_deg),
    )


def test_descends_a_negated_peak() -> None:
    """A well (negated peak) is reached from a port on its slope; the path falls."""
    pts = _fibonacci_sphere()
    well = np.array([0.0, 0.0, 1.0])
    f = -_gaussian_peak(pts, well, width_rad=math.radians(35.0))
    port = np.array([0.6, 0.0, 0.8])
    port = port / np.linalg.norm(port)

    out = trace_downhill(
        start_position=port,
        targets=[_min_target(well)],
        mesh_pts=pts,
        f_vals=f,
        sphere_radius=_R,
        delta_theta_mesh_rad=_PITCH,
        center=None,
    )
    assert out.status == "arrived"
    assert out.max_id == "MIN_0"  # the field names the reached MINIMUM
    # start_value is the seed-ring maximum (the documented climb-resume
    # semantics, inherited through the flip); the descent pays it off:
    # the negated field ends above where the climb began.
    assert out.end_value > out.start_value
    assert len(out.path) == out.n_steps + 1


def test_downhill_ignores_maxima_as_arrival_targets() -> None:
    """The pool's maxima are visited, never arrived: polarity filters the test."""
    pts = _fibonacci_sphere()
    well = np.array([0.0, 0.0, 1.0])
    ridge_top = np.array([0.7, 0.0, 0.7])
    ridge_top = ridge_top / np.linalg.norm(ridge_top)
    f = -_gaussian_peak(pts, well, math.radians(35.0), 1.0) + _gaussian_peak(
        pts, ridge_top, math.radians(35.0), 0.6
    )
    # Port between the peak and the well, pointing downhill past the disc.
    port = np.array([0.35, 0.0, 0.94])
    port = port / np.linalg.norm(port)

    out = trace_downhill(
        start_position=port,
        targets=[
            _min_target(well, cp_id="MIN_WELL"),
            ConnectionTarget(
                cp_id="MAX_RIDGE",
                polarity="maximum",
                position=ridge_top,
                value=0.6,
                angular_radius_rad=math.radians(8.0),
            ),
        ],
        mesh_pts=pts,
        f_vals=f,
        sphere_radius=_R,
        delta_theta_mesh_rad=_PITCH,
        center=None,
    )
    assert out.status == "arrived"
    assert out.max_id == "MIN_WELL"
    # The maximum is not an arrival test on the descent, and a path entering
    # its disc records it as visited, never as the answer.
    assert out.max_id != "MAX_RIDGE"


def test_downhill_flat_plateau_stalls_without_capture() -> None:
    """A constant field has no gradient: the descent stalls, it does not crash.

    Capture is disabled so this tests the INTEGRATOR on the negated field.
    What the total completion rule does with the same stall is the next test.
    """
    pts = _fibonacci_sphere()
    f = np.zeros(len(pts), dtype=np.float64)
    port = np.array([0.6, 0.0, 0.8])
    port = port / np.linalg.norm(port)

    out = trace_downhill(
        start_position=port,
        targets=[_min_target(np.array([0.0, 0.0, 1.0]), value=-1.0)],
        mesh_pts=pts,
        f_vals=f,
        sphere_radius=_R,
        delta_theta_mesh_rad=_PITCH,
        center=None,
        capture_pitch=0.0,
    )
    assert out.status == "stalled"
    assert out.max_id is None
    assert out.n_steps <= STALL_PATIENCE + 2


def test_downhill_plateau_stall_is_committed_to_the_nearest_minimum() -> None:
    """The descent keeps the ascent's inviolable property: TOTAL commitment.

    The same flat field, capture at its default: the path has nothing to fall
    through, so it is committed to the nearest minimum and the carry is
    published as a drawn leg — the association is never silently dropped.
    """
    pts = _fibonacci_sphere()
    f = np.zeros(len(pts), dtype=np.float64)
    port = np.array([0.6, 0.0, 0.8])
    port = port / np.linalg.norm(port)  # 36.87 deg from the pole

    out = trace_downhill(
        start_position=port,
        targets=[_min_target(np.array([0.0, 0.0, 1.0]), value=-1.0)],
        mesh_pts=pts,
        f_vals=f,
        sphere_radius=_R,
        delta_theta_mesh_rad=_PITCH,
        center=None,
    )
    assert out.status == "arrived"
    assert out.max_id == "MIN_0"
    assert out.completed_leg_deg is not None
    assert out.completed_leg_deg == pytest.approx(36.87, abs=1.0)


def test_downhill_pool_without_minima_never_arrives() -> None:
    """Arrival tests are filtered by polarity, exactly as the ascent filters.

    A pool of maxima (kept as maxima by the flip) is unarrivable on a field
    with no gradient to fall down; the descent ends uncommitted, the one
    legitimate exception to totality.
    """
    pts = _fibonacci_sphere()
    f = np.zeros(len(pts), dtype=np.float64)

    out = trace_downhill(
        start_position=(0.0, 0.0, 1.0),
        targets=[
            ConnectionTarget(
                cp_id="MAX_1",
                polarity="maximum",
                position=np.array([0.0, 1.0, 0.0]),
                value=-1.0,
                angular_radius_rad=math.radians(6.0),
            )
        ],
        mesh_pts=pts,
        f_vals=f,
        sphere_radius=_R,
        delta_theta_mesh_rad=_PITCH,
        center=None,
    )
    assert out.status in ("stalled", "budget")
    assert out.max_id is None


def test_downhill_short_budget_is_closed_by_the_total_completion() -> None:
    """A budget that dies mid-field is COMMITTED, not dropped: the descent
    inherits the ascent's totality rule verbatim. The integrator travels its
    5-pitch budget, the completion leg carries the rest, and the number
    publishes the cost — a reach gate would have lost this arc silently.
    """
    pts = _fibonacci_sphere()
    well = np.array([0.0, 0.0, 1.0])
    f = -_gaussian_peak(pts, well, width_rad=math.radians(35.0))
    port = np.array([0.9, 0.0, 0.44])
    port = port / np.linalg.norm(port)  # ~64 deg from the well

    out = trace_downhill(
        start_position=port,
        targets=[_min_target(well, value=-1.0)],
        mesh_pts=pts,
        f_vals=f,
        sphere_radius=_R,
        delta_theta_mesh_rad=_PITCH,
        center=None,
        max_arc_pitch=5.0,  # ~5 mesh pitches: cannot cross 64 deg on its own
    )
    assert out.status == "arrived"
    assert out.max_id == "MIN_0"
    assert out.completed_leg_deg is not None
    # Integrated leg <= the 15 deg budget; the carry spans the rest of the 64.
    assert out.arc_length_deg - out.completed_leg_deg <= 15.0 + 1e-6
    assert out.completed_leg_deg > 40.0


def test_downhill_is_deterministic_and_stays_on_the_sphere() -> None:
    """The integrator is a pure function: repeat runs are identical, and every
    emitted sample sits on the sphere of radius R (the frame contract Unit B
    builds on)."""
    pts = _fibonacci_sphere()
    well = np.array([0.0, 0.0, 1.0])
    f = -_gaussian_peak(pts, well, width_rad=math.radians(35.0))
    port = np.array([0.6, 0.0, 0.8])
    port = port / np.linalg.norm(port)
    pool = [_min_target(well)]

    a = trace_downhill(port, pool, pts, f, _R, _PITCH)
    b = trace_downhill(port, pool, pts, f, _R, _PITCH)
    assert a == b
    for p in a.path:
        assert abs(math.hypot(*p) - _R) < 1e-9


def test_downhill_visited_ids_records_crossed_other_polarity() -> None:
    """A saddle disc the descent crosses is named in visited_ids, never an arrival.

    Saddle polarity survives the pool flip, so saddles keep the ascent's
    visit-not-arrive semantics on the way down — the transit record that makes
    "the path ran through that region" answerable from the data.
    """
    pts = _fibonacci_sphere()
    well = np.array([0.0, 0.0, 1.0])
    f = -_gaussian_peak(pts, well, width_rad=math.radians(35.0))
    port = np.array([0.55, 0.25, 0.8])
    port = port / np.linalg.norm(port)
    cross = np.array([0.3, 0.1, 0.95])
    cross = cross / np.linalg.norm(cross)

    out = trace_downhill(
        start_position=port,
        targets=[
            _min_target(well, cp_id="MIN_WELL"),
            ConnectionTarget(
                cp_id="SAD_CROSS",
                polarity="saddle",
                position=cross,
                value=float(
                    _gaussian_peak(np.array([cross * _R]), well, math.radians(35.0))[0]
                ),
                angular_radius_rad=math.radians(8.0),
            ),
        ],
        mesh_pts=pts,
        f_vals=f,
        sphere_radius=_R,
        delta_theta_mesh_rad=_PITCH,
        center=None,
    )
    assert out.status == "arrived"
    assert out.max_id == "MIN_WELL"
    assert "SAD_CROSS" in out.visited_ids


def test_trace_downhill_is_the_negated_ascent_of_the_same_pool() -> None:
    """Parity: trace_downhill(f) == trace_uphill(-f) with the pool relabelled.

    Pins the wrapper's contract (the docstring's whole argument): the descent
    is exactly the ascent of the negated field, so the integrator, the stall
    rule, the budget, the completion and the transit rule cannot drift.
    """
    pts = _fibonacci_sphere()
    well = np.array([0.0, 0.0, 1.0])
    f = -_gaussian_peak(pts, well, width_rad=math.radians(30.0))
    port = np.array([0.55, 0.25, 0.8])
    port = port / np.linalg.norm(port)
    pool = [_min_target(well, cp_id="MIN_A")]

    down = trace_downhill(
        start_position=port,
        targets=pool,
        mesh_pts=pts,
        f_vals=f,
        sphere_radius=_R,
        delta_theta_mesh_rad=_PITCH,
    )
    up = trace_uphill(
        start_position=port,
        targets=[dataclasses.replace(t, polarity="maximum") for t in pool],
        mesh_pts=pts,
        f_vals=-f,
        sphere_radius=_R,
        delta_theta_mesh_rad=_PITCH,
    )
    assert down.status == up.status == "arrived"
    assert down.max_id == up.max_id == "MIN_A"
    assert down.path == up.path
    assert down.arc_length_deg == up.arc_length_deg
    assert down.n_steps == up.n_steps
    assert down.start_value == up.start_value  # negated-field values both
