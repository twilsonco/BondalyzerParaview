"""Unit tests for the Maximum Connection subroutine (Basin Locator Step 3).

The rule under test: an ascent branch launches from its PORT on the saddle's
excision boundary, follows that great circle outward, and claims the FIRST
critical point whose excision DISC it enters, inside the forward sector. First
hit a maximum -> connect; a minimum or saddle -> occluded.

Two design points the fixtures encode, both learned from real fields:

- The ray starts at the port, not the saddle centre. The disc inside the
  boundary is the degenerate core whose field is meaningless, so nothing in it
  is a candidate. A small ring radius (RING_DEG) is used so a target's
  along-track is (colatitude - RING_DEG) on the meridian.
- A critical point is a DISC of radius angular_radius_rad, not a point. A great
  circle that passes outside the disc never hits it. This is the MC18 case: a
  branch was blocked by a minimum at cross-track 7.4 deg with a 1.6 deg disc --
  the ray sails past it, so it is not an obstacle, and the maximum at cross 0.4
  with a 2.9 deg disc is the true first hit.

Every fixture is a unit sphere with the saddle at the north pole, so a branch at
azimuth phi is the meridian phi. The frame tests run the same geometry offset
from the origin and require identical answers, because Pd_20K sits at the origin
and hides any centred/world mixup.
"""

import math
import random
from dataclasses import replace

import numpy as np
import pytest

from gba_topology2.src.basin_center import BasinCenter
from gba_topology2.src.max_connection import (
    ARC_SAMPLES,
    CONE_PITCH,
    CONNECTION_STATUSES,
    DIRECTION_MODES,
    ConnectionTarget,
    DirectionMode,
    DiscGeometry,
    MaxConnection,
    basin_centers_to_targets,
    build_connection_pool,
    connect_seed_to_maxima,
    connection_ray,
    degenerate_arcs,
    directional_support_radius,
    disc_entry_angle,
    mark_mutual,
    refine_connections,
    summarize_connections,
    target_frame_angles,
)
from gba_topology2.src.morse_detector import CriticalPoint
from gba_topology2.src.saddle_launch import LaunchBranch, SaddleLaunchSeed
from gba_topology2.src.separatrix_tracer import ExcisionRegion
from gba_topology2.src.sphere_frame import (
    angular_distance,
    radial_offset,
    to_global,
)

R = 1.0
PITCH = math.radians(2.0)
CENTER = (2.0, -1.5, 0.8)
# Excision ring radius used by the fixtures: small, so a target on the meridian
# at colatitude t has along-track (t - RING_DEG) from the port.
RING_DEG = 3.0
CONE_DEG = CONE_PITCH * math.degrees(PITCH)


def _pt(
    colat_deg: float, az_deg: float, radius: float = R
) -> tuple[float, float, float]:
    """Point on the sphere at a colatitude from +z and an azimuth from +x."""
    t = math.radians(colat_deg)
    p = math.radians(az_deg)
    return (
        radius * math.sin(t) * math.cos(p),
        radius * math.sin(t) * math.sin(p),
        radius * math.cos(t),
    )


def _branch(
    index: int,
    az_deg: float,
    kind: str = "ascent",
    ring_deg: float = RING_DEG,
) -> LaunchBranch:
    """An ascent branch whose port sits at (ring_deg, az_deg) from the pole."""
    pos = np.array(_pt(ring_deg, az_deg), dtype=np.float64)
    origin = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    tangent = pos - float(np.dot(pos, origin)) * origin
    tangent /= float(np.linalg.norm(tangent))
    return LaunchBranch(
        kind=kind,
        sector_index=index,
        tangent_direction=(float(tangent[0]), float(tangent[1]), float(tangent[2])),
        launch_position=(float(pos[0]), float(pos[1]), float(pos[2])),
        sector_width_rad=math.radians(60.0),
        sector_centroid_value=1.0,
    )


def _seed(
    saddle_id: str,
    branches: list[LaunchBranch],
    position: tuple[float, float, float] = (0.0, 0.0, 1.0),
    index_agrees: bool = True,
    ring_deg: float = RING_DEG,
) -> SaddleLaunchSeed:
    ascent = [b for b in branches if b.kind == "ascent"]
    descent = [b for b in branches if b.kind == "descent"]
    return SaddleLaunchSeed(
        saddle_id=saddle_id,
        saddle_position=position,
        saddle_value=0.5,
        vertex_id=-1,
        multiplicity=1,
        fold_order=2,
        index_agrees=index_agrees,
        ascent_branches=ascent,
        descent_branches=descent,
        ring_radius_rad=math.radians(ring_deg),
    )


def _max(
    center_id: str,
    colat_deg: float,
    az_deg: float,
    value: float = 2.0,
    center: tuple[float, float, float] | None = None,
) -> BasinCenter:
    pos = _pt(colat_deg, az_deg)
    world = to_global(pos, center) if center is not None else pos
    return BasinCenter(
        center_id=center_id,
        polarity="maximum",
        position=world,
        value=value,
        is_effective=False,
    )


def _target(
    cp_id: str,
    polarity: str,
    colat_deg: float,
    az_deg: float,
    value: float = 0.0,
    disc_deg: float = 0.0,
) -> ConnectionTarget:
    """A raw pool entry (a saddle/minimum) at a spherical position, with a disc."""
    return ConnectionTarget(
        cp_id=cp_id,
        polarity=polarity,
        position=np.array(_pt(colat_deg, az_deg), dtype=np.float64),
        value=value,
        angular_radius_rad=math.radians(disc_deg),
    )


def _pool(
    *maxima: BasinCenter,
    extra: list[ConnectionTarget] | None = None,
    disc_deg: float = 0.0,
) -> list[ConnectionTarget]:
    """Basin centers -> targets (with a uniform disc), plus extra targets."""
    rho = math.radians(disc_deg)
    out = basin_centers_to_targets(
        list(maxima),
        {
            m.center_id: DiscGeometry(
                angular_radius_rad=rho, semi_major_rad=rho, semi_minor_rad=rho
            )
            for m in maxima
        },
    )
    if extra:
        out.extend(extra)
    return out


# --- disc_entry_angle: the geometry primitive ---------------------------------


def test_disc_entry_is_the_first_touch_of_the_great_circle() -> None:
    """Entry = centre along-track minus the half-chord, and None when missed."""
    # Target dead on the ray (cross 0): entry is (along - rho).
    entry = disc_entry_angle(math.radians(30.0), 0.0, math.radians(5.0))
    assert entry is not None
    assert math.degrees(entry) == pytest.approx(25.0, abs=1e-9)

    # A great circle that passes outside the disc never hits it.
    assert (
        disc_entry_angle(math.radians(30.0), math.radians(8.0), math.radians(5.0))
        is None
    )

    # Exactly tangent: |cross| == rho grazes the disc, entry == along.
    tangent = disc_entry_angle(math.radians(30.0), math.radians(5.0), math.radians(5.0))
    assert tangent is not None
    assert math.degrees(tangent) == pytest.approx(30.0, abs=1e-9)

    # Zero disc radius degenerates to the centre point.
    point = disc_entry_angle(math.radians(30.0), math.radians(2.0), 0.0)
    assert point is not None
    assert math.degrees(point) == pytest.approx(30.0, abs=1e-9)


# --- ray origin: the port, not the saddle centre ------------------------------


def test_port_tangent_is_the_default_and_starts_at_the_port() -> None:
    """The default mode launches from the port on the excision boundary."""
    seed = _seed("SAD_1", [_branch(0, 0.0)])
    import inspect

    sig = inspect.signature(connect_seed_to_maxima)
    assert sig.parameters["direction_mode"].default == "port_tangent"

    ray = connection_ray(seed, seed.ascent_branches[0], "port_tangent", R)
    assert ray is not None
    # Origin is the port at colatitude RING_DEG, not the pole.
    assert math.degrees(
        target_frame_angles(np.array(_pt(0.0, 0.0)), ray)[0]
    ) == pytest.approx(-RING_DEG, abs=1e-9)


def test_target_inside_the_ring_is_behind_the_port_origin() -> None:
    """A CP inside the excision boundary has along < 0 from the port.

    This is the mechanism that makes the degenerate core non-occluding: the ray
    starts at the boundary, so anything the ring swallowed is behind it.
    """
    seed = _seed("SAD_1", [_branch(0, 0.0)])
    ray = connection_ray(seed, seed.ascent_branches[0], "port_tangent", R)
    assert ray is not None
    inside = np.array(_pt(RING_DEG * 0.5, 0.0))  # colatitude 1.5 < ring 3
    along, _ = target_frame_angles(inside, ray)
    assert along < 0.0


# --- first-hit on the disc ----------------------------------------------------


def test_first_hit_beats_farther_on_axis() -> None:
    """A nearer maximum on the ray wins over a far one with cross-track ~0.

    The MC109/MC8 case: a ray through a near maximum toward a far one has
    cross ~0 for both, so cross-track cannot separate them; along-track must.
    """
    seed = _seed("SAD_1", [_branch(0, 0.0)])
    pool = _pool(
        _max("MAX_NEAR", 15.0, 0.001),
        _max("MAX_FAR", 30.0, 0.0),
        disc_deg=2.0,
    )
    conns = connect_seed_to_maxima([seed], pool, R, PITCH)
    assert conns[0].max_id == "MAX_NEAR"
    assert conns[0].status == "connected"
    # Entry is (colatitude - ring - disc) = 15 - 3 - 2 = 10.
    assert conns[0].along_track_deg == pytest.approx(10.0, abs=0.1)


def test_minimum_disc_on_the_ray_occludes_the_maximum_behind_it() -> None:
    """A minimum whose disc the ray enters blocks the maximum behind it."""
    seed = _seed("SAD_1", [_branch(0, 0.0)])
    pool = _pool(
        _max(
            "MAX_BEYOND",
            30.0,
            0.0,
        ),
        extra=[_target("MIN_BLOCK", "minimum", 15.0, 0.0, disc_deg=2.0)],
    )
    conns = connect_seed_to_maxima([seed], pool, R, PITCH)
    assert conns[0].status == "occluded"
    assert conns[0].max_id is None
    assert conns[0].blocker_id == "MIN_BLOCK"
    assert conns[0].blocker_polarity == "minimum"


def test_saddle_disc_on_the_ray_occludes() -> None:
    """A saddle whose disc the ray enters is a separatrix crossing: occluded."""
    seed = _seed("SAD_1", [_branch(0, 0.0)])
    pool = _pool(
        _max("MAX_BEYOND", 30.0, 0.0),
        extra=[_target("SAD_BLOCK", "saddle", 12.0, 0.0, disc_deg=1.5)],
    )
    conns = connect_seed_to_maxima([seed], pool, R, PITCH)
    assert conns[0].status == "occluded"
    assert conns[0].blocker_id == "SAD_BLOCK"
    assert conns[0].blocker_polarity == "saddle"


def test_off_ray_minimum_does_not_occlude_the_max_on_it() -> None:
    """The MC18 case: a minimum off the ray is not an obstacle.

    A minimum at cross-track 7.4 deg with a 1.6 deg disc: the great circle
    passes outside the disc, so it is not a candidate. The maximum at cross 0.4
    with a 2.9 deg disc is the true first hit, even though the minimum is nearer
    to the saddle.
    """
    seed = _seed("SAD_1", [_branch(0, 0.0)])
    pool = _pool(
        _max("MAX_ON_AXIS", 45.0, 0.0),
        extra=[_target("MIN_OFF", "minimum", 15.0, 30.0, disc_deg=1.61)],
        disc_deg=2.91,
    )
    conns = connect_seed_to_maxima([seed], pool, R, PITCH)
    assert conns[0].status == "connected"
    assert conns[0].max_id == "MAX_ON_AXIS"
    assert conns[0].blocker_id is None


def test_first_hit_is_a_maximum_so_it_connects() -> None:
    """When the nearest disc the ray enters is a maximum, the branch connects."""
    seed = _seed("SAD_1", [_branch(0, 0.0)])
    pool = _pool(
        _max("MAX_NEAR", 15.0, 0.0),
        extra=[_target("SAD_FAR", "saddle", 25.0, 0.0, disc_deg=2.0)],
        disc_deg=2.0,
    )
    conns = connect_seed_to_maxima([seed], pool, R, PITCH)
    assert conns[0].status == "connected"
    assert conns[0].max_id == "MAX_NEAR"


def test_source_saddle_is_not_its_own_target() -> None:
    """A branch never claims the saddle it launched from (skipped by id)."""
    seed = _seed("SAD_1", [_branch(0, 0.0)])
    pool = _pool(
        _max("MAX_20", 20.0, 0.0),
        extra=[_target("SAD_1", "saddle", 10.0, 0.0, disc_deg=3.0)],
        disc_deg=2.0,
    )
    conns = connect_seed_to_maxima([seed], pool, R, PITCH)
    assert conns[0].status == "connected"
    assert conns[0].max_id == "MAX_20"


# --- the forward sector (cone) gate -------------------------------------------


def test_cone_gate_excludes_an_off_sector_near_target() -> None:
    """A nearer target outside the sector loses to a farther one inside it."""
    seed = _seed("SAD_1", [_branch(0, 0.0)])
    pool = _pool(
        _max("OFF_SECTOR", 25.0, 60.0),
        _max("ON_AXIS", 30.0, 0.0),
        disc_deg=2.0,
    )
    assert CONE_DEG < 21.5
    conns = connect_seed_to_maxima([seed], pool, R, PITCH)
    assert conns[0].max_id == "ON_AXIS"


def test_cone_gate_with_nothing_inside_is_no_target() -> None:
    """A branch whose only forward disc is off-sector finds nothing."""
    seed = _seed("SAD_1", [_branch(0, 0.0)])
    pool = _pool(_max("OFF_SECTOR", 25.0, 60.0), disc_deg=2.0)
    conns = connect_seed_to_maxima([seed], pool, R, PITCH)
    assert conns[0].status == "no_target"
    assert conns[0].blocker_id is None


def test_arc_cap_produces_no_target() -> None:
    """A maximum whose disc entry is past the cap is not claimed."""
    seed = _seed("SAD_1", [_branch(0, 0.0)])
    pool = _pool(_max("FAR", 20.0, 0.0), disc_deg=0.0)
    # entry = 20 - 3 = 17 deg; a 10 deg cap < 17. The cap is in DEGREES, not
    # mesh pitches, so the same test means the same thing on every mesh.
    conns = connect_seed_to_maxima([seed], pool, R, PITCH, max_arc_deg=10.0)
    assert conns[0].status == "no_target"
    assert conns[0].arc_points == ()
    assert connect_seed_to_maxima([seed], pool, R, PITCH)[0].max_id == "FAR"


def test_the_cap_is_angular_not_mesh_relative() -> None:
    """Reach is a property of the sphere, so it must not scale with the mesh.

    The bug this pins: the cap was 40 MESH PITCHES, which is 64 deg on a Pd_20K
    pitch and 89 deg on an ethene3 pitch. On ethene3 C1 the potential V the saddle
    MC14 and the maximum MC13 are 178 deg apart and BOTH ascent branches point at
    it dead centre (cross-track 0.02 deg), so a hemisphere cap rejected a ray that
    had found its target perfectly. The same geometry, run at two different mesh
    pitches, must now give the same verdict.
    """
    seed = _seed("SAD_1", [_branch(0, 0.0)])
    # A maximum 150 deg out on the ray: far, aligned, and reachable by a geodesic.
    pool = _pool(_max("ANTIPODAL", 150.0, 0.0), disc_deg=2.0)
    fine = math.radians(1.6058)  # Pd_20K pitch
    coarse = math.radians(2.2278)  # ethene3 pitch
    a = connect_seed_to_maxima([seed], pool, R, fine)[0]
    b = connect_seed_to_maxima([seed], pool, R, coarse)[0]
    assert a.status == b.status == "connected"
    assert a.max_id == b.max_id == "ANTIPODAL"
    # And the old pitch-relative cap would have failed the coarse mesh: 40
    # pitches there is 89 deg, short of the 147 deg disc entry.
    assert (
        connect_seed_to_maxima([seed], pool, R, coarse, max_arc_deg=89.11)[0].status
        == "no_target"
    )


def test_rear_hemisphere_is_never_claimed() -> None:
    """A maximum behind the port is not a forward hit."""
    seed = _seed("SAD_1", [_branch(0, 0.0)])
    pool = _pool(_max("BEHIND", 160.0, 180.0), disc_deg=2.0)
    assert connect_seed_to_maxima([seed], pool, R, PITCH)[0].status == "no_target"


def test_opposite_branches_claim_opposite_maxima() -> None:
    """Two branches 180 deg apart must not collapse onto the same extremum."""
    seed = _seed("SAD_1", [_branch(0, 0.0), _branch(1, 180.0)])
    pool = _pool(_max("EAST", 25.0, 3.0), _max("WEST", 25.0, 177.0), disc_deg=2.0)
    conns = connect_seed_to_maxima([seed], pool, R, PITCH)
    assert [c.max_id for c in conns] == ["EAST", "WEST"]


def test_only_ascent_branches_are_connected() -> None:
    """Descent branches head for minima and are excluded from the table."""
    seed = _seed("SAD_1", [_branch(0, 0.0), _branch(0, 90.0, kind="descent")])
    pool = _pool(_max("A", 20.0, 0.0), _max("B", 20.0, 90.0), disc_deg=2.0)
    conns = connect_seed_to_maxima([seed], pool, R, PITCH)
    assert len(conns) == 1
    assert conns[0].max_id == "A"


# --- the two-part arc ---------------------------------------------------------


def test_arc_is_the_single_geodesic_between_the_two_centres() -> None:
    """The connector runs centre-to-centre, not through the port and boundary.

    The search (port ray -> boundary entry -> cone) is a solver whose job is to
    find the two endpoints. Once found, the connector is the single great circle
    between them, so it starts at the SADDLE CENTRE, not at the port.
    """
    seed = _seed("SAD_1", [_branch(0, 0.0)])
    pool = _pool(_max("T", 30.0, 0.0), disc_deg=2.0)
    conns = connect_seed_to_maxima([seed], pool, R, PITCH)
    arc = np.array(conns[0].arc_points)
    # First point is the saddle centre (the pole), NOT the port at RING_DEG.
    assert np.linalg.norm(arc[0] - np.array([0.0, 0.0, 1.0])) < 1e-12
    # Last point is the maximum's centre.
    assert np.linalg.norm(arc[-1] - np.array(_pt(30.0, 0.0))) < 1e-9
    assert np.allclose(np.linalg.norm(arc, axis=1), 1.0, atol=1e-9)


def test_arc_is_a_great_circle() -> None:
    """All samples share one plane through the centre: zero torsion.

    The two-part connector (ray to boundary, then geodesic to centre) was
    deliberately NOT a single great circle, so this assertion was dropped in
    Entry 046. The final connector is a single geodesic and the property is back.
    """
    seed = _seed("SAD_1", [_branch(0, 15.0)])
    conns = connect_seed_to_maxima([seed], _pool(_max("T", 40.0, 15.0)), R, PITCH)
    pts = np.array(conns[0].arc_points)
    normal = np.cross(pts[1] - pts[0], pts[-1] - pts[0])
    normal /= float(np.linalg.norm(normal))
    # A great circle through the origin: every point is perpendicular to the
    # plane normal (offset 0, not merely constant).
    assert np.max(np.abs(pts @ normal)) < 1e-9


def test_arc_has_no_kink_at_the_maximum_boundary() -> None:
    """The sharp turn the user reported is gone: consecutive steps stay collinear.

    The three-segment path bent hard where the launch ray met the excision
    boundary, because the first leg followed the launch direction and the second
    cut to the centre. The angular step of a uniformly sampled geodesic is
    span / ARC_SAMPLES; a kink shows up as one step far larger than the rest.
    """
    seed = _seed("SAD_1", [_branch(0, 0.0)])
    # A wide disc and a ray that meets it well off-axis: the old second leg had
    # to turn sharply to reach the centre.
    pool = _pool(_max("T", 30.0, 6.0), disc_deg=9.0)
    conns = connect_seed_to_maxima([seed], pool, R, PITCH)
    pts = np.array(conns[0].arc_points)
    dirs = pts[1:] - pts[:-1]
    dirs = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)
    kinks = [
        math.degrees(math.acos(float(np.clip(np.dot(dirs[i], dirs[i + 1]), -1, 1))))
        for i in range(len(dirs) - 1)
    ]
    uniform = 30.0 / ARC_SAMPLES  # the expected per-step angle, centre to centre
    assert max(kinks) < uniform * 1.5, (max(kinks), uniform)


def test_arc_points_lie_on_the_sphere_world_frame() -> None:
    """Every rendered sample sits at radius R from the nucleus, in world frame."""
    seed = _seed("SAD_1", [_branch(0, 35.0)])
    pool = _pool(_max("T", 30.0, 35.0, center=CENTER), disc_deg=2.0)
    conns = connect_seed_to_maxima([seed], pool, R, PITCH, center=CENTER)
    for p in conns[0].arc_points:
        assert float(np.linalg.norm(np.array(p) - np.array(CENTER))) == pytest.approx(
            R, rel=1e-9
        )


# --- pool construction (the frame trap) ---------------------------------------


def test_build_connection_pool_merges_extrema_and_saddles() -> None:
    """The pool carries maxima, minima AND saddles with their disc radii."""
    maxima = [_max("MAX_1", 20.0, 0.0)]
    minima = [
        BasinCenter(
            center_id="MIN_1",
            polarity="minimum",
            position=_pt(40.0, 0.0),
            value=-1.0,
            is_effective=False,
        )
    ]
    raw_sad = [
        CriticalPoint(vertex_id=7, cp_type="saddle", value=0.3, position=_pt(30.0, 0.0))
    ]
    pool = build_connection_pool(
        [*maxima, *minima],
        raw_sad,
        [],
        set(),
        disc_by_vertex={
            7: DiscGeometry(
                angular_radius_rad=math.radians(4.0),
                semi_major_rad=math.radians(4.0),
                semi_minor_rad=math.radians(4.0),
            )
        },
    )
    by_id = {t.cp_id: (t.polarity, t.angular_radius_rad) for t in pool}
    assert by_id["SAD_7"][0] == "saddle"
    assert math.degrees(by_id["SAD_7"][1]) == pytest.approx(4.0)


def test_build_connection_pool_excludes_hidden_raw_saddles() -> None:
    """A raw saddle collapsed into an effective entity is not visible."""
    raw_sad = [
        CriticalPoint(
            vertex_id=7, cp_type="saddle", value=0.3, position=_pt(30.0, 0.0)
        ),
        CriticalPoint(
            vertex_id=8, cp_type="saddle", value=0.3, position=_pt(35.0, 0.0)
        ),
    ]
    pool = build_connection_pool([], raw_sad, [], hidden_vertex_ids={8})
    assert {t.cp_id for t in pool} == {"SAD_7"}


def test_build_connection_pool_includes_effective_saddles() -> None:
    """E-SAD dicts (world frame, no vertex) enter the pool as saddles."""
    eff = [
        {
            "type": "E-SAD",
            "id_label": "E-SAD_MC7",
            "position": list(_pt(25.0, 0.0)),
            "value": 0.4,
        },
        {
            "type": "E-MAX",
            "id_label": "E-MAX_MC9",
            "position": list(_pt(50.0, 0.0)),
            "value": 1.0,
        },
    ]
    pool = build_connection_pool(
        [],
        [],
        eff,
        set(),
        disc_by_id={
            "E-SAD_MC7": DiscGeometry(
                angular_radius_rad=math.radians(8.0),
                semi_major_rad=math.radians(8.0),
                semi_minor_rad=math.radians(8.0),
            )
        },
    )
    by_id = {t.cp_id: t.polarity for t in pool}
    assert by_id == {"E-SAD_MC7": "saddle"}


def test_build_connection_pool_maps_raw_saddles_to_world_frame() -> None:
    """Raw saddles are centred; the pool is world frame. Checked with an offset."""
    raw_sad = [
        CriticalPoint(vertex_id=7, cp_type="saddle", value=0.3, position=_pt(30.0, 0.0))
    ]
    pool = build_connection_pool([], raw_sad, [], set(), center=CENTER)
    p = pool[0].position
    assert float(np.linalg.norm(p - np.array(CENTER))) == pytest.approx(R, rel=1e-9)


def test_build_connection_pool_dedups_by_id() -> None:
    """A saddle present in two sources is scored once."""
    raw_sad = [
        CriticalPoint(vertex_id=7, cp_type="saddle", value=0.3, position=_pt(30.0, 0.0))
    ]
    eff = [
        {
            "type": "E-SAD",
            "id_label": "SAD_7",
            "position": list(_pt(30.0, 0.0)),
            "value": 0.3,
        }
    ]
    pool = build_connection_pool([], raw_sad, eff, set())
    assert len([t for t in pool if t.cp_id == "SAD_7"]) == 1


# --- mutual, purity, determinism ----------------------------------------------


def test_mutual_flags_exactly_one_claim_per_maximum() -> None:
    """The tightest claim on a maximum is the mutual one, and only one is."""
    seeds = [
        _seed("SAD_A", [_branch(0, 0.0)], position=_pt(20.0, 0.0)),
        _seed("SAD_B", [_branch(0, 0.0)], position=_pt(60.0, 0.0)),
    ]
    pool = _pool(_max("TARGET", 20.0, 0.0), disc_deg=2.0)
    conns = connect_seed_to_maxima(seeds, pool, R, PITCH)
    connected = [c for c in conns if c.status == "connected"]
    assert len(connected) >= 1
    mutual = [c for c in conns if c.mutual]
    assert len(mutual) == len({c.max_id for c in connected})


def test_mark_mutual_preserves_blocker_fields() -> None:
    """mark_mutual rebuilds records; it must not drop the occlusion fields."""
    seed = _seed("SAD_1", [_branch(0, 0.0)])
    pool = _pool(
        _max("MAX_BEYOND", 30.0, 0.0),
        extra=[_target("MIN_BLOCK", "minimum", 15.0, 0.0, disc_deg=2.0)],
    )
    conns = mark_mutual(connect_seed_to_maxima([seed], pool, R, PITCH))
    assert conns[0].blocker_id == "MIN_BLOCK"
    assert conns[0].blocker_polarity == "minimum"


def test_world_frame_and_centred_runs_agree() -> None:
    """Offsetting the atom changes no metric -- the frame contract, pinned."""
    seed = _seed("SAD_1", [_branch(0, 35.0)])
    pool_local = _pool(_max("T", 30.0, 35.0), _max("OFF", 12.0, 120.0), disc_deg=2.0)
    pool_world = _pool(
        _max("T", 30.0, 35.0, center=CENTER),
        _max("OFF", 12.0, 120.0, center=CENTER),
        disc_deg=2.0,
    )
    a = connect_seed_to_maxima([seed], pool_local, R, PITCH)
    b = connect_seed_to_maxima([seed], pool_world, R, PITCH, center=CENTER)
    assert a[0].max_id == b[0].max_id
    assert a[0].cross_track_deg == pytest.approx(b[0].cross_track_deg, abs=1e-9)
    assert a[0].along_track_deg == pytest.approx(b[0].along_track_deg, abs=1e-9)


def test_result_is_independent_of_input_order() -> None:
    """Shuffling the pool and the seeds cannot change any claim."""
    seed = _seed("SAD_1", [_branch(0, 0.0), _branch(1, 120.0)])
    seed2 = _seed("SAD_2", [_branch(0, 240.0)], position=_pt(5.0, 240.0))
    pool = _pool(
        _max("A", 20.0, 2.0),
        _max("B", 25.0, 121.0),
        _max("C", 30.0, 241.0),
        _max("D", 12.0, 60.0),
        disc_deg=2.0,
    )
    reference = connect_seed_to_maxima([seed, seed2], pool, R, PITCH)

    rng = random.Random(7)
    for _ in range(8):
        shuffled = pool[:]
        rng.shuffle(shuffled)
        seeds = [seed, seed2][:]
        rng.shuffle(seeds)
        got = connect_seed_to_maxima(seeds, shuffled, R, PITCH)
        key = {
            (c.saddle_id, c.branch_index): (c.max_id, c.status, c.mutual) for c in got
        }
        want = {
            (c.saddle_id, c.branch_index): (c.max_id, c.status, c.mutual)
            for c in reference
        }
        assert key == want


def test_direction_modes_are_both_selectable() -> None:
    """Both modes run; the mode is recorded on every connection."""
    seed = _seed("SAD_1", [_branch(0, 20.0)])
    pool = _pool(_max("T", 30.0, 20.0), disc_deg=2.0)
    modes: tuple[DirectionMode, ...] = ("chord", "port_tangent")
    assert set(modes) == set(DIRECTION_MODES)
    for mode in modes:
        conns = connect_seed_to_maxima([seed], pool, R, PITCH, direction_mode=mode)
        assert conns[0].direction_mode == mode
        assert conns[0].max_id == "T"

    with pytest.raises(ValueError):
        connect_seed_to_maxima([seed], pool, R, PITCH, direction_mode="bogus")  # type: ignore[arg-type]


def test_degenerate_seed_position_yields_no_target() -> None:
    """A saddle at the nucleus has no radial, hence no ray."""
    seed = _seed("SAD_1", [_branch(0, 0.0)], position=(0.0, 0.0, 0.0))
    assert (
        connect_seed_to_maxima([seed], _pool(_max("T", 20.0, 0.0)), R, PITCH)[0].status
        == "no_target"
    )


def test_empty_inputs() -> None:
    """No seeds or no targets produce an empty table, not an exception."""
    assert connect_seed_to_maxima([], [], R, PITCH) == []
    seed = _seed("SAD_1", [_branch(0, 0.0)])
    assert connect_seed_to_maxima([seed], [], R, PITCH)[0].status == "no_target"


def test_summarize_connections_counts_all_three_statuses() -> None:
    """The summary separates connected / occluded / no_target, all finite."""
    seed = _seed(
        "SAD_1",
        [_branch(0, 0.0), _branch(1, 90.0), _branch(2, 200.0)],
    )
    pool = _pool(
        _max("M0", 25.0, 1.0),
        extra=[_target("MIN90", "minimum", 25.0, 90.0, disc_deg=2.0)],
        disc_deg=2.0,
    )
    stats = summarize_connections(connect_seed_to_maxima([seed], pool, R, PITCH))
    assert stats["n_branches"] == 3
    assert stats["n_connected"] == 1
    assert stats["n_occluded"] == 1
    assert stats["n_no_target"] == 1
    assert stats["n_blocked_by_minimum"] == 1
    assert stats["n_blocked_by_saddle"] == 0
    for key, value in stats.items():
        if isinstance(value, float):
            assert math.isfinite(value), key

    empty = summarize_connections([])
    assert empty["n_branches"] == 0
    assert empty["n_occluded"] == 0


def test_verified_flag_tracks_index_agrees() -> None:
    """Provisional (k_mismatch) seeds stay distinguishable in the table."""
    good = _seed("SAD_1", [_branch(0, 0.0)], index_agrees=True)
    bad = _seed("SAD_2", [_branch(0, 0.0)], index_agrees=False)
    conns = connect_seed_to_maxima(
        [good, bad], _pool(_max("T", 20.0, 0.0), disc_deg=2.0), R, PITCH
    )
    assert [c.verified for c in conns] == [True, False]
    assert summarize_connections(conns)["n_unverified"] == 1


def test_connection_statuses_are_the_three_verdicts() -> None:
    assert set(CONNECTION_STATUSES) == {"connected", "occluded", "no_target"}


# --- cone captures the target, disc blocks only real obstacles ----------------


def test_maximum_is_captured_by_the_cone_not_by_its_own_disc() -> None:
    """A ray that misses a maximum's boundary still connects to it.

    The regression the user reported. Every raw maximum's excision radius is one
    mesh pitch (1.61 deg on Pd_20K), while legitimate rays into a 3-saddle
    maximum arrive at cross-tracks 0.0, 1.35, 1.42 on one maximum and 0.0, 2.00,
    1.98 on its symmetric twin. Requiring the ray to hit the maximum's own disc
    put the gate inside the noise band, so the symmetric pattern produced 3
    connections on one and 1 on the other. The cone is the directional criterion
    for a TARGET; the disc is only what an OBSTACLE must be entered.
    """
    seed = _seed("SAD_1", [_branch(0, 0.0)])
    # MAX_OFF arrives at cross 2.0 with a 1.61 deg disc: outside its boundary.
    pool = _pool(_max("MAX_OFF", 20.81, 5.5), disc_deg=1.61)
    conns = connect_seed_to_maxima([seed], pool, R, PITCH)
    assert conns[0].status == "connected"
    assert conns[0].max_id == "MAX_OFF"
    assert abs(conns[0].cross_track_deg or 0.0) > 1.61


def test_symmetric_saddles_all_claim_one_maximum() -> None:
    """Three rays arriving at cross-track 0.0, 1.4, 2.0 all connect.

    The invariant that was violated. On Pd_20K shape index the eight raw maxima
    sit in a symmetric pattern, each ringed by three saddles at 19.5 deg, and
    the three rays into a maximum arrive at cross-tracks 0.0, 1.35, 1.42 on
    MAX_7 and 0.0, 2.00, 1.98 on MAX_4. Requiring the ray to enter the maximum's
    own 1.61 deg disc accepted MAX_7's three and rejected two of MAX_4's: the
    gate sat inside the band the field's own noise varies across, so identical
    geometry gave different answers.

    The 2.0 case is the one that pins the fix -- under the disc gate it is
    rejected, because 1.61 < 2.0.
    """
    seed = _seed("SAD_1", [_branch(0, 0.0)])
    # Azimuth chosen so asin(sin(colat) * sin(az)) hits the target cross-track.
    colat = 20.81
    sin_t = math.sin(math.radians(colat))
    results = []
    for want_cross in (0.0, 1.4, 2.0):
        az = math.degrees(math.asin(math.sin(math.radians(want_cross)) / sin_t))
        target = _target("TARGET", "maximum", colat, az, disc_deg=1.61)
        got = connect_seed_to_maxima([seed], [target], R, PITCH)[0]
        results.append((round(abs(got.cross_track_deg or 0.0), 2), got.status))
    assert [s for _, s in results] == ["connected"] * 3, results
    assert [round(c, 1) for c, _ in results] == [0.0, 1.4, 2.0], results


def test_obstacle_still_needs_its_disc_to_be_entered() -> None:
    """The disc test is retained for obstacles: a near-miss minimum is inert."""
    seed = _seed("SAD_1", [_branch(0, 0.0)])
    # Minimum at cross 7.4 with a 1.61 deg disc -- the ray sails past it.
    pool = _pool(
        _max("MAX_BEYOND", 45.0, 0.0),
        extra=[_target("MIN_NEAR", "minimum", 15.0, 30.0, disc_deg=1.61)],
        disc_deg=2.91,
    )
    conns = connect_seed_to_maxima([seed], pool, R, PITCH)
    assert conns[0].status == "connected"
    assert conns[0].max_id == "MAX_BEYOND"


def test_ranking_key_is_the_boundary_entry_when_the_ray_hits_the_disc() -> None:
    """along_track_deg records where the ray meets the basin, not the centre.

    The boundary entry is the search's ranking key and a diagnostic of how far
    the ray travels before it reaches the basin. It is NOT a vertex of the drawn
    connector, which runs centre to centre.
    """
    seed = _seed("SAD_1", [_branch(0, 0.0)])
    pool = _pool(_max("T", 30.0, 0.0), disc_deg=5.0)
    conns = connect_seed_to_maxima([seed], pool, R, PITCH)
    # Centre along-track is 30 - 3 = 27; boundary entry is 27 - 5 = 22.
    assert conns[0].along_track_deg == pytest.approx(22.0, abs=0.2)
    # The connector itself spans the full centre-to-centre separation.
    assert conns[0].separation_deg == pytest.approx(30.0, abs=0.2)


def test_ranking_key_is_the_centre_when_the_ray_misses_the_disc() -> None:
    """A ray that never enters the disc ranks on the centre's along-track."""
    seed = _seed("SAD_1", [_branch(0, 0.0)])
    pool = _pool(_max("T", 30.0, 4.0), disc_deg=1.61)
    conns = connect_seed_to_maxima([seed], pool, R, PITCH)
    assert conns[0].status == "connected"
    # No boundary crossing, so the key is the centre at 30 - 3 = 27.
    assert conns[0].along_track_deg == pytest.approx(27.0, abs=0.2)


def test_directional_support_radius_is_transverse_for_a_corridor() -> None:
    """An elongated boundary is narrow across and long along its axis."""
    a, b = math.radians(11.35), math.radians(2.43)
    centroid = np.array([0.0, 0.0, 1.0])
    axis = np.array([1.0, 0.0, 0.0])
    along_axis = directional_support_radius(
        centroid, a, b, axis, np.array([1.0, 0.0, 0.0])
    )
    across = directional_support_radius(centroid, a, b, axis, np.array([0.0, 1.0, 0.0]))
    assert math.degrees(along_axis) == pytest.approx(11.35, abs=1e-6)
    assert math.degrees(across) == pytest.approx(2.43, abs=1e-6)


def test_target_without_an_axis_is_isotropic() -> None:
    """A compact set (no fitted axis) falls back to the isotropic radius."""
    t = _target("X", "minimum", 20.0, 0.0, disc_deg=3.0)
    assert math.degrees(t.radius_toward(np.array([1.0, 0.0, 0.0]))) == pytest.approx(
        3.0
    )


def test_lifted_arc_endpoints_stay_radial() -> None:
    """radial_offset on the emitted arc is a true lift in the centred frame."""
    seed = _seed("SAD_1", [_branch(0, 35.0)])
    conns = connect_seed_to_maxima(
        [seed], _pool(_max("T", 30.0, 35.0), disc_deg=2.0), R, PITCH
    )
    lifted = [radial_offset(p, 1.02) for p in conns[0].arc_points]
    assert all(
        float(np.linalg.norm(np.array(p))) == pytest.approx(R * 1.02, rel=1e-12)
        for p in lifted
    )


# ---------------------------------------------------------------------------
# Step 4: the gradient fallback.
#
# A branch whose great-circle ray reaches nothing (the maximum lies behind the
# port, so along-track is negative) is INCOMPLETE. The fallback climbs the field
# from the port. These fixtures build a synthetic peak and check that the ray
# fails where the ascent succeeds, that the record is labelled method="gradient"
# with the gate that motivated it, and that omitting the field leaves the pure
# geodesic rule untouched (the Pd_20K no-op contract).
# ---------------------------------------------------------------------------


def _mesh_and_peak(
    peak_unit: tuple[float, float, float],
    width_deg: float = 30.0,
    n: int = 1200,
) -> tuple[np.ndarray, np.ndarray]:
    """A fibonacci mesh (centred, unit sphere) with one smooth peak."""
    idx = np.arange(n, dtype=float) + 0.5
    phi = np.arccos(1.0 - 2.0 * idx / n)
    theta = math.pi * (1.0 + 5.0**0.5) * idx
    pts = np.column_stack(
        [np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)]
    )
    p = np.array(peak_unit, dtype=np.float64)
    p = p / np.linalg.norm(p)
    d = np.arccos(np.clip(pts @ p, -1.0, 1.0))
    return pts, np.exp(-((d / math.radians(width_deg)) ** 2))


def test_gradient_fallback_finishes_a_behind_ray_branch() -> None:
    """Peak behind the port: the ray sees nothing, the climb arrives."""
    seed = _seed("SAD_1", [_branch(0, 0.0)])
    # Maximum at the north pole; the port at colat 3 deg launches outward, so the
    # pole is behind the ray (along < 0). The field peaks at the pole.
    targets = [_target("MAX_9", "maximum", 0.0, 0.0, value=1.0, disc_deg=2.0)]
    pts, f = _mesh_and_peak((0.0, 0.0, 1.0))

    geodesic = connect_seed_to_maxima([seed], targets, R, PITCH)
    assert geodesic[0].status == "no_target"
    assert geodesic[0].method == "geodesic"

    with_field = connect_seed_to_maxima(
        [seed], targets, R, PITCH, mesh_pts=pts, f_vals=f
    )
    conn = with_field[0]
    assert conn.status == "connected"
    assert conn.max_id == "MAX_9"
    assert conn.method == "gradient"
    assert conn.failure_gate == "behind"
    # The connector is the traced polyline, not a great circle: it has as many
    # points as the ascent took steps (plus the start).
    assert len(conn.arc_points) >= 2


def test_gradient_fallback_records_gate_when_it_stalls() -> None:
    """No maximum to reach: the record keeps the gate AND the path it travelled.

    The stalled climb is the answer to the question the operator asked -- follow
    the uphill direction from the excision boundary to its highest point and stop.
    So the record is unresolved (no max_id, status no_target) but carries the ridge
    line and its endpoint, labelled method="gradient" because the geometry was
    traced, not drawn.

    ``capture_pitch=0.0`` isolates the Step 4 RECORD contract from the completion
    rule. Under the default the same ascent is committed to the maximum it
    converged on, which is the next test's subject.
    """
    seed = _seed("SAD_1", [_branch(0, 0.0)])
    targets = [_target("MAX_9", "maximum", 0.0, 0.0, value=1.0, disc_deg=2.0)]
    # A field with a broad shoulder and no maximum in the pool: the climb travels
    # and stalls, so the path has real length.
    idx = np.arange(600, dtype=float) + 0.5
    phi = np.arccos(1.0 - 2.0 * idx / 600)
    theta = math.pi * (1.0 + 5.0**0.5) * idx
    pts = np.column_stack(
        [np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)]
    )
    # A peak at the SOUTH pole: the port at colat 3 launches outward (northward),
    # so the ray's gate is "behind", and the climb runs uphill across the sphere.
    south = pts @ np.array([0.0, 0.0, -1.0])
    f = np.exp(-((np.arccos(np.clip(south, -1.0, 1.0)) / 1.2) ** 2))

    conns = connect_seed_to_maxima(
        [seed], targets, R, PITCH, mesh_pts=pts, f_vals=f, capture_pitch=0.0
    )
    conn = conns[0]
    assert conn.status == "no_target"
    assert conn.failure_gate == "behind"
    assert conn.max_id is None
    # The path survives: this is the whole point of the change.
    assert len(conn.arc_points) >= 2
    assert conn.stall_point is not None
    assert conn.method == "gradient"
    assert conn.along_track_deg is not None and conn.along_track_deg > 0.0
    # The stall point is the path's last vertex, on the sphere.
    assert conn.stall_point == conn.arc_points[-1]
    assert float(np.linalg.norm(np.array(conn.stall_point))) == pytest.approx(
        1.0, rel=1e-6
    )


def test_a_stalled_gradient_path_is_committed_by_default() -> None:
    """THE INVARIANT, at the table level: no gradient path is left uncommitted.

    The same fixture as the record-contract test, with the completion at its
    default. The ascent that stalled short of the maximum is now committed to it,
    the connector terminates at the maximum's centre, and the carry is published as
    a drawn leg. This is the case a reach gate dropped: the gate was one mesh pitch
    because the first example overshot by 0.53, and the mirror-partner case overshot
    by 1.80, so a symmetric field was answered asymmetrically.
    """
    seed = _seed("SAD_1", [_branch(0, 0.0)])
    targets = [_target("MAX_9", "maximum", 0.0, 0.0, value=1.0, disc_deg=2.0)]
    idx = np.arange(600, dtype=float) + 0.5
    phi = np.arccos(1.0 - 2.0 * idx / 600)
    theta = math.pi * (1.0 + 5.0**0.5) * idx
    pts = np.column_stack(
        [np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)]
    )
    south = pts @ np.array([0.0, 0.0, -1.0])
    f = np.exp(-((np.arccos(np.clip(south, -1.0, 1.0)) / 1.2) ** 2))

    conn = connect_seed_to_maxima([seed], targets, R, PITCH, mesh_pts=pts, f_vals=f)[0]
    assert conn.status == "connected"
    assert conn.max_id == "MAX_9"
    assert conn.method == "gradient"
    # The gate is retained: it records why the GEODESIC failed, which is
    # independent of how the gradient path was finished.
    assert conn.failure_gate == "behind"
    # The connector ends AT the maximum, and the carry is published.
    assert conn.completed_leg_deg is not None and conn.completed_leg_deg > 0.0
    assert conn.stall_point is None
    end = np.asarray(conn.arc_points[-1], dtype=np.float64)
    assert float(np.linalg.norm(end)) == pytest.approx(R, rel=1e-6)
    # The summary reports the invariant, and it holds.
    stats = summarize_connections([conn])
    assert stats["n_completed"] == 1
    assert stats["n_gradient_uncommitted"] == 0
    assert stats["max_completed_leg_deg"] == pytest.approx(conn.completed_leg_deg)


def test_occluded_branch_is_not_rescued_by_the_gradient() -> None:
    """A branch blocked by a disc is a resolved verdict, not an incomplete one.

    The fallback runs only when the ray reaches NOTHING. A ray that flies over a
    minimum's disc has an answer (occluded); climbing the field from the port would
    contradict the geodesic rule that produced the occlusion.
    """
    seed = _seed("SAD_1", [_branch(0, 0.0)])
    # A minimum squarely on the outward ray, disc large enough to be entered.
    targets = [
        _target("MIN_1", "minimum", 20.0, 0.0, disc_deg=6.0),
        _target("MAX_9", "maximum", 40.0, 0.0, value=1.0, disc_deg=4.0),
    ]
    pts, f = _mesh_and_peak((0.0, 0.0, 1.0))
    conns = connect_seed_to_maxima([seed], targets, R, PITCH, mesh_pts=pts, f_vals=f)
    assert conns[0].status == "occluded"
    assert conns[0].blocker_id == "MIN_1"
    assert conns[0].method == "geodesic"
    assert conns[0].failure_gate == "occluded"


def test_no_field_leaves_the_geodesic_rule_untouched() -> None:
    """Without mesh_pts/f_vals the fallback is inert -- the Pd_20K no-op contract."""
    seed = _seed("SAD_1", [_branch(0, 0.0)])
    targets = [_target("MAX_9", "maximum", 0.0, 0.0, value=1.0, disc_deg=2.0)]
    pts, f = _mesh_and_peak((0.0, 0.0, 1.0))
    a = connect_seed_to_maxima([seed], targets, R, PITCH)
    b = connect_seed_to_maxima([seed], targets, R, PITCH, mesh_pts=None, f_vals=None)
    assert a[0].status == b[0].status == "no_target"
    assert a[0].method == b[0].method == "geodesic"
    del f, pts


def test_summary_counts_geodesic_and_gradient() -> None:
    """The method split is visible in the summary, so the fallback is measurable."""
    seed = _seed("SAD_1", [_branch(0, 0.0)])
    targets = [_target("MAX_9", "maximum", 0.0, 0.0, value=1.0, disc_deg=2.0)]
    pts, f = _mesh_and_peak((0.0, 0.0, 1.0))
    conns = connect_seed_to_maxima([seed], targets, R, PITCH, mesh_pts=pts, f_vals=f)
    stats = summarize_connections(conns)
    assert stats["n_gradient"] == 1
    assert stats["n_geodesic"] == 0
    assert stats["n_gate_behind"] == 1


# ---------------------------------------------------------------------------
# Entry 055: the antipodal connector, and the audit that an ascent branch is
# never left uncommitted.
#
# Two branches of one saddle that claim one maximum have the same endpoints, so
# the centre-to-centre geodesic gives them the same polyline. Away from the
# antipode that is honest. Near it the shortest geodesic is NOT unique, the
# branches genuinely travel different ways, and the scene shows one line where
# there should be two. Measured on ethene3 C1 (potential V): saddle MC15 and
# maximum MC14 are 178.01 deg apart, both branches reported connected, and the
# two polylines agreed point-for-point to 1.5e-6 deg.
# ---------------------------------------------------------------------------


def _antipodal_pair() -> tuple[list[SaddleLaunchSeed], list[ConnectionTarget]]:
    """A k=2 saddle with two ascent branches and a maximum 178 deg away.

    The two ports of a 2-fold saddle are antipodal on the ring, so they share ONE
    great circle through the centre; the branches differ in which way around it
    they travel. The target is placed at azimuth 90, OFF that circle, so the two
    sweeps separate (one over each pole) and the connector ends at the target's
    closest approach on the circle -- inside its excision disc.
    """
    seed = _seed("SAD_1", [_branch(0, 0.0), _branch(1, 180.0)])
    pool = _pool(_max("ANTIPODE", 178.0, 90.0), disc_deg=4.0)
    return [seed], pool


def test_antipodal_branches_draw_distinct_arcs() -> None:
    """The two branches must be visible as two lines, not one.

    Each branch's connector is built on the great circle through its OWN launch
    port, swept in the port's direction, so the pair separates.
    """
    seeds, pool = _antipodal_pair()
    conns = connect_seed_to_maxima(seeds, pool, R, PITCH)
    assert [c.status for c in conns] == ["connected", "connected"]
    assert {c.max_id for c in conns} == {"ANTIPODE"}
    a = np.array(conns[0].arc_points)
    b = np.array(conns[1].arc_points)
    dev = max(math.degrees(angular_distance(a[i], b[i])) for i in range(len(a)))
    assert dev > 90.0, f"the two arcs collapsed to one line: max deviation {dev:.2f}"
    # Both start at the saddle centre.
    assert np.linalg.norm(a[0] - np.array([0.0, 0.0, 1.0])) < 1e-12
    assert np.linalg.norm(b[0] - np.array([0.0, 0.0, 1.0])) < 1e-12
    # Each ends at the target's closest approach on its own circle, which is
    # inside the disc -- the basin, per Entry 044 -- not at the centre.
    target = np.array(_pt(178.0, 90.0))
    for arc in (a, b):
        d = math.degrees(float(angular_distance(arc[-1], target)))
        assert d <= 4.0 + 1e-6, f"arc ends {d:.2f} deg from the maximum"
    # The two endpoints are the SAME point: one circle, one projection.
    assert float(np.linalg.norm(a[-1] - b[-1])) < 1e-9


def test_the_antipodal_arcs_are_great_circles() -> None:
    """The bearing-anchored arc is still a geodesic, not a bent path."""
    seeds, pool = _antipodal_pair()
    conns = connect_seed_to_maxima(seeds, pool, R, PITCH)
    for c in conns:
        pts = np.array(c.arc_points)
        assert np.allclose(np.linalg.norm(pts, axis=1), 1.0, atol=1e-9)
        normal = np.cross(pts[1] - pts[0], pts[-1] - pts[0])
        normal /= float(np.linalg.norm(normal))
        assert float(np.max(np.abs(pts @ normal))) < 1e-9


def test_ordinary_separation_still_draws_the_centre_geodesic() -> None:
    """The fix is narrow: away from the antipode nothing changes.

    The shortest geodesic is unique there, so two branches claiming one maximum
    legitimately share a connector, and the arc must be the centre-to-centre
    geodesic exactly as before -- ending at the maximum's CENTRE.
    """
    seed = _seed("SAD_1", [_branch(0, 0.0)])
    pool = _pool(_max("NEAR", 40.0, 0.0), disc_deg=2.0)
    conns = connect_seed_to_maxima([seed], pool, R, PITCH)
    assert conns[0].status == "connected"
    a = np.array(conns[0].arc_points)
    assert float(np.linalg.norm(a[-1] - np.array(_pt(40.0, 0.0)))) < 1e-9
    # Every point on the meridian az=0: the centre-to-target geodesic.
    assert float(np.max(np.abs(a[:, 1]))) < 1e-12


def test_degenerate_arcs_fires_on_identical_polylines() -> None:
    """The audit catches the collapse a status check cannot see.

    Both records report `connected`, so the table looks complete; the drawing is
    not. This is the non-vacuity proof: the pre-fix arcs are fed in explicitly and
    must be flagged.
    """
    seeds, pool = _antipodal_pair()
    conns = connect_seed_to_maxima(seeds, pool, R, PITCH)
    assert degenerate_arcs(conns) == []

    shared = conns[0].arc_points
    collapsed = [replace(c, arc_points=shared) for c in conns]
    pairs = degenerate_arcs(collapsed)
    assert len(pairs) == 1
    a, b = pairs[0]
    assert (a.branch_index, b.branch_index) == (0, 1)
    assert a.saddle_id == b.saddle_id == "SAD_1"


def test_degenerate_arcs_ignores_different_maxima() -> None:
    """The audit compares branches that claim the SAME maximum."""
    seed = _seed("SAD_1", [_branch(0, 0.0), _branch(1, 90.0)])
    pool = _pool(_max("A", 30.0, 0.0), _max("B", 30.0, 90.0), disc_deg=2.0)
    conns = connect_seed_to_maxima([seed], pool, R, PITCH)
    assert all(c.status == "connected" for c in conns)
    assert degenerate_arcs(conns) == []


def test_every_branch_is_committed() -> None:
    """The audit the user asked for: no uphill launch seed is left uncommitted.

    Every branch must be claimed, occluded, or reported unresolved WITH A GATE
    naming why. A record that is none of those is a bug, not a finding.
    """
    cases = [
        _antipodal_pair(),
        (
            [_seed("SAD_1", [_branch(0, 0.0)])],
            _pool(_max("T", 30.0, 0.0), disc_deg=2.0),
        ),
        (
            [_seed("SAD_1", [_branch(0, 0.0)])],
            _pool(_max("FAR", 100.0, 0.0), disc_deg=2.0),
        ),
    ]
    for seeds, pool in cases:
        conns = connect_seed_to_maxima(seeds, pool, R, PITCH)
        stats = summarize_connections(conns)
        assert stats["n_uncommitted"] == 0
        assert stats["uncommitted_branches"] == []
        assert (
            stats["n_connected"] + stats["n_occluded"] + stats["n_no_target"]
            == stats["n_branches"]
        )
        for c in conns:
            if c.status == "no_target":
                assert c.failure_gate in ("behind", "cone", "cap", "occluded", "empty")


def test_summary_reports_the_degeneracy_count() -> None:
    """The audit is visible in the snapshot, not just in the function."""
    seeds, pool = _antipodal_pair()
    conns = connect_seed_to_maxima(seeds, pool, R, PITCH)
    stats = summarize_connections(conns)
    assert stats["n_degenerate_arcs"] == 0
    assert stats["degenerate_arcs"] == []
    shared = conns[0].arc_points
    collapsed = summarize_connections([replace(c, arc_points=shared) for c in conns])
    assert collapsed["n_degenerate_arcs"] == 1
    assert collapsed["degenerate_arcs"] == ["SAD_1/0~1"]


# --- refine_connections: the sawtooth-smoothing post-pass --------------------


def _gradient_conn(
    arc: tuple[tuple[float, float, float], ...],
    **kw: object,
) -> MaxConnection:
    """A connected gradient record carrying the given polyline."""
    base = MaxConnection(
        saddle_id="SAD_1",
        branch_index=0,
        max_id="MAX_9",
        status="connected",
        direction_mode="port_tangent",
        cross_track_deg=None,
        along_track_deg=10.0,
        separation_deg=20.0,
        mutual=False,
        arc_points=arc,
        saddle_value=0.0,
        max_value=1.0,
        verified=True,
        method="gradient",
    )
    return replace(base, **kw)  # type: ignore[arg-type]


def _sawtooth_arc(
    n: int,
    amplitude_pitch: float = 0.5,
    step_pitch: float = 1.0,
    span_deg: float = 60.0,
) -> tuple[tuple[float, float, float], ...]:
    """A traced connector that chatters across the equator: one-pitch teeth.

    The path advances ``step_pitch`` pitches in azimuth each vertex and alternates
    ``amplitude_pitch`` pitches of latitude about the equator, so the average
    direction is the straight crest while the point-to-point turn is huge. This is
    the geometry the smoother exists to remove.
    """
    amp = amplitude_pitch * math.degrees(PITCH)
    out: list[tuple[float, float, float]] = []
    for i in range(n):
        az = -span_deg / 2.0 + span_deg * i / (n - 1)
        lat = amp if i % 2 == 0 else -amp
        out.append(_pt(90.0 - lat, az))
    return tuple(out)


def _smooth_equator_arc(
    n: int, span_deg: float = 60.0
) -> tuple[tuple[float, float, float], ...]:
    """A clean path along the equator: no chatter, so the smoother leaves it alone."""
    return tuple(_pt(90.0, -span_deg / 2.0 + span_deg * i / (n - 1)) for i in range(n))


def test_refine_replaces_the_ascent_with_a_coarse_curve() -> None:
    """The coarse mode describes the connection, and the winding goes with the ascent.

    The total point-to-point turning collapses: a chattering path pays a large turn
    at every tooth, and a curve whose nodes are NODE_PITCH pitches apart pays almost
    none. The node count is the description length of the branch -- a handful of
    points, not one per mesh step.
    """
    arc = _sawtooth_arc(31)
    conn = _gradient_conn(arc)
    out = refine_connections([conn], R, PITCH, coarse=True)[0]
    assert out.refined
    assert out.n_nodes >= 2
    assert out.total_turn_before_deg > 1000.0, "fixture must be chattering"
    assert out.total_turn_after_deg < 0.15 * out.total_turn_before_deg, (
        f"chatter survived: {out.total_turn_before_deg:.0f} -> "
        f"{out.total_turn_after_deg:.0f} deg"
    )
    # The node set is a handful of points, not one per mesh step: the description
    # length of the branch is its arc divided by NODE_PITCH, and everything between
    # the nodes is interpolation.
    assert out.n_nodes <= len(arc) // 3, (
        f"the curve is not coarse: {out.n_nodes} nodes from {len(arc)} vertices"
    )
    # The endpoints are the topology's: the curve starts and finishes on them.
    assert math.dist(out.arc_points[0], arc[0]) < 1e-9
    assert math.dist(out.arc_points[-1], arc[-1]) < 1e-9


def test_refine_leaves_a_geodesic_connector_alone() -> None:
    """A great circle is not an ascent, so it carries no chatter to remove.

    The geodesic connector IS the straight line between two centres by definition.
    Smoothing it would move it off the line it is defined to be, which is the
    opposite of the intent -- and it is why the post-pass filters on ``method``.
    """
    arc = _sawtooth_arc(31)
    geo = _gradient_conn(arc, method="geodesic")
    out = refine_connections([geo], R, PITCH)[0]
    assert out is geo
    assert not out.refined
    assert out.arc_points == geo.arc_points


def test_coarse_curve_does_not_distort_a_smooth_path() -> None:
    """A path with no winding is described, not deformed: the curve sits on it.

    The non-vacuity guard for the coarse spacing. A curve through nodes 8 pitches
    apart is a licence to depart from the ascent by many pitches, and on a spiralling
    field it uses it. That licence must cost nothing where the path is already
    smooth: a great circle is reproduced by a great circle, so the deviation stays
    at the tessellation floor. If this test fails, the spacing is throwing away
    shape it was never asked to discard.
    """
    arc = _smooth_equator_arc(31)
    conn = _gradient_conn(arc)
    out = refine_connections([conn], R, PITCH, coarse=True)[0]
    assert out.refined, "a clean path is still described by a curve"
    assert out.refine_max_shift_deg < 0.5, (
        f"the coarse curve distorted a smooth path: "
        f"{out.refine_max_shift_deg:.2f} deg off it"
    )


def test_refine_leaves_a_short_path_alone() -> None:
    """Fewer vertices than the window needs: the record is returned unchanged."""
    arc = _sawtooth_arc(4)
    conn = _gradient_conn(arc)
    out = refine_connections([conn], R, PITCH)[0]
    assert out is conn
    assert not out.refined


def test_refine_pins_a_transited_excision_centre() -> None:
    """The centre of a region the path was carried across does not move.

    A transit is a straight carry over ground the field cannot speak for, so the
    centre is a position the pipeline vouches for. The smoothed curve is pinned to
    the vertex nearest it and passes exactly through it.
    """
    arc = _sawtooth_arc(31)
    mid = 15
    centre = np.asarray(arc[mid], dtype=float)
    conn = _gradient_conn(arc, transited_ids=("MC7",))
    region = ExcisionRegion(
        region_id="MC7", centre_unit=centre / np.linalg.norm(centre), radius_rad=0.1
    )
    out = refine_connections([conn], R, PITCH, regions=[region])[0]
    assert out.refined
    # The curve passes through the pinned vertex: some output point sits on it.
    dists = [math.dist(p, centre) for p in out.arc_points]
    assert min(dists) < 1e-6, "the smoothed curve did not pass through the transit"


def test_refine_preserves_the_endpoints_of_the_connector() -> None:
    """The launch port and the maximum stay put: they are the topology's, not the fit's.

    The endpoints are what the connection RECORD asserts -- a branch from this
    saddle to this maximum. A smoothing that moved one would change the
    connection, not the line representing it.
    """
    arc = _sawtooth_arc(31)
    conn = _gradient_conn(arc)
    out = refine_connections([conn], R, PITCH)[0]
    assert out.refined
    assert math.dist(out.arc_points[0], arc[0]) < 1e-9
    assert math.dist(out.arc_points[-1], arc[-1]) < 1e-9


def test_summary_reports_smoothing() -> None:
    """The smoothing's cost is in the snapshot, next to the record it interprets."""
    arc = _sawtooth_arc(31)
    conn = _gradient_conn(arc)
    out = refine_connections([conn], R, PITCH)
    stats = summarize_connections(out)
    assert stats["n_refined"] == 1
    assert stats["n_curve_nodes"] == out[0].n_nodes
    assert stats["total_turn_before_deg"] == pytest.approx(out[0].total_turn_before_deg)
    assert stats["total_turn_after_deg"] == pytest.approx(out[0].total_turn_after_deg)
    assert stats["max_refine_shift_deg"] == pytest.approx(out[0].refine_max_shift_deg)
    assert stats["total_turn_after_deg"] < stats["total_turn_before_deg"]
    assert summarize_connections([conn])["n_refined"] == 0


def test_smoothing_preserves_the_degenerate_arc_audit() -> None:
    """Resampling must not blind the audit that catches two branches drawing one line.

    ``degenerate_arcs`` compares polylines point-for-point and requires equal vertex
    counts. The smoother emits a variable count, so the risk is real: two branches
    that used to be identical at N vertices must remain identical after both are
    resampled to M, and two that differed must still differ. Determinism of the
    smoother is what makes the first half true -- the same input path gives the same
    output path -- and the audit runs on the REFINED table, so this is the property
    that keeps it working.
    """
    arc = _sawtooth_arc(31)
    twin = [
        _gradient_conn(arc, branch_index=0),
        _gradient_conn(arc, branch_index=1),
    ]
    smoothed_twins = refine_connections(twin, R, PITCH)
    assert all(c.refined for c in smoothed_twins)
    assert len(smoothed_twins[0].arc_points) == len(smoothed_twins[1].arc_points)
    pairs = degenerate_arcs(smoothed_twins)
    assert len(pairs) == 1, "resampling hid a collapsed pair from the audit"

    # Same span, deeper teeth: the two legs resample to the SAME vertex count, so
    # the audit's length guard cannot be what separates them -- the point-for-point
    # comparison must be. A fixture with a different span would pass on the length
    # check alone and prove nothing about the tolerance.
    other = _sawtooth_arc(31, amplitude_pitch=0.9)
    distinct = [
        _gradient_conn(arc, branch_index=0),
        _gradient_conn(other, branch_index=1),
    ]
    smoothed_distinct = refine_connections(distinct, R, PITCH)
    assert all(c.refined for c in smoothed_distinct)
    assert len(smoothed_distinct[0].arc_points) == len(
        smoothed_distinct[1].arc_points
    ), "fixture must reach the point-for-point comparison"
    assert degenerate_arcs(smoothed_distinct) == [], (
        "resampling made two different legs look identical"
    )


def test_gradient_path_is_prepended_with_saddle_to_port_bridge() -> None:
    """Gradient-traced paths now show the full trajectory from saddle center to target.

    When a gradient climb traces from the port uphill, the arc_points now include
    a prepended bridge from the saddle center to the port, making the full path
    visible. This parallels the snap-to-center behavior at the destination.
    """
    seed = _seed("SAD_1", [_branch(0, 0.0)])
    # Maximum at the north pole; the port launches outward, so it's behind the ray.
    targets = [_target("MAX_9", "maximum", 0.0, 0.0, value=1.0, disc_deg=2.0)]
    pts, f = _mesh_and_peak((0.0, 0.0, 1.0))

    conn = connect_seed_to_maxima([seed], targets, R, PITCH, mesh_pts=pts, f_vals=f)[0]
    assert conn.status == "connected"
    assert conn.method == "gradient"
    assert len(conn.arc_points) >= 2

    # The path should start near the saddle center (the seed position).
    saddle_center = np.asarray(seed.saddle_position, dtype=np.float64)
    first_point = np.asarray(conn.arc_points[0], dtype=np.float64)

    # First point should be close to saddle center (within a couple of arc samples).
    # The bridge spans from saddle (radius R) to port (also radius R), sampled into
    # ARC_SAMPLES+1 points. The first point should be very close to the saddle.
    distance_to_saddle = np.linalg.norm(first_point - saddle_center)
    # The max distance from saddle to port is at most 2*R (antipodal), so first point
    # should be very close (<< R) to the saddle.
    assert distance_to_saddle < 0.2 * R, (
        f"First point {distance_to_saddle} too far from saddle center"
    )

    # Verify the arc includes multiple points (bridge + ridge segments).
    # The bridge contributes ARC_SAMPLES+1 points, so total should be well above 2.
    assert len(conn.arc_points) > 2 * ARC_SAMPLES / 10, (
        f"Arc too short: {len(conn.arc_points)} points, expected > {2 * ARC_SAMPLES / 10}"
    )
