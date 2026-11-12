"""Maximum Connection Subroutine for gba_topology2 (Basin Locator Step 3).

A saddle's ascent (ridge) launch branch is the direction the separatrix leaves
the saddle on its way UP toward a maximum. This subroutine answers the question
that direction poses: *which* maximum is this branch aiming at?

The method: WALK THE RAY, CONE FOR TARGETS, DISC FOR OBSTACLES
--------------------------------------------------------------
The ray is a SOLVER, not the connector. Its job is to find the two endpoints of
a connection -- the saddle cluster centre and the maximum cluster centre -- and
the connector drawn is the single great circle between them. The port, the launch
bearing and the boundary crossing are all internal to the search.

Each branch supplies a ray on the sphere, launched from its PORT on the saddle's
excision boundary along the outward geodesic. Every VISIBLE CRITICAL POINT --
maxima, minima and saddles alike -- is scored by its position relative to that
ray, decomposed in the orthonormal frame ``(o, d, n)`` where ``o`` is the unit
origin, ``d`` the unit direction and ``n = o x d`` the pole of the ray's great
circle:

    along = atan2(t.d, t.o)      progress along the ray to the CP's CENTRE
    cross = asin(t.n)            perpendicular deviation from the ray

The candidates inside the forward sector (``0 < along <= max_arc_deg`` and
``|cross| <= cone_pitch * delta_theta``) are walked in increasing ``along``, and
the walk stops at the first CP that is decisive:

- a **maximum** ends the walk and the branch CONNECTS to it. A target is
  captured by the CONE. Where the ray crosses the maximum's excision boundary,
  ``entry = along - acos(cos rho / cos cross)``, is recorded as the ranking key
  (``along_track_deg``) and as how far the ray travels before it reaches the
  basin; it is not a vertex of the drawn connector.
- a **minimum or saddle** ends the walk as an OCCLUSION only if the great circle
  actually enters its excision disc, i.e. ``|cross| <= rho``. Otherwise the ray
  sails past it and the walk continues.

So the disc is the criterion for an OBSTACLE and the cone is the criterion for a
TARGET. Ties break on ``|cross|`` then on id, so the result is a pure function of
the input set and not of its order.

Why the cone captures a maximum, not the disc (measured, user-reported)
----------------------------------------------------------------------
Requiring the ray to enter the MAXIMUM's own boundary is tuned backwards. Every
raw maximum's excision radius is one mesh pitch -- 1.61 deg on Pd_20K -- while
the legitimate rays into a maximum ringed by three saddles arrive at cross-tracks
0.0, 1.35, 1.42 on MAX_7 and 0.0, 2.00, 1.98 on its symmetric twin MAX_4. A gate
at 1.61 sits inside that band, so it accepted all three of MAX_7's connections
and rejected two of MAX_4's: identical geometry, different answers, decided by
noise. The connector counts came out 3, 2, 2, 2, 2, 1, 1, 3 across the eight
symmetric maxima. Capturing by the cone, which is the directional criterion the
sector already encodes, gives all eight maxima exactly 3.

Why the disc blocks an obstacle, not the cone (measured, user-reported)
----------------------------------------------------------------------
Scoring an obstacle's CENTRE occludes on NEAR MISSES, because the cone is
necessarily wider than any single disc. On Pd_20K curvedness, saddle MC18's
branch 0 was blocked from its maximum MC38 by a minimum MIN_19165 at
``along 7.5, cross 7.4`` -- a 1.6 deg object the ray passes 7.4 deg clear of. It
is not an obstacle; the ray sails past it. It was a candidate only because the
9.6 deg cone is six times wider than its disc, and it won because its centre is
nearer than MC38's. Requiring the ray to enter the obstacle's disc takes the
occlusion count on that snapshot from 20 to 0 and MC18 connects to MC38.

Why the disc is an ELLIPSE evaluated toward the ray
---------------------------------------------------
An excision boundary around an elongated set is a fitted ellipse, not a circle.
Using its semi-major as a disc inflates the transverse reach by the aspect ratio:
on Pd_20K shape index the bridged corridor MS13 (a = 11.35, b = 2.43, aspect
10.6) occluded a ray passing 3.58 deg off its axis, because 3.58 < 11.35 while
3.58 > 2.78. :func:`directional_support_radius` evaluates the boundary in the
direction of the ray. This is the same failure the bridging gap test already
fixed for the same reason (``string_aggregator.cluster_directional_support_radius``),
whose docstring names it: an isotropic radius "turns a long thin corridor into a
disk of radius a and lets it swallow compact fragments sitting far off to the
side".

Why the previous rule was wrong (measured, user-reported)
--------------------------------------------------------
The first version took the smallest ``|cross|`` with ``along`` as a tie-break.
That scores AIM and never checks REACH, and it fails in three ways:

1. **Collinear maxima are the degenerate case, structurally.** A ray that
   passes through the maximum MC109 and on to MC143 has ``cross(MC109) = 0``
   and ``cross(MC143) = 0`` -- both lie on the great circle by construction, so
   cross-track is provably unable to separate them. Only ``along`` can, and it
   was the secondary key, consulted only on a bit-exact tie. On a dense field
   that is not rare: across the four saved Pd_20K snapshots the mean cross-track
   of a claim was 0.19-4.4 deg, i.e. many maxima sit essentially ON each other's
   rays, and float noise at 1e-11 rad was deciding a physical question.
2. **The pool was maxima-only**, so the rule could not express "passes through a
   critical point" at all -- it never looked at the critical points that were
   not maxima.
3. **The drawn arc is the saddle->winner geodesic**, so the renderer faithfully
   drew the occlusion the matcher had ignored. That is what made it visible.

Why the sector is a constant, not the branch's own sector width
---------------------------------------------------------------
The obvious per-branch, parameter-free cone is ``LaunchBranch.sector_width_rad``,
the angular gap between the two separatrix crossings bounding the branch's
ascent sector. Measured on the four snapshots it is unusable: on Curvedness 40
of 72 branches report width **0.0 deg** and several report **355-357 deg**; on V
four report 352.6 deg. It is the span between bounding crossings, which
degenerates whenever a sector contains all of them. So the half-angle is the
pitch-relative ``CONE_PITCH`` below, and the number of tunables stays at one.

This is deliberately the SIMPLEST possible connector: a great circle, no field
integration, no barrier test. Step 4 replaces it with the traced separatrix, and
the value of having it early is that the traced path has something to be
compared against -- a geodesic that lands on a different maximum than the
integrator does is a measurement, not a surprise.

The ray origin: the port, not the saddle centre
----------------------------------------------
The excision path (``excision_launch.py``) places ``launch_position`` on the
ring and ``tangent_direction`` at that port, pointing outward from the ring
centre. The saddle centre is the ring centre. Two rays are defensible:

- ``"port_tangent"`` (the DEFAULT): origin = the port, direction =
  ``tangent_direction``, the outward continuation measured AT the ring.
- ``"chord"``: origin = saddle centre, direction = the centre->port bearing.

They are the same great circle -- ``_outward_tangent`` builds the port direction
as ``radial * port - centre``, the parallel transport of the centre->port
geodesic, so both share the pole ``centre x port``. They differ ONLY in where
``along = 0`` sits, which shifts by the ring radius. That shift is the whole
reason the port is the origin: **the disc inside the excision boundary is the
degenerate core whose field is meaningless -- that is the entire reason the
boundary exists.** Launching from the centre means the connector's first ~ring
radius traverses that core, and any critical point the ring swallowed (a leaked
neighbour, per the RING > CENSUS finding) sits on the ray and occludes it. From
the port, every CP inside the boundary has ``along < 0`` and is automatically
excluded. Measured on Pd_20K curvedness, MC18's ring is 8.3 deg and the
occluding minima sit at along 7.5 -- inside the core, so the port origin removes
them by construction, and the disc test removes the off-ray ones.

The mode stays an API parameter (not a viewer control) because it cannot change
the winner on a clean field, and Step 4's integrator starts at the port.

Reciprocity
-----------
Claims are one-way: a branch claims the first critical point it runs into. A
connection is flagged ``mutual`` when it is the tightest claim on that maximum,
i.e. the maximum also points back at this branch. The flag is reported, never
used to gate, because a non-mutual claim is exactly the signal a crossing
separatrix produces -- hiding it would hide the phenomenon.

Occlusion is NOT reported as a mere absence. A branch blocked by a minimum or a
saddle carries ``status == "occluded"`` and the blocker's id, because "this ray
flies over a minimum on the way to that maximum" is a topological finding about
the field, not a failure to find a target.

Step 4: the ray resolves, the field finishes
--------------------------------------------
The ray is a straight-line rule, so it can only find a maximum that lies
FORWARD of the port. On the organic fields (ethene3) the separatrix leaves a
saddle, curves, and arrives at a maximum that sits BEHIND the launch direction:
the measured along-track is negative, so no cap or cone widens the ray enough to
see it. Such a branch is INCOMPLETE, and the field itself supplies the answer.

When the ray reaches nothing, the branch is climbed: ``separatrix_tracer``
integrates the surface gradient uphill from the port and stops when it enters a
maximum's disc. The record carries ``method == "gradient"`` and the traced
polyline as its connector -- flattening it to the geodesic between its endpoints
would erase the wrapping that made the ray fail, which is the finding. The
viewer draws these in cyan against the gold great circles.

The gate is recorded on every unresolved branch (``failure_gate``): ``behind``
(the maximum's along-track is negative), ``cone`` (outside the forward sector),
``cap`` (past the arc-length ceiling), ``occluded`` (a disc entered first) or
``empty`` (no maximum in the pool). This makes the two-stage design auditable --
"the ridge curves" and "the ridge is merely long" are different statements about
the field, and only the first is what the ascent exists to capture.

Occluded branches are NOT rescued. An occlusion is a resolved verdict: the ray
found the first critical point, and climbing the field from the port would
contradict the geodesic rule that produced it. Only ``no_target`` is incomplete.

The fallback is inert when ``mesh_pts``/``f_vals`` are omitted, which is the
contract on a dense metallic field: every Pd_20K branch resolves geodesically, so
the whole connection table is byte-identical with the field supplied (verified
against the HEAD baseline on both shape index and curvedness).

Frame contract: seeds carry CENTRED positions (``SaddleLaunchSeed``), maxima
carry WORLD positions (``BasinCenter``). Pass ``center`` (the world-frame
nucleus) to map the latter into the centred frame; the whole computation runs
centred, where ``p / ||p||`` is the outward normal, and every emitted coordinate
is mapped back out with ``to_global`` so the viewer can draw it. Omit ``center``
only for origin-centred synthetic meshes, where the two frames coincide.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

from gba_topology2.src.basin_center import BasinCenter
from gba_topology2.src.excision_boundary import KNearestFinder
from gba_topology2.src.micro_cluster import MicroCluster
from gba_topology2.src.morse_detector import CriticalPoint
from gba_topology2.src.ridge_refiner import (
    smooth_gradient_ridge,
    structural_curve_ridge,
)
from gba_topology2.src.saddle_launch import LaunchBranch, SaddleLaunchSeed
from gba_topology2.src.separatrix_tracer import (
    ExcisionRegion,
    trace_uphill,
)
from gba_topology2.src.sphere_frame import (
    angular_distance,
    radial_offset,
    rotate_towards,
    tangent_project,
    to_global,
    to_local,
    unit,
)

__all__ = [
    "ANTIPODAL_BAND_RING_MULTIPLE",
    "ARC_SAMPLES",
    "CONE_PITCH",
    "CONNECTION_METHODS",
    "CONNECTION_STATUSES",
    "DEGENERATE_ARC_TOL_DEG",
    "DIRECTION_MODES",
    "GATE_BEHIND",
    "GATE_CAP",
    "GATE_CONE",
    "GATE_EMPTY",
    "GATE_OCCLUDED",
    "MAX_ARC_DEG",
    "ConnectionRay",
    "ConnectionTarget",
    "DirectionMode",
    "DiscGeometry",
    "MaxConnection",
    "basin_centers_to_targets",
    "build_connection_pool",
    "connect_seed_to_maxima",
    "connection_ray",
    "degenerate_arcs",
    "directional_support_radius",
    "disc_entry_angle",
    "disc_geometry_from_cluster",
    "mark_mutual",
    "ray_frame",
    "refine_connections",
    "seed_ray_direction",
    "summarize_connections",
    "target_frame_angles",
]

# The ray's initial bearing is measured from the saddle centre toward the port.
# "port_tangent" starts at the port and follows the outward direction measured
# there. Both lie on the same great circle (see the module docstring); the mode
# is an API parameter for the integrator, not a viewer control.
DIRECTION_MODES: tuple[str, ...] = ("chord", "port_tangent")

DirectionMode = Literal["chord", "port_tangent"]

_STATUS_CONNECTED = "connected"
_STATUS_OCCLUDED = "occluded"
_STATUS_NO_TARGET = "no_target"

_KIND_MAXIMUM = "maximum"

# How a connection's endpoints were found. GEODESIC is the great-circle ray of
# Step 3; GRADIENT climbs the field, used when the ray reaches nothing.
_METHOD_GEODESIC = "geodesic"
_METHOD_GRADIENT = "gradient"
CONNECTION_METHODS: tuple[str, ...] = (_METHOD_GEODESIC, _METHOD_GRADIENT)

# Why the geodesic ray failed, so the fallback is attributable rather than
# opaque. See MaxConnection.failure_gate.
GATE_CONE = "cone"
GATE_CAP = "cap"
GATE_BEHIND = "behind"
GATE_OCCLUDED = "occluded"
GATE_EMPTY = "empty"

# The three verdicts a branch can receive, in decreasing order of success.
CONNECTION_STATUSES: tuple[str, ...] = (
    _STATUS_CONNECTED,
    _STATUS_OCCLUDED,
    _STATUS_NO_TARGET,
)

# Longest arc a branch may claim, in DEGREES -- not in mesh pitches. The units
# are the point: reach is an angular property of the sphere, while the cone is
# legitimately pitch-relative because it is about resolving structure at the mesh
# scale. Expressed in pitches the same constant means different things on
# different meshes -- 40 pitches is 64 deg on Pd_20K (pitch 1.6058) and 89 deg on
# ethene3 (2.2278) -- and it was calibrated on the metallic fields, where the
# largest saddle-to-claimed-maximum separation measured is 57.1 deg. Organic
# fields have near-ANTIPODAL saddle/maximum pairs: on ethene3 C1 the potential V
# the saddle MC14 and the maximum MC13 are 178.01 deg apart, and BOTH ascent
# branches point at it dead centre (cross-track -0.01 and +0.02 deg). A
# hemisphere cap rejects a ray that has found its target perfectly, and the
# gradient fallback then has to recover an answer the geodesic already had.
#
# 180 deg is the natural domain of a great-circle ray: every geodesic from a
# point reaches every point, uniquely up to the antipode. Under the first-hit
# rule the cap is a safety net, not the discriminator -- the nearer hit wins on
# along-track, and the CONE is what makes the claim directional. Measured across
# the saved snapshots, widening 89 deg -> 180 deg strictly improves and never
# regresses: ethene3 V 1/2 -> 2/2, ethene3 curvedness 9/14 -> 10/14 (the two new
# claims at cross-track 0.27 and -0.24 deg, one of which the gradient had
# independently reached at the SAME maximum), ethene3 Electron Density, and both
# Pd_20K fields unchanged.
MAX_ARC_DEG = 180.0

# How close to the antipode makes the centre-to-target geodesic degenerate, as a
# multiple of the excision ring angle. At separation 180 deg there is no unique
# shortest great circle -- infinitely many connect the two points -- so the
# centre-to-target formula cannot express WHICH one a branch travelled, and two
# branches of one saddle that claim one maximum collapse onto an identical
# polyline. The band is 2 x ring because that is the smallest separation at which
# two branches' own port bearings can differ by more than the ring: measured on
# ethene3 C1 (potential V) the saddle MC15 and maximum MC14 are 178.01 deg apart,
# 1.99 deg from antipodal, inside a 2 x 2.23 = 4.46 deg band, and the two arcs
# were identical to 0.000e+00 deg. Outside the band the shortest geodesic is
# unique and this code path is not taken, so every ordinary connector is
# bit-for-bit unchanged (verified byte-identical at 20, 60, 120, 160 and 170 deg).
ANTIPODAL_BAND_RING_MULTIPLE = 2.0

# Point-for-point deviation, in DEGREES, below which two connectors count as the
# SAME line for the degeneracy audit. Deliberately NOT pitch-relative, the inverse
# of the arc-cap lesson (Entry 054): the question is "are these two polylines
# numerically identical", which is a property of float arithmetic, not of the
# mesh, so an absolute threshold is the right unit. 1e-3 deg sits three orders of
# magnitude above the ~1.5e-6 deg float64 round-trip noise on world-frame arcs (an
# angle between two points 1.26 A from the nucleus loses precision to
# cancellation) and five below the 38 deg separation of the distinct pair that
# replaced the collapsed one. The gap is enormous, so the test is not sensitive to
# the exact value.
DEGENERATE_ARC_TOL_DEG = 1e-3

# Half-angle of the forward sector a branch may claim within, in mesh-pitch
# units. 6.0 pitches is 9.6 deg on Pd_20K (pitch 1.6058) and 13.4 deg on
# ethene3 (2.2278). Chosen as the smallest pitch multiple that admits every
# cross-track observed on the four saved Pd_20K snapshots: the largest was 7.82
# deg (V, 6 maxima), i.e. 4.9 pitches, so 6.0 leaves headroom without opening
# the sector wide enough to admit a neighbour off the branch's own gateway. In
# pitch units for the same reason as every other threshold here: the relevant
# scale is the intrinsic grid, and it keeps the constant transferable between
# the 1.6 deg and 2.2 deg meshes.
CONE_PITCH = 6.0

# Segments per rendered arc. The arc is a pure display artefact (the match itself
# is closed-form), so this only controls how smooth the tube looks.
ARC_SAMPLES = 24


@dataclass(frozen=True)
class ConnectionTarget:
    """One visible critical point a branch can run into.

    The occlusion test needs every critical point, not just maxima: a ray that
    flies over a minimum on its way to a distant maximum is not a connection.
    This is the uniform shape the pool is built from, so a raw CP, an effective
    CP and a :class:`BasinCenter` all enter the same search.

    A critical point is a DISC, not a point. Its excision boundary is the circle
    of ``angular_radius_rad`` about its centre, and that disc is the region where
    its own field is degenerate -- the reason the boundary exists. So a ray
    "hits" a critical point when it enters the disc, which is what makes a
    maximum capturable and a minimum sitting off to the side a non-obstacle.

    Attributes:
        cp_id: Identifier of the critical point.
        polarity: ``"maximum"``, ``"minimum"`` or ``"saddle"``.
        position: Position in the frame the caller's positions are in (world
            when built from the viewer's UI dicts). Not required to be unit
            length -- :func:`connect_seed_to_maxima` normalises it after mapping
            into the centred frame.
        value: Representative scalar field value.
        angular_radius_rad: Angular radius of this CP's excision disc, radians.
            0.0 means a point target (no disc), the conservative choice when the
            radius is unknown.
        semi_major_rad: Semi-major angular radius of the fitted boundary ellipse.
            Defaults to ``angular_radius_rad``; equal to the semi-minor for a
            compact set, in which case the disc is exact.
        semi_minor_rad: Semi-minor angular radius, radians. Defaults to
            ``angular_radius_rad``.
        principal_axis_unit: Major-axis direction of the ellipse, centred frame.
            Zero vector means no axis, and the radius is taken isotropically.
    """

    cp_id: str
    polarity: str
    position: npt.NDArray[np.float64]
    value: float
    angular_radius_rad: float = 0.0
    semi_major_rad: float = -1.0
    semi_minor_rad: float = -1.0
    principal_axis_unit: npt.NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(3, dtype=np.float64)
    )

    def __post_init__(self) -> None:
        # A target built with only the isotropic radius is a circle: both
        # semi-axes take it, so the directional radius returns it unchanged.
        if self.semi_major_rad < 0.0:
            object.__setattr__(self, "semi_major_rad", self.angular_radius_rad)
        if self.semi_minor_rad < 0.0:
            object.__setattr__(self, "semi_minor_rad", self.angular_radius_rad)

    def radius_toward(self, target_unit: npt.NDArray[np.float64]) -> float:
        """Excision-boundary radius in the direction of ``target_unit``.

        Returns the isotropic ``angular_radius_rad`` when the target has no
        fitted axis, so a compact set and a set of unknown shape behave exactly
        as they did before the ellipse was carried.
        """
        axis = np.asarray(self.principal_axis_unit, dtype=np.float64)
        if float(np.linalg.norm(axis)) <= 1e-12:
            return float(self.angular_radius_rad)
        return directional_support_radius(
            self.position, self.semi_major_rad, self.semi_minor_rad, axis, target_unit
        )


@dataclass(frozen=True)
class DiscGeometry:
    """The fitted excision boundary of one critical point, as an ellipse.

    A compact set has ``semi_major_rad == semi_minor_rad`` and the ellipse is a
    circle; a bridged corridor is elongated and the transverse radius is far
    smaller than the longitudinal one. The occlusion test needs the transverse
    value, so it must carry the shape, not just a scalar.

    Attributes:
        angular_radius_rad: Isotropic radius, the fallback when no axis is set.
        semi_major_rad: Semi-major angular radius, radians.
        semi_minor_rad: Semi-minor angular radius, radians.
        principal_axis_unit: Major-axis direction, centred frame. A zero vector
            means "no axis", and the radius is taken isotropically.
    """

    angular_radius_rad: float
    semi_major_rad: float
    semi_minor_rad: float
    principal_axis_unit: tuple[float, float, float] = (0.0, 0.0, 0.0)


def disc_geometry_from_cluster(cluster: MicroCluster) -> DiscGeometry:
    """Read a cluster's fitted boundary as a :class:`DiscGeometry`.

    A cluster with no fitted ellipse (morphology "compact") has its semi-axes set
    to the isotropic radius, so the geometry round-trips exactly.

    Args:
        cluster: The micro-cluster (or a corridor adapted to one).

    Returns:
        The disc geometry for its excision boundary.
    """
    axis = cluster.principal_axis_unit
    return DiscGeometry(
        angular_radius_rad=float(cluster.angular_radius_rad),
        semi_major_rad=float(cluster.semi_major_rad or cluster.angular_radius_rad),
        semi_minor_rad=float(cluster.semi_minor_rad or cluster.angular_radius_rad),
        principal_axis_unit=(float(axis[0]), float(axis[1]), float(axis[2])),
    )


def basin_centers_to_targets(
    centers: list[BasinCenter],
    discs: dict[str, DiscGeometry] | None = None,
) -> list[ConnectionTarget]:
    """Convert visible extrema into connection targets.

    Args:
        centers: Basin centers (maxima and minima) in the world frame, as
            produced by ``extract_basin_centers``.
        discs: Fitted excision boundary per entry, keyed by ``center_id``.
            Entries with no geometry get a zero radius and behave as points.

    Returns:
        One :class:`ConnectionTarget` per entry, positions left in the input
        frame -- :func:`connect_seed_to_maxima` maps them to the centred frame
        when ``center`` is given.
    """
    table = discs or {}
    out: list[ConnectionTarget] = []
    for c in centers:
        g = table.get(str(c.center_id))
        out.append(
            ConnectionTarget(
                cp_id=str(c.center_id),
                polarity=c.polarity,
                position=np.asarray(c.position, dtype=np.float64),
                value=float(c.value),
                angular_radius_rad=g.angular_radius_rad if g else 0.0,
                semi_major_rad=g.semi_major_rad if g else 0.0,
                semi_minor_rad=g.semi_minor_rad if g else 0.0,
                principal_axis_unit=(
                    np.asarray(g.principal_axis_unit, dtype=np.float64)
                    if g
                    else np.zeros(3, dtype=np.float64)
                ),
            )
        )
    return out


def build_connection_pool(
    basin_centers: list[BasinCenter],
    raw_saddles: list[CriticalPoint],
    effective_cps: list[dict[str, Any]],
    hidden_vertex_ids: set[int],
    center: tuple[float, float, float] | None = None,
    disc_by_vertex: dict[int, DiscGeometry] | None = None,
    disc_by_id: dict[str, DiscGeometry] | None = None,
) -> list[ConnectionTarget]:
    """Assemble every VISIBLE critical point into one occlusion pool.

    The first-hit rule can only express "this ray flies over a minimum/saddle on
    the way to that maximum" if the minimum/saddle is in the pool. Maxima and
    minima come from ``extract_basin_centers``; saddles do NOT -- that function
    emits extrema only. So the visible saddles are added here from two sources,
    which is the recurring "effective CPs have no mesh vertex" trap:

    - raw saddles that survived reduction (``vertex_id not in hidden``), whose
      ``CriticalPoint.position`` is in the CENTRED frame;
    - ``E-SAD`` entries of the viewer's effective-CP dicts, whose ``position`` is
      already in the WORLD frame (an effective CP has no vertex to scan).

    The two frames are reconciled here with ``to_global`` on the raw saddles, so
    every emitted position shares the world frame the basin centers use. Omit
    ``center`` only for origin-centred synthetic meshes, where the frames
    coincide -- on a real molecule they do not, and mixing them puts the raw
    saddles up to 4 R off the sphere.

    Args:
        basin_centers: Visible maxima and minima, world frame.
        raw_saddles: All raw saddles from the Morse detector, centred frame.
        effective_cps: The viewer's effective-CP dicts (world-frame positions).
        hidden_vertex_ids: Raw vertices collapsed into an effective entity; their
            saddles are not visible and must not occlude.
        center: World-frame nucleus, used to map the centred raw saddles out.
        disc_by_vertex: Fitted excision boundary per raw CP, keyed by mesh vertex
            id (``disc_geometry_from_cluster`` on its micro-cluster).
        disc_by_id: Fitted excision boundary per effective CP, keyed by entity id
            (the reduction unit's adapted cluster).

    Returns:
        A de-duplicated pool in the world frame (centred when ``center`` is
        None), ordered by id.
    """
    by_vertex = disc_by_vertex or {}
    by_id = disc_by_id or {}
    point = DiscGeometry(0.0, 0.0, 0.0)
    geo: dict[str, DiscGeometry] = dict(by_id)
    for c in basin_centers:
        if c.center_id in geo:
            continue
        if c.vertex_id >= 0 and c.vertex_id in by_vertex:
            geo[c.center_id] = by_vertex[c.vertex_id]

    pool: list[ConnectionTarget] = basin_centers_to_targets(basin_centers, geo)

    for sd in raw_saddles:
        if sd.vertex_id in hidden_vertex_ids:
            continue
        pos = (
            np.asarray(to_global(sd.position, center), dtype=np.float64)
            if center is not None
            else np.asarray(sd.position, dtype=np.float64)
        )
        g = by_vertex.get(sd.vertex_id, point)
        pool.append(
            ConnectionTarget(
                cp_id=f"SAD_{sd.vertex_id}",
                polarity="saddle",
                position=pos,
                value=float(sd.value),
                angular_radius_rad=g.angular_radius_rad,
                semi_major_rad=g.semi_major_rad,
                semi_minor_rad=g.semi_minor_rad,
                principal_axis_unit=np.asarray(g.principal_axis_unit, dtype=np.float64),
            )
        )

    for eff in effective_cps:
        if eff.get("type") != "E-SAD":
            continue
        pos = eff.get("position", (0.0, 0.0, 0.0))
        eid = str(eff.get("id_label", "E-SAD"))
        g = by_id.get(eid, point)
        pool.append(
            ConnectionTarget(
                cp_id=eid,
                polarity="saddle",
                position=np.asarray(pos, dtype=np.float64),
                value=float(eff.get("value", 0.0)),
                angular_radius_rad=g.angular_radius_rad,
                semi_major_rad=g.semi_major_rad,
                semi_minor_rad=g.semi_minor_rad,
                principal_axis_unit=np.asarray(g.principal_axis_unit, dtype=np.float64),
            )
        )

    # De-duplicate by id: a saddle can appear in more than one source, and a
    # duplicate would be scored twice with identical geometry.
    seen: set[str] = set()
    unique: list[ConnectionTarget] = []
    for t in pool:
        if t.cp_id in seen:
            continue
        seen.add(t.cp_id)
        unique.append(t)
    unique.sort(key=lambda t: t.cp_id)
    return unique


@dataclass(frozen=True)
class ConnectionRay:
    """A great-circle ray on the atom sphere, in the centred frame.

    Attributes:
        origin_unit: Unit radial vector of the ray's start.
        direction_unit: Unit tangent at the start giving the initial bearing.
        pole_unit: Unit normal of the ray's great circle, ``origin x direction``.
    """

    origin_unit: npt.NDArray[np.float64]
    direction_unit: npt.NDArray[np.float64]
    pole_unit: npt.NDArray[np.float64]


@dataclass(frozen=True)
class MaxConnection:
    """One ascent branch and the critical point it runs into first.

    Attributes:
        saddle_id: Source saddle's id (``"SAD_142"`` / ``"E-SAD_MC7"``).
        branch_index: Index of the branch within the seed's ascent list.
        max_id: Claimed maximum's ``center_id``. Set only when ``status`` is
            ``"connected"``.
        status: ``"connected"`` (the first hit was a maximum), ``"occluded"``
            (the first hit was a minimum or a saddle, so the ray flies over a
            critical point on the way), or ``"no_target"`` (nothing in the
            forward sector).
        direction_mode: Which ray definition produced this verdict.
        cross_track_deg: Perpendicular deviation of the hit from the ray, signed
            toward the ray's left (the pole). Negative for a hit on the other
            side. None when nothing was hit.
        along_track_deg: Arc-length along the ray at which the hit's excision
            disc is first entered. This is the quantity the rule minimises, and
            for a maximum it is where the connector MEETS the basin, not where
            its centre sits.
        separation_deg: Great-circle angle saddle-to-hit.
        mutual: True when this is the tightest claim on that maximum.
        arc_points: World-frame polyline of the FINAL CONNECTOR: the single
            great circle from the source saddle's centre to the hit's centre,
            sampled on the sphere. Empty when nothing was hit. The port and the
            boundary crossing are search internals and are not vertices here.
            A branch the gradient climb resolved carries the TRACED RIDGE LINE
            here instead, and a branch the climb could not resolve carries the
            part of the ridge it did travel -- the ascent to the highest point
            reachable from the port, which is what the operator asked to see.
        stall_point: World-frame position where a gradient climb ended when it
            reached no maximum: the highest point the ascent found. None for
            every other branch. This is the evidence for "the topology and the
            gradient disagree here" -- the path is drawn, the endpoint is named,
            and no correction is implied.
        transited_ids: Excision REGIONS the gradient climb crossed through, in
            crossing order. A transit is a straight line drawn over ground the
            field cannot speak for, so the record names that ground: the connector
            is continuous, and the segment that is not field-derived is
            attributable. Empty for every branch that never entered a region it
            had to leave.
        completed_leg_deg: Length of the straight leg that closed the last gap to
            the maximum. Set when the ascent CONVERGED on a maximum but did not
            enter its disc: a sharp peak makes a one-pitch integrator overshoot,
            land on the far side, and orbit just outside the disc. The completion is
            TOTAL -- every gradient path is committed to the maximum it converged on,
            at any distance -- so this field records how far that carry was, and the
            leg is drawn rather than integrated. See ``separatrix_tracer``.
        saddle_value: Field value at the saddle.
        max_value: Field value at the claimed maximum.
        verified: True when the source seed's ring converged on the fold order
            the reduction reports (``index_agrees``).
        blocker_id: Id of the critical point that occluded this branch, and its
            polarity. Recorded so "why is there no arc here?" is answerable from
            the table without re-running the geometry.
        method: ``"geodesic"`` when the great-circle ray found the endpoint, or
            ``"gradient"`` when the ray came up empty and the field was climbed
            instead. The two are drawn differently -- a geodesic connection is
            the straight connector between its endpoints, a gradient connection is
            the traced ridge line, because the wrapping IS the finding there.
        failure_gate: For an incomplete connection, which test rejected the
            maximum: ``"cone"`` (off the ray's direction), ``"cap"`` (past the
            arc-length ceiling), ``"behind"`` (negative along-track, so the ray
            points away), ``"occluded"`` (a disc the ray entered first), or
            ``"empty"`` (no maximum in the forward hemisphere at all). None when
            the geodesic succeeded. This is the diagnostic that says WHY the
            straight connector is not enough on a given field.
        refined: True when :func:`refine_connections` smoothed this connector's
            sawtooth chatter. The default (False) marks the raw ascent, which is
            the measurement; a refined record is an interpretation of it.
        refine_max_shift_deg: Largest angular distance from any output vertex of
            the smoothed curve to the traced polyline it replaces -- the cost of
            the smoothing, published not hidden.
        n_corners: Corners the smoother detected and kept sharp (pinned knots).
        n_runs: Chatter runs the fidelity smoother replaced. Zero on the coarse
            curve, which has no node set to count.
        n_nodes: Structural nodes the coarse curve passes through -- the description
            length of the branch (endpoints, knots, and points spaced NODE_PITCH
            mesh pitches along the arc between them). Zero on the fidelity smoother.
        total_turn_before_deg: Sum of point-to-point turns over the traced path.
        total_turn_after_deg: The same over the smoothed curve. The gap is the
            mesh-scale zigzag that was removed -- the number the operator judges.
    """

    saddle_id: str
    branch_index: int
    max_id: str | None
    status: str
    direction_mode: str
    cross_track_deg: float | None
    along_track_deg: float | None
    separation_deg: float | None
    mutual: bool
    arc_points: tuple[tuple[float, float, float], ...]
    saddle_value: float
    max_value: float | None
    verified: bool
    blocker_id: str | None = None
    blocker_polarity: str | None = None
    method: str = _METHOD_GEODESIC
    failure_gate: str | None = None
    stall_point: tuple[float, float, float] | None = None
    transited_ids: tuple[str, ...] = ()
    completed_leg_deg: float | None = None
    refined: bool = False
    refine_max_shift_deg: float = 0.0
    n_corners: int = 0
    n_runs: int = 0
    n_nodes: int = 0
    total_turn_before_deg: float = 0.0
    total_turn_after_deg: float = 0.0


def ray_frame(
    origin_unit: npt.NDArray[np.float64],
    direction_unit: npt.NDArray[np.float64],
) -> ConnectionRay:
    """Build the orthonormal ``(o, d, n)`` frame of a great-circle ray.

    Args:
        origin_unit: Unit radial vector of the ray's start.
        direction_unit: Unit tangent at the start.

    Returns:
        A :class:`ConnectionRay` whose pole is ``origin x direction``.
    """
    pole = np.cross(origin_unit, direction_unit)
    pole_unit = unit(pole)
    if pole_unit is None:
        # Degenerate: direction parallel to the radial. Fall back to any pole
        # perpendicular to the origin so the frame stays orthonormal.
        axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        if abs(float(np.dot(axis, origin_unit))) > 0.9:
            axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        pole_unit = unit(tangent_project(axis, origin_unit))
        if pole_unit is None:  # pragma: no cover - origin_unit is unit, so never
            pole_unit = axis
    return ConnectionRay(
        origin_unit=origin_unit, direction_unit=direction_unit, pole_unit=pole_unit
    )


def target_frame_angles(
    target_unit: npt.NDArray[np.float64],
    ray: ConnectionRay,
) -> tuple[float, float]:
    """Decompose a target direction into (along, cross) angles about a ray.

    ``along`` is the signed progress from the ray origin toward the target,
    positive ahead of the ray; ``cross`` is the latitude out of the ray's great
    circle, signed toward the pole. Both in radians.

    Args:
        target_unit: Unit radial vector of the candidate.
        ray: The ray frame.

    Returns:
        Tuple of (along_rad, cross_rad).
    """
    along = math.atan2(
        float(np.dot(target_unit, ray.direction_unit)),
        float(np.dot(target_unit, ray.origin_unit)),
    )
    cross = math.asin(
        float(np.clip(float(np.dot(target_unit, ray.pole_unit)), -1.0, 1.0))
    )
    return along, cross


def directional_support_radius(
    centroid_unit: npt.NDArray[np.float64],
    semi_major_rad: float,
    semi_minor_rad: float,
    principal_axis_unit: npt.NDArray[np.float64],
    target_unit: npt.NDArray[np.float64],
) -> float:
    """Angular radius of a fitted ellipse toward a target direction.

    Evaluates the polar form of the boundary ellipse,

        r(phi) = sqrt((a cos phi)^2 + (b sin phi)^2),

    at the bearing of ``target_unit`` on the tangent plane at ``centroid_unit``,
    with ``phi`` folded to [0, 90] degrees because the principal axis is an
    undirected line.

    Why the disc must be directional here. An excision boundary around an
    elongated set is an ELLIPSE, not a circle: a bridged corridor has a
    semi-major along the chain and a semi-minor across it. Treating it as a disc
    of radius ``a`` inflates the transverse extent by the aspect ratio, and the
    occlusion test then blocks rays that pass well clear of the set. Measured on
    Pd_20K shape index: corridor MS13 has a = 11.35, b = 2.43 (aspect 10.6). A
    ray to MAX_4 passed it at cross-track 3.58 deg -- outside the 2.78 deg
    transverse radius, inside the 11.35 deg isotropic one -- so the corridor
    occluded a maximum it never touched, and the eight raw maxima around the
    pattern lost one connection each.

    This is the same failure the bridging gap test already fixed for the same
    reason (``string_aggregator.cluster_directional_support_radius``), whose
    docstring names it: an isotropic radius "turns a long thin corridor into a
    disk of radius a and lets it swallow compact fragments sitting far off to
    the side". That function delegates here so the ellipse is evaluated in
    exactly one place.

    Args:
        centroid_unit: Unit radial of the ellipse centre, centred frame.
        semi_major_rad: Semi-major angular radius, radians.
        semi_minor_rad: Semi-minor angular radius, radians.
        principal_axis_unit: Major-axis direction (need not be normalised or
            tangent-projected; it is made tangent here).
        target_unit: Direction to evaluate toward, centred frame.

    Returns:
        Angular radius in radians toward the target. Falls back to the
        semi-major whenever the bearing or axis is undefined, which is the
        conservative choice: it never shrinks the reach on ambiguous geometry.
    """
    a = float(semi_major_rad)
    b = float(semi_minor_rad)
    if a <= 0.0:
        return 0.0
    if b <= 0.0:
        return a
    if abs(a - b) <= 1e-12:
        return a

    c = unit(centroid_unit)
    if c is None:
        return a
    t = np.asarray(target_unit, dtype=np.float64)
    bearing = unit(tangent_project(t, c))
    if bearing is None:
        return a
    axis = unit(tangent_project(np.asarray(principal_axis_unit, dtype=np.float64), c))
    if axis is None:
        return a

    cos_phi = abs(float(np.dot(bearing, axis)))
    cos2 = min(max(cos_phi * cos_phi, 0.0), 1.0)
    sin2 = max(0.0, 1.0 - cos2)
    return math.sqrt((a * a) * cos2 + (b * b) * sin2)


def disc_entry_angle(
    along_center_rad: float,
    cross_rad: float,
    disc_radius_rad: float,
) -> float | None:
    """Where a ray first touches a critical point's excision disc.

    A CP is a disc of angular radius ``rho`` about its centre, not a point. On
    the ray's great circle, a point at arc-length ``s`` from the origin sits at
    angular distance ``dist(s)`` from the centre with

        cos dist(s) = cos(c) * cos(s - a)

    where ``a`` is the centre's along-track and ``c`` its cross-track. Setting
    ``dist = rho`` gives the entry arc-length below. The solution exists exactly
    when ``|c| <= rho``: the great circle passes through the disc.

    Args:
        along_center_rad: Along-track angle of the disc centre.
        cross_rad: Cross-track angle of the disc centre (signed).
        disc_radius_rad: Angular radius of the disc. 0.0 degenerates to the
            centre point, whose entry is the centre itself.

    Returns:
        The arc-length at which the ray enters the disc, or None when the great
        circle misses the disc entirely.
    """
    if disc_radius_rad <= 0.0:
        return along_center_rad
    cos_c = math.cos(cross_rad)
    if cos_c <= 1e-15:
        return None
    ratio = math.cos(disc_radius_rad) / cos_c
    if ratio > 1.0:
        return None  # the great circle passes outside the disc
    return along_center_rad - math.acos(max(ratio, -1.0))


def connection_ray(
    seed: SaddleLaunchSeed,
    branch: LaunchBranch,
    direction_mode: DirectionMode,
    sphere_radius: float,
) -> ConnectionRay | None:
    """Build the great-circle ray of one launch branch.

    Args:
        seed: The source saddle.
        branch: One of the seed's ascent branches.
        direction_mode: ``"chord"`` starts at the saddle centre and aims at the
            port; ``"port_tangent"`` starts at the port along the port's own
            outward tangent.
        sphere_radius: Sphere radius, used only to normalise positions.

    Returns:
        The ray frame, or None when the geometry is degenerate (a centre at the
        nucleus, or a zero-length direction with no usable fallback).
    """
    centre = np.asarray(seed.saddle_position, dtype=np.float64)
    centre_unit = unit(centre)
    if centre_unit is None:
        return None
    port = np.asarray(branch.launch_position, dtype=np.float64)
    port_unit = unit(port)

    if direction_mode == "port_tangent":
        origin = port_unit if port_unit is not None else centre_unit
        raw = np.asarray(branch.tangent_direction, dtype=np.float64)
        direction = unit(tangent_project(raw, origin))
        if direction is None:
            return None
        return ray_frame(origin, direction)

    # "chord": the bearing from the saddle centre to the port. A port that
    # coincides with the centre carries no bearing, so fall back to the
    # branch's own tangent transported to the centre.
    if port_unit is not None:
        direction = unit(tangent_project(port_unit, centre_unit))
        if direction is not None:
            return ray_frame(centre_unit, direction)
    raw = np.asarray(branch.tangent_direction, dtype=np.float64)
    direction = unit(tangent_project(raw, centre_unit))
    if direction is None:
        return None
    return ray_frame(centre_unit, direction)


def seed_ray_direction(
    seed: SaddleLaunchSeed,
    branch: LaunchBranch,
    direction_mode: DirectionMode = "port_tangent",
    sphere_radius: float = 1.0,
) -> tuple[float, float, float] | None:
    """Unit initial bearing of one ascent branch, in the centred frame.

    A thin accessor over :func:`connection_ray` for callers (and tests) that
    want only the direction.

    Args:
        seed: The source saddle.
        branch: One of the seed's ascent branches.
        direction_mode: Which ray definition to use.
        sphere_radius: Sphere radius.

    Returns:
        The unit direction as a float tuple, or None when degenerate.
    """
    ray = connection_ray(seed, branch, direction_mode, sphere_radius)
    if ray is None:
        return None
    d = ray.direction_unit
    return (float(d[0]), float(d[1]), float(d[2]))


def _saddle_to_port_arc(
    saddle_unit: npt.NDArray[np.float64],
    port_unit: npt.NDArray[np.float64],
    sphere_radius: float,
    center: tuple[float, float, float] | None,
) -> tuple[tuple[float, float, float], ...]:
    """Sample the geodesic bridge from saddle center to launch port.

    The bridge is a great-circle arc from the saddle's effective CP center to
    the port on its excision boundary, both in the centred atom frame. This
    makes the full arc path visible: saddle center → (bridge) → port → (traced
    ridge) → target, instead of only showing the ridge-traced portion.

    Args:
        saddle_unit: Unit radial of the saddle CP center, centred frame.
        port_unit: Unit radial of the launch port, centred frame.
        sphere_radius: Atom sphere radius.
        center: World-frame nucleus. Output points are converted to world frame
            if center is not None; otherwise they stay in centred frame.

    Returns:
        ARC_SAMPLES + 1 points sampled on the geodesic from saddle to port,
        in world frame (if center given) or centred frame (if center is None).
        Endpoints (saddle center and port) are included.
    """
    span = angular_distance(saddle_unit, port_unit)

    def _emit(local: tuple[float, float, float]) -> tuple[float, float, float]:
        return to_global(local, center) if center is not None else local

    direction = unit(tangent_project(port_unit, saddle_unit))

    if direction is None or span <= 1e-12:
        # Degenerate: saddle and port are coincident (should not happen in
        # practice, but handle gracefully by returning both endpoints).
        start = _emit(radial_offset(saddle_unit, sphere_radius))
        end = _emit(radial_offset(port_unit, sphere_radius))
        return (start, end)

    out: list[tuple[float, float, float]] = []
    for i in range(ARC_SAMPLES + 1):
        frac = i / ARC_SAMPLES
        local = rotate_towards(saddle_unit, direction, span * frac, sphere_radius)
        out.append(_emit(local))
    return tuple(out)


def _arc_between(
    origin_unit: npt.NDArray[np.float64],
    target_unit: npt.NDArray[np.float64],
    sphere_radius: float,
    center: tuple[float, float, float] | None,
    bearing_unit: npt.NDArray[np.float64] | None = None,
    band_rad: float = 0.0,
) -> tuple[tuple[float, float, float], ...]:
    """Sample the great-circle arc from a saddle to a maximum, in world frame.

    The connector is the shortest geodesic between the two endpoints, which is
    what "straight line on the sphere" means. It is built with the exponential
    map so every sample lies exactly on the sphere.

    Near the antipode the shortest geodesic is not unique: every great circle
    through the two points has the same length, so the centre-to-target formula
    cannot say which one the branch travelled, and two branches of one saddle that
    claim one maximum draw as ONE line. When ``bearing_unit`` is supplied and the
    separation is within ``band_rad`` of 180 deg, the arc is built on the great
    circle through the branch's own launch port instead -- the bearing the search
    actually used, which is well conditioned because the port is a full ring angle
    from the centre. Outside the band the bearing is ignored, so ordinary
    connectors are unchanged to the last bit.

    Args:
        origin_unit: Unit radial of the saddle.
        target_unit: Unit radial of the maximum.
        sphere_radius: Sphere radius.
        center: World-frame nucleus, or None for an origin-centred mesh.
        bearing_unit: Unit radial of the branch's launch port, giving the
            centre->port bearing to fall back on near the antipode. None (the
            default) always uses the centre-to-target geodesic.
        band_rad: Antipodal band half-width, in radians.

    Returns:
        ``ARC_SAMPLES + 1`` world-frame points, endpoints included. The last point
        is the maximum's centre, except inside the antipodal band where it is the
        centre's closest approach on the branch's own great circle (see above).
    """
    span = angular_distance(origin_unit, target_unit)

    def _emit(local: tuple[float, float, float]) -> tuple[float, float, float]:
        return to_global(local, center) if center is not None else local

    direction = unit(tangent_project(target_unit, origin_unit))

    # Antipodal and a branch bearing is available: route the arc through the
    # port's great circle, which is the one this branch's ray lives on. Project
    # the target onto it so the connector still ENDS at the maximum.
    if bearing_unit is not None and band_rad > 0.0 and span >= math.pi - band_rad:
        port_dir = unit(tangent_project(bearing_unit, origin_unit))
        pole = None if port_dir is None else unit(np.cross(origin_unit, port_dir))
        on_circle = (
            None
            if pole is None
            else unit(target_unit - float(np.dot(target_unit, pole)) * pole)
        )
        if port_dir is not None and on_circle is not None:
            bearing_span = angular_distance(origin_unit, on_circle)
            samples: list[tuple[float, float, float]] = []
            for i in range(ARC_SAMPLES + 1):
                frac = i / ARC_SAMPLES
                local = rotate_towards(
                    origin_unit, port_dir, bearing_span * frac, sphere_radius
                )
                samples.append(_emit(local))
            return tuple(samples)

    if direction is None or span <= 1e-12:
        # Coincident endpoints (or a target at the antipode of the projection):
        # there is no unique geodesic, so draw the straight chord between them.
        start = _emit(radial_offset(origin_unit, sphere_radius))
        end = _emit(radial_offset(target_unit, sphere_radius))
        return (start, end)

    out: list[tuple[float, float, float]] = []
    for i in range(ARC_SAMPLES + 1):
        frac = i / ARC_SAMPLES
        local = rotate_towards(origin_unit, direction, span * frac, sphere_radius)
        out.append(_emit(local))
    return tuple(out)


def connect_seed_to_maxima(
    seeds: list[SaddleLaunchSeed],
    targets: list[ConnectionTarget],
    sphere_radius: float,
    delta_theta_mesh_rad: float,
    center: tuple[float, float, float] | None = None,
    max_arc_deg: float = MAX_ARC_DEG,
    cone_pitch: float = CONE_PITCH,
    direction_mode: DirectionMode = "port_tangent",
    mesh_pts: npt.NDArray[np.float64] | None = None,
    f_vals: npt.NDArray[np.float64] | None = None,
    find_k_nearest: KNearestFinder | None = None,
    excision_regions: Sequence[ExcisionRegion] = (),
    capture_pitch: float | None = None,
) -> list[MaxConnection]:
    """Find the first critical point each ascent branch runs into.

    Args:
        seeds: Saddle launch seeds, in the centred atom frame. Both the verified
            and the provisional (``k_mismatch``) sets are usually passed, so the
            connection table matches what the scene draws; each record carries
            ``verified`` so the two stay distinguishable.
        targets: EVERY visible critical point -- maxima, minima and saddles --
            each carrying its excision-disc radius. A branch that flies over a
            minimum toward a distant maximum is occluded, which the rule can only
            see if the minimum is in the pool. Build with
            :func:`build_connection_pool`.
        sphere_radius: Atom sphere radius R.
        delta_theta_mesh_rad: Intrinsic angular mesh pitch, to read the cap and
            the sector half-angle.
        center: World-frame nucleus. Required when ``targets`` carry world
            positions (they do, coming from ``extract_basin_centers``); the
            output arcs are then emitted in the world frame too.
        max_arc_deg: Arc-length cap in DEGREES (an angular reach, not a mesh
            multiple -- see the constant).
        cone_pitch: Forward-sector half-angle in mesh-pitch units.
        direction_mode: ``"port_tangent"`` (default) launches from the port on
            the excision boundary; ``"chord"`` launches from the saddle centre.
            See the module docstring for why the port is the origin.
        mesh_pts: (N, 3) mesh vertices, centred frame. With ``f_vals`` this enables
            the gradient fallback for branches the ray cannot resolve. Omit both
            to keep the pure geodesic rule (the unit tests do exactly that).
        f_vals: (N,) condensed field values on ``mesh_pts``.
        find_k_nearest: Spatial index over ``mesh_pts`` for the ascent's IDW
            samples. Strongly recommended alongside the two above.
        excision_regions: Noisy discs the gradient climb may cross but must not
            END inside -- the micro-cluster excision boundaries, built with
            :func:`separatrix_tracer.excision_regions`. An ascent that terminates
            in one is transited straight through to the far rim and resumes
            climbing, because a termination inside a boundary is a measurement
            taken in the noise the boundary exists to exclude. Empty (the
            default) keeps the pure Step 4 ascent, which is the contract the
            geodesic unit tests pin.
        capture_pitch: Limit the completion to a maximum whose disc is within this
            many mesh pitches of the terminal sample. None (the default) is TOTAL:
            every gradient path is committed to the maximum it converged on, so no
            ascent is left uncommitted. 0.0 disables the completion and restores
            the pure Step 4 record, where a stall is reported unresolved with its
            path and gate intact -- which is what the record-contract tests pin.

    Returns:
        One :class:`MaxConnection` per (seed, ascent branch), in input order.
        Deterministic: the winner depends only on the candidate SET, never on
        its order.
    """
    if direction_mode not in DIRECTION_MODES:
        raise ValueError(
            f"direction_mode {direction_mode!r} must be one of {DIRECTION_MODES}"
        )
    max_arc_rad = math.radians(max_arc_deg)
    cone_rad = cone_pitch * delta_theta_mesh_rad

    prepared: list[ConnectionTarget] = []
    for t in targets:
        pos = np.asarray(t.position, dtype=np.float64)
        local = to_local(pos, center) if center is not None else pos
        u = unit(local)
        if u is None:
            continue
        # The principal axis is a DIRECTION, so it is translation invariant and
        # needs no to_local. It is normalised here because radius_toward only
        # tests it for zero-length and then projects it.
        axis = np.asarray(t.principal_axis_unit, dtype=np.float64)
        axis_unit = unit(axis)
        prepared.append(
            ConnectionTarget(
                cp_id=str(t.cp_id),
                polarity=t.polarity,
                position=u,
                value=t.value,
                angular_radius_rad=float(t.angular_radius_rad),
                semi_major_rad=float(t.semi_major_rad),
                semi_minor_rad=float(t.semi_minor_rad),
                principal_axis_unit=(
                    axis_unit
                    if axis_unit is not None
                    else np.zeros(3, dtype=np.float64)
                ),
            )
        )
    # Order the candidate list by id so a tie in the metrics resolves the same
    # way regardless of how the caller assembled the list.
    prepared.sort(key=lambda t: t.cp_id)

    out: list[MaxConnection] = []
    for seed in seeds:
        centre = np.asarray(seed.saddle_position, dtype=np.float64)
        centre_unit = unit(centre)
        for branch in seed.ascent_branches:
            ray = connection_ray(seed, branch, direction_mode, sphere_radius)
            if ray is None or centre_unit is None:
                # connection_ray returns None exactly when the centre is at the
                # nucleus, so the second test is the type narrowing for the
                # connector below; both are the same degenerate seed.
                out.append(_no_target(seed, branch, direction_mode))
                continue

            # WALK THE RAY OUTWARD, in increasing along-track, and stop at the
            # first critical point that is either a maximum (a target) or a disc
            # the ray actually enters (an obstacle).
            #
            # The disc test applies to OBSTACLES ONLY. A minimum or saddle blocks
            # the ray only if the great circle runs into its excision boundary:
            # measured on Pd_20K curvedness, MC18's branch 0 was blocked by
            # MIN_19165 at cross -7.41 with a 1.61 deg disc -- the ray sails past
            # it, so it is not an obstacle, and E-MAX_MC38 at cross -0.39 is the
            # true hit.
            #
            # A MAXIMUM is captured by the CONE, not by its disc. Requiring the
            # ray to hit the maximum's own boundary is tuned backwards: every raw
            # maximum's excision radius is one mesh pitch (1.61 deg on Pd_20K),
            # while the legitimate rays into a 3-saddle maximum arrive at
            # cross-tracks 0.0, 1.35, 1.42 (MAX_7) and 0.0, 2.00, 1.98 (MAX_4) --
            # the gate sits inside the measurement-noise band, so it accepted
            # MAX_7's three connections and rejected two of MAX_4's. The pattern
            # is symmetric; only the gate's luck is not. The cone is the
            # directional criterion, and the disc is where the arc STOPS.
            #
            # rho is the boundary radius TOWARD the ray, not the isotropic one.
            # An excision boundary around an elongated set is an ellipse, and
            # using its semi-major as a disc inflates the transverse reach by the
            # aspect ratio: on Pd_20K shape index the corridor MS13 (a=11.35,
            # b=2.43) occluded a ray passing 3.58 deg off its axis, because
            # 3.58 < 11.35 while 3.58 > 2.78. See directional_support_radius.
            #
            # The Entry 043 rule minimised |cross| with along as a tie-break,
            # which scored aim without checking reach: a ray through a near
            # maximum toward a far one has cross ~ 0 for BOTH, so cross-track
            # cannot separate them and the tie-break was float noise.
            candidates: list[tuple[float, float, int]] = []
            for idx, t in enumerate(prepared):
                # A branch's own saddle is not a candidate: the ray starts there.
                if t.cp_id == seed.saddle_id:
                    continue
                along, cross = target_frame_angles(t.position, ray)
                if along <= 0.0 or along > max_arc_rad:
                    continue
                if abs(cross) > cone_rad:
                    continue
                candidates.append((along, cross, idx))
            candidates.sort(key=lambda r: (r[0], abs(r[1]), r[2]))

            hit_idx = -1
            entry_rad = 0.0
            cross = 0.0
            occluder_idx = -1
            for along, cross_c, idx in candidates:
                t = prepared[idx]
                if t.polarity == _KIND_MAXIMUM:
                    # Captured by the CONE. The boundary entry is recorded as the
                    # ranking key (along_track_deg) and as where the ray meets the
                    # basin, but it is NOT a vertex of the drawn connector: the
                    # search's job is to find the two endpoints, and the connector
                    # is the single geodesic between them.
                    rho = t.radius_toward(ray.direction_unit)
                    boundary = disc_entry_angle(along, cross_c, rho)
                    hit_idx = idx
                    entry_rad = (
                        boundary if boundary is not None and boundary > 0.0 else along
                    )
                    cross = cross_c
                    break
                rho = t.radius_toward(ray.direction_unit)
                obstacle = disc_entry_angle(along, cross_c, rho)
                if obstacle is not None and obstacle > 0.0:
                    hit_idx, entry_rad, cross, occluder_idx = (
                        idx,
                        obstacle,
                        cross_c,
                        idx,
                    )
                    break

            if hit_idx < 0:
                # The ray reached nothing. Say WHICH test rejected the maxima,
                # then climb the field from the port. The gate label is what makes
                # the two-stage design auditable: it separates "the ridge curves"
                # (cone/behind) from "the ridge is long" (cap) from "there is no
                # maximum anywhere ahead" (empty).
                gate = _classify_gate(prepared, ray, cone_rad, max_arc_rad)
                traced = _gradient_connection(
                    seed=seed,
                    branch=branch,
                    gate=gate,
                    direction_mode=direction_mode,
                    prepared=prepared,
                    mesh_pts=mesh_pts,
                    f_vals=f_vals,
                    sphere_radius=sphere_radius,
                    delta_theta_mesh_rad=delta_theta_mesh_rad,
                    center=center,
                    find_k_nearest=find_k_nearest,
                    centre_unit=centre_unit,
                    excision_regions=excision_regions,
                    capture_pitch=capture_pitch,
                )
                out.append(traced)
                continue

            hit = prepared[hit_idx]
            connected = occluder_idx < 0
            out.append(
                MaxConnection(
                    saddle_id=seed.saddle_id,
                    branch_index=branch.sector_index,
                    max_id=hit.cp_id if connected else None,
                    status=_STATUS_CONNECTED if connected else _STATUS_OCCLUDED,
                    direction_mode=direction_mode,
                    cross_track_deg=math.degrees(cross),
                    along_track_deg=math.degrees(entry_rad),
                    separation_deg=math.degrees(
                        angular_distance(centre_unit, hit.position)
                    ),
                    mutual=False,
                    # THE FINAL CONNECTOR: one great circle from the saddle
                    # cluster centre to the maximum cluster centre. The ray, the
                    # port and the boundary crossing were the means of finding
                    # these two endpoints; the connector itself is the geodesic
                    # between them, so it has no kink where it meets the basin.
                    #
                    # Near the antipode that geodesic is not unique, so the
                    # branch's own port bearing picks which great circle this
                    # ascent travelled -- without it, two branches claiming one
                    # maximum draw as a single line (Entry 055).
                    arc_points=_arc_between(
                        centre_unit,
                        hit.position,
                        sphere_radius,
                        center,
                        bearing_unit=unit(
                            np.asarray(branch.launch_position, dtype=np.float64)
                        ),
                        band_rad=ANTIPODAL_BAND_RING_MULTIPLE * seed.ring_radius_rad,
                    ),
                    saddle_value=float(seed.saddle_value),
                    max_value=float(hit.value) if connected else None,
                    verified=bool(seed.index_agrees),
                    blocker_id=None if connected else hit.cp_id,
                    blocker_polarity=None if connected else hit.polarity,
                    method=_METHOD_GEODESIC,
                    failure_gate=None if connected else GATE_OCCLUDED,
                )
            )

    return mark_mutual(out)


def _classify_gate(
    prepared: list[ConnectionTarget],
    ray: ConnectionRay,
    cone_rad: float,
    max_arc_rad: float,
) -> str:
    """Name the test that kept the ray from reaching a maximum.

    Every pool maximum is scored against the ray directly -- the pre-filtered
    candidate list cannot be used, because the maxima it excludes are exactly the
    ones being diagnosed. The label makes the two-stage design auditable: "the
    ridge curves away from the ray" (behind / cone) is the phenomenon the gradient
    ascent exists to capture, while "the ridge is merely long" (cap) and "there is
    no maximum on the atom" (empty) are different statements about the field.

    Args:
        prepared: Every pool target, centred unit positions.
        ray: The branch's great-circle ray.
        cone_rad: Forward-sector half-angle, radians.
        max_arc_rad: Arc-length cap, radians.

    Returns:
        The most informative gate: ``behind`` beats ``cone`` beats ``cap`` beats
        ``empty``, because a maximum that lies behind the launch direction is the
        strongest statement of "the straight connector cannot work here".
    """
    maxima = [t for t in prepared if t.polarity == _KIND_MAXIMUM]
    if not maxima:
        return GATE_EMPTY

    behind = cone_miss = cap_miss = 0
    for t in maxima:
        along, cross = target_frame_angles(t.position, ray)
        if along <= 0.0:
            behind += 1
        elif abs(cross) > cone_rad:
            cone_miss += 1
        elif along > max_arc_rad:
            cap_miss += 1

    if behind:
        return GATE_BEHIND
    if cone_miss:
        return GATE_CONE
    if cap_miss:
        return GATE_CAP
    # A maximum sat inside the sector and the walk still ended: the only way that
    # happens is that every one of them was screened by the disc-entry test, which
    # for a target is not a rejection, so this is the empty-sector case.
    return GATE_EMPTY


def _gradient_connection(
    seed: SaddleLaunchSeed,
    branch: LaunchBranch,
    gate: str,
    direction_mode: DirectionMode,
    prepared: list[ConnectionTarget],
    mesh_pts: npt.NDArray[np.float64] | None,
    f_vals: npt.NDArray[np.float64] | None,
    sphere_radius: float,
    delta_theta_mesh_rad: float,
    center: tuple[float, float, float] | None,
    find_k_nearest: KNearestFinder | None,
    centre_unit: npt.NDArray[np.float64],
    excision_regions: Sequence[ExcisionRegion] = (),
    capture_pitch: float | None = None,
) -> MaxConnection:
    """Climb the field from the port to finish an incomplete branch.

    The ascent starts at ``branch.launch_position`` -- the port on the saddle's
    excision boundary -- never at the saddle centre, because the gradient vanishes
    at a critical point and the first step would be pure noise. The returned path
    is prepended with a geodesic bridge from the saddle center to the port,
    making the full trajectory visible: saddle center → port → traced ridge.
    See ``separatrix_tracer`` for the integrator.

    Args:
        seed: The source saddle.
        branch: The unresolved ascent branch.
        gate: Why the geodesic ray failed, from :func:`_classify_gate`.
        direction_mode: The ray definition in force, carried onto the record so
            ``direction_mode`` always names the ray (port_tangent / chord) and
            ``method == "gradient"`` is what marks the fallback.
        prepared: The pool, centred unit positions (the ascent's own frame).
        mesh_pts: Mesh vertices, centred frame, or None to disable the fallback.
        f_vals: Field values, or None to disable the fallback.
        sphere_radius: Atom sphere radius R.
        delta_theta_mesh_rad: Mesh pitch, to read the step and budget.
        center: World-frame nucleus.
        find_k_nearest: Spatial index for the ascent's IDW samples.
        centre_unit: Unit radial of the saddle, for ``separation_deg``.
        excision_regions: Noisy discs the ascent may cross but must not end
            inside. Empty keeps the pure Step 4 ascent.

    Returns:
        A connected record when the ascent arrives, otherwise ``"no_target"``
        carrying the gate and the reason the fallback could not run.
    """
    base = _no_target(seed, branch, direction_mode, gate=gate)
    if mesh_pts is None or f_vals is None:
        return base

    outcome = trace_uphill(
        start_position=branch.launch_position,
        targets=prepared,
        mesh_pts=mesh_pts,
        f_vals=f_vals,
        sphere_radius=sphere_radius,
        delta_theta_mesh_rad=delta_theta_mesh_rad,
        center=center,
        find_k_nearest=find_k_nearest,
        excision_regions=excision_regions,
        capture_pitch=capture_pitch,
    )
    if outcome.status != "arrived" or outcome.max_id is None:
        # The climb found no maximum, but the PATH is the answer to the question
        # the user asked: follow the uphill direction from the excision boundary
        # to its highest point and stop. Keep the ridge line and the endpoint on
        # the record so the viewer can draw them and the disagreement with the
        # topology is visible, not inferred from an empty cell.
        # Prepend saddle-to-port bridge to the traced path for visibility.
        port_unit = unit(np.asarray(branch.launch_position, dtype=np.float64))
        if port_unit is not None:
            bridge = _saddle_to_port_arc(
                saddle_unit=centre_unit,
                port_unit=port_unit,
                sphere_radius=sphere_radius,
                center=center,
            )
            full_path = bridge + outcome.path[1:] if outcome.path else bridge
        else:
            full_path = outcome.path
        return replace(
            base,
            arc_points=full_path,
            stall_point=outcome.path[-1] if outcome.path else None,
            along_track_deg=outcome.arc_length_deg,
            transited_ids=outcome.transited_ids,
            # The path is a traced ridge, so it draws in the gradient colour even
            # though the record is unresolved. method names how the geometry was
            # produced; status names what it found.
            method=_METHOD_GRADIENT,
        )

    hit = next((t for t in prepared if t.cp_id == outcome.max_id), None)
    if hit is None:  # pragma: no cover - the tracer only returns pool ids
        return base
    # Prepend saddle-to-port bridge to the traced path for visibility.
    port_unit = unit(np.asarray(branch.launch_position, dtype=np.float64))
    if port_unit is not None:
        bridge = _saddle_to_port_arc(
            saddle_unit=centre_unit,
            port_unit=port_unit,
            sphere_radius=sphere_radius,
            center=center,
        )
        full_path = bridge + outcome.path[1:] if outcome.path else bridge
    else:
        full_path = outcome.path
    return MaxConnection(
        saddle_id=seed.saddle_id,
        branch_index=branch.sector_index,
        max_id=hit.cp_id,
        status=_STATUS_CONNECTED,
        direction_mode=direction_mode,
        cross_track_deg=None,
        along_track_deg=outcome.arc_length_deg,
        separation_deg=math.degrees(angular_distance(centre_unit, hit.position)),
        mutual=False,
        # The traced ridge line IS the connector here. Flattening it to the
        # geodesic between its endpoints would erase the wrapping that made the
        # ray rule fail, which is the finding. The bridge is prepended to show
        # the full path from saddle center through port to the target.
        arc_points=full_path,
        saddle_value=float(seed.saddle_value),
        max_value=float(hit.value),
        verified=bool(seed.index_agrees),
        method=_METHOD_GRADIENT,
        failure_gate=gate,
        transited_ids=outcome.transited_ids,
        completed_leg_deg=outcome.completed_leg_deg,
    )


def _no_target(
    seed: SaddleLaunchSeed,
    branch: LaunchBranch,
    direction_mode: str,
    gate: str | None = None,
) -> MaxConnection:
    """Build the record for a branch that claimed nothing.

    Args:
        seed: The source saddle.
        branch: The unclaimed ascent branch.
        direction_mode: The ray definition in force.
        gate: Which geodesic test rejected the maxima, from
            :func:`_classify_gate`. Carried through so an unresolved branch is
            still attributable: "the ray pointed away and the climb stalled" is a
            different finding from "there is no maximum on this atom".

    Returns:
        A :class:`MaxConnection` with ``status == "no_target"`` and an empty arc.
    """
    return MaxConnection(
        saddle_id=seed.saddle_id,
        branch_index=branch.sector_index,
        max_id=None,
        status=_STATUS_NO_TARGET,
        direction_mode=direction_mode,
        cross_track_deg=None,
        along_track_deg=None,
        separation_deg=None,
        mutual=False,
        arc_points=(),
        saddle_value=float(seed.saddle_value),
        max_value=None,
        verified=bool(seed.index_agrees),
        failure_gate=gate,
    )


def mark_mutual(connections: list[MaxConnection]) -> list[MaxConnection]:
    """Flag the tightest claim on each maximum as mutual.

    A maximum is "claimed by" every branch whose first hit is that maximum. The
    branch with the smallest cross-track is the one the maximum points back at,
    which is the reciprocal reading of the connection. Exactly one connection per
    maximum is flagged, and only connected ones are eligible (occluded and
    no-target records carry no ``max_id``).

    Args:
        connections: The connection list, in any order.

    Returns:
        A new list with ``mutual`` set; input order and all other fields are
        preserved.
    """
    winner: dict[str, tuple[float, float, int]] = {}
    for i, conn in enumerate(connections):
        if conn.max_id is None or conn.cross_track_deg is None:
            continue
        key = (abs(conn.cross_track_deg), float(conn.along_track_deg or 0.0), i)
        if conn.max_id not in winner or key < winner[conn.max_id]:
            winner[conn.max_id] = key

    out: list[MaxConnection] = []
    for i, conn in enumerate(connections):
        if conn.max_id is None:
            out.append(conn)
            continue
        is_mutual = winner.get(conn.max_id, (None, None, -1))[2] == i
        out.append(replace(conn, mutual=is_mutual))
    return out


def refine_connections(
    connections: Sequence[MaxConnection],
    sphere_radius: float,
    delta_theta_mesh_rad: float,
    center: tuple[float, float, float] | None = None,
    regions: Sequence[ExcisionRegion] = (),
    coarse: bool = False,
) -> list[MaxConnection]:
    """Replace every gradient-traced connector with a smooth curve.

    Two descriptions of the same drawn line, selected by ``coarse``:

    - ``coarse=False`` (the fidelity smoother): preserve the ascent, remove only
      the mesh-scale chatter. Each sawtooth run is replaced by a curve at
      sub-pitch spacing, so the output hugs the traced path -- the deviation from
      it is the cost, and it stays well under one pitch.
    - ``coarse=True`` (the structural curve): ONE smooth curve through a handful of
      nodes spaced ``NODE_PITCH`` mesh pitches apart. The exact path of the ascent
      is a property of the integrator's step size; this describes the connection
      instead, and the deviation from the traced path is expected and informational.
      The node spacing is a low-pass filter's cutoff: on a connector that spirals
      half a turn inside an excision region, 1-pitch spacing reproduces the winding
      (65935 of 66010 deg of turning) and 8-pitch spacing filters it out (651 deg).

    ``coarse`` defaults to False because that is what the snapshots on disk record:
    the ethene3 files were saved with the fidelity smoother ON and their
    ``connection_stats`` reproduce it exactly, so a replay must run it by default.

    A POST-PASS, deliberately not part of :func:`connect_seed_to_maxima`. The
    connection table is a record of what the field's gradient did; the smoothing is
    an interpretation of it, and the operator must be able to look at both. Running
    it after the fact also means the viewer can toggle it without re-integrating a
    single ascent.

    Only ``method == "gradient"`` is touched. A geodesic connector is a great circle
    between two centres -- it is not an approximation of anything, so there is no
    chatter in it to remove, and smoothing it would move it OFF the line it is
    defined to be.

    The transited regions are handed to the smoother as PINNED centres. A transit is
    a straight carry across ground the field cannot speak for, so the centre is a
    position the pipeline vouches for by construction; the curve is pinned to it.

    This replaces the Entry-058 transverse nudge, which needed the field (mesh,
    values, locator) to estimate a crest and moved the path CROSS-TRACK -- a motion
    that cannot remove a tooth, because the tooth IS the cross-track motion. The
    smoother is pure geometry: it needs no field, only the pitch (to size the corner
    window and the output spacing) and the radius (to scale the radials).

    Args:
        connections: The connection table, in any order.
        sphere_radius: Atom sphere radius R.
        delta_theta_mesh_rad: Intrinsic angular mesh pitch, to size the corner
            window and the output spacing.
        center: World-frame nucleus, for the frame conversion of the polyline.
        regions: The excision regions used by the ascent, so ``transited_ids`` can
            be resolved to centres.
        coarse: True for the structural curve, False for the fidelity smoother.

    Returns:
        A new list, same order. Records that were not smoothed are returned
        unchanged (identical objects, not copies).
    """
    by_id = {r.region_id: r for r in regions}
    out: list[MaxConnection] = []
    for conn in connections:
        if conn.method != _METHOD_GRADIENT or len(conn.arc_points) < 5:
            out.append(conn)
            continue
        centres: list[tuple[float, float, float]] = []
        for rid in conn.transited_ids:
            region = by_id.get(rid)
            if region is None:
                continue
            pos = (
                float(region.centre_unit[0]) * sphere_radius,
                float(region.centre_unit[1]) * sphere_radius,
                float(region.centre_unit[2]) * sphere_radius,
            )
            centres.append(to_global(pos, center) if center else pos)
        build = structural_curve_ridge if coarse else smooth_gradient_ridge
        res = build(
            arc_points=conn.arc_points,
            sphere_radius=sphere_radius,
            delta_theta_mesh_rad=delta_theta_mesh_rad,
            center=center,
            transited_centres=centres,
        )
        if not res.changed:
            out.append(conn)
            continue
        out.append(
            replace(
                conn,
                arc_points=res.path,
                refined=True,
                refine_max_shift_deg=res.max_deviation_deg,
                n_corners=len(res.corner_indices),
                n_runs=res.n_runs,
                n_nodes=res.n_nodes,
                total_turn_before_deg=res.total_turn_before_deg,
                total_turn_after_deg=res.total_turn_after_deg,
            )
        )
    return out


def degenerate_arcs(
    connections: Sequence[MaxConnection],
    tol_deg: float = DEGENERATE_ARC_TOL_DEG,
) -> list[tuple[MaxConnection, MaxConnection]]:
    """Pairs of branches that draw the same connector.

    Two ascent branches of one saddle that claim one maximum have the same two
    endpoints, so the centre-to-centre geodesic gives them the same polyline. Away
    from the antipode that is honest -- the shortest geodesic is unique and both
    rays lie near it. Near the antipode it is a collapse: the shortest geodesic is
    NOT unique, the branches genuinely travel different ways (one over each pole),
    and the scene shows one line where there should be two.

    This is the check that catches it, and the reason a status check is not
    enough: both records report `connected`, so the table looks complete while the
    drawing is not. The comparison is on the polylines, not the endpoints, so it
    also catches a future connector that is branch-dependent by construction.

    Args:
        connections: The connection table.
        tol_deg: Maximum point-for-point deviation, in degrees, that counts as the
            same line. Absolute, not pitch-relative: this measures numerical
            identity, not a physical scale (see the constant).

    Returns:
        Pairs of connected records that share a saddle, share a maximum, and draw
        the same polyline. Empty when every branch is visually distinct.
    """
    out: list[tuple[MaxConnection, MaxConnection]] = []
    by_saddle: dict[str, list[MaxConnection]] = {}
    for c in connections:
        if c.status != _STATUS_CONNECTED or len(c.arc_points) < 2:
            continue
        by_saddle.setdefault(c.saddle_id, []).append(c)
    for group in by_saddle.values():
        for i, a in enumerate(group):
            for b in group[i + 1 :]:
                if a.max_id != b.max_id:
                    continue
                if len(a.arc_points) != len(b.arc_points):
                    continue
                dev = max(
                    math.degrees(
                        angular_distance(
                            np.asarray(pa, dtype=np.float64),
                            np.asarray(pb, dtype=np.float64),
                        )
                    )
                    for pa, pb in zip(a.arc_points, b.arc_points, strict=True)
                )
                if dev <= tol_deg:
                    out.append((a, b))
    return out


def summarize_connections(connections: list[MaxConnection]) -> dict[str, Any]:
    """Reduce a connection list to the counters the viewer card shows.

    Args:
        connections: Output of :func:`connect_seed_to_maxima`.

    Returns:
        Dict of counts and angular statistics. Every float is finite, so the
        result is safe to write into a params snapshot and diff.
    """
    connected = [c for c in connections if c.status == _STATUS_CONNECTED]
    occluded = [c for c in connections if c.status == _STATUS_OCCLUDED]
    no_target = [c for c in connections if c.status == _STATUS_NO_TARGET]
    cross = [
        abs(float(c.cross_track_deg))
        for c in connected
        if c.cross_track_deg is not None
    ]
    sep = [float(c.separation_deg) for c in connected if c.separation_deg is not None]
    return {
        "direction_mode": connections[0].direction_mode if connections else "",
        "n_branches": len(connections),
        "n_connected": len(connected),
        "n_occluded": len(occluded),
        "n_no_target": len(no_target),
        "n_saddles_connected": len({c.saddle_id for c in connected}),
        "n_maxima_connected": len({c.max_id for c in connected}),
        "n_mutual": sum(1 for c in connections if c.mutual),
        "n_unverified": sum(1 for c in connected if not c.verified),
        # The two-stage split. n_geodesic is what Step 3's straight ray resolved;
        # n_gradient is what the field climb had to finish. On a dense metallic
        # field n_gradient is 0 and the fallback is provably a no-op; on ethene3
        # it is the whole answer.
        "n_geodesic": sum(1 for c in connected if c.method == _METHOD_GEODESIC),
        "n_gradient": sum(1 for c in connected if c.method == _METHOD_GRADIENT),
        "n_unresolved": len(no_target),
        # THE COMMITMENT AUDIT. Every ascent branch must be committed: claimed by
        # a maximum, occluded by a critical point it flew over, or reported
        # unresolved with a gate naming why. n_uncommitted is the count that has
        # none of those -- it must be zero, and a non-zero value is a bug, not a
        # finding. uncommitted_branches names them so the failure is locatable
        # without re-running the geometry.
        "n_uncommitted": len(connections)
        - len(connected)
        - len(occluded)
        - len(no_target),
        "uncommitted_branches": [
            f"{c.saddle_id}/{c.branch_index}"
            for c in connections
            if c.status not in CONNECTION_STATUSES
        ],
        # The degeneracy audit, which catches what the status check cannot. Two
        # branches of one saddle that claim one maximum near the antipode have
        # identical centre-to-centre connectors, so they DRAW AS ONE LINE while
        # both report `connected`. A status check sees zero problems; this sees
        # the collapsed pair. Measured on ethene3 C1 (potential V): saddle MC15
        # and maximum MC14 are 178.01 deg apart, both branches connected, and the
        # two polylines agreed point-for-point to 0.000e+00 deg.
        "n_degenerate_arcs": len(degenerate_arcs(connections)),
        "degenerate_arcs": [
            f"{a.saddle_id}/{a.branch_index}~{b.branch_index}"
            for a, b in degenerate_arcs(connections)
        ],
        "n_gate_behind": sum(1 for c in connections if c.failure_gate == GATE_BEHIND),
        "n_gate_cone": sum(1 for c in connections if c.failure_gate == GATE_CONE),
        "n_gate_cap": sum(1 for c in connections if c.failure_gate == GATE_CAP),
        "n_gate_occluded": sum(
            1 for c in connections if c.failure_gate == GATE_OCCLUDED
        ),
        "n_gate_empty": sum(1 for c in connections if c.failure_gate == GATE_EMPTY),
        # Transit accounting: how many branches had to be carried across an
        # excision region, and which regions. A non-zero count says the ascent
        # was terminating in noise, which is the ethene3 shape-index finding.
        "n_transited": sum(1 for c in connections if c.transited_ids),
        # How many gradient climbs were COMPLETED onto a maximum they converged on
        # but stepped past (a sharp peak makes a one-pitch integrator overshoot and
        # orbit just outside the disc). The association is unambiguous; only the
        # last leg is drawn rather than measured.
        "n_completed": sum(1 for c in connections if c.completed_leg_deg),
        # The cost of a TOTAL completion rule, made visible. A long leg is a
        # measurement of the field -- the ascent converged on this maximum from far
        # away -- not a rejection to hide. Read it as: how much of the connector is
        # drawn rather than integrated.
        "max_completed_leg_deg": max(
            (float(c.completed_leg_deg) for c in connections if c.completed_leg_deg),
            default=0.0,
        ),
        # THE INVARIANT. Every branch whose connector is a traced gradient path must
        # terminate at a maximum. A gradient path left uncommitted is a bug in the
        # rule, not a property of the field -- which is the lesson of Entries 051-056,
        # where each proximity gate silently dropped the next case (MC20 -> MC12, a
        # mirror partner of the case the gate was tuned on, at 1.80 pitch against a
        # 1.0 pitch limit). Must be zero.
        "n_gradient_uncommitted": sum(
            1
            for c in connections
            if c.method == _METHOD_GRADIENT and c.status != _STATUS_CONNECTED
        ),
        # Smoothing accounting. A refined connector is a REINTERPRETATION of the
        # ascent, not a new measurement, so the count and the cost are published
        # next to the record rather than folded into it: the operator sees how much
        # of the ridge line is the field's and how much is the fit's. The turning
        # sums are the chatter's signature -- the gap between before and after is the
        # mesh-scale zigzag the smoother removed.
        "n_refined": sum(1 for c in connections if c.refined),
        "max_refine_shift_deg": max(
            (float(c.refine_max_shift_deg) for c in connections if c.refined),
            default=0.0,
        ),
        "n_refined_corners": sum(int(c.n_corners) for c in connections if c.refined),
        "n_refined_runs": sum(int(c.n_runs) for c in connections if c.refined),
        "n_curve_nodes": sum(int(c.n_nodes) for c in connections if c.refined),
        "total_turn_before_deg": sum(
            float(c.total_turn_before_deg) for c in connections if c.refined
        ),
        "total_turn_after_deg": sum(
            float(c.total_turn_after_deg) for c in connections if c.refined
        ),
        "transited_regions": sorted({r for c in connections for r in c.transited_ids}),
        "n_blocked_by_minimum": sum(
            1 for c in occluded if c.blocker_polarity == "minimum"
        ),
        "n_blocked_by_saddle": sum(
            1 for c in occluded if c.blocker_polarity == "saddle"
        ),
        "mean_cross_track_deg": sum(cross) / len(cross) if cross else 0.0,
        "max_cross_track_deg": max(cross) if cross else 0.0,
        "mean_separation_deg": sum(sep) / len(sep) if sep else 0.0,
        "max_separation_deg": max(sep) if sep else 0.0,
    }
