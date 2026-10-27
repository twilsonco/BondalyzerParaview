"""Self-Calibrating Excision Ring Search for Launch Seeds (Basin Locator Step 2).

Chooses the excision radius for each saddle, then delegates the port reading to
``excision_launch``. This module owns the *policy*; ``excision_launch`` owns the
*measurement*.

Why the radius has to be searched
---------------------------------
An excision boundary must be large enough to sit outside the degenerate core of
the singularity it encloses, and small enough to stay local. At a mesh pitch of
1.6 deg, a lone saddle's ring from the excision-radius rule lands at exactly
1 dtheta -- inside the regime where the interpolated profile is dominated by
interpolation noise rather than by the field's own structure. The tell is
topological: a ring still in the core has no clean alternating signature, so its
valley and ridge counts disagree.

So balance is used as a convergence test. The ring is expanded until the port
signature is pure -- balanced, at the fold order the reduction stage reports, and
at least 2 (a balanced 1 + 1 is an extremum, not a saddle).

Why the sweep has no early exit
-------------------------------
The search walks radii from the excision-radius rule to the ceiling and stops
only on success. An earlier revision also stopped when two consecutive radii
agreed on an inflated ``k``, to save profiling time. That is unsound: "two
consecutive radii" counts ladder rungs, and rung spacing is the expand step, so
the criterion is measured in a unit the operator controls. IDW noise is
spatially correlated, so two closely spaced rings agree almost by construction --
a fine, thorough scan terminated prematurely while a coarse one survived, and
which saddles produced seeds depended on the step size. The expand step is a
resolution knob and must not be able to change the topology that is found.

Measured on Pd_20K: E-SAD_MC22 lost its seeds at expand step 0.75 and recovered
them at 1.25, purely from this effect. Removing the early exit makes the outcome
invariant in the step, which is the property the regression test pins.

What accepting on the target k costs
------------------------------------
Expanding until ``k == expected_k`` makes the ring a *constructor* of the index
rather than a measurement of it: a genuine index disagreement and a ring that
never resolve look identical from the outside. They are separated here by
recording the observed ``k`` at the tightest radius where the loop was balanced.
A ring that balanced at k = 2 while the reduction claimed k = 3 is reported as an
index mismatch, which is a finding about the Morse detector; a ring that never
balanced at all is a degenerate-core problem. Those are different diagnoses and
they get different counters.

Frame contract: centred atom frame throughout, matching ``excision_launch``.
"""

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

from gba_topology2.src.excision_boundary import KNearestFinder
from gba_topology2.src.excision_launch import extract_excision_launch_seed
from gba_topology2.src.micro_cluster import (
    BoundaryRingPoint,
    generate_boundary_ring,
)
from gba_topology2.src.saddle_launch import SaddleLaunchSeed

__all__ = [
    "EXPAND_PITCH",
    "MAX_RADIUS_PITCH",
    "RING_SAMPLES",
    "ExpansionOutcome",
    "SeedRequest",
    "expand_excision_seed",
    "summarize_outcomes",
]

# A ring around an extremum balances at 1 + 1; it is never a saddle signature.
_MIN_SADDLE_K = 2

# Ring search geometry, fixed so the topology found does not depend on operator
# tuning. The predecessor (gba_topology.topology_engine
# .compute_adaptive_excision_regions) hardcoded the same three quantities; the
# values here are in mesh-pitch units so they scale with the grid.
#
#   RING_SAMPLES      azimuthal samples used to PROFILE a ring for launch seeds.
#                     16 resolves 4 samples per lobe at k = 4, the floor for the
#                     smoothing and prominence filters. More samples is not
#                     better: finer azimuthal sampling resolves IDW interpolation
#                     noise as spurious extrema and inflates the measured k.
#   EXPAND_PITCH      radius increment per retry.
#   MAX_RADIUS_PITCH  ceiling for the radius, ~25.7 deg on Pd_20K, matching the
#                     predecessor's 24 deg sweep.
#
# The boundary rings the viewer DRAWS keep their own 48-sample count
# (num_boundary_points in compute_micro_clusters): that is a rendering
# resolution, and dropping it to 16 would draw visible 16-gons.
RING_SAMPLES = 16
EXPAND_PITCH = 0.5
MAX_RADIUS_PITCH = 16.0


@dataclass(frozen=True)
class SeedRequest:
    """One saddle needing launch seeds, with the ring its radius rule produced.

    Attributes:
        saddle_id: Identifier for the seed and for the failure list (e.g.
            ``"SAD_142"``, ``"E-SAD_MC7"``).
        centre: Saddle position in the centred atom frame.
        saddle_value: Field value at the saddle, the crossing level.
        expected_k: Fold order the reduction reports: ``multiplicity + 1`` for a
            raw saddle, ``1 - net_index`` for an effective one.
        a0_rad: Initial excision radius from the excision-radius rule.
        vertex_id: Mesh vertex index, -1 for an effective CP.
        multiplicity: Morse detector multiplicity, 0 for an effective CP.
        ring_points: Pre-generated ring at ``a0_rad``, reused when the rule is
            already right so the common case costs one profile.
    """

    saddle_id: str
    centre: tuple[float, float, float]
    saddle_value: float
    expected_k: int
    a0_rad: float
    vertex_id: int = -1
    multiplicity: int = 0
    ring_points: list[BoundaryRingPoint] = field(default_factory=list)


@dataclass(frozen=True)
class ExpansionOutcome:
    """Result of the radius search for one saddle.

    Attributes:
        seed: The converged seed, or the last seed attempted on failure.
        status: ``"converged_at_a0"``, ``"expanded"``, ``"unbalanced"``,
            ``"k_mismatch"`` or ``"no_ring"``.
        radii_tried: Number of ring radii profiled.
        a_final_rad: Radius the search stopped at.
        observed_k: Fold order measured at the tightest balanced radius, 0 if the
            loop never balanced. This is what separates an index disagreement
            from a degenerate core.
        expected_k: The fold order the search was looking for.
    """

    seed: SaddleLaunchSeed
    status: str
    radii_tried: int
    a_final_rad: float
    observed_k: int
    expected_k: int

    @property
    def converged(self) -> bool:
        """Whether a usable seed was produced."""
        return self.status in ("converged_at_a0", "expanded")


def _ring_at(
    request: SeedRequest,
    radius_rad: float,
    sphere_radius: float,
    num_samples: int,
    a0_rad: float,
) -> list[BoundaryRingPoint]:
    """Ring samples for one radius, reusing the caller's ring at the initial radius."""
    if abs(radius_rad - a0_rad) < 1e-15 and request.ring_points:
        return request.ring_points
    return generate_boundary_ring(
        centroid=request.centre,
        sphere_radius=sphere_radius,
        angular_radius_rad=radius_rad,
        num_samples=num_samples,
    )


def expand_excision_seed(
    request: SeedRequest,
    mesh_pts: npt.NDArray[np.float64],
    f_vals: npt.NDArray[np.float64],
    sphere_radius: float,
    delta_theta_mesh_rad: float,
    find_k_nearest: KNearestFinder | None = None,
    expand_pitch: float = EXPAND_PITCH,
    max_radius_pitch: float = MAX_RADIUS_PITCH,
    num_samples: int = RING_SAMPLES,
    smoothing_window: int = 3,
    min_prominence_frac: float = 0.05,
) -> ExpansionOutcome:
    """Expand a saddle's excision ring until its port signature is pure.

    Accepts at the first radius that is balanced, has fold order equal to
    ``expected_k``, and has ``k >= 2``. The sweep has no early exit: every radius
    down the ladder is profiled until one succeeds, so the expand step is a pure
    resolution knob that cannot change which saddles seed (see the module
    docstring for why an earlier version's stability exit was unsound). The
    step, ceiling and sample count default to module constants -- the viewer
    passes none of them -- and remain arguments so tests can vary them.

    Args:
        request: The saddle and its initial ring.
        mesh_pts: (N, 3) vertex coordinates in the centred atom frame.
        f_vals: (N,) scalar field values.
        sphere_radius: Atom sphere radius R.
        delta_theta_mesh_rad: Intrinsic angular mesh pitch.
        find_k_nearest: Spatial index over ``mesh_pts``; strongly recommended,
            since the search profiles up to a dozen rings per saddle.
        expand_pitch: Radius increment, in mesh-pitch units.
        max_radius_pitch: Ceiling for the radius, in mesh-pitch units.
        num_samples: Circumferential samples per ring.
        smoothing_window: Moving-average window for the loop profile.
        min_prominence_frac: Port prominence threshold fraction.

    Returns:
        An ``ExpansionOutcome``. On failure the seed is the last one attempted,
        so the caller keeps the saddle's identity and the final radius for
        reporting.
    """
    if request.expected_k < _MIN_SADDLE_K:
        # A saddle of fold order 1 is not a saddle; nothing to search for.
        seed = extract_excision_launch_seed(
            ring_points=[],
            mesh_pts=mesh_pts,
            f_vals=f_vals,
            saddle_id=request.saddle_id,
            saddle_position=request.centre,
            saddle_value=request.saddle_value,
            expected_k=request.expected_k,
            vertex_id=request.vertex_id,
            multiplicity=request.multiplicity,
        )
        return ExpansionOutcome(seed, "no_ring", 0, 0.0, 0, request.expected_k)

    a0 = max(float(request.a0_rad), 0.25 * delta_theta_mesh_rad)
    ceiling = max(float(max_radius_pitch), 0.0) * delta_theta_mesh_rad
    step = max(float(expand_pitch), 0.05) * delta_theta_mesh_rad

    radii: list[float] = []
    r = a0
    while r <= ceiling + 1e-15:
        radii.append(r)
        if r >= ceiling:
            break
        r += step
    if not radii:
        radii = [a0]

    best_k_at_balance = 0
    last_seed: SaddleLaunchSeed | None = None
    last_radius = a0

    for i, radius in enumerate(radii):
        ring = _ring_at(request, radius, sphere_radius, num_samples, a0)
        seed = extract_excision_launch_seed(
            ring_points=ring,
            mesh_pts=mesh_pts,
            f_vals=f_vals,
            saddle_id=request.saddle_id,
            saddle_position=request.centre,
            saddle_value=request.saddle_value,
            expected_k=request.expected_k,
            vertex_id=request.vertex_id,
            multiplicity=request.multiplicity,
            find_k_nearest=find_k_nearest,
            smoothing_window=smoothing_window,
            min_prominence_frac=min_prominence_frac,
        )
        last_seed = seed
        last_radius = radius
        tried = i + 1

        if seed.failure_reason is not None:
            if seed.failure_reason == "unbalanced":
                continue
            # empty_ring / too_few_samples / no_ports / below_min_k: the loop has
            # no structure at all yet, so a larger radius is the remedy.
            continue

        # Balanced with k >= 2. Record the tightest such measurement: it is the
        # evidence that separates an index disagreement from a degenerate core.
        if best_k_at_balance == 0:
            best_k_at_balance = seed.fold_order

        measured = seed.fold_order
        if measured == request.expected_k:
            status = "converged_at_a0" if tried == 1 else "expanded"
            return ExpansionOutcome(
                seed, status, tried, radius, measured, request.expected_k
            )

        # Any other reading keeps expanding. There is deliberately no early
        # exit: the sweep IS the search, and terminating on a "stable" inflated
        # count made the outcome depend on the step size, because IDW noise is
        # spatially correlated and two closely spaced radii agree almost by
        # construction. See the module docstring.

    observed = best_k_at_balance
    status = "k_mismatch" if observed > 0 else "unbalanced"
    # radii is non-empty, so the loop body ran and last_seed is set.
    seed = last_seed if last_seed is not None else _empty_seed(request)
    return ExpansionOutcome(
        seed, status, len(radii), last_radius, observed, request.expected_k
    )


def _empty_seed(request: SeedRequest) -> SaddleLaunchSeed:
    """A placeholder seed for a search that never profiled a ring."""
    return SaddleLaunchSeed(
        saddle_id=request.saddle_id,
        saddle_position=request.centre,
        saddle_value=request.saddle_value,
        vertex_id=request.vertex_id,
        multiplicity=int(request.multiplicity),
        fold_order=0,
        index_agrees=False,
    )


def summarize_outcomes(outcomes: list[ExpansionOutcome]) -> dict[str, Any]:
    """Aggregate expansion outcomes into the viewer's launch statistics.

    Args:
        outcomes: One outcome per saddle that was searched.

    Returns:
        Dict with the seed counts, branch counts, the five-way expansion
        taxonomy, the maximum fold order, and the ids of saddles whose ring
        failed to converge, so a flagged ring is findable in the scene.
    """
    seeds = [o.seed for o in outcomes if o.converged]
    counts: dict[str, Any] = {
        "n_saddles_seeded": len(seeds),
        "n_ascent": sum(len(s.ascent_branches) for s in seeds),
        "n_descent": sum(len(s.descent_branches) for s in seeds),
        "max_fold_order": max((s.fold_order for s in seeds), default=0),
        "n_rings_converged_at_a0": 0,
        "n_rings_expanded": 0,
        "n_rings_unbalanced": 0,
        "n_rings_k_mismatch": 0,
        "n_rings_no_ring": 0,
    }
    failures: list[dict[str, Any]] = []
    radii_tried: list[int] = []

    for o in outcomes:
        radii_tried.append(o.radii_tried)
        if o.status == "converged_at_a0":
            counts["n_rings_converged_at_a0"] += 1
        elif o.status == "expanded":
            counts["n_rings_expanded"] += 1
        elif o.status == "unbalanced":
            counts["n_rings_unbalanced"] += 1
        elif o.status == "k_mismatch":
            counts["n_rings_k_mismatch"] += 1
        elif o.status == "no_ring":
            counts["n_rings_no_ring"] += 1

        if not o.converged:
            failures.append(
                {
                    "id": o.seed.saddle_id,
                    "status": o.status,
                    "expected_k": o.expected_k,
                    "observed_k": o.observed_k,
                    "radii_tried": o.radii_tried,
                    "final_radius_deg": float(math.degrees(o.a_final_rad)),
                }
            )

    counts["n_index_mismatch"] = sum(
        1 for s in seeds if not s.index_agrees and s.fold_order > 0
    )
    counts["n_rings_failed"] = len(failures)
    counts["max_radii_tried"] = max(radii_tried, default=0)
    counts["mean_radii_tried"] = (
        sum(radii_tried) / len(radii_tried) if radii_tried else 0.0
    )
    counts["failures"] = failures[:40]
    return counts
