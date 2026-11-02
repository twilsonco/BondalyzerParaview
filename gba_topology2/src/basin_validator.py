"""Basin Boundary Validator for gba_topology2.

Validates that simplified CP configurations satisfy the topological constraint:
- Max basin boundary: alternating minimum and saddle (effective) CPs, forming
  a single closed cycle around the maximum.
- Min basin boundary: alternating maximum and saddle (effective) CPs, forming
  a single closed cycle around the minimum.

The boundary is extracted from the Morse-Smale complex, built by matching each
saddle's separatrix launch branches (ascent → maxima, descent → minima) to the
nearest visible basin center using tangent-direction projection.

Boundary validity requires:
1. Equal number of saddles and opposite-polarity extrema in the boundary.
2. Each opposite-polarity extremum appears in exactly 2 boundary saddles
   (degree-2 condition in the bipartite boundary graph).
3. The boundary graph forms a single connected cycle (not multiple loops).

Two entry points are provided:
- ``build_morse_smale_adjacency``: builds saddle→max/min mapping from seeds.
- ``validate_all_basin_boundaries``: runs the full validation.

Utility functions ``is_valid_max_basin_boundary`` and
``is_valid_min_basin_boundary`` operate on pre-ordered type sequences for unit
testing and fallback use.

Subroutine contract per AGENTS.md:
- Pure algorithmic computation (NumPy, standard library).
- Standalone testable via simple unit tests.
- Deterministic and fully specified.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gba_topology2.src.basin_center import BasinCenter
from gba_topology2.src.saddle_launch import SaddleLaunchSeed

__all__ = [
    "BasinBoundaryResult",
    "SaddleAdjacency",
    "SimplificationValidity",
    "build_morse_smale_adjacency",
    "is_valid_max_basin_boundary",
    "is_valid_min_basin_boundary",
    "validate_all_basin_boundaries",
    "validate_max_basin_boundary",
    "validate_min_basin_boundary",
]

# saddle_id → {'max': [max_id1, max_id2], 'min': [min_id1, min_id2]}
SaddleAdjacency = dict[str, dict[str, list[str]]]


@dataclass(frozen=True)
class BasinBoundaryResult:
    """Validation result for one basin's boundary.

    Attributes:
        basin_id: Identifier of the basin center CP.
        center_polarity: ``"maximum"`` or ``"minimum"``.
        boundary_saddle_ids: IDs of the saddles forming the basin boundary.
        boundary_extremum_ids: IDs of the opposite-polarity extrema in the
            boundary, in the order they appear interleaved with saddles.
        is_valid: True when the boundary forms a valid alternating cycle.
        n_saddles: Number of saddles in the boundary.
        n_extrema: Number of opposite-polarity extrema in the boundary.
        error: Short description of the failure, or empty string when valid.
    """

    basin_id: str
    center_polarity: str
    boundary_saddle_ids: list[str]
    boundary_extremum_ids: list[str]
    is_valid: bool
    n_saddles: int
    n_extrema: int
    error: str


@dataclass(frozen=True)
class SimplificationValidity:
    """Overall validity assessment of a simplified CP configuration.

    Attributes:
        is_valid: True if all basin boundaries satisfy the alternating cycle
            constraint.
        n_max_valid: Number of maximum basins with valid boundaries.
        n_min_valid: Number of minimum basins with valid boundaries.
        n_max_total: Total maximum basins evaluated.
        n_min_total: Total minimum basins evaluated.
        results: BasinBoundaryResult for every basin evaluated.
        failing_ids: basin_ids that failed validation.
        summary: Human-readable one-line summary.
    """

    is_valid: bool
    n_max_valid: int
    n_min_valid: int
    n_max_total: int
    n_min_total: int
    results: list[BasinBoundaryResult]
    failing_ids: list[str]
    summary: str


# ---------------------------------------------------------------------------
# Type-sequence helpers (unit-testable, no adjacency required)
# ---------------------------------------------------------------------------


def is_valid_max_basin_boundary(boundary_types: list[str]) -> tuple[bool, str]:
    """Check that a pre-ordered type sequence alternates minimum/saddle.

    A maximum basin's boundary must cycle through alternating minima and
    saddles (in either starting phase).  All types must be in {minimum, saddle}.

    Args:
        boundary_types: CP types in cyclic order around the basin.

    Returns:
        ``(is_valid, description)`` where description is the joined pattern on
        success or an error message on failure.
    """
    if not boundary_types:
        return False, "empty boundary"

    abbrev: list[str] = []
    for i, t in enumerate(boundary_types):
        if t not in ("minimum", "saddle"):
            return False, f"invalid type at position {i}: {t!r}"
        abbrev.append(t[0])

    n = len(boundary_types)
    for i in range(n):
        if boundary_types[i] == boundary_types[(i + 1) % n]:
            msg = (
                f"non-alternating at positions {i}→{(i + 1) % n}: "
                f"{boundary_types[i]} == {boundary_types[(i + 1) % n]}"
            )
            return (False, msg)
    return True, "-".join(abbrev)


def is_valid_min_basin_boundary(boundary_types: list[str]) -> tuple[bool, str]:
    """Check that a pre-ordered type sequence alternates maximum/saddle.

    A minimum basin's boundary must cycle through alternating maxima and
    saddles.  All types must be in {maximum, saddle}.

    Args:
        boundary_types: CP types in cyclic order around the basin.

    Returns:
        ``(is_valid, description)`` where description is the joined pattern on
        success or an error message on failure.
    """
    if not boundary_types:
        return False, "empty boundary"

    abbrev: list[str] = []
    for i, t in enumerate(boundary_types):
        if t not in ("maximum", "saddle"):
            return False, f"invalid type at position {i}: {t!r}"
        abbrev.append(t[0])

    n = len(boundary_types)
    for i in range(n):
        if boundary_types[i] == boundary_types[(i + 1) % n]:
            msg = (
                f"non-alternating at positions {i}→{(i + 1) % n}: "
                f"{boundary_types[i]} == {boundary_types[(i + 1) % n]}"
            )
            return (False, msg)
    return True, "-".join(abbrev)


# ---------------------------------------------------------------------------
# Morse-Smale adjacency builder
# ---------------------------------------------------------------------------


def _best_basin_by_tangent(
    saddle_pos: tuple[float, float, float],
    tangent_dir: tuple[float, float, float],
    candidates: list[BasinCenter],
    center: tuple[float, float, float],
) -> str | None:
    """Find the basin center whose centered position most aligns with a branch.

    Projects (basin_centered - saddle_pos) onto the branch's tangent direction
    and picks the candidate with the highest dot product.

    Args:
        saddle_pos: Saddle position in the centred atom frame.
        tangent_dir: Unit tangent of the separatrix branch (centred frame).
        candidates: Basin centers to search (all in world frame).
        center: World-frame nucleus used to convert basin positions.

    Returns:
        center_id of the best matching BasinCenter, or None if empty.
    """
    if not candidates:
        return None
    t = np.asarray(tangent_dir, dtype=np.float64)
    t_n = t / (np.linalg.norm(t) + 1e-15)
    s = np.asarray(saddle_pos, dtype=np.float64)
    ctr = np.asarray(center, dtype=np.float64)
    best_id: str | None = None
    best_dot = -2.0
    for bc in candidates:
        bp_centered = np.asarray(bc.position, dtype=np.float64) - ctr
        diff = bp_centered - s
        diff_n = diff / (np.linalg.norm(diff) + 1e-15)
        dot = float(np.dot(t_n, diff_n))
        if dot > best_dot:
            best_dot, best_id = dot, bc.center_id
    return best_id


def build_morse_smale_adjacency(
    seeds: list[SaddleLaunchSeed],
    basin_centers: list[BasinCenter],
    center: tuple[float, float, float],
) -> SaddleAdjacency:
    """Build the Morse-Smale saddle adjacency from separatrix launch seeds.

    For each saddle, identifies the maximum basin each ascent branch flows
    toward and the minimum basin each descent branch flows toward, using
    tangent-direction projection onto the basin center positions.

    Args:
        seeds: SaddleLaunchSeed objects from the launch stage.
        basin_centers: All visible basin centers (maxima + minima).
        center: World-frame nucleus, used to translate basin positions to the
            centred atom frame used by the seeds.

    Returns:
        SaddleAdjacency mapping each saddle_id to
        ``{"max": [max_id, ...], "min": [min_id, ...]}``.
    """
    basin_max = [b for b in basin_centers if b.polarity == "maximum"]
    basin_min = [b for b in basin_centers if b.polarity == "minimum"]

    adj: SaddleAdjacency = {}
    for seed in seeds:
        max_targets = [
            bid
            for br in seed.ascent_branches
            if (
                bid := _best_basin_by_tangent(
                    seed.saddle_position, br.tangent_direction, basin_max, center
                )
            )
            is not None
        ]
        min_targets = [
            bid
            for br in seed.descent_branches
            if (
                bid := _best_basin_by_tangent(
                    seed.saddle_position, br.tangent_direction, basin_min, center
                )
            )
            is not None
        ]
        adj[seed.saddle_id] = {"max": max_targets, "min": min_targets}
    return adj


# ---------------------------------------------------------------------------
# Per-basin boundary validation
# ---------------------------------------------------------------------------


def _validate_boundary_cycle(
    basin_id: str,
    center_polarity: str,
    boundary_saddle_ids: list[str],
    saddle_adj: SaddleAdjacency,
    extremum_key: str,
) -> BasinBoundaryResult:
    """Validate the boundary cycle for one basin.

    Checks:
    1. Each opposite-polarity extremum appears in exactly 2 boundary saddles.
    2. The bipartite boundary graph (saddles ↔ extrema) forms a single cycle.

    Args:
        basin_id: Center basin ID (for reporting).
        center_polarity: ``"maximum"`` or ``"minimum"``.
        boundary_saddle_ids: Saddle IDs whose adjacency includes ``basin_id``.
        saddle_adj: Full adjacency mapping.
        extremum_key: ``"min"`` (for max basins) or ``"max"`` (for min basins).

    Returns:
        BasinBoundaryResult with validation outcome.
    """
    # Count how many times each extremum appears across boundary saddles
    extremum_counts: dict[str, int] = {}
    for sid in boundary_saddle_ids:
        for eid in saddle_adj.get(sid, {}).get(extremum_key, []):
            extremum_counts[eid] = extremum_counts.get(eid, 0) + 1

    n_sad = len(boundary_saddle_ids)
    extremum_ids = list(extremum_counts.keys())
    n_ext = len(extremum_ids)

    # Degree-2 check: each extremum must appear in exactly 2 boundary saddles
    bad_degree = [(eid, cnt) for eid, cnt in extremum_counts.items() if cnt != 2]
    if bad_degree:
        bd_str = ", ".join(f"{eid}(deg={cnt})" for eid, cnt in bad_degree)
        return BasinBoundaryResult(
            basin_id=basin_id,
            center_polarity=center_polarity,
            boundary_saddle_ids=boundary_saddle_ids,
            boundary_extremum_ids=extremum_ids,
            is_valid=False,
            n_saddles=n_sad,
            n_extrema=n_ext,
            error=f"degree≠2 extrema: {bd_str}",
        )

    # Equal count check (necessary for alternating cycle)
    if n_sad != n_ext:
        return BasinBoundaryResult(
            basin_id=basin_id,
            center_polarity=center_polarity,
            boundary_saddle_ids=boundary_saddle_ids,
            boundary_extremum_ids=extremum_ids,
            is_valid=False,
            n_saddles=n_sad,
            n_extrema=n_ext,
            error=f"unequal counts: {n_sad} saddles ≠ {n_ext} extrema",
        )

    if n_sad == 0:
        return BasinBoundaryResult(
            basin_id=basin_id,
            center_polarity=center_polarity,
            boundary_saddle_ids=[],
            boundary_extremum_ids=[],
            is_valid=False,
            n_saddles=0,
            n_extrema=0,
            error="no adjacent saddles found",
        )

    # Connectivity check: trace the boundary cycle
    # Build extremum → [saddle, saddle] mapping
    ext_to_saddles: dict[str, list[str]] = {eid: [] for eid in extremum_ids}
    for sid in boundary_saddle_ids:
        for eid in saddle_adj.get(sid, {}).get(extremum_key, []):
            if eid in ext_to_saddles:
                ext_to_saddles[eid].append(sid)

    # Trace: start at the first extremum, walk the bipartite cycle
    start_ext = extremum_ids[0]
    saddle_neighbors = ext_to_saddles[start_ext]
    if len(saddle_neighbors) < 1:
        return BasinBoundaryResult(
            basin_id=basin_id,
            center_polarity=center_polarity,
            boundary_saddle_ids=boundary_saddle_ids,
            boundary_extremum_ids=extremum_ids,
            is_valid=False,
            n_saddles=n_sad,
            n_extrema=n_ext,
            error="extremum has no saddle neighbors (graph error)",
        )
    start_saddle = saddle_neighbors[0]

    visited_saddles: list[str] = []
    cur_ext = start_ext
    cur_sad = start_saddle

    for _ in range(2 * n_sad + 1):
        visited_saddles.append(cur_sad)
        # From cur_sad, go to the other extremum (not cur_ext)
        ext_neighbors = saddle_adj.get(cur_sad, {}).get(extremum_key, [])
        if len(ext_neighbors) < 2:
            break
        next_ext = ext_neighbors[0] if ext_neighbors[1] == cur_ext else ext_neighbors[1]
        if next_ext == start_ext and len(visited_saddles) > 1:
            break  # completed the cycle
        # From next_ext, go to the other saddle (not cur_sad)
        sad_neighbors = ext_to_saddles.get(next_ext, [])
        if len(sad_neighbors) < 2:
            break
        next_sad = sad_neighbors[0] if sad_neighbors[1] == cur_sad else sad_neighbors[1]
        cur_ext = next_ext
        cur_sad = next_sad

    if len(set(visited_saddles)) != n_sad:
        n_visited = len(set(visited_saddles))
        return BasinBoundaryResult(
            basin_id=basin_id,
            center_polarity=center_polarity,
            boundary_saddle_ids=boundary_saddle_ids,
            boundary_extremum_ids=extremum_ids,
            is_valid=False,
            n_saddles=n_sad,
            n_extrema=n_ext,
            error=f"boundary not a single cycle: {n_visited}/{n_sad} saddles reachable",
        )

    return BasinBoundaryResult(
        basin_id=basin_id,
        center_polarity=center_polarity,
        boundary_saddle_ids=boundary_saddle_ids,
        boundary_extremum_ids=extremum_ids,
        is_valid=True,
        n_saddles=n_sad,
        n_extrema=n_ext,
        error="",
    )


def validate_max_basin_boundary(
    max_id: str,
    saddle_adj: SaddleAdjacency,
) -> BasinBoundaryResult:
    """Validate the boundary of a maximum basin.

    Args:
        max_id: center_id of the maximum basin.
        saddle_adj: Morse-Smale adjacency from ``build_morse_smale_adjacency``.

    Returns:
        BasinBoundaryResult for this maximum basin.
    """
    boundary_saddles = [sid for sid, v in saddle_adj.items() if max_id in v["max"]]
    return _validate_boundary_cycle(
        basin_id=max_id,
        center_polarity="maximum",
        boundary_saddle_ids=boundary_saddles,
        saddle_adj=saddle_adj,
        extremum_key="min",
    )


def validate_min_basin_boundary(
    min_id: str,
    saddle_adj: SaddleAdjacency,
) -> BasinBoundaryResult:
    """Validate the boundary of a minimum basin.

    Args:
        min_id: center_id of the minimum basin.
        saddle_adj: Morse-Smale adjacency from ``build_morse_smale_adjacency``.

    Returns:
        BasinBoundaryResult for this minimum basin.
    """
    boundary_saddles = [sid for sid, v in saddle_adj.items() if min_id in v["min"]]
    return _validate_boundary_cycle(
        basin_id=min_id,
        center_polarity="minimum",
        boundary_saddle_ids=boundary_saddles,
        saddle_adj=saddle_adj,
        extremum_key="max",
    )


# ---------------------------------------------------------------------------
# Full-configuration validation
# ---------------------------------------------------------------------------


def validate_all_basin_boundaries(
    basin_centers: list[BasinCenter],
    saddle_adj: SaddleAdjacency,
) -> SimplificationValidity:
    """Validate every basin boundary in a simplified CP configuration.

    Args:
        basin_centers: All visible basin centers (maxima and minima).
        saddle_adj: Morse-Smale adjacency from ``build_morse_smale_adjacency``.

    Returns:
        SimplificationValidity with per-basin results and aggregate status.
    """
    results: list[BasinBoundaryResult] = []
    failing_ids: list[str] = []
    n_max_valid = n_min_valid = 0
    n_max_total = n_min_total = 0

    for bc in basin_centers:
        if bc.polarity == "maximum":
            res = validate_max_basin_boundary(bc.center_id, saddle_adj)
            n_max_total += 1
            if res.is_valid:
                n_max_valid += 1
            else:
                failing_ids.append(bc.center_id)
        else:
            res = validate_min_basin_boundary(bc.center_id, saddle_adj)
            n_min_total += 1
            if res.is_valid:
                n_min_valid += 1
            else:
                failing_ids.append(bc.center_id)
        results.append(res)

    all_valid = len(failing_ids) == 0
    if all_valid:
        summary = (
            f"✓ all {n_max_total} max and {n_min_total} min basin boundaries valid"
        )
    else:
        fail_strs = []
        for res in results:
            if not res.is_valid:
                fail_strs.append(f"{res.basin_id}: {res.error}")
        summary = "✗ " + "; ".join(fail_strs)

    return SimplificationValidity(
        is_valid=all_valid,
        n_max_valid=n_max_valid,
        n_min_valid=n_min_valid,
        n_max_total=n_max_total,
        n_min_total=n_min_total,
        results=results,
        failing_ids=failing_ids,
        summary=summary,
    )
