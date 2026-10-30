"""Basin Boundary Validator for gba_topology2.

Validates that simplified CP configurations satisfy the topological constraint:
- Max basin boundary: alternating minimum and saddle CPs
- Min basin boundary: alternating maximum and saddle CPs

Intended to validate candidate solutions from parameter space searches and rank
them by correctness + parameter cost.

Subroutine contract per AGENTS.md:
- Pure algorithmic computation (NumPy, standard library).
- Standalone testable via simple unit tests.
- Deterministic and fully specified.
"""

from dataclasses import dataclass
from typing import Any

from gba_topology2.src.morse_detector import CriticalPoint

__all__ = [
    "BasinBoundaryResult",
    "SimplificationValidity",
    "validate_basin_boundaries",
    "is_valid_max_basin_boundary",
    "is_valid_min_basin_boundary",
]


@dataclass(frozen=True)
class BasinBoundaryResult:
    """Result of extracting and validating basin boundaries.

    Attributes:
        basin_id: Identifier of the basin center CP.
        center_polarity: "maximum" or "minimum".
        boundary_cp_ids: Vertex IDs of boundary CPs in cyclic order.
        boundary_types: List of CP types ('maximum', 'minimum', 'saddle') in
            cyclic order.
        is_alternating: True if types alternate correctly.
        alternation_pattern: String description of the pattern found
            (e.g., 'max-sad-min-sad-max-sad').
        alternation_errors: List of (index, expected, found) tuples marking
            where alternation fails.
    """

    basin_id: int  # vertex_id of basin center
    center_polarity: str  # "maximum" or "minimum"
    boundary_cp_ids: list[int]  # vertex IDs of boundary CPs
    boundary_types: list[str]
    is_alternating: bool
    alternation_pattern: str
    alternation_errors: list[tuple[int, str, str]]


@dataclass(frozen=True)
class SimplificationValidity:
    """Overall validity assessment of a simplified CP configuration.

    Attributes:
        is_valid: True if all basin boundaries satisfy alternating constraint.
        num_max_basins: Number of valid maximum basins.
        num_min_basins: Number of valid minimum basins.
        validation_results: List of BasinBoundaryResult for each basin.
        failing_basins: Indices into validation_results for basins that fail.
        error_summary: Human-readable description of validation failures.
    """

    is_valid: bool
    num_max_basins: int
    num_min_basins: int
    validation_results: list[BasinBoundaryResult]
    failing_basins: list[int]
    error_summary: str


def _get_cp_type(cp: CriticalPoint) -> str:
    """Extract CP type from CriticalPoint.

    Returns:
        One of: 'maximum', 'minimum', 'saddle'.
    """
    return cp.cp_type


def _find_alternation_errors(
    boundary_types: list[str], is_max_basin: bool
) -> list[tuple[int, str, str]]:
    """Find and return list of alternation errors.

    Args:
        boundary_types: List of CP types.
        is_max_basin: True if validating max basin (expects min/sad alternation),
                     False if validating min basin (expects max/sad alternation).

    Returns:
        List of (index, expected_type, found_type) tuples.
    """
    errors: list[tuple[int, str, str]] = []
    if not boundary_types:
        return errors

    for i in range(len(boundary_types)):
        if is_max_basin:
            expected = "minimum" if i % 2 == 0 else "saddle"
        else:
            expected = "maximum" if i % 2 == 0 else "saddle"
        found = boundary_types[i]
        if found != expected:
            errors.append((i, expected, found))
    return errors


def is_valid_max_basin_boundary(boundary_types: list[str]) -> tuple[bool, str]:
    """Check if a max basin boundary alternates minimum-saddle-minimum-saddle...

    A maximum basin's boundary should be a cycle of alternating minima and
    saddles (in cyclic order). This is the topological constraint from
    Morse theory: maxima are surrounded by minima separated by saddles.

    Args:
        boundary_types: List of CP types in cyclic order.

    Returns:
        (is_valid, pattern_description) where is_valid is True if types
        alternate correctly, and pattern_description is a string like
        'min-sad-min-sad'.
    """
    if not boundary_types:
        return False, "empty boundary"

    # Cyclic alternation check: for each adjacent pair (and wrap around),
    # verify they are not the same type and both are in {min, sad}.
    pattern_parts: list[str] = []
    for i, cp_type in enumerate(boundary_types):
        if cp_type not in ("minimum", "saddle"):
            return False, f"invalid type at position {i}: {cp_type}"
        pattern_parts.append(cp_type[0])  # 'm' for min, 's' for sad

    pattern = "-".join(pattern_parts)

    # Check alternation: no two consecutive types are the same
    n = len(boundary_types)
    for i in range(n):
        curr = boundary_types[i]
        next_idx = (i + 1) % n
        next_type = boundary_types[next_idx]
        if curr == next_type:
            return False, f"non-alternating at positions {i}->{next_idx}: {curr} == {next_type}"

    return True, pattern


def is_valid_min_basin_boundary(boundary_types: list[str]) -> tuple[bool, str]:
    """Check if a min basin boundary alternates maximum-saddle-maximum-saddle...

    A minimum basin's boundary should be a cycle of alternating maxima and
    saddles (in cyclic order). This is the topological dual of max basin
    validation.

    Args:
        boundary_types: List of CP types in cyclic order.

    Returns:
        (is_valid, pattern_description) where is_valid is True if types
        alternate correctly.
    """
    if not boundary_types:
        return False, "empty boundary"

    pattern_parts: list[str] = []
    for i, cp_type in enumerate(boundary_types):
        if cp_type not in ("maximum", "saddle"):
            return False, f"invalid type at position {i}: {cp_type}"
        pattern_parts.append(cp_type[0])  # 'M' for max, 's' for sad

    pattern = "-".join(pattern_parts)

    # Check alternation
    n = len(boundary_types)
    for i in range(n):
        curr = boundary_types[i]
        next_idx = (i + 1) % n
        next_type = boundary_types[next_idx]
        if curr == next_type:
            return False, f"non-alternating at positions {i}->{next_idx}: {curr} == {next_type}"

    return True, pattern


def validate_basin_boundaries(
    all_cps: list[CriticalPoint],
    visible_max_ids: set[int] | None = None,
    visible_min_ids: set[int] | None = None,
) -> SimplificationValidity:
    """Validate all basin boundaries in a simplified CP configuration.

    For each visible maximum and minimum, extract its boundary CPs (saddles
    and the opposite polarity extrema) and verify the alternating type
    constraint. Collect results and determine overall validity.

    Args:
        all_cps: All critical points (raw + effective, before/after simplification).
        visible_max_ids: Set of vertex IDs of visible maximum CPs. If None,
            all maxima are considered visible.
        visible_min_ids: Set of vertex IDs of visible minimum CPs. If None,
            all minima are considered visible.

    Returns:
        SimplificationValidity containing validation results for all basins.
    """
    # Partition CPs by type
    maxima = [cp for cp in all_cps if cp.cp_type == "maximum"]
    minima = [cp for cp in all_cps if cp.cp_type == "minimum"]
    saddles = [cp for cp in all_cps if cp.cp_type == "saddle"]

    if visible_max_ids is None:
        visible_max_ids = {cp.vertex_id for cp in maxima}
    if visible_min_ids is None:
        visible_min_ids = {cp.vertex_id for cp in minima}

    validation_results: list[BasinBoundaryResult] = []
    failing_basin_indices: list[int] = []
    num_max_basins = 0
    num_min_basins = 0

    # Validate each visible maximum basin
    for max_cp in maxima:
        if max_cp.vertex_id not in visible_max_ids:
            continue

        # Boundary: saddles + minima
        boundary_cp_ids = [cp.vertex_id for cp in (saddles + minima)]
        boundary_types = [_get_cp_type(cp) for cp in (saddles + minima)]

        is_valid, pattern = is_valid_max_basin_boundary(boundary_types)

        # Identify alternation errors for max basin
        alternation_errors = _find_alternation_errors(boundary_types, is_max_basin=True)

        result = BasinBoundaryResult(
            basin_id=max_cp.vertex_id,
            center_polarity="maximum",
            boundary_cp_ids=boundary_cp_ids,
            boundary_types=boundary_types,
            is_alternating=is_valid,
            alternation_pattern=pattern,
            alternation_errors=alternation_errors,
        )
        validation_results.append(result)
        if is_valid:
            num_max_basins += 1
        else:
            failing_basin_indices.append(len(validation_results) - 1)

    # Validate each visible minimum basin
    for min_cp in minima:
        if min_cp.vertex_id not in visible_min_ids:
            continue

        # Boundary: saddles + maxima
        boundary_cp_ids = [cp.vertex_id for cp in (saddles + maxima)]
        boundary_types = [_get_cp_type(cp) for cp in (saddles + maxima)]

        is_valid, pattern = is_valid_min_basin_boundary(boundary_types)

        # Identify alternation errors for min basin
        alternation_errors = _find_alternation_errors(boundary_types, is_max_basin=False)

        result = BasinBoundaryResult(
            basin_id=min_cp.vertex_id,
            center_polarity="minimum",
            boundary_cp_ids=boundary_cp_ids,
            boundary_types=boundary_types,
            is_alternating=is_valid,
            alternation_pattern=pattern,
            alternation_errors=alternation_errors,
        )
        validation_results.append(result)
        if is_valid:
            num_min_basins += 1
        else:
            failing_basin_indices.append(len(validation_results) - 1)

    is_overall_valid = len(failing_basin_indices) == 0

    # Build error summary
    if is_overall_valid:
        error_summary = (
            f"✓ All {num_max_basins} max and {num_min_basins} min basins valid"
        )
    else:
        failing_ids = [
            validation_results[i].basin_id for i in failing_basin_indices
        ]
        error_summary = f"✗ {len(failing_basin_indices)} basin(s) fail alternation: {', '.join(str(vid) for vid in failing_ids)}"

    return SimplificationValidity(
        is_valid=is_overall_valid,
        num_max_basins=num_max_basins,
        num_min_basins=num_min_basins,
        validation_results=validation_results,
        failing_basins=failing_basin_indices,
        error_summary=error_summary,
    )
