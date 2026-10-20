"""Unit tests for the pair scale distribution diagnostic (pair_scales.py).

The diagnostic exists to answer one question: is there a gap between the
separation of fragments of one feature and the separation of genuinely distinct
features? That only works if the measured separations are right, so these tests
pin the geometry against hand-computed values.

Two measures are checked for the same pair layouts:

- centroid separation, which for elongated entities is dominated by the
  half-lengths, and
- directional edge clearance, which measures the fitted boundary toward the
  neighbour and so collapses to zero for touching entities at any orientation.

The perpendicular case is the load-bearing one: it is the geometry that made the
isotropic semi-major subtraction absorb an off-axis minimum, so a clearance
measure that reports it as "close" is measuring the wrong shape.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from itertools import pairwise

import numpy as np
import pytest

from gba_topology2.src.micro_cluster import (
    MicroCluster,
    generate_elliptical_boundary_ring,
)
from gba_topology2.src.morse_detector import CriticalPoint
from gba_topology2.src.pair_scales import (
    PairwiseClearance,
    compute_pairwise_clearances,
    histogram_bins,
    render_ascii_histogram,
    render_intermediate_registry,
    write_svg_histogram,
)

DELTA_THETA = 0.025
RADIUS = 1.0


def _on_sphere(
    centroid_unit: tuple[float, float, float],
    axis_unit: tuple[float, float, float],
    offset: float,
) -> tuple[float, float, float]:
    """Step ``offset`` radians from a centroid along a tangent axis."""
    c = np.array(centroid_unit, dtype=np.float64)
    c = c / np.linalg.norm(c)
    ax = np.array(axis_unit, dtype=np.float64)
    ax = ax - float(np.dot(ax, c)) * c
    n = float(np.linalg.norm(ax))
    ax = ax / n if n > 1e-12 else np.zeros(3, dtype=np.float64)
    p = math.cos(offset) * c + math.sin(offset) * ax
    p = p / np.linalg.norm(p)
    return (float(p[0]), float(p[1]), float(p[2]))


def _string_cluster(
    cluster_id: str,
    centroid_unit: tuple[float, float, float],
    axis_unit: tuple[float, float, float],
    semi_major: float,
    semi_minor: float,
    n_max: int,
    n_sad: int,
    start_vid: int,
) -> MicroCluster:
    """Build an elongated string cluster with members spread along its axis."""
    c = np.array(centroid_unit, dtype=np.float64)
    c = c / np.linalg.norm(c)
    total = n_max + n_sad
    members: list[CriticalPoint] = []
    vid = start_vid
    for i, cp_type in enumerate(["maximum"] * n_max + ["saddle"] * n_sad):
        frac = (i / (total - 1) - 0.5) if total > 1 else 0.0
        pos = _on_sphere(centroid_unit, axis_unit, 2.0 * frac * semi_major)
        members.append(
            CriticalPoint(
                vertex_id=vid,
                cp_type=cp_type,
                value=1.0 if cp_type == "maximum" else 0.5,
                position=pos,
                multiplicity=1,
            )
        )
        vid += 1

    centroid = (float(c[0] * RADIUS), float(c[1] * RADIUS), float(c[2] * RADIUS))
    return MicroCluster(
        cluster_id=cluster_id,
        members=members,
        centroid=centroid,
        centroid_unit=(float(c[0]), float(c[1]), float(c[2])),
        angular_radius_rad=semi_major,
        angular_radius_deg=math.degrees(semi_major),
        angular_diameter_rad=2.0 * semi_major,
        angular_diameter_deg=math.degrees(2.0 * semi_major),
        spatial_diameter=2.0 * semi_major * RADIUS,
        n_maxima=n_max,
        n_minima=0,
        n_saddles=n_sad,
        total_saddle_index=n_sad,
        local_euler_index=n_max - n_sad,
        is_multi_cp=True,
        boundary_ring=generate_elliptical_boundary_ring(
            centroid=centroid,
            sphere_radius=RADIUS,
            semi_major_rad=semi_major,
            semi_minor_rad=semi_minor,
            principal_axis_unit=axis_unit,
            num_samples=24,
        ),
        morphology="string",
        aspect_ratio=8.0,
        principal_axis_unit=axis_unit,
        semi_major_rad=semi_major,
        semi_minor_rad=semi_minor,
    )


def _compact_cluster(
    cluster_id: str,
    centroid_unit: tuple[float, float, float],
    cp_type: str,
    radius: float,
    vid: int,
) -> MicroCluster:
    """Build a single-CP compact cluster (circular ring, null axis)."""
    c = np.array(centroid_unit, dtype=np.float64)
    c = c / np.linalg.norm(c)
    chi = 1 if cp_type in ("maximum", "minimum") else -1
    return MicroCluster(
        cluster_id=cluster_id,
        members=[
            CriticalPoint(
                vertex_id=vid,
                cp_type=cp_type,
                value=1.0,
                position=(float(c[0]), float(c[1]), float(c[2])),
                multiplicity=1,
            )
        ],
        centroid=(float(c[0] * RADIUS), float(c[1] * RADIUS), float(c[2] * RADIUS)),
        centroid_unit=(float(c[0]), float(c[1]), float(c[2])),
        angular_radius_rad=radius,
        angular_radius_deg=math.degrees(radius),
        angular_diameter_rad=0.0,
        angular_diameter_deg=0.0,
        spatial_diameter=0.0,
        n_maxima=1 if cp_type == "maximum" else 0,
        n_minima=1 if cp_type == "minimum" else 0,
        n_saddles=1 if cp_type == "saddle" else 0,
        total_saddle_index=1 if cp_type == "saddle" else 0,
        local_euler_index=chi,
        is_multi_cp=False,
        boundary_ring=[],
        morphology="compact",
        aspect_ratio=1.0,
        semi_major_rad=radius,
        semi_minor_rad=radius,
    )


def _collinear_pair() -> list[MicroCluster]:
    """Two string fragments tip-to-tip along +x, 0.015 rad uncovered gap."""
    return [
        _string_cluster(
            "MC1",
            (math.sin(-0.065), 0.0, math.cos(-0.065)),
            (1.0, 0.0, 0.0),
            0.05,
            0.01,
            4,
            5,
            0,
        ),
        _string_cluster(
            "MC2",
            (math.sin(0.05), 0.0, math.cos(0.05)),
            (1.0, 0.0, 0.0),
            0.05,
            0.01,
            4,
            5,
            100,
        ),
    ]


def _perpendicular_pair() -> list[MicroCluster]:
    """A long corridor at the pole and a lone minimum 90 degrees off its axis."""
    return [
        _string_cluster("MC10", (0.0, 0.0, 1.0), (1.0, 0.0, 0.0), 0.30, 0.01, 8, 9, 0),
        _compact_cluster(
            "MC131", (0.0, math.sin(0.35), math.cos(0.35)), "minimum", 0.02, 1310
        ),
    ]


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


def test_collinear_pair_clearance_matches_tip_gap() -> None:
    """On-axis fragments measure the uncovered gap between their ellipse tips."""
    records = compute_pairwise_clearances(_collinear_pair(), DELTA_THETA)

    assert len(records) == 1
    rec = records[0]
    assert (rec.id_a, rec.id_b) == ("MC1", "MC2")
    # Centroid separation 0.115 rad, each cluster reaches 0.05 along the axis.
    assert rec.centroid_rad == pytest.approx(0.115, abs=1e-9)
    assert rec.gap_rad == pytest.approx(0.015, abs=1e-9)
    assert rec.gap_dtheta == pytest.approx(0.015 / DELTA_THETA, abs=1e-6)
    # Both neighbours sit on the shared principal axis.
    assert rec.bearing_a_deg == pytest.approx(0.0, abs=1e-6)


def test_perpendicular_pair_measures_transverse_width() -> None:
    """A transverse neighbour is measured against the minor axis, not the major.

    This is the geometry that the isotropic semi-major subtraction got wrong.
    The corridor is 0.30 rad long, so subtracting the half-length would report
    the pair as overlapping; the directional measure must report a wide gap.
    """
    records = compute_pairwise_clearances(_perpendicular_pair(), DELTA_THETA)

    assert len(records) == 1
    rec = records[0]
    assert rec.centroid_rad == pytest.approx(0.35, abs=1e-9)
    # 0.35 - b(0.01) - compact radius(0.02)
    assert rec.gap_rad == pytest.approx(0.32, abs=1e-9)
    assert rec.bearing_a_deg == pytest.approx(90.0, abs=1e-5)
    # The isotropic measure would have reported this as touching.
    isotropic = 0.35 - 0.30 - 0.02
    assert isotropic < rec.gap_rad


def test_compact_pair_bearing_is_reported_as_unknown() -> None:
    """A compact cluster has no axis, so its bearing is flagged, not guessed."""
    pair = _perpendicular_pair()
    records = compute_pairwise_clearances(pair, DELTA_THETA)
    assert records[0].bearing_a_deg == pytest.approx(90.0)

    # Reversed order puts the compact cluster first, where the axis is null.
    flipped = compute_pairwise_clearances(list(reversed(pair)), DELTA_THETA)
    assert flipped[0].bearing_a_deg == -1.0


def test_overlap_produces_negative_clearance() -> None:
    """Crossing corridors report a negative gap, as the bridging test expects."""
    ang = math.radians(60.0)
    pair = [
        _string_cluster("MC1", (0.0, 0.0, 1.0), (1.0, 0.0, 0.0), 0.30, 0.01, 4, 5, 0),
        _string_cluster(
            "MC2",
            (0.0, 0.0, 1.0),
            (math.cos(ang), math.sin(ang), 0.0),
            0.30,
            0.01,
            4,
            5,
            100,
        ),
    ]
    records = compute_pairwise_clearances(pair, DELTA_THETA)

    assert records[0].centroid_rad == pytest.approx(0.0, abs=1e-9)
    assert records[0].gap_rad < 0.0


def test_chi_sum_and_composition_are_carried() -> None:
    """The registry needs the index balance to judge a candidate merge."""
    records = compute_pairwise_clearances(_perpendicular_pair(), DELTA_THETA)
    rec = records[0]
    assert rec.chi_sum == -1 + 1
    assert rec.composition_a == "8 Max, 9 Sad"
    assert rec.composition_b == "1 Min"


# ---------------------------------------------------------------------------
# Enumeration
# ---------------------------------------------------------------------------


def test_pair_count_is_n_choose_two() -> None:
    """Every unordered pair appears exactly once, with no self-pairs."""
    clusters = _collinear_pair() + _perpendicular_pair()
    records = compute_pairwise_clearances(clusters, DELTA_THETA)

    n = len(clusters)
    assert len(records) == n * (n - 1) // 2
    keys = {(r.id_a, r.id_b) for r in records}
    assert len(keys) == len(records)
    assert all(r.id_a != r.id_b for r in records)


def test_records_sorted_by_clearance() -> None:
    """Sorting by gap makes the trough readable straight off the list."""
    clusters = _collinear_pair() + _perpendicular_pair()
    records = compute_pairwise_clearances(clusters, DELTA_THETA)

    gaps = [r.gap_dtheta for r in records]
    assert gaps == sorted(gaps)


@pytest.mark.parametrize("count", [0, 1])
def test_fewer_than_two_entities_yields_nothing(count: int) -> None:
    """Boundary case: a single entity has no pairs to measure."""
    clusters = _collinear_pair()[:count]
    assert compute_pairwise_clearances(clusters, DELTA_THETA) == []


def test_rejects_nonpositive_pitch() -> None:
    """The normalising unit must be positive."""
    with pytest.raises(ValueError, match="must be positive"):
        compute_pairwise_clearances(_collinear_pair(), 0.0)


def test_centroid_distance_is_symmetric() -> None:
    """Swapping the pair order must not change the measured separations."""
    pair = _perpendicular_pair()
    forward = compute_pairwise_clearances(pair, DELTA_THETA)[0]
    backward = compute_pairwise_clearances(list(reversed(pair)), DELTA_THETA)[0]

    assert forward.centroid_rad == pytest.approx(backward.centroid_rad)
    assert forward.gap_rad == pytest.approx(backward.gap_rad)


# ---------------------------------------------------------------------------
# Binning
# ---------------------------------------------------------------------------


def test_bins_conserve_every_value() -> None:
    """Counts must sum to the input size, including negatives and overflow."""
    values = [-3.0, -0.1, 0.0, 0.4, 0.5, 1.2, 11.9, 12.0, 40.0]
    bins = histogram_bins(values, bin_width=0.5, max_value=12.0)

    assert sum(count for _, _, count in bins) == len(values)
    assert bins[0][0] == -math.inf
    assert bins[-1][1] == math.inf


def test_negative_values_land_in_the_underflow_bin() -> None:
    """Overlapping boundaries must be visible, not folded into the first bin.

    Negative clearance is the signal that two fitted ellipses overlap, which is
    the signature of a merge candidate, so losing it would hide the very pairs
    the diagnostic exists to find.
    """
    bins = histogram_bins([-5.0, -0.2, 0.1], bin_width=0.5, max_value=12.0)

    assert bins[0] == (-math.inf, 0.0, 2)
    assert bins[1] == (0.0, 0.5, 1)


def test_bin_edges_are_contiguous() -> None:
    """No gaps between buckets, so a value cannot be counted twice."""
    bins = histogram_bins([1.0], bin_width=0.5, max_value=6.0)

    for lower, upper in pairwise(bins):
        assert lower[1] == upper[0]


def test_empty_input_gives_zero_counts() -> None:
    """An empty distribution renders as an all-zero chart, not an error."""
    bins = histogram_bins([], bin_width=1.0, max_value=4.0)

    assert sum(count for _, _, count in bins) == 0
    assert len(bins) == 6


def test_rejects_nonpositive_bin_width() -> None:
    """Bucket width is a divisor and must be positive."""
    with pytest.raises(ValueError, match="must be positive"):
        histogram_bins([1.0], bin_width=0.0)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def test_ascii_histogram_labels_zones() -> None:
    """The chart must name the scale zone of each bin for visual reading."""
    chart = render_ascii_histogram([0.5, 3.0, 9.0, -1.0], "TEST")

    assert "overlap" in chart
    assert "intra" in chart
    assert "meso" in chart
    assert "macro" in chart
    assert "n = 4 pairs" in chart


def test_ascii_histogram_marks_empty_meso_bins() -> None:
    """An empty transition zone is the headline result and must be flagged."""
    chart = render_ascii_histogram([0.5, 1.0, 9.0, 10.0], "TEST")

    assert "empty" in chart


def test_ascii_histogram_survives_empty_input() -> None:
    """Zero pairs must not divide by zero when scaling the bars."""
    chart = render_ascii_histogram([], "TEST")

    assert "n = 0 pairs" in chart


def test_overflow_bin_does_not_flatten_the_chart() -> None:
    """Bars scale to the finite peak, so the meso trough stays readable.

    On a sphere most pairs are distant, so the overflow bin dwarfs everything
    else. Scaling to it would render the interesting bins as blank space and
    hide the separation the chart exists to reveal.
    """
    values = [1.0, 3.0, 5.0] + [50.0] * 200
    chart = render_ascii_histogram(values, "TEST", bar_width=40)

    rows = [line for line in chart.splitlines() if "dtheta [" in line]
    finite_row = next(r for r in rows if r.split("|")[0].strip().startswith("1.0 -"))
    overflow_row = next(r for r in rows if r.split("|")[0].strip().startswith(">12"))

    # A single pair fills the finite bar, so the scale is set by the finite bins.
    assert len(finite_row.split("|")[1]) >= 38
    assert "off scale" in overflow_row
    assert "bars scaled to 1)" in chart


def test_registry_lists_meso_pairs() -> None:
    """Pairs in the transition zone are reported with their decision data."""
    clusters = _collinear_pair() + _perpendicular_pair()
    records = compute_pairwise_clearances(clusters, DELTA_THETA)

    text = render_intermediate_registry(records, low_dtheta=0.0, high_dtheta=1.0)
    assert "MC1" in text and "MC2" in text
    assert "chi_sum" in text


def test_registry_reports_clean_separation() -> None:
    """An empty transition zone is the good outcome and must say so."""
    clusters = _collinear_pair()
    records = compute_pairwise_clearances(clusters, DELTA_THETA)

    text = render_intermediate_registry(records, low_dtheta=8.0, high_dtheta=9.0)
    assert "empty" in text
    assert "clean scale separation" in text


def test_registry_truncates_long_lists() -> None:
    """A dense field must not flood the terminal."""
    records = [
        PairwiseClearance(
            id_a=f"A{i}",
            id_b="B",
            composition_a="1 Max",
            composition_b="1 Min",
            centroid_rad=0.1,
            centroid_dtheta=4.0,
            gap_rad=0.1,
            gap_dtheta=4.0,
            bearing_a_deg=10.0,
            chi_sum=2,
        )
        for i in range(50)
    ]
    text = render_intermediate_registry(records, limit=5)

    assert "45 more" in text


# ---------------------------------------------------------------------------
# SVG output
# ---------------------------------------------------------------------------


def test_svg_is_well_formed_and_titled(tmp_path: object) -> None:
    """The chart must open in a browser, so the XML has to parse."""
    out = f"{tmp_path}/chart.svg"  # type: ignore[operator]
    write_svg_histogram([0.5, 3.0, 9.0], [1.0, 4.0, 10.0], "curvedness", out)

    with open(out, encoding="utf-8") as handle:
        payload = handle.read()
    root = ET.fromstring(payload)
    assert root.tag.endswith("svg")
    assert "curvedness" in payload


def test_svg_escapes_field_names(tmp_path: object) -> None:
    """Field names carry metacharacters and non-ASCII prefixes."""
    out = f"{tmp_path}/chart.svg"  # type: ignore[operator]
    write_svg_histogram([1.0], [2.0], 'Ï\x81 <curvedness> & "willmore"', out)

    with open(out, encoding="utf-8") as handle:
        payload = handle.read()
    ET.fromstring(payload)
    assert "&lt;curvedness&gt;" in payload
    assert "&amp;" in payload


def test_svg_handles_empty_distributions(tmp_path: object) -> None:
    """A field with no pairs must still produce a valid chart."""
    out = f"{tmp_path}/empty.svg"  # type: ignore[operator]
    write_svg_histogram([], [], "empty field", out)

    with open(out, encoding="utf-8") as handle:
        ET.fromstring(handle.read())
