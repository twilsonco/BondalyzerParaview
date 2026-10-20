"""Pair Scale Distribution Subroutine for gba_topology2.

Diagnostic mathematics for the three-scale hypothesis:

    L1 (micro)    mesh-scale chatter, absorbed by micro-clustering
    L2 (meso)     corridor fragmentation, absorbed by bridging
    L3 (macro)    distinct physical basins, which must NOT be merged

If the field has genuine scale separation, the distribution of pairwise
separations between consolidated entities shows a peak at L2 and a peak at L3
with a trough between them. The bottom of that trough is the objective place to
put a macro-clustering cutoff, rather than a guessed one.

Two separations are measured for every pair, both normalised by the mesh pitch
so results from spheres of different radius and mesh density share one axis:

- **Centroid separation** ``theta_ij / dtheta``. Simple, but for elongated
  entities it is dominated by the two half-lengths, so tip-to-tip fragments look
  far apart and crossing corridors look close.
- **Directional edge clearance** ``(theta_ij - r_i(phi_i) - r_j(phi_j)) / dtheta``,
  where ``r`` is the fitted boundary evaluated toward the neighbour (see
  :func:`gba_topology2.src.string_aggregator.cluster_directional_support_radius`).
  Entities that touch sit near zero regardless of orientation, so the trough is
  far sharper than in the centroid distribution.

This module is pure: it imports no viewer, VTK, or Trame code, and mutates
nothing. It consumes the ``MicroCluster`` interface, so a bridged corridor
(``ReductionUnit.as_cluster``) is measured exactly like a lone cluster.

Run the CLI in ``gba_topology2/inspect_pair_scales.py`` to apply this to a
``.plt`` file.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from gba_topology2.src.micro_cluster import MicroCluster
from gba_topology2.src.string_aggregator import cluster_directional_support_radius

__all__ = [
    "PairwiseClearance",
    "compute_pairwise_clearances",
    "histogram_bins",
    "render_ascii_histogram",
    "render_intermediate_registry",
    "write_svg_histogram",
]

# Zone boundaries, in units of the mesh pitch. Chosen to bracket the operating
# range of the pipeline: micro-clustering reaches k_pitch (~2), bridging reaches
# k_bridge (~3) beyond the fitted boundary, and true basins sit well outside.
ZONE_EDGES_DTHETA: tuple[float, float] = (2.5, 6.0)
ZONE_LABELS: tuple[str, str, str] = ("intra", "meso", "macro")


@dataclass(frozen=True)
class PairwiseClearance:
    """Separation between two consolidated entities, in radians and pitch units.

    Attributes:
        id_a: Identifier of the first entity.
        id_b: Identifier of the second entity.
        composition_a: Human-readable composition of the first entity.
        composition_b: Human-readable composition of the second entity.
        centroid_rad: Great-circle distance between centroids, radians.
        centroid_dtheta: Same distance in mesh-pitch units.
        gap_rad: Directional edge-to-edge clearance, radians. Negative when the
            fitted boundaries overlap.
        gap_dtheta: Same clearance in mesh-pitch units.
        bearing_a_deg: Angle between the bearing to entity B and entity A's
            principal axis, folded to [0, 90]. 0 is on-axis, 90 is transverse.
            -1.0 when A has no meaningful axis (compact morphology).
        chi_sum: Summed local Euler index of the pair.
    """

    id_a: str
    id_b: str
    composition_a: str
    composition_b: str
    centroid_rad: float
    centroid_dtheta: float
    gap_rad: float
    gap_dtheta: float
    bearing_a_deg: float
    chi_sum: int


def _bearing_to_axis_deg(
    origin_unit: np.ndarray,
    target_unit: np.ndarray,
    axis_unit: np.ndarray,
) -> float:
    """Angle from an entity's principal axis to the bearing of a neighbour.

    The bearing is the direction to ``target_unit`` projected onto the tangent
    plane at ``origin_unit``, and the axis is re-orthogonalised the same way, so
    the result is a true tangent-plane angle. It is folded to [0, 90] degrees
    because a principal axis is an undirected line.

    Args:
        origin_unit: (3,) unit centroid direction of the measuring entity.
        target_unit: (3,) unit centroid direction of the neighbour.
        axis_unit: (3,) principal axis of the measuring entity.

    Returns:
        Angle in degrees in [0, 90], or -1.0 when the axis is degenerate (a
        compact cluster has no meaningful major direction).
    """
    axis = np.asarray(axis_unit, dtype=np.float64)
    axis = axis - float(np.dot(axis, origin_unit)) * origin_unit
    if float(np.linalg.norm(axis)) < 1e-9:
        return -1.0
    axis = axis / float(np.linalg.norm(axis))

    bearing = np.asarray(target_unit, dtype=np.float64)
    bearing = bearing - float(np.dot(bearing, origin_unit)) * origin_unit
    if float(np.linalg.norm(bearing)) < 1e-12:
        return -1.0
    bearing = bearing / float(np.linalg.norm(bearing))

    cos_v = abs(float(np.clip(float(np.dot(bearing, axis)), -1.0, 1.0)))
    return math.degrees(math.acos(cos_v))


def compute_pairwise_clearances(
    clusters: Sequence[MicroCluster],
    delta_theta_mesh_rad: float,
) -> list[PairwiseClearance]:
    """Measure every unique pair of entities.

    Args:
        clusters: Entities to measure. Pass ``ReductionUnit.as_cluster`` values
            to measure bridged corridors as single entities.
        delta_theta_mesh_rad: Mesh pitch in radians, used as the normalising
            unit. Must be positive.

    Returns:
        One record per unordered pair (``i < j``), ordered by ascending
        directional clearance. Length is ``N * (N - 1) / 2``.

    Raises:
        ValueError: If ``delta_theta_mesh_rad`` is not positive.
    """
    if delta_theta_mesh_rad <= 0:
        raise ValueError(
            f"delta_theta_mesh_rad must be positive, got {delta_theta_mesh_rad}"
        )

    n = len(clusters)
    if n < 2:
        return []

    units = np.array([c.centroid_unit for c in clusters], dtype=np.float64)
    norms = np.linalg.norm(units, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    units = units / norms

    records: list[PairwiseClearance] = []
    for i in range(n):
        for j in range(i + 1, n):
            cos_ij = float(np.clip(float(np.dot(units[i], units[j])), -1.0, 1.0))
            centroid_rad = math.acos(cos_ij)
            reach_i = cluster_directional_support_radius(clusters[i], units[j])
            reach_j = cluster_directional_support_radius(clusters[j], units[i])
            gap_rad = centroid_rad - reach_i - reach_j
            records.append(
                PairwiseClearance(
                    id_a=clusters[i].cluster_id,
                    id_b=clusters[j].cluster_id,
                    composition_a=clusters[i].composition_summary,
                    composition_b=clusters[j].composition_summary,
                    centroid_rad=centroid_rad,
                    centroid_dtheta=centroid_rad / delta_theta_mesh_rad,
                    gap_rad=gap_rad,
                    gap_dtheta=gap_rad / delta_theta_mesh_rad,
                    bearing_a_deg=_bearing_to_axis_deg(
                        units[i], units[j], np.array(clusters[i].principal_axis_unit)
                    ),
                    chi_sum=clusters[i].local_euler_index
                    + clusters[j].local_euler_index,
                )
            )

    records.sort(key=lambda r: r.gap_dtheta)
    return records


def histogram_bins(
    values: Sequence[float],
    bin_width: float = 0.5,
    max_value: float = 12.0,
) -> list[tuple[float, float, int]]:
    """Bin values into fixed-width buckets with overflow and underflow bins.

    Args:
        values: Values to bin.
        bin_width: Bucket width. Must be positive.
        max_value: Upper edge of the last finite bucket.

    Returns:
        List of ``(low, high, count)`` tuples, ordered ascending. The first
        bucket is the underflow bin ``(-inf, 0)``, which collects negative
        clearances (overlapping fitted boundaries); the last is the overflow bin
        ``[max_value, +inf)``. Every input value lands in exactly one bucket, so
        the counts sum to ``len(values)``.

    Raises:
        ValueError: If ``bin_width`` is not positive.
    """
    if bin_width <= 0:
        raise ValueError(f"bin_width must be positive, got {bin_width}")

    n_finite = max(1, math.ceil(max_value / bin_width))
    edges = [float(k) * bin_width for k in range(n_finite + 1)]

    # counts[0] is the underflow bin for negative separations, counts[1..n]
    # are the finite buckets, counts[n + 1] is the overflow bin.
    counts = [0] * (n_finite + 2)

    for v in values:
        if v < 0.0:
            counts[0] += 1
            continue
        idx = math.floor(v / bin_width)
        if idx >= n_finite:
            counts[n_finite + 1] += 1
        else:
            counts[idx + 1] += 1

    bins: list[tuple[float, float, int]] = [(-math.inf, 0.0, counts[0])]
    for k in range(n_finite):
        bins.append((edges[k], edges[k + 1], counts[k + 1]))
    bins.append((edges[-1], math.inf, counts[n_finite + 1]))
    return bins


def _zone_for(low: float, high: float) -> str:
    """Label the scale zone a bin sits in, in pitch units.

    Args:
        low: Lower bin edge.
        high: Upper bin edge.

    Returns:
        ``"overlap"`` for negative bins, else the zone whose range covers the
        bin centre: ``"intra"``, ``"meso"``, or ``"macro"``.
    """
    if high <= 0.0:
        return "overlap"
    centre = low if math.isinf(high) else 0.5 * (low + high)
    if math.isinf(centre):
        return ZONE_LABELS[2]
    lo, hi = ZONE_EDGES_DTHETA
    if centre < lo:
        return ZONE_LABELS[0]
    if centre < hi:
        return ZONE_LABELS[1]
    return ZONE_LABELS[2]


def render_ascii_histogram(
    values: Sequence[float],
    label: str,
    bin_width: float = 0.5,
    max_value: float = 12.0,
    bar_width: int = 40,
) -> str:
    """Render a terminal bar chart of a distribution.

    Args:
        values: Values to plot, in mesh-pitch units.
        label: Title line.
        bin_width: Bucket width.
        max_value: Upper edge of the last finite bucket.
        bar_width: Character width of the longest bar.

    Returns:
        Multi-line string, ready to print. Bars scale to the largest bin, so an
        empty distribution renders as an all-zero chart rather than failing.
    """
    bins = histogram_bins(values, bin_width=bin_width, max_value=max_value)
    # The overflow bin collects every distant pair, so on a sphere it dwarfs
    # everything else and flattens the chart. Scaling to the finite peak keeps
    # the meso trough -- the whole point of the chart -- readable.
    finite = [count for _, high, count in bins if not math.isinf(high)]
    peak = max(finite, default=0)

    lines: list[str] = [
        (
            f"{label}  (n = {len(values)} pairs, bins of {bin_width:g} dtheta, "
            f"bars scaled to {peak})"
        ),
        "-" * 78,
    ]
    for low, high, count in bins:
        if math.isinf(high):
            edge = f">{low:g}".rjust(13)
        elif math.isinf(low):
            edge = f"<{high:g}".rjust(13)
        else:
            edge = f"{low:5.1f} -{high:5.1f}".rjust(13)
        zone = _zone_for(low, high)
        # Finite bins cannot exceed the peak, but the overflow bin can exceed it
        # by orders of magnitude, so the bar is clamped to the chart width.
        filled = 0 if peak == 0 else min(bar_width, round(bar_width * count / peak))
        if math.isinf(high):
            marker = "  (off scale)" if count else ""
        else:
            marker = "  <-- empty" if count == 0 and zone == ZONE_LABELS[1] else ""
        lines.append(
            f"{edge} dtheta [{zone.ljust(8)}] ({count:4d}) |{'#' * filled}{marker}"
        )
    lines.append("-" * 78)
    lo, hi = ZONE_EDGES_DTHETA
    lines.append(
        f"zones: overlap < 0 | intra < {lo:g} | meso {lo:g}-{hi:g} | macro > {hi:g}"
    )
    return "\n".join(lines)


def render_intermediate_registry(
    records: Sequence[PairwiseClearance],
    low_dtheta: float | None = None,
    high_dtheta: float | None = None,
    limit: int = 40,
) -> str:
    """List the pairs whose clearance falls in the meso (transition) zone.

    These are the decisions a macro-clustering rule would have to make, so they
    are the pairs worth inspecting in the viewer.

    Args:
        records: Output of :func:`compute_pairwise_clearances`.
        low_dtheta: Lower clearance bound; defaults to the meso zone floor.
        high_dtheta: Upper clearance bound; defaults to the meso zone ceiling.
        limit: Maximum rows to print; the remainder is summarised.

    Returns:
        Multi-line string, ready to print.
    """
    lo, hi = ZONE_EDGES_DTHETA
    low = lo if low_dtheta is None else low_dtheta
    high = hi if high_dtheta is None else high_dtheta

    hits = [r for r in records if low <= r.gap_dtheta <= high]
    if not hits:
        return (
            f"INTERMEDIATE REGISTRY ({low:g} - {high:g} dtheta): "
            "empty -- clean scale separation at this threshold"
        )

    lines = [
        f"INTERMEDIATE REGISTRY ({low:g} - {high:g} dtheta): {len(hits)} pair(s)",
        "-" * 78,
    ]
    for r in hits[:limit]:
        bearing = "n/a" if r.bearing_a_deg < 0.0 else f"{r.bearing_a_deg:5.1f} deg"
        lines.append(f"  {r.id_a} [{r.composition_a}] <-> {r.id_b} [{r.composition_b}]")
        lines.append(
            f"      gap {r.gap_dtheta:6.2f} dtheta | centroid "
            f"{r.centroid_dtheta:6.2f} dtheta | bearing from {r.id_a} axis: "
            f"{bearing} | chi_sum {r.chi_sum:+d}"
        )
    if len(hits) > limit:
        lines.append(f"  ... {len(hits) - limit} more")
    lines.append("-" * 78)
    return "\n".join(lines)


def _svg_escape(text: str) -> str:
    """Escape text for inclusion in an SVG XML text node.

    Field names in these files carry non-ASCII prefixes, so the string is left
    as UTF-8 and only the XML metacharacters are escaped.

    Args:
        text: Raw text.

    Returns:
        XML-safe text.
    """
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def write_svg_histogram(
    gap_values: Sequence[float],
    centroid_values: Sequence[float],
    title: str,
    output_path: str,
    bin_width: float = 0.5,
    max_value: float = 12.0,
) -> None:
    """Write a self-contained SVG chart of the two distributions.

    Draws the directional clearance and the centroid separation as overlaid bar
    series on a shared axis, with vertical guides at the zone edges so the
    chart can be read against the pipeline's own thresholds. No third-party
    plotting library is required, so the file opens in VS Code or a browser next
    to the viewer.

    Args:
        gap_values: Directional clearances in pitch units.
        centroid_values: Centroid separations in pitch units.
        title: Chart title (typically the field name).
        output_path: Destination ``.svg`` path.
        bin_width: Bucket width.
        max_value: Upper edge of the last finite bucket.
    """
    gap_bins = histogram_bins(gap_values, bin_width=bin_width, max_value=max_value)
    cent_bins = histogram_bins(
        centroid_values, bin_width=bin_width, max_value=max_value
    )

    width, height = 900, 420
    left, right, top, bottom = 70, 30, 60, 60
    plot_w = width - left - right
    plot_h = height - top - bottom
    n_bins = len(gap_bins)
    # Scale to the finite bins only: the overflow bin holds every distant pair
    # and would compress the meso trough, the feature the chart exists to show,
    # into a flat line.
    finite = [c for _, high, c in gap_bins if not math.isinf(high)] + [
        c for _, high, c in cent_bins if not math.isinf(high)
    ]
    peak = max(finite + [1])
    lo, hi = ZONE_EDGES_DTHETA

    # Slot 0 is the underflow bin for negative clearances, so value v sits in
    # slot (v / bin_width) + 1 and the axis spans [-bin_width, n_bins*bw).
    span = n_bins * bin_width

    def x_of(value: float) -> float:
        return left + plot_w * ((value + bin_width) / span)

    def y_of(count: int) -> float:
        return top + plot_h * (1.0 - count / peak)

    slot = plot_w / n_bins
    parts: list[str] = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}">'
        ),
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        (
            f'<text x="{left}" y="26" font-family="sans-serif" font-size="15" '
            f'font-weight="bold">{_svg_escape(title)}</text>'
        ),
        (
            f'<text x="{left}" y="45" font-family="sans-serif" font-size="11" '
            f'fill="#555555">pair separations in mesh-pitch units '
            f"(gap = directional edge clearance, centroid = centre to centre)"
            f"</text>"
        ),
    ]

    tick_step = max(bin_width, 1.0)
    v = 0.0
    while v <= max_value + 1e-9:
        xv = x_of(v)
        parts.append(
            f'<line x1="{xv:.1f}" y1="{top}" x2="{xv:.1f}" y2="{top + plot_h}" '
            f'stroke="#eeeeee" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{xv:.1f}" y="{top + plot_h + 16}" font-family="sans-serif" '
            f'font-size="10" text-anchor="middle">{v:g}</text>'
        )
        v += tick_step

    zero_x = x_of(0.0)
    parts.append(
        f'<line x1="{zero_x:.1f}" y1="{top}" x2="{zero_x:.1f}" y2="{top + plot_h}" '
        f'stroke="#666666" stroke-width="1.5"/>'
    )
    parts.append(
        f'<text x="{zero_x - 4:.1f}" y="{top + plot_h + 16}" '
        f'font-family="sans-serif" font-size="10" text-anchor="end">touch</text>'
    )

    for guide, colour in ((lo, "#e8a33d"), (hi, "#c0392b")):
        xv = x_of(guide)
        parts.append(
            f'<line x1="{xv:.1f}" y1="{top}" x2="{xv:.1f}" y2="{top + plot_h}" '
            f'stroke="{colour}" stroke-width="1.5" stroke-dasharray="5,4"/>'
        )
        parts.append(
            f'<text x="{xv + 4:.1f}" y="{top + 12}" font-family="sans-serif" '
            f'font-size="10" fill="{colour}">{guide:g}</text>'
        )

    half = slot * 0.44
    for bins, colour, opacity in (
        (cent_bins, "#2c7fb8", "0.45"),
        (gap_bins, "#d95f02", "0.70"),
    ):
        for k, (low, high, count) in enumerate(bins):
            if count == 0:
                continue
            centre = left + slot * (k + 0.5)
            # Overflow bars exceed the plot once scaling ignores them, so clip
            # to the top and print the true count above the bar.
            clipped = count > peak
            y = top if clipped else y_of(count)
            parts.append(
                f'<rect x="{centre - half:.1f}" y="{y:.1f}" width="{2 * half:.1f}" '
                f'height="{top + plot_h - y:.1f}" fill="{colour}" '
                f'fill-opacity="{opacity}"/>'
            )
            if clipped:
                parts.append(
                    f'<text x="{centre:.1f}" y="{top - 5:.1f}" '
                    f'font-family="sans-serif" font-size="9" fill="{colour}" '
                    f'text-anchor="middle">{count}</text>'
                )

    parts.append(
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" '
        f'y2="{top + plot_h}" stroke="#333333" stroke-width="1.5"/>'
    )
    parts.append(
        f'<text x="{left + plot_w / 2:.0f}" y="{height - 16}" '
        f'font-family="sans-serif" font-size="11" text-anchor="middle">'
        f"separation in mesh-pitch units (dtheta); left of 'touch' = overlap</text>"
    )
    parts.append(
        f'<text x="18" y="{top + plot_h / 2:.0f}" font-family="sans-serif" '
        f'font-size="11" text-anchor="middle" transform="rotate(-90 18 '
        f'{top + plot_h / 2:.0f})">pairs</text>'
    )
    parts.append(
        f'<text x="{left + plot_w - 250}" y="{top + 14}" font-family="sans-serif" '
        f'font-size="11"><tspan fill="#d95f02">gap</tspan>  '
        f'<tspan fill="#2c7fb8">centroid</tspan>  '
        f'<tspan fill="#555555">peak bin = {peak}</tspan></text>'
    )
    parts.append("</svg>")

    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(parts))
