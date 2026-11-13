"""diagnose_params.py - headless runner for saved sphere_viewer parameter snapshots.

Takes a JSON file written by the viewer's "Save Parameters" button and re-runs
the analysis subroutines that the GUI controls trigger, in the viewer's own
order. Every stage is callable in isolation with --only, so an observation on
screen can be reproduced and bisected without launching the viewer.

Stages (viewer order, --only stops after the named one, inclusive):
    geometry      analyze_mesh_geometry
    morse         compute_discrete_morse_cps
    clusters      compute_micro_clusters
    bridge        bridge_micro_clusters (Step 2 corridor bridging)
    units         build_reduction_units (Step 4 corridor bridging)
    catastrophes  classify_catastrophes (Step 5)
    ports         profile_boundary_loop (+ perfect/reposition)
    reduction     Step 6 effective-CP reduction + harmonic nudge
    basins        extract_basin_centers
    launch        expand_excision_seed (excision-boundary launch seeds)
    connect       connect_seed_to_maxima (Step 3 ascent-branch first-hit), then the
                  Step 3 ridge post-pass: refine_connections runs for
                  refine_ridge_paths (fidelity smoother) and coarse_ridge_curves
                  (structural curve), exactly as the viewer's card drives it
    probe         probe_gradient_ridge (Step 3 CP probing) on every traced ascent

The connect stage's summary carries the card's per-connector ridge output as well as
the aggregate: ``connection_ridge_details`` / ``summarize_ridge_postpass`` publish
what the post-pass did to each leg (``refined``, ``refine_max_shift_deg``,
``n_nodes`` for the coarse curve, ``n_runs`` for the smoother, and the turn totals),
and ``--out`` writes them per connector. The probe stage exists because
``probe_ridge_points`` changes no geometry at all, so no other metric can show it:
``--set probe_ridge_points=true`` runs it and reports the viewer's own
``ridge_probe_stats`` counters.

The saved file's derived_metrics are diffed against the recomputed ones, so the
run answers "does this parameter set reproduce what I saw?" with an exit code
(0 = no drift, 1 = drift). Display-only controls (show_*, cp_glyph_scale,
num_contours) have no headless analogue and are accepted but inert.

Run with:
    uv run python gba_topology2/diagnose_params.py ethene4_atom1_params.json
    uv run python gba_topology2/diagnose_params.py <params.json> --only clusters
    uv run python gba_topology2/diagnose_params.py <params.json> --set k_pitch=3.0
    uv run python gba_topology2/diagnose_params.py <params.json> --out results.json
    uv run python gba_topology2/diagnose_params.py <params.json> \
        --set probe_ridge_points=true --out probed.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt
from vtkmodules.util import numpy_support

workspace_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)

from gba_topology2.src.basin_boundary import (
    build_connection_graph,
    find_minimum_basins,
)
from gba_topology2.src.basin_center import BasinCenter, extract_basin_centers
from gba_topology2.src.catastrophe_classifier import (
    EffectiveCatastrophe,
    classify_catastrophes,
)
from gba_topology2.src.excision_boundary import (
    KNearestFinder,
    profile_boundary_loop,
)
from gba_topology2.src.excision_launch_seeds import (
    expand_excision_seed,
    summarize_outcomes,
)
from gba_topology2.src.harmonic_nudge import (
    harmonic_nudge_critical_point,
    perfect_boundary_ports,
    reposition_ports_on_ring,
)
from gba_topology2.src.max_connection import (
    ConnectionTarget,
    DiscGeometry,
    MaxConnection,
    build_connection_pool,
    connect_seed_to_maxima,
    disc_geometry_from_cluster,
    refine_connections,
    summarize_connections,
)
from gba_topology2.src.mesh_geometry import analyze_mesh_geometry
from gba_topology2.src.micro_cluster import MicroCluster, compute_micro_clusters
from gba_topology2.src.morse_detector import (
    CriticalPoint,
    MorseDetectionResult,
    compute_discrete_morse_cps,
)
from gba_topology2.src.point_locator import make_kdtree_nearest_finder
from gba_topology2.src.polarity_detector import (
    classify_cluster_polarity,
    effective_reduction_decision,
)
from gba_topology2.src.ridge_refiner import probe_gradient_ridge
from gba_topology2.src.saddle_launch import SaddleLaunchSeed
from gba_topology2.src.separatrix_tracer import excision_regions
from gba_topology2.src.sphere_frame import to_global
from gba_topology2.src.sphere_viewer import (
    DEFAULT_UI_STATE,
    PARAM_BOOL_KEYS,
    PARAM_CONTEXT_STRING_KEYS,
    PARAM_FLOAT_KEYS,
    PARAM_STRING_KEYS,
    ReductionUnit,
    build_launch_requests,
    build_reduction_units,
    catastrophe_claimed_cluster_ids,
    effective_cp_to_ui,
    extract_atom_sphere_mesh,
    fusable_catastrophes,
    load_params_from_dict,
    make_ridge_probe_stats,
)
from gba_topology2.src.string_aggregator import (
    BridgeLink,
    compute_bridge_links,
    compute_group_geometry,
    is_degenerate_group,
    select_elongated_core,
)

__all__ = [
    "STAGES",
    "MetricDiff",
    "RunContext",
    "apply_overrides",
    "build_context",
    "computed_metrics",
    "diff_leaf",
    "diff_metrics",
    "run_stages",
]


@dataclass
class RunContext:
    """Everything the pipeline stages read, plus their accumulated results.

    Attributes:
        params: Flat control dict from ``load_params_from_dict``.
        plt_path: Absolute .plt path actually loaded.
        atom_num: Target atom number.
        field_name: Condensed field array analysed.
        center: World-frame nucleus.
        pts: (N, 3) centre-relative mesh vertices.
        triangles: (M, 3) triangle vertex indices.
        f_vals: (N,) scalar field values.
        sphere_radius: Atom sphere radius R.
        delta_theta_rad: Intrinsic angular mesh pitch.
        geom_info: Recomputed mesh geometry metrics (viewer state shape).
        morse: Raw Morse detection result, set by the morse stage.
        morse_counts: Stage summary matching the viewer's state key.
        raw_cps: All raw critical points, centred frame.
        clusters: Micro-clusters.
        bridge_links: Per-pair bridging verdicts from the bridge stage.
        bridge_components: Connected components of the bridging graph.
        bridge_gate: Per-component elongation-gate outcome.
        units: Reduction units (cluster or bridged corridor).
        n_bridged: Clusters folded into a corridor.
        catastrophes: Detected catastrophe constellations.
        catastrophes_ui: Viewer-shaped catastrophe dicts.
        n_ports_perfected: Ports re-placed by the perfect-angles toggle.
        clusters_ui: Viewer-shaped reduction-unit dicts.
        effective_cps: Effective CP dicts produced by the reduction stage.
        hidden_vertex_ids: Raw vertices collapsed into an effective entity.
        reduction_stats: Step 6 counters, matching the viewer's state key.
        display_counts: Rendered CP counts and index sum.
        basin_centers: Candidate basin centers, world frame.
        seeds: Saddle launch seeds.
        launch_stats: Launch counters, matching the viewer's state key.
        unverified_seeds: Provisional seeds (ring k != reduction k), rendered
            faded by the viewer and connected alongside the verified ones.
        connections: Ascent-branch to maximum claims (Step 3).
        connection_stats: Step 3 counters, matching the viewer's state key.
        connection_pool: The visible-CP pool the claims were made against.
    """

    params: dict[str, Any]
    plt_path: str
    atom_num: int
    field_name: str
    center: tuple[float, float, float]
    pts: npt.NDArray[np.float64]
    triangles: npt.NDArray[np.integer[Any]]
    f_vals: npt.NDArray[np.float64]
    sphere_radius: float
    delta_theta_rad: float
    geom_info: dict[str, Any]
    morse: MorseDetectionResult | None = None
    morse_counts: dict[str, Any] = field(default_factory=dict)
    raw_cps: list[CriticalPoint] = field(default_factory=list)
    clusters: list[MicroCluster] = field(default_factory=list)
    bridge_links: list[BridgeLink] = field(default_factory=list)
    bridge_components: list[list[str]] = field(default_factory=list)
    bridge_gate: list[dict[str, Any]] = field(default_factory=list)
    units: list[ReductionUnit] = field(default_factory=list)
    n_bridged: int = 0
    catastrophes: list[EffectiveCatastrophe] = field(default_factory=list)
    catastrophes_ui: list[dict[str, Any]] = field(default_factory=list)
    n_ports_perfected: int = 0
    clusters_ui: list[dict[str, Any]] = field(default_factory=list)
    effective_cps: list[dict[str, Any]] = field(default_factory=list)
    hidden_vertex_ids: set[int] = field(default_factory=set)
    eff_ring_radius: dict[str, float] = field(default_factory=dict)
    eff_disc_geometry: dict[str, DiscGeometry] = field(default_factory=dict)
    locator: KNearestFinder | None = None
    reduction_stats: dict[str, Any] = field(default_factory=dict)
    display_counts: dict[str, Any] = field(default_factory=dict)
    basin_centers: list[BasinCenter] = field(default_factory=list)
    seeds: list[SaddleLaunchSeed] = field(default_factory=list)
    launch_stats: dict[str, Any] = field(default_factory=dict)
    unverified_seeds: list[SaddleLaunchSeed] = field(default_factory=list)
    connections: list[MaxConnection] = field(default_factory=list)
    connection_stats: dict[str, Any] = field(default_factory=dict)
    #: The traced table as it left ``connect_seed_to_maxima``, BEFORE the Step 3
    #: ridge post-pass replaced it. Mirrors the viewer's
    #: ``runtime_traced_connections``: the probe measures the ASCENT, so it must
    #: read this table and not the smoothed one drawn beside it.
    traced_connections: list[MaxConnection] = field(default_factory=list)
    #: Per-connection output of the Step 3 ridge switches, for ``--out``.
    connection_details: list[dict[str, Any]] = field(default_factory=list)
    #: Ridge-probe counters under the viewer's ``state.ridge_probe_stats`` names.
    ridge_probe_stats: dict[str, Any] = field(default_factory=dict)
    #: Per-connection ridge-probe measurements, for ``--out``.
    probe_details: list[dict[str, Any]] = field(default_factory=list)
    #: The viewer's arc-selection scope for the probe, read from the snapshot's
    #: ``selection`` section. The probe measures the connectors ON SCREEN, so a
    #: replay has to know what was on screen to reproduce the counters.
    probe_basin_id: str | None = None
    #: The saddle end of the same scope. Naming a saddle is how a stalled ascent
    #: gets seen, and it is mutually exclusive with ``probe_basin_id``.
    probe_saddle_id: str | None = None
    #: ``--probe-all``: ignore the display gates and the selection, and probe
    #: every gradient connector. A superset of the screen, for comparing two
    #: parameter sets over the whole run.
    probe_all: bool = False
    connection_pool: list[ConnectionTarget] = field(default_factory=list)
    #: Minimum basin boundaries computed from connections.
    minimum_basins: dict[str, Any] = field(default_factory=dict)
    connection_graph: dict[str, list[Any]] = field(default_factory=dict)

    def require_morse(self) -> MorseDetectionResult:
        """Return the Morse result, requiring the morse stage to have run.

        Returns:
            The stored MorseDetectionResult.

        Raises:
            RuntimeError: If the morse stage has not run yet.
        """
        if self.morse is None:
            raise RuntimeError("the 'morse' stage must run before this stage")
        return self.morse


def _p(ctx: RunContext, key: str) -> Any:
    """Read one control value from the context params.

    A control absent from the params reads as its default, which is the same rule
    :func:`load_params_from_dict` applies to a legacy snapshot: a file predating a
    control shows that control off-screen, so filling it from the default is the
    faithful reading, not a papering-over. Contexts built in code (the stage tests)
    carry only the controls their fixture cares about, and a stage must not be
    forced to enumerate controls invented after it was written.
    """
    if key in ctx.params:
        return ctx.params[key]
    return DEFAULT_UI_STATE[key]


def _require_morse_first(ctx: RunContext) -> None:
    """Guard stages that consume raw_cps set by the morse stage.

    Args:
        ctx: The run context.

    Raises:
        RuntimeError: If the morse stage has not run yet.
    """
    if ctx.morse is None:
        raise RuntimeError("the 'morse' stage must run before this stage")


def stage_geometry(ctx: RunContext) -> dict[str, Any]:
    """Report the mesh geometry metrics the viewer's header card shows."""
    return dict(ctx.geom_info)


def stage_morse(ctx: RunContext) -> dict[str, Any]:
    """Run the Discrete Morse detector at the viewer's SoS tie-break epsilon."""
    morse = compute_discrete_morse_cps(
        ctx.pts, ctx.triangles, ctx.f_vals, tie_break_epsilon=1e-15
    )
    ctx.morse = morse
    ctx.raw_cps = morse.minima + morse.maxima + morse.saddles
    ctx.morse_counts = {
        "minima": len(morse.minima),
        "maxima": len(morse.maxima),
        "saddles": len(morse.saddles),
        "euler": morse.euler_characteristic,
        "valid": morse.is_valid_euler,
    }
    return dict(ctx.morse_counts)


def stage_clusters(ctx: RunContext) -> dict[str, Any]:
    """Run micro-clustering with the saved k_pitch."""
    _require_morse_first(ctx)
    ctx.clusters = compute_micro_clusters(
        cps=ctx.raw_cps,
        delta_theta_mesh_rad=ctx.delta_theta_rad,
        k_pitch=float(_p(ctx, "k_pitch")),
        sphere_radius=ctx.sphere_radius,
        min_boundary_margin_deg=1.5,
        num_boundary_points=48,
    )
    return {
        "n_clusters": len(ctx.clusters),
        "n_multi_cp": sum(1 for c in ctx.clusters if c.is_multi_cp),
        "index_sum": sum(c.local_euler_index for c in ctx.clusters),
    }


def stage_bridge(ctx: RunContext) -> dict[str, Any]:
    """Report the bridging graph and the pooled-elongation gate per component.

    This stage exists because the "Bridge Radius" control has a non-obvious
    effect on the number of effective critical points. The bridging rule is
    two-stage: a graph edge is created when the uncovered gap is within
    ``k_bridge`` delta_theta units *and* the principal axes are collinear, and
    the connected components of that graph are then gated by their **pooled**
    tangent-plane aspect ratio. Because the gate is a pooled property it is not
    decomposable -- a single off-axis satellite can sink a group that was
    otherwise a valid corridor -- so the response to ``k_bridge`` is not
    necessarily monotone.

    The stage reports the per-pair verdicts and the pooled aspect ratio of
    every component, which is what makes a change in the effective-CP count
    explainable rather than mysterious.

    Returns:
        Summary of the bridging graph, its components and the gate outcome.
    """
    _require_morse_first(ctx)
    if not ctx.clusters:
        return {
            "n_clusters": 0,
            "n_links": 0,
            "n_components": 0,
            "n_qualified": 0,
        }

    k_bridge = float(_p(ctx, "k_bridge"))
    links = compute_bridge_links(
        clusters=ctx.clusters,
        delta_theta_mesh_rad=ctx.delta_theta_rad,
        k_bridge=k_bridge,
        bridge_all=True,
    )

    # Rebuild the component partition the aggregator uses.
    by_id = {c.cluster_id: c for c in ctx.clusters}
    adj: dict[str, set[str]] = {c.cluster_id: set() for c in ctx.clusters}
    for link in links:
        if link.is_linked:
            adj[link.cluster_a].add(link.cluster_b)
            adj[link.cluster_b].add(link.cluster_a)

    components: list[list[str]] = []
    seen: set[str] = set()
    for cid in adj:
        if cid in seen:
            continue
        comp: list[str] = []
        queue = [cid]
        seen.add(cid)
        while queue:
            curr = queue.pop()
            comp.append(curr)
            for nbr in adj[curr]:
                if nbr not in seen:
                    seen.add(nbr)
                    queue.append(nbr)
        components.append(sorted(comp))

    ctx.bridge_links = links
    ctx.bridge_components = components

    # The gate outcome per component, plus what the core selection keeps.
    qualified = 0
    promoted = 0
    for comp in components:
        clusters = [by_id[cid] for cid in comp]
        pooled = compute_group_geometry(clusters, ctx.sphere_radius)
        core, satellites = select_elongated_core(
            component=clusters,
            aspect_ratio_threshold=2.0,
            sphere_radius=ctx.sphere_radius,
            adjacency=adj,
        )
        if (
            len(clusters) > 1
            and not is_degenerate_group(pooled)
            and pooled.aspect_ratio >= 2.0
        ):
            qualified += 1
        if len(core) > 1:
            promoted += 1
        ctx.bridge_gate.append(
            {
                "component": comp,
                "n_clusters": len(clusters),
                "pooled_aspect_ratio": pooled.aspect_ratio,
                "pooled_is_degenerate": is_degenerate_group(pooled),
                "pooled_chi": sum(by_id[cid].local_euler_index for cid in comp),
                "gate_passed": (
                    len(clusters) > 1
                    and not is_degenerate_group(pooled)
                    and pooled.aspect_ratio >= 2.0
                ),
                "core": [c.cluster_id for c in core],
                "satellites": [c.cluster_id for c in satellites],
                "n_satellites": len(satellites),
            }
        )

    return {
        "n_clusters": len(ctx.clusters),
        "n_links": sum(1 for link in links if link.is_linked),
        "n_components": len(components),
        "n_qualified": qualified,
        "n_promoted": promoted,
        "threshold_deg": math.degrees(k_bridge * ctx.delta_theta_rad),
    }


def stage_units(ctx: RunContext) -> dict[str, Any]:
    """Bridge collinear corridor fragments into reduction units (Step 4)."""
    ctx.units, ctx.n_bridged = build_reduction_units(
        clusters=ctx.clusters,
        delta_theta_mesh_rad=ctx.delta_theta_rad,
        sphere_radius=ctx.sphere_radius,
        aggregate_strings=bool(_p(ctx, "aggregate_strings")),
        k_bridge=float(_p(ctx, "k_bridge")),
    )
    return {
        "n_units": len(ctx.units),
        "n_strings": sum(1 for u in ctx.units if u.is_string),
        "n_bridged": ctx.n_bridged,
        "index_sum": sum(u.local_euler_index for u in ctx.units),
    }


def stage_catastrophes(ctx: RunContext) -> dict[str, Any]:
    """Classify catastrophe constellations (Step 5) on the raw clusters."""
    ctx.catastrophes, _ = classify_catastrophes(
        clusters=ctx.clusters,
        field_vals=ctx.f_vals,
        delta_theta_mesh_rad=ctx.delta_theta_rad,
        sphere_radius=ctx.sphere_radius,
        barrier_tol=float(_p(ctx, "cat_barrier_tol")),
        angular_tol_mult=float(_p(ctx, "cat_angular_tol_mult")),
    )
    return {
        "n_catastrophes": len(ctx.catastrophes),
        "n_fused": sum(1 for c in ctx.catastrophes if c.is_fused),
        "n_split": sum(1 for c in ctx.catastrophes if not c.is_fused),
        # How many constellations the reduction stage may actually collapse.
        # Only fused (spurious) ones qualify; a split constellation keeps its
        # distinct poles, so its core cluster's extrema stay visible.
        "n_fusable": len(fusable_catastrophes(ctx.catastrophes)),
        "max_fold_order": max((c.fold_order for c in ctx.catastrophes), default=0),
    }


def stage_ports(ctx: RunContext) -> dict[str, Any]:
    """Profile every catastrophe excision boundary and count its ports."""
    do_perfect = bool(_p(ctx, "perfect_port_angles"))
    n_valley = n_ridge = n_balanced = 0
    for cat in ctx.catastrophes:
        b_res = profile_boundary_loop(
            ring_points=cat.boundary_ring, mesh_pts=ctx.pts, f_vals=ctx.f_vals
        )
        v_ports, r_ports = b_res.valley_ports, b_res.ridge_ports
        if do_perfect:
            v_ports = reposition_ports_on_ring(
                perfect_boundary_ports(v_ports, cat.fold_order),
                cat.boundary_ring,
                ctx.sphere_radius,
                ctx.f_vals,
                ctx.pts,
            )
            r_ports = reposition_ports_on_ring(
                perfect_boundary_ports(r_ports, cat.fold_order),
                cat.boundary_ring,
                ctx.sphere_radius,
                ctx.f_vals,
                ctx.pts,
            )
        n_valley += len(v_ports)
        n_ridge += len(r_ports)
        n_balanced += int(b_res.is_topologically_balanced)
        ctx.catastrophes_ui.append(
            {
                "id": cat.entity_id,
                "type": cat.catastrophe_type,
                "classification": cat.classification,
                "is_fused": cat.is_fused,
                "fold_order": cat.fold_order,
                "net_euler_index": cat.net_euler_index,
                "centroid": list(to_global(cat.centroid, ctx.center)),
                "angular_span_deg": cat.angular_span_deg,
                "barrier_depth": cat.barrier_depth,
                "relative_barrier_depth": cat.relative_barrier_depth,
                "num_valleys": len(v_ports),
                "num_ridges": len(r_ports),
                "is_balanced": b_res.is_topologically_balanced,
                "constituent_cluster_ids": [
                    cl.cluster_id for cl in cat.constituent_clusters
                ],
                "all_member_count": len(cat.all_members),
            }
        )
    ctx.n_ports_perfected = (n_valley + n_ridge) if do_perfect else 0
    return {
        "n_catastrophes_profiled": len(ctx.catastrophes),
        "n_valley_ports": n_valley,
        "n_ridge_ports": n_ridge,
        "n_balanced_loops": n_balanced,
        "ports_perfected": do_perfect,
    }


def stage_reduction(ctx: RunContext) -> dict[str, Any]:
    """Collapse catastrophes and multi-CP units into effective CPs (Step 6).

    Mirrors ``sphere_viewer.update_topology`` stages 4-7: catastrophe
    collapse, standalone multi-cluster reduction with the polarity/index
    decision, annulment/irreducible accounting, the harmonic nudge, and the
    resulting displayed-CP counts and index sum.
    """
    morse = ctx.require_morse()
    reduce_cats = bool(_p(ctx, "reduce_catastrophes"))
    reduce_clusts = bool(_p(ctx, "reduce_multi_clusters"))
    do_nudge = bool(_p(ctx, "nudge_cp_positions"))
    do_perfect = bool(_p(ctx, "perfect_port_angles"))
    max_nudge_pitch = float(_p(ctx, "nudge_max_pitch"))

    hidden: set[int] = set()
    effs: list[dict[str, Any]] = []
    eff_ring: dict[str, float] = {}
    eff_disc: dict[str, DiscGeometry] = {}
    n_cats = n_clusts = n_annul = n_irred = 0
    nudge_degs: list[float] = []
    pitch_deg = max(math.degrees(ctx.delta_theta_rad), 1e-9)

    for cat in ctx.catastrophes:
        # Only *fused* (spurious) catastrophes are collapsed; a split
        # constellation keeps its distinct poles. Mirrors the viewer.
        fuse_this = reduce_cats and cat.is_fused
        cat_nudge = None
        if fuse_this and do_nudge:
            cat_nudge = harmonic_nudge_critical_point(
                initial_pos=cat.centroid,
                mesh_pts=ctx.pts,
                f_vals=ctx.f_vals,
                fold_order=cat.fold_order,
                delta_theta_mesh_rad=ctx.delta_theta_rad,
                sphere_radius=ctx.sphere_radius,
                max_displacement_pitch=max_nudge_pitch,
                probe_radius_pitch=max(1.5, cat.angular_span_deg / 2.0 / pitch_deg),
            )
        if fuse_this:
            eff_pos = cat_nudge.optimized_position if cat_nudge else cat.centroid
            if cat_nudge is not None:
                nudge_degs.append(cat_nudge.displacement_ang_deg)
            cat_entity_id = f"E-SAD_{cat.entity_id}"
            effs.append(
                effective_cp_to_ui(
                    entity_id=cat_entity_id,
                    eff_type="E-SAD",
                    position=eff_pos,
                    value=max((m.value for m in cat.all_members), default=0.0),
                    index=cat.net_euler_index,
                    members=cat.all_members,
                    origin_id=cat.entity_id,
                    origin_label="Catastrophe",
                    fold_order=cat.fold_order,
                    nudge=cat_nudge if do_nudge else None,
                    center=ctx.center,
                )
            )
            eff_ring[cat_entity_id] = cat.angular_radius_rad
            eff_disc[cat_entity_id] = DiscGeometry(
                angular_radius_rad=float(cat.angular_radius_rad),
                semi_major_rad=float(cat.angular_radius_rad),
                semi_minor_rad=float(cat.angular_radius_rad),
            )
            hidden.update(m.vertex_id for m in cat.all_members)
            n_cats += 1

    claimed = catastrophe_claimed_cluster_ids(ctx.catastrophes, reduce_cats)
    annihilated: set[str] = set()
    irreducible: set[str] = set()

    if reduce_clusts:
        for unit in ctx.units:
            if len(unit.members) <= 1:
                continue
            if any(oid in claimed for oid in unit.origin_ids):
                continue
            eff = classify_cluster_polarity(
                cluster=unit.as_cluster,
                mesh_pts=ctx.pts,
                f_vals=ctx.f_vals,
                delta_theta_mesh_rad=ctx.delta_theta_rad,
                sphere_radius=ctx.sphere_radius,
            )
            decision = effective_reduction_decision(
                unit.local_euler_index, eff.extremum_type
            )
            if decision == "E-NONE":
                hidden.update(m.vertex_id for m in unit.members)
                annihilated.add(unit.unit_id)
                n_annul += 1
                continue
            if decision == "IRREDUCIBLE":
                irreducible.add(unit.unit_id)
                n_irred += 1
                continue

            eff_index = unit.local_euler_index
            eff_val = eff.effective_value
            fold_order = 0
            if decision == "E-SAD":
                eff_val = max((m.value for m in unit.members), default=0.0)
                fold_order = 1 - eff_index
            cl_nudge = None
            if do_nudge:
                cl_nudge = harmonic_nudge_critical_point(
                    initial_pos=unit.centroid,
                    mesh_pts=ctx.pts,
                    f_vals=ctx.f_vals,
                    fold_order=fold_order,
                    delta_theta_mesh_rad=ctx.delta_theta_rad,
                    sphere_radius=ctx.sphere_radius,
                    max_displacement_pitch=max_nudge_pitch,
                    probe_radius_pitch=max(
                        1.5, unit.as_cluster.angular_radius_deg / pitch_deg
                    ),
                )
            eff_pos = cl_nudge.optimized_position if cl_nudge else unit.centroid
            if cl_nudge is not None:
                nudge_degs.append(cl_nudge.displacement_ang_deg)
            unit_entity_id = f"{decision}_{unit.unit_id}"
            effs.append(
                effective_cp_to_ui(
                    entity_id=unit_entity_id,
                    eff_type=decision,
                    position=eff_pos,
                    value=eff_val,
                    index=eff_index,
                    members=unit.members,
                    origin_id=unit.unit_id,
                    origin_label=(
                        "Saddle Corridor" if unit.is_string else "Micro-Cluster"
                    ),
                    fold_order=fold_order,
                    nudge=cl_nudge if do_nudge else None,
                    center=ctx.center,
                )
            )
            eff_ring[unit_entity_id] = unit.as_cluster.angular_radius_rad
            eff_disc[unit_entity_id] = disc_geometry_from_cluster(unit.as_cluster)
            hidden.update(m.vertex_id for m in unit.members)
            n_clusts += 1

    # Displayed counts mirror the viewer's glyph bookkeeping: every visible
    # raw CP contributes one glyph, and every effective CP contributes one
    # glyph of its own kind. The index sum uses multiplicity for raw saddles
    # and the net index for effective CPs, so collapsing a set never changes
    # the displayed chi.
    n_max = sum(1 for m in morse.maxima if m.vertex_id not in hidden)
    n_min = sum(1 for m in morse.minima if m.vertex_id not in hidden)
    n_sad = sum(1 for m in morse.saddles if m.vertex_id not in hidden)
    index_sum = (
        n_max
        + n_min
        - sum(m.multiplicity for m in morse.saddles if m.vertex_id not in hidden)
    )
    for eff_cp in effs:
        if eff_cp["type"] == "E-MAX":
            n_max += 1
        elif eff_cp["type"] == "E-MIN":
            n_min += 1
        else:
            n_sad += 1
        index_sum += int(eff_cp["net_index"])

    ctx.hidden_vertex_ids = hidden
    ctx.effective_cps = effs
    ctx.eff_ring_radius = eff_ring
    ctx.eff_disc_geometry = eff_disc
    ctx.display_counts = {
        "minima": n_min,
        "maxima": n_max,
        "saddles": n_sad,
        "euler": index_sum,
        "valid": index_sum == 2,
    }
    ctx.reduction_stats = {
        "n_catastrophes_reduced": n_cats,
        "n_clusters_reduced": n_clusts,
        "n_clusters_annihilated": n_annul,
        "n_clusters_irreducible": n_irred,
        "n_clusters_bridged": ctx.n_bridged,
        "n_effective_cps": len(effs),
        "n_raw_cps_hidden": len(hidden),
        "n_nudged": len(nudge_degs) if do_nudge else 0,
        "max_nudge_deg": max(nudge_degs) if (do_nudge and nudge_degs) else 0.0,
        "mean_nudge_deg": (
            sum(nudge_degs) / len(nudge_degs) if (do_nudge and nudge_degs) else 0.0
        ),
        "n_ports_perfected": ctx.n_ports_perfected if do_perfect else 0,
    }
    ctx.clusters_ui = [
        {
            "id": u.unit_id,
            "size": len(u.members),
            "local_euler_index": u.local_euler_index,
            "morphology": u.morphology,
            "aspect_ratio": u.aspect_ratio,
            "is_string": u.is_string,
            "origin_ids": list(u.origin_ids),
            "centroid": list(to_global(u.centroid, ctx.center)),
            "is_annulled": u.unit_id in annihilated,
            "is_irreducible": u.unit_id in irreducible,
            "members": [
                {
                    "vertex_id": m.vertex_id,
                    "type": m.cp_type,
                    "value": m.value,
                    "multiplicity": m.multiplicity,
                }
                for m in u.members
            ],
        }
        for u in ctx.units
    ]
    return {**ctx.reduction_stats, "display_counts": dict(ctx.display_counts)}


def stage_basins(ctx: RunContext) -> dict[str, Any]:
    """Extract candidate basin centers from the visible extrema."""
    _require_morse_first(ctx)
    ctx.basin_centers = extract_basin_centers(
        raw_cps=ctx.raw_cps,
        effective_cps=ctx.effective_cps,
        hidden_vertex_ids=ctx.hidden_vertex_ids,
        center=ctx.center,
    )
    return {
        "n_basin_centers": len(ctx.basin_centers),
        "n_maxima": sum(1 for b in ctx.basin_centers if b.polarity == "maximum"),
        "n_minima": sum(1 for b in ctx.basin_centers if b.polarity == "minimum"),
        "n_effective": sum(1 for b in ctx.basin_centers if b.is_effective),
    }


def stage_launch(ctx: RunContext) -> dict[str, Any]:
    """Extract saddle separatrix launch seeds for the visible saddles.

    Seeds are read off each saddle's excision boundary, so the set is the
    *displayed* saddles: raw saddles that survived reduction plus every effective
    saddle. The cyclic-link scan needs a vertex_id and an effective CP has none,
    which is why the ring substrate covers a set the link scan could not.
    """
    morse = ctx.require_morse()
    requests = build_launch_requests(
        saddles=morse.saddles,
        effective_cps=ctx.effective_cps,
        hidden_vertex_ids=ctx.hidden_vertex_ids,
        eff_ring_radius=ctx.eff_ring_radius,
        center=ctx.center,
        delta_theta_mesh_rad=ctx.delta_theta_rad,
        k_pitch=float(_p(ctx, "k_pitch")),
    )
    # Step, ceiling and sample count come from the excision_launch_seeds module
    # constants (the defaults), not from params: the harness must exercise the
    # same fixed geometry the viewer does, so the two cannot drift.
    outcomes = [
        expand_excision_seed(
            request=req,
            mesh_pts=ctx.pts,
            f_vals=ctx.f_vals,
            sphere_radius=ctx.sphere_radius,
            delta_theta_mesh_rad=ctx.delta_theta_rad,
            find_k_nearest=ctx.locator,
        )
        for req in requests
    ]
    ctx.seeds = [o.seed for o in outcomes if o.converged]
    ctx.unverified_seeds = [o.seed for o in outcomes if o.unverified]
    ctx.launch_stats = summarize_outcomes(outcomes)
    return dict(ctx.launch_stats)


def connection_ridge_details(
    connections: list[MaxConnection],
) -> list[dict[str, Any]]:
    """Record what the Step 3 ridge switches did to each connector.

    ``summarize_connections`` reports the AGGREGATE of the post-pass table, which
    cannot answer "which leg was smoothed, and by how much did it move?". The
    card's own per-row fields can, and they are exactly the ones the viewer
    publishes in ``state.max_connections_list``: the post-pass marks a connector it
    rewrote with ``refined=True`` and reports the deviation as
    ``refine_max_shift_deg``, whichever switch fired; ``n_nodes`` (>0) is the
    coarse curve's structural node count, ``n_runs`` (>0) the fidelity smoother's
    chatter-run count, ``n_corners`` the bends it refused to flatten, and the turn
    totals are the fidelity cost of the line on screen.

    Args:
        connections: The connection table, refined or not.

    Returns:
        One dict per connector, in table order.
    """
    return [
        {
            "id": f"{c.saddle_id}#{c.branch_index}",
            "saddle_id": c.saddle_id,
            "branch_index": c.branch_index,
            "max_id": c.max_id,
            "status": c.status,
            "method": c.method,
            "refined": bool(c.refined),
            "refine_max_shift_deg": float(c.refine_max_shift_deg),
            "n_corners": int(c.n_corners),
            "n_runs": int(c.n_runs),
            "n_nodes": int(c.n_nodes),
            "total_turn_before_deg": float(c.total_turn_before_deg),
            "total_turn_after_deg": float(c.total_turn_after_deg),
        }
        for c in connections
    ]


def summarize_ridge_postpass(
    connections: list[MaxConnection],
) -> dict[str, Any]:
    """The card footer: how many legs the two ridge switches rewrote.

    Three counters, because the switches are separable and a single number cannot
    tell them apart (measured on the ethene3 atom-1 snapshot): both off gives 0/0/0;
    "Coarse Structural Curves" gives ``n_refined=4, n_coarse=4, n_curve_nodes=38``;
    "Smooth Sawtooth Chatter" gives ``n_refined=3, n_coarse=0, n_curve_nodes=0``.
    ``n_refined`` is the UNION -- the post-pass sets ``refined`` for either
    description -- while ``n_coarse`` (structural nodes) and ``n_smoothed``
    (chatter runs) identify which one ran.

    Args:
        connections: The connection table after the post-pass.

    Returns:
        Footer counters, all numeric, safe to print in the stage summary.
    """
    details = connection_ridge_details(connections)
    shifts = [d["refine_max_shift_deg"] for d in details if d["refined"]]
    return {
        "n_refined": sum(1 for d in details if d["refined"]),
        "n_coarse": sum(1 for d in details if d["n_nodes"] > 0),
        "n_smoothed": sum(1 for d in details if d["n_runs"] > 0),
        "max_refine_shift_deg": max(shifts, default=0.0),
        "n_curve_nodes": sum(d["n_nodes"] for d in details),
    }


def stage_connect(ctx: RunContext) -> dict[str, Any]:
    """Find the first critical point each ascent branch runs into (Step 3).

    Mirrors the viewer. The seeds are the verified and provisional sets together,
    because the scene draws both, and each connection carries ``verified`` so the
    two stay separable in the output.

    The pool is every VISIBLE critical point, not just the maxima: a ray that
    flies over a minimum on its way to a distant maximum is occluded, and the
    rule can only see that if the minimum is in the pool. ``extract_basin_centers``
    emits extrema only, so the visible saddles are added from the two places they
    live -- the raw Morse saddles the reduction left unhidden (centred frame) and
    the ``E-SAD`` entries of the reduced set (world frame, no vertex to scan).
    ``build_connection_pool`` reconciles the two frames.

    Each CP carries its excision-disc radius (raw micro-cluster radius keyed by
    vertex, ``eff_ring_radius`` keyed by entity id), so the rule tests the disc a
    ray must clear rather than the centre point it must miss exactly, and a
    maximum is captured where the arc meets its boundary.

    The ray launches from the branch's PORT on the excision boundary and follows
    the outward geodesic; the claim is the critical point whose disc is entered
    first, inside the forward sector. The cap, the sector half-angle and the ray
    definition are module constants, not controls, matching the ring geometry.

    Returns:
        Summary of the connection table, matching the viewer's state key.

    Raises:
        RuntimeError: If the morse stage has not run (the pool needs its saddles).
    """
    morse = ctx.require_morse()
    pool = build_connection_pool(
        basin_centers=ctx.basin_centers,
        raw_saddles=morse.saddles,
        effective_cps=ctx.effective_cps,
        hidden_vertex_ids=ctx.hidden_vertex_ids,
        center=ctx.center,
        disc_by_vertex={
            m.vertex_id: disc_geometry_from_cluster(cl)
            for cl in ctx.clusters
            for m in cl.members
        },
        disc_by_id=ctx.eff_disc_geometry,
    )
    ctx.connection_pool = pool
    regions = excision_regions([u.as_cluster for u in ctx.units])
    ctx.connections = connect_seed_to_maxima(
        seeds=[*ctx.seeds, *ctx.unverified_seeds],
        targets=pool,
        sphere_radius=ctx.sphere_radius,
        delta_theta_mesh_rad=ctx.delta_theta_rad,
        center=ctx.center,
        mesh_pts=ctx.pts,
        f_vals=ctx.f_vals,
        find_k_nearest=ctx.locator,
        excision_regions=regions,
    )
    ctx.traced_connections = list(ctx.connections)
    # The same controls the viewer's switches drive, read the same way every other
    # viewer control is read here: a snapshot saved with either ON must reproduce the
    # line that was on screen, not the raw ascent. Both off is a no-op --
    # refine_connections returns the input records by identity when nothing changes.
    # A v6 snapshot recorded the FIDELITY smoother, so coarse_ridge_curves is absent
    # from it, defaults to off, and the replay runs the smoother it shows.
    if bool(_p(ctx, "refine_ridge_paths")) or bool(_p(ctx, "coarse_ridge_curves")):
        ctx.connections = refine_connections(
            ctx.connections,
            ctx.sphere_radius,
            ctx.delta_theta_rad,
            center=ctx.center,
            regions=regions,
            coarse=bool(_p(ctx, "coarse_ridge_curves")),
        )
    ctx.connection_stats = summarize_connections(ctx.connections)
    stats = dict(ctx.connection_stats)
    # Per-connector records for --out: the aggregate above is the whole story the
    # metrics diff needs, and the footer below is the whole story the card shows.
    ctx.connection_details = connection_ridge_details(ctx.connections)
    stats.update(summarize_ridge_postpass(ctx.connections))
    return stats


def stage_minimum_basins(ctx: RunContext) -> dict[str, Any]:
    """Compute minimum basin boundaries from the connection topology.

    Uses the verified connections to detect cycles that enclose each minimum CP.
    Each basin boundary is an ordered cycle of arcs (MaxConnection objects) that
    forms a closed path enclosing a minimum in the connection graph.

    Returns:
        Summary of the basin detection, including count of detected basins.
    """
    if not ctx.connections:
        ctx.minimum_basins = {}
        ctx.connection_graph = {}
        return {"n_minimum_basins": 0}

    try:
        # Build the graph for diagnostics
        ctx.connection_graph = build_connection_graph(ctx.connections)
        ctx.minimum_basins = find_minimum_basins(
            ctx.connections,
            ctx.basin_centers,
        )
    except (ValueError, Exception) as e:
        # Topology error or malformed data; use empty dict
        ctx.minimum_basins = {}
        ctx.connection_graph = {}
        return {
            "n_minimum_basins": 0,
            "error": str(e),
        }

    # Build detailed diagnostics
    summary: dict[str, Any] = {"n_minimum_basins": len(ctx.minimum_basins)}
    if ctx.minimum_basins:
        summary["minima_with_boundaries"] = [
            {
                "minimum_id": basin.minimum_id,
                "n_boundary_arcs": len(basin.boundary_arcs),
                "boundary_saddles": list(
                    {arc.saddle_id for arc in basin.boundary_arcs}
                ),
                "boundary_maxima": list(
                    {arc.max_id for arc in basin.boundary_arcs if arc.max_id}
                ),
            }
            for basin in ctx.minimum_basins.values()
        ]
    if ctx.connection_graph:
        summary["n_saddles_in_graph"] = len(ctx.connection_graph)
        summary["n_saddle_to_max_arcs"] = sum(
            len(conns) for conns in ctx.connection_graph.values()
        )
    return summary


def stage_probe(ctx: RunContext) -> dict[str, Any]:
    """Step 3 "Step 3 CP Probing": measure the ridge beside each traced ascent.

    The probe is the one control on the card that changes NO geometry, so a replay
    that never runs it reports nothing about it. This stage mirrors the viewer's
    ``_emit_max_arcs`` probe block: it measures the ASCENT (the traced table, not
    the smoothed line drawn beside it), it runs on ``method == "gradient"``
    connectors only, and it publishes the same counters under the viewer's own
    ``state.ridge_probe_stats`` names.

    The scope is copied deliberately. The viewer probes the connectors ON SCREEN --
    ``_emit_max_arcs`` selects them by the current basin or saddle selection, and
    gates the whole block on ``show_max_saddle_arcs`` -- so the saved counters are a
    property of the selection, not of the run. A replay that probed every gradient
    connector would report a bigger number than the screen ever showed and call the
    snapshot unreproducible. ``--probe-all`` opts into the full-field superset when
    the point is comparing two parameter sets over the whole run.

    Args:
        ctx: Context after the connect stage.

    Returns:
        The probe stats, with ``on`` False and zero counters when the control is
        off or nothing was on screen, so the stage always reports the same keys.
    """
    basin_id = None if ctx.probe_all else ctx.probe_basin_id
    saddle_id = None if ctx.probe_all else ctx.probe_saddle_id
    # The switch, the overlay's own visibility, and a selection: all three, exactly
    # as the drawer computes them. Nothing is on screen without all three, so a
    # replay that ran the probe with the arcs hidden would measure a scene nobody
    # saw. ``--probe-all`` drops the latter two -- it cannot drop the switch, since
    # a probe the operator never turned on was never on screen either.
    on = bool(_p(ctx, "probe_ridge_points")) and (
        ctx.probe_all
        or (
            bool(_p(ctx, "show_max_saddle_arcs"))
            and (basin_id is not None or saddle_id is not None)
        )
    )
    if not on:
        ctx.ridge_probe_stats = make_ridge_probe_stats([], on=False)
        ctx.probe_details = []
        return dict(ctx.ridge_probe_stats)

    # Two scopes, two sources, and the difference is the whole point. On screen, the
    # rows are the drawn arcs -- basin-culled and fusion-culled -- selected by the
    # saddle and basin under the cursor, exactly as ``_emit_max_arcs`` walks them;
    # the ascent is then looked up by ``(saddle_id, branch_index)``, which is the
    # viewer's ``traced_by_key`` and the reason this stage reads traced_connections
    # at all. ``--probe-all`` has nothing to do with the screen, so it goes straight
    # to the run's traced table.
    traced_by_key = {(c.saddle_id, c.branch_index): c for c in ctx.traced_connections}
    if ctx.probe_all:
        scope_rows = list(ctx.traced_connections)
    else:
        scope_rows = [
            conn
            for conn in ctx.connections
            if (basin_id is None or conn.max_id == basin_id)
            and (saddle_id is None or conn.saddle_id == saddle_id)
        ]
    probes: list[dict[str, Any]] = []
    for conn in scope_rows:
        # A geodesic ray is a great circle, not an ascent: the ridge beside it is
        # not a property of the field, so the viewer leaves it unprobed.
        if conn.method != "gradient" or len(conn.arc_points) < 2:
            continue
        target = traced_by_key.get((conn.saddle_id, conn.branch_index), conn)
        probe = probe_gradient_ridge(
            target.arc_points,
            ctx.pts,
            ctx.f_vals,
            ctx.sphere_radius,
            ctx.delta_theta_rad,
            center=ctx.center,
            find_k_nearest=ctx.locator,
        )
        probes.append(
            {
                "id": f"{conn.saddle_id}#{conn.branch_index}",
                "saddle_id": conn.saddle_id,
                "branch_index": conn.branch_index,
                "max_id": conn.max_id,
                "n_knots": len(probe.knot_positions_world),
                "n_probe": int(probe.n_accepted),
                "probe_refused": len(probe.samples) - int(probe.n_accepted),
                "probe_offset_pitch": float(probe.median_offset_pitch),
                "probe_max_offset_pitch": float(probe.max_offset_pitch),
            }
        )
    ctx.probe_details = probes
    # The same aggregator the viewer runs, so the two sides cannot drift by a sum.
    ctx.ridge_probe_stats = make_ridge_probe_stats(
        probes,
        on=True,
        basin_id=basin_id,
        saddle_id=saddle_id,
    )
    return dict(ctx.ridge_probe_stats)


STAGES: list[tuple[str, Callable[[RunContext], dict[str, Any]]]] = [
    ("geometry", stage_geometry),
    ("morse", stage_morse),
    ("clusters", stage_clusters),
    ("bridge", stage_bridge),
    ("units", stage_units),
    ("catastrophes", stage_catastrophes),
    ("ports", stage_ports),
    ("reduction", stage_reduction),
    ("basins", stage_basins),
    ("launch", stage_launch),
    ("connect", stage_connect),
    ("minimum_basins", stage_minimum_basins),
    ("probe", stage_probe),
]


def run_stages(
    ctx: RunContext, stop_after: str | None = None
) -> list[tuple[str, dict[str, Any]]]:
    """Run the pipeline in viewer order, stopping after ``stop_after``.

    Args:
        ctx: Populated run context.
        stop_after: Last stage to run (inclusive), or None for the full run.

    Returns:
        List of (stage name, summary dict) in execution order.

    Raises:
        ValueError: If stop_after is not a known stage name.
    """
    names = [name for name, _ in STAGES]
    if stop_after is not None and stop_after not in names:
        raise ValueError(f"Unknown stage {stop_after!r}; choose from {names}")
    results: list[tuple[str, dict[str, Any]]] = []
    for name, fn in STAGES:
        results.append((name, fn(ctx)))
        if name == stop_after:
            break
    return results


@dataclass(frozen=True)
class MetricDiff:
    """One leaf comparison between a saved metric and its recomputation.

    Attributes:
        path: Dotted path, e.g. ``morse_counts.saddles``.
        saved: Value recorded in the params JSON.
        computed: Value recomputed by this run.
        matches: True when equal (floats within tolerance).
    """

    path: str
    saved: Any
    computed: Any
    matches: bool


def diff_leaf(a: Any, b: Any, rtol: float) -> bool:
    """Compare two scalar leaves, with float tolerance and bool strictness.

    Args:
        a: Saved value.
        b: Recomputed value.
        rtol: Relative tolerance for float comparison.

    Returns:
        True when the two values agree.
    """
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(float(a), float(b), rel_tol=rtol, abs_tol=1e-12)
    return bool(a == b)


def diff_metrics(
    saved: dict[str, Any], computed: dict[str, Any], *, rtol: float = 1e-6
) -> list[MetricDiff]:
    """Recursively compare saved derived metrics against recomputed ones.

    Walks the saved structure, so metrics this run did not reach (an early
    --only stop, signalled by a missing top-level key) are skipped. Leaf
    lists (e.g. geom_info.center) compare element-wise.

    Args:
        saved: ``derived_metrics`` section from the params JSON.
        computed: Recomputed metric dicts keyed the same way.
        rtol: Relative tolerance for float comparison.

    Returns:
        One MetricDiff per leaf found in both structures.
    """
    out: list[MetricDiff] = []

    def walk(prefix: str, a: Any, b: Any) -> None:
        if isinstance(a, dict):
            if not isinstance(b, dict):
                out.append(MetricDiff(prefix, a, b, False))
                return
            for key in a:
                walk(f"{prefix}.{key}" if prefix else key, a[key], b.get(key))
            return
        if isinstance(a, list):
            same = isinstance(b, list) and len(a) == len(b)
            eq = same and all(diff_leaf(x, y, rtol) for x, y in zip(a, b, strict=True))
            out.append(MetricDiff(prefix, a, b, eq))
            return
        out.append(MetricDiff(prefix, a, b, diff_leaf(a, b, rtol)))

    for key, value in saved.items():
        if key not in computed:
            continue
        walk(key, value, computed[key])
    return out


def build_context(
    params: dict[str, Any],
    *,
    plt_path: str | None,
    atom_num: int | None,
    field_name: str | None,
    selection: dict[str, Any] | None = None,
) -> RunContext:
    """Load the mesh described by the params file and prepare the pipeline.

    Args:
        params: Flat dict from ``load_params_from_dict``.
        plt_path: Override for the recorded .plt path.
        atom_num: Override for the recorded atom number.
        field_name: Override for the recorded condensed field.
        selection: The snapshot's ``selection`` section. It is the arc overlay's
            scope, and the ridge probe measures the arcs ON SCREEN, so the probe
            stage cannot reproduce a saved counter without it. A pre-v8 snapshot
            records the basin end only; a pre-v7 one records no saddle end at all.

    Returns:
        A RunContext ready for ``run_stages``.

    Raises:
        FileNotFoundError: If no .plt path resolves to an existing file.
        KeyError: If the requested field is absent from the mesh.
    """
    context = params.get("context", {})
    resolved_plt = plt_path or context.get("plt_path")
    if not resolved_plt or not os.path.exists(resolved_plt):
        raise FileNotFoundError(
            f"PLT file not found: {resolved_plt!r}. Pass --plt with a local path."
        )
    resolved_atom = int(
        atom_num if atom_num is not None else context.get("atom_number", 1)
    )

    sphere_poly, field_items, atom_info = extract_atom_sphere_mesh(
        resolved_plt, target_atom_num=resolved_atom
    )
    center: tuple[float, float, float] = tuple(atom_info["center"])  # type: ignore[assignment]

    # ``vtk_to_numpy`` is typed to return an array of *some* dtype -- the VTK
    # signature admits uint8 for a point field, which is not assignable to the
    # float64/int32 the pipeline is built on. np.asarray with an explicit dtype is
    # the honest narrowing: it is a no-op copy-free pass-through when the array
    # already matches (it always does for a sphere mesh and a double field), and it
    # copies rather than lying when it does not.
    pts: npt.NDArray[np.float64] = np.asarray(
        numpy_support.vtk_to_numpy(sphere_poly.GetPoints().GetData()),
        dtype=np.float64,
    )
    n_cells = sphere_poly.GetNumberOfPolys()
    cell_arr = numpy_support.vtk_to_numpy(sphere_poly.GetPolys().GetData())
    triangles: npt.NDArray[np.int32] = np.asarray(
        cell_arr.reshape((n_cells, 4))[:, 1:4], dtype=np.int32
    )

    geom = analyze_mesh_geometry(
        pts, triangles, sphere_radius=float(atom_info["sphere_radius"])
    )
    geom_info = {
        "num_vertices": geom["num_vertices"],
        "num_triangles": geom["num_triangles"],
        "num_edges": geom["num_edges"],
        "sphere_radius": float(geom["sphere_radius"]),
        "avg_edge_length": float(geom["avg_edge_length"]),
        "mesh_pitch_deg": float(geom["mesh_pitch_deg"]),
        "center": list(center),
        "center_offset": float(atom_info["center_offset"]),
        "radius_source": str(atom_info["radius_source"]),
    }

    resolved_field = field_name or params.get("selected_field")
    if resolved_field is None:
        resolved_field = field_items[0]["value"]
    arr = sphere_poly.GetPointData().GetArray(resolved_field)
    if arr is None:
        available = [f["value"] for f in field_items]
        raise KeyError(f"Field {resolved_field!r} absent; available: {available}")
    f_vals: npt.NDArray[np.float64] = np.asarray(
        numpy_support.vtk_to_numpy(arr), dtype=np.float64
    )

    return RunContext(
        params=params,
        plt_path=os.path.abspath(resolved_plt),
        atom_num=resolved_atom,
        field_name=resolved_field,
        center=center,
        pts=pts,
        triangles=triangles,
        f_vals=f_vals,
        sphere_radius=geom_info["sphere_radius"],
        delta_theta_rad=geom["mesh_pitch_rad"],
        geom_info=geom_info,
        locator=make_kdtree_nearest_finder(sphere_poly),
        probe_basin_id=_scope_id(selection, "selected_basin_center_id"),
        probe_saddle_id=_scope_id(selection, "selected_saddle_id"),
    )


def _scope_id(selection: dict[str, Any] | None, key: str) -> str | None:
    """Read one selection id, tolerating a snapshot that predates the key.

    Args:
        selection: The snapshot's ``selection`` section, or None.
        key: Which id to read.

    Returns:
        The id, or None when the section is absent, the key is missing, or the
        selection was empty at save time (which is what the export writes then).
    """
    if not selection:
        return None
    value = selection.get(key)
    return str(value) if isinstance(value, str) and value else None


def apply_overrides(params: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    """Apply ``key=value`` control overrides with the widget's type.

    Args:
        params: Flat control dict to update in place.
        overrides: Strings like ``k_pitch=3.0`` or ``reduce_catastrophes=true``.

    Returns:
        The updated params dict.

    Raises:
        KeyError: If the key is not a known control.
        ValueError: If the value cannot be parsed for that control.
    """
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override {item!r} must be key=value")
        key, raw = item.split("=", 1)
        if key in PARAM_BOOL_KEYS:
            if raw.strip().lower() not in ("true", "false"):
                raise ValueError(f"Override {key}=... must be true/false, got {raw!r}")
            params[key] = raw.strip().lower() == "true"
        elif key in PARAM_FLOAT_KEYS:
            try:
                params[key] = float(raw)
            except ValueError as exc:
                raise ValueError(f"Override {key}=... must be a number") from exc
        elif key in PARAM_STRING_KEYS or key in PARAM_CONTEXT_STRING_KEYS:
            params[key] = raw
        else:
            raise KeyError(f"Unknown control {key!r} in override")
    return params


def computed_metrics(ctx: RunContext) -> dict[str, Any]:
    """Collect the recomputed metric dicts under the viewer's state key names.

    Args:
        ctx: Finished run context.

    Returns:
        Dict keyed like the export's derived_metrics section, containing only
        the stages this run actually reached.
    """
    metrics: dict[str, Any] = {"geom_info": ctx.geom_info}
    if ctx.morse_counts:
        metrics["morse_counts"] = ctx.morse_counts
    if ctx.display_counts:
        metrics["display_counts"] = ctx.display_counts
    if ctx.reduction_stats:
        metrics["reduction_stats"] = ctx.reduction_stats
    if ctx.launch_stats:
        metrics["launch_stats"] = ctx.launch_stats
    if ctx.connection_stats:
        metrics["connection_stats"] = ctx.connection_stats
    # Gated on non-empty, not on the toggle: stage_probe always writes a dict, and
    # an OLD snapshot (<= v7) has no probe block at all. diff_metrics walks the
    # saved side, so a missing saved key is skipped -- the run reports "no
    # comparable metric" for a file that never measured, instead of a false drift.
    if ctx.ridge_probe_stats:
        metrics["ridge_probe_stats"] = ctx.ridge_probe_stats
    return metrics


def dump_results(ctx: RunContext, path: str) -> None:
    """Write per-entity results to a JSON file for programmatic diffing.

    Args:
        ctx: Finished run context.
        path: Destination path.
    """
    payload = {
        "plt_path": ctx.plt_path,
        "atom_number": ctx.atom_num,
        "field": ctx.field_name,
        "params": {k: v for k, v in ctx.params.items() if not isinstance(v, dict)},
        "geom_info": ctx.geom_info,
        "morse_counts": ctx.morse_counts,
        "display_counts": ctx.display_counts,
        "reduction_stats": ctx.reduction_stats,
        "launch_stats": ctx.launch_stats,
        "clusters": ctx.clusters_ui,
        "bridge_links": [
            {
                "cluster_a": link.cluster_a,
                "cluster_b": link.cluster_b,
                "gap_deg": math.degrees(link.gap_rad),
                "axis_angle_deg": link.axis_angle_deg,
                "is_linked": link.is_linked,
                "reason": link.reason,
            }
            for link in ctx.bridge_links
        ],
        "bridge_gate": ctx.bridge_gate,
        "catastrophes": ctx.catastrophes_ui,
        "effective_cps": ctx.effective_cps,
        "basin_centers": [
            {
                "id": b.center_id,
                "polarity": b.polarity,
                "position": list(b.position),
                "value": b.value,
                "is_effective": b.is_effective,
                "origin_id": b.origin_id,
                "net_index": b.net_index,
            }
            for b in ctx.basin_centers
        ],
        "launch_seeds": [
            {
                "saddle_id": s.saddle_id,
                "vertex_id": s.vertex_id,
                "fold_order": s.fold_order,
                "index_agrees": s.index_agrees,
                "n_ascent": len(s.ascent_branches),
                "n_descent": len(s.descent_branches),
            }
            for s in ctx.seeds
        ],
        "connection_stats": ctx.connection_stats,
        "max_connections": [
            {
                "saddle_id": c.saddle_id,
                "branch_index": c.branch_index,
                "max_id": c.max_id,
                "status": c.status,
                "direction_mode": c.direction_mode,
                "cross_track_deg": c.cross_track_deg,
                "along_track_deg": c.along_track_deg,
                "separation_deg": c.separation_deg,
                "mutual": c.mutual,
                "verified": c.verified,
                "saddle_value": c.saddle_value,
                "max_value": c.max_value,
                "blocker_id": c.blocker_id,
                "blocker_polarity": c.blocker_polarity,
                "method": c.method,
                "failure_gate": c.failure_gate,
                "stall_point": list(c.stall_point) if c.stall_point else None,
                "transited_ids": list(c.transited_ids),
                "completed_leg_deg": c.completed_leg_deg,
                # The Step 3 ridge switches, per connector. Without these the dump
                # cannot show the card's output at all: "refined" is the fidelity
                # smoother's mark, "n_nodes" the coarse curve's, and the turn totals
                # are the smoothing cost.
                "refined": bool(c.refined),
                "refine_max_shift_deg": float(c.refine_max_shift_deg),
                "n_corners": int(c.n_corners),
                "n_runs": int(c.n_runs),
                "n_nodes": int(c.n_nodes),
                "total_turn_before_deg": float(c.total_turn_before_deg),
                "total_turn_after_deg": float(c.total_turn_after_deg),
            }
            for c in ctx.connections
        ],
        "connection_ridge_details": ctx.connection_details,
        "ridge_probe_stats": ctx.ridge_probe_stats,
        "probe_details": ctx.probe_details,
        "connection_pool": [
            {
                "cp_id": t.cp_id,
                "polarity": t.polarity,
                "position": [float(x) for x in t.position],
                "value": t.value,
            }
            for t in ctx.connection_pool
        ],
    }
    # Add minimum basins to payload for debugging
    if ctx.minimum_basins:
        payload["minimum_basins"] = {
            min_id: {
                "minimum_id": basin.minimum_id,
                "n_boundary_arcs": len(basin.boundary_arcs),
                "boundary_arcs": [
                    {
                        "saddle_id": arc.saddle_id,
                        "branch_index": arc.branch_index,
                        "max_id": arc.max_id,
                    }
                    for arc in basin.boundary_arcs
                ],
            }
            for min_id, basin in ctx.minimum_basins.items()
        }
    # Add connection graph for diagnostics
    if ctx.connection_graph:
        payload["connection_graph"] = {
            saddle_id: [
                {
                    "max_id": conn.max_id,
                    "branch_index": conn.branch_index,
                    "status": conn.status,
                    "verified": conn.verified,
                }
                for conn in conns
            ]
            for saddle_id, conns in ctx.connection_graph.items()
        }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write("\n")


def main() -> int:
    """CLI entrypoint. Returns the process exit code.

    Returns:
        0 when the recomputed metrics match the saved observation (or no
        metrics were comparable), 1 when any leaf drifts.
    """
    parser = argparse.ArgumentParser(
        prog="diagnose_params.py",
        description='Re-run sphere_viewer analysis stages from a saved "Save '
        'Parameters" JSON.',
    )
    parser.add_argument("params_json", help="Path to *_params.json")
    parser.add_argument(
        "--only",
        choices=[name for name, _ in STAGES],
        default=None,
        help="Stop after this stage (inclusive)",
    )
    parser.add_argument("--plt", default=None, help="Override the recorded .plt path")
    parser.add_argument("-a", "--atom", type=int, default=None, help="Override atom")
    parser.add_argument("-f", "--field", default=None, help="Override condensed field")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override one control, e.g. --set k_pitch=3.0 (repeatable)",
    )
    parser.add_argument("--out", default=None, help="Write detailed results to JSON")
    parser.add_argument(
        "--probe-all",
        action="store_true",
        help="Ridge-probe every gradient connector, ignoring the snapshot's arc "
        "selection and overlay visibility. A superset of what the screen showed, "
        "for comparing parameter sets over the whole run; it also makes the probe "
        "counters incomparable with a saved file.",
    )
    parser.add_argument(
        "--quiet", action="store_true", help="Print only the reproduce-check summary"
    )
    args = parser.parse_args()

    with open(args.params_json, encoding="utf-8") as fh:
        payload = json.load(fh)
    params = load_params_from_dict(payload)
    apply_overrides(params, args.overrides)

    ctx = build_context(
        params,
        plt_path=args.plt,
        atom_num=args.atom,
        field_name=args.field,
        selection=payload.get("selection"),
    )
    ctx.probe_all = bool(args.probe_all)
    if not args.quiet:
        print(f"[diag] {ctx.plt_path}  atom={ctx.atom_num}  field={ctx.field_name}")
        print(
            "[diag] stages: "
            + ", ".join(name for name, _ in STAGES)
            + (f"  (stop after {args.only})" if args.only else "")
        )

    results = run_stages(ctx, stop_after=args.only)
    if not args.quiet:
        for name, summary in results:
            print(f"\n=== {name} ===")
            for key, value in summary.items():
                print(f"  {key}: {value}")

    saved_metrics = payload.get("derived_metrics", {})
    diffs = diff_metrics(saved_metrics, computed_metrics(ctx))
    drift = [d for d in diffs if not d.matches]
    print(f"\n=== reproduce check: {len(diffs) - len(drift)}/{len(diffs)} match ===")
    for d in drift:
        print(f"  DRIFT  {d.path}: saved={d.saved!r} computed={d.computed!r}")
    if not drift and diffs:
        print("  PASS - recomputed metrics match the saved observation")
    elif not diffs:
        print("  (no comparable metrics in the saved file)")

    if args.out:
        dump_results(ctx, args.out)
        print(f"\n[diag] results written to: {os.path.abspath(args.out)}")

    return 1 if drift else 0


if __name__ == "__main__":
    sys.exit(main())
