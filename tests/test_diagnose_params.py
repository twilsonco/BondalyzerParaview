"""Unit tests for the diagnose_params headless analysis runner.

Covers the pure, UI-free pieces of ``gba_topology2/diagnose_params.py``:
1. ``diff_leaf`` / ``diff_metrics`` -- the reproduce-check comparison,
   including float tolerance, bool strictness, list leaves, and the
   skip-missing-keys rule that makes --only stops comparable.
2. ``apply_overrides`` -- typed key=value overrides for sweeps.
3. ``run_stages`` -- stage ordering, --only early stop, and the morse-first
   guard.
4. The Step 3 max-connections card -- ``connection_ridge_details`` /
   ``summarize_ridge_postpass`` (the two ridge switches' per-connector output) and
   ``stage_probe`` (CP probing, which changes no geometry and so is invisible to
   every other metric).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.vtkCommonDataModel import vtkPolyData

from gba_topology2 import diagnose_params
from gba_topology2.diagnose_params import (
    STAGES,
    MaxConnection,
    MetricDiff,
    RunContext,
    apply_overrides,
    connection_ridge_details,
    diff_leaf,
    diff_metrics,
    dump_results,
    run_stages,
    stage_connect,
    stage_probe,
    summarize_ridge_postpass,
)
from gba_topology2.src.point_locator import make_kdtree_nearest_finder


def _poly_from_points(pts: np.ndarray) -> vtkPolyData:
    """Wrap an (N, 3) array as a point-only vtkPolyData for the locator."""
    vp = vtkPoints()
    for p in pts:
        vp.InsertNextPoint(float(p[0]), float(p[1]), float(p[2]))
    poly = vtkPolyData()
    poly.SetPoints(vp)
    return poly


def _make_ctx(**overrides: Any) -> RunContext:
    """Build a minimal RunContext on a tiny octahedron sphere mesh.

    Args:
        overrides: params keys to override on top of the defaults.

    Returns:
        A RunContext whose morse stage can run on the octahedron with an
        f_vals field chosen to yield a valid chi = 2 Morse function.
    """
    pts = np.array(
        [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ],
        dtype=np.float64,
    )
    triangles = np.array(
        [
            [0, 2, 4],
            [2, 1, 4],
            [1, 3, 4],
            [3, 0, 4],
            [2, 0, 5],
            [1, 2, 5],
            [3, 1, 5],
            [0, 3, 5],
        ],
        dtype=np.int32,
    )
    params: dict[str, Any] = {
        "k_pitch": 2.0,
        "aggregate_strings": False,
        "k_bridge": 3.0,
        "cat_barrier_tol": 0.005,
        "cat_angular_tol_mult": 4.0,
        "reduce_catastrophes": False,
        "reduce_multi_clusters": False,
        "nudge_cp_positions": False,
        "perfect_port_angles": False,
        "nudge_max_pitch": 1.5,
        "refine_ridge_paths": False,
        "selected_field": "V (condensed)",
    }
    params.update(overrides)
    # Quadrupole field f = x^2 - y^2: the detector's own reference case yields
    # 2 maxima (v0, v1), 2 minima (v2, v3) and 2 simple saddles (v4, v5), so
    # chi = 2 and the saddle-consuming stages have exact expected counts.
    f_vals = pts[:, 0] ** 2 - pts[:, 1] ** 2
    return RunContext(
        params=params,
        plt_path="synthetic.plt",
        atom_num=1,
        field_name="V (condensed)",
        center=(0.0, 0.0, 0.0),
        pts=pts,
        triangles=triangles,
        f_vals=f_vals,
        sphere_radius=1.0,
        delta_theta_rad=math.pi / 2.0,
        geom_info={"num_vertices": 6, "mesh_pitch_deg": 90.0},
        locator=make_kdtree_nearest_finder(_poly_from_points(pts)),
    )


class TestDiffLeaf:
    """Scalar leaf comparison semantics."""

    def test_bools_compare_strictly(self) -> None:
        assert diff_leaf(True, True, 1e-6)
        assert not diff_leaf(True, 1, 1e-6)
        assert not diff_leaf(1, True, 1e-6)

    def test_ints_and_floats_use_tolerance(self) -> None:
        assert diff_leaf(2, 2.0, 1e-6)
        assert diff_leaf(1.0, 1.0 + 1e-12, 1e-6)
        assert not diff_leaf(2, 3, 1e-6)

    def test_strings_compare_by_equality(self) -> None:
        assert diff_leaf("metadata", "metadata", 1e-6)
        assert not diff_leaf("metadata", "fitted", 1e-6)

    def test_mixed_types_are_not_equal(self) -> None:
        assert not diff_leaf("2", 2, 1e-6)


class TestDiffMetrics:
    """Recursive metric comparison against the saved derived_metrics."""

    def test_matching_nested_dicts_all_pass(self) -> None:
        saved = {"morse_counts": {"minima": 6, "euler": 2, "valid": True}}
        computed = {"morse_counts": {"minima": 6, "euler": 2, "valid": True}}

        diffs = diff_metrics(saved, computed)

        assert len(diffs) == 3
        assert all(d.matches for d in diffs)

    def test_drift_is_isolated_to_the_drifting_leaf(self) -> None:
        saved = {"morse_counts": {"minima": 6, "maxima": 8}}
        computed = {"morse_counts": {"minima": 6, "maxima": 7}}

        diffs = diff_metrics(saved, computed)
        drift = [d for d in diffs if not d.matches]

        assert [d.path for d in drift] == ["morse_counts.maxima"]
        assert drift[0].saved == 8
        assert drift[0].computed == 7

    def test_missing_computed_section_is_skipped(self) -> None:
        """An early --only stop leaves later sections uncomputed: skip them."""
        saved = {
            "morse_counts": {"minima": 6},
            "launch_stats": {"n_saddles_seeded": 12},
        }
        computed = {"morse_counts": {"minima": 6}}

        diffs = diff_metrics(saved, computed)

        assert [d.path for d in diffs] == ["morse_counts.minima"]
        assert diffs[0].matches

    def test_list_leaf_compares_elementwise(self) -> None:
        saved = {"geom_info": {"center": [1.0, 2.0, 3.0]}}
        same = {"geom_info": {"center": [1.0, 2.0, 3.0]}}
        moved = {"geom_info": {"center": [1.0, 2.0, -3.0]}}
        short = {"geom_info": {"center": [1.0, 2.0]}}

        assert diff_metrics(saved, same)[0].matches
        assert not diff_metrics(saved, moved)[0].matches
        assert not diff_metrics(saved, short)[0].matches

    def test_dict_vs_scalar_shape_mismatch_is_drift(self) -> None:
        saved = {"morse_counts": {"minima": 6}}
        computed = {"morse_counts": 6}

        diffs = diff_metrics(saved, computed)

        assert len(diffs) == 1
        assert not diffs[0].matches

    def test_empty_saved_yields_no_diffs(self) -> None:
        assert diff_metrics({}, {"geom_info": {"num_vertices": 6}}) == []

    def test_diff_is_immutable(self) -> None:
        """MetricDiff is frozen: assignment is rejected, not silently accepted."""
        import dataclasses

        d = MetricDiff(path="a", saved=1, computed=2, matches=False)
        with pytest.raises(dataclasses.FrozenInstanceError):
            d.path = "b"  # type: ignore[misc]

    def test_diff_fields_are_ordered_for_reporting(self) -> None:
        """Reports index by dotted path, so the field order is contractual."""
        import dataclasses

        names = [f.name for f in dataclasses.fields(MetricDiff)]
        assert names == ["path", "saved", "computed", "matches"]


class TestApplyOverrides:
    """Typed key=value overrides used for parameter sweeps."""

    def test_float_override(self) -> None:
        params: dict[str, Any] = {"k_pitch": 2.0}
        apply_overrides(params, ["k_pitch=3.5"])
        assert params["k_pitch"] == 3.5

    def test_bool_override_accepts_true_false_case_insensitive(self) -> None:
        params: dict[str, Any] = {"reduce_catastrophes": False}
        apply_overrides(params, ["reduce_catastrophes=TRUE"])
        assert params["reduce_catastrophes"] is True

    def test_string_override_for_selected_field(self) -> None:
        params: dict[str, Any] = {"selected_field": "V (condensed)"}
        apply_overrides(params, ["selected_field=ELF (condensed)"])
        assert params["selected_field"] == "ELF (condensed)"

    def test_bool_override_rejects_numbers(self) -> None:
        params: dict[str, Any] = {"show_rings": True}
        with pytest.raises(ValueError, match="true/false"):
            apply_overrides(params, ["show_rings=1"])

    def test_float_override_rejects_words(self) -> None:
        params: dict[str, Any] = {"k_pitch": 2.0}
        with pytest.raises(ValueError, match="number"):
            apply_overrides(params, ["k_pitch=wide"])

    def test_unknown_key_raises_keyerror(self) -> None:
        with pytest.raises(KeyError, match="not_a_control"):
            apply_overrides({}, ["not_a_control=1"])

    def test_missing_equals_raises_valueerror(self) -> None:
        with pytest.raises(ValueError, match="key=value"):
            apply_overrides({}, ["k_pitch"])


class TestRunStages:
    """Stage ordering, early stop, and the morse-first guard."""

    def test_stage_order_matches_viewer_pipeline(self) -> None:
        names = [name for name, _ in STAGES]
        assert names == [
            "geometry",
            "morse",
            "clusters",
            "bridge",
            "units",
            "catastrophes",
            "ports",
            "reduction",
            "basins",
            "launch",
            "connect",
            # Minimum basins are computed from the connections topology.
            "minimum_basins",
            # The probe runs last: it measures the traced ascent, which only exists
            # once the connect stage has produced it. In the viewer the same order
            # holds -- the probe lives in _emit_max_arcs, after the post-pass.
            "probe",
        ]

    def test_geometry_only_stop_runs_single_stage(self) -> None:
        ctx = _make_ctx()
        results = run_stages(ctx, stop_after="geometry")

        assert [name for name, _ in results] == ["geometry"]
        assert results[0][1]["num_vertices"] == 6
        assert ctx.morse is None

    def test_morse_stage_populates_counts(self) -> None:
        ctx = _make_ctx()
        results = dict(run_stages(ctx, stop_after="morse"))

        assert ctx.morse is not None
        assert results["morse"] == {
            "minima": 2,
            "maxima": 2,
            "saddles": 2,
            "euler": 2,
            "valid": True,
        }

    def test_full_run_on_octahedron_preserves_euler(self) -> None:
        """Quadrupole octahedron: 2 Max + 2 Min + 2 Sad, chi = 2 end to end.

        k_pitch=2.0 on a 90-degree mesh pitch merges all six CPs into one
        unit, so with reduction off every glyph stays visible and the
        displayed index sum must still equal 2.
        """
        ctx = _make_ctx()
        results = run_stages(ctx)

        by_name = dict(results)
        assert by_name["morse"]["euler"] == 2
        assert by_name["clusters"]["index_sum"] == 2
        assert by_name["units"]["n_units"] == 1
        assert by_name["reduction"]["display_counts"]["euler"] == 2
        assert by_name["basins"]["n_basin_centers"] == 4
        assert by_name["launch"]["n_saddles_seeded"] == 2

    def test_connect_stage_populates_connection_table(self) -> None:
        """The connect stage runs last and fills the Step 3 context fields.

        On the quadrupole octahedron the two saddles (poles) each launch ascent
        branches toward the equatorial maxima, so a full run must produce a
        non-empty connection table whose counters agree with the records.
        """
        ctx = _make_ctx()
        by_name = dict(run_stages(ctx))

        assert "connect" in by_name
        assert len(ctx.connections) == by_name["connect"]["n_branches"]
        # The stage summary is the viewer's aggregate stats PLUS the card's ridge
        # footer. ``connection_stats`` itself stays exactly the viewer's state key,
        # so the reproduce check compares like for like.
        assert by_name["connect"] == {
            **ctx.connection_stats,
            **summarize_ridge_postpass(ctx.connections),
        }
        # Every record is one of the three verdicts, and the summary counts agree.
        assert all(
            c.status in ("connected", "occluded", "no_target") for c in ctx.connections
        )
        n_connected = sum(1 for c in ctx.connections if c.status == "connected")
        assert n_connected == by_name["connect"]["n_connected"]
        # The pool carries the visible saddles, not just the extrema: that is what
        # lets a branch be occluded. On the octahedron the 2 saddles are in it.
        assert any(t.polarity == "saddle" for t in ctx.connection_pool)
        # Arc points, when present, sit on the unit sphere (center is origin).
        for c in ctx.connections:
            for p in c.arc_points:
                assert float(np.linalg.norm(p)) == pytest.approx(1.0, rel=1e-9)

    def test_connect_requires_morse_first(self) -> None:
        """The connect stage needs the morse saddles to build the occlusion pool."""
        ctx = _make_ctx()
        # No prior stages: the pool is built from the Morse saddles, so connect
        # fails loudly rather than silently scoring against an empty pool.
        with pytest.raises(RuntimeError, match="morse"):
            stage_connect(ctx)

    def test_unknown_stop_stage_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown stage"):
            run_stages(_make_ctx(), stop_after="teleport")

    def test_reduction_requires_morse_stage(self) -> None:
        """Calling a consumer stage directly without morse fails loudly."""
        from gba_topology2.diagnose_params import stage_reduction

        ctx = _make_ctx()
        with pytest.raises(RuntimeError, match="morse"):
            stage_reduction(ctx)

    def test_display_counts_include_effective_cps(self) -> None:
        """Displayed counts must include the effective CPs, not just raw ones.

        Regression guard: the viewer's glyph bookkeeping adds one glyph per
        effective CP (E-MAX -> maxima, E-SAD -> saddles) on top of the
        visible raw CPs. Counting only the raw CPs under-reports both the
        glyph counts and the index sum, and the reproduce check then reports
        drift that is an artefact of the runner rather than of the data.
        """
        from gba_topology2.diagnose_params import stage_reduction  # noqa: F401

        ctx = _make_ctx(reduce_multi_clusters=True)
        run_stages(ctx, stop_after="reduction")

        n_eff_max = sum(1 for e in ctx.effective_cps if e["type"] == "E-MAX")
        n_eff_min = sum(1 for e in ctx.effective_cps if e["type"] == "E-MIN")
        n_eff_sad = sum(1 for e in ctx.effective_cps if e["type"] == "E-SAD")

        raw_max = sum(
            1
            for m in ctx.require_morse().maxima
            if m.vertex_id not in ctx.hidden_vertex_ids
        )
        raw_min = sum(
            1
            for m in ctx.require_morse().minima
            if m.vertex_id not in ctx.hidden_vertex_ids
        )
        raw_sad = sum(
            1
            for m in ctx.require_morse().saddles
            if m.vertex_id not in ctx.hidden_vertex_ids
        )

        assert ctx.display_counts["maxima"] == raw_max + n_eff_max
        assert ctx.display_counts["minima"] == raw_min + n_eff_min
        assert ctx.display_counts["saddles"] == raw_sad + n_eff_sad
        # The index sum must survive the collapse: chi is preserved.
        assert ctx.display_counts["euler"] == 2

    def test_display_counts_match_with_reduction_off(self) -> None:
        """With reduction off there are no effective CPs, so counts are raw."""
        ctx = _make_ctx()
        results = dict(run_stages(ctx, stop_after="reduction"))

        assert ctx.effective_cps == []
        assert results["reduction"]["display_counts"] == {
            "minima": 2,
            "maxima": 2,
            "saddles": 2,
            "euler": 2,
            "valid": True,
        }


def _conn(**overrides: Any) -> MaxConnection:
    """Build one connector: saddle on +x, maximum on +z, a 3-point arc between.

    Args:
        overrides: ``MaxConnection`` fields to replace.

    Returns:
        A connector with every required field filled and the ridge post-pass
        fields at their defaults.
    """
    base: dict[str, Any] = {
        "saddle_id": "SAD_1",
        "branch_index": 0,
        "max_id": "MAX_1",
        "status": "connected",
        "direction_mode": "port_tangent",
        "cross_track_deg": 1.0,
        "along_track_deg": 2.0,
        "separation_deg": 3.0,
        "mutual": True,
        "arc_points": ((1.0, 0.0, 0.0), (0.7071, 0.0, 0.7071), (0.0, 0.0, 1.0)),
        "saddle_value": 0.0,
        "max_value": 1.0,
        "verified": True,
        "method": "gradient",
    }
    base.update(overrides)
    return MaxConnection(**base)


class TestConnectionRidgeDetails:
    """The per-connector record of the two ridge switches."""

    def test_rows_are_one_per_connector_in_table_order(self) -> None:
        conns = [_conn(saddle_id="SAD_1"), _conn(saddle_id="SAD_2", branch_index=3)]
        rows = connection_ridge_details(conns)
        assert [r["id"] for r in rows] == ["SAD_1#0", "SAD_2#3"]

    def test_carries_the_cards_fields(self) -> None:
        """Every field the card's row/footer shows must be in the record.

        The card reads these out of ``state.max_connections_list``; a replay that
        cannot report them cannot be compared against what was on screen.
        """
        row = connection_ridge_details(
            [_conn(refined=True, refine_max_shift_deg=1.5, n_nodes=9, n_runs=0)]
        )[0]
        for key in (
            "refined",
            "refine_max_shift_deg",
            "n_corners",
            "n_runs",
            "n_nodes",
            "total_turn_before_deg",
            "total_turn_after_deg",
        ):
            assert key in row, f"{key} missing from the per-connector record"
        assert row["refined"] is True
        assert row["n_nodes"] == 9

    def test_values_are_plain_python_for_json(self) -> None:
        """Numpy scalars in the record would make ``--out`` raise on dump."""
        row = connection_ridge_details([_conn(refined=True)])[0]
        assert all(type(v) in (bool, int, float, str, type(None)) for v in row.values())

    def test_untouched_connector_reads_zeroed(self) -> None:
        """Both switches off: the post-pass never ran, so every counter is 0."""
        assert summarize_ridge_postpass([_conn()]) == {
            "n_refined": 0,
            "n_coarse": 0,
            "n_smoothed": 0,
            "max_refine_shift_deg": 0.0,
            "n_curve_nodes": 0,
        }

    def test_coarse_and_smooth_signatures_are_separable(self) -> None:
        """The footer must distinguish the two switches, not just "something ran".

        Measured on the ethene3 atom-1 snapshot: coarse on gives
        ``n_refined=4, n_coarse=4, n_curve_nodes=38``; smooth on gives
        ``n_refined=3, n_coarse=0, n_curve_nodes=0``. ``refined`` alone cannot tell
        those apart, which is why ``n_nodes`` and ``n_runs`` are both recorded.
        """
        coarse = summarize_ridge_postpass([_conn(refined=True, n_nodes=9)])
        smooth = summarize_ridge_postpass([_conn(refined=True, n_runs=4)])
        assert coarse["n_coarse"] == 1 and coarse["n_smoothed"] == 0
        assert smooth["n_coarse"] == 0 and smooth["n_smoothed"] == 1
        assert coarse["n_curve_nodes"] == 9


class TestStageProbe:
    """``probe_ridge_points`` changes no geometry, so it needs its own stage.

    These tests set ``probe_all``: they ask WHICH connectors the measurement runs
    over and what it computes, and the full-field mode is the scope-free way to
    ask that. The display gates that pick the scope in a replay are
    :class:`TestProbeScope`.
    """

    def test_off_reports_zeroed_stats(self) -> None:
        """The default is off, and the stage must say so in the viewer's keys."""
        ctx = _make_ctx()
        ctx.traced_connections = [_conn()]
        ctx.probe_all = True
        stats = stage_probe(ctx)
        assert stats["on"] is False
        assert stats["n_connections"] == 0
        assert ctx.probe_details == []

    def test_stats_use_the_viewers_state_names(self) -> None:
        """Same names as ``state.ridge_probe_stats``, so the two are comparable."""
        ctx = _make_ctx(probe_ridge_points=True)
        ctx.probe_all = True
        ctx.traced_connections = [_conn()]
        stats = stage_probe(ctx)
        assert set(stats) >= {
            "on",
            "n_knots",
            "n_points",
            "n_refused",
            "median_offset_pitch",
        }

    def test_only_gradient_connectors_are_probed(self) -> None:
        """A geodesic ray is a great circle, not an ascent: no ridge to measure."""
        ctx = _make_ctx(probe_ridge_points=True)
        ctx.probe_all = True
        ctx.traced_connections = [
            _conn(saddle_id="SAD_1", method="gradient"),
            _conn(saddle_id="SAD_2", method="geodesic"),
        ]
        stats = stage_probe(ctx)
        assert stats["n_connections"] == 1
        assert [p["saddle_id"] for p in ctx.probe_details] == ["SAD_1"]

    def test_measures_the_traced_table_not_the_postpassed_one(self) -> None:
        """The probe measures the ASCENT, so it reads ``traced_connections``.

        This mirrors the viewer, which hands the probe
        ``runtime_traced_connections``: a smoothed record does not even carry the
        vertices the probe was sized for.
        """
        ctx = _make_ctx(probe_ridge_points=True)
        ctx.probe_all = True
        ctx.traced_connections = [_conn(method="gradient")]
        ctx.connections = [_conn(method="geodesic")]
        assert stage_probe(ctx)["n_connections"] == 1

    def test_short_arcs_are_skipped(self) -> None:
        """A connector with no interior path has no ridge beside it."""
        ctx = _make_ctx(probe_ridge_points=True)
        ctx.probe_all = True
        ctx.traced_connections = [_conn(arc_points=((1.0, 0.0, 0.0),))]
        assert stage_probe(ctx)["n_connections"] == 0


class TestProbeScope:
    """A replayed probe measures the arcs the operator had on screen.

    In the viewer the probe is not a run-wide measurement: it is the diagnostic on
    the saddle-to-basin overlay, so it walks the branches the selection names --
    the selected saddle's launching ones and the selected basin's arriving ones --
    and runs only while the overlay is visible. Those counters are what a snapshot
    saves, so a replay that measured every connector would print numbers no screen
    ever showed and a diff against the saved file would report a change that is the
    tool's, not the parameters'.
    """

    def test_hidden_overlay_means_no_probe(self) -> None:
        """Nothing is on screen, so there is nothing to reproduce."""
        ctx = _make_ctx(probe_ridge_points=True, show_max_saddle_arcs=False)
        ctx.probe_basin_id = "MAX_1"
        ctx.traced_connections = [_conn(method="gradient")]
        stats = stage_probe(ctx)
        assert stats["on"] is False
        assert stats["n_connections"] == 0

    def test_no_selection_means_no_probe(self) -> None:
        """Visible arcs but none selected: the probe walks the selection only."""
        ctx = _make_ctx(probe_ridge_points=True, show_max_saddle_arcs=True)
        ctx.traced_connections = [_conn(method="gradient")]
        stats = stage_probe(ctx)
        assert stats["on"] is False

    def test_basin_scope_takes_the_arriving_branches_only(self) -> None:
        """One saddle, two basins: selecting a basin selects its own branch."""
        ctx = _make_ctx(probe_ridge_points=True, show_max_saddle_arcs=True)
        ctx.probe_basin_id = "MAX_2"
        ctx.connections = [
            _conn(max_id="MAX_1", branch_index=0, method="gradient"),
            _conn(max_id="MAX_2", branch_index=1, method="gradient"),
        ]
        ctx.traced_connections = list(ctx.connections)
        stats = stage_probe(ctx)
        assert stats["n_connections"] == 1
        assert stats["scope_basin_id"] == "MAX_2"
        assert [d["branch_index"] for d in ctx.probe_details] == [1]

    def test_saddle_scope_takes_the_launching_branches_only(self) -> None:
        """One basin, two saddles: selecting a saddle selects its own launches."""
        ctx = _make_ctx(probe_ridge_points=True, show_max_saddle_arcs=True)
        ctx.probe_saddle_id = "SAD_2"
        ctx.connections = [
            _conn(saddle_id="SAD_1", max_id="MAX_1", method="gradient"),
            _conn(saddle_id="SAD_2", max_id="MAX_1", method="gradient"),
        ]
        ctx.traced_connections = list(ctx.connections)
        stats = stage_probe(ctx)
        assert stats["n_connections"] == 1
        assert stats["scope_saddle_id"] == "SAD_2"
        assert [d["saddle_id"] for d in ctx.probe_details] == ["SAD_2"]

    def test_selection_beats_the_traced_table(self) -> None:
        """The scope is the display table; the traced record supplies the path.

        The viewer walks ``max_arcs_list`` (drawn arcs, already basin-culled and
        fusion-culled) and looks the ascent up by ``(saddle_id, branch_index)``. A
        replay that scoped itself to the traced table would probe branches the
        operator never saw.
        """
        ctx = _make_ctx(probe_ridge_points=True, show_max_saddle_arcs=True)
        ctx.probe_basin_id = "MAX_1"
        ctx.connections = [_conn(max_id="MAX_1", method="gradient")]
        ctx.traced_connections = [
            _conn(max_id="MAX_1", method="gradient"),
            _conn(max_id="MAX_2", method="gradient"),
        ]
        stats = stage_probe(ctx)
        assert stats["n_connections"] == 1

    def test_probe_all_drops_the_display_gates(self) -> None:
        """``--probe-all`` is the superset: no selection, hidden overlay, all run.

        It keeps the switch itself. A probe the operator never turned on was never
        on screen, and inventing one is not what the flag asks for.
        """
        ctx = _make_ctx(probe_ridge_points=True, show_max_saddle_arcs=False)
        ctx.probe_all = True
        ctx.traced_connections = [_conn(method="gradient")]
        stats = stage_probe(ctx)
        assert stats["on"] is True
        assert stats["scope_basin_id"] is None

    def test_scope_is_reported_even_when_empty(self) -> None:
        """The card says what it measured, including "nothing", in both modes."""
        ctx = _make_ctx(probe_ridge_points=True)
        stats = stage_probe(ctx)
        assert {"scope_basin_id", "scope_saddle_id"} <= set(stats)
        assert stats["scope_basin_id"] is None

    def test_a_pre_v8_selection_degrades_to_no_scope(self) -> None:
        """A snapshot that never recorded a scope has none, and the replay says so.

        ``selection`` is a passthrough section, so a v7 file arrives with the keys it
        wrote -- no ``selected_saddle_id``, and no probe counters to compare against.
        The reader must not invent a basin out of a missing key: an empty selection
        exports as ``None``, which is the same shape as ``""`` in the schema default.
        """
        assert diagnose_params._scope_id(None, "selected_saddle_id") is None
        assert diagnose_params._scope_id({}, "selected_saddle_id") is None
        assert (
            diagnose_params._scope_id(
                {"selected_saddle_id": None}, "selected_saddle_id"
            )
            is None
        )
        assert (
            diagnose_params._scope_id({"selected_saddle_id": ""}, "selected_saddle_id")
            is None
        )
        assert (
            diagnose_params._scope_id(
                {"selected_basin_center_id": 7}, "selected_basin_center_id"
            )
            is None
        )
        # A real id is a string, and only a string scopes the probe.
        assert (
            diagnose_params._scope_id(
                {"selected_basin_center_id": "MAX_4864"}, "selected_basin_center_id"
            )
            == "MAX_4864"
        )


class TestRidgeOutputPlumbing:
    """The card's output reaches ``--out`` and the stage order is right."""

    def test_probe_runs_after_connect(self) -> None:
        """Probing needs a traced table, so the stage order is load-bearing."""
        names = [name for name, _ in STAGES]
        assert names.index("connect") < names.index("probe")

    def test_connect_saves_the_traced_table(self) -> None:
        """``stage_connect`` must publish the pre-post-pass table for the probe."""
        ctx = _make_ctx()
        for name, fn in STAGES:
            fn(ctx)
            if name == "connect":
                break
        assert len(ctx.traced_connections) == len(ctx.connections)
        assert ctx.connection_details, "the card's per-connector rows are missing"

    def test_dump_records_the_card_output(self, tmp_path: Path) -> None:
        """``--out`` must carry the ridge fields, the footer, and the probe."""
        ctx = _make_ctx(probe_ridge_points=True)
        ctx.probe_all = True
        for name, fn in STAGES:
            fn(ctx)
            if name == "connect":
                # The octahedron resolves its pairs by straight ray, so it yields
                # no ascent for the probe to measure. Hand it a traced gradient
                # connector to exercise the real probe->dump path.
                ctx.traced_connections = [_conn(method="gradient")]
            if name == "probe":
                break
        out = tmp_path / "dump.json"
        dump_results(ctx, str(out))
        payload = json.loads(out.read_text(encoding="utf-8"))
        assert payload["ridge_probe_stats"]["on"] is True
        assert payload["probe_details"][0]["id"] == "SAD_1#0"
        assert payload["connection_ridge_details"]
        assert {"refined", "n_nodes", "total_turn_after_deg"} <= set(
            payload["max_connections"][0]
        )
