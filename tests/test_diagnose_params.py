"""Unit tests for the diagnose_params headless analysis runner.

Covers the pure, UI-free pieces of ``gba_topology2/diagnose_params.py``:
1. ``diff_leaf`` / ``diff_metrics`` -- the reproduce-check comparison,
   including float tolerance, bool strictness, list leaves, and the
   skip-missing-keys rule that makes --only stops comparable.
2. ``apply_overrides`` -- typed key=value overrides for sweeps.
3. ``run_stages`` -- stage ordering, --only early stop, and the morse-first
   guard.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest
from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.vtkCommonDataModel import vtkPolyData

from gba_topology2.diagnose_params import (
    STAGES,
    MetricDiff,
    RunContext,
    apply_overrides,
    diff_leaf,
    diff_metrics,
    run_stages,
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
        "launch_seed_pitch": 0.35,
        "ring_samples": 48,
        "ring_expand_pitch": 1.0,
        "ring_max_pitch": 12.0,
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
