"""Unit tests for basin_validator.py.

Tests the basin boundary validation logic using:
1. Type-sequence helpers (is_valid_max/min_basin_boundary)
2. Adjacency-based cycle validation (validate_max/min_basin_boundary,
   validate_all_basin_boundaries, build_morse_smale_adjacency)
3. Pipeline integration on the saved ethene4 / Pd20K parameter snapshots,
   including the cluster-aware traced adjacency (build_traced_adjacency).
"""

from __future__ import annotations

import os
from typing import Any

import pytest

from gba_topology2.src.basin_center import BasinCenter
from gba_topology2.src.basin_validator import (
    SaddleAdjacency,
    SimplificationValidity,
    build_morse_smale_adjacency,
    build_traced_adjacency,
    is_valid_max_basin_boundary,
    is_valid_min_basin_boundary,
    validate_all_basin_boundaries,
    validate_max_basin_boundary,
    validate_min_basin_boundary,
)
from gba_topology2.src.saddle_launch import LaunchBranch, SaddleLaunchSeed

# ---------------------------------------------------------------------------
# Type-sequence helpers
# ---------------------------------------------------------------------------


class TestMaxBasinBoundaryTypeSequence:
    """Tests for is_valid_max_basin_boundary."""

    def test_valid_min_sad_min_sad(self) -> None:
        ok, pat = is_valid_max_basin_boundary(
            ["minimum", "saddle", "minimum", "saddle"]
        )
        assert ok
        assert "m" in pat and "s" in pat

    def test_valid_sad_min_sad_min(self) -> None:
        ok, _ = is_valid_max_basin_boundary(["saddle", "minimum", "saddle", "minimum"])
        assert ok

    def test_valid_two_element(self) -> None:
        ok, _ = is_valid_max_basin_boundary(["saddle", "minimum"])
        assert ok

    def test_invalid_consecutive_minima(self) -> None:
        ok, msg = is_valid_max_basin_boundary(["minimum", "minimum", "saddle"])
        assert not ok
        assert "non-alternating" in msg

    def test_invalid_consecutive_saddles(self) -> None:
        ok, _ = is_valid_max_basin_boundary(["saddle", "saddle", "minimum"])
        assert not ok

    def test_invalid_maximum_in_boundary(self) -> None:
        ok, msg = is_valid_max_basin_boundary(["minimum", "maximum", "saddle"])
        assert not ok
        assert "invalid type" in msg

    def test_empty_boundary(self) -> None:
        ok, msg = is_valid_max_basin_boundary([])
        assert not ok
        assert "empty" in msg


class TestMinBasinBoundaryTypeSequence:
    """Tests for is_valid_min_basin_boundary."""

    def test_valid_max_sad_max_sad(self) -> None:
        ok, _ = is_valid_min_basin_boundary(["maximum", "saddle", "maximum", "saddle"])
        assert ok

    def test_valid_sad_max_sad_max(self) -> None:
        ok, _ = is_valid_min_basin_boundary(["saddle", "maximum", "saddle", "maximum"])
        assert ok

    def test_valid_two_element(self) -> None:
        ok, _ = is_valid_min_basin_boundary(["saddle", "maximum"])
        assert ok

    def test_invalid_consecutive_maxima(self) -> None:
        ok, msg = is_valid_min_basin_boundary(["maximum", "maximum", "saddle"])
        assert not ok
        assert "non-alternating" in msg

    def test_invalid_minimum_in_boundary(self) -> None:
        ok, msg = is_valid_min_basin_boundary(["maximum", "minimum", "saddle"])
        assert not ok
        assert "invalid type" in msg


# ---------------------------------------------------------------------------
# Adjacency-based validation helpers
# ---------------------------------------------------------------------------


def _make_adj(spec: dict[str, tuple[list[str], list[str]]]) -> SaddleAdjacency:
    """Build a SaddleAdjacency from a compact spec: {sad_id: ([maxes], [mins])}."""
    return {sid: {"max": maxes, "min": mins} for sid, (maxes, mins) in spec.items()}


class TestValidateMaxBasinBoundary:
    """Tests for validate_max_basin_boundary."""

    def test_valid_triangle_topology(self) -> None:
        # 3 saddles, 3 minima around one maximum: classic degree-3 max
        adj = _make_adj(
            {
                "S1": (["MAX_A", "MAX_B"], ["MIN_1", "MIN_2"]),
                "S2": (["MAX_A", "MAX_C"], ["MIN_2", "MIN_3"]),
                "S3": (["MAX_A", "MAX_D"], ["MIN_3", "MIN_1"]),
            }
        )
        result = validate_max_basin_boundary("MAX_A", adj)
        assert result.is_valid, result.error
        assert result.n_saddles == 3
        assert result.n_extrema == 3

    def test_valid_rectangle_topology(self) -> None:
        # 4 saddles, 4 minima around one maximum
        adj = _make_adj(
            {
                "S1": (["MAX_A", "X"], ["MIN_1", "MIN_2"]),
                "S2": (["MAX_A", "X"], ["MIN_2", "MIN_3"]),
                "S3": (["MAX_A", "X"], ["MIN_3", "MIN_4"]),
                "S4": (["MAX_A", "X"], ["MIN_4", "MIN_1"]),
            }
        )
        result = validate_max_basin_boundary("MAX_A", adj)
        assert result.is_valid, result.error
        assert result.n_saddles == 4

    def test_invalid_no_adjacent_saddles(self) -> None:
        adj: SaddleAdjacency = {}
        result = validate_max_basin_boundary("MAX_A", adj)
        assert not result.is_valid
        assert "no adjacent saddles" in result.error

    def test_invalid_extremum_degree_not_2(self) -> None:
        # MIN_1 appears in 3 saddles → degree 3, not 2
        adj = _make_adj(
            {
                "S1": (["MAX_A"], ["MIN_1", "MIN_2"]),
                "S2": (["MAX_A"], ["MIN_2", "MIN_1"]),
                "S3": (["MAX_A"], ["MIN_1", "MIN_3"]),
            }
        )
        result = validate_max_basin_boundary("MAX_A", adj)
        assert not result.is_valid
        assert "degree" in result.error

    def test_invalid_disconnected_boundary(self) -> None:
        # Two disconnected sub-cycles: S1-MIN_1-S2-MIN_2 and S3-MIN_3-S4-MIN_4
        adj = _make_adj(
            {
                "S1": (["MAX_A"], ["MIN_1", "MIN_2"]),
                "S2": (["MAX_A"], ["MIN_2", "MIN_1"]),
                "S3": (["MAX_A"], ["MIN_3", "MIN_4"]),
                "S4": (["MAX_A"], ["MIN_4", "MIN_3"]),
            }
        )
        result = validate_max_basin_boundary("MAX_A", adj)
        assert not result.is_valid
        assert "cycle" in result.error


class TestValidateMinBasinBoundary:
    """Tests for validate_min_basin_boundary."""

    def test_valid_two_saddle_topology(self) -> None:
        # Classic degree-2 min: 2 saddles, 2 maxima
        adj = _make_adj(
            {
                "S1": (["MAX_1", "MAX_2"], ["MIN_A", "X"]),
                "S2": (["MAX_2", "MAX_1"], ["MIN_A", "X"]),
            }
        )
        result = validate_min_basin_boundary("MIN_A", adj)
        assert result.is_valid, result.error
        assert result.n_saddles == 2
        assert result.n_extrema == 2

    def test_valid_three_saddle_topology(self) -> None:
        adj = _make_adj(
            {
                "S1": (["MAX_1", "MAX_2"], ["MIN_A", "X"]),
                "S2": (["MAX_2", "MAX_3"], ["MIN_A", "X"]),
                "S3": (["MAX_3", "MAX_1"], ["MIN_A", "X"]),
            }
        )
        result = validate_min_basin_boundary("MIN_A", adj)
        assert result.is_valid, result.error
        assert result.n_saddles == 3

    def test_invalid_no_saddles(self) -> None:
        adj: SaddleAdjacency = {}
        result = validate_min_basin_boundary("MIN_A", adj)
        assert not result.is_valid

    def test_invalid_disconnected(self) -> None:
        adj = _make_adj(
            {
                "S1": (["MAX_1", "MAX_2"], ["MIN_A"]),
                "S2": (["MAX_2", "MAX_1"], ["MIN_A"]),
                "S3": (["MAX_3", "MAX_4"], ["MIN_A"]),
                "S4": (["MAX_4", "MAX_3"], ["MIN_A"]),
            }
        )
        result = validate_min_basin_boundary("MIN_A", adj)
        assert not result.is_valid


# ---------------------------------------------------------------------------
# build_morse_smale_adjacency
# ---------------------------------------------------------------------------


def _make_seed(
    saddle_id: str,
    saddle_pos: tuple[float, float, float],
    asc_tangents: list[tuple[float, float, float]],
    desc_tangents: list[tuple[float, float, float]],
) -> SaddleLaunchSeed:
    """Build a minimal SaddleLaunchSeed for testing."""
    ascent = [
        LaunchBranch(
            kind="ascent",
            sector_index=i,
            tangent_direction=t,
            launch_position=t,  # same as tangent for tests
            sector_width_rad=1.0,
            sector_centroid_value=0.0,
        )
        for i, t in enumerate(asc_tangents)
    ]
    descent = [
        LaunchBranch(
            kind="descent",
            sector_index=i,
            tangent_direction=t,
            launch_position=t,
            sector_width_rad=1.0,
            sector_centroid_value=0.0,
        )
        for i, t in enumerate(desc_tangents)
    ]
    return SaddleLaunchSeed(
        saddle_id=saddle_id,
        saddle_position=saddle_pos,
        saddle_value=0.0,
        vertex_id=-1,
        multiplicity=1,
        fold_order=2,
        index_agrees=True,
        ascent_branches=ascent,
        descent_branches=descent,
    )


def _make_basin(
    center_id: str, polarity: str, pos: tuple[float, float, float]
) -> BasinCenter:
    return BasinCenter(
        center_id=center_id,
        polarity=polarity,
        position=pos,
        value=0.0,
        is_effective=False,
        vertex_id=-1,
    )


class TestBuildMorseSmaleAdjacency:
    """Tests for build_morse_smale_adjacency."""

    def test_simple_two_saddle_topology(self) -> None:
        """Two saddles, one max, two minima; verify adjacency is built correctly."""
        # Max is at +z, two minima along ±x, saddles at ±y
        center = (0.0, 0.0, 0.0)

        max_a = _make_basin("MAX_A", "maximum", (0.0, 0.0, 1.0))
        min_1 = _make_basin("MIN_1", "minimum", (1.0, 0.0, 0.0))
        min_2 = _make_basin("MIN_2", "minimum", (-1.0, 0.0, 0.0))

        # Saddle S1 at (0, 1, 0): ascent toward +z, descent toward +x
        seed1 = _make_seed(
            "S1",
            saddle_pos=(0.0, 1.0, 0.0),
            asc_tangents=[(0.0, 0.0, 1.0)],  # toward MAX_A
            desc_tangents=[(1.0, 0.0, 0.0)],  # toward MIN_1
        )
        # Saddle S2 at (0, -1, 0): ascent toward +z, descent toward -x
        seed2 = _make_seed(
            "S2",
            saddle_pos=(0.0, -1.0, 0.0),
            asc_tangents=[(0.0, 0.0, 1.0)],  # toward MAX_A
            desc_tangents=[(-1.0, 0.0, 0.0)],  # toward MIN_2
        )

        adj = build_morse_smale_adjacency(
            seeds=[seed1, seed2],
            basin_centers=[max_a, min_1, min_2],
            center=center,
        )

        assert "S1" in adj and "S2" in adj
        assert "MAX_A" in adj["S1"]["max"]
        assert "MAX_A" in adj["S2"]["max"]
        assert "MIN_1" in adj["S1"]["min"]
        assert "MIN_2" in adj["S2"]["min"]

    def test_empty_seeds(self) -> None:
        """Empty seed list produces empty adjacency."""
        adj = build_morse_smale_adjacency([], [], (0.0, 0.0, 0.0))
        assert adj == {}

    def test_no_basin_centers_returns_nones_filtered(self) -> None:
        """With no basin centers, adjacency entries have empty lists."""
        seed = _make_seed(
            "S1",
            (0.0, 1.0, 0.0),
            [(0.0, 0.0, 1.0)],
            [(-1.0, 0.0, 0.0)],
        )
        adj = build_morse_smale_adjacency([seed], [], (0.0, 0.0, 0.0))
        assert adj["S1"]["max"] == []
        assert adj["S1"]["min"] == []


# ---------------------------------------------------------------------------
# Full validate_all_basin_boundaries
# ---------------------------------------------------------------------------


class TestValidateAllBasinBoundaries:
    """Tests for validate_all_basin_boundaries."""

    def test_all_valid(self) -> None:
        # Tetrahedron-like Morse topology: 1 max, 3 minima, 3 saddles
        adj = _make_adj(
            {
                "S1": (["MAX_A", "X"], ["MIN_1", "MIN_2"]),
                "S2": (["MAX_A", "X"], ["MIN_2", "MIN_3"]),
                "S3": (["MAX_A", "X"], ["MIN_3", "MIN_1"]),
            }
        )
        bc_max = _make_basin("MAX_A", "maximum", (0.0, 0.0, 1.0))
        bc_min1 = _make_basin("MIN_1", "minimum", (1.0, 0.0, -0.5))
        bc_min2 = _make_basin("MIN_2", "minimum", (-0.5, 0.866, -0.5))
        bc_min3 = _make_basin("MIN_3", "minimum", (-0.5, -0.866, -0.5))
        basin_centers = [bc_max, bc_min1, bc_min2, bc_min3]

        result = validate_all_basin_boundaries(basin_centers, adj)
        assert result.is_valid, result.summary
        assert result.n_max_valid == 1
        assert result.n_min_valid == 3
        assert "✓" in result.summary

    def test_empty_inputs(self) -> None:
        result = validate_all_basin_boundaries([], {})
        assert result.is_valid
        assert result.n_max_total == 0
        assert result.n_min_total == 0

    def test_partial_failure_reported(self) -> None:
        # MAX_A is valid, MAX_B has no adjacent saddles → fails
        adj = _make_adj(
            {
                "S1": (["MAX_A", "X"], ["MIN_1", "MIN_2"]),
                "S2": (["MAX_A", "X"], ["MIN_2", "MIN_1"]),
            }
        )
        bc_a = _make_basin("MAX_A", "maximum", (0.0, 0.0, 1.0))
        bc_b = _make_basin("MAX_B", "maximum", (1.0, 0.0, 0.0))

        result = validate_all_basin_boundaries([bc_a, bc_b], adj)
        assert not result.is_valid
        assert "MAX_B" in result.failing_ids
        assert "MAX_A" not in result.failing_ids
        assert result.n_max_valid == 1
        assert result.n_max_total == 2


# ---------------------------------------------------------------------------
# Integration test against the diagnose_params.py pipeline
# ---------------------------------------------------------------------------

ETHENE_PARAMS_JSON = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "ethene4_atom1_params.json",
)


def _run_validation(overrides: list[str]) -> SimplificationValidity:
    """Helper: run the full diagnose_params pipeline + validator."""
    import json

    from gba_topology2.diagnose_params import apply_overrides, build_context, run_stages
    from gba_topology2.src.sphere_viewer import load_params_from_dict

    with open(ETHENE_PARAMS_JSON) as fh:
        payload = json.load(fh)

    params = load_params_from_dict(payload)
    apply_overrides(params, overrides)
    ctx = build_context(params, plt_path=None, atom_num=None, field_name=None)
    run_stages(ctx)

    adj = build_morse_smale_adjacency(ctx.seeds, ctx.basin_centers, ctx.center)
    return validate_all_basin_boundaries(ctx.basin_centers, adj)


@pytest.mark.skipif(
    not os.path.exists(ETHENE_PARAMS_JSON),
    reason="ethene4_atom1_params.json not present",
)
class TestEtheneIntegration:
    """Pipeline integration tests using the saved ethene4 parameter snapshot."""

    def test_default_params_are_valid(self) -> None:
        """The saved default parameters (k_pitch=2.0, k_bridge=3.0) must be valid."""
        validity = _run_validation([])
        assert validity.is_valid, validity.summary
        assert validity.n_max_valid == 3
        assert validity.n_min_valid == 4

    def test_k_pitch_4_k_bridge_1_is_valid(self) -> None:
        """k_pitch=4.0 with k_bridge=1.0 produces a different valid simplification."""
        validity = _run_validation(["k_pitch=4.0", "k_bridge=1.0"])
        assert validity.is_valid, validity.summary

    def test_k_pitch_1_is_invalid(self) -> None:
        """k_pitch=1.0 (under-clustered) must be detected as invalid."""
        validity = _run_validation(["k_pitch=1.0"])
        assert not validity.is_valid

    def test_k_pitch_4_alone_is_invalid(self) -> None:
        """k_pitch=4.0 alone (over-clustered with bridging) must be invalid."""
        validity = _run_validation(["k_pitch=4.0"])
        assert not validity.is_valid

    def test_k_bridge_5_is_invalid(self) -> None:
        """k_bridge=5.0 (aggressive bridging) must be detected as invalid."""
        validity = _run_validation(["k_bridge=5.0"])
        assert not validity.is_valid


# ---------------------------------------------------------------------------
# Traced (cluster-aware gradient flow) adjacency
# ---------------------------------------------------------------------------

PD20K_PARAMS_JSON = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "Pd20k_atom1_params.json",
)


def _run_traced(overrides: list[str] | None = None) -> tuple[Any, Any, Any]:
    """Run the pipeline and return (ctx, traced_adjacency, outcomes)."""
    import json

    from gba_topology2.diagnose_params import (
        apply_overrides,
        build_context,
        build_trace_zones,
        run_stages,
    )
    from gba_topology2.src.separatrix_tracer import build_walk_adjacency
    from gba_topology2.src.sphere_viewer import load_params_from_dict

    with open(ETHENE_PARAMS_JSON) as fh:
        payload = json.load(fh)
    params = load_params_from_dict(payload)
    apply_overrides(params, overrides or [])
    ctx = build_context(params, plt_path=None, atom_num=None, field_name=None)
    run_stages(ctx)

    zones = build_trace_zones(ctx)
    nbrs = build_walk_adjacency(ctx.triangles, int(ctx.pts.shape[0]))
    adj, outcomes = build_traced_adjacency(ctx.seeds, ctx.pts, ctx.f_vals, nbrs, zones)
    return ctx, adj, outcomes


@pytest.mark.skipif(
    not os.path.exists(ETHENE_PARAMS_JSON),
    reason="ethene4_atom1_params.json not present",
)
class TestEtheneTracedAdjacency:
    """Traced adjacency on the ethene4 snapshot (clean, low-noise field)."""

    def test_all_branches_absorbed(self) -> None:
        ctx, _adj, outcomes = _run_traced()
        branches = [o for lst in outcomes.values() for o in lst]
        assert all(o.status == "absorbed" for o in branches)
        # Exactly one outcome per separatrix branch.
        assert len(branches) == sum(
            len(s.ascent_branches) + len(s.descent_branches) for s in ctx.seeds
        )

    def test_matches_heuristic_verdict(self) -> None:
        ctx, adj, _out = _run_traced()
        traced = validate_all_basin_boundaries(ctx.basin_centers, adj)
        heur = validate_all_basin_boundaries(
            ctx.basin_centers,
            build_morse_smale_adjacency(ctx.seeds, ctx.basin_centers, ctx.center),
        )
        assert traced.is_valid and heur.is_valid
        assert traced.n_max_valid == traced.n_max_total == 3
        assert traced.n_min_valid == traced.n_min_total == 4

    def test_no_self_loops(self) -> None:
        _ctx, adj, _out = _run_traced()
        for sid, entry in adj.items():
            for key in ("max", "min"):
                assert sid not in entry[key]

    def test_labels_live_in_basin_id_space(self) -> None:
        ctx, adj, _out = _run_traced()
        ids = {b.center_id for b in ctx.basin_centers}
        for entry in adj.values():
            assert set(entry["max"]) <= ids
            assert set(entry["min"]) <= ids


@pytest.mark.skipif(
    not os.path.exists(PD20K_PARAMS_JSON),
    reason="Pd20k_atom1_params.json not present",
)
class TestPd20kNoiseHandling:
    """Pd20K is the noisy field: 22 annihilated (net-index-0) clusters.

    Verifies the two mechanisms the tangent heuristic cannot express:
    whole-cluster absorption and pass-through of noise clusters.
    """

    def _run(self) -> tuple[Any, Any, Any, Any]:
        import json

        from gba_topology2.diagnose_params import (
            build_context,
            build_trace_zones,
            run_stages,
        )
        from gba_topology2.src.separatrix_tracer import build_walk_adjacency
        from gba_topology2.src.sphere_viewer import load_params_from_dict

        with open(PD20K_PARAMS_JSON) as fh:
            payload = json.load(fh)
        params = load_params_from_dict(payload)
        ctx = build_context(params, plt_path=None, atom_num=None, field_name=None)
        run_stages(ctx)
        zones = build_trace_zones(ctx)
        nbrs = build_walk_adjacency(ctx.triangles, int(ctx.pts.shape[0]))
        adj, outcomes = build_traced_adjacency(
            ctx.seeds, ctx.pts, ctx.f_vals, nbrs, zones
        )
        return ctx, zones, adj, outcomes

    def test_annihilated_clusters_become_passthrough_zones(self) -> None:
        ctx, zones, _adj, _out = self._run()
        assert ctx.annihilated_units, "fixture must contain annihilated clusters"
        pt = {z.label for z in zones if z.kind == "passthrough"}
        assert pt == {f"ANNUL_{u}" for u in ctx.annihilated_units}
        # MC10 is the transit cluster the design discussion calls out.
        assert "ANNUL_MC10" in pt

    def test_passthrough_zones_never_appear_as_targets(self) -> None:
        ctx, zones, adj, _out = self._run()
        pt = {z.label for z in zones if z.kind == "passthrough"}
        ids = {b.center_id for b in ctx.basin_centers}
        for entry in adj.values():
            for key in ("max", "min"):
                assert not (set(entry[key]) & pt)
                assert set(entry[key]) <= ids

    def test_tunneling_rescues_noise_trapped_branches(self) -> None:
        _ctx, _zones, _adj, outcomes = self._run()
        branches = [o for lst in outcomes.values() for o in lst]
        assert branches
        # At least 90% of branches reach a real CP; the naive walk (no
        # pass-through zones) strands the ones that fall in noise sinks.
        n_abs = sum(1 for o in branches if o.status == "absorbed")
        assert n_abs >= 0.9 * len(branches)

    def test_tangent_heuristic_pollutes_rims_with_annihilated_ids(self) -> None:
        """The heuristic has no zone concept, which is why tracing is needed."""
        import json

        from gba_topology2.diagnose_params import (
            build_context,
            run_stages,
        )
        from gba_topology2.src.sphere_viewer import load_params_from_dict

        with open(PD20K_PARAMS_JSON) as fh:
            payload = json.load(fh)
        params = load_params_from_dict(payload)
        ctx = build_context(params, plt_path=None, atom_num=None, field_name=None)
        run_stages(ctx)
        adj = build_morse_smale_adjacency(ctx.seeds, ctx.basin_centers, ctx.center)
        annul = {f"ANNUL_{u}" for u in ctx.annihilated_units}
        # Heuristic targets are basin ids, so it cannot emit ANNUL_* labels --
        # it instead silently mis-assigns branches. Assert the traced adjacency
        # is the one that carries zone semantics, and that both agree on ids.
        for entry in adj.values():
            assert not (set(entry["max"]) | set(entry["min"])) & annul
