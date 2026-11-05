"""Unit tests for morse_smale.py (shared Morse-Smale zone construction).

The zone builder is the single source of truth for the separatrix tracer's
boundary conditions, used by both the viewer and the headless harness. These
tests pin the labelling scheme, the cap-radius rules, and the two behaviours
the reduction's conclusions encode: whole-cluster absorption and pass-through of
annihilated clusters.
"""

from __future__ import annotations

import json
import math
import os
from typing import Any

import pytest
from snapshot_params import load_snapshot

from gba_topology2.src.morse_detector import CriticalPoint
from gba_topology2.src.morse_smale import (
    lone_cp_excision_radius_rad,
    trace_zones_from_reduction,
)

PARAMS_DIR = os.path.dirname(os.path.dirname(__file__))
ETHENE_PARAMS_JSON = os.path.join(PARAMS_DIR, "ethene4_atom1_params.json")


def _cp(vid: int, kind: str, pos: tuple[float, float, float]) -> CriticalPoint:
    return CriticalPoint(vertex_id=vid, cp_type=kind, value=0.0, position=pos)


class _FakeCluster:
    def __init__(self, radius_rad: float) -> None:
        self.angular_radius_rad = radius_rad


class _FakeUnit:
    def __init__(
        self, unit_id: str, centroid: tuple[float, float, float], radius: float
    ):
        self.unit_id = unit_id
        self.centroid = centroid
        self.as_cluster = _FakeCluster(radius)


def _eff(label: str, kind: str, pos: tuple[float, float, float]) -> dict[str, Any]:
    return {"id_label": label, "type": kind, "position": list(pos)}


BASE: dict[str, Any] = {
    "maxima": [_cp(1, "maximum", (0.0, 0.0, 1.0))],
    "minima": [_cp(2, "minimum", (0.0, 0.0, -1.0))],
    "saddles": [_cp(3, "saddle", (1.0, 0.0, 0.0))],
    "effective_cps": [],
    "units": [],
    "eff_ring_radius": {},
    "hidden_vertex_ids": set(),
    "annihilated_unit_ids": [],
    "center": (0.0, 0.0, 0.0),
    "delta_theta_mesh_rad": math.radians(7.0),
    "k_pitch": 2.0,
}


class TestLoneCpRadius:
    def test_uses_half_pitch_when_large(self) -> None:
        assert math.isclose(
            lone_cp_excision_radius_rad(4.0, math.radians(7.0)),
            math.radians(14.0),
        )

    def test_floor_is_the_boundary_margin(self) -> None:
        assert math.isclose(
            lone_cp_excision_radius_rad(1.0, math.radians(1.0)), math.radians(1.5)
        )


class TestZoneLabelsAndKinds:
    def test_raw_cps_get_id_space_labels(self) -> None:
        zones = trace_zones_from_reduction(**BASE)
        by_label = {z.label: z for z in zones}
        assert set(by_label) == {"MAX_1", "MIN_2", "SAD_3"}
        assert by_label["MAX_1"].kind == "absorb_max"
        assert by_label["MIN_2"].kind == "absorb_min"
        assert by_label["SAD_3"].kind == "absorb_sad"

    def test_raw_caps_use_the_lone_cp_rule(self) -> None:
        zones = trace_zones_from_reduction(**BASE)
        expect = math.cos(lone_cp_excision_radius_rad(2.0, math.radians(7.0)))
        assert all(math.isclose(z.cos_radius, expect) for z in zones)

    def test_hidden_vertices_get_no_cap(self) -> None:
        zones = trace_zones_from_reduction(**{**BASE, "hidden_vertex_ids": {1, 3}})
        assert {z.label for z in zones} == {"MIN_2"}

    def test_effective_cps_use_their_id_label(self) -> None:
        zones = trace_zones_from_reduction(
            **{
                **BASE,
                "effective_cps": [
                    _eff("E-MAX_MC28", "E-MAX", (0.0, 1.0, 0.0)),
                    _eff("E-MIN_MC9", "E-MIN", (0.0, -1.0, 0.0)),
                    _eff("E-SAD_MC1", "E-SAD", (0.0, 0.0, 1.0)),
                ],
            }
        )
        by_label = {z.label: z for z in zones}
        assert by_label["E-MAX_MC28"].kind == "absorb_max"
        assert by_label["E-MIN_MC9"].kind == "absorb_min"
        assert by_label["E-SAD_MC1"].kind == "absorb_sad"

    def test_effective_cps_prefer_their_recorded_ring_radius(self) -> None:
        zones = trace_zones_from_reduction(
            **{
                **BASE,
                "effective_cps": [_eff("E-MAX_MC28", "E-MAX", (0.0, 1.0, 0.0))],
                "eff_ring_radius": {"E-MAX_MC28": 0.4},
            }
        )
        by_label = {z.label: z for z in zones}
        assert math.isclose(by_label["E-MAX_MC28"].cos_radius, math.cos(0.4))
        # A lone raw CP keeps the rule-based radius.
        assert math.isclose(
            by_label["MAX_1"].cos_radius,
            math.cos(lone_cp_excision_radius_rad(2.0, math.radians(7.0))),
        )

    def test_unknown_effective_type_is_skipped(self) -> None:
        zones = trace_zones_from_reduction(
            **{
                **BASE,
                "effective_cps": [_eff("E-NONE_MC5", "E-NONE", (0.0, 1.0, 0.0))],
            }
        )
        assert "E-NONE_MC5" not in {z.label for z in zones}


class TestPassThroughZones:
    def test_annihilated_units_become_passthrough(self) -> None:
        zones = trace_zones_from_reduction(
            **{
                **BASE,
                "units": [_FakeUnit("MC10", (0.0, 0.0, 1.0), 0.3)],
                "annihilated_unit_ids": ["MC10"],
            }
        )
        by_label = {z.label: z for z in zones}
        assert by_label["ANNUL_MC10"].kind == "passthrough"
        assert math.isclose(by_label["ANNUL_MC10"].cos_radius, math.cos(0.3))

    def test_small_annihilated_unit_is_floored_to_lone_rule(self) -> None:
        zones = trace_zones_from_reduction(
            **{
                **BASE,
                "units": [_FakeUnit("MC10", (0.0, 0.0, 1.0), 1e-9)],
                "annihilated_unit_ids": ["MC10"],
            }
        )
        by_label = {z.label: z for z in zones}
        expect = math.cos(lone_cp_excision_radius_rad(2.0, math.radians(7.0)))
        assert math.isclose(by_label["ANNUL_MC10"].cos_radius, expect)

    def test_unknown_annihilated_id_is_ignored(self) -> None:
        zones = trace_zones_from_reduction(
            **{**BASE, "units": [], "annihilated_unit_ids": ["MC99"]}
        )
        assert not any(z.kind == "passthrough" for z in zones)

    def test_absorption_zones_come_before_passthrough(self) -> None:
        zones = trace_zones_from_reduction(
            **{
                **BASE,
                "units": [_FakeUnit("MC10", (0.0, 0.0, 1.0), 0.3)],
                "annihilated_unit_ids": ["MC10"],
            }
        )
        kinds = [z.kind for z in zones]
        assert kinds.index("passthrough") == len(kinds) - 1


class TestFrameContract:
    def test_effective_positions_are_mapped_out_of_world_frame(self) -> None:
        """Effective CP dicts carry world positions; caps must use directions
        measured from the nucleus, so an offset centre is subtracted."""
        center = (10.0, 0.0, 0.0)
        # World position radially outward from the nucleus along +z.
        zones = trace_zones_from_reduction(
            **{
                **BASE,
                "center": center,
                "effective_cps": [_eff("E-MAX_MC1", "E-MAX", (10.0, 0.0, 1.0))],
            }
        )
        by_label = {z.label: z for z in zones}
        assert by_label["E-MAX_MC1"].center_unit == pytest.approx((0.0, 0.0, 1.0))

    def test_raw_positions_are_already_centred(self) -> None:
        center = (10.0, 0.0, 0.0)
        zones = trace_zones_from_reduction(**{**BASE, "center": center})
        by_label = {z.label: z for z in zones}
        # Raw CP positions are centred-frame, so the nucleus offset is ignored.
        assert by_label["MAX_1"].center_unit == pytest.approx((0.0, 0.0, 1.0))


@pytest.mark.skipif(
    not os.path.exists(ETHENE_PARAMS_JSON),
    reason="ethene4_atom1_params.json not present",
)
class TestHarnessAdapterAgreement:
    """The harness adapter must be a pure pass-through to the shared builder."""

    def test_adapter_matches_direct_call(self) -> None:
        from gba_topology2.diagnose_params import (
            build_context,
            build_trace_zones,
            run_stages,
        )
        from gba_topology2.src.sphere_viewer import load_params_from_dict

        payload = load_snapshot(ETHENE_PARAMS_JSON)
        params = load_params_from_dict(payload)
        ctx = build_context(params, plt_path=None, atom_num=None, field_name=None)
        run_stages(ctx)

        adapted = build_trace_zones(ctx)
        direct = trace_zones_from_reduction(
            maxima=ctx.require_morse().maxima,
            minima=ctx.require_morse().minima,
            saddles=ctx.require_morse().saddles,
            effective_cps=ctx.effective_cps,
            units=ctx.units,
            eff_ring_radius=ctx.eff_ring_radius,
            hidden_vertex_ids=ctx.hidden_vertex_ids,
            annihilated_unit_ids=ctx.annihilated_units,
            center=ctx.center,
            delta_theta_mesh_rad=ctx.delta_theta_rad,
            k_pitch=float(params["k_pitch"]),
        )
        assert [(z.label, z.kind, z.cos_radius) for z in adapted] == [
            (z.label, z.kind, z.cos_radius) for z in direct
        ]
        # Every zone label is in the id space the validator expects: extrema
        # match basin-center ids, saddles match launch-seed ids, annihilated
        # units are ANNUL_-prefixed and never targets.
        extrema_ids = {b.center_id for b in ctx.basin_centers}
        seed_ids = {s.saddle_id for s in ctx.seeds}
        for z in adapted:
            if z.kind == "passthrough":
                assert z.label.startswith("ANNUL_")
            elif z.kind == "absorb_sad":
                assert z.label in seed_ids
            else:
                assert z.label in extrema_ids


@pytest.mark.skipif(
    not os.path.exists(ETHENE_PARAMS_JSON),
    reason="ethene4_atom1_params.json not present",
)
class TestViewerTraceEntryAgreement:
    """The viewer's trace entry point must match the harness step for step.

    ``trace_morse_smale_complex`` is the seam both sides call; this pins that
    feeding it the harness's own pipeline output reproduces the harness's
    adjacency, outcomes and verdict exactly, so an on-screen trace is the trace
    a saved snapshot reproduces headlessly.
    """

    @staticmethod
    def _pipeline() -> tuple[Any, Any]:
        from gba_topology2.diagnose_params import build_context, run_stages
        from gba_topology2.src.sphere_viewer import load_params_from_dict

        payload = load_snapshot(ETHENE_PARAMS_JSON)
        params = load_params_from_dict(payload)
        ctx = build_context(params, plt_path=None, atom_num=None, field_name=None)
        run_stages(ctx)
        return ctx, params

    @classmethod
    def _trace(cls) -> tuple[Any, Any]:
        """Run the pipeline, then the shared viewer entry point over its output."""
        from gba_topology2.src.sphere_viewer import trace_morse_smale_complex

        ctx, params = cls._pipeline()
        traced = trace_morse_smale_complex(
            seeds=ctx.seeds,
            pts=ctx.pts,
            triangles=ctx.triangles,
            f_vals=ctx.f_vals,
            maxima=ctx.require_morse().maxima,
            minima=ctx.require_morse().minima,
            saddles=ctx.require_morse().saddles,
            effective_cps=ctx.effective_cps,
            units=ctx.units,
            eff_ring_radius=ctx.eff_ring_radius,
            hidden_vertex_ids=ctx.hidden_vertex_ids,
            annihilated_unit_ids=ctx.annihilated_units,
            basin_centers=ctx.basin_centers,
            center=ctx.center,
            delta_theta_mesh_rad=ctx.delta_theta_rad,
            k_pitch=float(params["k_pitch"]),
        )
        return ctx, traced

    def test_adjacency_and_verdict_match_harness(self) -> None:
        """The shared entry point reproduces the harness step for step."""
        from gba_topology2.diagnose_params import (
            build_trace_zones,
            stage_validate,
        )
        from gba_topology2.src.basin_validator import build_traced_adjacency
        from gba_topology2.src.separatrix_tracer import build_walk_adjacency

        ctx, traced = self._trace()
        harness = stage_validate(ctx)

        # The harness path, computed independently of the shared entry point.
        nbrs = build_walk_adjacency(ctx.triangles, int(ctx.pts.shape[0]))
        harness_adj, _ = build_traced_adjacency(
            ctx.seeds, ctx.pts, ctx.f_vals, nbrs, build_trace_zones(ctx)
        )

        assert traced.adjacency == harness_adj
        assert traced.stats["n_branches"] == harness["n_branches"]
        assert traced.stats["n_absorbed"] == harness["n_branches_absorbed"]
        assert traced.stats["is_valid"] == harness["is_valid"]
        assert (
            f"{traced.stats['n_max_valid']}/{traced.stats['n_max_total']}"
            == harness["n_max_valid"]
        )
        assert (
            f"{traced.stats['n_min_valid']}/{traced.stats['n_min_total']}"
            == harness["n_min_valid"]
        )
        assert traced.stats["summary"] == harness["summary"]

    def test_zone_count_matches_adapter(self) -> None:
        from gba_topology2.diagnose_params import build_trace_zones

        ctx, traced = self._trace()
        assert [(z.label, z.kind) for z in traced.zones] == [
            (z.label, z.kind) for z in build_trace_zones(ctx)
        ]

    def test_stats_shape_is_json_serialisable(self) -> None:
        """The stats dict is bound into the Vue template, so it must be plain."""
        ctx, traced = self._trace()
        json.dumps(traced.stats)  # must not raise
        assert traced.stats["n_saddles_traced"] == len(ctx.seeds)
        assert traced.stats["n_saddles_traced"] == len(ctx.seeds)
