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
    def __init__(self, unit_id: str, centroid: tuple[float, float, float], radius: float):
        self.unit_id = unit_id
        self.centroid = centroid
        self.as_cluster = _FakeCluster(radius)


def _eff(label: str, kind: str, pos: tuple[float, float, float]) -> dict[str, Any]:
    return {"id_label": label, "type": kind, "position": list(pos)}


BASE: dict[str, Any] = dict(
    maxima=[_cp(1, "maximum", (0.0, 0.0, 1.0))],
    minima=[_cp(2, "minimum", (0.0, 0.0, -1.0))],
    saddles=[_cp(3, "saddle", (1.0, 0.0, 0.0))],
    effective_cps=[],
    units=[],
    eff_ring_radius={},
    hidden_vertex_ids=set(),
    annihilated_unit_ids=[],
    center=(0.0, 0.0, 0.0),
    delta_theta_mesh_rad=math.radians(7.0),
    k_pitch=2.0,
)


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

        with open(ETHENE_PARAMS_JSON) as fh:
            payload = json.load(fh)
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
