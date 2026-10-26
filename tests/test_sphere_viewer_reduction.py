"""Unit tests for the sphere_viewer reduction-ownership rule and UI defaults.

Covers two Step 6 additions:
1. ``catastrophe_claimed_cluster_ids`` -- the rule deciding which micro-clusters
   the catastrophe stage owns, and therefore which the standalone multi-cluster
   stage must skip.
2. ``DEFAULT_UI_STATE`` -- the single canonical baseline consumed by both the
   initial state seeding and the "Reset to Defaults" action.
"""

from __future__ import annotations

import math
import pathlib
import re
from typing import Any

import pytest

import gba_topology2.src.sphere_viewer as sv
from gba_topology2.src.catastrophe_classifier import EffectiveCatastrophe
from gba_topology2.src.micro_cluster import MicroCluster
from gba_topology2.src.morse_detector import CriticalPoint
from gba_topology2.src.sphere_viewer import (
    DEFAULT_UI_STATE,
    catastrophe_claimed_cluster_ids,
    fusable_catastrophes,
)


def _make_cluster(cluster_id: str, local_chi: int) -> MicroCluster:
    """Build a minimal MicroCluster carrying a given net index.

    Args:
        cluster_id: Identifier to assign.
        local_chi: Net local Euler index to record.

    Returns:
        A MicroCluster sufficient for ownership tests (geometry is unused).
    """
    pos = (0.0, 0.0, 1.0)
    cp = CriticalPoint(
        vertex_id=abs(hash(cluster_id)) % 10000,
        cp_type="maximum" if local_chi >= 0 else "saddle",
        value=1.0,
        position=pos,
        multiplicity=1,
    )
    return MicroCluster(
        cluster_id=cluster_id,
        members=[cp],
        centroid=pos,
        centroid_unit=(0.0, 0.0, 1.0),
        angular_radius_rad=0.02,
        angular_radius_deg=math.degrees(0.02),
        angular_diameter_rad=0.0,
        angular_diameter_deg=0.0,
        spatial_diameter=0.0,
        n_maxima=1,
        n_minima=0,
        n_saddles=0,
        total_saddle_index=0,
        local_euler_index=local_chi,
        is_multi_cp=True,
    )


def _make_cat(
    entity_id: str,
    cluster_ids: list[str],
    *,
    is_fused: bool = True,
    classification: str = "spurious_unfolding",
) -> EffectiveCatastrophe:
    """Build a minimal catastrophe constellation over the named clusters.

    Args:
        entity_id: Catastrophe identifier.
        cluster_ids: Constituent micro-cluster IDs.
        is_fused: Whether the constellation is a spurious unfolding.
        classification: The classification label carried by the constellation.

    Returns:
        An EffectiveCatastrophe sufficient for ownership tests.
    """
    return EffectiveCatastrophe(
        entity_id=entity_id,
        catastrophe_type="E-MSAD",
        classification=classification,
        is_fused=is_fused,
        fold_order=3,
        net_euler_index=-2,
        centroid=(0.0, 0.0, 1.0),
        centroid_unit=(0.0, 0.0, 1.0),
        angular_radius_rad=0.1,
        angular_radius_deg=math.degrees(0.1),
        barrier_depth=0.0,
        relative_barrier_depth=0.0,
        angular_span_deg=math.degrees(0.2),
        bifurcation_score=0.0,
        constituent_clusters=[_make_cluster(cid, -1) for cid in cluster_ids],
        all_members=[],
        boundary_ring=[],
    )


def test_no_claim_when_catastrophe_reduction_disabled() -> None:
    """With reduce_catastrophes off, nothing is claimed, so no cluster is stranded.

    This is the regression guard for the symmetric saddle-string case: a
    chi = -1 cluster absorbed into an unreduced catastrophe must stay eligible
    for standalone multi-cluster reduction.
    """
    cats = [_make_cat("CAT_1", ["MC21", "MC22"])]

    assert catastrophe_claimed_cluster_ids(cats, reduce_catastrophes=False) == set()


def test_claims_constituents_when_catastrophe_reduction_enabled() -> None:
    """With reduce_catastrophes on, every constellation member is claimed."""
    cats = [_make_cat("CAT_1", ["MC21", "MC22"]), _make_cat("CAT_2", ["MC34"])]

    claimed = catastrophe_claimed_cluster_ids(cats, reduce_catastrophes=True)

    assert claimed == {"MC21", "MC22", "MC34"}


def test_empty_catastrophe_list_claims_nothing() -> None:
    """Boundary case: no catastrophes yields an empty claim set for both toggle states."""
    assert catastrophe_claimed_cluster_ids([], reduce_catastrophes=False) == set()
    assert catastrophe_claimed_cluster_ids([], reduce_catastrophes=True) == set()


def test_shared_cluster_claimed_once() -> None:
    """A cluster appearing in two constellations yields a single claim entry."""
    cats = [_make_cat("CAT_1", ["MC21", "MC22"]), _make_cat("CAT_2", ["MC22"])]

    claimed = catastrophe_claimed_cluster_ids(cats, reduce_catastrophes=True)

    assert claimed == {"MC21", "MC22"}
    assert len(claimed) == 2


def test_default_ui_state_covers_every_reactive_control() -> None:
    """Every control wired to on_param_change has a canonical default.

    A missing key would leave that control un-reset by "Reset to Defaults". The
    key list is read from the module source rather than duplicated here, so the
    test cannot silently drift out of sync with the decorator.
    """
    src = pathlib.Path(sv.__file__).read_text(encoding="utf-8")
    marker = "    @state.change(\n"
    start = src.index(marker) + len(marker)
    end = src.index("    )\n", start)
    wired = set(re.findall(r'"([a-z_]+)"', src[start:end]))

    assert wired, "no keys parsed from the @state.change decorator"
    # selected_field is data-dependent (first discovered condensed array) and is
    # supplied at runtime, so it is intentionally absent from the static table.
    assert wired - set(DEFAULT_UI_STATE) == {"selected_field"}
    # Guard the parse itself: a silently empty or tiny match would vacuate the
    # assertion above.
    assert len(wired) >= 20


def test_default_ui_state_values_are_the_documented_baseline() -> None:
    """Pin the numeric/boolean baseline that the sliders and switches expect."""
    assert DEFAULT_UI_STATE["k_pitch"] == 2.0
    assert DEFAULT_UI_STATE["k_bridge"] == 3.0
    assert DEFAULT_UI_STATE["aggregate_strings"] is False
    assert DEFAULT_UI_STATE["cat_barrier_tol"] == 0.005
    assert DEFAULT_UI_STATE["cat_angular_tol_mult"] == 4.0
    assert DEFAULT_UI_STATE["nudge_max_pitch"] == 1.5
    assert DEFAULT_UI_STATE["num_contours"] == 15
    assert DEFAULT_UI_STATE["cp_glyph_scale"] == 1.0

    reduction_toggles = (
        "reduce_catastrophes",
        "reduce_multi_clusters",
        "nudge_cp_positions",
        "perfect_port_angles",
    )
    for key in reduction_toggles:
        assert DEFAULT_UI_STATE[key] is False, f"{key} must default to off"


def test_default_ui_state_is_not_mutated_by_callers() -> None:
    """The table is module-level; guard against accidental shared-state mutation."""
    snapshot = dict(DEFAULT_UI_STATE)
    working = dict(DEFAULT_UI_STATE)
    working["k_pitch"] = 4.0
    working["reduce_multi_clusters"] = True

    assert DEFAULT_UI_STATE == snapshot
    assert DEFAULT_UI_STATE["k_pitch"] == 2.0


@pytest.mark.parametrize("toggle", [True, False])
def test_claim_set_is_a_plain_set_of_str(toggle: bool) -> None:
    """The pipeline unions this result into a set[str]; keep the type stable."""
    cats = [_make_cat("CAT_1", ["MC7"])]

    claimed = catastrophe_claimed_cluster_ids(cats, reduce_catastrophes=toggle)

    assert isinstance(claimed, set)
    assert all(isinstance(cid, str) for cid in claimed)


CENTER = (1.261797, -0.000090, -0.000193)


class _FakeNudge:
    """Minimal stand-in for harmonic_nudge.NudgeResult."""

    initial_position = (0.0, 0.0, 0.412733)
    optimized_position = (0.01, 0.02, 0.412200)
    displacement_ang_deg = 2.5
    initial_energy = 1.0
    final_energy = 0.25
    iterations = 7
    converged = True


def test_raw_cp_to_ui_maps_position_to_world_frame() -> None:
    """A centre argument shifts the emitted position into molecular coordinates."""
    cp = CriticalPoint(
        vertex_id=5, cp_type="maximum", value=2.0, position=(0.0, 0.0, 1.0)
    )

    local = sv.raw_cp_to_ui(cp)
    world = sv.raw_cp_to_ui(cp, CENTER)

    assert local["position"] == [0.0, 0.0, 1.0]
    assert world["position"] == pytest.approx(
        [0.0 + CENTER[0], 0.0 + CENTER[1], 1.0 + CENTER[2]]
    )
    # Everything else is frame-independent.
    assert world["vertex_id"] == local["vertex_id"]
    assert world["value"] == local["value"]
    assert world["id_label"] == local["id_label"]


def test_effective_cp_to_ui_maps_every_coordinate() -> None:
    """Position, nudge pre/post and member positions all gain the centre."""
    members = [
        CriticalPoint(
            vertex_id=1, cp_type="maximum", value=1.0, position=(0.1, 0.0, 0.4)
        ),
        CriticalPoint(
            vertex_id=2, cp_type="saddle", value=0.5, position=(0.0, 0.1, 0.4)
        ),
    ]

    local = sv.effective_cp_to_ui(
        entity_id="E-SAD_MC1",
        eff_type="E-SAD",
        position=(0.0, 0.0, 0.412733),
        value=0.5,
        index=-1,
        members=members,
        origin_id="MC1",
        origin_label="Micro-Cluster",
        fold_order=2,
        nudge=_FakeNudge(),
    )
    world = sv.effective_cp_to_ui(
        entity_id="E-SAD_MC1",
        eff_type="E-SAD",
        position=(0.0, 0.0, 0.412733),
        value=0.5,
        index=-1,
        members=members,
        origin_id="MC1",
        origin_label="Micro-Cluster",
        fold_order=2,
        nudge=_FakeNudge(),
        center=CENTER,
    )

    assert world["position"] == pytest.approx(
        [0.0 + CENTER[0], 0.0 + CENTER[1], 0.412733 + CENTER[2]]
    )
    assert world["nudge"]["initial_position"] == pytest.approx(
        [c + CENTER[i] for i, c in enumerate(_FakeNudge.initial_position)]
    )
    assert world["nudge"]["optimized_position"] == pytest.approx(
        [c + CENTER[i] for i, c in enumerate(_FakeNudge.optimized_position)]
    )
    for mem_w, mem in zip(world["members"], members):
        assert mem_w["position"] == pytest.approx(
            [mem.position[i] + CENTER[i] for i in range(3)]
        )

    # The local call is unchanged, so the pre-centring behaviour still works.
    assert local["position"] == [0.0, 0.0, 0.412733]
    assert local["nudge"]["initial_position"] == list(_FakeNudge.initial_position)
    # Scalars are frame-independent.
    assert world["net_index"] == local["net_index"] == -1
    assert world["nudge"]["displacement_ang_deg"] == 2.5


def test_only_fused_catastrophes_are_fusable() -> None:
    """A split constellation must never be collapsed into one effective saddle.

    A split constellation is genuine physical symmetry breaking whose poles are
    distinct critical points. Collapsing it fabricates a saddle at a centroid
    that is not a critical point and hides the extrema its core cluster
    carries -- the defect that swallowed the global maximum on ethene4.
    """
    fused = _make_cat("CAT_1", ["MC1"], is_fused=True)
    split = _make_cat(
        "CAT_2",
        ["MC2"],
        is_fused=False,
        classification="physical_symmetry_breaking",
    )

    assert fusable_catastrophes([fused, split]) == [fused]


def test_no_catastrophes_are_fusable_when_all_are_split() -> None:
    """A set of only split constellations collapses to nothing."""
    split = _make_cat(
        "CAT_1",
        ["MC1"],
        is_fused=False,
        classification="physical_symmetry_breaking",
    )

    assert fusable_catastrophes([split]) == []


def test_split_catastrophe_does_not_claim_its_clusters() -> None:
    """A split constellation's clusters stay eligible for standalone reduction.

    Since the catastrophe stage no longer collapses them, claiming them would
    strand them: the standalone multi-cluster stage would skip them and no
    effective CP would ever be emitted, so the cluster's extrema would vanish
    from the reduced view entirely.
    """
    split = _make_cat(
        "CAT_1",
        ["MC3"],
        is_fused=False,
        classification="physical_symmetry_breaking",
    )

    assert catastrophe_claimed_cluster_ids([split], reduce_catastrophes=True) == set()


def test_fused_catastrophe_still_claims_its_clusters() -> None:
    """A fused constellation claims its constituents, so no duplicate ECP is emitted."""
    fused = _make_cat("CAT_1", ["MC3"], is_fused=True)

    assert catastrophe_claimed_cluster_ids([fused], reduce_catastrophes=True) == {"MC3"}


# ---------------------------------------------------------------------------
# Step 9 seed-set rule: which saddles get excision-ring launch seeds.
#
# The set is the *displayed* saddles. A raw saddle absorbed by reduction has no
# glyph, so seeding it launches separatrices from a point the reduction replaced;
# an effective saddle has a glyph and no vertex_id, so it can only be seeded from
# a loop. Both halves of that rule are pinned here.
# ---------------------------------------------------------------------------


def _raw_saddle(vertex_id: int, multiplicity: int = 1) -> CriticalPoint:
    """A raw saddle at the north pole with a given multiplicity."""
    return CriticalPoint(
        vertex_id=vertex_id,
        cp_type="saddle",
        value=0.5,
        position=(0.0, 0.0, 1.0),
        multiplicity=multiplicity,
    )


def _eff_sad(
    id_label: str,
    fold_order: int,
    position: tuple[float, float, float] = (0.0, 0.0, 1.0),
) -> dict[str, Any]:
    """An effective-saddle UI dict, world frame, as effective_cp_to_ui emits."""
    return {
        "id_label": id_label,
        "type": "E-SAD",
        "position": list(position),
        "value": 0.25,
        "fold_order": fold_order,
        "net_index": 1 - fold_order,
        "is_effective": True,
    }


def test_visible_raw_saddles_are_requested() -> None:
    """Every raw saddle that survived reduction gets a request."""
    reqs = sv.build_launch_requests(
        saddles=[_raw_saddle(10), _raw_saddle(20)],
        effective_cps=[],
        hidden_vertex_ids=set(),
        eff_ring_radius={},
        center=(0.0, 0.0, 0.0),
        delta_theta_mesh_rad=math.radians(1.606),
        k_pitch=2.0,
    )
    assert [r.saddle_id for r in reqs] == ["SAD_10", "SAD_20"]
    assert all(r.vertex_id >= 0 for r in reqs)


def test_hidden_raw_saddles_are_excluded() -> None:
    """A raw saddle collapsed into an effective one is not seeded."""
    reqs = sv.build_launch_requests(
        saddles=[_raw_saddle(10), _raw_saddle(20), _raw_saddle(30)],
        effective_cps=[],
        hidden_vertex_ids={20},
        eff_ring_radius={},
        center=(0.0, 0.0, 0.0),
        delta_theta_mesh_rad=math.radians(1.606),
        k_pitch=2.0,
    )
    assert [r.saddle_id for r in reqs] == ["SAD_10", "SAD_30"]


def test_effective_saddles_are_included() -> None:
    """The case the link scan could not serve: an E-SAD with no vertex_id."""
    reqs = sv.build_launch_requests(
        saddles=[],
        effective_cps=[_eff_sad("E-SAD_CAT_1", 3)],
        hidden_vertex_ids=set(),
        eff_ring_radius={"E-SAD_CAT_1": math.radians(6.0)},
        center=(0.0, 0.0, 0.0),
        delta_theta_mesh_rad=math.radians(1.606),
        k_pitch=2.0,
    )
    assert len(reqs) == 1
    req = reqs[0]
    assert req.saddle_id == "E-SAD_CAT_1"
    assert req.vertex_id == -1
    assert req.expected_k == 3
    # The constellation's own ring radius, not the lone-CP rule.
    assert req.a0_rad == pytest.approx(math.radians(6.0))


def test_extrema_are_not_seeded() -> None:
    """E-MAX and E-MIN carry no separatrix; only E-SAD rows become requests."""
    eff = [_eff_sad("E-SAD_CAT_1", 3), _eff_sad("E-MAX_MC7", 0)]
    eff[1]["type"] = "E-MAX"
    reqs = sv.build_launch_requests(
        saddles=[],
        effective_cps=eff,
        hidden_vertex_ids=set(),
        eff_ring_radius={},
        center=(0.0, 0.0, 0.0),
        delta_theta_mesh_rad=math.radians(1.606),
        k_pitch=2.0,
    )
    assert [r.saddle_id for r in reqs] == ["E-SAD_CAT_1"]


def test_expected_k_from_multiplicity_for_raw_saddles() -> None:
    """A raw saddle of multiplicity mu has fold order k = mu + 1."""
    reqs = sv.build_launch_requests(
        saddles=[_raw_saddle(1, multiplicity=1), _raw_saddle(2, multiplicity=2)],
        effective_cps=[],
        hidden_vertex_ids=set(),
        eff_ring_radius={},
        center=(0.0, 0.0, 0.0),
        delta_theta_mesh_rad=math.radians(1.606),
        k_pitch=2.0,
    )
    assert [r.expected_k for r in reqs] == [2, 3]
    assert [r.multiplicity for r in reqs] == [1, 2]


def test_lone_saddle_radius_follows_the_excision_rule() -> None:
    """A lone raw saddle's a0 is max(margin, k_pitch * dtheta / 2)."""
    dtheta = math.radians(1.606)
    reqs = sv.build_launch_requests(
        saddles=[_raw_saddle(1)],
        effective_cps=[],
        hidden_vertex_ids=set(),
        eff_ring_radius={},
        center=(0.0, 0.0, 0.0),
        delta_theta_mesh_rad=dtheta,
        k_pitch=2.0,
    )
    # k_pitch * dtheta / 2 = 1.606 deg beats the 1.5 deg margin.
    assert reqs[0].a0_rad == pytest.approx(dtheta)

    # A small pitch makes the fixed margin win instead.
    small = sv.build_launch_requests(
        saddles=[_raw_saddle(1)],
        effective_cps=[],
        hidden_vertex_ids=set(),
        eff_ring_radius={},
        center=(0.0, 0.0, 0.0),
        delta_theta_mesh_rad=math.radians(0.3),
        k_pitch=2.0,
    )
    assert small[0].a0_rad == pytest.approx(math.radians(1.5))


def test_k_pitch_slider_moves_the_initial_radius() -> None:
    """The clustering pitch is a live control and must reach the ring rule."""
    dtheta = math.radians(1.606)
    reqs = sv.build_launch_requests(
        saddles=[_raw_saddle(1)],
        effective_cps=[],
        hidden_vertex_ids=set(),
        eff_ring_radius={},
        center=(0.0, 0.0, 0.0),
        delta_theta_mesh_rad=dtheta,
        k_pitch=6.0,
    )
    assert reqs[0].a0_rad == pytest.approx(3.0 * dtheta)


def test_effective_centre_is_mapped_to_the_centred_frame() -> None:
    """UI positions are world frame; the ring search works in the centred frame."""
    centre = (10.0, -3.0, 0.5)
    reqs = sv.build_launch_requests(
        saddles=[],
        effective_cps=[_eff_sad("E-SAD_CAT_1", 3, position=(10.0, -2.0, 0.5))],
        hidden_vertex_ids=set(),
        eff_ring_radius={},
        center=centre,
        delta_theta_mesh_rad=math.radians(1.606),
        k_pitch=2.0,
    )
    assert reqs[0].centre == pytest.approx((0.0, 1.0, 0.0))


def test_effective_falls_back_to_the_lone_rule_without_a_radius() -> None:
    """An effective saddle with no recorded ring radius still gets a sane a0."""
    dtheta = math.radians(1.606)
    reqs = sv.build_launch_requests(
        saddles=[],
        effective_cps=[_eff_sad("E-SAD_CAT_9", 2)],
        hidden_vertex_ids=set(),
        eff_ring_radius={},
        center=(0.0, 0.0, 0.0),
        delta_theta_mesh_rad=dtheta,
        k_pitch=2.0,
    )
    assert reqs[0].a0_rad == pytest.approx(dtheta)


def test_raw_saddles_precede_effective_ones() -> None:
    """Ordering is stable, so the sidebar list and the seed list agree."""
    reqs = sv.build_launch_requests(
        saddles=[_raw_saddle(7)],
        effective_cps=[_eff_sad("E-SAD_CAT_1", 3)],
        hidden_vertex_ids=set(),
        eff_ring_radius={},
        center=(0.0, 0.0, 0.0),
        delta_theta_mesh_rad=math.radians(1.606),
        k_pitch=2.0,
    )
    assert [r.saddle_id for r in reqs] == ["SAD_7", "E-SAD_CAT_1"]
