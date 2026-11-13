"""Basin v11 viewer logic: occlusion, draw order, connection class, routing.

Minimum basin boundaries shipped in v10. v11 draws the same rule from the other
end -- the closed separatrix ring encircling a MAXIMUM -- and both rings can be
on screen at once, which forces four things that a single-polarity viewer never
needed:

* one occlusion rule between two attractors (:func:`occludes`);
* one draw order computed over the FULL set, before anything is drawn
  (:func:`basin_draw_order`);
* one classification of a group's arcs that feeds BOTH the render gate and the
  warning text, so the message cannot contradict the screen
  (:func:`classify_basin_connection`);
* one toggle routing, so all four switches skip the pipeline rebuild that erased
  the rings in v10 (:data:`BASIN_ONLY_TOGGLES`, :func:`is_basin_only_change`).

These are the module-level half of the feature and they are pure -- no trame, no
VTK, no browser. The nested half (the actor groups, the caches) is pinned by the
source guards at the bottom of this file, in the idiom the rest of the viewer
suite uses.
"""

from __future__ import annotations

import ast
import pathlib
from typing import Any

import pytest

import gba_topology2.src.sphere_viewer as sv
from gba_topology2.src.sphere_viewer import (
    BASIN_ONLY_TOGGLES,
    MAX_BASIN_GATE_DEG,
    MAX_BASIN_RING_COLOR,
    MIN_BASIN_GATE_DEG,
    MIN_BASIN_RING_COLOR,
    RAYCAST_OCCLUDERS,
    basin_draw_order,
    basin_ring_color,
    basin_ring_gate_deg,
    basin_stats_template,
    classify_basin_connection,
    is_basin_only_change,
    occludes,
)

SOURCE = pathlib.Path(sv.__file__)
MIN_KEY = "show_basin_for_selected_minimum"
MIN_ALL_KEY = "show_all_minimum_basins"
MAX_KEY = "show_basin_for_selected_maximum"
MAX_ALL_KEY = "show_all_maximum_basins"


def _ring(
    polarity: str, sep: float | None = 12.0, endpoint: str = "X"
) -> dict[str, Any]:
    """One projected RING row: an arc that closes on an endpoint."""
    return {
        "polarity": polarity,
        "kind": "ring",
        "separation_deg": sep,
        "endpoint_id": endpoint,
        "occluders": ["maximum"] if polarity == "minimum" else [],
    }


def _ray(polarity: str, hit: str = "saddle") -> dict[str, Any]:
    """One projected RAY row: a branch that left and never closed."""
    return {
        "polarity": polarity,
        "kind": "ray",
        "separation_deg": None,
        "endpoint_id": None,
        "occluders": [hit],
    }


# ---------------------------------------------------------------------------
# A. The occlusion rule
# ---------------------------------------------------------------------------


class TestOccludes:
    def test_a_group_never_occludes_itself(self) -> None:
        """Same attractor on both sides: a maximum's ring is not hidden by other
        maxima's rings. Only the OPPOSITE shell can be in the way."""
        assert occludes("maximum", "maximum") is False
        assert occludes("minimum", "minimum") is False

    def test_a_maximums_shell_hides_a_minimums_ring(self) -> None:
        """A maximum's stable shell is the OUTER one, so it sits between the
        camera and a valley's boundary."""
        assert occludes("maximum", "minimum") is True

    def test_a_minimums_shell_does_not_hide_a_maximums_ring(self) -> None:
        """The asymmetry, and the reason the two sides are not mirror images: a
        minimum's stable shell is the INNER one -- a hill is not inside a valley
        -- so it cannot screen a hill's ring."""
        assert occludes("minimum", "maximum") is False

    def test_a_saddle_hit_occludes_a_maximums_ring(self) -> None:
        """The ascent-side mirror of the saddle raycast that already occludes a
        minimum's basin."""
        assert occludes("saddle", "maximum") is True

    def test_a_saddle_hit_occludes_a_minimums_ring(self) -> None:
        assert occludes("saddle", "minimum") is True

    def test_the_raycast_occluders_are_the_attractors_a_ray_can_stop_on(self) -> None:
        """A ray that stops ON an attractor is screened by that attractor's
        shell; a ray that flies past everything is screened by nothing."""
        assert set(RAYCAST_OCCLUDERS) == {"saddle", "maximum"}

    def test_an_unknown_occluder_hides_nothing_on_the_maximum_side(self) -> None:
        """Conservative default: a hit kind we do not know is not a shell."""
        assert occludes("minimum", "maximum") is False
        assert occludes("", "maximum") is False


# ---------------------------------------------------------------------------
# B. The draw order, decided over the full set
# ---------------------------------------------------------------------------


class TestBasinDrawOrder:
    def test_minimum_first(self) -> None:
        """The group added first is underneath. The minimum side has been on
        screen since v10, so it keeps the primacy slot and the maximum's ring --
        which occludes it -- is added last."""
        assert basin_draw_order([_ring("minimum"), _ring("maximum")]) == ["maximum"]

    def test_a_group_with_no_rings_is_not_visible(self) -> None:
        assert basin_draw_order([_ring("minimum")]) == ["minimum"]
        assert basin_draw_order([_ring("maximum")]) == ["maximum"]

    def test_rays_never_occlude(self) -> None:
        """Rays are decorations. A ray-only maximum set leaves the minimum's
        rings on screen -- the v10 output is unchanged by v11."""
        assert basin_draw_order([_ring("minimum"), _ray("maximum")]) == ["minimum"]

    def test_an_all_ray_group_occludes_nothing(self) -> None:
        """An open hill does not hide a valley: no rings, so the group is not
        even a candidate, and the minimum keeps its rings."""
        assert basin_draw_order(
            [_ring("minimum"), _ray("maximum"), _ray("maximum")]
        ) == ["minimum"]

    def test_no_rows_at_all_draws_nothing(self) -> None:
        assert basin_draw_order([]) == []

    def test_the_decision_is_made_over_the_full_set_not_per_group(self) -> None:
        """Row order must not matter: emitting the maximum first is the same
        scene as emitting it last. Interleaving the decision with the drawing is
        what made v10's rings vanish, so the order of INPUTS is pinned."""
        forward = [_ring("minimum"), _ring("maximum")]
        backward = list(reversed(forward))
        assert basin_draw_order(forward) == basin_draw_order(backward) == ["maximum"]

    def test_more_minima_do_not_outvote_one_maximum(self) -> None:
        """Presence, not count: ten valley rings are still behind one hill's
        shell. The shipped rule is geometric; a count-based rule would make the
        visible set depend on how many CPs the classifier happened to find."""
        rows = [_ring("minimum", endpoint=f"MIN_{i}") for i in range(10)]
        rows.append(_ring("maximum"))
        assert basin_draw_order(rows) == ["maximum"]


# ---------------------------------------------------------------------------
# C. The gates and colours, one per polarity
# ---------------------------------------------------------------------------


class TestGatesAndColors:
    def test_the_maximum_gate_is_the_number_v10_invented(self) -> None:
        """One number judges a hill and a valley."""
        assert MAX_BASIN_GATE_DEG == pytest.approx(0.5)
        assert basin_ring_gate_deg("maximum") == pytest.approx(0.5)

    def test_the_minimum_side_is_ungated(self) -> None:
        """Giving the minimum side a real gate would change v10's rendered
        output. None means every separation passes, which is v10's behaviour."""
        assert MIN_BASIN_GATE_DEG is None
        assert basin_ring_gate_deg("minimum") is None

    def test_the_drawer_and_the_classifier_read_one_gate(self) -> None:
        """The pair cannot drift: there is one lookup, not a constant at each
        call site."""
        assert basin_ring_gate_deg("minimum") is MIN_BASIN_GATE_DEG
        assert basin_ring_gate_deg("maximum") == MAX_BASIN_GATE_DEG

    def test_an_unknown_polarity_is_treated_as_the_minimum_side(self) -> None:
        """Fail to the permissive gate: an unrecognised label draws, it does not
        withhold geometry."""
        assert basin_ring_gate_deg("saddle") is None

    def test_each_polarity_rings_in_the_colour_of_its_centre(self) -> None:
        """Purple hill, gold valley, and the purple is the palette's maximum
        colour so a ring reads as the entity it bounds."""
        assert basin_ring_color("maximum") == MAX_BASIN_RING_COLOR
        assert basin_ring_color("minimum") == MIN_BASIN_RING_COLOR
        assert MAX_BASIN_RING_COLOR == (0.294, 0.0, 0.514)
        assert MIN_BASIN_RING_COLOR == (1.0, 0.84, 0.0)

    def test_the_two_rings_are_distinguishable(self) -> None:
        """Two full layers on one sphere: if the colours matched, the screen
        would show one undiagnosable tangle."""
        assert MAX_BASIN_RING_COLOR != MIN_BASIN_RING_COLOR

    def test_both_rings_lift_off_the_sphere(self) -> None:
        """The lift is shared by the one geometry builder, so a z-fight between
        the two polarities' rings is structurally impossible."""
        src = _body_src("_basin_group_geometry")
        assert "radial_offset" in src
        assert src.count("1.020") == 1


# ---------------------------------------------------------------------------
# D. The stats template, built once for both panels
# ---------------------------------------------------------------------------


class TestStatsTemplate:
    def test_each_polarity_reads_its_own_stats_block(self) -> None:
        for polarity in ("minimum", "maximum"):
            assert f"basin_group_stats['{polarity}']" in basin_stats_template(polarity)

    def test_the_templates_differ_only_by_polarity(self) -> None:
        """One template, two instantiations: a field added on one panel appears
        on the other, which is the point of not inlining the Vue string twice."""
        lo = basin_stats_template("minimum")
        hi = basin_stats_template("maximum")
        assert len(lo) == len(hi)
        for field in ("status", "n_rings", "n_rays", "min_sep_text", "gate_text"):
            assert field in lo and field in hi, field

    def test_the_active_count_uses_the_policys_own_list(self) -> None:
        assert "active_min_basins" in basin_stats_template("minimum")
        assert "active_max_basins" in basin_stats_template("maximum")


# ---------------------------------------------------------------------------
# E. classify_basin_connection -- the render gate and the text, in one function
# ---------------------------------------------------------------------------


class TestClassifyBasinConnection:
    def test_no_arcs_is_not_a_basin_at_all(self) -> None:
        """The empty case names the reason, because the drawer shows it: an
        unclassified maximum's group is empty, and that is NOT 'the hill is
        open'."""
        out = classify_basin_connection("minimum", [], MIN_BASIN_GATE_DEG)
        assert out["status"] == "no_basin"
        assert out["n_rings"] == 0 and out["n_rays"] == 0
        assert out["unbounded"] is False
        assert out["warnings"] == []
        assert "other polarity" in out["reason"]

    def test_the_ring_gate_is_visible_in_the_payload(self) -> None:
        """The asymmetry must be reportable, not folklore: the number that gated
        the drawing travels with the summary."""
        assert (
            classify_basin_connection("minimum", [_ring("minimum")], None)[
                "ring_gate_deg"
            ]
            is None
        )
        assert classify_basin_connection("maximum", [_ring("maximum")], 0.5)[
            "ring_gate_deg"
        ] == pytest.approx(0.5)

    def test_rings_tightly_connected_warn_on_the_maximum_side(self) -> None:
        """Plan acceptance: a bunched ring pair WARNS, it does not withhold the
        ring -- the gate reports, the drawer draws.

        The shipped gate is 0.5 deg, so "tightly connected" means every gap
        between consecutive rings is under half a degree.
        """
        rows = [_ring("maximum", sep=10.0), _ring("maximum", sep=10.3)]
        out = classify_basin_connection("maximum", rows, MAX_BASIN_GATE_DEG)
        assert out["min_separation_deg"] == pytest.approx(0.3)
        assert out["status"] == "bounded"
        gate_warnings = [w for w in out["warnings"] if "gate" in w]
        assert gate_warnings, out["warnings"]
        assert "0.3" in gate_warnings[0]

    def test_a_spread_ring_pair_is_quiet(self) -> None:
        """Rings on opposite sides of the hill close a basin: no gate warning.

        The ring-only set still carries the "no rays" note, which is a different
        statement (the separatrix set is incomplete), so this asserts on the gate
        warnings only.
        """
        rows = [_ring("maximum", sep=10.0), _ring("maximum", sep=30.0)]
        out = classify_basin_connection("maximum", rows, MAX_BASIN_GATE_DEG)
        assert [w for w in out["warnings"] if "gate" in w] == []
        assert out["status"] == "bounded"
        assert out["unbounded"] is False

    def test_the_gate_never_withholds_a_basin(self) -> None:
        """The gate's whole job is the warning: a tightly-connected hill is still
        a basin on screen (it is BOUNDED BY MINIMA, and the minima are drawn)."""
        rows = [_ring("maximum", sep=10.0), _ring("maximum", sep=10.2)]
        out = classify_basin_connection("maximum", rows, MAX_BASIN_GATE_DEG)
        assert out["status"] == "bounded"
        assert out["n_rings"] == 2

    def test_one_ring_cannot_be_measured_against_itself(self) -> None:
        """The gap statistic is measured between PAIRS; a lone ring has none, so
        the gate cannot fire and the text says n/a."""
        out = classify_basin_connection(
            "maximum", [_ring("maximum")], MAX_BASIN_GATE_DEG
        )
        assert out["min_separation_deg"] is None
        assert out["min_sep_text"] == "n/a"
        assert [w for w in out["warnings"] if "gate" in w] == []

    def test_raycast_only_is_bounded_with_a_warning_on_the_minimum_side(self) -> None:
        """v10's output, pinned: a raycast-only valley ring is drawn, with the
        'not closed' note."""
        out = classify_basin_connection(
            "minimum", [_ray("minimum")], MIN_BASIN_GATE_DEG
        )
        assert out["status"] == "bounded"
        assert out["n_rays"] == 1
        assert any("raycast arcs only" in w for w in out["warnings"])

    def test_raycast_only_is_unbounded_on_the_maximum_side(self) -> None:
        """A hill with no ring has no closed boundary: that is what 'unbounded'
        means, and the drawer draws nothing for it."""
        out = classify_basin_connection(
            "maximum", [_ray("maximum")], MAX_BASIN_GATE_DEG
        )
        assert out["status"] == "unbounded"
        assert out["unbounded"] is True
        assert out["n_rings"] == 0

    def test_a_rayless_ring_set_is_bounded_and_says_its_boundary_is_open(self) -> None:
        """A ring with no rays closes the basin but reports that the separatrix
        set is incomplete -- both facts at once, one status."""
        out = classify_basin_connection(
            "minimum", [_ring("minimum")], MIN_BASIN_GATE_DEG
        )
        assert out["status"] == "bounded"
        assert any("no rays" in w for w in out["warnings"])

    def test_the_gate_ignores_rays(self) -> None:
        """The gate measures separation between RINGS; rays have no separation,
        and averaging them in would invent gaps of None."""
        rows = [_ring("maximum", sep=10.0), _ray("maximum"), _ring("maximum", sep=10.1)]
        out = classify_basin_connection("maximum", rows, MAX_BASIN_GATE_DEG)
        assert out["min_separation_deg"] == pytest.approx(0.1)
        assert out["n_rings"] == 2 and out["n_rays"] == 1

    def test_the_minimum_side_is_never_declared_unbounded(self) -> None:
        """v10's parity: no gate, and a raycast-only basin stays bounded."""
        cases: list[list[dict[str, Any]]] = [
            [],
            [_ray("minimum")],
            [_ring("minimum")],
            [_ray("minimum"), _ring("minimum")],
        ]
        for rows in cases:
            out = classify_basin_connection("minimum", rows, MIN_BASIN_GATE_DEG)
            assert out["unbounded"] is False, rows
            assert out["status"] in ("bounded", "no_basin"), rows

    def test_the_gate_reads_the_widest_gap_between_consecutive_rings(self) -> None:
        """The statistic is the MAX of the consecutive gaps, not the min.

        The question a closure gate asks is "is there any room in this ring set
        at all?" A set with one wide gap encloses the centre; a set whose every
        gap is sub-gate is a bunch of near-coincident arcs, which bounds nothing.
        Taking the min would call the second set fine as soon as one pair was
        far apart -- and the field is named ``min_separation_deg``, so the name
        reads against the value. Pinned here for exactly that reason.
        """
        rows = [
            _ring("maximum", sep=30.0),
            _ring("maximum", sep=5.0),
            _ring("maximum", sep=5.2),
            _ring("maximum", sep=40.0),
        ]
        out = classify_basin_connection("maximum", rows, MAX_BASIN_GATE_DEG)
        # sorted seps: 5.0, 5.2, 30.0, 40.0 -> gaps 0.2, 24.8, 10.0 -> max 24.8
        assert out["min_separation_deg"] == pytest.approx(24.8)
        assert [w for w in out["warnings"] if "gate" in w] == []

    def test_a_bunch_of_rings_is_the_case_the_gate_is_for(self) -> None:
        rows = [_ring("maximum", sep=float(i) * 0.1) for i in range(5)]
        out = classify_basin_connection("maximum", rows, MAX_BASIN_GATE_DEG)
        assert out["min_separation_deg"] == pytest.approx(0.1)
        assert [w for w in out["warnings"] if "gate" in w], out["warnings"]

    def test_the_text_fields_are_preformatted(self) -> None:
        """A Vue template cannot format a float, and a bare None renders as
        'None' on a diagnostics panel."""
        out = classify_basin_connection("maximum", [_ring("maximum", sep=9.0)], 0.5)
        assert out["gate_text"] == "0.50°"
        assert out["min_sep_text"] == "n/a"
        pair = classify_basin_connection(
            "maximum", [_ring("maximum", sep=9.0), _ring("maximum", sep=9.25)], 0.5
        )
        assert pair["min_sep_text"] == "0.250°"


# ---------------------------------------------------------------------------
# F. Toggle routing: four switches, one behaviour
# ---------------------------------------------------------------------------


class TestToggleRouting:
    def test_all_four_basin_switches_skip_the_pipeline_rebuild(self) -> None:
        """A basin switch is a pure overlay: a rebuild clears the caches and
        re-emits the selection arcs over the ring (v10's incident)."""
        assert set(BASIN_ONLY_TOGGLES) == {MIN_KEY, MIN_ALL_KEY, MAX_KEY, MAX_ALL_KEY}
        for key in BASIN_ONLY_TOGGLES:
            assert is_basin_only_change({key}), key

    def test_every_basin_switch_together_still_skips(self) -> None:
        assert is_basin_only_change(set(BASIN_ONLY_TOGGLES))

    def test_a_mixed_batch_rebuilds(self) -> None:
        """A batch that also changes a pipeline control must not skip the
        rebuild -- the non-basin control in the same flush is real work."""
        assert not is_basin_only_change({MAX_ALL_KEY, "k_pitch"})

    def test_an_empty_batch_is_not_a_basin_change(self) -> None:
        assert not is_basin_only_change(set())

    def test_the_maximum_toggles_are_declared_on_state(self) -> None:
        """A switch with no state key is invisible: the v_model binds to nothing
        and the card renders dead."""
        src = SOURCE.read_text(encoding="utf-8")
        for key in (MAX_KEY, MAX_ALL_KEY):
            assert f"state.{key} = False" in src, f"no state declaration for {key}"

    def test_the_maximum_card_is_declared(self) -> None:
        src = SOURCE.read_text(encoding="utf-8")
        for key in (MAX_KEY, MAX_ALL_KEY):
            assert f'v_model=("{key}",)' in src, f"no card for {key}"

    def test_the_maximum_switches_are_registered_as_controls(self) -> None:
        """Unregistered keys silently disable the parameter SAVE (Unit B).

        PARAM_CONTROL_KEYS is derived from DEFAULT_UI_STATE, so the real gate is
        PARAM_BOOL_KEYS: a bool missing from it arrives from JSON as a string and
        reads truthy, which turns "off" into "on" for a switch whose whole job is
        to stay off.
        """
        from gba_topology2.src.params_schema import (
            DEFAULT_UI_STATE,
            PARAM_BOOL_KEYS,
            PARAM_CONTROL_KEYS,
        )

        for key in (MAX_KEY, MAX_ALL_KEY):
            assert key in DEFAULT_UI_STATE, key
            assert key in PARAM_CONTROL_KEYS, key
            assert key in PARAM_BOOL_KEYS, key

    def test_the_maximum_switches_default_off(self) -> None:
        """The DEFAULT_UI_STATE entry IS the pipeline-snapshot default: a
        snapshot with no UI block is a pipeline run, and the v10 incident's
        symptom was rings appearing unasked."""
        from gba_topology2.src.params_schema import DEFAULT_UI_STATE

        assert DEFAULT_UI_STATE[MAX_KEY] is False
        assert DEFAULT_UI_STATE[MAX_ALL_KEY] is False
        assert DEFAULT_UI_STATE[MIN_KEY] is False
        assert DEFAULT_UI_STATE[MIN_ALL_KEY] is False


# ---------------------------------------------------------------------------
# G. Structural guards on the nested half
# ---------------------------------------------------------------------------


def _tree() -> ast.Module:
    return ast.parse(SOURCE.read_text(encoding="utf-8"))


def _node(name: str) -> ast.FunctionDef:
    for node in ast.walk(_tree()):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} is not defined in {SOURCE.name}")


def _body_src(name: str) -> str:
    segment = ast.get_source_segment(SOURCE.read_text(encoding="utf-8"), _node(name))
    assert segment is not None, name
    return segment


def _calls_in(name: str) -> set[str]:
    out: set[str] = set()
    for node in ast.walk(_node(name)):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                out.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                out.add(node.func.attr)
    return out


def _assigns_in(name: str) -> set[str]:
    out: set[str] = set()
    for node in ast.walk(_node(name)):
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Attribute):
                    out.add(tgt.attr)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Attribute):
            out.add(str(node.target.attr))
    return out


class TestViewerWiring:
    def test_the_rebuild_refreshes_the_pair(self) -> None:
        """v10's fix, widened: refreshing one group from a rebuild is half a
        refresh the moment two groups order against each other."""
        calls = _calls_in("update_topology")
        assert "_refresh_basin_overlays" in calls
        assert "compute_and_show_basin" not in calls
        assert "compute_and_show_max_basin" not in calls

    def test_the_toggle_handler_refreshes_the_pair(self) -> None:
        calls = _calls_in("on_param_change")
        assert "_refresh_basin_overlays" in calls
        assert "compute_and_show_basin" not in calls

    def test_the_refresh_covers_both_groups(self) -> None:
        calls = _calls_in("_refresh_basin_overlays")
        assert {"compute_and_show_basin", "compute_and_show_max_basin"} <= calls

    def test_both_computers_project_and_apply_the_same_group_pass(self) -> None:
        """Each side writes its OWN arc list, then hands the pair to the one
        occlusion pass. A side that drew directly would bypass the order."""
        for name, arcs in (
            ("compute_and_show_basin", "min_basin_arcs"),
            ("compute_and_show_max_basin", "max_basin_arcs"),
        ):
            assigned = _assigns_in(name)
            assert arcs in assigned, f"{name} does not write {arcs}"
            assert "_apply_basin_groups" in _calls_in(name), name
            assert "_project_basin_arcs" in _calls_in(name), name

    def test_the_computers_share_the_rehydrators(self) -> None:
        """Three builders, one implementation each: the centre pool, the ascent
        table, the descent table. Two copies is where a null-safety fix lands on
        one side only."""
        for name in ("compute_and_show_basin", "compute_and_show_max_basin"):
            calls = _calls_in(name)
            assert "_basin_center_objects" in calls, name
            assert "_rehydrate_ascent_rows" in calls, name
            assert "_rehydrate_descent_rows" in calls, name

    def test_the_maximum_side_reads_the_descent_table_for_geometry(self) -> None:
        """A maximum's boundary is traced DOWNHILL from its bounding saddles, so
        the arc points come from the descent (saddle->minimum) table. Reading the
        ascent table for geometry is the two-copies bug this pins out."""
        src = _body_src("compute_and_show_max_basin")
        assert "min_connections_list" in src
        assert "max_connections_list" not in src

    def test_the_minimum_side_reads_the_ascent_table_for_geometry(self) -> None:
        src = _body_src("compute_and_show_basin")
        assert "max_connections_list" in src

    def test_occlusion_is_an_actor_swap_not_a_visibility_flip(self) -> None:
        """VTK draws actors in the order they were added. A visibility flip
        leaves the hidden group in the depth buffer -- the exact artefact the
        rule exists to prevent. The renderer is SHARED with the surface, the
        ports and the CP overlay, so the removal is per-group, never
        RemoveAllViewProps."""
        src = _body_src("_set_basin_visible")
        assert "RemoveActor" in src
        assert "AddActor" in src
        assert "RemoveAllViewProps" not in src
        assert "SetVisibility" not in src

    def test_the_groups_are_closure_locals_not_state(self) -> None:
        """vtkActors are not JSON-safe, and `state` is serialized into SAVE
        PARAMETERS."""
        assert "vtkActor" in _body_src("run_sphere_viewer")
        src = SOURCE.read_text(encoding="utf-8")
        assert "self._min_basin_actors" not in src
        assert "self._max_basin_actors" not in src

    def test_the_class_is_reported_from_the_render_truth(self) -> None:
        """The stats name what was DRAWN, so a group whose rows exist in the
        table but were dropped by occlusion is not reported as visible."""
        src = _body_src("_apply_basin_groups")
        assert "visible[" in src
        assert "classify_basin_connection" in src
        assert "basin_draw_order" in src

    def test_active_ids_come_from_the_drawn_rows(self) -> None:
        src = _body_src("_apply_basin_groups")
        assert "active_min_basins" in src
        assert "active_max_basins" in src
        assert "endpoint_id" in src

    def test_the_selection_handlers_do_not_write_either_group(self) -> None:
        """v10's incident, generalized.

        The selection overlay used to overwrite the basin group's arc table, and
        the rings vanished a moment after appearing. A selection refresh goes
        through the group computers, which read the toggles and can therefore
        clear -- it never writes the arc lists by hand.
        """
        for name in (
            "select_basin_center",
            "select_basin_center_from_list",
            "clear_basin_center_selection",
        ):
            assigned = _assigns_in(name)
            assert not assigned & {"min_basin_arcs", "max_basin_arcs"}, name
            assert "basin_arcs" not in assigned, name

    def test_selection_refreshes_under_the_toggles_and_clears_without_them(
        self,
    ) -> None:
        """The guarded form of the pair refresh: selecting a centre with a switch
        on draws that centre's ring; selecting with both switches off must still
        run the occlusion pass, which removes every ring."""
        src = _body_src("select_basin_center")
        assert "compute_and_show_basin" in src
        assert "compute_and_show_max_basin" in src
        assert "_apply_basin_groups" in src

    def test_an_empty_group_removes_its_actors(self) -> None:
        """A recompute that yields no rings must not leave the previous
        parameter set's ring on screen -- the failure v10 spent a release
        eliminating."""
        src = _body_src("_set_basin_visible")
        assert "if not rows:" in src

    def test_the_export_writes_both_serializers(self) -> None:
        """A parameter-only file is enough to reproduce the ring (v10's lesson):
        the topology each side draws is exported by the function that draws it."""
        assert "serialize_minimum_basins" in _body_src("compute_and_show_basin")
        assert "serialize_maximum_basins" in _body_src("compute_and_show_max_basin")


def test_module_exports_the_basin_helpers() -> None:
    """The report side and the tests read these names; keep them importable."""
    for name in (
        "occludes",
        "basin_draw_order",
        "classify_basin_connection",
        "basin_ring_gate_deg",
        "basin_ring_color",
        "basin_stats_template",
        "is_basin_only_change",
    ):
        assert callable(getattr(sv, name)), name
