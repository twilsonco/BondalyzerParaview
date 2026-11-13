"""Unit tests for the sphere_viewer "Save Parameters" diagnostic export.

Covers the JSON parameter snapshot helpers added to ``sphere_viewer``:
1. ``save_params_to_dict`` -- builds the JSON-safe snapshot from a duck-typed
   state object (the trame ``state`` satisfies the same protocol).
2. ``load_params_from_dict`` -- validates and coerces a saved payload back
   into a flat dict safe for ``state.update()``.
3. ``default_params_path`` -- the deterministic output naming.

The control-key coverage test reads the module source the same way
``test_sphere_viewer_reduction.py`` does, so the export cannot silently drift
out of sync with the controls wired to ``@state.change``.
"""

from __future__ import annotations

import json
import pathlib
import re
from typing import Any

import pytest

import gba_topology2.src.sphere_viewer as sv
from gba_topology2.src.sphere_viewer import (
    DEFAULT_UI_STATE,
    PARAMS_SCHEMA_VERSION,
    default_params_path,
    load_params_from_dict,
    make_ridge_probe_stats,
    save_params_to_dict,
)


class _FakeState:
    """Duck-typed stand-in for the trame state object.

    Args:
        values: Attribute names mapped to their values.
    """

    # Attributes the tests touch directly; the rest live in __dict__ via
    # setattr in __init__.
    morse_counts: dict[str, Any]
    k_pitch: float

    def __init__(self, values: dict[str, Any]) -> None:
        for key, value in values.items():
            setattr(self, key, value)


ATOM_INFO: dict[str, Any] = {
    "atom_number": 1,
    "atom_symbol": "C",
    "atom_name": "C1",
    "num_nodes": 100,
    "center": [0.0, 0.0, 0.0],
    "field_items": [
        {"title": "Electron Density", "value": "Electron Density (condensed)"}
    ],
}


def _make_state(**overrides: Any) -> _FakeState:
    """Build a fake state seeded with the canonical defaults.

    Args:
        overrides: Keys whose default value should be replaced.

    Returns:
        A _FakeState carrying every control, metric, context and selection key.
    """
    values: dict[str, Any] = dict(DEFAULT_UI_STATE)
    values["selected_field"] = "Electron Density (condensed)"
    values["geom_info"] = {
        "num_vertices": 100,
        "num_triangles": 196,
        "sphere_radius": 1.5,
        "mesh_pitch_deg": 2.0,
        "center": [0.0, 0.0, 0.0],
        "center_offset": 0.0,
        "radius_source": "metadata",
    }
    values["morse_counts"] = {
        "minima": 2,
        "maxima": 3,
        "saddles": 4,
        "euler": 2,
        "valid": True,
    }
    values["display_counts"] = {
        "minima": 2,
        "maxima": 3,
        "saddles": 4,
        "euler": 2,
        "valid": True,
    }
    values["reduction_stats"] = {"n_effective_cps": 0}
    values["launch_stats"] = {"n_saddles_seeded": 4, "max_fold_order": 2}
    values["connection_stats"] = {"n_branches": 8, "n_connected": 8}
    # The probe's measurement, in the shape ``make_ridge_probe_stats`` writes. The
    # exporter skips a metric that is None, so a fake state that never carried it
    # silently exported a subset of PARAM_METRIC_KEYS.
    values["ridge_probe_stats"] = make_ridge_probe_stats([], on=False)
    # v9: the arc table, its CP scoring targets, and the derived basin topology.
    # The exporter skips a None metric, so a state that never carried these
    # exports a subset and the strict export test fails on the key set.
    values["max_connections_list"] = []
    values["basin_centers_list"] = []
    values["connection_graph"] = {}
    values["minimum_basins"] = {}
    values["maximum_basins"] = {}
    # v10: the descent drain table (basin adjacency) and its summary.
    values["min_connections_list"] = []
    values["descent_connection_stats"] = {}
    # v11: what each basin GROUP drew and what the shared classifier said. The
    # exporter skips a None metric, so a state that never carried these exports a
    # subset and the strict key-set assertion fails on the new names.
    values["min_basin_arcs"] = []
    values["max_basin_arcs"] = []
    values["active_min_basins"] = []
    values["active_max_basins"] = []
    values["basin_group_stats"] = {"minimum": {}, "maximum": {}}
    values["selected_cp"] = {"id_label": "SAD_42", "is_effective": False}
    values["selected_cluster"] = {"id": "MC3"}
    values["selected_catastrophe"] = None
    values["selected_basin_center"] = {"id": "BC1"}
    values.update(overrides)
    return _FakeState(values)


def _export(**overrides: Any) -> dict[str, Any]:
    """Snapshot a fake state with the standard atom context."""
    return save_params_to_dict(
        _make_state(**overrides),
        plt_path="ethene4.plt",
        atom_info=ATOM_INFO,
        atom_num=1,
    )


def test_export_covers_every_reactive_control() -> None:
    """Every key wired to @state.change appears in the export params.

    The key list is parsed from the module source (same technique as
    test_default_ui_state_covers_every_reactive_control), so a control added
    to the decorator but forgotten in DEFAULT_UI_STATE fails here too.
    """
    src = pathlib.Path(sv.__file__).read_text(encoding="utf-8")
    marker = "    @state.change(\n"
    start = src.index(marker) + len(marker)
    end = src.index("    )\n", start)
    wired = set(re.findall(r'"([a-z_]+)"', src[start:end]))
    assert wired, "no keys parsed from the @state.change decorator"

    payload = _export()
    exported = set(payload["params"]["bools"]) | set(payload["params"]["floats"])
    exported |= set(payload["params"]["strings"])

    assert wired <= exported
    assert exported == set(DEFAULT_UI_STATE) | {"selected_field"}


def test_export_types_match_widgets() -> None:
    """Switches export as bool, sliders as float, selectors as str."""
    payload = _export()
    bools = payload["params"]["bools"]
    floats = payload["params"]["floats"]
    strings = payload["params"]["strings"]

    assert all(isinstance(v, bool) for v in bools.values())
    assert all(isinstance(v, float) for v in floats.values())
    assert all(isinstance(v, str) for v in strings.values())
    # num_contours is an int slider default; it must widen to float in JSON.
    assert isinstance(floats["num_contours"], float)
    assert floats["num_contours"] == 15.0
    assert bools["reduce_catastrophes"] is False
    assert strings["selected_field"] == "Electron Density (condensed)"


def test_export_is_json_serializable_and_round_trips() -> None:
    """The snapshot survives json.dumps and load_params_from_dict."""
    payload = _export()
    text = json.dumps(payload)
    reloaded = json.loads(text)

    params = load_params_from_dict(reloaded)
    for key, expected in DEFAULT_UI_STATE.items():
        got = params[key]
        if isinstance(expected, bool):
            assert got is expected, key
        else:
            assert got == pytest.approx(expected), key
    assert params["selected_field"] == "Electron Density (condensed)"


def test_export_records_context_metrics_and_selection() -> None:
    """Context, derived metrics and selection ids are captured verbatim."""
    payload = _export()

    assert payload["schema_version"] == PARAMS_SCHEMA_VERSION
    assert payload["tool"] == "gba_topology2.sphere_viewer"
    assert payload["timestamp_utc"].endswith("+00:00")

    ctx = payload["context"]
    assert ctx["plt_path"].endswith("ethene4.plt")
    assert pathlib.Path(ctx["plt_path"]).is_absolute()
    assert ctx["atom_number"] == 1
    assert ctx["atom_name"] == "C1"
    assert ctx["field_items"][0]["value"] == "Electron Density (condensed)"

    metrics = payload["derived_metrics"]
    assert set(metrics) == set(sv.PARAM_METRIC_KEYS)
    assert metrics["morse_counts"]["euler"] == 2

    sel = payload["selection"]
    assert sel["selected_cp_id"] == "SAD_42"
    assert sel["selected_cluster_id"] == "MC3"
    assert sel["selected_catastrophe_id"] is None
    assert sel["selected_basin_center_id"] == "BC1"


def test_export_deep_copies_metrics() -> None:
    """Mutating the state metrics after export must not alter the snapshot."""
    state = _make_state()
    payload = save_params_to_dict(
        state, plt_path="ethene4.plt", atom_info=ATOM_INFO, atom_num=1
    )
    state.morse_counts["euler"] = 999

    assert payload["derived_metrics"]["morse_counts"]["euler"] == 2


def test_export_missing_control_key_raises() -> None:
    """A state object missing a control is a hard error, not a silent gap."""
    state = _make_state()
    del state.k_pitch

    with pytest.raises((AttributeError, KeyError)):
        save_params_to_dict(
            state, plt_path="ethene4.plt", atom_info=ATOM_INFO, atom_num=1
        )


def test_export_rejects_typed_mismatch() -> None:
    """A slider carrying a str is rejected at export time."""
    with pytest.raises(TypeError):
        _export(k_pitch="wide")


def test_load_rejects_unknown_schema_version() -> None:
    """A stale/broken schema version must be rejected, not guessed at.

    Version 1 is the case that matters: it is a real file on disk, written
    before the excision ring controls existed. It lacks every new key, so
    without the version gate it would fail as a confusing "Missing parameter
    'ring_samples'" KeyError instead of a clear version message.

    The 99 is deliberate, not PARAMS_SCHEMA_VERSION + 1: version 4 is admitted as
    LEGACY (it was the refinement control's version, and that step was reverted),
    so a v4 snapshot on disk must still open. Only a version above every known
    one is rejected.
    """
    payload = _export()
    for bad in (0, 1, 99, "one", None):
        payload["schema_version"] = bad
        with pytest.raises(ValueError):
            load_params_from_dict(payload)


def test_legacy_version_loads_with_defaults_for_new_controls() -> None:
    """A version 2 snapshot reproduces: absent controls take their defaults.

    The saved Pd_20K observations we diagnose against are version 2 files. A
    control invented after they were written had exactly one value on screen --
    its default -- so filling it is the faithful reading, and rejecting the file
    would destroy the evidence.
    """
    payload = _export()
    payload["schema_version"] = 2
    del payload["params"]["bools"]["show_max_saddle_arcs"]

    params = load_params_from_dict(payload)
    assert params["show_max_saddle_arcs"] is DEFAULT_UI_STATE["show_max_saddle_arcs"]
    # Everything the file did record survives untouched.
    assert params["k_pitch"] == pytest.approx(payload["params"]["floats"]["k_pitch"])


def test_a_version_3_snapshot_loads_the_ridge_refinement_control_as_off() -> None:
    """Every ethene3 and Pd_20K snapshot on disk is version 3.

    The ridge-refinement control is invented after them, so a v3 file has no
    ``refine_ridge_paths`` key. Filling it from the default is the faithful
    reading -- the control did not exist, so it was not on screen, and its default
    is off, which is exactly the state those observations were made in. Rejecting
    the file would make every saved observation on the branch unreadable.
    """
    payload = _export()
    payload["schema_version"] = 3
    del payload["params"]["bools"]["refine_ridge_paths"]

    params = load_params_from_dict(payload)
    assert params["refine_ridge_paths"] is False
    assert DEFAULT_UI_STATE["refine_ridge_paths"] is False
    assert params["k_pitch"] == pytest.approx(payload["params"]["floats"]["k_pitch"])


def test_v6_snapshot_replays_the_fidelity_smoother_not_the_coarse_curve() -> None:
    """A v6 file recorded the FIDELITY smoother, and must keep recording it.

    The coarse structural curve is a NEW key (``coarse_ridge_curves``, v7) rather
    than a new meaning for ``refine_ridge_paths``, and the reason is on disk: the
    ethene3 snapshots were saved with ``refine_ridge_paths`` ON, and their recorded
    connection_stats reproduce the fidelity smoother exactly (n_refined 4,
    total_turn_after_deg 5987.902, max_refine_shift_deg 0.898). Overloading the key
    would replay those files with geometry that was never on screen -- the one thing
    a snapshot must not do.

    So a v6 payload with the fidelity control on loads with the coarse control OFF,
    and the pipeline runs the smoother the file describes.
    """
    payload = _export()
    payload["schema_version"] = 6
    payload["params"]["bools"]["refine_ridge_paths"] = True
    del payload["params"]["bools"]["coarse_ridge_curves"]

    params = load_params_from_dict(payload)
    assert params["refine_ridge_paths"] is True, "the recorded control must survive"
    assert params["coarse_ridge_curves"] is False, (
        "a v6 file predates the coarse curve; it was not on screen"
    )
    assert DEFAULT_UI_STATE["coarse_ridge_curves"] is False


def test_current_version_uses_defaults_for_missing_controls() -> None:
    """Missing parameters fall back to defaults, enabling forward compatibility.

    When a new parameter is added to a feature (like show_all_minimum_basins),
    old parameter files saved before the feature was added should still load.
    Missing parameters take their defaults from DEFAULT_UI_STATE.
    """
    payload = _export()
    assert payload["schema_version"] == PARAMS_SCHEMA_VERSION
    del payload["params"]["bools"]["show_max_saddle_arcs"]

    params = load_params_from_dict(payload)
    assert params["show_max_saddle_arcs"] is DEFAULT_UI_STATE["show_max_saddle_arcs"]


def test_load_missing_control_key_uses_default() -> None:
    """Missing control keys are filled from DEFAULT_UI_STATE.

    This enables old parameter files (saved before a new feature was added)
    to load successfully by using the default value for missing parameters.
    """
    payload = _export()
    del payload["params"]["floats"]["k_bridge"]

    params = load_params_from_dict(payload)
    assert params["k_bridge"] == DEFAULT_UI_STATE["k_bridge"]


def test_load_rejects_bool_for_slider_and_number_for_toggle() -> None:
    """Type coercion is strict in both directions."""
    payload = _export()
    payload["params"]["floats"]["k_pitch"] = True
    with pytest.raises(TypeError):
        load_params_from_dict(payload)

    payload = _export()
    payload["params"]["bools"]["show_rings"] = 1.0
    with pytest.raises(TypeError):
        load_params_from_dict(payload)


def test_load_widens_int_sliders_to_float() -> None:
    """json.loads gives int for 15.0 written as 15; the loader widens it."""
    payload = _export()
    payload["params"]["floats"]["num_contours"] = 15  # int, as json may emit
    params = load_params_from_dict(payload)
    assert isinstance(params["num_contours"], float)
    assert params["num_contours"] == 15.0


def test_load_sections_passthrough() -> None:
    """context/derived_metrics/selection ride along for diagnostics."""
    params = load_params_from_dict(_export())
    assert params["context"]["atom_name"] == "C1"
    assert params["derived_metrics"]["launch_stats"]["max_fold_order"] == 2
    assert params["selection"]["selected_cluster_id"] == "MC3"


def test_default_params_path_naming(tmp_path: pathlib.Path) -> None:
    """The default path sits next to the .plt with the atom-tagged stem."""
    plt = tmp_path / "sub" / "Pd_20K.plt"
    plt.parent.mkdir()
    plt.write_text("")

    out = default_params_path(str(plt), 7)

    assert out == str(tmp_path / "sub" / "Pd_20K_atom7_params.json")


def test_control_key_tables_partition_defaults() -> None:
    """bool/float/string tables cover DEFAULT_UI_STATE exactly once each."""
    bools, floats, strings = (
        sv.PARAM_BOOL_KEYS,
        sv.PARAM_FLOAT_KEYS,
        sv.PARAM_STRING_KEYS,
    )
    assert bools & floats == frozenset()
    assert bools & strings == frozenset()
    assert floats & strings == frozenset()
    assert bools | floats | strings == set(DEFAULT_UI_STATE)


def test_ring_geometry_is_not_a_control() -> None:
    """The excision ring step, ceiling and sample count are not adjustable.

    They are module constants in ``excision_launch_seeds``. Exposing them as
    controls let the operator change WHICH saddles produce seeds, not just how
    finely the search runs -- the topology found must not depend on tuning. An
    earlier revision's early-exit had exactly this defect: it counted consecutive
    ladder rungs, and rung spacing is the expand step.

    Pinned here so the three keys cannot be re-added to DEFAULT_UI_STATE as a
    side effect of re-exposing a slider.
    """
    for key in ("ring_samples", "ring_expand_pitch", "ring_max_pitch"):
        assert key not in DEFAULT_UI_STATE, f"{key} must be a constant, not a control"
        assert key not in sv.PARAM_BOOL_KEYS
        assert key not in sv.PARAM_FLOAT_KEYS


def test_snapshot_with_retired_ring_keys_still_loads() -> None:
    """Files saved while the ring controls existed load unchanged.

    The loader iterates the SCHEMA's keys and reads them from the payload, so an
    extra key in an older file is ignored. Retiring a control therefore needs no
    schema version bump and no fixture migration -- only ADDING a key breaks old
    files, because a missing required key raises.
    """
    payload = _export()
    payload["params"]["floats"]["ring_samples"] = 96.0
    payload["params"]["floats"]["ring_expand_pitch"] = 0.5
    payload["params"]["floats"]["ring_max_pitch"] = 20.0

    params = load_params_from_dict(payload)

    assert "ring_samples" not in params
    assert "ring_expand_pitch" not in params
    assert "ring_max_pitch" not in params
    assert params["k_pitch"] == DEFAULT_UI_STATE["k_pitch"]


def test_probe_output_is_recorded_in_the_snapshot() -> None:
    """The ridge probe's MEASUREMENT travels with the switch that asked for it.

    ``probe_ridge_points`` was already in the file, and it recorded only the request:
    the counters lived in ``state`` for the drawer and died with the process. A saved
    snapshot is the citation for an observation, so the numbers are the point.
    """
    payload = _export(
        probe_ridge_points=True,
        show_max_saddle_arcs=True,
        selected_basin_center={"id": "MAX_4864"},
        ridge_probe_stats=make_ridge_probe_stats(
            [
                {
                    "n_knots": 4,
                    "n_probe": 3,
                    "probe_refused": 1,
                    "probe_offset_pitch": 0.42,
                }
            ],
            on=True,
            basin_id="MAX_4864",
        ),
    )

    stats = payload["derived_metrics"]["ridge_probe_stats"]
    assert stats["on"] is True
    assert stats["n_connections"] == 1
    assert stats["n_points"] == 3
    assert stats["n_refused"] == 1
    assert stats["median_offset_pitch"] == pytest.approx(0.42)
    # The scope, named: these counters describe the arcs on screen, and without the
    # ids a reader cannot tell "4 of one basin" from "4 of 928".
    assert stats["scope_basin_id"] == "MAX_4864"
    assert stats["scope_saddle_id"] is None


def test_selection_records_the_saddle_end() -> None:
    """The arc overlay has two ends, and the probe scopes itself to either.

    Selecting a saddle is how a STALLED ascent gets seen -- it has no maximum to
    select -- so a saddle-scoped measurement was unreproducible from the file alone.
    """
    payload = _export(selected_saddle_id="SAD_4864")

    assert payload["selection"]["selected_saddle_id"] == "SAD_4864"
    assert (
        load_params_from_dict(payload)["selection"]["selected_saddle_id"] == "SAD_4864"
    )


def test_selection_without_a_saddle_writes_null_not_empty_string() -> None:
    """An empty selection is null, the same shape as the other four ids.

    The default is ``""`` and the ids round-trip through ``_selection_id``, which
    writes None for nothing selected; a bare ``state.selected_saddle_id`` would have
    put ``""`` in some files and ``null`` in others for the same state.
    """
    assert _export()["selection"]["selected_saddle_id"] is None


def test_a_v7_file_loads_with_the_probe_scope_absent() -> None:
    """Schema 8 adds the saddle end; a v7 file has no saddle scope to restore.

    v7 is listed as legacy, so its absent controls come from the defaults and it
    loads. What cannot be recovered is the SCOPE of a saved probe, which v7 did not
    record -- the replay says so in ``n_missing`` rather than inventing a selection.
    """
    payload = _export(probe_ridge_points=True)
    payload["schema_version"] = 7
    payload["selection"].pop("selected_saddle_id")

    params = load_params_from_dict(payload)

    assert params["probe_ridge_points"] is True
    assert "selected_saddle_id" not in params["selection"]
    assert params["derived_metrics"]["ridge_probe_stats"]["scope_basin_id"] is None


def test_basin_topology_is_recorded_in_the_snapshot() -> None:
    """The saddle->max graph and each minimum's basin cycle travel in the file.

    Basin boundaries are cycles in the connection graph, so a saved observation
    that renders them must CARRY the graph: toggles that draw nothing were
    undiagnosable from a file that recorded neither the arcs nor the cycles.
    """
    graph = {
        "SAD_1": [
            {
                "max_id": "MAX_1",
                "branch_index": 0,
                "status": "connected",
                "verified": True,
            }
        ]
    }
    basins = {
        "MIN_1": {
            "minimum_id": "MIN_1",
            "n_boundary_arcs": 1,
            "boundary_arcs": [
                {"saddle_id": "SAD_1", "branch_index": 0, "max_id": "MAX_1"}
            ],
            "boundary_saddles": ["SAD_1"],
            "boundary_maxima": ["MAX_1"],
        }
    }
    payload = _export(
        connection_graph=graph,
        minimum_basins=basins,
        max_connections_list=[{"saddle_id": "SAD_1", "max_id": "MAX_1"}],
        basin_centers_list=[{"id": "MIN_1", "polarity": "minimum"}],
    )

    metrics = payload["derived_metrics"]
    assert metrics["connection_graph"] == graph
    assert metrics["minimum_basins"] == basins
    assert metrics["max_connections_list"] == [
        {"saddle_id": "SAD_1", "max_id": "MAX_1"}
    ]
    assert metrics["basin_centers_list"] == [{"id": "MIN_1", "polarity": "minimum"}]
    # A v9 file loads, and the sections pass through untouched (metrics are
    # outputs: the loader never feeds them back into the pipeline).
    loaded = load_params_from_dict(payload)
    assert loaded["derived_metrics"]["minimum_basins"] == basins
    # schema_version is a top-level key of the exported payload; the restored
    # params dict carries sections, not the version (the loader consumed it).
    assert payload["schema_version"] == PARAMS_SCHEMA_VERSION


def test_selection_records_the_minimum_end() -> None:
    """Which minimum's basin was on screen is part of the observation."""
    payload = _export(selected_minimum_id="MIN_5011")

    assert payload["selection"]["selected_minimum_id"] == "MIN_5011"
    assert (
        load_params_from_dict(payload)["selection"]["selected_minimum_id"] == "MIN_5011"
    )


def test_a_v8_file_loads_without_the_basin_sections() -> None:
    """Schema 9 adds the basin exports; a v8 file has no graph to restore.

    v8 is listed as legacy: its controls come from the defaults, and the
    derived_metrics section simply has no basin keys -- the absence is the
    honest record of a snapshot taken before the topology was exportable.
    """
    payload = _export(show_all_minimum_basins=True)
    payload["schema_version"] = 8
    for key in (
        "max_connections_list",
        "basin_centers_list",
        "connection_graph",
        "minimum_basins",
    ):
        payload["derived_metrics"].pop(key)
    payload["selection"].pop("selected_minimum_id")

    params = load_params_from_dict(payload)

    assert params["show_all_minimum_basins"] is True
    assert "minimum_basins" not in params["derived_metrics"]
    assert "selected_minimum_id" not in params["selection"]


#: One descent row in the exact shape min_connection_to_state_dict writes --
#: the basin v10 adjacency as it crosses the file boundary, nulls included
#: (min_id None means "drained nowhere", a stored value, not corruption).
DESCENT_ROW: dict[str, Any] = {
    "saddle_id": "SAD_1",
    "branch_index": 0,
    "min_id": "MIN_1",
    "status": "connected",
    "direction_mode": "port_tangent",
    "cross_track_deg": 0.0,
    "along_track_deg": 25.0,
    "separation_deg": 30.0,
    "arc_points": [[0.0, 0.0, 1.0], [0.6, 0.0, 0.8]],
    "saddle_value": 0.5,
    "min_value": 0.0003,
    "verified": True,
    "blocker_id": None,
    "blocker_polarity": None,
    "method": "geodesic",
    "failure_gate": None,
    "stall_point": None,
    "transited_ids": [],
    "completed_leg_deg": None,
}


def test_v10_export_carries_the_descent_table() -> None:
    """basin v10: the drain table is the adjacency; the file must carry it.

    A v9 file recorded the arcs and rebuilt the ring from proximity -- the rule
    that saturated. The rows cross the boundary verbatim here; their field-for-
    field rehydration is pinned in test_basin_boundary.py (the v9 None-safety
    bug class included).
    """
    stats = {
        "n_branches": 2,
        "n_connected": 1,
        "n_occluded": 1,
        "n_no_target": 0,
        "n_gradient": 0,
    }
    payload = _export(
        min_connections_list=[DESCENT_ROW], descent_connection_stats=stats
    )

    metrics = payload["derived_metrics"]
    assert metrics["min_connections_list"] == [DESCENT_ROW]
    assert metrics["descent_connection_stats"] == stats
    assert payload["schema_version"] == 11
    loaded = load_params_from_dict(json.loads(json.dumps(payload)))
    assert loaded["derived_metrics"]["min_connections_list"] == [DESCENT_ROW]


def test_v11_export_carries_the_maximum_basin_group() -> None:
    """basin v11: the hill side's ring becomes citable.

    The arcs a group drew, the centres on screen, the shared classification, and
    the selected maximum. v10 recorded `selected_minimum_id` and never the
    maximum's id, so a snapshot of a maximum's basin boundary named the valley
    centre -- the asymmetry this pins shut from the file side.
    """
    stats = {
        "minimum": {"status": "bounded", "n_rings": 1},
        "maximum": {
            "status": "bounded",
            "n_rings": 2,
            "min_separation_deg": 24.8,
            "ring_gate_deg": 0.5,
        },
    }
    payload = _export(
        max_basin_arcs=[{"arc_kind": "ring", "endpoint_id": "MAX_1"}],
        basin_group_stats=stats,
        selected_maximum_id="MAX_7",
    )
    assert payload["schema_version"] == 11
    assert payload["derived_metrics"]["max_basin_arcs"] == [
        {"arc_kind": "ring", "endpoint_id": "MAX_1"}
    ]
    assert payload["derived_metrics"]["basin_group_stats"]["maximum"]["n_rings"] == 2
    loaded = load_params_from_dict(payload)
    assert (
        loaded["derived_metrics"]["max_basin_arcs"]
        == payload["derived_metrics"]["max_basin_arcs"]
    )
    assert loaded["selection"]["selected_maximum_id"] == "MAX_7"


def test_a_v10_snapshot_loads_and_its_new_toggles_fill_off() -> None:
    """The v10 incident was rings appearing unasked.

    A v10 file says nothing about the maximum group, and "nothing" must read as
    both switches off. The metrics v11 adds are ABSENT, not empty: an empty
    `max_basin_arcs` would claim the group was computed and drew nothing, which is
    a different statement about the past than "this file predates the group".
    """
    payload = _export()
    payload["schema_version"] = 10
    for key in (
        "min_basin_arcs",
        "max_basin_arcs",
        "active_min_basins",
        "active_max_basins",
        "basin_group_stats",
        "maximum_basins",
    ):
        payload["derived_metrics"].pop(key, None)
    payload["selection"].pop("selected_maximum_id", None)

    params = load_params_from_dict(payload)
    assert params["show_basin_for_selected_maximum"] is False
    assert params["show_all_maximum_basins"] is False
    assert "max_basin_arcs" not in params["derived_metrics"]
    # The v10 replay wrote no `maximum_basins` key at all. Its absence is the
    # honest record; `minimum_basins` is present because v10 computed it.
    assert "maximum_basins" not in params["derived_metrics"]
    assert "minimum_basins" in params["derived_metrics"]
    assert "selected_maximum_id" not in params["selection"]


def test_a_v9_file_loads_without_the_descent_table() -> None:
    """v10 adds the descent export; v9 is legacy: no drain table, honest absence.

    The replay of a v9 file reports its basins as un-derivable-by-v10: the
    adjacency was never recorded, and proximity -- what the old rule used -- is
    exactly what v10 rejected. The loader must not invent it.
    """
    payload = _export(show_all_minimum_basins=True)
    payload["schema_version"] = 9
    for key in ("min_connections_list", "descent_connection_stats"):
        payload["derived_metrics"].pop(key)

    params = load_params_from_dict(payload)

    assert params["show_all_minimum_basins"] is True
    assert "min_connections_list" not in params["derived_metrics"]
