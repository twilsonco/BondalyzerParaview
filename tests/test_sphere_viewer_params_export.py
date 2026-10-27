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
    """
    payload = _export()
    for bad in (0, 1, PARAMS_SCHEMA_VERSION + 1, "one", None):
        payload["schema_version"] = bad
        with pytest.raises(ValueError):
            load_params_from_dict(payload)


def test_load_requires_every_control_key() -> None:
    """Dropping any control key raises KeyError naming the key."""
    payload = _export()
    del payload["params"]["floats"]["k_bridge"]

    with pytest.raises(KeyError, match="k_bridge"):
        load_params_from_dict(payload)


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
