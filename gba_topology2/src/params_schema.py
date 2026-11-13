"""Parameter Snapshot Schema for gba_topology2.

The sphere viewer can export its complete observable state to JSON ("Save
Parameters") and read it back ("Load Parameters"). This module owns the
*format* of that snapshot, so the viewer, the headless diagnostic runner and
the tests all agree on one definition instead of three copies.

The snapshot has five top-level sections:

``schema_version``
    Bumped on any breaking change to the layout below, so a stale file is
    rejected rather than silently misread.
``params``
    Every editable control, split into ``bools`` / ``floats`` / ``strings``
    by the type its widget produces. This is the only section that is fed
    back into the viewer's state.
``context``
    The run context: the absolute .plt path, the atom number and the
    discovered condensed field list. A loader uses it to decide whether a
    different file or atom has to be opened.
``camera``
    The view camera (direction, zoom, projection), stored relative to the
    atom's nucleus so it travels with the atom when a different file or atom
    is loaded. Optional: files written before the camera existed still load.
``derived_metrics`` / ``selection``
    Diagnostics captured at observation time. These are outputs, never inputs:
    a loader must not feed them back into the pipeline.

Everything here is pure and UI-free, so the format can be exercised without a
browser, a render window or a viewer instance.
"""

from __future__ import annotations

import json
import os
import statistics
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import Any

from gba_topology2.src.camera_state import (
    CameraState,
    camera_section_to_dict,
)

__all__ = [
    "DEFAULT_UI_STATE",
    "PARAMS_SCHEMA_VERSION",
    "PARAM_BOOL_KEYS",
    "PARAM_CONTEXT_STRING_KEYS",
    "PARAM_CONTROL_KEYS",
    "PARAM_FLOAT_KEYS",
    "PARAM_LEGACY_SCHEMA_VERSIONS",
    "PARAM_METRIC_KEYS",
    "PARAM_STRING_KEYS",
    "RIDGE_PROBE_STAT_KEYS",
    "default_params_path",
    "load_params_from_dict",
    "make_ridge_probe_stats",
    "save_params_to_dict",
]


# Canonical baseline for every interactive control in the viewer. Consumed
# both by the initial state seeding in ``run_sphere_viewer`` and by the
# "Reset to Defaults" action, so the two can never drift apart.
# NOTE: "selected_field" is intentionally absent -- the default field is data
# dependent (the first discovered condensed array) and supplied at runtime.
DEFAULT_UI_STATE: dict[str, Any] = {
    # Field & mesh display
    "show_flood": True,
    "show_contours": True,
    "show_sphere_wire": False,
    "num_contours": 15,
    # Step 1: Morse detection
    "show_cps": True,
    "show_maxima": True,
    "show_minima": True,
    "show_saddles": True,
    "cp_glyph_scale": 1.0,
    # Step 2: Micro-clustering
    "k_pitch": 2.0,
    "show_rings": True,
    # Step 4: Saddle-string / corridor aggregation
    "aggregate_strings": False,
    "k_bridge": 3.0,
    # Step 5: Catastrophes & excision boundary ports
    "show_catastrophes": True,
    "show_ports": True,
    "cat_barrier_tol": 0.005,
    "cat_angular_tol_mult": 4.0,
    # Step 6: Effective CP reduction & harmonic nudge
    "reduce_catastrophes": False,
    "reduce_multi_clusters": False,
    "nudge_cp_positions": False,
    "perfect_port_angles": False,
    "nudge_max_pitch": 1.5,
    # Basin Locator Step 2: saddle separatrix launch seeds. Seeds launch from
    # each saddle's excision boundary, expanded until the port signature is pure.
    # The ring step, ceiling and sample count are deliberately NOT controls: they
    # are module constants in excision_launch_seeds, because the topology found
    # must not depend on operator tuning.
    "show_launch_seeds": False,
    # Basin Locator Step 3: maximum connections. Each ascent branch extrapolates
    # a great-circle ray and claims the maximum it points at; the arcs are drawn
    # only from the SELECTED maximum, so this is an inspection overlay. The arc
    # length cap is likewise a module constant (max_connection.MAX_ARC_DEG, in
    # degrees), not a slider, for the same reason as the ring geometry above.
    "show_max_saddle_arcs": False,
    # Basin Locator Step 6: the coarse structural curve. A gradient-traced
    # connector is an ascent at one mesh pitch per step, so it carries the
    # integrator's step size as chatter and, on a spiralling field, as winding.
    # This replaces each CYAN connector with one smooth spherical curve through its
    # structural nodes -- the launch port, the maximum, the detected corners (which
    # stay sharp), the transited excision centres, and points spaced several mesh
    # pitches along the arc between them. Geodesic connectors are never touched -- a
    # great circle is not an approximation of anything. Off by default: the traced
    # ascent is the measurement and the curve is an interpretation of it, and the
    # operator must be able to see both.
    #
    # The key name is kept from the Entry-058 transverse nudge that this replaced.
    # The semantic change is invisible to the file format: every snapshot on disk
    # has this control OFF, so a v6 snapshot replays identically, and a snapshot
    # saved with it on asks for "the interpretation of the ascent", which is what
    # this is. Renaming it would force a v7 for no change in what a file means.
    "refine_ridge_paths": False,
    # Basin Locator Step 6, the coarse rendering. The structural curve: ONE smooth
    # spherical curve through a handful of nodes -- the launch port, the maximum, the
    # detected corners (which stay sharp), the transited excision centres, and points
    # spaced NODE_PITCH mesh pitches along the arc between them. Where the fidelity
    # smoother preserves the ascent and removes only mesh-scale chatter, this replaces
    # it: the exact path of the ascent is a property of the integrator's step size, and
    # the 3D representation wants the connection. The node spacing is a low-pass
    # filter's cutoff, not a density knob -- at 1 pitch a spiralling connector is
    # reproduced almost exactly (65935 of 66010 deg of turning), at 8 it is filtered to
    # 651 deg. Takes precedence over refine_ridge_paths when both are on, since the
    # two are alternative descriptions of the same drawn line.
    "coarse_ridge_curves": False,
    # Ridge PROBE (diagnostic). Marks, on the SELECTED connector, the sharp
    # continuous bends (black) and the ridge the field puts beside the traced path
    # (white), computed from the Hessian of log f over a wide disc. The path is NOT
    # moved: this measures how far off-crest the ascent runs -- the drift the
    # smoother deliberately leaves alone, since removing a tooth is a change to the
    # path's shape, not a shift across it. Off by default; it costs one disc fit per
    # sample on the visible paths only.
    "probe_ridge_points": False,
    # Minimum basin boundaries: visualization of closed cycles enclosing each
    # minimum CP in the connection topology. Show individual minimum or all at once.
    "show_basin_for_selected_minimum": False,
    "show_all_minimum_basins": False,
}

# Schema version of the JSON file written by the "Save Parameters" button.
# Bump on any breaking change to the layout below so loaders can reject
# stale files instead of silently misreading them.
#
# Version 1 covers the original layout (params / context / derived_metrics /
# selection). The "camera" section was added later as an *optional* section,
# so it did not warrant a version bump: files without it still load, and
# files with it are recognised by the presence of the key.
#
# Version 2 adds the excision ring controls. Adding a control is breaking on its
# own: load_params_from_dict requires every control key to be present, so a
# version 1 file would fail with a confusing "Missing parameter
# 'ring_samples'" KeyError. Bumping makes the rejection a clear version message.
#
# Version 3 adds the maximum-connection overlay control. Because the snapshots we
# diagnose WITH are version 2 files on disk, version 2 is admitted as a legacy
# version (see PARAM_LEGACY_SCHEMA_VERSIONS) and its absent controls are filled
# from DEFAULT_UI_STATE, so an existing observation still reproduces.
#
# Version 4 added the connector-refinement control. That step was reverted -- the
# refinement will be rebuilt differently -- so the control is gone from
# DEFAULT_UI_STATE, but snapshots saved while it existed are still on disk. v4 is
# admitted as legacy: the loader reads the keys it knows and ignores the one it
# no longer has, so an observation made with refinement on still opens and
# reproduces the pipeline it describes.
#
# Version 5 adds the ridge-refinement control (refine_ridge_paths), the rebuilt
# Step 6. Version 3 is the one that matters as legacy: every ethene3 and Pd_20K
# snapshot on disk is v3, and a control invented after them had exactly one value
# on screen -- its default, off -- so filling it is the faithful reading.
#
# Version 6 adds the ridge-probe control (probe_ridge_points). Version 5 is
# admitted as legacy for the same reason v3 was: a snapshot predating the control
# shows it off, which is its default.
#
# Version 7 adds the coarse structural curve (coarse_ridge_curves). It is a NEW key
# rather than a new meaning for refine_ridge_paths, and the reason is evidence, not
# tidiness: the ethene3 snapshots on disk were saved with refine_ridge_paths ON, and
# their recorded connection_stats reproduce the FIDELITY smoother exactly
# (n_refined 4, total_turn_after_deg 5987.902, max_refine_shift_deg 0.898 -- all four
# match to the last digit). Overloading the key would replay those snapshots with
# geometry that was never on screen, which is the one thing a snapshot must not do.
# Version 6 is admitted as legacy: it predates coarse_ridge_curves, so the control
# was off, which is its default, and the snapshot replays the line it recorded.
#
# Version 8 exports the ridge probe's OUTPUT (derived_metrics.ridge_probe_stats), not
# just its switch. No control key was invented, so the strict/legacy split is
# unaffected: metrics are outputs the loader never feeds back into the pipeline, and
# a v7 file simply has no probe block to compare. v7 is admitted as legacy because
# every snapshot on disk predates the export, and the absence of a measurement is not
# a measurement of zero -- diagnose_params reports "no comparable metric" for those
# files rather than a mismatch.
PARAMS_SCHEMA_VERSION = 8

# Older schema versions that still load, with absent controls defaulted. A
# snapshot is a record of what was ON SCREEN, so refusing to read a file written
# before a control was invented destroys the evidence. The current version stays
# strict: a truncated CURRENT file is a bug to surface, not a gap to paper over.
PARAM_LEGACY_SCHEMA_VERSIONS: frozenset[int] = frozenset({2, 3, 4, 5, 6, 7})

# Every editable control, in DEFAULT_UI_STATE's canonical order.
PARAM_CONTROL_KEYS: tuple[str, ...] = tuple(DEFAULT_UI_STATE)

# Type coercion tables: the viewer state is duck-typed, so the export pins
# each control to the JSON type its widget produces (switch -> bool,
# slider -> float, select -> str). Every control key must appear in exactly
# one table; the module-level checks below enforce that at import time.
PARAM_BOOL_KEYS: frozenset[str] = frozenset(
    k for k, v in DEFAULT_UI_STATE.items() if isinstance(v, bool)
)
PARAM_FLOAT_KEYS: frozenset[str] = frozenset(
    k
    for k, v in DEFAULT_UI_STATE.items()
    if isinstance(v, (int, float)) and not isinstance(v, bool)
)
# No control produces a string today: every selector in the viewer is
# data-dependent and lives in PARAM_CONTEXT_STRING_KEYS. Derived from the
# defaults so the first str widget registers itself.
PARAM_STRING_KEYS: frozenset[str] = frozenset(
    k for k, v in DEFAULT_UI_STATE.items() if isinstance(v, str)
)
if set(PARAM_BOOL_KEYS | PARAM_FLOAT_KEYS | PARAM_STRING_KEYS) != set(
    PARAM_CONTROL_KEYS
):
    raise RuntimeError(
        "PARAM_*_KEYS tables must cover exactly the DEFAULT_UI_STATE controls"
    )

# Data-dependent string control, supplied at runtime (absent from
# DEFAULT_UI_STATE), exported under params["strings"].
PARAM_CONTEXT_STRING_KEYS: tuple[str, ...] = ("selected_field",)

# Derived metric snapshots copied verbatim into the export. These are
# pipeline outputs, not inputs: the loader returns them for diagnostics but
# they must never be fed back into state.update().
PARAM_METRIC_KEYS: tuple[str, ...] = (
    "geom_info",
    "morse_counts",
    "display_counts",
    "reduction_stats",
    "launch_stats",
    "connection_stats",
    # The ridge probe measures the ascent beside the connectors on screen, and
    # until now the ONLY record of that measurement was the pixels in the drawer.
    # Saving it makes a diagnostic observation citable: the switch was already in
    # the snapshot, but a switch says what was asked for, not what was found.
    "ridge_probe_stats",
)

# The probe's summary, in the order ``make_ridge_probe_stats`` writes it. Exported so
# the drawer, the replay and the tests all key off one definition.
RIDGE_PROBE_STAT_KEYS: tuple[str, ...] = (
    "on",
    "n_connections",
    "n_knots",
    "n_points",
    "n_refused",
    "median_offset_pitch",
    "scope_basin_id",
    "scope_saddle_id",
)

# Per-connection probe fields the aggregator reads off a row. The viewer stamps these
# onto its arc-table rows; diagnose_params builds the same dicts headlessly.
_PROBE_ROW_KEYS = {
    "n_knots": "n_knots",
    "n_probe": "n_points",
    "probe_refused": "n_refused",
}


def make_ridge_probe_stats(
    probe_rows: Sequence[Mapping[str, Any]],
    *,
    on: bool,
    basin_id: str | None = None,
    saddle_id: str | None = None,
) -> dict[str, Any]:
    """Aggregate per-connector ridge-probe rows into the saved summary.

    ONE implementation of the aggregation, deliberately: the viewer measures the
    probe while drawing and the headless replay measures it while reproducing, and
    a saved number is only worth diffing if both sides compute it identically. The
    two callers used to sum by hand, and the sum is exactly the kind of thing that
    drifts -- the viewer's init block had already dropped ``n_refused`` from its
    copy of this dict, so a file saved before any probe ran reported a different
    key set than one saved after.

    Args:
        probe_rows: One mapping per PROBED connector, carrying ``n_knots``,
            ``n_probe``, ``probe_refused`` and ``probe_offset_pitch`` (the field
            names the viewer's arc table uses). Rows are counted, not filtered:
            pass only the connectors the probe actually measured.
        on: Whether the probe ran, i.e. the toggle AND its display gates. False
            zeroes every counter; the scope ids are dropped with it, because a
            scope that measured nothing is not a fact about the field.
        basin_id: The selected maximum whose arcs were on screen, when the probe
            was selection-scoped. Recorded so a reader can tell 4 connectors on
            one basin from 4 connectors in the whole run.
        saddle_id: The selected saddle whose launch branches were probed, when it
            was the saddle that was selected. Mutually exclusive with basin_id,
            as in ``_emit_max_arcs``.

    Returns:
        A JSON-safe dict with exactly ``RIDGE_PROBE_STAT_KEYS``. The median is over
        the PER-CONNECTOR medians, matching ``np.median`` on both parities, so the
        viewer and this function agree bit-for-bit.
    """
    offsets = (
        [
            float(row["probe_offset_pitch"])
            for row in probe_rows
            if "probe_offset_pitch" in row
        ]
        if on
        else []
    )
    stats: dict[str, Any] = {
        "on": bool(on),
        "n_connections": len(probe_rows) if on else 0,
        "n_knots": sum(int(row.get("n_knots", 0)) for row in probe_rows) if on else 0,
        "n_points": sum(int(row.get("n_probe", 0)) for row in probe_rows) if on else 0,
        "n_refused": (
            sum(int(row.get("probe_refused", 0)) for row in probe_rows) if on else 0
        ),
        "median_offset_pitch": (float(statistics.median(offsets)) if offsets else 0.0),
        "scope_basin_id": str(basin_id) if (on and basin_id is not None) else None,
        "scope_saddle_id": str(saddle_id) if (on and saddle_id is not None) else None,
    }
    return {key: stats[key] for key in RIDGE_PROBE_STAT_KEYS}


# Name of the optional top-level section holding the view camera.
PARAM_CAMERA_SECTION = "camera"


def default_params_path(plt_path: str, atom_num: int) -> str:
    """Return the default output path for a saved parameter snapshot.

    Args:
        plt_path: Path to the .plt file the viewer was launched with.
        atom_num: Target atom number (1-based).

    Returns:
        ``<plt_dir>/<plt_stem>_atom<atom_num>_params.json``.
    """
    abs_plt = os.path.abspath(plt_path)
    directory = os.path.dirname(abs_plt)
    stem = os.path.splitext(os.path.basename(abs_plt))[0]
    return os.path.join(directory, f"{stem}_atom{atom_num}_params.json")


def _coerce_bool(key: str, raw: Any) -> bool:
    """Coerce an exported toggle value, rejecting non-bool types.

    Args:
        key: Control key, for the error message.
        raw: Value read from the viewer state or a JSON payload.

    Returns:
        The value as a bool.

    Raises:
        TypeError: If the value is not a bool.
    """
    if not isinstance(raw, bool):
        raise TypeError(f"Parameter '{key}' must be a bool, got {type(raw).__name__}")
    return raw


def _coerce_float(key: str, raw: Any) -> float:
    """Coerce an exported slider value to float, rejecting bools.

    Args:
        key: Control key, for the error message.
        raw: Value read from the viewer state or a JSON payload.

    Returns:
        The value as a float (ints from JSON, e.g. num_contours, widen to
        float).

    Raises:
        TypeError: If the value is a bool or non-numeric.
    """
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise TypeError(f"Parameter '{key}' must be a number, got {raw!r}")
    return float(raw)


def _coerce_str(key: str, raw: Any) -> str:
    """Coerce an exported selector value to str, rejecting non-strings.

    Args:
        key: Control key, for the error message.
        raw: Value read from the viewer state or a JSON payload.

    Returns:
        The value as a str.

    Raises:
        TypeError: If the value is not a str.
    """
    if not isinstance(raw, str):
        raise TypeError(f"Parameter '{key}' must be a str, got {type(raw).__name__}")
    return raw


def _selection_id(selected: Any, key: str = "id_label") -> str | None:
    """Extract a stable id from a selection dict for the export.

    Args:
        selected: The selection state value (a dict or None).
        key: Which key carries the id ("id_label" for CPs, "id" otherwise).

    Returns:
        The id string, or None when nothing is selected.
    """
    if not isinstance(selected, dict):
        return None
    value = selected.get(key)
    return value if isinstance(value, str) else None


def save_params_to_dict(
    state_like: Any,
    *,
    plt_path: str,
    atom_info: dict[str, Any],
    atom_num: int,
    camera: CameraState | None = None,
    camera_center: tuple[float, float, float] | None = None,
) -> dict[str, Any]:
    """Snapshot every editable viewer control plus context and metrics.

    The returned dict is JSON-serializable and fully describes the GUI state
    observed on screen: every toggle (bool), slider (float) and selector
    (str), the run context (absolute .plt path, atom, selected condensed
    field, discovered field list), the view camera, the derived pipeline
    metrics, and the active selection ids. Diagnostic scripts feed the file
    back through ``load_params_from_dict`` to reproduce an observation
    headlessly.

    Args:
        state_like: Object exposing the control/metric keys as attributes
            (the trame ``state`` object satisfies this).
        plt_path: .plt path the viewer was launched with; stored absolute.
        atom_info: Atom metadata dict from ``extract_atom_sphere_mesh``.
        atom_num: Target atom number (1-based).
        camera: Optional view-camera snapshot. When given it is written to
            the optional ``camera`` section so the zoom level and viewing
            direction travel with the parameters.
        camera_center: The world-frame nucleus the camera was captured
            around. Required when ``camera`` is given, so the camera can be
            re-based when it is restored against a different atom.

    Returns:
        Schema-versioned export dict with keys: schema_version, tool,
        timestamp_utc, params (bools/floats/strings), context, camera,
        derived_metrics, selection.

    Raises:
        KeyError: If a control key is missing from the state object.
        TypeError: If a control value has an unexpected type.
        ValueError: If a camera is given without its reference nucleus.
    """
    bools = {
        k: _coerce_bool(k, getattr(state_like, k)) for k in sorted(PARAM_BOOL_KEYS)
    }
    floats = {
        k: _coerce_float(k, getattr(state_like, k)) for k in sorted(PARAM_FLOAT_KEYS)
    }
    strings = {
        k: _coerce_str(k, getattr(state_like, k)) for k in sorted(PARAM_STRING_KEYS)
    }
    for key in PARAM_CONTEXT_STRING_KEYS:
        strings[key] = _coerce_str(key, getattr(state_like, key))

    context: dict[str, Any] = {
        "plt_path": os.path.abspath(plt_path),
        "atom_number": int(atom_num),
        "atom_name": str(atom_info.get("atom_name", "")),
        "atom_symbol": str(atom_info.get("atom_symbol", "")),
        "field_items": [
            {"title": str(f["title"]), "value": str(f["value"])}
            for f in atom_info.get("field_items", [])
        ],
    }

    metrics: dict[str, Any] = {
        key: json.loads(json.dumps(getattr(state_like, key)))
        for key in PARAM_METRIC_KEYS
        if getattr(state_like, key, None) is not None
    }

    selection: dict[str, str | None] = {
        "selected_cp_id": _selection_id(getattr(state_like, "selected_cp", None)),
        "selected_cluster_id": _selection_id(
            getattr(state_like, "selected_cluster", None), "id"
        ),
        "selected_catastrophe_id": _selection_id(
            getattr(state_like, "selected_catastrophe", None), "id"
        ),
        "selected_basin_center_id": _selection_id(
            getattr(state_like, "selected_basin_center", None), "id"
        ),
        # The arc overlay has two ends and the snapshot used to record one of them.
        # Naming a saddle is a first-class selection -- it is how a STALLED ascent
        # gets seen, since it has no maximum to select -- and the ridge probe scopes
        # itself to whatever is on screen, so a saddle-scoped measurement was
        # unreproducible from the file alone.
        "selected_saddle_id": (getattr(state_like, "selected_saddle_id", None) or None),
        # Basin visualization: the minimum selected for boundary display.
        "selected_minimum_id": (getattr(state_like, "selected_minimum_id", None) or None),
    }

    payload: dict[str, Any] = {
        "schema_version": PARAMS_SCHEMA_VERSION,
        "tool": "gba_topology2.sphere_viewer",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "params": {"bools": bools, "floats": floats, "strings": strings},
        "context": context,
        "derived_metrics": metrics,
        "selection": selection,
    }
    if camera is not None:
        if camera_center is None:
            raise ValueError(
                "A camera snapshot requires the nucleus it was captured around"
            )
        payload[PARAM_CAMERA_SECTION] = camera_section_to_dict(camera, camera_center)
    return payload


def load_params_from_dict(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate a saved parameter snapshot and prepare it for state.update().

    Checks ``schema_version``, requires every control key to be present, and
    coerces values to their widget types (bools strict, ints widened to
    float). The result's top level is safe to pass to ``state.update()``
    directly: it contains exactly the control keys plus ``selected_field``.
    The ``context``, ``derived_metrics``, ``selection`` and ``camera``
    sections are returned alongside for diagnostics; they are outputs/ids, not
    inputs, so callers must not feed them back into the pipeline state.

    Args:
        payload: Dict loaded from a file written by ``save_params_to_dict``
            (e.g. ``json.load(f)``).

    Returns:
        Dict with the flat control params, plus "context",
        "derived_metrics", "selection" and "camera" passthrough sections.

    Raises:
        ValueError: If schema_version is missing, unsupported, or a legacy
            version is missing a control that has no default.
        KeyError: If any required control key is absent at the current version.
        TypeError: If any value has the wrong type.
    """
    version = payload.get("schema_version")
    strict = version == PARAMS_SCHEMA_VERSION
    if not strict and version not in PARAM_LEGACY_SCHEMA_VERSIONS:
        raise ValueError(
            f"Unsupported params schema_version {version!r}; "
            f"expected {PARAMS_SCHEMA_VERSION} "
            f"(legacy: {sorted(PARAM_LEGACY_SCHEMA_VERSIONS)})"
        )
    params = payload.get("params")
    if params is None:
        raise KeyError("Params payload is missing the 'params' section")
    if not isinstance(params, dict):
        raise TypeError("Params payload section 'params' must be a dict")

    def _section(name: str, keys: list[str], coerce: Any) -> dict[str, Any]:
        section = params.get(name)
        if not isinstance(section, dict):
            raise KeyError(f"Params section '{name}' is missing")
        out: dict[str, Any] = {}
        for key in keys:
            if key not in section:
                # A snapshot predates the control (either from an older schema
                # version or an older file from the current version), so the
                # control had a known value: its default. Filling it keeps an
                # old observation reproducible instead of unreadable.
                if key in DEFAULT_UI_STATE:
                    out[key] = DEFAULT_UI_STATE[key]
                    continue
                raise ValueError(
                    f"Params section '{name}' is missing '{key}' and no default exists"
                )
            out[key] = coerce(key, section[key])
        return out

    bools = _section("bools", sorted(PARAM_BOOL_KEYS), _coerce_bool)
    floats = _section("floats", sorted(PARAM_FLOAT_KEYS), _coerce_float)
    strings = _section(
        "strings",
        sorted(PARAM_STRING_KEYS | set(PARAM_CONTEXT_STRING_KEYS)),
        _coerce_str,
    )

    result: dict[str, Any] = {**bools, **floats, **strings}
    for section in ("context", "derived_metrics", "selection", PARAM_CAMERA_SECTION):
        value = payload.get(section)
        if isinstance(value, dict):
            result[section] = value
    return result
