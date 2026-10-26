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
    "PARAM_METRIC_KEYS",
    "PARAM_STRING_KEYS",
    "default_params_path",
    "load_params_from_dict",
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
    # Basin Locator Step 2: saddle separatrix launch seeds. The seeds launch from
    # each saddle's excision boundary, profiled at ring_samples azimuths and
    # expanded by ring_expand_pitch * dtheta until the port signature is pure, up
    # to ring_max_pitch * dtheta.
    "show_launch_seeds": False,
    "ring_samples": 48,
    "ring_expand_pitch": 1.0,
    "ring_max_pitch": 12.0,
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
PARAMS_SCHEMA_VERSION = 2

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
PARAM_STRING_KEYS: frozenset[str] = frozenset()
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
)

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
        ValueError: If schema_version is missing or unsupported.
        KeyError: If any required control key is absent.
        TypeError: If any value has the wrong type.
    """
    version = payload.get("schema_version")
    if version != PARAMS_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported params schema_version {version!r}; "
            f"expected {PARAMS_SCHEMA_VERSION}"
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
                raise KeyError(f"Missing parameter '{key}' in section '{name}'")
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
