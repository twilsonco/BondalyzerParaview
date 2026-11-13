"""Parameter Snapshot Resolution for gba_topology2.

Loading a saved snapshot is more than copying a dict into the viewer's state:
the snapshot may describe a *different* file or a *different* atom than the
one currently open, and the camera may have been captured around a different
nucleus.

This module turns a raw snapshot into an explicit, testable *restore plan*:

- which ``.plt`` file to open,
- which atom to isolate,
- which condensed field to select,
- the control values to push into the state,
- the camera to apply, re-based onto the target atom.

The plan is deliberately inert: it resolves *what* to restore, and the
viewer decides *how*. That split keeps the interesting logic (path
resolution, field fallback, camera re-basing) testable without a browser or
a render window.

Everything here is pure and UI-free.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from gba_topology2.src.camera_state import (
    CameraState,
    camera_reference_center,
    camera_state_from_dict,
    rebase_camera_state,
)

__all__ = [
    "RestorePlan",
    "resolve_restore_plan",
]


@dataclass(frozen=True)
class RestorePlan:
    """What a loaded snapshot asks the viewer to restore.

    Attributes:
        plt_path: Absolute path of the .plt to open, or None to keep the
            currently open one.
        atom_number: Atom to isolate, or None to keep the current atom.
        selected_field: Condensed field to select, or None to keep the
            current field.
        controls: Flat dict of control values to push into the state.
        camera: Camera to apply, already re-based onto the target atom, or
            None when the snapshot carried no camera.
        field_items: The field list recorded in the snapshot, used to check
            that the requested field actually exists.
        source_path: Absolute path of the snapshot file, for messages.
    """

    plt_path: str | None
    atom_number: int | None
    selected_field: str | None
    controls: dict[str, Any] = field(default_factory=dict)
    camera: CameraState | None = None
    field_items: list[dict[str, str]] = field(default_factory=list)
    source_path: str | None = None

    @property
    def requires_reload(self) -> bool:
        """Whether restoring this plan means opening a different file/atom.

        Returns:
            True when the snapshot names a .plt or atom that differs from the
            ones currently loaded, so the viewer must re-extract the mesh.
        """
        return self.plt_path is not None or self.atom_number is not None


def _resolve_field(
    requested: str | None,
    field_items: list[dict[str, str]],
) -> str | None:
    """Pick the field to select, falling back to the first discovered one.

    A snapshot may name a field that the target file does not have -- the
    two files can carry different condensed arrays. Rather than failing the
    whole restore, the first available field is used, which keeps the viewer
    usable and is reported through the plan.

    Args:
        requested: The field named by the snapshot, if any.
        field_items: The field list discovered on the target mesh.

    Returns:
        The field to select, or None when there is nothing to select.
    """
    available = [str(item.get("value", "")) for item in field_items]
    if requested and requested in available:
        return requested
    if available:
        return available[0]
    return requested


def resolve_restore_plan(
    params: dict[str, Any],
    *,
    current_plt_path: str | None = None,
    current_atom_number: int | None = None,
    source_path: str | None = None,
) -> RestorePlan:
    """Turn a loaded snapshot into an actionable restore plan.

    Args:
        params: Flat control dict from ``load_params_from_dict``, with the
            optional "context" and "camera" sections attached.
        current_plt_path: Absolute path of the .plt currently open, used to
            decide whether the mesh has to be re-extracted.
        current_atom_number: Atom currently isolated.
        source_path: Absolute path of the snapshot file, for messages.

    Returns:
        The resolved :class:`RestorePlan`.
    """
    context = params.get("context")
    context = context if isinstance(context, dict) else {}
    field_items = [
        {"title": str(item.get("title", "")), "value": str(item.get("value", ""))}
        for item in context.get("field_items", [])
        if isinstance(item, dict)
    ]

    raw_plt = context.get("plt_path")
    target_plt: str | None = None
    if isinstance(raw_plt, str) and raw_plt:
        candidate = os.path.abspath(raw_plt)
        if current_plt_path is None or os.path.abspath(current_plt_path) != candidate:
            target_plt = candidate

    raw_atom = context.get("atom_number")
    target_atom: int | None = None
    if (
        isinstance(raw_atom, int)
        and not isinstance(raw_atom, bool)
        and (current_atom_number is None or raw_atom != current_atom_number)
    ):
        target_atom = raw_atom

    requested_field = params.get("selected_field")
    if not isinstance(requested_field, str):
        requested_field = None
    selected_field = _resolve_field(requested_field, field_items)

    controls = {
        key: value
        for key, value in params.items()
        if not isinstance(value, dict) and key != "selected_field"
    }
    if selected_field is not None:
        controls["selected_field"] = selected_field

    # Extract and restore selection values (they are stored in a separate
    # dict but need to be applied as state controls). Only restore string-based
    # ID selections that can be directly assigned; object-based selections
    # (selected_cp, selected_cluster, etc.) cannot be restored from just their IDs
    # without looking them up in the pipeline data.
    selection = params.get("selection")
    if isinstance(selection, dict):
        # Restore string ID selections that are stored directly in state
        id_selections = {
            "selected_saddle_id": "selected_saddle_id",
            "selected_minimum_id": "selected_minimum_id",
        }
        for sel_key, state_key in id_selections.items():
            value = selection.get(sel_key)
            if value is not None:
                controls[state_key] = value

    camera = _resolve_camera(params)

    return RestorePlan(
        plt_path=target_plt,
        atom_number=target_atom,
        selected_field=selected_field,
        controls=controls,
        camera=camera,
        field_items=field_items,
        source_path=source_path,
    )


def _resolve_camera(params: dict[str, Any]) -> CameraState | None:
    """Read the camera section of a snapshot, if it has one.

    The camera is stored in world coordinates together with the nucleus it
    was captured around. When that nucleus is known, the camera is re-based
    onto the origin so it can be applied to whichever atom ends up loaded;
    the viewer then translates it onto the actual nucleus. When the nucleus
    is not known -- a file written before the camera existed -- the camera is
    returned as captured and the viewer applies it unchanged.

    Args:
        params: Flat control dict with the optional "camera" section.

    Returns:
        The camera to apply, or None when the snapshot carried no camera.
    """
    section = params.get("camera")
    if not isinstance(section, dict):
        return None
    state = camera_state_from_dict(section)
    reference = camera_reference_center(section)
    if reference is None:
        return state
    # Re-base onto the origin; the viewer re-bases onto the live nucleus.
    return rebase_camera_state(state, reference, (0.0, 0.0, 0.0))
