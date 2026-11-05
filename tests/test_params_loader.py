"""Unit tests for the parameter-snapshot loader and its camera round trip.

Covers:
1. ``params_schema.save_params_to_dict`` -- the camera reaches the JSON and
   is recorded relative to the nucleus it was captured around.
2. ``params_schema.load_params_from_dict`` -- the camera section rides along
   as a passthrough, and a file without one still loads.
3. ``params_loader.resolve_restore_plan`` -- the plan resolves the file,
   atom, field, controls and camera, and knows when a reload is needed.
4. The end-to-end round trip: save -> load -> plan -> apply, on real data.

The loader is pure, so none of this needs a browser or a render window.
"""

from __future__ import annotations

import json
import os
import pathlib
from types import SimpleNamespace
from typing import Any

import pytest
from snapshot_params import load_snapshot

import gba_topology2.src.sphere_viewer as sv
from gba_topology2.src.camera_state import (
    CameraState,
    capture_camera_state,
    rebase_camera_state,
)
from gba_topology2.src.params_loader import resolve_restore_plan
from gba_topology2.src.params_schema import (
    DEFAULT_UI_STATE,
    load_params_from_dict,
    save_params_to_dict,
)

PLT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ethene4.plt"
)

CENTER = (1.0, 2.0, 3.0)

CAMERA = CameraState(
    position=(4.0, 5.0, 6.0),
    focal_point=(1.0, 2.0, 3.0),
    view_up=(0.0, 0.0, 1.0),
    parallel_projection=0,
    parallel_scale=1.25,
    view_angle=30.0,
)


def _make_state(**overrides: Any) -> SimpleNamespace:
    """Build a duck-typed state carrying every control and metric key.

    Args:
        overrides: Keys whose default value should be replaced.

    Returns:
        The fake state object.
    """
    values: dict[str, Any] = dict(DEFAULT_UI_STATE)
    values["selected_field"] = "Electron Density (condensed)"
    values["geom_info"] = {"num_vertices": 10}
    values["morse_counts"] = {"euler": 2}
    values["display_counts"] = {"euler": 2}
    values["reduction_stats"] = {"n_effective_cps": 0}
    values["launch_stats"] = {"n_saddles_seeded": 0}
    values["selected_cp"] = None
    values["selected_cluster"] = None
    values["selected_catastrophe"] = None
    values["selected_basin_center"] = None
    values.update(overrides)
    return SimpleNamespace(**values)


def _atom_info(field_items: list[dict[str, str]] | None = None) -> dict[str, Any]:
    """Build an atom-info dict for the export.

    Args:
        field_items: The field list to record.

    Returns:
        The atom metadata.
    """
    return {
        "atom_number": 1,
        "atom_symbol": "C",
        "atom_name": "C1",
        "center": list(CENTER),
        "field_items": field_items
        if field_items is not None
        else [{"title": "Electron Density", "value": "Electron Density (condensed)"}],
    }


def _export(
    camera: CameraState | None = None,
    camera_center: tuple[float, float, float] | None = None,
    **state_overrides: Any,
) -> dict[str, Any]:
    """Export a fake state, optionally with a camera.

    Args:
        camera: The camera to include.
        camera_center: The nucleus the camera was captured around.
        state_overrides: Control overrides for the fake state.

    Returns:
        The exported payload.
    """
    return save_params_to_dict(
        _make_state(**state_overrides),
        plt_path=PLT,
        atom_info=_atom_info(),
        atom_num=1,
        camera=camera,
        camera_center=camera_center,
    )


# ---------------------------------------------------------------------------
# Export: the camera reaches the JSON
# ---------------------------------------------------------------------------


def test_export_includes_the_camera_section() -> None:
    """A camera snapshot is written to the optional "camera" section."""
    payload = _export(camera=CAMERA, camera_center=CENTER)

    assert "camera" in payload
    section = payload["camera"]
    assert section["position"] == pytest.approx([4.0, 5.0, 6.0])
    assert section["focal_point"] == pytest.approx([1.0, 2.0, 3.0])
    assert section["view_up"] == pytest.approx([0.0, 0.0, 1.0])
    assert section["parallel_projection"] == 0
    assert section["parallel_scale"] == pytest.approx(1.25)
    assert section["view_angle"] == pytest.approx(30.0)


def test_export_records_the_reference_nucleus() -> None:
    """The camera section remembers which atom it was captured around."""
    payload = _export(camera=CAMERA, camera_center=CENTER)

    assert payload["camera"]["atom_center"] == pytest.approx(list(CENTER))


def test_export_without_a_camera_omits_the_section() -> None:
    """The camera section is optional, so a file without one stays valid."""
    payload = _export()

    assert "camera" not in payload


def test_export_with_a_camera_but_no_center_is_rejected() -> None:
    """A camera without its reference nucleus cannot be re-based."""
    with pytest.raises(ValueError, match="nucleus"):
        _export(camera=CAMERA)


def test_exported_payload_is_json_serializable() -> None:
    """The whole snapshot, camera included, survives json.dumps."""
    payload = _export(camera=CAMERA, camera_center=CENTER)

    text = json.dumps(payload)
    assert json.loads(text)["camera"]["parallel_scale"] == pytest.approx(1.25)


# ---------------------------------------------------------------------------
# Load: the camera rides along
# ---------------------------------------------------------------------------


def test_load_returns_the_camera_section() -> None:
    """The loader hands the camera section back for the plan builder."""
    payload = _export(camera=CAMERA, camera_center=CENTER)

    params = load_params_from_dict(json.loads(json.dumps(payload)))

    assert "camera" in params
    assert params["camera"]["atom_center"] == pytest.approx(list(CENTER))


def test_load_without_a_camera_section_still_works() -> None:
    """Files written before the camera existed remain loadable."""
    payload = _export()

    params = load_params_from_dict(payload)

    assert "camera" not in params


def test_load_keeps_the_flat_controls() -> None:
    """The camera section must not leak into the flat control set."""
    payload = _export(camera=CAMERA, camera_center=CENTER)

    params = load_params_from_dict(payload)

    for key, expected in DEFAULT_UI_STATE.items():
        assert params[key] == expected
    assert params["selected_field"] == "Electron Density (condensed)"


# ---------------------------------------------------------------------------
# Plan: resolving what to restore
# ---------------------------------------------------------------------------


def test_plan_keeps_the_current_file_and_atom() -> None:
    """A snapshot of the open file needs no reload."""
    payload = _export()
    params = load_params_from_dict(payload)

    plan = resolve_restore_plan(
        params,
        current_plt_path=os.path.abspath(PLT),
        current_atom_number=1,
    )

    assert plan.plt_path is None
    assert plan.atom_number is None
    assert plan.requires_reload is False


def test_plan_requests_a_reload_for_a_different_file() -> None:
    """A snapshot naming another .plt asks for that file to be opened."""
    payload = _export()
    params = load_params_from_dict(payload)

    plan = resolve_restore_plan(
        params,
        current_plt_path="/somewhere/else.plt",
        current_atom_number=1,
    )

    assert plan.plt_path == os.path.abspath(PLT)
    assert plan.requires_reload is True


def test_plan_requests_a_reload_for_a_different_atom() -> None:
    """A snapshot naming another atom asks for that atom to be isolated."""
    payload = _export()
    params = load_params_from_dict(payload)

    plan = resolve_restore_plan(
        params,
        current_plt_path=os.path.abspath(PLT),
        current_atom_number=7,
    )

    assert plan.atom_number == 1
    assert plan.requires_reload is True


def test_plan_carries_the_controls() -> None:
    """The plan carries the control values to push into the state."""
    payload = _export(k_pitch=3.5, show_rings=False)
    params = load_params_from_dict(payload)

    plan = resolve_restore_plan(
        params,
        current_plt_path=os.path.abspath(PLT),
        current_atom_number=1,
    )

    assert plan.controls["k_pitch"] == pytest.approx(3.5)
    assert plan.controls["show_rings"] is False


def test_plan_falls_back_when_the_field_is_absent() -> None:
    """A field the target file lacks falls back to the first available one."""
    payload = _export()
    params = load_params_from_dict(payload)
    params["selected_field"] = "Not A Field"

    plan = resolve_restore_plan(
        params,
        current_plt_path=os.path.abspath(PLT),
        current_atom_number=1,
    )

    assert plan.selected_field == "Electron Density (condensed)"


def test_plan_rebases_the_camera_onto_the_origin() -> None:
    """The plan's camera is expressed relative to the atom's nucleus.

    That makes it portable: the viewer can re-base it onto whichever atom
    ends up loaded.
    """
    payload = _export(camera=CAMERA, camera_center=CENTER)
    params = load_params_from_dict(payload)

    plan = resolve_restore_plan(
        params,
        current_plt_path=os.path.abspath(PLT),
        current_atom_number=1,
    )

    assert plan.camera is not None
    assert plan.camera.position == pytest.approx((3.0, 3.0, 3.0))
    assert plan.camera.focal_point == pytest.approx((0.0, 0.0, 0.0))
    # Orientation and zoom are untouched by the translation.
    assert plan.camera.view_up == pytest.approx(CAMERA.view_up)
    assert plan.camera.parallel_scale == pytest.approx(CAMERA.parallel_scale)


def test_plan_without_a_camera_has_none() -> None:
    """A snapshot without a camera resolves to no camera."""
    params = load_params_from_dict(_export())

    plan = resolve_restore_plan(
        params,
        current_plt_path=os.path.abspath(PLT),
        current_atom_number=1,
    )

    assert plan.camera is None


def test_plan_camera_can_be_rebased_onto_the_live_atom() -> None:
    """The plan's camera can be moved onto a different nucleus."""
    payload = _export(camera=CAMERA, camera_center=CENTER)
    params = load_params_from_dict(payload)

    plan = resolve_restore_plan(
        params,
        current_plt_path=os.path.abspath(PLT),
        current_atom_number=1,
    )
    assert plan.camera is not None

    rebased = rebase_camera_state(plan.camera, (0.0, 0.0, 0.0), (10.0, 10.0, 10.0))

    assert rebased.position == pytest.approx((13.0, 13.0, 13.0))
    assert rebased.focal_point == pytest.approx((10.0, 10.0, 10.0))


# ---------------------------------------------------------------------------
# End to end on real data
# ---------------------------------------------------------------------------


def test_real_capture_export_load_plan_round_trip() -> None:
    """A camera captured from a real render window survives the round trip.

    This is the regression that matters: the numbers that come out of the
    export are the numbers that go back in.
    """
    from vtkmodules.vtkRenderingCore import vtkCamera

    camera = vtkCamera()
    camera.SetPosition(2.0, 1.0, 6.0)
    camera.SetFocalPoint(0.5, 0.5, 0.5)
    camera.SetViewUp(0.0, 0.0, 1.0)
    camera.SetParallelScale(2.5)
    captured = capture_camera_state(camera)

    payload = _export(camera=captured, camera_center=CENTER)
    params = load_params_from_dict(json.loads(json.dumps(payload)))
    plan = resolve_restore_plan(
        params,
        current_plt_path=os.path.abspath(PLT),
        current_atom_number=1,
    )

    assert plan.camera is not None
    assert plan.camera.view_up == pytest.approx(captured.view_up)
    assert plan.camera.parallel_scale == pytest.approx(captured.parallel_scale)
    assert plan.camera.view_angle == pytest.approx(captured.view_angle)

    # And the rebased camera still looks at the atom.
    restored = rebase_camera_state(plan.camera, (0.0, 0.0, 0.0), CENTER)
    assert restored.focal_point == pytest.approx(captured.focal_point)
    assert restored.position == pytest.approx(captured.position)


def test_a_saved_file_without_a_camera_still_resolves() -> None:
    """The two committed sample snapshots predate the camera and still load."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sample = os.path.join(root, "ethene4_atom1_params b.json")
    if not os.path.exists(sample):
        pytest.skip("sample snapshot not present")

    # The snapshot predates the separatrix control, so it is migrated first;
    # this test is about the missing camera section, not the schema version.
    params = load_params_from_dict(load_snapshot(sample))
    plan = resolve_restore_plan(
        params,
        current_plt_path=os.path.abspath(PLT),
        current_atom_number=1,
    )

    assert plan.controls
    assert plan.selected_field


def test_the_viewer_still_exports_the_camera() -> None:
    """The viewer's save path passes the camera through to the schema."""
    src = pathlib.Path(sv.__file__).read_text(encoding="utf-8")

    assert "camera=capture_current_camera()" in src
    assert "camera_center=session.center" in src
