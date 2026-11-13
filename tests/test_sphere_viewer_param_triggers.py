"""Every analysis control must be registered as a recompute trigger.

The viewer recomputes the pipeline through one handler, ``on_param_change``,
which is subscribed to an explicit list of state keys with ``@state.change(...)``
and calls ``update_field()`` -> ``update_topology()``. The list is hand-maintained,
so a control added to ``DEFAULT_UI_STATE`` and read inside the pipeline can be
wired to the UI, written to snapshots, and consumed by the analysis, and still be
dead when toggled: no key change, no event, no recompute. The operator sees the
old geometry until an UNRELATED control fires the handler, which then picks up the
pending value -- so the toggle appears to work only in combination with another
switch. That is how ``coarse_ridge_curves`` (schema v7, "Coarse Structural Curves
(Cyan Ridges)") shipped: the pipeline reads it at the refinement call, and the key
was missing from the subscription list.

The registry is the single source of truth for the fix, so the guard is mechanical:
every control the viewer exposes must appear in the subscription list. Parsed from
source with ``ast`` because the handler lives in the ``run_sphere_viewer`` closure
and the list is decorator data -- importing and launching the app to read a
decorator would need a render window and a browser.
"""

from __future__ import annotations

import ast
import pathlib
from typing import Final

from gba_topology2.src.params_schema import DEFAULT_UI_STATE
from gba_topology2.src.sphere_viewer import (
    BASIN_ONLY_TOGGLES,
    PARAM_CONTEXT_STRING_KEYS,
    is_basin_only_change,
)

SOURCE: Final[pathlib.Path] = (
    pathlib.Path(__file__).resolve().parent.parent
    / "gba_topology2"
    / "src"
    / "sphere_viewer.py"
)

#: Runtime string that ``on_param_change`` calls when a watched key changes.
RECOMPUTE_CALL: Final[str] = "update_field"


def _state_change_keys() -> tuple[list[str], str]:
    """Read the ``@state.change(...)`` subscription lists out of the viewer source.

    Returns:
        Tuple ``(keys, handler_name)`` of the first ``state.change`` decorator
        found and the function it decorates.

    Raises:
        AssertionError: If the viewer has no ``@state.change`` subscription at all.
    """
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"), filename=str(SOURCE))
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        for dec in node.decorator_list:
            if (
                isinstance(dec, ast.Call)
                and isinstance(dec.func, ast.Attribute)
                and dec.func.attr == "change"
                and isinstance(dec.func.value, ast.Name)
                and dec.func.value.id == "state"
            ):
                keys = [
                    arg.value
                    for arg in dec.args
                    if isinstance(arg, ast.Constant) and isinstance(arg.value, str)
                ]
                return keys, node.name
    raise AssertionError("no @state.change(...) subscription found in sphere_viewer.py")


def _calls_recompute(handler_name: str) -> bool:
    """Return True when the subscribed handler calls the recompute entry point.

    Args:
        handler_name: Name of the function ``@state.change`` decorates.
    """
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"), filename=str(SOURCE))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == handler_name:
            body_nodes: list[ast.AST] = list(ast.walk(node))
            return any(
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Name)
                and call.func.id == RECOMPUTE_CALL
                for call in body_nodes
            )
    return False


class TestParamTriggerRegistry:
    """The subscription list covers every control the viewer exposes."""

    def test_basin_pipeline_is_fed_the_descent_table(self) -> None:
        """basin v10: the viewer must call find_minimum_basins(conn, centers, descents).

        The v9 two-argument call rebuilt every ring from spatial proximity --
        the rule that saturated. The third argument is the adjacency, so the
        short call is a behavior regression, not a style choice; it is pinned
        structurally on the source because the pipeline itself lives in a
        closure the AST harness can read and pytest cannot invoke.
        """
        tree = ast.parse(SOURCE.read_text(encoding="utf-8"), filename=str(SOURCE))
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "find_minimum_basins"
        ]
        assert calls, "the viewer no longer calls find_minimum_basins"
        for call in calls:
            assert len(call.args) == 3, (
                f"find_minimum_basins at line {call.lineno} takes "
                f"{len(call.args)} args; basin v10 requires the descent table"
            )

    def test_every_control_is_a_recompute_trigger(self) -> None:
        """A control that changes the analysis must fire the handler.

        This is the regression guard for the ``coarse_ridge_curves`` symptom: the
        switch, its schema default, its snapshot key and its pipeline consumer all
        existed; only the subscription was missing, so toggling it did nothing
        until an unrelated toggle forced a recompute.
        """
        keys, handler = _state_change_keys()
        watched = set(keys)
        unregistered = sorted(set(DEFAULT_UI_STATE) - watched)
        assert not unregistered, (
            f"{handler} is not subscribed to these viewer controls, so toggling "
            f"them cannot recompute the pipeline: {unregistered}"
        )

    def test_no_duplicate_keys_in_subscription(self) -> None:
        """A key listed twice is a merge artifact, not a second subscription."""
        keys, _handler = _state_change_keys()
        assert len(keys) == len(set(keys)), "duplicate key in @state.change(...)"

    def test_subscription_keys_are_known_state_names(self) -> None:
        """Every watched key is a control or a declared context string.

        Guards the other direction: a typo'd key subscribes to nothing and the
        control it was meant to cover stays dead.
        """
        keys, _handler = _state_change_keys()
        known = set(DEFAULT_UI_STATE) | set(PARAM_CONTEXT_STRING_KEYS)
        unknown = sorted(set(keys) - known)
        assert not unknown, f"@state.change(...) watches unknown state keys: {unknown}"

    def test_handler_calls_the_recompute_entry_point(self) -> None:
        """The subscribed handler is the one that rebuilds the scene.

        Registering the key on a handler that never recomputes would satisfy the
        key-list check and still leave the toggle dead, which is the failure mode
        this file exists to catch.
        """
        _keys, handler = _state_change_keys()
        assert _calls_recompute(handler), (
            f"{handler} does not call {RECOMPUTE_CALL}(), so a change to a "
            "watched key cannot recompute the analysis"
        )

    def test_ridge_cards_share_one_subscription(self) -> None:
        """The three Step 3 ridge switches are watched side by side.

        ``refine_ridge_paths``, ``coarse_ridge_curves`` and ``probe_ridge_points``
        are one feature family reading one call in ``update_topology``; the card
        is only predictable if all three behave identically.
        """
        keys, _handler = _state_change_keys()
        for name in ("refine_ridge_paths", "coarse_ridge_curves", "probe_ridge_points"):
            assert name in keys, f"{name} is not a recompute trigger"


class TestBasinOnlyChangePredicate:
    """Basin overlay toggles draw without a pipeline rebuild.

    Regression for the "boundaries flash, then vanish" symptom: the basin
    toggles ran compute_and_show_basin (draw) and then update_field (rebuild),
    and the rebuild's final select_basin_center re-emitted the selection arcs
    over the basin ring. The observer now early-returns for basin-only batches,
    so the draw is the LAST write to the arc actor.
    """

    def test_each_basin_toggle_alone_is_basin_only(self) -> None:
        for key in BASIN_ONLY_TOGGLES:
            assert is_basin_only_change({key}), key

    def test_both_toggles_together_is_basin_only(self) -> None:
        assert is_basin_only_change(set(BASIN_ONLY_TOGGLES))

    def test_mixed_batch_rebuilds(self) -> None:
        """A batch that also changes a pipeline control must not skip the
        rebuild — the non-basin control in the same flush is real work."""
        assert not is_basin_only_change({"show_all_minimum_basins", "k_pitch"})

    def test_empty_batch_is_not_basin_only(self) -> None:
        """An empty change set is not a basin change (no draw, no rebuild)."""
        assert not is_basin_only_change(set())

    def test_toggles_are_registered_as_triggers(self) -> None:
        """The predicate's keys must be the real subscribed control names."""
        keys, _handler = _state_change_keys()
        assert set(BASIN_ONLY_TOGGLES) <= set(keys)

    def test_observer_uses_the_predicate(self) -> None:
        """on_param_change must branch on the predicate, not re-derive it.

        Guards the wiring: the ast shape of the handler's basin branch is
        ``if is_basin_only_change(...)`` — an inline set comparison drifting
        from BASIN_ONLY_TOGGLES is exactly how this bug class ships twice.
        """
        tree = ast.parse(SOURCE.read_text(encoding="utf-8"), filename=str(SOURCE))
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "on_param_change":
                calls = [
                    c.func.id
                    for c in ast.walk(node)
                    if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)
                ]
                assert "is_basin_only_change" in calls
                return
        raise AssertionError("on_param_change not found in sphere_viewer.py")
