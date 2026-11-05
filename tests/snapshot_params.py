"""Shared helpers for tests that load saved sphere_viewer parameter snapshots.

The ``*_params.json`` files in the repo root are operator-captured observations
written by the viewer's "Save Parameters" button. They are data, not fixtures we
regenerate, so they stay pinned to the schema version in force when they were
saved. ``load_params_from_dict`` rejects a stale version by design -- a missing
control key would otherwise surface as a confusing KeyError.

A control added after a snapshot was saved has exactly one sensible value for
that snapshot: its default. ``load_snapshot`` applies that reasoning explicitly,
so pipeline tests can run against real captured data without rewriting files on
disk, and the migration is stated in one place instead of being scattered across
test bodies.
"""

from __future__ import annotations

import json
from typing import Any

from gba_topology2.src.params_schema import (
    DEFAULT_UI_STATE,
    PARAM_CONTEXT_STRING_KEYS,
    PARAMS_SCHEMA_VERSION,
)

__all__ = ["load_snapshot"]


def load_snapshot(path: str) -> dict[str, Any]:
    """Read a saved parameter snapshot and migrate it to the current schema.

    Args:
        path: Path to a ``*_params.json`` file written by the viewer.

    Returns:
        The payload with ``schema_version`` set to the current version, any
        control added since the file was saved filled in from
        ``DEFAULT_UI_STATE``, and any control retired since then dropped (the
        loader ignores unknown keys, so carrying one is not a version gap). The
        ``params`` sections are created if absent, so a minimal hand-written
        fixture also loads.

    Raises:
        TypeError: If a known control carries a value of the wrong widget type.
    """
    with open(path, encoding="utf-8") as fh:
        payload: dict[str, Any] = json.load(fh)

    params = payload.setdefault("params", {})

    # "is a float" must exclude bool: isinstance(True, int) is True in Python,
    # so a plain (int, float) test would copy every switch into the floats.
    def _matches(value: Any, name: str) -> bool:
        if name == "bools":
            return isinstance(value, bool)
        if name == "floats":
            return isinstance(value, (int, float)) and not isinstance(value, bool)
        return isinstance(value, str)

    def _known(key: str) -> bool:
        # selected_field is a context string, not a DEFAULT_UI_STATE control:
        # its default is data-dependent and supplied at runtime.
        return key in DEFAULT_UI_STATE or key in PARAM_CONTEXT_STRING_KEYS

    for name in ("bools", "floats", "strings"):
        section = params.setdefault(name, {})
        # Retired controls stay in old files and are dropped here, exactly as
        # load_params_from_dict ignores them: removing a control is not a
        # breaking change, so a snapshot carrying one is still valid.
        for key, value in list(section.items()):
            if not _known(key):
                del section[key]
            elif not _matches(value, name):
                raise TypeError(f"Control '{key}' is not a {name[:-1]}: {value!r}")
        for key, default in DEFAULT_UI_STATE.items():
            if _matches(default, name) and key not in section:
                section[key] = default

    payload["schema_version"] = PARAMS_SCHEMA_VERSION
    return payload
