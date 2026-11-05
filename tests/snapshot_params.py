"""Load saved sphere_viewer parameter snapshots in tests.

The ``*_params.json`` files in the repo root are operator-captured observations
written by the viewer's "Save Parameters" button. They are data, not fixtures we
regenerate, so they stay pinned to the schema version in force when they were
saved. The viewer's loader rejects a stale version by design; the headless
harness migrates first (``params_schema.migrate_params_payload``), and tests use
the same path so a pipeline test runs against real captured data.

This module is the thin file-reading wrapper over that migration.
"""

from __future__ import annotations

import json
from typing import Any

from gba_topology2.src.params_schema import migrate_params_payload

__all__ = ["load_snapshot"]


def load_snapshot(path: str) -> dict[str, Any]:
    """Read a saved parameter snapshot and migrate it to the current schema.

    Args:
        path: Path to a ``*_params.json`` file written by the viewer.

    Returns:
        The payload with ``schema_version`` set to the current version and any
        control added since the file was saved filled in from its default.
        See ``params_schema.migrate_params_payload``.

    Raises:
        TypeError: If a known control carries a value of the wrong widget type.
    """
    with open(path, encoding="utf-8") as fh:
        payload: dict[str, Any] = json.load(fh)
    return migrate_params_payload(payload)
