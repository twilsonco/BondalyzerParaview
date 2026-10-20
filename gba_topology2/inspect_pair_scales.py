"""Read-only diagnostic: measure the scale separation of consolidated entities.

Tests the three-scale hypothesis:

    L1 (micro)  mesh chatter, absorbed by micro-clustering   (~k_pitch dtheta)
    L2 (meso)   corridor fragmentation, absorbed by bridging (~k_bridge dtheta)
    L3 (macro)  distinct physical basins, which must NOT merge

For each selected field this builds the pipeline's own entities -- micro-clusters
at ``--k-pitch``, then bridged reduction units at ``--k-bridge`` -- and measures
every pair of entities by centroid separation and by directional edge clearance,
both normalised by the mesh pitch. The output is a terminal histogram per
measure plus a registry of the pairs that sit in the transition zone, which are
exactly the decisions a macro-clustering rule would have to make.

The measurement runs BEFORE catastrophe classification and before the effective
CP reduction, so the distribution reflects the geometry of the field rather than
the shape imposed by the constellation templates.

Run with:
    uv run python gba_topology2/inspect_pair_scales.py ethene3.plt
    uv run python gba_topology2/inspect_pair_scales.py Pd_20K.plt -f curvedness
    uv run python gba_topology2/inspect_pair_scales.py Pd_20K.plt --all --svg
    uv run python gba_topology2/inspect_pair_scales.py ethene3.plt --list-fields
"""

import argparse
import os
import re
import sys

import numpy as np
from vtkmodules.util import numpy_support

workspace_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)

from gba_topology2.src.mesh_geometry import analyze_mesh_geometry
from gba_topology2.src.micro_cluster import compute_micro_clusters
from gba_topology2.src.morse_detector import compute_discrete_morse_cps
from gba_topology2.src.pair_scales import (
    compute_pairwise_clearances,
    render_ascii_histogram,
    render_intermediate_registry,
    write_svg_histogram,
)
from gba_topology2.src.sphere_viewer import (
    build_reduction_units,
    extract_atom_sphere_mesh,
)

# The fields worth reading by default. Matched case-insensitively against the
# discovered array names, which carry non-ASCII prefixes (for example the
# curvedness array on Pd_20K is "Ï\x81 curvedness (condensed)"), so substring
# matching is required and exact equality is not. The volume field is stored as
# plain "V", so it is selected by the single-letter pattern "v", which matches
# only on the exact base name (see _field_matches) -- as a substring it would
# also select "curvedness" and "distance".
DEFAULT_FIELD_PATTERNS: tuple[str, ...] = (
    "density",
    "v",
    "shape",
    "curvedness",
    "willmore",
)


def _field_matches(name: str, pattern: str) -> bool:
    """Test whether a field array name is described by a pattern.

    Multi-character patterns match as case-insensitive substrings. Single-letter
    patterns must equal the base name -- the array name with the ``(condensed)``
    suffix and any non-ASCII field prefix stripped -- because a single letter as
    a substring is far too loose to be useful.

    Args:
        name: Discovered array name.
        pattern: Case-insensitive user pattern.

    Returns:
        True when the pattern describes this field.
    """
    low = name.lower()
    key = pattern.lower().strip()
    if not key:
        return False
    if len(key) == 1:
        base = low.replace("(condensed)", "")
        base = "".join(ch for ch in base if ch.isascii() and ch.isalnum())
        return base.strip() == key
    return key in low


def select_fields(
    field_items: list[dict[str, str]],
    patterns: list[str],
    take_all: bool,
) -> list[str]:
    """Pick the array names to analyse.

    Args:
        field_items: Discovered fields, each with ``title`` and ``value`` keys.
        patterns: Case-insensitive substrings, or exact base names; a field
            matches if any pattern describes it. Ignored when ``take_all``.
        take_all: Select every discovered field.

    Returns:
        Matching array names, in discovery order.
    """
    names = [str(item["value"]) for item in field_items]
    if take_all:
        return names
    picked: list[str] = []
    for pattern in patterns:
        for name in names:
            if _field_matches(name, pattern) and name not in picked:
                picked.append(name)
    return picked


def _slug(name: str) -> str:
    """Build a filesystem-safe stem from a field array name.

    Args:
        name: Field array name, possibly containing non-ASCII characters.

    Returns:
        Lowercase ASCII slug of letters, digits, and underscores.
    """
    ascii_only = name.encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", "_", ascii_only.lower()).strip("_")[:48]


def main() -> None:
    """Run the diagnostic over the requested fields and print the histograms."""
    parser = argparse.ArgumentParser(prog="inspect_pair_scales.py")
    parser.add_argument("plt_file", nargs="?", default="Pd_20K.plt")
    parser.add_argument("-a", "--atom", type=int, default=1)
    parser.add_argument(
        "-f",
        "--fields",
        nargs="+",
        default=None,
        help="case-insensitive field substrings (default: the five core fields)",
    )
    parser.add_argument(
        "--all", action="store_true", help="analyse every discovered field"
    )
    parser.add_argument("--list-fields", action="store_true")
    parser.add_argument("--k-pitch", type=float, default=2.0)
    parser.add_argument("--k-bridge", type=float, default=3.0)
    parser.add_argument("--bin-width", type=float, default=0.5)
    parser.add_argument("--max-value", type=float, default=12.0)
    parser.add_argument(
        "--svg", action="store_true", help="write an SVG chart per field"
    )
    args, _unknown = parser.parse_known_args()

    sphere_poly, field_items, atom_info = extract_atom_sphere_mesh(
        args.plt_file, target_atom_num=args.atom
    )

    if args.list_fields:
        print(f"fields in {args.plt_file} (atom {args.atom}):")
        for item in field_items:
            print(f"  {item['value']}")
        return

    patterns = args.fields or list(DEFAULT_FIELD_PATTERNS)
    selected = select_fields(field_items, patterns, args.all)
    if not selected:
        print(f"No field matched {patterns}. Try --list-fields.")
        return

    R = float(atom_info["sphere_radius"])
    pts_np = numpy_support.vtk_to_numpy(sphere_poly.GetPoints().GetData())
    n_cells = sphere_poly.GetNumberOfPolys()
    cell_arr = numpy_support.vtk_to_numpy(sphere_poly.GetPolys().GetData())
    tris = cell_arr.reshape((n_cells, 4))[:, 1:4]
    geom = analyze_mesh_geometry(pts_np, tris, sphere_radius=R)
    dth = float(geom["mesh_pitch_rad"])

    stem = _slug(os.path.basename(args.plt_file))
    print("=" * 78)
    print(f"file  : {args.plt_file}")
    print(f"atom  : {atom_info['atom_name']}   R = {R:.4f} A")
    print(
        f"mesh  : {len(pts_np)} verts, dtheta = {np.degrees(dth):.4f} deg "
        f"({dth:.6f} rad)"
    )
    print(f"scales: k_pitch = {args.k_pitch}, k_bridge = {args.k_bridge}")
    print(f"fields: {len(selected)} selected")
    print("=" * 78)

    point_data = sphere_poly.GetPointData()
    for field in selected:
        arr = point_data.GetArray(field)
        if arr is None:
            print(f"\n[skip] {field}: array not found")
            continue
        f_vals = numpy_support.vtk_to_numpy(arr).astype(np.float64)

        morse = compute_discrete_morse_cps(
            pts_np, tris, f_vals, tie_break_epsilon=1e-15
        )
        all_raw = morse.minima + morse.maxima + morse.saddles
        n_extrema = len(morse.maxima) + len(morse.minima)
        saddle_index = sum(cp.multiplicity for cp in morse.saddles)
        euler = n_extrema - saddle_index
        clusters = compute_micro_clusters(
            cps=all_raw,
            delta_theta_mesh_rad=dth,
            k_pitch=args.k_pitch,
            sphere_radius=R,
        )
        units, n_bridged = build_reduction_units(
            clusters=clusters,
            delta_theta_mesh_rad=dth,
            sphere_radius=R,
            aggregate_strings=True,
            k_bridge=args.k_bridge,
        )
        entities = [u.as_cluster for u in units]
        records = compute_pairwise_clearances(entities, dth)

        print(f"\n### FIELD: {field}")
        print(
            f"    CPs {len(all_raw)} (max {len(morse.maxima)} / min "
            f"{len(morse.minima)} / sad {len(morse.saddles)})  chi {euler}"
        )
        print(
            f"    clusters {len(clusters)} -> units {len(units)} "
            f"({n_bridged} bridged)  pairs {len(records)}"
        )
        if len(records) < 2:
            print("    too few entities to distribute")
            continue

        gaps = [r.gap_dtheta for r in records]
        centroids = [r.centroid_dtheta for r in records]
        print()
        print(
            render_ascii_histogram(
                gaps,
                "DIRECTIONAL EDGE CLEARANCE",
                bin_width=args.bin_width,
                max_value=args.max_value,
            )
        )
        print()
        print(
            render_ascii_histogram(
                centroids,
                "CENTROID SEPARATION",
                bin_width=args.bin_width,
                max_value=args.max_value,
            )
        )
        print()
        print(render_intermediate_registry(records))

        if args.svg:
            out = f"{stem}_atom{args.atom}_{_slug(field)}_scales.svg"
            write_svg_histogram(
                gaps,
                centroids,
                f"{atom_info['atom_name']} | {field}",
                out,
                bin_width=args.bin_width,
                max_value=args.max_value,
            )
            print(f"SVG written: {out}")


if __name__ == "__main__":
    main()
