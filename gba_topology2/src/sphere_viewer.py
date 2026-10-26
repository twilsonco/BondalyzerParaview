"""Sphere Viewer Subroutine for gba_topology2.

Step 2 Interactive Viewer:
- Displays atom sphere, raw critical points (minima, maxima, saddles), and micro-cluster boundary rings.
- Interactive 3D Picking: Clicking directly on a cluster or CP in the 3D viewport or sidebar table
  immediately opens its inspector card displaying all constituent CPs, coordinates, scalar values,
  angular radius, and local Euler index.
- Clean separation from algorithmic subroutines: imports mesh_geometry, morse_detector, and micro_cluster.

Run with:
    uv run python -m gba_topology2.src.sphere_viewer Pd_20K.plt
or:
    uv run python gba_topology2/src/sphere_viewer.py Pd_20K.plt
"""

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

# VTK Imports
from vtkmodules.util import numpy_support
from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkPolyData
from vtkmodules.vtkFiltersCore import vtkContourFilter, vtkGlyph3D, vtkTubeFilter
from vtkmodules.vtkFiltersSources import vtkCubeSource, vtkSphereSource
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera
from vtkmodules.vtkInteractionWidgets import vtkOrientationMarkerWidget
from vtkmodules.vtkRenderingAnnotation import vtkAxesActor
from vtkmodules.vtkRenderingCore import (
    vtkActor,
    vtkCellPicker,
    vtkColorTransferFunction,
    vtkCoordinate,
    vtkPolyDataMapper,
    vtkRenderer,
    vtkRenderWindow,
    vtkRenderWindowInteractor,
)

# Trame Imports
try:
    from trame.app import get_server
    from trame.ui.vuetify3 import SinglePageWithDrawerLayout
    from trame.widgets import html
    from trame.widgets import vuetify3 as v3
    from trame.widgets.vtk import VtkRemoteView
except ImportError:
    from trame.app import get_server  # type: ignore
    from trame.ui.vuetify import SinglePageWithDrawerLayout  # type: ignore
    from trame.widgets import html  # type: ignore
    from trame.widgets import vuetify as v3  # type: ignore
    from trame.widgets.vtk import VtkRemoteView  # type: ignore

# Add workspace parent dir so imports resolve cleanly
workspace_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)

from BondalyzerParaView.plt_gba_to_vtm import extract_gba_zones_from_plt
from BondalyzerParaView.trame_viewer import get_display_title, get_robust_scalar_bounds
from gba_topology2.src.basin_center import (
    extract_basin_centers,
)
from gba_topology2.src.camera_state import (
    CameraState,
    apply_camera_state,
    capture_camera_state,
    rebase_camera_state,
)
from gba_topology2.src.catastrophe_classifier import (
    EffectiveCatastrophe,
    classify_catastrophes,
)
from gba_topology2.src.excision_boundary import (
    profile_boundary_loop,
)
from gba_topology2.src.harmonic_nudge import (
    harmonic_nudge_critical_point,
    perfect_boundary_ports,
    reposition_ports_on_ring,
)
from gba_topology2.src.mesh_geometry import analyze_mesh_geometry
from gba_topology2.src.micro_cluster import (
    BoundaryRingPoint,
    MicroCluster,
    compute_micro_clusters,
)
from gba_topology2.src.morse_detector import CriticalPoint, compute_discrete_morse_cps
from gba_topology2.src.params_loader import RestorePlan, resolve_restore_plan
from gba_topology2.src.params_schema import (
    DEFAULT_UI_STATE,
    PARAM_BOOL_KEYS,
    PARAM_CONTEXT_STRING_KEYS,
    PARAM_FLOAT_KEYS,
    PARAM_METRIC_KEYS,
    PARAM_STRING_KEYS,
    PARAMS_SCHEMA_VERSION,
    default_params_path,
    load_params_from_dict,
    save_params_to_dict,
)
from gba_topology2.src.polarity_detector import (
    classify_cluster_polarity,
    effective_reduction_decision,
)
from gba_topology2.src.saddle_launch import (
    SaddleLaunchSeed,
    compute_all_saddle_launch_seeds,
)
from gba_topology2.src.sphere_frame import (
    compute_atom_frame,
    radial_offset,
    to_global,
    to_global_array,
    to_local,
)
from gba_topology2.src.string_aggregator import (
    MacroString,
    aggregate_cluster_strings,
)

__all__ = [
    "DEFAULT_UI_STATE",
    "PARAMS_SCHEMA_VERSION",
    "PARAM_BOOL_KEYS",
    "PARAM_CONTEXT_STRING_KEYS",
    "PARAM_FLOAT_KEYS",
    "PARAM_METRIC_KEYS",
    "PARAM_STRING_KEYS",
    "VUE3_THEME_CONFIG",
    "ReductionUnit",
    "build_reduction_units",
    "catastrophe_claimed_cluster_ids",
    "default_params_path",
    "extract_atom_sphere_mesh",
    "fusable_catastrophes",
    "load_params_from_dict",
    "macro_string_to_cluster",
    "run_sphere_viewer",
    "save_params_to_dict",
]

# Vuetify theme handed to createVuetify() through the layout's
# ``vuetify_config=`` argument (which stores it as ``trame__vuetify3_config``).
#
# Why a theme and not CSS overrides: the drawer previously carried an
# ``html.Style`` block forcing dark cards with ``!important``. Vue strips
# ``<style>`` from component templates, so that block never reached the document
# and the app rendered entirely in Vuetify's stock *light* theme -- where
# ``surface-variant`` is a #424242 panel, emphasis text is black at 0.6 opacity
# (1.73:1 on that panel), and the Material accents the cards name (``.text-teal``
# #009688, ``.text-grey`` #9e9e9e) are light-theme mid-tones. That is the
# grey-on-grey this palette removes.
#
# Vuetify generates ``.text-{color}`` / ``.bg-{color}`` rules from this palette
# into its own stylesheet, injected after the static CSS and with ``!important``
# at equal specificity -- so a palette entry wins over the static Material rule.
# Naming the accents here therefore fixes all of them at once, with no per-widget
# edits.
#
# Every value is the darkest shade of its own hue family that clears WCAG AA
# (4.5:1) in *both* roles a color plays, measured against every surface it can
# land on:
#   - as text (``text-teal`` on a card), and
#   - as a chip label, where VChip defaults to ``variant="tonal"``: the label is
#     the accent and the background is that same accent at 12% over the card, so
#     the accent must survive its own tint.
# Hues are deliberately preserved, so teal still reads as teal.
# ``medium-emphasis-opacity`` is raised from the stock 0.6 to 0.80, which lifts
# the ~50 ``text-medium-emphasis`` labels from 1.7:1 to 11.8:1 in one place.
VUE3_THEME_CONFIG: dict[str, Any] = {
    "defaultTheme": "light",
    # Cards are white-on-white in a light theme, so they need the edge the dark
    # theme gave for free. Setting it as a Vuetify *default* applies it to every
    # VCard from one place; the border color comes from the theme's
    # ``border-color``/``border-opacity`` variables below.
    "defaults": {"VCard": {"border": True}},
    "theme": {
        "themes": {
            "light": {
                "dark": False,
                "colors": {
                    # Neutrals. surface-variant is the card fill used by every
                    # step card; the stock light value (#424242) is a dark panel
                    # that only makes sense in a dark theme.
                    "background": "#FFFFFF",
                    "on-background": "#000000",
                    "surface": "#FFFFFF",
                    "on-surface": "#000000",
                    "surface-bright": "#FFFFFF",
                    "surface-light": "#E8EBF0",
                    "surface-variant": "#F2F4F7",
                    "on-surface-variant": "#1A1A1A",
                    # Semantic accents.
                    "primary": "#1256A3",
                    "secondary": "#006256",
                    "tertiary": "#6A1B9A",
                    "error": "#A5231C",
                    "info": "#0B4F9E",
                    "success": "#1B5E20",
                    "warning": "#7A4100",
                    # Material accents named by the cards. Without these entries
                    # the static Material CSS supplies light-theme mid-tones that
                    # fail on white (.text-teal is #009688, .bg-grey is #9e9e9e).
                    "teal": "#006256",
                    "cyan": "#006064",
                    "purple": "#6A1B9A",
                    "amber": "#7A4100",
                    "orange": "#7A4100",
                    "deep-orange": "#9F2D0A",
                    "grey": "#5A5A5A",
                    # trame's layout footer hardcodes ``text-grey-lighten-1`` on
                    # its "Powered by trame" links. The static Material value
                    # (#bdbdbd) is legible on the dark footer it was written for
                    # and 1.88:1 on a white one, so the palette overrides it.
                    # Kept lighter than ``grey`` so the ramp order holds.
                    "grey-lighten-1": "#6A6A6A",
                    "blue": "#1256A3",
                    "green": "#1B5E20",
                    "red": "#A5231C",
                    # Explicit -darken-N entries for the chips the source names,
                    # so the result does not depend on Vuetify's generated
                    # variation arithmetic. Ordered so each -darken-N stays darker
                    # than its base.
                    "amber-darken-2": "#6D5200",
                    "amber-darken-3": "#874400",
                    "orange-darken-2": "#913C00",
                    "deep-orange-darken-1": "#993414",
                    "cyan-darken-2": "#006069",
                    "purple-darken-2": "#7B1FA2",
                    "grey-darken-1": "#4A4A4A",
                },
                "variables": {
                    "border-color": "#000000",
                    "border-opacity": 0.24,
                    "high-emphasis-opacity": 0.87,
                    "medium-emphasis-opacity": 0.80,
                    "disabled-opacity": 0.60,
                },
            }
        }
    },
}


def _parse_aux_radius(aux: dict[str, Any]) -> float | None:
    """Read an authoritative sphere radius from a zone's auxiliary metadata.

    Args:
        aux: Zone auxiliary key/value map from the PLT header.

    Returns:
        The Aux SphereRadius in Angstrom, or None when absent, unparsable, or
        not strictly positive.
    """
    raw = aux.get("SphereRadius")
    if raw is None:
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    return value if value > 0.0 else None


def extract_atom_sphere_mesh(
    plt_path: str, target_atom_num: int = 1
) -> tuple[vtkPolyData, list[dict[str, str]], dict[str, Any]]:
    """Extract the AtomSphereData mesh and condensed scalar fields from a .plt file.

    The returned mesh is translated into the atom's own frame: its vertices are
    centre-relative, so the nucleus sits at (0, 0, 0) and every angular
    formula in the pipeline (normals ``p / ||p||``, tangent planes, great-circle
    rings, ``dir * R`` reconstructions) is correct without knowing where the
    atom lives in the molecule. ``atom_info["center"]`` carries the world-frame
    nucleus, and ``sphere_frame.to_global`` maps coordinates back for drawing
    and display. An isolated atom centred on the origin yields center ~ 0, so
    the pipeline is unchanged for it.

    Args:
        plt_path: Path to the .plt file.
        target_atom_num: Atom number (1-based index).

    Returns:
        Tuple of (sphere_poly, field_items, atom_info), where sphere_poly is in
        the centred atom frame and atom_info records the world-frame centre,
        its distance from the origin, the sphere radius and where that radius
        came from ("metadata" = the zone's Aux SphereRadius, "fitted" = the
        mean centre-relative vertex distance).
    """
    abs_plt = os.path.abspath(plt_path)
    if not os.path.exists(abs_plt):
        raise FileNotFoundError(f"PLT file not found: {abs_plt}")

    gba_mb, gba_meta = extract_gba_zones_from_plt(
        abs_plt,
        output_vtm=None,
        include_sphere_patches=False,
        include_surfaces=False,
        include_atom_spheres=True,
    )

    sphere_poly: vtkPolyData | None = None
    atom_symbol = "C"
    atom_aux: dict[str, Any] = {}

    for entry in gba_meta:
        ztype = entry.get("zone_type")
        block_idx = entry.get("block_index")
        if block_idx is None:
            continue
        poly = gba_mb.GetBlock(int(block_idx))
        if not isinstance(poly, vtkPolyData) or poly.GetNumberOfPoints() == 0:
            continue

        atom_nr = entry.get("atom_number", 1)
        try:
            atom_nr = int(atom_nr)
        except (ValueError, TypeError):
            atom_nr = 1

        if atom_nr != target_atom_num:
            continue

        if ztype == "AtomSphereData":
            sphere_poly = poly
            atom_symbol = entry.get("atom_type", "C")
            atom_aux = dict(entry.get("aux", {}))
            break

    if sphere_poly is None:
        raise RuntimeError(
            f"Could not find AtomSphereData for Atom #{target_atom_num} in {plt_path}"
        )

    # Discover condensed surface fields
    pd = sphere_poly.GetPointData()
    discovered_fields: list[str] = []
    for i in range(pd.GetNumberOfArrays()):
        aname = pd.GetArrayName(i)
        if not aname:
            continue
        lower_a = aname.lower().strip()
        is_condensed = "(condensed)" in lower_a or lower_a in (
            "v",
            "v (condensed)",
            "î±",
            "α",
            "alpha",
            "trajectory parameter",
        )
        is_ignored = aname in (
            "X",
            "Y",
            "Z",
            "RGBColor",
            "Normals",
            "atomic_number",
            "AtomicNumber",
        )
        if is_condensed and not is_ignored and aname not in discovered_fields:
            discovered_fields.append(aname)

    if not discovered_fields:
        discovered_fields = ["Electron Density (condensed)"]

    field_items = [
        {"title": get_display_title(f), "value": f} for f in discovered_fields
    ]
    atom_nodes = int(sphere_poly.GetNumberOfPoints())
    num_tris = int(sphere_poly.GetNumberOfPolys())

    # Recover the atom's own frame and translate the mesh into it. The numpy
    # array is a view on the VTK points, so the subtraction recentres the mesh
    # in place; scalar point data is unaffected by a translation.
    pts_arr = numpy_support.vtk_to_numpy(sphere_poly.GetPoints().GetData())
    center, fitted_radius = compute_atom_frame(pts_arr)
    pts_arr -= center
    sphere_poly.GetPoints().Modified()
    sphere_poly.Modified()

    meta_radius = _parse_aux_radius(atom_aux)
    if meta_radius is not None:
        sphere_radius = meta_radius
        radius_source = "metadata"
        if abs(meta_radius - fitted_radius) / fitted_radius > 0.01:
            print(
                f"[sphere_viewer] Aux SphereRadius {meta_radius:.6f} A disagrees "
                f"with the fitted radius {fitted_radius:.6f} A for "
                f"{atom_symbol}{target_atom_num}; using the metadata value."
            )
    else:
        sphere_radius = fitted_radius
        radius_source = "fitted"

    atom_info = {
        "atom_number": target_atom_num,
        "atom_symbol": atom_symbol,
        "atom_name": f"{atom_symbol}{target_atom_num}",
        "num_nodes": atom_nodes,
        "num_triangles": num_tris,
        "sphere_radius": sphere_radius,
        "fitted_radius": fitted_radius,
        "radius_source": radius_source,
        # World-frame nucleus. The returned mesh is centre-relative; add this
        # back with sphere_frame.to_global for anything drawn or displayed.
        "center": [float(center[0]), float(center[1]), float(center[2])],
        "center_offset": float(np.linalg.norm(center)),
    }

    return sphere_poly, field_items, atom_info


def build_discrete_colormap(
    n_colors: int, f_min: float, f_max: float
) -> vtkColorTransferFunction:
    """Build smooth Viridis transfer function."""
    ctf = vtkColorTransferFunction()
    anchors = [
        (0.00, 0.267, 0.004, 0.329),  # dark purple
        (0.25, 0.190, 0.407, 0.556),  # dark blue
        (0.50, 0.127, 0.566, 0.550),  # teal
        (0.75, 0.369, 0.788, 0.382),  # bright green
        (1.00, 0.993, 0.906, 0.143),  # bright yellow
    ]
    n_steps = max(2, min(n_colors, 64))
    for step in range(n_steps):
        frac = step / (n_steps - 1)
        for a_i in range(len(anchors) - 1):
            f0, r0, g0, b0 = anchors[a_i]
            f1, r1, g1, b1 = anchors[a_i + 1]
            if f0 <= frac <= f1:
                t = (frac - f0) / (f1 - f0) if f1 > f0 else 0.0
                r = r0 + t * (r1 - r0)
                g = g0 + t * (g1 - g0)
                b = b0 + t * (b1 - b0)
                break
        else:
            r, g, b = anchors[-1][1:]

        val = f_min + frac * (f_max - f_min)
        ctf.AddRGBPoint(val, r, g, b)
    return ctf


_RAW_TYPE_META: dict[str, tuple[str, str]] = {
    "maximum": ("Maximum", "error"),
    "minimum": ("Minimum", "info"),
    "saddle": ("Saddle", "success"),
}

_EFF_TYPE_META: dict[str, tuple[str, str]] = {
    "E-MAX": ("Effective Maximum", "error"),
    "E-MIN": ("Effective Minimum", "info"),
    "E-SAD": ("Effective Saddle", "success"),
}


def catastrophe_claimed_cluster_ids(
    catastrophes: list[EffectiveCatastrophe],
    reduce_catastrophes: bool,
) -> set[str]:
    """Return the cluster IDs whose reduction is owned by the catastrophe stage.

    The Step 6 pipeline reduces a micro-cluster in exactly one place. A cluster
    absorbed into a catastrophe constellation is collapsed by the catastrophe
    stage (into a single ``E-SAD`` at the constellation centroid), so the
    standalone multi-cluster stage must skip it to avoid emitting a duplicate
    effective CP and double-counting its index.

    When ``reduce_catastrophes`` is off, the catastrophe stage collapses
    nothing. Claiming those clusters anyway would strand them: the cluster
    stage would skip them and no effective CP would ever be emitted, so a
    symmetric set carrying a real net index (e.g. a chi = -1 saddle string)
    silently disappears from the reduced view. Ownership therefore follows the
    toggle that actually performs the collapse.

    Only *fused* constellations are collapsed, so only they claim their
    constituents (see ``fusable_catastrophes``). A split constellation is
    left standing, which means its clusters stay eligible for standalone
    reduction -- that is what keeps the extrema of a split constellation's core
    cluster visible instead of being swallowed.

    Args:
        catastrophes: Detected catastrophe constellations.
        reduce_catastrophes: Whether the catastrophe reduction toggle is active.

    Returns:
        Set of ``MicroCluster.cluster_id`` values to exclude from standalone
        multi-cluster reduction. Empty when catastrophe reduction is disabled.
    """
    if not reduce_catastrophes:
        return set()

    claimed: set[str] = set()
    for cat in fusable_catastrophes(catastrophes):
        for cl in cat.constituent_clusters:
            claimed.add(cl.cluster_id)
    return claimed


def fusable_catastrophes(
    catastrophes: list[EffectiveCatastrophe],
) -> list[EffectiveCatastrophe]:
    """Return the catastrophes that may be collapsed into one effective saddle.

    A catastrophe is collapsed only when it is *fused* -- i.e. classified as a
    ``spurious_unfolding``. A fused constellation is a numerical artefact: its
    poles are a single genuine critical point that the mesh happens to split
    into several, so collapsing them into one ``E-SAD`` removes the artefact
    without destroying real structure.

    A *split* constellation (``physical_symmetry_breaking``) is the opposite
    case. Its members are genuinely distinct critical points that a real
    symmetry breaking has separated; the classifier deliberately keeps them
    apart. Collapsing them into a single synthetic saddle at the constellation
    centroid fabricates a critical point that does not exist on the mesh, and
    -- because the core cluster of such a constellation can itself be a
    composite extremum set -- it hides real extrema as well. On the ethene4
    Electron Density field this swallowed the field's global maximum: the core
    cluster MC3 (2 Max + 1 Sad) was absorbed into an ``E-SAD`` whose centroid is
    not a critical point, and the maximum glyph vanished.

    The spurious / physical distinction is therefore the gate, and the
    classification must be honoured rather than merely displayed.

    Args:
        catastrophes: Detected catastrophe constellations.

    Returns:
        The subset of catastrophes that may be collapsed into one effective
        saddle each. Empty when every constellation is a genuine split.
    """
    return [cat for cat in catastrophes if cat.is_fused]


@dataclass(frozen=True)
class ReductionUnit:
    """One set of critical points reduced to at most one effective CP.

    A unit is either a single micro-cluster or a macro-string bridged from
    several collinear micro-clusters along one ridge/valley corridor. The Step 6
    reduction loop consumes units rather than raw clusters so that a corridor
    fragmented by the isotropic clustering radius collapses to a single
    effective CP carrying the summed index of all its fragments.

    Attributes:
        unit_id: Display identifier (``MC12`` for a lone cluster, ``MS1`` for a
            bridged corridor).
        origin_ids: Cluster IDs folded into this unit.
        members: Every critical point in the unit, de-duplicated by vertex id.
        centroid: (3,) Euclidean centroid on the sphere.
        centroid_unit: (3,) unit centroid direction.
        local_euler_index: Net index of the whole unit.
        n_maxima: Count of maximum members.
        n_minima: Count of minimum members.
        n_saddles: Count of saddle members.
        total_saddle_index: Multiplicity-weighted saddle index.
        boundary_ring: Ring enclosing the whole unit.
        is_string: True when the unit bridges two or more clusters.
        morphology: ``"string"`` or ``"compact"``.
        aspect_ratio: Tangent-plane anisotropy of the unit.
        spatial_diameter: Extent of the unit across its members (Angstrom).
        as_cluster: MicroCluster view of the unit, so subroutines written
            against the cluster interface (catastrophe classification, polarity
            probing) consume a bridged corridor unchanged.
    """

    unit_id: str
    origin_ids: tuple[str, ...]
    members: list[CriticalPoint]
    centroid: tuple[float, float, float]
    centroid_unit: tuple[float, float, float]
    local_euler_index: int
    n_maxima: int
    n_minima: int
    n_saddles: int
    total_saddle_index: int
    boundary_ring: list[BoundaryRingPoint]
    is_string: bool
    morphology: str
    aspect_ratio: float
    spatial_diameter: float
    as_cluster: MicroCluster

    @property
    def composition_summary(self) -> str:
        """Human-readable composition string for the sidebar table."""
        parts: list[str] = []
        if self.n_maxima > 0:
            parts.append(f"{self.n_maxima} Max")
        if self.n_saddles > 0:
            if self.total_saddle_index > self.n_saddles:
                parts.append(f"{self.n_saddles} Sad (mult {self.total_saddle_index})")
            else:
                parts.append(f"{self.n_saddles} Sad")
        if self.n_minima > 0:
            parts.append(f"{self.n_minima} Min")
        return ", ".join(parts) if parts else "0 CPs"


def macro_string_to_cluster(ms: MacroString) -> MicroCluster:
    """Adapt a MacroString to the MicroCluster interface.

    Downstream subroutines (notably ``classify_cluster_polarity``) take a
    MicroCluster. This adapter exposes a bridged corridor through that same
    interface, so a string is polarity-classified by exactly the same rule as a
    lone cluster instead of needing a parallel code path.

    Args:
        ms: The bridged macro-string.

    Returns:
        A MicroCluster carrying the string's members, index and elliptical ring.
    """
    extent = _unit_spatial_diameter(ms.all_members, ms.centroid)
    return MicroCluster(
        cluster_id=ms.string_id,
        members=list(ms.all_members),
        centroid=ms.centroid,
        centroid_unit=ms.centroid_unit,
        angular_radius_rad=ms.semi_major_rad,
        angular_radius_deg=ms.angular_length_deg / 2.0,
        angular_diameter_rad=ms.angular_length_rad,
        angular_diameter_deg=ms.angular_length_deg,
        spatial_diameter=extent,
        n_maxima=ms.n_maxima,
        n_minima=ms.n_minima,
        n_saddles=ms.n_saddles,
        total_saddle_index=ms.total_saddle_index,
        local_euler_index=ms.local_euler_index,
        is_multi_cp=len(ms.all_members) > 1,
        boundary_ring=list(ms.boundary_ring),
        morphology="string",
        aspect_ratio=min(ms.aspect_ratio, 999.0),
        principal_axis_unit=ms.principal_axis_unit,
        semi_major_rad=ms.semi_major_rad,
        semi_minor_rad=ms.semi_minor_rad,
    )


def _unit_spatial_diameter(
    members: list[CriticalPoint], centroid: tuple[float, float, float]
) -> float:
    """Return twice the largest member-to-centroid distance.

    Args:
        members: Critical points in the unit.
        centroid: Unit centroid.

    Returns:
        Twice the maximum member-to-centroid distance, in Angstrom.
    """
    if not members:
        return 0.0
    cx, cy, cz = centroid
    far = max(
        math.sqrt(
            (m.position[0] - cx) ** 2
            + (m.position[1] - cy) ** 2
            + (m.position[2] - cz) ** 2
        )
        for m in members
    )
    return 2.0 * far


def _unit_from_cluster(cl: MicroCluster) -> ReductionUnit:
    """Wrap a lone micro-cluster as a reduction unit.

    Args:
        cl: The micro-cluster.

    Returns:
        A single-cluster ReductionUnit.
    """
    return ReductionUnit(
        unit_id=cl.cluster_id,
        origin_ids=(cl.cluster_id,),
        members=list(cl.members),
        centroid=cl.centroid,
        centroid_unit=cl.centroid_unit,
        local_euler_index=cl.local_euler_index,
        n_maxima=cl.n_maxima,
        n_minima=cl.n_minima,
        n_saddles=cl.n_saddles,
        total_saddle_index=cl.total_saddle_index,
        boundary_ring=list(cl.boundary_ring),
        is_string=False,
        morphology=cl.morphology,
        aspect_ratio=cl.aspect_ratio,
        spatial_diameter=cl.spatial_diameter,
        as_cluster=cl,
    )


def _unit_from_macro_string(ms: MacroString) -> ReductionUnit:
    """Wrap a bridged macro-string as a reduction unit.

    A one-cluster macro-string is the clustering stage's own string, not a
    bridge, so it keeps its ``MC`` identity. Renaming it ``MS3`` would imply a
    merge that never happened.

    Args:
        ms: The bridged corridor.

    Returns:
        A ReductionUnit carrying the summed index of its constituents.
    """
    bridged = len(ms.constituent_clusters) > 1
    unit_id = ms.string_id if bridged else ms.constituent_clusters[0].cluster_id
    return ReductionUnit(
        unit_id=unit_id,
        origin_ids=tuple(cl.cluster_id for cl in ms.constituent_clusters),
        members=list(ms.all_members),
        centroid=ms.centroid,
        centroid_unit=ms.centroid_unit,
        local_euler_index=ms.local_euler_index,
        n_maxima=ms.n_maxima,
        n_minima=ms.n_minima,
        n_saddles=ms.n_saddles,
        total_saddle_index=ms.total_saddle_index,
        boundary_ring=list(ms.boundary_ring),
        is_string=bridged,
        morphology="string",
        aspect_ratio=min(ms.aspect_ratio, 999.0),
        spatial_diameter=_unit_spatial_diameter(ms.all_members, ms.centroid),
        as_cluster=macro_string_to_cluster(ms),
    )


def build_reduction_units(
    clusters: list[MicroCluster],
    delta_theta_mesh_rad: float,
    sphere_radius: float,
    aggregate_strings: bool,
    k_bridge: float = 3.0,
    aspect_ratio_threshold: float = 2.0,
) -> tuple[list[ReductionUnit], int]:
    """Partition micro-clusters into the sets reduced to one effective CP each.

    With ``aggregate_strings`` off this is the identity partition: one unit per
    micro-cluster, reproducing the pre-aggregation pipeline exactly. With it on,
    collinear neighbours along one ridge/valley corridor are bridged first, so a
    corridor fragmented by the isotropic clustering radius -- for example a
    saddle/dipole/saddle chain whose fragments individually carry chi = 0 and
    chi = +1 -- is evaluated on its summed index and collapses to a single
    effective saddle at the corridor centre.

    Bridging is disjoint: every cluster lands in exactly one unit, so the sum of
    unit indices always equals the sum of cluster indices and the displayed
    Euler characteristic is invariant under aggregation.

    Args:
        clusters: Micro-clusters from the clustering stage.
        delta_theta_mesh_rad: Mesh pitch in radians.
        sphere_radius: Sphere radius R.
        aggregate_strings: Whether to bridge collinear corridors.
        k_bridge: Maximum centroid separation, in delta_theta units, for
            bridging two clusters into one corridor.
        aspect_ratio_threshold: Minimum tangent-plane aspect ratio for a bridged
            group to qualify as a corridor.

    Returns:
        Tuple of (reduction units, number of clusters folded into a corridor).
    """
    if not aggregate_strings:
        return [_unit_from_cluster(cl) for cl in clusters], 0

    macro_strings, isolated = aggregate_cluster_strings(
        clusters=clusters,
        delta_theta_mesh_rad=delta_theta_mesh_rad,
        sphere_radius=sphere_radius,
        k_bridge=k_bridge,
        aspect_ratio_threshold=aspect_ratio_threshold,
        bridge_all=True,
    )

    units: list[ReductionUnit] = [_unit_from_cluster(cl) for cl in isolated]
    n_bridged = 0
    for ms in macro_strings:
        units.append(_unit_from_macro_string(ms))
        if len(ms.constituent_clusters) > 1:
            n_bridged += len(ms.constituent_clusters)

    # Order like compute_micro_clusters does (largest first, then earliest
    # source cluster) so the sidebar table reads the same with aggregation on
    # or off. Sorting by unit_id would interleave MC10 before MC2.
    order = {cl.cluster_id: i for i, cl in enumerate(clusters)}
    units.sort(key=lambda u: (-len(u.members), min(order[o] for o in u.origin_ids)))
    return units, n_bridged


def raw_cp_to_ui(
    cp: CriticalPoint, center: tuple[float, float, float] | None = None
) -> dict[str, Any]:
    """Convert a raw Discrete Morse critical point into its sidebar/inspector dict.

    Args:
        cp: The raw CriticalPoint produced by the Morse detector, in the centred
            atom frame.
        center: World-frame nucleus. When given, the emitted position is mapped
            into world coordinates so the inspector, the VTK glyphs and the
            picker all agree.

    Returns:
        JSON-serializable dict consumed by the CP table and inspector card.
    """
    title, badge = _RAW_TYPE_META.get(cp.cp_type, (cp.cp_type, "grey"))
    return {
        "vertex_id": cp.vertex_id,
        "type": cp.cp_type,
        "type_title": title,
        "badge_color": badge,
        "value": cp.value,
        "position": list(to_global(cp.position, center) if center else cp.position),
        "multiplicity": cp.multiplicity,
        "id_label": f"{cp.cp_type[:3].upper()}_{cp.vertex_id}",
        "is_effective": False,
        "is_hidden": False,
    }


def effective_cp_to_ui(
    entity_id: str,
    eff_type: str,
    position: tuple[float, float, float],
    value: float,
    index: int,
    members: list[CriticalPoint],
    origin_id: str,
    origin_label: str,
    fold_order: int,
    nudge: Any,
    center: tuple[float, float, float] | None = None,
) -> dict[str, Any]:
    """Convert a reduced effective critical point into its sidebar/inspector dict.

    Args:
        entity_id: Identifier of the effective entity (e.g. ``E-MAX_MC12``).
        eff_type: One of ``"E-MAX"``, ``"E-MIN"``, ``"E-SAD"``.
        position: Displayed (3,) coordinates in the centred atom frame
            (nudged when the nudge toggle is on).
        value: Representative scalar value of the collapsed set.
        index: Net topological index carried by the collapsed set.
        members: Constituent raw critical points.
        origin_id: Cluster / catastrophe identifier this entity was reduced from.
        origin_label: Human-readable origin kind ("Micro-Cluster" or "Catastrophe").
        fold_order: Symmetry fold order k used by the harmonic nudge.
        nudge: Optional ``NudgeResult`` when the harmonic nudge was applied.
        center: World-frame nucleus. When given, every emitted coordinate
            (position, nudge pre/post, member positions) is mapped into world
            coordinates.

    Returns:
        JSON-serializable dict consumed by the CP table and inspector card.
    """
    title, badge = _EFF_TYPE_META.get(eff_type, (eff_type, "grey"))
    nudge_applied = nudge is not None
    nudge_dict: dict[str, Any] | None = None
    if nudge is not None:
        nudge_dict = {
            "initial_position": list(
                to_global(nudge.initial_position, center)
                if center
                else nudge.initial_position
            ),
            "optimized_position": list(
                to_global(nudge.optimized_position, center)
                if center
                else nudge.optimized_position
            ),
            "displacement_ang_deg": nudge.displacement_ang_deg,
            "initial_energy": nudge.initial_energy,
            "final_energy": nudge.final_energy,
            "energy_drop": nudge.initial_energy - nudge.final_energy,
            "iterations": nudge.iterations,
            "converged": nudge.converged,
        }
    return {
        "vertex_id": -1,
        "type": eff_type,
        "type_title": title,
        "badge_color": badge,
        "value": value,
        "position": list(to_global(position, center) if center else position),
        "multiplicity": 1,
        "id_label": entity_id,
        "is_effective": True,
        "is_hidden": False,
        "net_index": index,
        "member_count": len(members),
        "origin_id": origin_id,
        "origin_label": origin_label,
        "fold_order": fold_order,
        "nudge_applied": nudge_applied,
        "nudge": nudge_dict,
        "members": [
            {
                "vertex_id": m.vertex_id,
                "type": m.cp_type,
                "value": m.value,
                "multiplicity": m.multiplicity,
                "index_contrib": -m.multiplicity if m.cp_type == "saddle" else 1,
                "position": list(
                    to_global(m.position, center) if center else m.position
                ),
                "id_label": f"{m.cp_type[:3].upper()}_{m.vertex_id}",
            }
            for m in members
        ],
    }


@dataclass
class ViewerSession:
    """The mesh-bound state of the viewer for one atom of one file.

    Everything in here changes when a different .plt or a different atom is
    loaded, so it is kept together in one object rather than spread over
    dozen closure variables. That is what makes a runtime reload possible:
    the session is replaced wholesale, and the pipeline reads from it.

    Attributes:
        plt_path: Absolute path of the loaded .plt.
        atom_number: The isolated atom (1-based).
        sphere_poly: The atom's sphere mesh, in the centred atom frame.
        field_items: The condensed fields discovered on that mesh.
        atom_info: Atom metadata (name, symbol, radius, nucleus, ...).
        center: World-frame nucleus of the atom.
        pts_np: (N, 3) centre-relative mesh vertices.
        triangles_np: (M, 3) triangle connectivity.
        render_poly: The mesh translated into the world frame, for drawing.
        geom_info: Mesh geometry metrics (radius, pitch, counts).
        sphere_radius: Sphere radius R, in Angstrom.
        delta_theta_rad: Intrinsic angular mesh pitch, in radians.
    """

    plt_path: str
    atom_number: int
    sphere_poly: vtkPolyData
    field_items: list[dict[str, str]]
    atom_info: dict[str, Any]
    center: tuple[float, float, float]
    pts_np: npt.NDArray[np.float64]
    triangles_np: npt.NDArray[np.int32]
    render_poly: vtkPolyData
    geom_info: dict[str, Any]
    sphere_radius: float
    delta_theta_rad: float


def build_viewer_session(plt_path: str, atom_number: int) -> ViewerSession:
    """Extract a mesh and assemble the viewer state bound to it.

    The extracted mesh is centre-relative: ``pts_np`` and every coordinate
    the algorithms produce live in the atom's own frame, where the outward
    normal of a surface point is ``p / ||p||``. VTK draws in the world
    frame, so the surface mesh gets a translated copy and every glyph, ring
    and port is mapped back with ``to_global(center)``.

    Args:
        plt_path: Path to the .plt file.
        atom_number: Atom to isolate (1-based).

    Returns:
        The assembled :class:`ViewerSession`.

    Raises:
        FileNotFoundError: If the .plt file does not exist.
        RuntimeError: If the requested atom has no sphere mesh.
    """
    abs_plt = os.path.abspath(plt_path)
    sphere_poly, field_items, atom_info = extract_atom_sphere_mesh(
        abs_plt, target_atom_num=atom_number
    )

    center: tuple[float, float, float] = tuple(atom_info["center"])  # type: ignore[assignment]

    pts_np: npt.NDArray[np.float64] = numpy_support.vtk_to_numpy(
        sphere_poly.GetPoints().GetData()
    )
    n_cells = sphere_poly.GetNumberOfPolys()
    cell_arr = numpy_support.vtk_to_numpy(sphere_poly.GetPolys().GetData())
    triangles_np: npt.NDArray[np.int32] = cell_arr.reshape((n_cells, 4))[:, 1:4]

    render_poly = vtkPolyData()
    render_pts = vtkPoints()
    for p in to_global_array(pts_np, center):
        render_pts.InsertNextPoint(p[0], p[1], p[2])
    render_poly.SetPoints(render_pts)
    render_poly.SetPolys(sphere_poly.GetPolys())
    render_poly.GetPointData().ShallowCopy(sphere_poly.GetPointData())

    geom_info = analyze_mesh_geometry(
        pts_np, triangles_np, sphere_radius=float(atom_info["sphere_radius"])
    )

    return ViewerSession(
        plt_path=abs_plt,
        atom_number=atom_number,
        sphere_poly=sphere_poly,
        field_items=field_items,
        atom_info=atom_info,
        center=center,
        pts_np=pts_np,
        triangles_np=triangles_np,
        render_poly=render_poly,
        geom_info=geom_info,
        sphere_radius=float(geom_info["sphere_radius"]),
        delta_theta_rad=float(geom_info["mesh_pitch_rad"]),
    )


def _uploaded_file_content(uploaded: Any) -> bytes | None:
    """Extract the bytes of a file uploaded from the browser.

    Trame's file widgets deliver the selected file as a dict carrying the
    file name, size, mime type and the content, which arrives either as a
    single ``bytes`` blob or as a list of chunks that have to be joined.

    Args:
        uploaded: The state value bound to the file input, or None.

    Returns:
        The file content as bytes, or None when nothing was uploaded.
    """
    if not isinstance(uploaded, dict):
        return None
    content = uploaded.get("content")
    if isinstance(content, list):
        chunks: list[bytes] = []
        for chunk in content:
            if isinstance(chunk, bytes):
                chunks.append(chunk)
        return b"".join(chunks) if chunks else None
    if isinstance(content, bytes):
        return content
    return None


def _geom_info_for_state(session: ViewerSession) -> dict[str, Any]:
    """Build the viewer's ``geom_info`` state entry from a session.

    Args:
        session: The session to read the geometry from.

    Returns:
        The state dict describing the mesh.
    """
    geom_info = session.geom_info
    atom_info = session.atom_info
    return {
        "num_vertices": geom_info["num_vertices"],
        "num_triangles": geom_info["num_triangles"],
        "num_edges": geom_info["num_edges"],
        "sphere_radius": float(geom_info["sphere_radius"]),
        "avg_edge_length": float(geom_info["avg_edge_length"]),
        "mesh_pitch_deg": float(geom_info["mesh_pitch_deg"]),
        "center": list(session.center),
        "center_offset": float(atom_info["center_offset"]),
        "radius_source": str(atom_info["radius_source"]),
    }


def run_sphere_viewer(
    plt_file: str,
    atom_num: int = 1,
    port: int | None = None,
    open_browser: bool = True,
    params_path: str | None = None,
) -> None:
    """Launch Step 2 sphere viewer application for a given atom and PLT file.

    Args:
        plt_file: Path to the .plt file to open.
        atom_num: Atom to isolate (1-based).
        port: Trame web server port.
        open_browser: Whether to open a browser on startup.
        params_path: Optional path to a parameter snapshot to restore on
            startup. The snapshot may name a different file, atom or field;
            the viewer opens whatever it asks for.
    """
    session = build_viewer_session(plt_file, atom_num)
    field_items = session.field_items
    atom_info = session.atom_info
    render_poly = session.render_poly

    # Renderer and Window
    renderer = vtkRenderer()
    renderer.SetBackground(0.12, 0.13, 0.16)
    renderer.SetBackground2(0.20, 0.22, 0.26)
    renderer.SetGradientBackground(True)

    render_window = vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window.SetSize(1000, 750)
    render_window.SetWindowName(
        f"gba_topology2 - Sphere Viewer: {atom_info['atom_name']}"
    )
    render_window.SetOffScreenRendering(1)

    interactor = vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)
    interactor.SetInteractorStyle(vtkInteractorStyleTrackballCamera())
    interactor.Initialize()

    # Orientation axes
    axes_actor = vtkAxesActor()
    axes_actor.SetShaftTypeToCylinder()
    axes_actor.SetCylinderRadius(0.03)
    axes_actor.SetTotalLength(1.0, 1.0, 1.0)
    axes_actor.SetNormalizedShaftLength(0.75, 0.75, 0.75)
    axes_actor.SetNormalizedTipLength(0.25, 0.25, 0.25)
    for cap_actor in (
        axes_actor.GetXAxisCaptionActor2D(),
        axes_actor.GetYAxisCaptionActor2D(),
        axes_actor.GetZAxisCaptionActor2D(),
    ):
        cap_actor.GetTextActor().SetTextScaleModeToNone()
        cap_prop = cap_actor.GetCaptionTextProperty()
        cap_prop.SetFontSize(16)
        cap_prop.BoldOn()
        cap_prop.ShadowOn()

    orientation_widget = vtkOrientationMarkerWidget()
    orientation_widget.SetOrientationMarker(axes_actor)
    orientation_widget.SetInteractor(interactor)
    orientation_widget.SetViewport(0.72, 0.72, 0.98, 0.98)
    orientation_widget.SetEnabled(1)
    orientation_widget.InteractiveOff()

    # 1. Sphere Surface Flood Actor
    flood_mapper = vtkPolyDataMapper()
    flood_mapper.SetInputData(render_poly)
    flood_mapper.SetScalarModeToUsePointFieldData()
    flood_mapper.ScalarVisibilityOn()

    flood_actor = vtkActor()
    flood_actor.SetMapper(flood_mapper)
    flood_actor.GetProperty().SetAmbient(0.35)
    flood_actor.GetProperty().SetDiffuse(0.75)
    flood_actor.GetProperty().SetSpecular(0.40)
    flood_actor.GetProperty().SetSpecularPower(30)
    renderer.AddActor(flood_actor)

    # 2. Sphere Isocontours Actor
    contour_filter = vtkContourFilter()
    contour_filter.SetInputData(render_poly)

    contour_mapper = vtkPolyDataMapper()
    contour_mapper.SetInputConnection(contour_filter.GetOutputPort())
    contour_mapper.ScalarVisibilityOff()

    contour_actor = vtkActor()
    contour_actor.SetMapper(contour_mapper)
    contour_actor.GetProperty().SetColor(1.0, 1.0, 1.0)
    contour_actor.GetProperty().SetLineWidth(1.8)
    contour_actor.GetProperty().SetLighting(False)
    renderer.AddActor(contour_actor)

    # 3. Sphere Wireframe Actor
    sphere_wire_mapper = vtkPolyDataMapper()
    sphere_wire_mapper.SetInputData(render_poly)
    sphere_wire_mapper.ScalarVisibilityOff()

    sphere_wire_actor = vtkActor()
    sphere_wire_actor.SetMapper(sphere_wire_mapper)
    sphere_wire_actor.GetProperty().SetRepresentationToWireframe()
    sphere_wire_actor.GetProperty().SetColor(0.85, 0.85, 0.85)
    sphere_wire_actor.GetProperty().SetOpacity(0.20)
    sphere_wire_actor.GetProperty().SetLineWidth(1.0)
    sphere_wire_actor.SetVisibility(False)
    renderer.AddActor(sphere_wire_actor)

    # 4. Critical Point Glyphs
    # Maxima (Red Spheres)
    maxima_pts = vtkPoints()
    maxima_poly = vtkPolyData()
    maxima_poly.SetPoints(maxima_pts)

    max_sphere_src = vtkSphereSource()
    max_sphere_src.SetRadius(0.045)
    max_sphere_src.SetThetaResolution(16)
    max_sphere_src.SetPhiResolution(16)

    max_glyph = vtkGlyph3D()
    max_glyph.SetSourceConnection(max_sphere_src.GetOutputPort())
    max_glyph.SetInputData(maxima_poly)
    max_glyph.ScalingOff()

    max_mapper = vtkPolyDataMapper()
    max_mapper.SetInputConnection(max_glyph.GetOutputPort())
    max_mapper.ScalarVisibilityOff()

    max_actor = vtkActor()
    max_actor.SetMapper(max_mapper)
    max_actor.GetProperty().SetColor(0.95, 0.15, 0.15)
    max_actor.GetProperty().SetAmbient(0.6)
    max_actor.GetProperty().SetDiffuse(0.6)
    renderer.AddActor(max_actor)

    # Minima (Blue Spheres)
    minima_pts = vtkPoints()
    minima_poly = vtkPolyData()
    minima_poly.SetPoints(minima_pts)

    min_sphere_src = vtkSphereSource()
    min_sphere_src.SetRadius(0.045)
    min_sphere_src.SetThetaResolution(16)
    min_sphere_src.SetPhiResolution(16)

    min_glyph = vtkGlyph3D()
    min_glyph.SetSourceConnection(min_sphere_src.GetOutputPort())
    min_glyph.SetInputData(minima_poly)
    min_glyph.ScalingOff()

    min_mapper = vtkPolyDataMapper()
    min_mapper.SetInputConnection(min_glyph.GetOutputPort())
    min_mapper.ScalarVisibilityOff()

    min_actor = vtkActor()
    min_actor.SetMapper(min_mapper)
    min_actor.GetProperty().SetColor(0.15, 0.55, 0.95)
    min_actor.GetProperty().SetAmbient(0.6)
    min_actor.GetProperty().SetDiffuse(0.6)
    renderer.AddActor(min_actor)

    # Saddles (Green Cubes)
    saddles_pts = vtkPoints()
    saddles_poly = vtkPolyData()
    saddles_poly.SetPoints(saddles_pts)

    sad_cube_src = vtkCubeSource()
    sad_cube_src.SetXLength(0.065)
    sad_cube_src.SetYLength(0.065)
    sad_cube_src.SetZLength(0.065)

    sad_glyph = vtkGlyph3D()
    sad_glyph.SetSourceConnection(sad_cube_src.GetOutputPort())
    sad_glyph.SetInputData(saddles_poly)
    sad_glyph.ScalingOff()

    sad_mapper = vtkPolyDataMapper()
    sad_mapper.SetInputConnection(sad_glyph.GetOutputPort())
    sad_mapper.ScalarVisibilityOff()

    sad_actor = vtkActor()
    sad_actor.SetMapper(sad_mapper)
    sad_actor.GetProperty().SetColor(0.20, 0.85, 0.35)
    sad_actor.GetProperty().SetAmbient(0.6)
    sad_actor.GetProperty().SetDiffuse(0.6)
    renderer.AddActor(sad_actor)

    # 5. Micro-Cluster Boundary Rings (Tubes on sphere surface)
    cluster_rings_poly = vtkPolyData()
    cluster_rings_tubes = vtkTubeFilter()
    cluster_rings_tubes.SetInputData(cluster_rings_poly)
    cluster_rings_tubes.SetRadius(0.010)
    cluster_rings_tubes.SetNumberOfSides(12)
    cluster_rings_tubes.CappingOn()

    cluster_rings_mapper = vtkPolyDataMapper()
    cluster_rings_mapper.SetInputConnection(cluster_rings_tubes.GetOutputPort())
    cluster_rings_mapper.ScalarVisibilityOff()

    cluster_rings_actor = vtkActor()
    cluster_rings_actor.SetMapper(cluster_rings_mapper)
    cluster_rings_actor.GetProperty().SetColor(0.05, 0.05, 0.05)  # Crisp black rings
    cluster_rings_actor.GetProperty().SetAmbient(0.9)
    cluster_rings_actor.GetProperty().SetDiffuse(0.1)
    renderer.AddActor(cluster_rings_actor)

    # 5a. Annulled Cluster Rings (ghost-grey tubes): boundary of a chi = 0 set that
    # reduced to NOTHING. The glyph is gone; the ring is the record of the annulment.
    annulled_rings_poly = vtkPolyData()
    annulled_rings_tubes = vtkTubeFilter()
    annulled_rings_tubes.SetInputData(annulled_rings_poly)
    annulled_rings_tubes.SetRadius(0.014)
    annulled_rings_tubes.SetNumberOfSides(12)
    annulled_rings_tubes.CappingOn()

    annulled_rings_mapper = vtkPolyDataMapper()
    annulled_rings_mapper.SetInputConnection(annulled_rings_tubes.GetOutputPort())
    annulled_rings_mapper.ScalarVisibilityOff()

    annulled_rings_actor = vtkActor()
    annulled_rings_actor.SetMapper(annulled_rings_mapper)
    annulled_rings_actor.GetProperty().SetColor(0.78, 0.78, 0.82)  # Ghost grey
    annulled_rings_actor.GetProperty().SetAmbient(0.9)
    annulled_rings_actor.GetProperty().SetDiffuse(0.1)
    annulled_rings_actor.GetProperty().SetOpacity(0.85)
    annulled_rings_actor.SetVisibility(False)
    renderer.AddActor(annulled_rings_actor)

    # 5b. Catastrophe Rings (Purple tubes on sphere surface)
    cat_rings_poly = vtkPolyData()
    cat_rings_tubes = vtkTubeFilter()
    cat_rings_tubes.SetInputData(cat_rings_poly)
    cat_rings_tubes.SetRadius(0.012)
    cat_rings_tubes.SetNumberOfSides(12)
    cat_rings_tubes.CappingOn()

    cat_rings_mapper = vtkPolyDataMapper()
    cat_rings_mapper.SetInputConnection(cat_rings_tubes.GetOutputPort())
    cat_rings_mapper.ScalarVisibilityOff()

    cat_rings_actor = vtkActor()
    cat_rings_actor.SetMapper(cat_rings_mapper)
    cat_rings_actor.GetProperty().SetColor(0.60, 0.20, 0.85)  # Purple
    cat_rings_actor.GetProperty().SetAmbient(0.8)
    cat_rings_actor.GetProperty().SetDiffuse(0.2)
    renderer.AddActor(cat_rings_actor)

    # 5c. Boundary Ports: Valley Ports (Orange spheres) and Ridge Ports (Cyan spheres)
    vports_pts = vtkPoints()
    vports_poly = vtkPolyData()
    vports_poly.SetPoints(vports_pts)

    vport_src = vtkSphereSource()
    vport_src.SetRadius(0.035)
    vport_src.SetThetaResolution(14)
    vport_src.SetPhiResolution(14)

    vport_glyph = vtkGlyph3D()
    vport_glyph.SetSourceConnection(vport_src.GetOutputPort())
    vport_glyph.SetInputData(vports_poly)
    vport_glyph.SetScaleModeToDataScalingOff()

    vport_mapper = vtkPolyDataMapper()
    vport_mapper.SetInputConnection(vport_glyph.GetOutputPort())
    vport_mapper.ScalarVisibilityOff()

    vport_actor = vtkActor()
    vport_actor.SetMapper(vport_mapper)
    vport_actor.GetProperty().SetColor(1.0, 0.55, 0.0)  # Vivid Orange
    vport_actor.GetProperty().SetAmbient(0.7)
    vport_actor.GetProperty().SetDiffuse(0.5)
    renderer.AddActor(vport_actor)

    rports_pts = vtkPoints()
    rports_poly = vtkPolyData()
    rports_poly.SetPoints(rports_pts)

    rport_src = vtkSphereSource()
    rport_src.SetRadius(0.035)
    rport_src.SetThetaResolution(14)
    rport_src.SetPhiResolution(14)

    rport_glyph = vtkGlyph3D()
    rport_glyph.SetSourceConnection(rport_src.GetOutputPort())
    rport_glyph.SetInputData(rports_poly)
    rport_glyph.SetScaleModeToDataScalingOff()

    rport_mapper = vtkPolyDataMapper()
    rport_mapper.SetInputConnection(rport_glyph.GetOutputPort())
    rport_mapper.ScalarVisibilityOff()

    rport_actor = vtkActor()
    rport_actor.SetMapper(rport_mapper)
    rport_actor.GetProperty().SetColor(0.0, 0.85, 0.95)  # Cyan
    rport_actor.GetProperty().SetAmbient(0.7)
    rport_actor.GetProperty().SetDiffuse(0.5)
    renderer.AddActor(rport_actor)

    # 6. Selection Highlight Actor
    highlight_src = vtkSphereSource()
    highlight_src.SetRadius(0.08)
    highlight_src.SetThetaResolution(18)
    highlight_src.SetPhiResolution(18)

    highlight_mapper = vtkPolyDataMapper()
    highlight_mapper.SetInputConnection(highlight_src.GetOutputPort())

    highlight_actor = vtkActor()
    highlight_actor.SetMapper(highlight_mapper)
    highlight_actor.GetProperty().SetColor(1.0, 0.95, 0.1)  # Vivid yellow
    highlight_actor.GetProperty().SetRepresentationToWireframe()
    highlight_actor.GetProperty().SetLineWidth(3.0)
    highlight_actor.SetVisibility(False)
    renderer.AddActor(highlight_actor)

    # 7. Cluster Member Highlight Rings (Cyan wireframe spheres)
    cluster_members_pts = vtkPoints()
    cluster_members_poly = vtkPolyData()
    cluster_members_poly.SetPoints(cluster_members_pts)

    cluster_member_src = vtkSphereSource()
    cluster_member_src.SetRadius(0.065)
    cluster_member_src.SetThetaResolution(16)
    cluster_member_src.SetPhiResolution(16)

    cluster_members_glyph = vtkGlyph3D()
    cluster_members_glyph.SetSourceConnection(cluster_member_src.GetOutputPort())
    cluster_members_glyph.SetInputData(cluster_members_poly)
    cluster_members_glyph.ScalingOff()

    cluster_members_mapper = vtkPolyDataMapper()
    cluster_members_mapper.SetInputConnection(cluster_members_glyph.GetOutputPort())
    cluster_members_mapper.ScalarVisibilityOff()

    cluster_members_actor = vtkActor()
    cluster_members_actor.SetMapper(cluster_members_mapper)
    cluster_members_actor.GetProperty().SetColor(0.2, 0.9, 1.0)  # Cyan
    cluster_members_actor.GetProperty().SetRepresentationToWireframe()
    cluster_members_actor.GetProperty().SetLineWidth(2.5)
    cluster_members_actor.SetVisibility(False)
    renderer.AddActor(cluster_members_actor)

    # 8. Active Basin Center Halo Actor (Gold wireframe sphere)
    basin_center_src = vtkSphereSource()
    basin_center_src.SetRadius(0.09)
    basin_center_src.SetThetaResolution(20)
    basin_center_src.SetPhiResolution(20)

    basin_center_mapper = vtkPolyDataMapper()
    basin_center_mapper.SetInputConnection(basin_center_src.GetOutputPort())

    basin_center_actor = vtkActor()
    basin_center_actor.SetMapper(basin_center_mapper)
    basin_center_actor.GetProperty().SetColor(1.0, 0.84, 0.0)  # Pure Gold
    basin_center_actor.GetProperty().SetRepresentationToWireframe()
    basin_center_actor.GetProperty().SetLineWidth(3.5)
    basin_center_actor.SetVisibility(False)
    renderer.AddActor(basin_center_actor)

    # 9. Saddle Launch Seed Rays (Basin Locator Step 2)
    # Ascent = ridge / unstable manifold (towards maxima), Descent = valley /
    # stable manifold (towards minima). Rendered as short tubes so they read as
    # directional stubs on the surface rather than flat lines.
    launch_ascent_poly = vtkPolyData()
    launch_ascent_tubes = vtkTubeFilter()
    launch_ascent_tubes.SetInputData(launch_ascent_poly)
    launch_ascent_tubes.SetRadius(0.008)
    launch_ascent_tubes.SetNumberOfSides(8)
    launch_ascent_tubes.CappingOn()

    launch_ascent_mapper = vtkPolyDataMapper()
    launch_ascent_mapper.SetInputConnection(launch_ascent_tubes.GetOutputPort())
    launch_ascent_mapper.ScalarVisibilityOff()

    launch_ascent_actor = vtkActor()
    launch_ascent_actor.SetMapper(launch_ascent_mapper)
    launch_ascent_actor.GetProperty().SetColor(1.0, 0.35, 0.10)  # Vivid orange-red
    launch_ascent_actor.GetProperty().SetAmbient(0.8)
    launch_ascent_actor.GetProperty().SetDiffuse(0.2)
    launch_ascent_actor.SetVisibility(False)
    renderer.AddActor(launch_ascent_actor)

    launch_descent_poly = vtkPolyData()
    launch_descent_tubes = vtkTubeFilter()
    launch_descent_tubes.SetInputData(launch_descent_poly)
    launch_descent_tubes.SetRadius(0.008)
    launch_descent_tubes.SetNumberOfSides(8)
    launch_descent_tubes.CappingOn()

    launch_descent_mapper = vtkPolyDataMapper()
    launch_descent_mapper.SetInputConnection(launch_descent_tubes.GetOutputPort())
    launch_descent_mapper.ScalarVisibilityOff()

    launch_descent_actor = vtkActor()
    launch_descent_actor.SetMapper(launch_descent_mapper)
    launch_descent_actor.GetProperty().SetColor(0.10, 0.75, 1.0)  # Vivid cyan
    launch_descent_actor.GetProperty().SetAmbient(0.8)
    launch_descent_actor.GetProperty().SetDiffuse(0.2)
    launch_descent_actor.SetVisibility(False)
    renderer.AddActor(launch_descent_actor)

    renderer.ResetCamera()

    # Trame Server Setup
    server = get_server("sphere_viewer_step2")
    state, ctrl = server.state, server.controller

    def request_view_update() -> None:
        if hasattr(ctrl, "view_update") and callable(ctrl.view_update):
            try:
                ctrl.view_update()
            except (RuntimeError, Exception):
                pass

    # Initial state. The Vuetify theme is not set here: ``trame__theme`` is read
    # by nothing in trame, so the palette is supplied to ``createVuetify()``
    # through ``vuetify_config=`` on the layout below.
    default_field = field_items[0]["value"] if field_items else "Electron Density"
    state.plt_file = os.path.basename(session.plt_path)
    state.atom_info = session.atom_info
    state.geom_info = _geom_info_for_state(session)
    state.field_items = session.field_items
    state.selected_field = default_field
    # Seed every interactive control from the single canonical defaults table so
    # the initial state and the "Reset to Defaults" action cannot drift apart.
    state.update(DEFAULT_UI_STATE)
    state.reduction_stats = {
        "n_catastrophes_reduced": 0,
        "n_clusters_reduced": 0,
        "n_clusters_annihilated": 0,
        "n_clusters_irreducible": 0,
        "n_clusters_bridged": 0,
        "n_effective_cps": 0,
        "n_raw_cps_hidden": 0,
        "n_nudged": 0,
        "max_nudge_deg": 0.0,
        "mean_nudge_deg": 0.0,
        "n_ports_perfected": 0,
    }

    # Computed lists & selection
    state.morse_counts = {
        "minima": 0,
        "maxima": 0,
        "saddles": 0,
        "euler": 2,
        "valid": True,
    }
    # Counts of the critical points actually rendered (after reduction)
    state.display_counts = {
        "minima": 0,
        "maxima": 0,
        "saddles": 0,
        "euler": 2,
        "valid": True,
    }
    state.all_cps_list = []
    state.clusters_list = []
    state.catastrophes_list = []
    state.basin_centers_list = []
    state.launch_stats = {
        "n_saddles_seeded": 0,
        "n_ascent": 0,
        "n_descent": 0,
        "n_index_mismatch": 0,
        "max_fold_order": 0,
    }
    state.selected_cp = None
    state.selected_cluster = None
    state.selected_catastrophe = None
    state.selected_basin_center = None
    # Absolute path of the most recent "Save Parameters" JSON, for the
    # drawer confirmation chip. Export-only: deliberately absent from
    # DEFAULT_UI_STATE and the @state.change decorator (it triggers no
    # recompute and must survive "Reset to Defaults").
    state.params_last_saved_path = ""
    # Name of the most recently loaded snapshot, and the error from the last
    # load attempt. Both are display-only and never feed the pipeline.
    state.params_last_loaded_path = ""
    state.params_load_error = ""

    # Cached in memory for 3D picking ray-cast
    runtime_cps: list[CriticalPoint] = []
    runtime_clusters: list[MicroCluster] = []
    runtime_catastrophes: list[EffectiveCatastrophe] = []
    runtime_effective_cps: list[dict[str, Any]] = []
    runtime_hidden_vertex_ids: set[int] = set()
    runtime_annulled_cluster_ids: set[str] = set()
    runtime_irreducible_cluster_ids: set[str] = set()

    # Re-entrancy latch for bulk state writes (see reset_defaults). Trame fires a
    # @state.change callback per changed key, so a 19-key batched reset would
    # otherwise rebuild the whole pipeline 19 times.
    suppress_param_change = False

    def update_topology() -> None:
        nonlocal runtime_cps, runtime_clusters, runtime_catastrophes
        nonlocal runtime_effective_cps, runtime_hidden_vertex_ids
        cur_field = state.selected_field
        pd = session.sphere_poly.GetPointData()
        arr = pd.GetArray(cur_field)
        if not arr:
            return

        f_vals_np: npt.NDArray[np.float64] = numpy_support.vtk_to_numpy(arr)

        # 1. Run Morse Detector (Subroutine 2)
        morse_res = compute_discrete_morse_cps(
            session.pts_np,
            session.triangles_np,
            f_vals_np,
            tie_break_epsilon=1e-15,
        )

        state.morse_counts = {
            "minima": len(morse_res.minima),
            "maxima": len(morse_res.maxima),
            "saddles": len(morse_res.saddles),
            "euler": morse_res.euler_characteristic,
            "valid": morse_res.is_valid_euler,
        }

        # Update Glyph Points
        scale = float(state.cp_glyph_scale)
        max_sphere_src.SetRadius(0.045 * scale)
        max_sphere_src.Update()
        min_sphere_src.SetRadius(0.045 * scale)
        min_sphere_src.Update()
        sad_cube_src.SetXLength(0.065 * scale)
        sad_cube_src.SetYLength(0.065 * scale)
        sad_cube_src.SetZLength(0.065 * scale)
        sad_cube_src.Update()

        # NOTE: The CP glyph point sets (maxima_pts / minima_pts / saddles_pts) are
        # populated at the END of this routine, after the effective-CP reduction stage,
        # so that sets collapsed into effective critical points can be hidden while
        # their boundary rings stay on screen.

        # =====================================================================
        # [PIPELINE EXTENSION POINT - Step 3: Adjacent Extrema Fusion]
        # Currently, Simulation of Simplicity (SoS) tie-breaking (tie_break_epsilon=1e-15)
        # in morse_detector.py resolves flat plateau ties deterministically, guaranteeing
        # integer chi = 2 without leaving adjacent same-type extrema (d_G = 1) across the mesh.
        # Downstream, any sub-grid positioning is further optimized in Step 8 (Harmonic Nudge).
        #
        # If raw unperturbed data (tie_break_epsilon = 0.0) or explicit plateau clique
        # collapsing is required in the future, insert the adjacency scan/fusion stage
        # here BEFORE micro-clustering:
        #
        #     # from gba_topology2.src.extrema_fusion import fuse_adjacent_extrema
        #     # fused_minima = fuse_adjacent_extrema(morse_res.minima, pts, triangles, f_vals)
        #     # fused_maxima = fuse_adjacent_extrema(morse_res.maxima, pts, triangles, f_vals)
        #     # all_raw_cps = fused_minima + fused_maxima + morse_res.saddles
        # =====================================================================

        # 2. Run Micro-Cluster (Subroutine 3)
        all_raw_cps = morse_res.minima + morse_res.maxima + morse_res.saddles
        runtime_cps = all_raw_cps
        k_val = float(state.k_pitch)
        clusters = compute_micro_clusters(
            cps=all_raw_cps,
            delta_theta_mesh_rad=session.delta_theta_rad,
            k_pitch=k_val,
            sphere_radius=session.sphere_radius,
            min_boundary_margin_deg=1.5,
            num_boundary_points=48,
        )
        runtime_clusters = clusters

        # 2b. Bridge collinear corridor fragments into single reduction units.
        # The clustering radius is isotropic, so a ridge/valley corridor whose
        # node spacing varies is routinely cut into fragments that individually
        # carry chi = 0 (annulled to nothing) or chi = +1 (reduced to a spurious
        # extremum). Bridging first lets each corridor be judged on its summed
        # index, so it collapses to one effective saddle at its centre.
        do_aggregate = bool(state.aggregate_strings)
        units, n_bridged = build_reduction_units(
            clusters=clusters,
            delta_theta_mesh_rad=session.delta_theta_rad,
            sphere_radius=session.sphere_radius,
            aggregate_strings=do_aggregate,
            k_bridge=float(state.k_bridge),
        )

        # Build VTK geometry for cluster boundary rings
        ring_pts = vtkPoints()
        ring_lines = vtkCellArray()
        pt_counter = 0

        multi_units = [u for u in units if len(u.members) > 1]
        for unit in multi_units:
            ring = unit.boundary_ring
            n_r = len(ring)
            if n_r < 3:
                continue

            for rp in ring:
                # Lift the ring radially in the atom frame to prevent z-fighting
                # with the sphere, then place it in the world frame.
                lifted = radial_offset(rp.position, 1.008)
                g = to_global(lifted, session.center)
                ring_pts.InsertNextPoint(g[0], g[1], g[2])

            ring_lines.InsertNextCell(n_r + 1)
            for k in range(n_r):
                ring_lines.InsertCellPoint(pt_counter + k)
            ring_lines.InsertCellPoint(pt_counter)  # Close the ring
            pt_counter += n_r

        cluster_rings_poly.SetPoints(ring_pts)
        cluster_rings_poly.SetLines(ring_lines)
        cluster_rings_poly.Modified()
        cluster_rings_actor.SetVisibility(bool(state.show_rings and pt_counter > 0))

        # Format reduction units for the sidebar table. With aggregation off a
        # unit is exactly one micro-cluster, so the table is unchanged; with it
        # on, a bridged corridor appears as one MS-prefixed row.
        clusters_ui: list[dict[str, Any]] = []
        for unit in units:
            clusters_ui.append(
                {
                    "id": unit.unit_id,
                    "size": len(unit.members),
                    "composition_summary": unit.composition_summary,
                    "local_euler_index": unit.local_euler_index,
                    "n_maxima": unit.n_maxima,
                    "n_minima": unit.n_minima,
                    "n_saddles": unit.n_saddles,
                    "total_saddle_index": unit.total_saddle_index,
                    "is_multi_cp": len(unit.members) > 1,
                    "morphology": unit.morphology,
                    "aspect_ratio": unit.aspect_ratio,
                    "angular_radius_deg": unit.as_cluster.angular_radius_deg,
                    "angular_diameter_deg": unit.as_cluster.angular_diameter_deg,
                    "spatial_diameter": unit.spatial_diameter,
                    "centroid": list(to_global(unit.centroid, session.center)),
                    "centroid_unit": list(unit.centroid_unit),
                    "is_string": unit.is_string,
                    "origin_ids": list(unit.origin_ids),
                    "members": [
                        {
                            "vertex_id": m.vertex_id,
                            "type": m.cp_type,
                            "value": m.value,
                            "multiplicity": m.multiplicity,
                            "index_contrib": -m.multiplicity
                            if m.cp_type == "saddle"
                            else 1,
                            "position": list(to_global(m.position, session.center)),
                            "id_label": f"{m.cp_type[:3].upper()}_{m.vertex_id}",
                        }
                        for m in unit.members
                    ],
                    "is_annulled": False,
                    "is_irreducible": False,
                }
            )
        # NOTE: state.clusters_list is assigned after the Step 6 reduction stage,
        # once the annulled / irreducible flags are known.

        # 3. Run Catastrophe Classifier (Phase 2 & Phase 3)
        # Runs on the raw micro-clusters, not the bridged units, so catastrophe
        # detection is identical whether or not aggregation is enabled.
        b_tol = float(state.cat_barrier_tol)
        ang_mult = float(state.cat_angular_tol_mult)
        catastrophes, _ = classify_catastrophes(
            clusters=clusters,
            field_vals=f_vals_np,
            delta_theta_mesh_rad=session.delta_theta_rad,
            sphere_radius=session.sphere_radius,
            barrier_tol=b_tol,
            angular_tol_mult=ang_mult,
        )
        runtime_catastrophes = catastrophes

        # Build VTK geometry for catastrophe boundary rings and boundary ports
        c_ring_pts = vtkPoints()
        c_ring_lines = vtkCellArray()
        c_pt_counter = 0

        vports_pts.Reset()
        rports_pts.Reset()

        # 4. Effective Critical Point Reduction & Harmonic Nudge (Step 6 assessment)
        # Four independent switches let each algorithmic stage be judged in isolation:
        #   reduce_catastrophes  -> collapse catastrophe constellations to one E-SAD
        #   reduce_multi_clusters -> collapse multi-CP micro-clusters to E-MAX / E-MIN
        #   nudge_cp_positions   -> symmetry-energy pattern search on effective CPs
        #   perfect_port_angles  -> enforce ideal 360/k azimuthal port spacing
        reduce_cats = bool(state.reduce_catastrophes)
        reduce_clusts = bool(state.reduce_multi_clusters)
        do_nudge = bool(state.nudge_cp_positions)
        do_perfect = bool(state.perfect_port_angles)
        max_nudge_pitch = float(state.nudge_max_pitch)

        # Only *fused* (spurious) catastrophes may be collapsed. A split
        # constellation is genuine physical symmetry breaking whose poles are
        # distinct critical points; collapsing it would fabricate a saddle at a
        # centroid that is not a critical point and would hide real extrema
        # carried by its core cluster. See fusable_catastrophes.
        hidden_vertex_ids: set[int] = set()
        effective_cps_ui: list[dict[str, Any]] = []
        n_cats_reduced = 0
        n_clusts_reduced = 0
        n_clusts_annihilated = 0
        n_clusts_irreducible = 0
        n_ports_perfected = 0
        nudge_degs: list[float] = []

        catastrophes_ui: list[dict[str, Any]] = []
        for cat in catastrophes:
            # Excision boundary profiling for ports
            b_res = profile_boundary_loop(
                ring_points=cat.boundary_ring,
                mesh_pts=session.pts_np,
                f_vals=f_vals_np,
            )

            v_ports = b_res.valley_ports
            r_ports = b_res.ridge_ports
            if do_perfect:
                v_in = perfect_boundary_ports(v_ports, cat.fold_order)
                r_in = perfect_boundary_ports(r_ports, cat.fold_order)
                v_ports = reposition_ports_on_ring(
                    v_in,
                    cat.boundary_ring,
                    session.sphere_radius,
                    f_vals_np,
                    session.pts_np,
                )
                r_ports = reposition_ports_on_ring(
                    r_in,
                    cat.boundary_ring,
                    session.sphere_radius,
                    f_vals_np,
                    session.pts_np,
                )
                n_ports_perfected += len(v_ports) + len(r_ports)

            for v_port in v_ports:
                g = to_global(radial_offset(v_port.position, 1.008), session.center)
                vports_pts.InsertNextPoint(g[0], g[1], g[2])
            for r_port in r_ports:
                g = to_global(radial_offset(r_port.position, 1.008), session.center)
                rports_pts.InsertNextPoint(g[0], g[1], g[2])

            # Tube geometry for purple catastrophe ring
            c_ring = cat.boundary_ring
            n_cr = len(c_ring)
            if n_cr >= 3:
                for c_pt in c_ring:
                    g = to_global(radial_offset(c_pt.position, 1.012), session.center)
                    c_ring_pts.InsertNextPoint(g[0], g[1], g[2])
                c_ring_lines.InsertNextCell(n_cr + 1)
                for k in range(n_cr):
                    c_ring_lines.InsertCellPoint(c_pt_counter + k)
                c_ring_lines.InsertCellPoint(c_pt_counter)
                c_pt_counter += n_cr

            # Collapse the constellation into a single effective saddle, keeping the
            # purple excision ring on screen as the record of the collapsed set.
            # Only *fused* constellations qualify: a split constellation is
            # genuine physical symmetry breaking whose poles are distinct
            # critical points, so collapsing it would fabricate a saddle at a
            # centroid that is not a critical point and would hide the real
            # extrema its core cluster carries.
            fuse_this = reduce_cats and cat.is_fused
            cat_nudge = None
            if fuse_this and do_nudge:
                probe_pitch = max(
                    1.5,
                    cat.angular_span_deg
                    / 2.0
                    / max(math.degrees(session.delta_theta_rad), 1e-9),
                )
                cat_nudge = harmonic_nudge_critical_point(
                    initial_pos=cat.centroid,
                    mesh_pts=session.pts_np,
                    f_vals=f_vals_np,
                    fold_order=cat.fold_order,
                    delta_theta_mesh_rad=session.delta_theta_rad,
                    sphere_radius=session.sphere_radius,
                    max_displacement_pitch=max_nudge_pitch,
                    probe_radius_pitch=probe_pitch,
                )
            if fuse_this:
                if cat_nudge is not None:
                    eff_pos = cat_nudge.optimized_position
                    nudge_degs.append(cat_nudge.displacement_ang_deg)
                else:
                    eff_pos = cat.centroid
                eff_val = (
                    max(m.value for m in cat.all_members) if cat.all_members else 0.0
                )
                effective_cps_ui.append(
                    effective_cp_to_ui(
                        entity_id=f"E-SAD_{cat.entity_id}",
                        eff_type="E-SAD",
                        position=eff_pos,
                        value=eff_val,
                        index=cat.net_euler_index,
                        members=cat.all_members,
                        origin_id=cat.entity_id,
                        origin_label="Catastrophe",
                        fold_order=cat.fold_order,
                        nudge=cat_nudge if do_nudge else None,
                        center=session.center,
                    )
                )
                for m in cat.all_members:
                    hidden_vertex_ids.add(m.vertex_id)
                n_cats_reduced += 1

            catastrophes_ui.append(
                {
                    "id": cat.entity_id,
                    "type": cat.catastrophe_type,
                    "classification": cat.classification,
                    "is_fused": cat.is_fused,
                    "fold_order": cat.fold_order,
                    "net_euler_index": cat.net_euler_index,
                    "centroid": list(to_global(cat.centroid, session.center)),
                    "centroid_unit": list(cat.centroid_unit),
                    "angular_radius_deg": cat.angular_radius_deg,
                    "angular_span_deg": cat.angular_span_deg,
                    "barrier_depth": cat.barrier_depth,
                    "relative_barrier_depth": cat.relative_barrier_depth,
                    "bifurcation_score": cat.bifurcation_score,
                    "num_valleys": len(v_ports),
                    "num_ridges": len(r_ports),
                    "is_balanced": b_res.is_topologically_balanced,
                    "constituent_cluster_ids": [
                        cl.cluster_id for cl in cat.constituent_clusters
                    ],
                    "all_member_count": len(cat.all_members),
                    "is_reduced": fuse_this,
                    "ports_perfected": do_perfect,
                    "nudge_deg": cat_nudge.displacement_ang_deg
                    if (fuse_this and do_nudge and cat_nudge is not None)
                    else None,
                }
            )

        cat_rings_poly.SetPoints(c_ring_pts)
        cat_rings_poly.SetLines(c_ring_lines)
        cat_rings_poly.Modified()
        cat_rings_actor.SetVisibility(
            bool(state.show_catastrophes and c_pt_counter > 0)
        )

        vports_poly.SetPoints(vports_pts)
        vports_poly.Modified()
        vport_actor.SetVisibility(
            bool(state.show_ports and vports_pts.GetNumberOfPoints() > 0)
        )

        rports_poly.SetPoints(rports_pts)
        rports_poly.Modified()
        rport_actor.SetVisibility(
            bool(state.show_ports and rports_pts.GetNumberOfPoints() > 0)
        )

        state.catastrophes_list = catastrophes_ui

        # 5. Collapse each multi-CP reduction unit into one effective CP.
        # A unit is a lone micro-cluster, or a bridged corridor of collinear
        # fragments. Units absorbed by a catastrophe are handled above and
        # skipped here; a corridor is claimed when any of its fragments is, so
        # aggregation cannot smuggle a catastrophe's index into a second glyph.
        # The reduction kind is dictated by the set's net Euler index (topological
        # contract, see effective_reduction_decision):
        #   chi == 0  -> annihilate: members hidden, NO glyph, ring kept (annulled)
        #   chi == +1 -> E-MAX / E-MIN via the radial-slope polarity classifier
        #   chi <= -1 -> E-SAD carrying the net index (fold order k = 1 - chi)
        #   chi >= +2 -> irreducible: a single effective CP carries at most +1,
        #                so the set stays fully expanded (counted, not reduced)
        #
        # Ownership follows the toggle that performs the collapse: with
        # reduce_catastrophes off, nothing is claimed, so a chi = -1 string
        # absorbed into an unreduced catastrophe still reduces here (see
        # catastrophe_claimed_cluster_ids for the full rationale).
        cat_cluster_ids = catastrophe_claimed_cluster_ids(catastrophes, reduce_cats)

        annihilated_unit_ids: set[str] = set()
        irreducible_unit_ids: set[str] = set()

        if reduce_clusts:
            for unit in units:
                if len(unit.members) <= 1:
                    continue
                if any(oid in cat_cluster_ids for oid in unit.origin_ids):
                    continue
                eff = classify_cluster_polarity(
                    cluster=unit.as_cluster,
                    mesh_pts=session.pts_np,
                    f_vals=f_vals_np,
                    delta_theta_mesh_rad=session.delta_theta_rad,
                    sphere_radius=session.sphere_radius,
                )
                decision = effective_reduction_decision(
                    unit.local_euler_index, eff.extremum_type
                )

                if decision == "E-NONE":
                    # Topologically neutral set: reduce to nothing. Hide the
                    # member glyphs and keep the boundary ring (rendered in the
                    # annulment style) as the record that a set was collapsed.
                    for m in unit.members:
                        hidden_vertex_ids.add(m.vertex_id)
                    annihilated_unit_ids.add(unit.unit_id)
                    n_clusts_annihilated += 1
                    continue

                if decision == "IRREDUCIBLE":
                    # Net index >= +2 cannot be carried by one effective CP.
                    irreducible_unit_ids.add(unit.unit_id)
                    n_clusts_irreducible += 1
                    continue

                eff_type = decision
                eff_index = unit.local_euler_index
                eff_val = eff.effective_value
                fold_order = 0
                if eff_type == "E-SAD":
                    eff_val = (
                        max(m.value for m in unit.members) if unit.members else 0.0
                    )
                    fold_order = 1 - eff_index
                cl_nudge = None
                if do_nudge:
                    cl_nudge = harmonic_nudge_critical_point(
                        initial_pos=unit.centroid,
                        mesh_pts=session.pts_np,
                        f_vals=f_vals_np,
                        fold_order=fold_order,
                        delta_theta_mesh_rad=session.delta_theta_rad,
                        sphere_radius=session.sphere_radius,
                        max_displacement_pitch=max_nudge_pitch,
                        probe_radius_pitch=max(
                            1.5,
                            unit.as_cluster.angular_radius_deg
                            / max(math.degrees(session.delta_theta_rad), 1e-9),
                        ),
                    )
                if cl_nudge is not None:
                    eff_pos = cl_nudge.optimized_position
                    nudge_degs.append(cl_nudge.displacement_ang_deg)
                else:
                    eff_pos = unit.centroid
                effective_cps_ui.append(
                    effective_cp_to_ui(
                        entity_id=f"{eff_type}_{unit.unit_id}",
                        eff_type=eff_type,
                        position=eff_pos,
                        value=eff_val,
                        index=eff_index,
                        members=unit.members,
                        origin_id=unit.unit_id,
                        origin_label=(
                            "Saddle Corridor" if unit.is_string else "Micro-Cluster"
                        ),
                        fold_order=fold_order,
                        nudge=cl_nudge if do_nudge else None,
                        center=session.center,
                    )
                )
                for m in unit.members:
                    hidden_vertex_ids.add(m.vertex_id)
                n_clusts_reduced += 1

        # Annulment rings: a chi = 0 set leaves no glyph, so re-emit its boundary
        # ring in the ghost-grey annulment style (drawn over the black ring).
        a_ring_pts = vtkPoints()
        a_ring_lines = vtkCellArray()
        a_pt_counter = 0
        for unit in units:
            if unit.unit_id not in annihilated_unit_ids:
                continue
            a_ring = unit.boundary_ring
            n_ar = len(a_ring)
            if n_ar < 3:
                continue
            for rp in a_ring:
                g = to_global(radial_offset(rp.position, 1.014), session.center)
                a_ring_pts.InsertNextPoint(g[0], g[1], g[2])
            a_ring_lines.InsertNextCell(n_ar + 1)
            for k in range(n_ar):
                a_ring_lines.InsertCellPoint(a_pt_counter + k)
            a_ring_lines.InsertCellPoint(a_pt_counter)
            a_pt_counter += n_ar

        annulled_rings_poly.SetPoints(a_ring_pts)
        annulled_rings_poly.SetLines(a_ring_lines)
        annulled_rings_poly.Modified()
        annulled_rings_actor.SetVisibility(bool(state.show_rings and a_pt_counter > 0))
        runtime_annulled_cluster_ids.clear()
        runtime_annulled_cluster_ids.update(annihilated_unit_ids)
        runtime_irreducible_cluster_ids.clear()
        runtime_irreducible_cluster_ids.update(irreducible_unit_ids)

        for cl_dict in clusters_ui:
            cl_dict["is_annulled"] = cl_dict["id"] in annihilated_unit_ids
            cl_dict["is_irreducible"] = cl_dict["id"] in irreducible_unit_ids
        state.clusters_list = clusters_ui

        # 6. Populate CP glyph point sets from the (possibly reduced) CP set.
        maxima_pts.Reset()
        minima_pts.Reset()
        saddles_pts.Reset()
        n_disp_max = 0
        n_disp_min = 0
        n_disp_sad = 0
        # Index sum uses multiplicity for raw saddles and the net index for
        # effective CPs, so collapsing a set never changes the displayed chi.
        index_sum = 0

        # Glyph point sets are drawn by VTK, so they carry world coordinates.
        # The effective-CP loop below reads eff_dict["position"], which the UI
        # builder already mapped to the world frame.
        for mx in morse_res.maxima:
            if mx.vertex_id in hidden_vertex_ids:
                continue
            p = to_global(mx.position, session.center)
            maxima_pts.InsertNextPoint(p[0], p[1], p[2])
            n_disp_max += 1
            index_sum += 1
        for mn in morse_res.minima:
            if mn.vertex_id in hidden_vertex_ids:
                continue
            p = to_global(mn.position, session.center)
            minima_pts.InsertNextPoint(p[0], p[1], p[2])
            n_disp_min += 1
            index_sum += 1
        for sd in morse_res.saddles:
            if sd.vertex_id in hidden_vertex_ids:
                continue
            p = to_global(sd.position, session.center)
            saddles_pts.InsertNextPoint(p[0], p[1], p[2])
            n_disp_sad += 1
            index_sum -= sd.multiplicity

        for eff_dict in effective_cps_ui:
            pos = eff_dict["position"]
            if eff_dict["type"] == "E-MAX":
                maxima_pts.InsertNextPoint(pos[0], pos[1], pos[2])
                n_disp_max += 1
            elif eff_dict["type"] == "E-MIN":
                minima_pts.InsertNextPoint(pos[0], pos[1], pos[2])
                n_disp_min += 1
            else:
                saddles_pts.InsertNextPoint(pos[0], pos[1], pos[2])
                n_disp_sad += 1
            index_sum += int(eff_dict["net_index"])

        maxima_poly.SetPoints(maxima_pts)
        maxima_poly.Modified()
        max_actor.SetVisibility(bool(state.show_cps and state.show_maxima))

        minima_poly.SetPoints(minima_pts)
        minima_poly.Modified()
        min_actor.SetVisibility(bool(state.show_cps and state.show_minima))

        saddles_poly.SetPoints(saddles_pts)
        saddles_poly.Modified()
        sad_actor.SetVisibility(bool(state.show_cps and state.show_saddles))

        # 7. Flatten CP list for UI: visible raw CPs first, then effective CPs.
        cps_ui: list[dict[str, Any]] = []
        for cp in all_raw_cps:
            cp_dict = raw_cp_to_ui(cp, session.center)
            cp_dict["is_hidden"] = cp.vertex_id in hidden_vertex_ids
            cps_ui.append(cp_dict)
        cps_ui.extend(effective_cps_ui)
        state.all_cps_list = sorted(cps_ui, key=lambda x: -x["value"])
        runtime_effective_cps = effective_cps_ui
        runtime_hidden_vertex_ids = hidden_vertex_ids

        # 8. Extract Basin Centers (Basin Locator - Step 1)
        # Raw CPs are centred-frame, the effective dicts are already world-frame;
        # `session.center` lets the subroutine map the former so both agree on the sphere.
        basin_centers = extract_basin_centers(
            raw_cps=all_raw_cps,
            effective_cps=effective_cps_ui,
            hidden_vertex_ids=hidden_vertex_ids,
            center=session.center,
        )
        state.basin_centers_list = [
            {
                "id": bc.center_id,
                "polarity": bc.polarity,
                "badge_color": "error" if bc.polarity == "maximum" else "info",
                "position": list(bc.position),
                "value": bc.value,
                "is_effective": bc.is_effective,
                "vertex_id": bc.vertex_id,
                "origin_id": bc.origin_id,
                "net_index": bc.net_index,
                "member_count": bc.member_count,
            }
            for bc in basin_centers
        ]

        # Refresh active basin session.center selection if previous selection exists
        if state.selected_basin_center:
            curr_id = state.selected_basin_center.get("id")
            matching_bc = next(
                (b for b in state.basin_centers_list if b["id"] == curr_id), None
            )
            if matching_bc:
                select_basin_center(matching_bc)
            else:
                select_basin_center(None)

        # 9. Saddle separatrix launch seeds (Basin Locator - Step 2)
        # Extracted for every visible saddle, including effective saddles, whose
        # centre replaces the degenerate core of the constellation it collapsed.
        launch_saddles = [
            cp for cp in morse_res.saddles if cp.vertex_id not in hidden_vertex_ids
        ]
        launch_pitch = float(state.launch_seed_pitch)
        seeds = compute_all_saddle_launch_seeds(
            saddles=launch_saddles,
            mesh_pts=session.pts_np,
            triangles=session.triangles_np,
            f_vals=f_vals_np,
            sphere_radius=session.sphere_radius,
            delta_theta_mesh_rad=session.delta_theta_rad,
            launch_step_pitch=launch_pitch,
        )

        a_pts, a_lines = vtkPoints(), vtkCellArray()
        d_pts, d_lines = vtkPoints(), vtkCellArray()
        n_ascent = n_descent = n_mismatch = 0
        max_fold = 0

        def _emit_branch(
            seed: SaddleLaunchSeed,
            branch: Any,
            lines: vtkCellArray,
            pts_acc: vtkPoints,
        ) -> None:
            base = pts_acc.GetNumberOfPoints()
            s0 = to_global(radial_offset(seed.saddle_position, 1.016), session.center)
            s1 = to_global(radial_offset(branch.launch_position, 1.016), session.center)
            pts_acc.InsertNextPoint(s0[0], s0[1], s0[2])
            pts_acc.InsertNextPoint(s1[0], s1[1], s1[2])
            lines.InsertNextCell(2)
            lines.InsertCellPoint(base)
            lines.InsertCellPoint(base + 1)

        for seed in seeds:
            max_fold = max(max_fold, seed.fold_order)
            if not seed.index_agrees and seed.fold_order > 0:
                n_mismatch += 1
            for br in seed.ascent_branches:
                _emit_branch(seed, br, a_lines, a_pts)
                n_ascent += 1
            for br in seed.descent_branches:
                _emit_branch(seed, br, d_lines, d_pts)
                n_descent += 1

        launch_ascent_poly.SetPoints(a_pts)
        launch_ascent_poly.SetLines(a_lines)
        launch_ascent_poly.Modified()
        launch_descent_poly.SetPoints(d_pts)
        launch_descent_poly.SetLines(d_lines)
        launch_descent_poly.Modified()

        show_seeds = bool(state.show_launch_seeds)
        launch_ascent_actor.SetVisibility(show_seeds and n_ascent > 0)
        launch_descent_actor.SetVisibility(show_seeds and n_descent > 0)

        state.launch_stats = {
            "n_saddles_seeded": len(seeds),
            "n_ascent": n_ascent,
            "n_descent": n_descent,
            "n_index_mismatch": n_mismatch,
            "max_fold_order": max_fold,
        }

        n_raw_hidden = len(hidden_vertex_ids)
        state.display_counts = {
            "minima": n_disp_min,
            "maxima": n_disp_max,
            "saddles": n_disp_sad,
            "euler": index_sum,
            "valid": index_sum == 2,
        }
        state.reduction_stats = {
            "n_catastrophes_reduced": n_cats_reduced,
            "n_clusters_reduced": n_clusts_reduced,
            "n_clusters_annihilated": n_clusts_annihilated,
            "n_clusters_irreducible": n_clusts_irreducible,
            "n_clusters_bridged": n_bridged,
            "n_effective_cps": len(effective_cps_ui),
            "n_raw_cps_hidden": n_raw_hidden,
            "n_nudged": len(nudge_degs) if do_nudge else 0,
            "max_nudge_deg": max(nudge_degs) if (do_nudge and nudge_degs) else 0.0,
            "mean_nudge_deg": (sum(nudge_degs) / len(nudge_degs))
            if (do_nudge and nudge_degs)
            else 0.0,
            "n_ports_perfected": n_ports_perfected if do_perfect else 0,
        }

    def update_field() -> None:
        cur_field = state.selected_field
        pd = session.sphere_poly.GetPointData()
        arr = pd.GetArray(cur_field)
        if arr:
            f_min, f_max = get_robust_scalar_bounds(arr, lower_pct=2.0, upper_pct=98.0)
            n_levels = max(2, int(state.num_contours))

            if state.show_flood:
                ctf = build_discrete_colormap(n_levels, f_min, f_max)
                flood_mapper.SelectColorArray(cur_field)
                flood_mapper.SetLookupTable(ctf)
                flood_mapper.SetScalarRange(f_min, f_max)
                flood_actor.SetVisibility(True)
            else:
                flood_actor.SetVisibility(False)

            if state.show_contours:
                contour_filter.SetInputArrayToProcess(0, 0, 0, 0, cur_field)
                step_lin = (f_max - f_min) / (n_levels - 1)
                contour_filter.SetNumberOfContours(n_levels)
                for i in range(n_levels):
                    contour_filter.SetValue(i, f_min + i * step_lin)
                contour_filter.Update()
                contour_actor.SetVisibility(True)
            else:
                contour_actor.SetVisibility(False)

        sphere_wire_actor.SetVisibility(bool(state.show_sphere_wire))
        update_topology()
        render_window.Render()
        request_view_update()

    def select_cp(cp_item: dict[str, Any] | None) -> None:
        state.selected_cp = cp_item
        if cp_item is None:
            highlight_actor.SetVisibility(False)
        else:
            p = cp_item["position"]
            scale = float(state.cp_glyph_scale)
            highlight_src.SetRadius(0.075 * scale)
            highlight_src.SetCenter(p[0], p[1], p[2])
            highlight_src.Update()
            highlight_actor.SetVisibility(True)
        render_window.Render()
        request_view_update()

    def select_cluster(cluster_item: dict[str, Any] | None) -> None:
        state.selected_cluster = cluster_item
        if cluster_item is None:
            cluster_members_pts.Reset()
            cluster_members_poly.SetPoints(cluster_members_pts)
            cluster_members_poly.Modified()
            cluster_members_actor.SetVisibility(False)
            highlight_actor.SetVisibility(False)
        else:
            c_pos = cluster_item["centroid"]
            highlight_src.SetRadius(
                max(float(cluster_item.get("spatial_diameter", 0.08)) * 0.75, 0.08)
            )
            highlight_src.SetCenter(c_pos[0], c_pos[1], c_pos[2])
            highlight_src.Update()
            highlight_actor.SetVisibility(True)

            # Highlight all individual member CPs in cyan rings
            cluster_members_pts.Reset()
            for mem in cluster_item.get("members", []):
                m_pos = mem["position"]
                cluster_members_pts.InsertNextPoint(m_pos[0], m_pos[1], m_pos[2])
            cluster_members_poly.SetPoints(cluster_members_pts)
            cluster_members_poly.Modified()
            cluster_members_actor.SetVisibility(
                len(cluster_item.get("members", [])) > 0
            )

        render_window.Render()
        request_view_update()

    def select_catastrophe(cat_item: dict[str, Any] | None) -> None:
        state.selected_catastrophe = cat_item
        if cat_item is None:
            highlight_actor.SetVisibility(False)
        else:
            c_pos = cat_item["centroid"]
            highlight_src.SetRadius(
                max(float(cat_item.get("angular_radius_deg", 5.0)) * 0.02, 0.12)
            )
            highlight_src.SetCenter(c_pos[0], c_pos[1], c_pos[2])
            highlight_src.Update()
            highlight_actor.SetVisibility(True)
        render_window.Render()
        request_view_update()

    def select_basin_center(basin_item: dict[str, Any] | None) -> None:
        state.selected_basin_center = basin_item
        if basin_item is None:
            basin_center_actor.SetVisibility(False)
        else:
            p = basin_item["position"]
            scale = float(state.cp_glyph_scale)
            basin_center_src.SetRadius(0.085 * scale)
            basin_center_src.SetCenter(p[0], p[1], p[2])
            basin_center_src.Update()
            basin_center_actor.SetVisibility(True)
        render_window.Render()
        request_view_update()

    # 3D Direct Picking Ray-Caster
    picker = vtkCellPicker()
    picker.SetTolerance(0.02)
    world_coord = vtkCoordinate()
    world_coord.SetCoordinateSystemToWorld()

    @ctrl.add("on_scene_click")
    def on_scene_click(
        click_x: float | None = None,
        click_y: float | None = None,
        client_w: float | None = None,
        client_h: float | None = None,
    ) -> None:
        try:
            if click_x is None or click_y is None:
                return

            rw_size = render_window.GetSize()
            w, h = rw_size[0], rw_size[1]
            cw = float(client_w) if client_w else float(w)
            ch = float(client_h) if client_h else float(h)
            norm_x = float(click_x) / cw
            norm_y = float(click_y) / ch
            disp_x = norm_x * w
            disp_y = (1.0 - norm_y) * h

            # Ray-pick in 3D world coordinates
            picker.Pick(disp_x, disp_y, 0, renderer)
            picked_actor = picker.GetActor()

            if picked_actor is not None:
                pick_pos = picker.GetPickPosition()
                # runtime_clusters / runtime_cps are algorithm objects living in
                # the centred atom frame, so measure against the pick in that
                # same frame. The UI dicts are already world coordinates.
                pick_local = to_local(pick_pos, session.center)
                # Check closest cluster centroid
                min_cluster_dist = float("inf")
                best_cluster: MicroCluster | None = None
                for cl in runtime_clusters:
                    c_pos = cl.centroid
                    d = math.sqrt(
                        (c_pos[0] - pick_local[0]) ** 2
                        + (c_pos[1] - pick_local[1]) ** 2
                        + (c_pos[2] - pick_local[2]) ** 2
                    )
                    if d < min_cluster_dist:
                        min_cluster_dist = d
                        best_cluster = cl

                # Check closest effective critical point (reduced sets)
                min_eff_dist = float("inf")
                best_eff_dict: dict[str, Any] | None = None
                for eff_dict in runtime_effective_cps:
                    p = eff_dict["position"]
                    d = math.sqrt(
                        (p[0] - pick_pos[0]) ** 2
                        + (p[1] - pick_pos[1]) ** 2
                        + (p[2] - pick_pos[2]) ** 2
                    )
                    if d < min_eff_dist:
                        min_eff_dist = d
                        best_eff_dict = eff_dict

                # Check closest CP (raw CPs collapsed into an effective CP are skipped)
                min_cp_dist = float("inf")
                best_cp: CriticalPoint | None = None
                for cp in runtime_cps:
                    if cp.vertex_id in runtime_hidden_vertex_ids:
                        continue
                    p = cp.position
                    d = math.sqrt(
                        (p[0] - pick_local[0]) ** 2
                        + (p[1] - pick_local[1]) ** 2
                        + (p[2] - pick_local[2]) ** 2
                    )
                    if d < min_cp_dist:
                        min_cp_dist = d
                        best_cp = cp

                # If clicked close to a cluster centroid (within spatial radius or 0.15 A)
                if best_cluster is not None and min_cluster_dist <= max(
                    best_cluster.spatial_diameter, 0.15
                ):
                    for cl_dict in state.clusters_list:
                        # A bridged corridor row is keyed by its MS id, so match
                        # the picked raw cluster against the row's origin list.
                        if (
                            cl_dict["id"] == best_cluster.cluster_id
                            or best_cluster.cluster_id in cl_dict.get("origin_ids", [])
                        ):
                            select_cluster(cl_dict)
                            return

                # Effective CP glyphs sit at set centroids, so test them first
                if best_eff_dict is not None and min_eff_dist <= 0.15:
                    select_cp(best_eff_dict)
                    return

                # Otherwise check CP
                if best_cp is not None and min_cp_dist <= 0.15:
                    for cp_dict in state.all_cps_list:
                        if (
                            not cp_dict.get("is_effective", False)
                            and cp_dict["vertex_id"] == best_cp.vertex_id
                        ):
                            select_cp(cp_dict)
                            return

            # Screen-space proximity fallback
            min_screen_dist = float("inf")
            screen_cluster_dict: dict[str, Any] | None = None
            for cl_dict in state.clusters_list:
                c_pos = cl_dict["centroid"]
                world_coord.SetValue(c_pos[0], c_pos[1], c_pos[2])
                item_disp = world_coord.GetComputedDisplayValue(renderer)
                sdx = item_disp[0] - disp_x
                sdy = item_disp[1] - disp_y
                dist = math.sqrt(sdx * sdx + sdy * sdy)
                if dist < min_screen_dist:
                    min_screen_dist = dist
                    screen_cluster_dict = cl_dict

            if screen_cluster_dict is not None and min_screen_dist <= 35.0:
                select_cluster(screen_cluster_dict)
                return

            # Screen-space fallback for effective CP glyphs
            min_eff_screen = float("inf")
            eff_screen_dict: dict[str, Any] | None = None
            for eff_dict in runtime_effective_cps:
                p = eff_dict["position"]
                world_coord.SetValue(p[0], p[1], p[2])
                item_disp = world_coord.GetComputedDisplayValue(renderer)
                sdx = item_disp[0] - disp_x
                sdy = item_disp[1] - disp_y
                dist = math.sqrt(sdx * sdx + sdy * sdy)
                if dist < min_eff_screen:
                    min_eff_screen = dist
                    eff_screen_dict = eff_dict

            if eff_screen_dict is not None and min_eff_screen <= 25.0:
                select_cp(eff_screen_dict)
                return

            # Clear selection if background clicked
            select_cluster(None)
            select_cp(None)
        except (ValueError, TypeError, RuntimeError) as e:
            print(f"[sphere_viewer] Picking error: {e}")

    @state.change(
        "selected_field",
        "show_flood",
        "show_contours",
        "show_sphere_wire",
        "num_contours",
        "show_cps",
        "show_maxima",
        "show_minima",
        "show_saddles",
        "k_pitch",
        "cp_glyph_scale",
        "show_rings",
        "aggregate_strings",
        "k_bridge",
        "show_catastrophes",
        "show_ports",
        "cat_barrier_tol",
        "cat_angular_tol_mult",
        "reduce_catastrophes",
        "reduce_multi_clusters",
        "nudge_cp_positions",
        "perfect_port_angles",
        "nudge_max_pitch",
        "show_launch_seeds",
        "launch_seed_pitch",
    )
    def on_param_change(**kwargs: Any) -> None:
        if suppress_param_change:
            return
        update_field()

    @ctrl.add("reset_defaults")
    def reset_defaults() -> None:
        """Restore every analysis control to its canonical baseline.

        Re-seeds all sliders, switches and reduction toggles from
        ``DEFAULT_UI_STATE`` and clears any active selection. The selected
        condensed field is deliberately preserved: switching fields is a data
        choice, not an analysis parameter, and resetting it would discard the
        operator's context.

        The whole write is performed with ``suppress_param_change`` raised so
        the batched state update rebuilds the pipeline exactly once instead of
        once per changed key.
        """
        nonlocal suppress_param_change
        suppress_param_change = True
        try:
            state.update(dict(DEFAULT_UI_STATE))
            state.selected_cp = None
            state.selected_cluster = None
            state.selected_catastrophe = None
            state.selected_basin_center = None
        finally:
            suppress_param_change = False

        highlight_actor.SetVisibility(False)
        basin_center_actor.SetVisibility(False)
        launch_ascent_actor.SetVisibility(False)
        launch_descent_actor.SetVisibility(False)
        cluster_members_pts.Reset()
        cluster_members_poly.SetPoints(cluster_members_pts)
        cluster_members_poly.Modified()
        cluster_members_actor.SetVisibility(False)

        update_field()

    def active_camera() -> Any:
        """Return the render window's active camera.

        Returns:
            The ``vtkCamera`` the renderer is currently using.
        """
        return renderer.GetActiveCamera()

    def capture_current_camera() -> CameraState:
        """Snapshot the camera the operator is currently looking through.

        Trame mirrors the browser's camera onto the server-side camera
        whenever the user interacts, so the server camera reflects the
        current viewing direction and zoom.

        Returns:
            The captured camera state.
        """
        return capture_camera_state(active_camera())

    def apply_camera(camera_state: CameraState) -> None:
        """Apply a camera state to the renderer, re-based onto the live atom.

        The snapshot is stored relative to the nucleus it was captured
        around, so it is translated onto the atom that is currently loaded
        before being applied. That keeps the camera pointing at the atom
        even when the snapshot was taken against a different one.

        Args:
            camera_state: The camera to apply.
        """
        rebased = rebase_camera_state(camera_state, (0.0, 0.0, 0.0), session.center)
        apply_camera_state(active_camera(), rebased)
        renderer.ResetCameraClippingRange()
        render_window.Render()
        request_view_update()

    def apply_restore_plan(plan: RestorePlan) -> None:
        """Apply a resolved restore plan to the live viewer.

        Re-opens the file or atom the snapshot names, pushes the control
        values into the state, and restores the camera. The whole write is
        performed with ``suppress_param_change`` raised so the batched
        state update rebuilds the pipeline exactly once instead of once per
        changed key.

        Args:
            plan: The plan to apply.
        """
        nonlocal suppress_param_change

        needs_reload = plan.plt_path is not None or plan.atom_number is not None
        if needs_reload:
            target_plt = plan.plt_path or session.plt_path
            target_atom = (
                plan.atom_number
                if plan.atom_number is not None
                else session.atom_number
            )
            if not os.path.exists(target_plt):
                raise FileNotFoundError(f"PLT file not found: {target_plt}")
            new_session = build_viewer_session(target_plt, target_atom)
            # Swap the mesh-bound state in place so every closure that reads
            # from ``session`` picks up the new mesh.
            session.plt_path = new_session.plt_path
            session.atom_number = new_session.atom_number
            session.sphere_poly = new_session.sphere_poly
            session.field_items = new_session.field_items
            session.atom_info = new_session.atom_info
            session.center = new_session.center
            session.pts_np = new_session.pts_np
            session.triangles_np = new_session.triangles_np
            session.render_poly = new_session.render_poly
            session.geom_info = new_session.geom_info
            session.sphere_radius = new_session.sphere_radius
            session.delta_theta_rad = new_session.delta_theta_rad

            # Re-point the drawing pipeline at the new mesh.
            flood_mapper.SetInputData(session.render_poly)
            contour_filter.SetInputData(session.render_poly)
            sphere_wire_mapper.SetInputData(session.render_poly)
            state.plt_file = os.path.basename(session.plt_path)
            state.atom_info = session.atom_info
            state.geom_info = _geom_info_for_state(session)
            state.field_items = session.field_items
            render_window.SetWindowName(
                f"gba_topology2 - Sphere Viewer: {session.atom_info['atom_name']}"
            )

        suppress_param_change = True
        try:
            state.update(plan.controls)
        finally:
            suppress_param_change = False

        # A reload may have changed which fields exist, so re-resolve the
        # requested field against the mesh that is now loaded.
        if plan.selected_field is not None:
            available = [f["value"] for f in session.field_items]
            if plan.selected_field in available:
                state.selected_field = plan.selected_field
            elif available:
                state.selected_field = available[0]

        # The camera is captured relative to the nucleus it was taken around;
        # the plan re-based it onto the origin, so applying it re-bases it
        # onto the live atom.
        if plan.camera is not None:
            apply_camera(plan.camera)

        update_field()

    def load_params_from_file(path: str) -> None:
        """Load a parameter snapshot from disk and restore the viewer.

        Args:
            path: Path of the JSON snapshot to read.

        Raises:
            OSError: If the file cannot be read.
            ValueError: If the snapshot is malformed or has the wrong schema.
            KeyError: If a required control key is missing.
            TypeError: If a control value has the wrong type.
            FileNotFoundError: If the snapshot names a .plt that is missing.
        """
        with open(path, encoding="utf-8") as fh:
            payload = json.load(fh)
        params = load_params_from_dict(payload)
        plan = resolve_restore_plan(
            params,
            current_plt_path=session.plt_path,
            current_atom_number=session.atom_number,
            source_path=os.path.abspath(path),
        )
        apply_restore_plan(plan)
        state.params_last_loaded_path = os.path.abspath(path)
        print(f"[sphere_viewer] Parameters loaded from: {os.path.abspath(path)}")

    @ctrl.add("save_parameters")
    def save_parameters() -> None:
        """Write every editable control, context, camera and metric to JSON.

        Diagnostic capture for reproducing an observed viewer state in
        headless scripts. The snapshot goes to
        ``default_params_path(plt_file, atom_num)`` next to the .plt file,
        with ``indent=2`` and sorted keys for stable diffs. The absolute
        path is printed to the terminal and mirrored into
        ``state.params_last_saved_path`` for the drawer confirmation chip.
        The write is a pure state read, so no pipeline recompute is needed.
        """
        try:
            payload = save_params_to_dict(
                state,
                plt_path=session.plt_path,
                atom_info={
                    **session.atom_info,
                    "field_items": session.field_items,
                },
                atom_num=session.atom_number,
                camera=capture_current_camera(),
                camera_center=session.center,
            )
            out_path = default_params_path(session.plt_path, session.atom_number)
            with open(out_path, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2, sort_keys=True)
                fh.write("\n")
            state.params_last_saved_path = out_path
            print(f"[sphere_viewer] Parameters saved to: {out_path}")
        except OSError as exc:
            print(f"[sphere_viewer] Parameter save failed: {exc}")

    @ctrl.add("load_parameters")
    def load_parameters(params_path: str = "") -> None:
        """Restore the viewer from a parameter snapshot.

        The snapshot may name a different .plt, a different atom and a
        different field; the viewer opens whatever it asks for. Failures
        are reported in the terminal and surfaced in the drawer's status
        chip rather than raised into the trame event loop.

        Args:
            params_path: Path of the JSON snapshot to read.
        """
        if not params_path:
            state.params_load_error = "No parameter file selected"
            return
        try:
            load_params_from_file(params_path)
            state.params_load_error = ""
        except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
            state.params_load_error = str(exc)
            print(f"[sphere_viewer] Parameter load failed: {exc}")

    @ctrl.add("open_load_dialog")
    def open_load_dialog() -> None:
        """Open the "Load Parameters" dialog.

        Any previous error is cleared so the dialog never shows a stale
        message from an earlier attempt.
        """
        state.params_load_error = ""
        state.load_dialog_open = True

    @ctrl.add("close_load_dialog")
    def close_load_dialog() -> None:
        """Close the "Load Parameters" dialog without loading anything."""
        state.load_dialog_open = False
        state.load_dialog_file = None

    @ctrl.add("confirm_load_parameters")
    def confirm_load_parameters() -> None:
        """Load the file chosen in the "Load Parameters" dialog.

        The file is read from the uploaded content, so it does not need to
        exist at a path the server knows about.
        """
        uploaded = state.load_dialog_file
        content = _uploaded_file_content(uploaded)
        if content is None:
            state.params_load_error = "No parameter file selected"
            return
        try:
            payload = json.loads(content.decode("utf-8"))
            params = load_params_from_dict(payload)
            plan = resolve_restore_plan(
                params,
                current_plt_path=session.plt_path,
                current_atom_number=session.atom_number,
                source_path=str(uploaded.get("name", "")),
            )
            apply_restore_plan(plan)
            state.params_load_error = ""
            state.params_last_loaded_path = str(uploaded.get("name", ""))
            state.load_dialog_open = False
            state.load_dialog_file = None
            print(
                "[sphere_viewer] Parameters loaded from: "
                f"{state.params_last_loaded_path}"
            )
        except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
            state.params_load_error = str(exc)
            print(f"[sphere_viewer] Parameter load failed: {exc}")

    @ctrl.add("select_basin_center_from_list")
    def select_basin_center_from_list(basin_id: str) -> None:
        for bc in state.basin_centers_list:
            if bc.get("id") == basin_id:
                select_basin_center(bc)
                return

    @ctrl.add("clear_basin_center_selection")
    def clear_basin_center_selection() -> None:
        select_basin_center(None)

    # Initial pipeline update
    update_field()

    # UI Layout
    with SinglePageWithDrawerLayout(server, vuetify_config=VUE3_THEME_CONFIG) as layout:
        layout.title.set_text(f"gba_topology2: Atom Sphere ({atom_info['atom_name']})")
        layout.drawer.width = 400

        with layout.drawer:
            # No CSS here on purpose. ``html.Style`` renders a <style> tag inside
            # the drawer's Vue template, and Vue strips <style> from templates,
            # so nothing in a scoped block ever reaches the document. All colors
            # and edges come from VUE3_THEME_CONFIG (see the layout construction
            # above), which is the only mechanism that actually applies.
            with v3.VContainer(fluid=True, classes="pa-3"):
                # Confirmation chip for the last "Save Parameters" write.
                with html.Div(v_if="params_last_saved_path", classes="mb-2"):
                    v3.VChip(
                        "Saved: {{ params_last_saved_path }}",
                        size="x-small",
                        color="success",
                        classes="font-mono",
                    )

                # Confirmation chip for the last "Load Parameters" read.
                with html.Div(v_if="params_last_loaded_path", classes="mb-2"):
                    v3.VChip(
                        "Loaded: {{ params_last_loaded_path }}",
                        size="x-small",
                        color="info",
                        classes="font-mono",
                    )

                # Error chip for a failed "Load Parameters" attempt.
                with html.Div(v_if="params_load_error", classes="mb-2"):
                    v3.VChip(
                        "Load failed: {{ params_load_error }}",
                        size="x-small",
                        color="error",
                        classes="font-mono",
                    )

                # Atom & Mesh Geometry Card (Step 1 metrics)
                with v3.VCard(elevation=2, classes="mb-3", color="surface-variant"):
                    with v3.VCardItem():
                        with v3.VCardTitle(
                            classes="text-subtitle-1 font-weight-bold d-flex align-center"
                        ):
                            v3.VIcon("mdi-atom", classes="mr-2", color="success")
                            html.Span(f"Atom: {atom_info['atom_name']}")
                        v3.VCardSubtitle(f"File: {state.plt_file}")

                    v3.VDivider()
                    with v3.VCardText(classes="pt-2 pb-2"):
                        with v3.VRow(dense=True):
                            with v3.VCol(cols=6):
                                html.Div(
                                    "Mesh Vertices",
                                    classes="text-caption text-medium-emphasis",
                                )
                                html.Div(
                                    "{{ geom_info.num_vertices }}",
                                    classes="text-body-2 font-weight-bold",
                                )
                            with v3.VCol(cols=6):
                                html.Div(
                                    "Mesh Triangles",
                                    classes="text-caption text-medium-emphasis",
                                )
                                html.Div(
                                    "{{ geom_info.num_triangles }}",
                                    classes="text-body-2 font-weight-bold",
                                )

                        with v3.VRow(dense=True, classes="mt-1"):
                            with v3.VCol(cols=6):
                                html.Div(
                                    "Avg Edge Length (l_e)",
                                    classes="text-caption text-medium-emphasis",
                                )
                                html.Div(
                                    "{{ geom_info.avg_edge_length.toFixed(4) }} Å",
                                    classes="text-body-2 font-weight-bold",
                                )
                            with v3.VCol(cols=6):
                                html.Div(
                                    "Mesh Pitch (δθ)",
                                    classes="text-caption text-medium-emphasis",
                                )
                                html.Div(
                                    "{{ geom_info.mesh_pitch_deg.toFixed(2) }}°",
                                    classes="text-body-2 font-weight-bold text-success",
                                )

                        with v3.VRow(dense=True, classes="mt-1"):
                            with v3.VCol(cols=6):
                                html.Div(
                                    "Sphere Radius R ({{ geom_info.radius_source }})",
                                    classes="text-caption text-medium-emphasis",
                                )
                                html.Div(
                                    "{{ geom_info.sphere_radius.toFixed(4) }} Å",
                                    classes="text-body-2 font-weight-bold",
                                )
                            with v3.VCol(cols=6):
                                html.Div(
                                    "Nucleus |x₀| from origin",
                                    classes="text-caption text-medium-emphasis",
                                )
                                html.Div(
                                    "{{ geom_info.center_offset.toFixed(4) }} Å "
                                    "({{ geom_info.center[0].toFixed(3) }}, "
                                    "{{ geom_info.center[1].toFixed(3) }}, "
                                    "{{ geom_info.center[2].toFixed(3) }})",
                                    classes="text-caption font-mono font-weight-bold text-info",
                                )

                # Field Controls
                with v3.VCard(elevation=2, classes="mb-3", color="surface-variant"):
                    with v3.VCardItem():
                        with v3.VCardTitle(classes="text-subtitle-2 font-weight-bold"):
                            html.Span("Condensed Field & Mesh")

                    v3.VDivider()
                    with v3.VCardText(classes="pt-3 pb-2"):
                        v3.VSelect(
                            label="Select Field",
                            items=("field_items",),
                            v_model=("selected_field",),
                            density="compact",
                            variant="outlined",
                            classes="mb-2",
                        )

                        v3.VSwitch(
                            label="Surface Color Flood",
                            v_model=("show_flood",),
                            density="compact",
                            color="primary",
                            hide_details=True,
                            classes="mb-1",
                        )

                        v3.VSwitch(
                            label="Isocontour Lines",
                            v_model=("show_contours",),
                            density="compact",
                            color="primary",
                            hide_details=True,
                            classes="mb-1",
                        )

                        with html.Div(v_if="show_contours", classes="mb-2"):
                            with html.Div(
                                classes="d-flex justify-space-between align-center mt-1"
                            ):
                                html.Div(
                                    "Number of Contours",
                                    classes="text-caption font-weight-bold text-medium-emphasis",
                                )
                                html.Div(
                                    "{{ num_contours }}",
                                    classes="text-caption font-weight-bold text-primary",
                                )

                            v3.VSlider(
                                min=3,
                                max=50,
                                step=1,
                                v_model=("num_contours",),
                                density="compact",
                                thumb_label=False,
                                color="primary",
                                classes="mt-1",
                            )

                        v3.VSwitch(
                            label="Show Mesh Wireframe",
                            v_model=("show_sphere_wire",),
                            density="compact",
                            color="secondary",
                            hide_details=True,
                        )

                # Step 1: Discrete Morse Detection
                with v3.VCard(elevation=2, classes="mb-3", color="surface-variant"):
                    with v3.VCardItem():
                        with v3.VCardTitle(
                            classes="text-subtitle-2 font-weight-bold d-flex align-center justify-space-between"
                        ):
                            html.Span("Step 1: Morse Detection (τ = 0.0)")
                            v3.VChip(
                                "Euler χ={{ morse_counts.euler }}",
                                size="x-small",
                                color="success"
                                if ("morse_counts.valid", True)
                                else "error",
                                classes="font-weight-bold",
                            )

                    v3.VDivider()
                    with v3.VCardText(classes="pt-2 pb-2"):
                        v3.VSwitch(
                            label="Display Critical Points",
                            v_model=("show_cps",),
                            density="compact",
                            color="primary",
                            hide_details=True,
                            classes="mb-1",
                        )

                        with html.Div(v_if="show_cps"):
                            with v3.VRow(dense=True, classes="align-center mb-1"):
                                with v3.VCol(cols=4):
                                    v3.VCheckbox(
                                        label="Max (Red)",
                                        v_model=("show_maxima",),
                                        density="compact",
                                        color="error",
                                        hide_details=True,
                                    )
                                with v3.VCol(cols=4):
                                    v3.VCheckbox(
                                        label="Min (Blue)",
                                        v_model=("show_minima",),
                                        density="compact",
                                        color="info",
                                        hide_details=True,
                                    )
                                with v3.VCol(cols=4):
                                    v3.VCheckbox(
                                        label="Sad (Green)",
                                        v_model=("show_saddles",),
                                        density="compact",
                                        color="success",
                                        hide_details=True,
                                    )

                            with html.Div(classes="mt-2 pa-2 rounded surface"):
                                with v3.VRow(
                                    dense=True, classes="text-caption text-center"
                                ):
                                    with v3.VCol(cols=4):
                                        html.Div(
                                            "Maxima", classes="text-medium-emphasis"
                                        )
                                        html.Div(
                                            "{{ morse_counts.maxima }}",
                                            classes="text-subtitle-2 font-weight-bold text-error",
                                        )
                                    with v3.VCol(cols=4):
                                        html.Div(
                                            "Minima", classes="text-medium-emphasis"
                                        )
                                        html.Div(
                                            "{{ morse_counts.minima }}",
                                            classes="text-subtitle-2 font-weight-bold text-info",
                                        )
                                    with v3.VCol(cols=4):
                                        html.Div(
                                            "Saddles", classes="text-medium-emphasis"
                                        )
                                        html.Div(
                                            "{{ morse_counts.saddles }}",
                                            classes="text-subtitle-2 font-weight-bold text-success",
                                        )

                            with html.Div(
                                v_if="reduction_stats.n_effective_cps > 0",
                                classes="text-caption font-mono text-center mt-1 text-teal",
                            ):
                                html.Div(
                                    "Displayed after reduction: {{ display_counts.maxima }} Max / {{ display_counts.minima }} Min / {{ display_counts.saddles }} Sad (χ = {{ display_counts.euler }})"
                                )

                            with html.Div(
                                classes="d-flex justify-space-between align-center mt-3"
                            ):
                                html.Div(
                                    "Glyph Size Scale",
                                    classes="text-caption font-weight-bold text-medium-emphasis",
                                )
                                html.Div(
                                    "{{ Number(cp_glyph_scale).toFixed(1) }}x",
                                    classes="text-caption font-weight-bold text-primary",
                                )

                            v3.VSlider(
                                min=0.2,
                                max=3.0,
                                step=0.1,
                                v_model=("cp_glyph_scale",),
                                density="compact",
                                thumb_label=False,
                                color="primary",
                                classes="mt-1",
                            )

                # Step 2: Micro-Clustering & Interactive 3D Picking
                with v3.VCard(elevation=2, classes="mb-3", color="surface-variant"):
                    with v3.VCardItem():
                        with v3.VCardTitle(
                            classes="text-subtitle-2 font-weight-bold d-flex align-center justify-space-between"
                        ):
                            html.Span("Step 2: Micro-Clusters")
                            v3.VChip(
                                "{{ clusters_list.filter(c => c.is_multi_cp).length }} multi-CP",
                                size="x-small",
                                color="warning",
                            )

                    v3.VDivider()
                    with v3.VCardText(classes="pt-2 pb-2"):
                        v3.VSwitch(
                            label="Show Cluster Boundary Rings",
                            v_model=("show_rings",),
                            density="compact",
                            color="warning",
                            hide_details=True,
                            classes="mb-1",
                        )

                        with html.Div(
                            classes="d-flex justify-space-between align-center mt-2"
                        ):
                            html.Div(
                                "Cluster Pitch Multiplier (k · δθ)",
                                classes="text-caption font-weight-bold text-medium-emphasis",
                            )
                            html.Div(
                                "{{ Number(k_pitch).toFixed(1) }}x ({{ (k_pitch * geom_info.mesh_pitch_deg).toFixed(1) }}°)",
                                classes="text-caption font-weight-bold text-warning",
                            )

                        v3.VSlider(
                            min=1.0,
                            max=5.0,
                            step=0.2,
                            v_model=("k_pitch",),
                            density="compact",
                            thumb_label=False,
                            color="warning",
                            classes="mt-1",
                        )

                        # Clusters table
                        html.Div(
                            "Micro-Clusters (Click in 3D or list to inspect)",
                            classes="text-caption font-weight-bold text-medium-emphasis mt-2 mb-1",
                        )
                        with v3.VTable(
                            density="compact",
                            classes="elevation-0",
                            style="max-height: 180px; overflow-y: auto;",
                        ):
                            with html.Thead():
                                with html.Tr():
                                    html.Th(
                                        "ID",
                                        classes="text-left text-caption font-weight-bold",
                                    )
                                    html.Th(
                                        "Composition",
                                        classes="text-left text-caption font-weight-bold",
                                    )
                                    html.Th(
                                        "χ",
                                        classes="text-right text-caption font-weight-bold",
                                    )
                            with html.Tbody():
                                with html.Tr(
                                    v_for="cl in clusters_list",
                                    key="cl.id",
                                    click=(ctrl.select_cluster_from_list, "[cl.id]"),
                                    classes="cursor-pointer",
                                ):
                                    with html.Td(
                                        classes="text-caption font-weight-bold"
                                    ):
                                        html.Span("{{ cl.id }} ({{ cl.size }}) ")
                                        v3.VChip(
                                            "annulled",
                                            size="x-small",
                                            color="grey-darken-1",
                                            classes="ml-1",
                                            v_if="cl.is_annulled",
                                        )
                                        v3.VChip(
                                            "irred",
                                            size="x-small",
                                            color="deep-orange-darken-1",
                                            classes="ml-1",
                                            v_if="cl.is_irreducible",
                                        )
                                        v3.VChip(
                                            "corridor",
                                            size="x-small",
                                            color="cyan-darken-2",
                                            classes="ml-1",
                                            v_if="cl.is_string",
                                        )
                                    with html.Td():
                                        v3.VChip(
                                            "{{ cl.composition_summary }}",
                                            size="x-small",
                                            color="amber-darken-3"
                                            if ("cl.is_multi_cp", True)
                                            else "default",
                                        )
                                    html.Td(
                                        "{{ cl.local_euler_index >= 0 ? '+' + cl.local_euler_index : cl.local_euler_index }}",
                                        classes="text-right text-caption font-mono",
                                    )

                # Inspector Card: Selected Micro-Cluster
                with v3.VCard(
                    v_if="selected_cluster",
                    elevation=3,
                    classes="mb-3 border-warning",
                    color="surface",
                ):
                    with v3.VCardItem():
                        with v3.VCardTitle(
                            classes="text-subtitle-1 font-weight-bold d-flex align-center justify-space-between"
                        ):
                            with html.Div(classes="d-flex align-center"):
                                v3.VIcon(
                                    "mdi-chart-bubble", classes="mr-2", color="warning"
                                )
                                html.Span(
                                    "{{ selected_cluster.is_string ? 'Corridor' : 'Cluster' }} {{ selected_cluster.id }} ({{ selected_cluster.size }} CPs)"
                                )
                            v3.VBtn(
                                icon="mdi-close",
                                variant="text",
                                density="compact",
                                click=ctrl.clear_cluster_selection,
                            )
                        v3.VCardSubtitle(
                            "Composition: {{ selected_cluster.composition_summary }} | Net Index: {{ selected_cluster.local_euler_index >= 0 ? '+' + selected_cluster.local_euler_index : selected_cluster.local_euler_index }}"
                        )
                        html.Div(
                            "Index breakdown: +{{ selected_cluster.n_maxima }} (Max) + {{ selected_cluster.n_minima }} (Min) - {{ selected_cluster.total_saddle_index }} (Saddles) = {{ selected_cluster.local_euler_index >= 0 ? '+' + selected_cluster.local_euler_index : selected_cluster.local_euler_index }}",
                            classes="text-caption text-medium-emphasis font-italic px-4 pb-1",
                        )
                        with html.Div(
                            v_if="selected_cluster.is_string",
                            classes="text-caption font-mono px-4 pb-1 text-cyan",
                        ):
                            html.Div(
                                "Bridged from: {{ selected_cluster.origin_ids.join(', ') }}"
                            )

                    v3.VDivider()
                    with v3.VCardText(classes="pt-2"):
                        with v3.VRow(dense=True):
                            with v3.VCol(cols=6):
                                html.Div(
                                    "Morphology / Aspect Ratio",
                                    classes="text-caption text-medium-emphasis",
                                )
                                html.Div(
                                    "{{ selected_cluster.morphology === 'string' ? 'String (Type 2)' : 'Compact (Type 1)' }} (AR: {{ selected_cluster.aspect_ratio ? selected_cluster.aspect_ratio.toFixed(2) : '1.00' }})",
                                    classes="text-body-2 font-weight-bold",
                                )
                            with v3.VCol(cols=6):
                                html.Div(
                                    "Spatial Diam",
                                    classes="text-caption text-medium-emphasis",
                                )
                                html.Div(
                                    "{{ selected_cluster.spatial_diameter.toFixed(4) }} Å",
                                    classes="text-body-2 font-weight-bold",
                                )

                        with v3.VRow(dense=True, classes="mt-1"):
                            with v3.VCol(cols=6):
                                html.Div(
                                    "Angular Radius / Diam",
                                    classes="text-caption text-medium-emphasis",
                                )
                                html.Div(
                                    "{{ selected_cluster.angular_radius_deg.toFixed(1) }}° / {{ selected_cluster.angular_diameter_deg.toFixed(1) }}°",
                                    classes="text-body-2 font-weight-bold",
                                )
                            with v3.VCol(cols=6):
                                html.Div(
                                    "Centroid Unit Vector (X, Y, Z)",
                                    classes="text-caption text-medium-emphasis",
                                )
                                html.Div(
                                    "({{ selected_cluster.centroid_unit[0].toFixed(3) }}, {{ selected_cluster.centroid_unit[1].toFixed(3) }}, {{ selected_cluster.centroid_unit[2].toFixed(3) }})",
                                    classes="text-caption font-mono",
                                )

                        html.Div(
                            "Constituent Critical Points",
                            classes="text-caption font-weight-bold text-medium-emphasis mt-2 mb-1",
                        )
                        with v3.VList(density="compact", classes="pa-0 bg-transparent"):
                            with v3.VListItem(
                                v_for="mem in selected_cluster.members",
                                key="mem.id_label",
                                click=(ctrl.select_cp_from_list, "[mem.id_label]"),
                                classes="px-1 py-0 cursor-pointer",
                            ):
                                with html.Template(v_slot_prepend=True):
                                    v3.VChip(
                                        "{{ mem.id_label }}",
                                        size="x-small",
                                        color="primary",
                                        classes="mr-2",
                                    )
                                html.Span(
                                    "{{ mem.type }}{{ mem.type === 'saddle' && mem.multiplicity > 1 ? ' (mult ' + mem.multiplicity + ', idx -' + mem.multiplicity + ')' : '' }} | Val: {{ mem.value.toFixed(5) }}",
                                    classes="text-caption",
                                )

                # Step 4: Saddle String / Corridor Aggregation Card
                with v3.VCard(elevation=2, classes="mb-3", color="surface-variant"):
                    with v3.VCardItem():
                        with v3.VCardTitle(
                            classes="text-subtitle-2 font-weight-bold d-flex align-center justify-space-between"
                        ):
                            html.Span("Step 4: Saddle String Aggregation")
                            v3.VChip(
                                "{{ reduction_stats.n_clusters_bridged }} bridged",
                                size="x-small",
                                color="cyan",
                            )

                    v3.VDivider()
                    with v3.VCardText(classes="pt-2 pb-2"):
                        v3.VSwitch(
                            label="Bridge Collinear Corridor Fragments",
                            v_model=("aggregate_strings",),
                            density="compact",
                            color="cyan",
                            hide_details=True,
                            classes="mb-1",
                        )
                        html.Div(
                            "The clustering radius is isotropic, so a ridge/valley "
                            "corridor is often cut into fragments that individually "
                            "annul (chi = 0) or reduce to a spurious extremum. "
                            "Bridging rejoins them so each corridor collapses to one "
                            "effective CP carrying its summed index.",
                            classes="text-caption text-medium-emphasis mb-2",
                        )

                        with html.Div(
                            classes="d-flex justify-space-between align-center mt-1"
                        ):
                            html.Div(
                                "Bridge Radius (k · δθ)",
                                classes="text-caption font-weight-bold text-medium-emphasis",
                            )
                            html.Div(
                                "{{ Number(k_bridge).toFixed(1) }}x ({{ (k_bridge * geom_info.mesh_pitch_deg).toFixed(1) }}°)",
                                classes="text-caption font-weight-bold text-cyan",
                            )

                        v3.VSlider(
                            min=1.0,
                            max=6.0,
                            step=0.25,
                            v_model=("k_bridge",),
                            density="compact",
                            thumb_label=False,
                            color="cyan",
                            classes="mt-1",
                        )

                # Step 5: Catastrophes (Monkey & Octupolar Saddles) & Boundary Ports Card
                with v3.VCard(elevation=2, classes="mb-3", color="surface-variant"):
                    with v3.VCardItem():
                        with v3.VCardTitle(
                            classes="text-subtitle-2 font-weight-bold d-flex align-center justify-space-between"
                        ):
                            html.Span("Step 5: Catastrophes & Ports")
                            v3.VChip(
                                "{{ catastrophes_list.length }} found",
                                size="x-small",
                                color="purple",
                            )

                    v3.VDivider()
                    with v3.VCardText(classes="pt-2 pb-2"):
                        v3.VSwitch(
                            label="Show Catastrophe Rings (Purple)",
                            v_model=("show_catastrophes",),
                            density="compact",
                            color="purple",
                            hide_details=True,
                            classes="mb-1",
                        )
                        v3.VSwitch(
                            label="Show Boundary Ports (Orange=Valley, Cyan=Ridge)",
                            v_model=("show_ports",),
                            density="compact",
                            color="info",
                            hide_details=True,
                            classes="mb-2",
                        )

                        with html.Div(
                            classes="d-flex justify-space-between align-center mt-1"
                        ):
                            html.Div(
                                "Barrier Tol (Relative)",
                                classes="text-caption font-weight-bold text-medium-emphasis",
                            )
                            html.Div(
                                "{{ (cat_barrier_tol * 100).toFixed(2) }}%",
                                classes="text-caption font-weight-bold text-purple",
                            )

                        v3.VSlider(
                            min=0.001,
                            max=0.05,
                            step=0.001,
                            v_model=("cat_barrier_tol",),
                            density="compact",
                            thumb_label=False,
                            color="purple",
                            classes="mt-1",
                        )

                        with html.Div(
                            classes="d-flex justify-space-between align-center mt-1"
                        ):
                            html.Div(
                                "Angular Tol (Span Multiplier)",
                                classes="text-caption font-weight-bold text-medium-emphasis",
                            )
                            html.Div(
                                "{{ Number(cat_angular_tol_mult).toFixed(1) }}x δθ",
                                classes="text-caption font-weight-bold text-purple",
                            )

                        v3.VSlider(
                            min=2.0,
                            max=10.0,
                            step=0.5,
                            v_model=("cat_angular_tol_mult",),
                            density="compact",
                            thumb_label=False,
                            color="purple",
                            classes="mt-1",
                        )

                        # Catastrophes table
                        html.Div(
                            "Catastrophes (Click to inspect)",
                            classes="text-caption font-weight-bold text-medium-emphasis mt-2 mb-1",
                        )
                        with v3.VTable(
                            density="compact",
                            classes="elevation-0",
                            style="max-height: 150px; overflow-y: auto;",
                        ):
                            with html.Thead():
                                with html.Tr():
                                    html.Th(
                                        "ID",
                                        classes="text-left text-caption font-weight-bold",
                                    )
                                    html.Th(
                                        "Type",
                                        classes="text-left text-caption font-weight-bold",
                                    )
                                    html.Th(
                                        "χ",
                                        classes="text-center text-caption font-weight-bold",
                                    )
                                    html.Th(
                                        "Class",
                                        classes="text-right text-caption font-weight-bold",
                                    )
                            with html.Tbody():
                                with html.Tr(
                                    v_for="cat in catastrophes_list",
                                    key="cat.id",
                                    click=(
                                        ctrl.select_catastrophe_from_list,
                                        "[cat.id]",
                                    ),
                                    classes="cursor-pointer",
                                ):
                                    html.Td(
                                        "{{ cat.id }}",
                                        classes="text-caption font-weight-bold",
                                    )
                                    with html.Td():
                                        v3.VChip(
                                            "{{ cat.type }}",
                                            size="x-small",
                                            color="purple-darken-2",
                                        )
                                    html.Td(
                                        "{{ cat.net_euler_index }}",
                                        classes="text-center text-caption font-mono",
                                    )
                                    with html.Td(classes="text-right"):
                                        v3.VChip(
                                            "{{ cat.is_fused ? 'Fused' : 'Split' }}",
                                            size="x-small",
                                            color="success"
                                            if ("cat.is_fused", True)
                                            else "amber-darken-3",
                                        )

                # Step 6: Effective CP Reduction & Harmonic Nudge Assessment Card
                with v3.VCard(elevation=2, classes="mb-3", color="surface-variant"):
                    with v3.VCardItem():
                        with v3.VCardTitle(
                            classes="text-subtitle-2 font-weight-bold d-flex align-center justify-space-between"
                        ):
                            html.Span("Step 6: Effective CPs & Harmonic Nudge")
                            v3.VChip(
                                "{{ reduction_stats.n_effective_cps }} ECPs",
                                size="x-small",
                                color="teal",
                            )

                    v3.VDivider()
                    with v3.VCardText(classes="pt-2 pb-2"):
                        v3.VSwitch(
                            label="Reduce Catastrophes to Effective Saddles",
                            v_model=("reduce_catastrophes",),
                            density="compact",
                            color="purple",
                            hide_details=True,
                            classes="mb-1",
                        )
                        v3.VSwitch(
                            label="Reduce Multi-CP Clusters to Effective Extrema",
                            v_model=("reduce_multi_clusters",),
                            density="compact",
                            color="warning",
                            hide_details=True,
                            classes="mb-1",
                        )
                        v3.VSwitch(
                            label="Harmonic Nudge CP Positions",
                            v_model=("nudge_cp_positions",),
                            density="compact",
                            color="teal",
                            hide_details=True,
                            classes="mb-1",
                        )
                        v3.VSwitch(
                            label="Perfect Boundary Port Angles (360/k)",
                            v_model=("perfect_port_angles",),
                            density="compact",
                            color="info",
                            hide_details=True,
                            classes="mb-2",
                        )

                        with html.Div(
                            classes="d-flex justify-space-between align-center mt-1"
                        ):
                            html.Div(
                                "Max Nudge Displacement",
                                classes="text-caption font-weight-bold text-medium-emphasis",
                            )
                            html.Div(
                                "{{ Number(nudge_max_pitch).toFixed(1) }}x δθ",
                                classes="text-caption font-weight-bold text-teal",
                            )

                        v3.VSlider(
                            min=0.25,
                            max=4.0,
                            step=0.25,
                            v_model=("nudge_max_pitch",),
                            density="compact",
                            thumb_label=False,
                            color="teal",
                            classes="mt-1",
                        )

                        with html.Div(classes="mt-2 pa-2 rounded surface"):
                            with v3.VRow(
                                dense=True, classes="text-caption text-center"
                            ):
                                with v3.VCol(cols=4):
                                    html.Div(
                                        "Cats Reduced",
                                        classes="text-medium-emphasis",
                                    )
                                    html.Div(
                                        "{{ reduction_stats.n_catastrophes_reduced }}",
                                        classes="text-subtitle-2 font-weight-bold text-purple",
                                    )
                                with v3.VCol(cols=4):
                                    html.Div(
                                        "Clusters Reduced",
                                        classes="text-medium-emphasis",
                                    )
                                    html.Div(
                                        "{{ reduction_stats.n_clusters_reduced }}",
                                        classes="text-subtitle-2 font-weight-bold text-warning",
                                    )
                                with v3.VCol(cols=4):
                                    html.Div(
                                        "CPs Hidden",
                                        classes="text-medium-emphasis",
                                    )
                                    html.Div(
                                        "{{ reduction_stats.n_raw_cps_hidden }}",
                                        classes="text-subtitle-2 font-weight-bold",
                                    )

                        with html.Div(
                            v_if="reduction_stats.n_clusters_bridged > 0",
                            classes="text-caption font-mono mt-1 text-cyan",
                        ):
                            html.Div(
                                "Corridor fragments bridged: {{ reduction_stats.n_clusters_bridged }}"
                            )
                        with html.Div(
                            v_if="reduction_stats.n_clusters_annihilated > 0",
                            classes="text-caption font-mono mt-1 text-grey",
                        ):
                            html.Div(
                                "Annulled (χ=0, ring kept): {{ reduction_stats.n_clusters_annihilated }}"
                            )
                        with html.Div(
                            v_if="reduction_stats.n_clusters_irreducible > 0",
                            classes="text-caption font-mono mt-1 text-deep-orange",
                        ):
                            html.Div(
                                "Irreducible (χ≥+2, kept expanded): {{ reduction_stats.n_clusters_irreducible }}"
                            )
                        with html.Div(
                            v_if="nudge_cp_positions",
                            classes="text-caption font-mono mt-1 text-teal",
                        ):
                            html.Div(
                                "Nudged {{ reduction_stats.n_nudged }} | mean {{ reduction_stats.mean_nudge_deg.toFixed(2) }}° | max {{ reduction_stats.max_nudge_deg.toFixed(2) }}°"
                            )
                        with html.Div(
                            v_if="perfect_port_angles",
                            classes="text-caption font-mono mt-1 text-info",
                        ):
                            html.Div(
                                "Ports perfected: {{ reduction_stats.n_ports_perfected }}"
                            )

                        # Effective critical point table
                        with html.Div(
                            v_if="reduction_stats.n_effective_cps > 0", classes="mt-2"
                        ):
                            html.Div(
                                "Effective Critical Points (click to inspect)",
                                classes="text-caption font-weight-bold text-medium-emphasis mb-1",
                            )
                            with v3.VTable(
                                density="compact",
                                classes="elevation-0",
                                style="max-height: 150px; overflow-y: auto;",
                            ):
                                with html.Thead():
                                    with html.Tr():
                                        html.Th(
                                            "ID",
                                            classes="text-left text-caption font-weight-bold",
                                        )
                                        html.Th(
                                            "χ",
                                            classes="text-center text-caption font-weight-bold",
                                        )
                                        html.Th(
                                            "Nudge",
                                            classes="text-right text-caption font-weight-bold",
                                        )
                                with html.Tbody():
                                    with html.Tr(
                                        v_for="ecp in all_cps_list.filter(c => c.is_effective)",
                                        key="ecp.id_label",
                                        click=(
                                            ctrl.select_cp_from_list,
                                            "[ecp.id_label]",
                                        ),
                                        classes="cursor-pointer",
                                    ):
                                        html.Td(
                                            "{{ ecp.id_label }}",
                                            classes="text-caption font-weight-bold",
                                        )
                                        html.Td(
                                            "{{ ecp.net_index }}",
                                            classes="text-center text-caption font-mono",
                                        )
                                        html.Td(
                                            "{{ ecp.nudge ? ecp.nudge.displacement_ang_deg.toFixed(2) + '°' : '-' }}",
                                            classes="text-right text-caption font-mono",
                                        )

                # Step 7: Basin Center Selection Card
                with v3.VCard(elevation=2, classes="mb-3", color="surface-variant"):
                    with v3.VCardItem():
                        with v3.VCardTitle(
                            classes="text-subtitle-2 font-weight-bold d-flex align-center justify-space-between"
                        ):
                            with html.Div(classes="d-flex align-center"):
                                v3.VIcon(
                                    "mdi-bullseye-arrow", classes="mr-2", color="amber"
                                )
                                html.Span("Basins: Step 1 Centers")
                            v3.VChip(
                                "{{ basin_centers_list.length }} Extrema",
                                size="x-small",
                                color="amber-darken-3",
                            )

                    v3.VDivider()
                    with v3.VCardText(classes="pt-2 pb-2"):
                        html.Div(
                            "Candidate Basin Centers (Click to highlight & inspect)",
                            classes="text-caption font-weight-bold text-medium-emphasis mb-1",
                        )
                        with v3.VTable(
                            density="compact",
                            classes="elevation-0",
                            style="max-height: 180px; overflow-y: auto;",
                        ):
                            with html.Thead():
                                with html.Tr():
                                    html.Th(
                                        "ID",
                                        classes="text-left text-caption font-weight-bold",
                                    )
                                    html.Th(
                                        "Type",
                                        classes="text-center text-caption font-weight-bold",
                                    )
                                    html.Th(
                                        "Value",
                                        classes="text-right text-caption font-weight-bold",
                                    )
                            with html.Tbody():
                                with html.Tr(
                                    v_for="bc in basin_centers_list",
                                    key="bc.id",
                                    click=(
                                        ctrl.select_basin_center_from_list,
                                        "[bc.id]",
                                    ),
                                    classes="cursor-pointer",
                                ):
                                    with html.Td(
                                        classes="text-caption font-weight-bold"
                                    ):
                                        html.Span("{{ bc.id }}")
                                        v3.VChip(
                                            "eff",
                                            size="x-small",
                                            color="teal",
                                            classes="ml-1",
                                            v_if="bc.is_effective",
                                        )
                                    with html.Td(classes="text-center"):
                                        v3.VChip(
                                            "{{ bc.polarity === 'maximum' ? 'MAX' : 'MIN' }}",
                                            size="x-small",
                                            color=("bc.badge_color",),
                                        )
                                    html.Td(
                                        "{{ bc.value.toFixed(5) }}",
                                        classes="text-right text-caption font-mono",
                                    )

                # Inspector Card: Selected Basin Center
                with v3.VCard(
                    v_if="selected_basin_center",
                    elevation=3,
                    classes="mb-3 border-warning",
                    color="surface",
                ):
                    with v3.VCardItem():
                        with v3.VCardTitle(
                            classes="text-subtitle-1 font-weight-bold d-flex align-center justify-space-between"
                        ):
                            with html.Div(classes="d-flex align-center"):
                                v3.VIcon(
                                    "mdi-bullseye-arrow",
                                    classes="mr-2",
                                    color="amber",
                                )
                                html.Span(
                                    "Basin Center: {{ selected_basin_center.id }}"
                                )
                            v3.VBtn(
                                icon="mdi-close",
                                variant="text",
                                density="compact",
                                click=ctrl.clear_basin_center_selection,
                            )
                        v3.VCardSubtitle(
                            "{{ selected_basin_center.is_effective ? 'Effective Extremum from ' + selected_basin_center.origin_id + ' (' + selected_basin_center.member_count + ' CPs)' : 'Raw Discrete Morse Extremum (Vertex ' + selected_basin_center.vertex_id + ')' }}"
                        )

                    v3.VDivider()
                    with v3.VCardText(classes="pt-2"):
                        with v3.VRow(dense=True):
                            with v3.VCol(cols=6):
                                html.Div(
                                    "Polarity",
                                    classes="text-caption text-medium-emphasis",
                                )
                                v3.VChip(
                                    "{{ selected_basin_center.polarity.toUpperCase() }}",
                                    size="x-small",
                                    color=("selected_basin_center.badge_color",),
                                )
                            with v3.VCol(cols=6):
                                html.Div(
                                    "Scalar Value",
                                    classes="text-caption text-medium-emphasis",
                                )
                                html.Div(
                                    "{{ selected_basin_center.value.toFixed(6) }}",
                                    classes="text-body-2 font-weight-bold",
                                )

                        with v3.VRow(dense=True, classes="mt-1"):
                            with v3.VCol(cols=12):
                                html.Div(
                                    "Coordinates (X, Y, Z)",
                                    classes="text-caption text-medium-emphasis",
                                )
                                html.Div(
                                    "({{ selected_basin_center.position[0].toFixed(4) }}, {{ selected_basin_center.position[1].toFixed(4) }}, {{ selected_basin_center.position[2].toFixed(4) }})",
                                    classes="text-body-2 font-mono",
                                )

                # Basin Locator Step 2: Saddle Launch Seeds Card
                with v3.VCard(elevation=2, classes="mb-3", color="surface-variant"):
                    with v3.VCardItem():
                        with v3.VCardTitle(
                            classes="text-subtitle-2 font-weight-bold d-flex align-center justify-space-between"
                        ):
                            with html.Div(classes="d-flex align-center"):
                                v3.VIcon(
                                    "mdi-arrow-decision", classes="mr-2", color="orange"
                                )
                                html.Span("Basins: Step 2 Launch Seeds")
                            v3.VChip(
                                "{{ launch_stats.n_saddles_seeded }} saddles",
                                size="x-small",
                                color="orange-darken-2",
                            )

                    v3.VDivider()
                    with v3.VCardText(classes="pt-2 pb-2"):
                        v3.VSwitch(
                            label="Show Launch Seeds (Orange=Ascent, Cyan=Descent)",
                            v_model=("show_launch_seeds",),
                            density="compact",
                            color="orange",
                            hide_details=True,
                            classes="mb-1",
                        )

                        with html.Div(v_if="show_launch_seeds"):
                            with html.Div(
                                classes="d-flex justify-space-between align-center mt-1"
                            ):
                                html.Div(
                                    "Launch Offset",
                                    classes="text-caption font-weight-bold text-medium-emphasis",
                                )
                                html.Div(
                                    "{{ Number(launch_seed_pitch).toFixed(2) }}x δθ",
                                    classes="text-caption font-weight-bold text-orange",
                                )

                            v3.VSlider(
                                min=0.1,
                                max=1.0,
                                step=0.05,
                                v_model=("launch_seed_pitch",),
                                density="compact",
                                thumb_label=False,
                                color="orange",
                                classes="mt-1",
                            )

                            with html.Div(classes="mt-2 pa-2 rounded surface"):
                                with v3.VRow(
                                    dense=True, classes="text-caption text-center"
                                ):
                                    with v3.VCol(cols=4):
                                        html.Div(
                                            "Ascent", classes="text-medium-emphasis"
                                        )
                                        html.Div(
                                            "{{ launch_stats.n_ascent }}",
                                            classes="text-subtitle-2 font-weight-bold text-orange",
                                        )
                                    with v3.VCol(cols=4):
                                        html.Div(
                                            "Descent", classes="text-medium-emphasis"
                                        )
                                        html.Div(
                                            "{{ launch_stats.n_descent }}",
                                            classes="text-subtitle-2 font-weight-bold text-info",
                                        )
                                    with v3.VCol(cols=4):
                                        html.Div(
                                            "Max k", classes="text-medium-emphasis"
                                        )
                                        html.Div(
                                            "{{ launch_stats.max_fold_order }}",
                                            classes="text-subtitle-2 font-weight-bold",
                                        )

                            with html.Div(
                                v_if="launch_stats.n_index_mismatch > 0",
                                classes="text-caption font-mono mt-1 text-deep-orange",
                            ):
                                html.Div(
                                    "Index mismatch (k != mu+1): {{ launch_stats.n_index_mismatch }}"
                                )
                            with html.Div(
                                classes="text-caption text-medium-emphasis mt-1 font-italic",
                            ):
                                html.Div(
                                    "k=2 simple saddle, k=3 monkey (3-basin junction), k=4 octupolar."
                                )

                # Inspector Card: Selected Catastrophe
                with v3.VCard(
                    v_if="selected_catastrophe",
                    elevation=3,
                    classes="mb-3 border-purple",
                    color="surface",
                ):
                    with v3.VCardItem():
                        with v3.VCardTitle(
                            classes="text-subtitle-1 font-weight-bold d-flex align-center justify-space-between"
                        ):
                            with html.Div(classes="d-flex align-center"):
                                v3.VIcon(
                                    "mdi-atom-variant", classes="mr-2", color="purple"
                                )
                                html.Span(
                                    "{{ selected_catastrophe.id }} ({{ selected_catastrophe.type }})"
                                )
                            v3.VBtn(
                                icon="mdi-close",
                                variant="text",
                                density="compact",
                                click=ctrl.clear_catastrophe_selection,
                            )
                        v3.VCardSubtitle(
                            "Net χ = {{ selected_catastrophe.net_euler_index }} | {{ selected_catastrophe.classification === 'spurious_unfolding' ? 'Spurious Noise (Fused)' : 'Physical Symmetry Breaking (Split)' }}"
                        )

                    v3.VDivider()
                    with v3.VCardText(classes="pt-2"):
                        with v3.VRow(dense=True):
                            with v3.VCol(cols=6):
                                html.Div(
                                    "Fold Order / Ports",
                                    classes="text-caption text-medium-emphasis",
                                )
                                html.Div(
                                    "k = {{ selected_catastrophe.fold_order }} ({{ selected_catastrophe.num_valleys }}V + {{ selected_catastrophe.num_ridges }}R)",
                                    classes="text-body-2 font-weight-bold",
                                )
                            with v3.VCol(cols=6):
                                html.Div(
                                    "Bifurcation Score",
                                    classes="text-caption text-medium-emphasis",
                                )
                                html.Div(
                                    "{{ (selected_catastrophe.bifurcation_score * 100).toFixed(1) }}%",
                                    classes="text-body-2 font-weight-bold text-purple",
                                )

                        with v3.VRow(dense=True, classes="mt-1"):
                            with v3.VCol(cols=6):
                                html.Div(
                                    "Barrier Depth",
                                    classes="text-caption text-medium-emphasis",
                                )
                                html.Div(
                                    "{{ selected_catastrophe.barrier_depth.toFixed(5) }} ({{ (selected_catastrophe.relative_barrier_depth * 100).toFixed(2) }}%)",
                                    classes="text-caption font-mono font-weight-bold",
                                )
                            with v3.VCol(cols=6):
                                html.Div(
                                    "Angular Span",
                                    classes="text-caption text-medium-emphasis",
                                )
                                html.Div(
                                    "{{ selected_catastrophe.angular_span_deg.toFixed(1) }}°",
                                    classes="text-caption font-weight-bold",
                                )

                        with v3.VRow(dense=True, classes="mt-1"):
                            with v3.VCol(cols=12):
                                html.Div(
                                    "Constituent Clusters",
                                    classes="text-caption text-medium-emphasis",
                                )
                                html.Div(
                                    "{{ selected_catastrophe.constituent_cluster_ids.join(', ') }} ({{ selected_catastrophe.all_member_count }} total CPs)",
                                    classes="text-caption font-mono",
                                )

                # Inspector Card: Selected Critical Point
                with v3.VCard(
                    v_if="selected_cp",
                    elevation=3,
                    classes="mb-3 border-primary",
                    color="surface",
                ):
                    with v3.VCardItem():
                        with v3.VCardTitle(
                            classes="text-subtitle-1 font-weight-bold d-flex align-center justify-space-between"
                        ):
                            with html.Div(classes="d-flex align-center"):
                                v3.VIcon(
                                    "mdi-crosshairs-gps",
                                    classes="mr-2",
                                    color="amber-darken-2",
                                )
                                html.Span("{{ selected_cp.id_label }}")
                            v3.VBtn(
                                icon="mdi-close",
                                variant="text",
                                density="compact",
                                click=ctrl.clear_cp_selection,
                            )
                        v3.VCardSubtitle(
                            "Vertex Index: {{ selected_cp.vertex_id }}",
                            v_if="!selected_cp.is_effective",
                        )
                        v3.VCardSubtitle(
                            "Reduced from {{ selected_cp.origin_label }} {{ selected_cp.origin_id }} ({{ selected_cp.member_count }} CPs) | Net χ = {{ selected_cp.net_index }}",
                            v_if="!!selected_cp.is_effective",
                        )

                    v3.VDivider()
                    with v3.VCardText(classes="pt-2"):
                        with v3.VRow(dense=True):
                            with v3.VCol(cols=6):
                                html.Div(
                                    "Type", classes="text-caption text-medium-emphasis"
                                )
                                v3.VChip(
                                    "{{ selected_cp.type_title }}",
                                    size="x-small",
                                    color=("selected_cp.badge_color",),
                                )
                            with v3.VCol(cols=6):
                                html.Div(
                                    "Scalar Value",
                                    classes="text-caption text-medium-emphasis",
                                )
                                html.Div(
                                    "{{ selected_cp.value.toFixed(6) }}",
                                    classes="text-body-2 font-weight-bold",
                                )

                        with v3.VRow(dense=True, classes="mt-1"):
                            with v3.VCol(cols=12):
                                html.Div(
                                    "Coordinates (X, Y, Z) — molecular frame",
                                    classes="text-caption text-medium-emphasis",
                                )
                                html.Div(
                                    "({{ selected_cp.position[0].toFixed(4) }}, {{ selected_cp.position[1].toFixed(4) }}, {{ selected_cp.position[2].toFixed(4) }})",
                                    classes="text-body-2 font-mono",
                                )

                        with html.Div(
                            v_if="!!selected_cp.nudge",
                            classes="mt-2 pa-2 rounded surface",
                        ):
                            html.Div(
                                "Harmonic Nudge (k = {{ selected_cp.fold_order }})",
                                classes="text-caption font-weight-bold text-teal",
                            )
                            html.Div(
                                "Displacement: {{ selected_cp.nudge.displacement_ang_deg.toFixed(3) }}° in {{ selected_cp.nudge.iterations }} iters",
                                classes="text-caption font-mono",
                            )
                            html.Div(
                                "Symmetry energy: {{ selected_cp.nudge.initial_energy.toFixed(6) }} → {{ selected_cp.nudge.final_energy.toFixed(6) }} (Δ = {{ selected_cp.nudge.energy_drop.toFixed(6) }})",
                                classes="text-caption font-mono",
                            )
                            html.Div(
                                "Pre-nudge center: ({{ selected_cp.nudge.initial_position[0].toFixed(4) }}, {{ selected_cp.nudge.initial_position[1].toFixed(4) }}, {{ selected_cp.nudge.initial_position[2].toFixed(4) }})",
                                classes="text-caption font-mono text-medium-emphasis",
                            )

                        with html.Div(v_if="!!selected_cp.members", classes="mt-2"):
                            html.Div(
                                "Constituent Critical Points",
                                classes="text-caption font-weight-bold text-medium-emphasis mb-1",
                            )
                            with html.Div(
                                v_for="mem in selected_cp.members",
                                key="mem.id_label",
                                classes="text-caption font-mono d-flex justify-space-between",
                            ):
                                html.Span("{{ mem.id_label }}")
                                html.Span("{{ mem.value.toFixed(6) }}")

        # Toolbar
        with layout.toolbar:
            v3.VSpacer()
            v3.VBtn(
                "Save Parameters",
                prepend_icon="mdi-content-save",
                click=ctrl.save_parameters,
                variant="tonal",
                density="compact",
                color="success",
                tooltip=(
                    "Write every editable control, the run context, the view "
                    "camera and the derived metrics to a JSON file next to "
                    "the .plt for headless reproduction"
                ),
            )
            v3.VBtn(
                "Load Parameters",
                prepend_icon="mdi-content-load",
                click=ctrl.open_load_dialog,
                variant="tonal",
                density="compact",
                color="info",
                tooltip=(
                    "Restore a saved parameter snapshot: the file, atom, "
                    "field, controls and camera it describes"
                ),
            )
            v3.VBtn(
                "Reset Defaults",
                prepend_icon="mdi-restore",
                click=ctrl.reset_defaults,
                variant="tonal",
                density="compact",
                color="warning",
                tooltip="Restore all analysis sliders, switches and reduction toggles",
            )
            v3.VBtn(
                "Reset Camera",
                prepend_icon="mdi-camera-flip-outline",
                click=ctrl.reset_camera,
                variant="tonal",
                density="compact",
                color="primary",
            )

        # 3D Viewport with direct clicking
        with layout.content:
            with html.Div(
                style="position: relative; width: 100%; height: 100%; cursor: pointer;",
                click=(
                    ctrl.on_scene_click,
                    "[$event.offsetX, $event.offsetY, $event.currentTarget.clientWidth, $event.currentTarget.clientHeight]",
                ),
            ):
                view = VtkRemoteView(render_window, interactive_ratio=1.0)
                ctrl.view_update = view.update
                ctrl.view_reset_camera = view.reset_camera

        # Load Parameters dialog. The file is picked in the browser and
        # uploaded to the server, so a snapshot can be restored without the
        # server knowing where it lives on disk.
        with v3.VDialog(v_model=("load_dialog_open", False), max_width="640"):
            with v3.VCard():
                with v3.VCardTitle(classes="text-subtitle-1 font-weight-bold"):
                    html.Span("Load Parameters")
                v3.VDivider()
                with v3.VCardText(classes="pt-3"):
                    html.Div(
                        "Choose a parameter snapshot (.json) written by "
                        '"Save Parameters". The file, atom, field, controls '
                        "and camera it describes are restored.",
                        classes="text-caption text-medium-emphasis mb-3",
                    )
                    v3.VFileInput(
                        label="Parameter file",
                        v_model=("load_dialog_file", None),
                        accept=".json",
                        prepend_icon="mdi-file-code-outline",
                        density="compact",
                        variant="outlined",
                        show_size=True,
                        truncate_length=40,
                        hide_details=True,
                        __properties=["accept"],
                    )
                    with html.Div(v_if="params_load_error", classes="mt-3"):
                        v3.VAlert(
                            "{{ params_load_error }}",
                            type="error",
                            variant="tonal",
                            density="compact",
                        )
                with v3.VCardActions():
                    v3.VSpacer()
                    v3.VBtn(
                        "Cancel",
                        variant="text",
                        density="compact",
                        click=ctrl.close_load_dialog,
                    )
                    v3.VBtn(
                        "Load",
                        variant="tonal",
                        density="compact",
                        color="info",
                        click=ctrl.confirm_load_parameters,
                    )

    # Optional startup restore. The snapshot may name a different file, atom
    # or field than the command line asked for; the snapshot wins, because it
    # is the more specific description of the state to reproduce.
    if params_path:
        try:
            load_params_from_file(params_path)
            state.params_last_loaded_path = os.path.abspath(params_path)
        except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
            state.params_load_error = str(exc)
            print(f"[sphere_viewer] Parameter load failed: {exc}")

    print(
        f"\n[gba_topology2] Starting Sphere Viewer for {session.atom_info['atom_name']} "
        f"from {session.plt_path}"
    )
    server.start(port=port, open_browser=open_browser)


def main() -> None:
    """CLI entrypoint for sphere_viewer."""
    parser = argparse.ArgumentParser(
        prog="sphere_viewer.py",
        description="gba_topology2 Step 2: Isolated Atom Sphere & Micro-Cluster Interactive Viewer",
    )
    parser.add_argument(
        "plt_file",
        nargs="?",
        default="Pd_20K.plt",
        help="Input Tecplot .plt file (default: Pd_20K.plt)",
    )
    parser.add_argument(
        "-a",
        "--atom",
        type=int,
        default=1,
        help="Target atom number to isolate (default: 1 for Pd1)",
    )
    parser.add_argument(
        "-p", "--port", type=int, default=None, help="Trame web server port"
    )
    parser.add_argument(
        "--params",
        default=None,
        help=(
            "Path to a parameter snapshot (.json) to restore on startup. "
            "The snapshot may name its own .plt, atom and field, which then "
            "take precedence over the positional arguments."
        ),
    )
    parser.add_argument(
        "--server", action="store_true", help="Run without auto-opening browser"
    )
    args, unknown = parser.parse_known_args()

    sys.argv = [sys.argv[0], args.plt_file] + unknown
    run_sphere_viewer(
        args.plt_file,
        atom_num=args.atom,
        port=args.port,
        open_browser=not args.server,
        params_path=args.params,
    )


if __name__ == "__main__":
    main()
