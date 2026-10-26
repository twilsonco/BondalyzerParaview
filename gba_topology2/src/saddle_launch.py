"""Saddle Launch Vector Subroutine for gba_topology2 (Basin Locator Step 2).

Given a saddle critical point, determine the directions in which the separatrix
(streamline) branches leave it, and produce the on-sphere seed positions from
which Step 3 integrates them.

Why not the Hessian?
--------------------
The obvious approach -- diagonalise the surface Hessian and take its
eigenvectors -- works only for a simple 2-fold saddle. A monkey saddle
(fold order k = 3) has an identically vanishing quadratic term at its centre
(the normal form is f = x^3 - 3xy^2), so the Hessian is degenerate and its
eigenvectors are arbitrary. The same failure hits every higher-order saddle
this project exists to resolve, and the effective saddles produced by the
catastrophe stage are exactly those cases.

The method used here
--------------------
Work with the cyclic link, which is discrete Morse theory's own description of
a saddle. Order the neighbours of v cyclically around v on the sphere and read
off the sign of f(u) - f(v) around the cycle. The sign alternates in exactly
2k sectors: k upper sectors (f > f(v), ridges, ascending / unstable manifold)
and k lower sectors (f < f(v), valleys, descending / stable manifold),
interleaved. The crossings between them are the separatrix directions, and the
bisector of each sector is the branch launch direction. This is uniform in k:
k = 2 gives the familiar 2 + 2 cross, k = 3 the 3 + 3 monkey, k = 4 the 4 + 4
octupolar cross, with no case analysis and no degenerate linear algebra.

The sector count is also a measurement, not an assumption: it is compared
against the multiplicity the Morse detector already reported
(multiplicity mu = k - 1) and the discrepancy is reported, so a saddle whose
observed fold order disagrees with its index is visible in the viewer rather
than silently traced with the wrong number of branches.

Frame contract: all inputs and outputs are in the CENTRED atom frame, where the
nucleus is the origin and the outward normal of a surface point p is p / ||p||.
The viewer maps the launch positions to the world frame with
``sphere_frame.to_global`` before drawing.
"""

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

from gba_topology2.src.morse_detector import CriticalPoint

__all__ = [
    "LaunchBranch",
    "SaddleLaunchSeed",
    "build_cyclic_link",
    "compute_all_saddle_launch_seeds",
    "extract_saddle_launch_seed",
]


@dataclass(frozen=True)
class LaunchBranch:
    """One separatrix branch leaving a saddle.

    Attributes:
        kind: ``"ascent"`` (ridge, unstable manifold, towards maxima) or
            ``"descent"`` (valley, stable manifold, towards minima).
        sector_index: Index of this branch among branches of the same kind,
            ordered counter-clockwise about the saddle.
        tangent_direction: Unit tangent vector at the saddle pointing along the
            branch (perpendicular to the radial normal, by construction).
        launch_position: Point on the sphere, offset from the saddle centre
            along ``tangent_direction``, from which Step 3 starts integrating.
        sector_width_rad: Angular width of the sector this branch bisects.
        sector_centroid_value: Mean field value of the link nodes inside the
            sector, used as a prominence measure.
    """

    kind: str
    sector_index: int
    tangent_direction: tuple[float, float, float]
    launch_position: tuple[float, float, float]
    sector_width_rad: float
    sector_centroid_value: float


@dataclass(frozen=True)
class SaddleLaunchSeed:
    """All launch branches extracted from one saddle.

    Attributes:
        saddle_id: Identifier of the source saddle (e.g. ``"SAD_142"``).
        saddle_position: Saddle centre in the centred atom frame.
        saddle_value: Field value at the saddle.
        vertex_id: Mesh vertex index of the saddle.
        multiplicity: Multiplicity reported by the Morse detector
            (mu = k - 1 for a saddle of fold order k).
        fold_order: Fold order k observed from the link sector count.
        index_agrees: True when the observed fold order matches the reported
            multiplicity, i.e. k == mu + 1.
        ascent_branches: Ridge branches (towards maxima), counter-clockwise.
        descent_branches: Valley branches (towards minima), counter-clockwise.
        separatrix_directions: The 2k crossing directions where f equals the
            saddle value, ordered counter-clockwise. These are the separatrix
            tangents themselves, i.e. the boundary between basins.
    """

    saddle_id: str
    saddle_position: tuple[float, float, float]
    saddle_value: float
    vertex_id: int
    multiplicity: int
    fold_order: int
    index_agrees: bool
    ascent_branches: list[LaunchBranch] = field(default_factory=list)
    descent_branches: list[LaunchBranch] = field(default_factory=list)
    separatrix_directions: list[tuple[float, float, float]] = field(
        default_factory=list
    )
    # Excision substrate only: the ring radius the branches were read from, and
    # why the ring produced nothing. Both are zero/None for the cyclic-link path.
    ring_radius_rad: float = 0.0
    failure_reason: str | None = None

    @property
    def all_branches(self) -> list[LaunchBranch]:
        """Ascent branches followed by descent branches."""
        return [*self.ascent_branches, *self.descent_branches]


def build_cyclic_link(
    vertex_id: int,
    triangles: npt.NDArray[np.integer[Any]],
) -> list[int]:
    """Order the neighbours of a vertex cyclically around it.

    Builds the link graph from the triangles incident to ``vertex_id`` and walks
    it into a cycle. For an interior vertex of a closed triangulated surface the
    link is a single cycle, so the walk visits every neighbour exactly once.

    Args:
        vertex_id: Index of the centre vertex.
        triangles: (M, 3) triangle vertex indices.

    Returns:
        Neighbour vertex indices in cyclic order around ``vertex_id``. Empty if
        the vertex is not referenced by any triangle.

    Raises:
        ValueError: If the link is not a single simple cycle (a non-manifold
            vertex), which would make a cyclic sector scan meaningless.
    """
    tris = np.asarray(triangles)

    link_edges: set[tuple[int, int]] = set()
    for tri in tris:
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
        if vertex_id not in (a, b, c):
            continue
        others = [x for x in (a, b, c) if x != vertex_id]
        if len(others) == 2:
            link_edges.add((others[0], others[1]))
            link_edges.add((others[1], others[0]))

    if not link_edges:
        return []

    adjacency: dict[int, set[int]] = {}
    for u, w in link_edges:
        adjacency.setdefault(u, set()).add(w)

    for node, nbrs in adjacency.items():
        if len(nbrs) != 2:
            raise ValueError(
                f"Link of vertex {vertex_id} is not a cycle: neighbour {node} "
                f"has {len(nbrs)} link neighbours (expected 2)."
            )

    start = min(adjacency)
    order = [start]
    previous = -1
    current = start
    while True:
        nxt_set = adjacency[current] - {previous}
        if not nxt_set:
            break
        nxt = next(iter(nxt_set))
        if nxt == start:
            break
        order.append(nxt)
        previous, current = current, nxt
        if len(order) > len(adjacency):
            raise ValueError(
                f"Link walk around vertex {vertex_id} did not close into a cycle."
            )

    if len(order) != len(adjacency):
        raise ValueError(
            f"Link of vertex {vertex_id} is disconnected: walked {len(order)} of "
            f"{len(adjacency)} link nodes."
        )
    return order


def _unit(value: npt.NDArray[np.float64]) -> npt.NDArray[np.float64] | None:
    """Normalise ``value``, returning None for a (near) zero vector."""
    norm = float(np.linalg.norm(value))
    if norm <= 1e-15:
        return None
    return value / norm


def _tangent_project(
    vector: npt.NDArray[np.float64], normal: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Remove the component of ``vector`` along the unit ``normal``."""
    return vector - float(np.dot(vector, normal)) * normal


def _rotate_towards(
    origin_unit: npt.NDArray[np.float64],
    direction_unit: npt.NDArray[np.float64],
    angle_rad: float,
    radius: float,
) -> tuple[float, float, float]:
    """Move on the sphere from ``origin_unit`` along ``direction_unit``.

    Uses the great-circle (geodesic) exponential map, so the result lies exactly
    on the sphere of radius ``radius`` rather than being projected back onto it
    after a linear step.

    Args:
        origin_unit: Unit radial vector of the start point.
        direction_unit: Unit tangent vector giving the travel direction.
        angle_rad: Angular step in radians.
        radius: Sphere radius.

    Returns:
        The new position as a float tuple, at distance ``radius`` from the
        origin.
    """
    if angle_rad <= 0.0:
        return (
            float(origin_unit[0] * radius),
            float(origin_unit[1] * radius),
            float(origin_unit[2] * radius),
        )
    moved = math.cos(angle_rad) * origin_unit + math.sin(angle_rad) * direction_unit
    moved_unit = _unit(moved)
    if moved_unit is None:
        moved_unit = origin_unit
    return (
        float(moved_unit[0] * radius),
        float(moved_unit[1] * radius),
        float(moved_unit[2] * radius),
    )


def _sector_width(
    a: npt.NDArray[np.float64],
    b: npt.NDArray[np.float64],
    radial: npt.NDArray[np.float64],
) -> float:
    """Angular width at the centre between two crossing directions on S^2."""
    ta = _unit(_tangent_project(a, radial))
    tb = _unit(_tangent_project(b, radial))
    if ta is None or tb is None:
        return 0.0
    dot = float(np.clip(np.dot(ta, tb), -1.0, 1.0))
    return float(math.acos(dot))


def extract_saddle_launch_seed(
    saddle: CriticalPoint,
    mesh_pts: npt.NDArray[np.float64],
    triangles: npt.NDArray[np.integer[Any]],
    f_vals: npt.NDArray[np.float64],
    sphere_radius: float,
    delta_theta_mesh_rad: float,
    launch_step_pitch: float = 0.35,
) -> SaddleLaunchSeed:
    """Extract the separatrix launch branches of one saddle.

    Args:
        saddle: The saddle critical point, in the centred atom frame.
        mesh_pts: (N, 3) vertex coordinates in the centred atom frame.
        triangles: (M, 3) triangle vertex indices.
        f_vals: (N,) scalar field values.
        sphere_radius: Atom sphere radius R.
        delta_theta_mesh_rad: Intrinsic angular mesh pitch.
        launch_step_pitch: Launch offset in units of the mesh pitch. Must be
            positive and small enough to stay inside the saddle's own sector;
            the default 0.35 puts the seed just off the degenerate centre.

    Returns:
        A SaddleLaunchSeed with the ascent and descent branches. If the link
        scan finds fewer than four sign crossings the vertex is not a traceable
        saddle in this field, and the branch lists come back empty.

    Raises:
        ValueError: If ``launch_step_pitch`` is not positive, the saddle vertex
            sits at the sphere centre, or the link is not a simple cycle.
    """
    if launch_step_pitch <= 0.0:
        raise ValueError(f"launch_step_pitch must be > 0, got {launch_step_pitch}")

    pts = np.asarray(mesh_pts, dtype=np.float64)
    values = np.asarray(f_vals, dtype=np.float64)
    v = int(saddle.vertex_id)
    p_v = pts[v]
    f_v = float(values[v])

    radial = _unit(p_v)
    if radial is None:
        raise ValueError(f"Saddle vertex {v} sits at the sphere centre; R degenerate.")

    link = build_cyclic_link(v, triangles)
    n_link = len(link)
    if n_link < 3:
        return SaddleLaunchSeed(
            saddle_id=f"SAD_{v}",
            saddle_position=(float(p_v[0]), float(p_v[1]), float(p_v[2])),
            saddle_value=f_v,
            vertex_id=v,
            multiplicity=int(saddle.multiplicity),
            fold_order=0,
            index_agrees=False,
        )

    link_pos = pts[link]
    link_f = values[link]

    # Sign of f - f(v) around the cyclic link. A node exactly equal to f(v) is
    # given a positive sign so the scan cannot stall on an exact tie.
    delta = link_f - f_v
    signs = np.where(np.abs(delta) < 1e-15, 1.0, np.sign(delta))

    # Crossings between consecutive link nodes, positioned by linear
    # interpolation of the field along the link chord.
    crossings: list[npt.NDArray[np.float64]] = []
    crossing_edges: list[tuple[int, int]] = []
    for i in range(n_link):
        j = (i + 1) % n_link
        if signs[i] == signs[j]:
            continue
        di, dj = float(delta[i]), float(delta[j])
        denom = dj - di
        t = (-di / denom) if abs(denom) > 1e-300 else 0.5
        t = min(max(t, 0.0), 1.0)
        chord = link_pos[i] + t * (link_pos[j] - link_pos[i])
        chord_unit = _unit(chord)
        if chord_unit is None:
            continue
        crossings.append(chord_unit)
        crossing_edges.append((i, j))

    n_cross = len(crossings)
    if n_cross < 4:
        return SaddleLaunchSeed(
            saddle_id=f"SAD_{v}",
            saddle_position=(float(p_v[0]), float(p_v[1]), float(p_v[2])),
            saddle_value=f_v,
            vertex_id=v,
            multiplicity=int(saddle.multiplicity),
            fold_order=max(0, n_cross // 2),
            index_agrees=False,
        )

    separatrix_dirs: list[tuple[float, float, float]] = []
    for chord_unit in crossings:
        tangent = _unit(_tangent_project(chord_unit, radial))
        if tangent is not None:
            separatrix_dirs.append(
                (float(tangent[0]), float(tangent[1]), float(tangent[2]))
            )

    # Sectors: the link nodes strictly between consecutive crossings.
    # Sector c is bounded by crossing c (on edge (i_c, j_c)) and crossing c+1
    # (on edge (i_{c+1}, j_{c+1})). It contains link nodes from j_c to i_{c+1}.
    ascent: list[LaunchBranch] = []
    descent: list[LaunchBranch] = []
    step_angle = float(launch_step_pitch) * float(delta_theta_mesh_rad)

    for c in range(n_cross):
        j_c = crossing_edges[c][1]
        i_next = crossing_edges[(c + 1) % n_cross][0]
        members: list[int] = []
        cursor = j_c
        for _ in range(n_link + 1):
            members.append(link[cursor])
            if cursor == i_next:
                break
            cursor = (cursor + 1) % n_link
        if not members:
            continue

        sector_sign = int(signs[j_c])
        acc = np.zeros(3, dtype=np.float64)
        for node in members:
            u = _unit(pts[node])
            acc += pts[node] if u is None else u
        acc = acc + crossings[c] + crossings[(c + 1) % n_cross]

        bisector = _unit(_tangent_project(acc, radial))
        if bisector is None:
            continue

        width = _sector_width(crossings[c], crossings[(c + 1) % n_cross], radial)
        centroid_value = float(np.mean([float(values[m]) for m in members]))
        launch = _rotate_towards(radial, bisector, step_angle, sphere_radius)

        branch = LaunchBranch(
            kind="ascent" if sector_sign > 0 else "descent",
            sector_index=0,
            tangent_direction=(
                float(bisector[0]),
                float(bisector[1]),
                float(bisector[2]),
            ),
            launch_position=launch,
            sector_width_rad=width,
            sector_centroid_value=centroid_value,
        )
        (ascent if sector_sign > 0 else descent).append(branch)

    ascent = [
        LaunchBranch(
            kind=b.kind,
            sector_index=i,
            tangent_direction=b.tangent_direction,
            launch_position=b.launch_position,
            sector_width_rad=b.sector_width_rad,
            sector_centroid_value=b.sector_centroid_value,
        )
        for i, b in enumerate(ascent)
    ]
    descent = [
        LaunchBranch(
            kind=b.kind,
            sector_index=i,
            tangent_direction=b.tangent_direction,
            launch_position=b.launch_position,
            sector_width_rad=b.sector_width_rad,
            sector_centroid_value=b.sector_centroid_value,
        )
        for i, b in enumerate(descent)
    ]

    fold_order = len(ascent)
    index_agrees = fold_order == int(saddle.multiplicity) + 1

    return SaddleLaunchSeed(
        saddle_id=f"SAD_{v}",
        saddle_position=(float(p_v[0]), float(p_v[1]), float(p_v[2])),
        saddle_value=f_v,
        vertex_id=v,
        multiplicity=int(saddle.multiplicity),
        fold_order=fold_order,
        index_agrees=index_agrees,
        ascent_branches=ascent,
        descent_branches=descent,
        separatrix_directions=separatrix_dirs,
    )


def compute_all_saddle_launch_seeds(
    saddles: list[CriticalPoint],
    mesh_pts: npt.NDArray[np.float64],
    triangles: npt.NDArray[np.integer[Any]],
    f_vals: npt.NDArray[np.float64],
    sphere_radius: float,
    delta_theta_mesh_rad: float,
    launch_step_pitch: float = 0.35,
) -> list[SaddleLaunchSeed]:
    """Extract launch seeds for every saddle.

    Args:
        saddles: Saddle critical points in the centred atom frame.
        mesh_pts: (N, 3) vertex coordinates in the centred atom frame.
        triangles: (M, 3) triangle vertex indices.
        f_vals: (N,) scalar field values.
        sphere_radius: Atom sphere radius R.
        delta_theta_mesh_rad: Intrinsic angular mesh pitch.
        launch_step_pitch: Launch offset in mesh-pitch units.

    Returns:
        One SaddleLaunchSeed per saddle, in the input order. Saddles whose link
        is non-manifold are skipped with a warning rather than aborting the run,
        so one bad vertex cannot take the viewer down.
    """
    seeds: list[SaddleLaunchSeed] = []
    for s in saddles:
        try:
            seeds.append(
                extract_saddle_launch_seed(
                    saddle=s,
                    mesh_pts=mesh_pts,
                    triangles=triangles,
                    f_vals=f_vals,
                    sphere_radius=sphere_radius,
                    delta_theta_mesh_rad=delta_theta_mesh_rad,
                    launch_step_pitch=launch_step_pitch,
                )
            )
        except ValueError as exc:
            print(f"[saddle_launch] skipped saddle {s.vertex_id}: {exc}")
    return seeds
