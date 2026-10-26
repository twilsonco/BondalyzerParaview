"""Unit tests for local (spatially-indexed) IDW interpolation and the locator adapter.

The load-bearing case is parity: ``interpolate_scalar_at_points_local`` is a
performance rewrite of ``interpolate_scalar_at_points`` for the excision-ring
launch path, so every value it produces must match the reference kernel to
floating-point noise. The remaining cases pin the behaviours a spatial index can
get wrong -- the exact-vertex snap, the k-nearest equivalence argument that
justifies a radius-free query, and the clamping when k exceeds the mesh size.
"""

import math
import time
from collections.abc import Sequence

import numpy as np
import pytest
from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.vtkCommonDataModel import vtkPolyData

from gba_topology2.src.excision_boundary import (
    KNearestFinder,
    interpolate_scalar_at_points,
    interpolate_scalar_at_points_local,
)
from gba_topology2.src.micro_cluster import generate_boundary_ring
from gba_topology2.src.point_locator import (
    KdTreeNearestFinder,
    make_kdtree_nearest_finder,
)


def _fibonacci_sphere(num_points: int = 1500, radius: float = 1.0) -> np.ndarray:
    """Generate uniform points on a sphere."""
    indices = np.arange(0, num_points, dtype=float) + 0.5
    phi = np.arccos(1 - 2 * indices / num_points)
    theta = math.pi * (1 + 5**0.5) * indices
    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    z = radius * np.cos(phi)
    return np.column_stack([x, y, z])


def _brute_force_k_nearest(mesh_pts: np.ndarray) -> KNearestFinder:
    """Return an exact chord-distance k-nearest finder over ``mesh_pts``."""
    m = np.asarray(mesh_pts, dtype=np.float64)

    def find(point: Sequence[float], k: int) -> list[int]:
        d = np.linalg.norm(m - np.asarray(point, dtype=np.float64), axis=1)
        order = np.argsort(d, kind="stable")[: int(k)]
        return [int(i) for i in order]

    return find


def _poly_from_points(pts: np.ndarray) -> vtkPolyData:
    """Wrap an (N, 3) array as a point-only ``vtkPolyData``."""
    vp = vtkPoints()
    for p in pts:
        vp.InsertNextPoint(float(p[0]), float(p[1]), float(p[2]))
    poly = vtkPolyData()
    poly.SetPoints(vp)
    return poly


def test_local_matches_global_on_ring_samples() -> None:
    """Ring samples over a quadrupole field interpolate identically to the global scan."""
    pts = _fibonacci_sphere(num_points=1500, radius=1.0)
    f_vals = pts[:, 0] ** 2 - pts[:, 1] ** 2

    ring = generate_boundary_ring(
        centroid=(0.0, 0.0, 1.0),
        sphere_radius=1.0,
        angular_radius_rad=math.radians(18.0),
        num_samples=72,
    )
    q = np.array([rp.position for rp in ring], dtype=np.float64)

    reference = interpolate_scalar_at_points(q, pts, f_vals)
    local = interpolate_scalar_at_points_local(
        q,
        pts,
        f_vals,
        find_k_nearest=_brute_force_k_nearest(pts),
    )

    assert np.allclose(local, reference, atol=1e-12)


def test_local_matches_global_with_kdtree_finder() -> None:
    """The production VTK locator reproduces the reference kernel too."""
    pts = _fibonacci_sphere(num_points=1200, radius=1.0)
    rng = np.random.default_rng(7)
    f_vals = np.exp(-((pts[:, 2] - 0.3) ** 2)) * np.cos(
        3.0 * np.arctan2(pts[:, 1], pts[:, 0])
    )
    f_vals = f_vals + rng.normal(0.0, 1e-3, size=len(pts))

    ring = generate_boundary_ring(
        centroid=(0.2, -0.1, 0.97),
        sphere_radius=1.0,
        angular_radius_rad=math.radians(12.0),
        num_samples=48,
    )
    q = np.array([rp.position for rp in ring], dtype=np.float64)

    reference = interpolate_scalar_at_points(q, pts, f_vals, k_neighbors=4, power=2.0)
    local = interpolate_scalar_at_points_local(
        q,
        pts,
        f_vals,
        find_k_nearest=make_kdtree_nearest_finder(_poly_from_points(pts)),
        k_neighbors=4,
        power=2.0,
    )

    assert np.allclose(local, reference, atol=1e-12)


def test_chord_k_nearest_equals_angular_k_nearest_on_sphere() -> None:
    """The radius-free query is justified: on a sphere, chord order == angular order.

    ``interpolate_scalar_at_points_local`` asks the index for k nearest by chord
    distance and then evaluates an angular kernel on the result. That is only the
    true k nearest by angle if the two orders agree, which holds exactly when all
    points share a radius.
    """
    pts = _fibonacci_sphere(num_points=2000, radius=1.0)
    u = pts / np.linalg.norm(pts, axis=1, keepdims=True)
    kd = make_kdtree_nearest_finder(_poly_from_points(pts))

    rng = np.random.default_rng(11)
    k = 4
    for _ in range(200):
        v = rng.normal(size=3)
        q = v / np.linalg.norm(v)
        by_chord = set(kd(q.tolist(), k))
        ang = np.arccos(np.clip(u @ q, -1.0, 1.0))
        by_angle = {int(i) for i in np.argsort(ang, kind="stable")[:k]}
        assert by_chord == by_angle


def test_local_preserves_exact_vertex_snap() -> None:
    """A query landing on a mesh vertex returns that vertex's value verbatim."""
    pts = np.array(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64
    )
    f_vals = np.array([10.0, 20.0, 30.0], dtype=np.float64)

    res = interpolate_scalar_at_points_local(
        pts,
        pts,
        f_vals,
        find_k_nearest=_brute_force_k_nearest(pts),
        k_neighbors=2,
    )

    assert np.allclose(res, f_vals, atol=1e-12)


def test_local_clamps_when_k_exceeds_mesh_size() -> None:
    """Asking for more neighbours than exist uses every vertex, without raising."""
    pts = _fibonacci_sphere(num_points=12, radius=1.0)
    f_vals = pts[:, 2]
    q = np.array([[0.6, 0.0, math.sqrt(1.0 - 0.36)]], dtype=np.float64)

    reference = interpolate_scalar_at_points(q, pts, f_vals, k_neighbors=50)
    local = interpolate_scalar_at_points_local(
        q,
        pts,
        f_vals,
        find_k_nearest=make_kdtree_nearest_finder(_poly_from_points(pts)),
        k_neighbors=50,
    )

    assert np.allclose(local, reference, atol=1e-12)
    assert np.isfinite(local).all()


def test_local_handles_query_at_origin() -> None:
    """A zero-norm query has no direction; the routine must not divide by zero."""
    pts = _fibonacci_sphere(num_points=200, radius=1.0)
    f_vals = pts[:, 2]

    res = interpolate_scalar_at_points_local(
        np.array([[0.0, 0.0, 0.0]]),
        pts,
        f_vals,
        find_k_nearest=_brute_force_k_nearest(pts),
    )

    assert np.isfinite(res).all()


def test_local_rejects_nonpositive_k() -> None:
    """k_neighbors <= 0 has no meaning for a k-nearest kernel."""
    pts = _fibonacci_sphere(num_points=50)
    for bad in (0, -3):
        with pytest.raises(ValueError, match="k_neighbors"):
            interpolate_scalar_at_points_local(
                pts[:2],
                pts,
                pts[:, 2],
                find_k_nearest=_brute_force_k_nearest(pts),
                k_neighbors=bad,
            )


def test_local_accepts_single_query_point() -> None:
    """A bare (3,) query is widened to one row rather than misread as three."""
    pts = _fibonacci_sphere(num_points=300, radius=1.0)
    f_vals = pts[:, 0] ** 2 - pts[:, 1] ** 2
    q = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    res = interpolate_scalar_at_points_local(
        q,
        pts,
        f_vals,
        find_k_nearest=_brute_force_k_nearest(pts),
    )

    assert res.shape == (1,)
    assert np.isfinite(res).all()


def test_kdtree_finder_matches_brute_force() -> None:
    """The VTK locator returns exactly the brute-force k-nearest set."""
    pts = _fibonacci_sphere(num_points=800, radius=1.0)
    kd = make_kdtree_nearest_finder(_poly_from_points(pts))
    bf = _brute_force_k_nearest(pts)

    rng = np.random.default_rng(3)
    for _ in range(25):
        v = rng.normal(size=3)
        q = (v / np.linalg.norm(v)).tolist()
        assert sorted(kd(q, 4)) == sorted(bf(q, 4))
        assert sorted(kd(q, 9)) == sorted(bf(q, 9))


def test_kdtree_finder_returns_closest_first() -> None:
    """The kernel relies on index 0 being the nearest vertex."""
    pts = _fibonacci_sphere(num_points=400, radius=1.0)
    kd = make_kdtree_nearest_finder(_poly_from_points(pts))

    rng = np.random.default_rng(5)
    for _ in range(20):
        v = rng.normal(size=3)
        q = (v / np.linalg.norm(v)).tolist()
        ids = kd(q, 5)
        d = np.linalg.norm(pts - np.asarray(q), axis=1)
        assert ids[0] == int(np.argmin(d))
        assert d[ids[0]] <= d[ids[-1]]


def test_kdtree_finder_reports_closest_point() -> None:
    """``closest`` returns the index of the nearest vertex."""
    pts = _fibonacci_sphere(num_points=300, radius=1.0)
    kd = KdTreeNearestFinder(_poly_from_points(pts))

    target = 42
    q = (pts[target] * 1.02).tolist()
    assert kd.closest(q) == target


def test_kdtree_finder_rejects_empty_dataset() -> None:
    """Building a tree over no points is a configuration error, not a silent empty index."""
    with pytest.raises(ValueError, match="empty dataset"):
        make_kdtree_nearest_finder(vtkPolyData())


def test_kdtree_finder_clamps_k_to_num_points() -> None:
    """Requesting more neighbours than the mesh has returns the whole mesh."""
    pts = _fibonacci_sphere(num_points=30, radius=1.0)
    kd = make_kdtree_nearest_finder(_poly_from_points(pts))

    assert len(kd([0.0, 0.0, 1.0], 1000)) == 30
    assert kd([0.0, 0.0, 1.0], 0) == []
    assert kd([0.0, 0.0, 1.0], -1) == []


def test_local_is_faster_than_global_on_many_rings() -> None:
    """The rewrite must actually pay for itself at launch-seed scale.

    Guards against a regression where the per-query Python loop ends up slower
    than the vectorised (M, N) scan it replaces.
    """
    pts = _fibonacci_sphere(num_points=6000, radius=1.0)
    f_vals = pts[:, 0] ** 2 - pts[:, 1] ** 2
    kd = make_kdtree_nearest_finder(_poly_from_points(pts))

    queries = []
    for i in range(40):
        ring = generate_boundary_ring(
            (0.0, 0.0, 1.0), 1.0, math.radians(6.0 + i * 0.3), num_samples=48
        )
        queries.append(np.array([rp.position for rp in ring], dtype=np.float64))
    all_q = np.vstack(queries)

    t0 = time.perf_counter()
    ref = interpolate_scalar_at_points(all_q, pts, f_vals)
    t_global = time.perf_counter() - t0

    t0 = time.perf_counter()
    local = interpolate_scalar_at_points_local(all_q, pts, f_vals, find_k_nearest=kd)
    t_local = time.perf_counter() - t0

    assert np.allclose(local, ref, atol=1e-12)
    assert t_local < t_global, f"local {t_local:.3f}s vs global {t_global:.3f}s"
