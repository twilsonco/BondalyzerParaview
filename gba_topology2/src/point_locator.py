"""Spatial point locator adapter for gba_topology2.

Wraps a VTK kd-tree locator in the ``KNearestFinder`` callable that
``excision_boundary.interpolate_scalar_at_points_local`` expects, so the IDW
kernel can query a spatial index instead of scanning all N mesh vertices per
query point.

The adapter lives here, and not in ``excision_boundary``, for two reasons:
``excision_boundary`` advertises a pure NumPy/stdlib contract and must stay
unit-testable without a VTK build, and the seam is the callable -- tests pass a
brute-force lambda, production passes the wrapper built here.

Asking the tree for exactly ``k`` nearest is sufficient for the IDW kernel
because both the mesh vertices and the excision ring samples lie on the atom
sphere: chord distance ``2 R sin(theta / 2)`` is strictly monotone in the
angular distance the kernel uses, so the k nearest by chord are the k nearest by
angle. No radius search, widening, or global fallback is needed.

Subroutine contract per AGENTS.md:
- Thin adapter only: no numerical policy lives in this module.
- Standalone testable via simple unit tests.
"""

from collections.abc import Sequence

from vtkmodules.vtkCommonCore import vtkIdList
from vtkmodules.vtkCommonDataModel import vtkKdTreePointLocator, vtkPolyData

from gba_topology2.src.excision_boundary import KNearestFinder

__all__ = ["KdTreeNearestFinder", "make_kdtree_nearest_finder"]


class KdTreeNearestFinder:
    """Callable returning the indices of the k mesh vertices nearest a point.

    The underlying ``vtkKdTreePointLocator`` is built once over a dataset and
    reused for every query, which is what makes profiling hundreds of excision
    rings affordable: the tree costs one build per mesh, not one per ring.

    Attributes:
        num_points: Number of indexed points.
    """

    __slots__ = ("_ids", "_locator", "num_points")

    def __init__(self, poly: vtkPolyData) -> None:
        """Build the kd-tree over ``poly``'s points.

        Args:
            poly: The dataset to index. Typically the atom's sphere mesh, in the
                centred frame shared with the mesh points given to the kernel.

        Raises:
            ValueError: If the dataset carries no points.
        """
        self.num_points = int(poly.GetNumberOfPoints())
        if self.num_points == 0:
            raise ValueError("Cannot build a point locator over an empty dataset")
        self._locator = vtkKdTreePointLocator()
        self._locator.SetDataSet(poly)
        self._locator.BuildLocator()
        self._ids = vtkIdList()

    def __call__(self, point: Sequence[float], k: int) -> list[int]:
        """Return the indices of the ``k`` points nearest ``point``.

        Args:
            point: (3,) query coordinates.
            k: Number of neighbours to return. Values above the number of indexed
                points are clamped, so the caller never needs to know the mesh
                size; the kernel then simply uses every available vertex.

        Returns:
            Nearest point indices, ordered from closest to furthest.
        """
        kk = min(int(k), self.num_points)
        if kk <= 0:
            return []
        self._ids.Reset()
        self._locator.FindClosestNPoints(
            kk, (float(point[0]), float(point[1]), float(point[2])), self._ids
        )
        return [int(self._ids.GetId(i)) for i in range(self._ids.GetNumberOfIds())]

    def closest(self, point: Sequence[float]) -> int:
        """Return the index of the nearest point to ``point``.

        Args:
            point: (3,) query coordinates.

        Returns:
            The nearest point index, or -1 when the locator finds none.
        """
        return int(
            self._locator.FindClosestPoint(
                (float(point[0]), float(point[1]), float(point[2]))
            )
        )


def make_kdtree_nearest_finder(poly: vtkPolyData) -> KNearestFinder:
    """Build a ``KNearestFinder`` backed by a kd-tree over ``poly``.

    Args:
        poly: The dataset to index, in the same frame as the mesh points passed
            to the IDW routine (the centred atom frame).

    Returns:
        A reusable callable suitable for
        ``excision_boundary.interpolate_scalar_at_points_local``.

    Raises:
        ValueError: If the dataset carries no points.
    """
    return KdTreeNearestFinder(poly)
