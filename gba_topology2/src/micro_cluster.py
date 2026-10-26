"""Micro-Cluster Identification Subroutine.

Groups raw critical points into micro-clusters based on angular separation
scaled by mesh pitch:
    theta_ij <= k * delta_theta_mesh

Computes for each cluster:
- Centroid projected onto the sphere of radius R
- Member critical point IDs and types
- Angular span and spatial diameter
- Initial boundary ring vertices on the sphere
- Local Euler index: chi_micro = N_max + N_min - sum(saddle_multiplicities)
"""

import math
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from gba_topology2.src.morse_detector import CriticalPoint

__all__ = [
    "BoundaryRingPoint",
    "MicroCluster",
    "compute_micro_clusters",
    "compute_tangent_pca",
    "generate_boundary_ring",
    "generate_elliptical_boundary_ring",
    "ring_frame",
]


@dataclass(frozen=True)
class BoundaryRingPoint:
    """A point along the circular or elliptical boundary ring on the sphere."""

    phi_rad: float
    position: tuple[float, float, float]


@dataclass(frozen=True)
class MicroCluster:
    """Represents a micro-cluster of one or more critical points."""

    cluster_id: str
    members: list[CriticalPoint]
    centroid: tuple[float, float, float]
    centroid_unit: tuple[float, float, float]
    angular_radius_rad: float
    angular_radius_deg: float
    angular_diameter_rad: float
    angular_diameter_deg: float
    spatial_diameter: float
    n_maxima: int
    n_minima: int
    n_saddles: int
    total_saddle_index: int = 0
    local_euler_index: int = 0
    is_multi_cp: bool = False
    boundary_ring: list[BoundaryRingPoint] = field(default_factory=list)
    # Morphology and anisotropy metrics
    morphology: str = "compact"  # "compact" (Type 1) or "string" (Type 2)
    aspect_ratio: float = 1.0  # lambda_1 / lambda_2 on tangent plane (>= 1.0)
    principal_axis_unit: tuple[float, float, float] = (
        0.0,
        0.0,
        0.0,
    )  # Major axis tangent vector
    semi_major_rad: float = 0.0  # Semi-major angular radius (radians)
    semi_minor_rad: float = 0.0  # Semi-minor angular radius (radians)

    @property
    def composition_summary(self) -> str:
        """Human-readable composition string, e.g. '2 Max, 1 Sad' or '4 Max, 1 Sad (mult 3)'."""
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


def generate_boundary_ring(
    centroid: tuple[float, float, float],
    sphere_radius: float,
    angular_radius_rad: float,
    num_samples: int = 48,
) -> list[BoundaryRingPoint]:
    """Generate sample coordinates for a circular boundary ring on the sphere.

    Args:
        centroid: (x, y, z) coordinates of cluster centroid.
        sphere_radius: Radius R of the sphere.
        angular_radius_rad: Angular radius of the boundary ring in radians.
        num_samples: Number of circumferential sample points.

    Returns:
        List of BoundaryRingPoint objects forming the closed loop.
    """
    c_arr = np.array(centroid, dtype=np.float64)
    c_norm = float(np.linalg.norm(c_arr))
    if c_norm < 1e-12:
        c_unit = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    else:
        c_unit = c_arr / c_norm

    # Find tangent orthonormal vectors t1, t2
    if abs(float(c_unit[2])) < 0.9:
        v_temp = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    else:
        v_temp = np.array([1.0, 0.0, 0.0], dtype=np.float64)

    t1 = np.cross(c_unit, v_temp)
    norm_t1 = float(np.linalg.norm(t1))
    if norm_t1 < 1e-12:
        v_temp = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        t1 = np.cross(c_unit, v_temp)
        norm_t1 = float(np.linalg.norm(t1))
    t1 /= norm_t1
    t2 = np.cross(c_unit, t1)
    norm_t2 = float(np.linalg.norm(t2))
    if norm_t2 > 1e-12:
        t2 /= norm_t2

    cos_a = math.cos(angular_radius_rad)
    sin_a = math.sin(angular_radius_rad)

    ring: list[BoundaryRingPoint] = []
    for i in range(num_samples):
        phi = 2.0 * math.pi * float(i) / float(num_samples)
        dir_vec = cos_a * c_unit + sin_a * (math.cos(phi) * t1 + math.sin(phi) * t2)
        dir_vec /= np.linalg.norm(dir_vec)
        p = dir_vec * sphere_radius
        ring.append(
            BoundaryRingPoint(
                phi_rad=phi,
                position=(float(p[0]), float(p[1]), float(p[2])),
            )
        )
    return ring


def generate_elliptical_boundary_ring(
    centroid: tuple[float, float, float],
    sphere_radius: float,
    semi_major_rad: float,
    semi_minor_rad: float,
    principal_axis_unit: tuple[float, float, float],
    num_samples: int = 48,
) -> list[BoundaryRingPoint]:
    """Generate sample coordinates for an oriented elliptical boundary ring on the sphere.

    Args:
        centroid: (x, y, z) coordinates of cluster centroid on sphere.
        sphere_radius: Radius R of the sphere.
        semi_major_rad: Semi-major angular radius in radians (along principal axis).
        semi_minor_rad: Semi-minor angular radius in radians (transverse).
        principal_axis_unit: (3,) unit tangent vector along principal major axis.
        num_samples: Number of circumferential sample points.

    Returns:
        List of BoundaryRingPoint objects forming the closed oriented loop.
    """
    c_arr = np.array(centroid, dtype=np.float64)
    c_norm = float(np.linalg.norm(c_arr))
    if c_norm < 1e-12:
        c_unit = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    else:
        c_unit = c_arr / c_norm

    # Tangent vector 1 (major axis)
    t1 = np.array(principal_axis_unit, dtype=np.float64)
    # Project to tangent plane to ensure exact orthogonality to c_unit
    t1 = t1 - float(np.dot(t1, c_unit)) * c_unit
    norm_t1 = float(np.linalg.norm(t1))
    if norm_t1 < 1e-6:
        # Fallback to arbitrary tangent vector if degenerate
        if abs(float(c_unit[2])) < 0.9:
            v_temp = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        else:
            v_temp = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        t1 = np.cross(c_unit, v_temp)
        norm_t1 = float(np.linalg.norm(t1))
    t1 /= norm_t1

    # Tangent vector 2 (minor axis, orthonormal)
    t2 = np.cross(c_unit, t1)
    norm_t2 = float(np.linalg.norm(t2))
    if norm_t2 > 1e-12:
        t2 /= norm_t2

    ring: list[BoundaryRingPoint] = []
    for i in range(num_samples):
        phi = 2.0 * math.pi * float(i) / float(num_samples)
        # Angular offset in tangent plane: u = a * cos(phi) * t1 + b * sin(phi) * t2
        theta_phi = math.sqrt(
            (semi_major_rad * math.cos(phi)) ** 2
            + (semi_minor_rad * math.sin(phi)) ** 2
        )
        if theta_phi > 1e-12:
            tan_dir = (
                (semi_major_rad * math.cos(phi)) * t1
                + (semi_minor_rad * math.sin(phi)) * t2
            ) / theta_phi
        else:
            tan_dir = t1

        cos_t = math.cos(theta_phi)
        sin_t = math.sin(theta_phi)
        dir_vec = cos_t * c_unit + sin_t * tan_dir
        dir_vec /= np.linalg.norm(dir_vec)
        p = dir_vec * sphere_radius
        ring.append(
            BoundaryRingPoint(
                phi_rad=phi,
                position=(float(p[0]), float(p[1]), float(p[2])),
            )
        )
    return ring


def ring_frame(
    ring_points: list[BoundaryRingPoint],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Recover the spherical frame of a boundary ring from its samples.

    The analytic inverse of :func:`generate_boundary_ring`: given the samples,
    return ``(c_unit, t1, t2, angular_radius_rad)`` such that a point on the loop
    at azimuth ``phi`` is

        p(phi) = R * [cos(a) * c + sin(a) * (cos(phi) * t1 + sin(phi) * t2)]

    Exact for circular rings. For an elliptical ring, whose angular radius varies
    with ``phi``, it recovers the mean radius and the principal frame, which is
    the best circular approximation of the loop.

    Args:
        ring_points: Ordered ring samples carrying ``phi_rad`` and ``position``.

    Returns:
        Tuple of (c_unit, t1, t2, angular_radius_rad).

    Raises:
        ValueError: If fewer than four ring samples are supplied.
    """
    if len(ring_points) < 4:
        raise ValueError("Need at least 4 ring points to recover a ring frame")

    pts = np.array([rp.position for rp in ring_points], dtype=np.float64)
    pts_unit = pts / np.linalg.norm(pts, axis=1, keepdims=True)

    c_unit = pts_unit.sum(axis=0)
    n_c = float(np.linalg.norm(c_unit))
    c_unit = c_unit / n_c if n_c > 1e-12 else np.array([0.0, 0.0, 1.0])

    # Angular radius: mean angle of ring samples from the ring center
    cos_a = float(np.clip(np.mean(pts_unit @ c_unit), -1.0, 1.0))
    ang_radius = math.acos(cos_a)

    phis = np.array([rp.phi_rad for rp in ring_points], dtype=np.float64)

    def _tangent_at(target_phi: float) -> np.ndarray:
        idx = int(np.argmin(np.abs(np.angle(np.exp(1j * (phis - target_phi))))))
        vec = pts_unit[idx] - cos_a * c_unit
        nrm = float(np.linalg.norm(vec))
        return vec / nrm if nrm > 1e-12 else np.zeros(3)

    t1 = _tangent_at(0.0)
    if float(np.linalg.norm(t1)) < 1e-12:
        v_tmp = (
            np.array([0.0, 0.0, 1.0])
            if abs(float(c_unit[2])) < 0.9
            else np.array([1.0, 0.0, 0.0])
        )
        t1 = np.cross(c_unit, v_tmp)
        t1 /= np.linalg.norm(t1)

    t2 = np.cross(c_unit, t1)
    t2 /= np.linalg.norm(t2)
    # Enforce the ring's own orientation (counter-clockwise about c_unit)
    if float(np.dot(_tangent_at(math.pi / 2.0), t2)) < 0.0:
        t2 = -t2

    return c_unit, t1, t2, ang_radius


def compute_tangent_pca(
    comp_units: npt.NDArray[np.float64],
    centroid_unit: npt.NDArray[np.float64],
) -> tuple[float, npt.NDArray[np.float64], float, float]:
    """Compute 2D tangent plane Principal Component Analysis for member points.

    Projects unit vectors onto the 2D tangent plane at centroid_unit using an
    orthonormal basis (t1, t2), computes the covariance tensor, and extracts
    eigenvalues and principal axis.

    Args:
        comp_units: (N, 3) unit direction vectors of member CPs.
        centroid_unit: (3,) unit normal vector of cluster centroid.

    Returns:
        Tuple of (aspect_ratio, principal_tangent_3d, max_major_rad, max_minor_rad)
        where aspect_ratio >= 1.0.
    """
    n = len(comp_units)
    if n < 2:
        # Single point: isotropic aspect ratio 1.0
        # Create arbitrary tangent vector
        if abs(float(centroid_unit[2])) < 0.9:
            v_temp = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        else:
            v_temp = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        t1 = np.cross(centroid_unit, v_temp)
        t1 /= np.linalg.norm(t1)
        return 1.0, t1, 0.0, 0.0

    # Build orthonormal tangent basis t1, t2
    if abs(float(centroid_unit[2])) < 0.9:
        v_temp = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    else:
        v_temp = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    t1 = np.cross(centroid_unit, v_temp)
    t1 /= np.linalg.norm(t1)
    t2 = np.cross(centroid_unit, t1)
    t2 /= np.linalg.norm(t2)

    # Project member unit vectors onto tangent plane using logarithmic map / central projection:
    # displacement x = cos_c * centroid_unit + sin_c * (u_1 * t1 + u_2 * t2)
    # (u_1, u_2) coords on tangent plane in angular radians:
    coords_2d = np.zeros((n, 2), dtype=np.float64)
    for i in range(n):
        cos_c = float(np.clip(np.dot(centroid_unit, comp_units[i]), -1.0, 1.0))
        ang_c = math.acos(cos_c)
        if ang_c > 1e-12:
            tan_proj = comp_units[i] - cos_c * centroid_unit
            norm_tan = float(np.linalg.norm(tan_proj))
            if norm_tan > 1e-12:
                tan_dir = tan_proj / norm_tan
                coords_2d[i, 0] = ang_c * float(np.dot(tan_dir, t1))
                coords_2d[i, 1] = ang_c * float(np.dot(tan_dir, t2))

    # Center coordinates on tangent plane (mean should be close to 0)
    mean_2d = np.mean(coords_2d, axis=0)
    centered = coords_2d - mean_2d

    # Covariance tensor (2, 2)
    cov = np.dot(centered.T, centered) / float(n)
    eigvals, eigvecs = np.linalg.eigh(cov)

    # Sort descending: lambda_1 >= lambda_2
    idx = np.argsort(eigvals)[::-1]
    val1 = max(float(eigvals[idx[0]]), 1e-16)
    val2 = max(float(eigvals[idx[1]]), 1e-16)
    v1_2d = eigvecs[:, idx[0]]

    # 3D tangent vector along principal axis
    princ_axis_3d = v1_2d[0] * t1 + v1_2d[1] * t2
    norm_pa = float(np.linalg.norm(princ_axis_3d))
    if norm_pa > 1e-12:
        princ_axis_3d /= norm_pa

    aspect_ratio = math.sqrt(val1 / val2)

    # Project centered coords onto 2D eigenvectors to find max extents along major & minor axes
    v2_2d = eigvecs[:, idx[1]]
    proj_major = np.abs(np.dot(centered, v1_2d))
    proj_minor = np.abs(np.dot(centered, v2_2d))
    max_major_rad = float(np.max(proj_major))
    max_minor_rad = float(np.max(proj_minor))

    return aspect_ratio, princ_axis_3d, max_major_rad, max_minor_rad


def compute_micro_clusters(
    cps: list[CriticalPoint],
    delta_theta_mesh_rad: float,
    k_pitch: float = 2.0,
    sphere_radius: float = 1.0,
    min_boundary_margin_deg: float = 1.5,
    num_boundary_points: int = 48,
    aspect_ratio_threshold: float = 2.0,
) -> list[MicroCluster]:
    """Cluster critical points within distance threshold d <= k * delta_theta_mesh.

    Args:
        cps: List of input CriticalPoint objects.
        delta_theta_mesh_rad: Mesh pitch delta_theta in radians.
        k_pitch: Multiplier for mesh pitch to define clustering radius (default 2.0).
        sphere_radius: Sphere radius R.
        min_boundary_margin_deg: Minimum angular boundary margin (default 1.5 deg).
        num_boundary_points: Sample points for boundary ring (default 48).
        aspect_ratio_threshold: Aspect ratio threshold (lambda_1/lambda_2 >= thresh)
            to classify multi-CP cluster as "string" vs "compact" (default 2.0).

    Returns:
        List of MicroCluster objects sorted by member count descending.

    Raises:
        ValueError: If delta_theta_mesh_rad <= 0, k_pitch <= 0, or aspect_ratio_threshold < 1.0.
    """
    if delta_theta_mesh_rad <= 0:
        raise ValueError(
            f"delta_theta_mesh_rad must be positive, got {delta_theta_mesh_rad}"
        )
    if k_pitch <= 0:
        raise ValueError(f"k_pitch must be positive, got {k_pitch}")
    if aspect_ratio_threshold < 1.0:
        raise ValueError(
            f"aspect_ratio_threshold must be >= 1.0, got {aspect_ratio_threshold}"
        )
    if not cps:
        return []

    n = len(cps)
    threshold_rad = k_pitch * delta_theta_mesh_rad

    positions: npt.NDArray[np.float64] = np.array(
        [cp.position for cp in cps], dtype=np.float64
    )
    norms = np.linalg.norm(positions, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    unit_positions = positions / norms

    # Build adjacency based on angular distance <= threshold_rad
    adj: dict[int, list[int]] = {i: [] for i in range(n)}
    for i in range(n):
        for j in range(i + 1, n):
            cos_ij = float(
                np.clip(np.dot(unit_positions[i], unit_positions[j]), -1.0, 1.0)
            )
            ang_ij = math.acos(cos_ij)
            if ang_ij <= threshold_rad:
                adj[i].append(j)
                adj[j].append(i)

    # Connected components search
    visited: set[int] = set()
    components: list[list[int]] = []
    for i in range(n):
        if i not in visited:
            comp: list[int] = []
            queue = [i]
            visited.add(i)
            while queue:
                curr = queue.pop(0)
                comp.append(curr)
                for nbr in adj[curr]:
                    if nbr not in visited:
                        visited.add(nbr)
                        queue.append(nbr)
            components.append(comp)

    # Sort components: multi-element first, then largest first, then by earliest index
    components.sort(key=lambda comp: (-len(comp), min(comp)))

    min_margin_rad = math.radians(min_boundary_margin_deg)
    clusters: list[MicroCluster] = []

    for cl_idx, comp in enumerate(components):
        members = [cps[i] for i in comp]
        comp_positions = positions[comp]
        comp_units = unit_positions[comp]

        # Centroid projected onto sphere
        mean_unit = np.mean(comp_units, axis=0)
        norm_mean = float(np.linalg.norm(mean_unit))
        if norm_mean > 1e-8:
            centroid_unit = mean_unit / norm_mean
            centroid = centroid_unit * sphere_radius
        else:
            centroid_unit = comp_units[0]
            centroid = comp_positions[0]

        # Compute maximum pairwise angular distance and chord distance
        max_ang = 0.0
        max_dist = 0.0
        for i_idx in range(len(comp)):
            for j_idx in range(i_idx + 1, len(comp)):
                d = float(np.linalg.norm(comp_positions[i_idx] - comp_positions[j_idx]))
                cos_v = float(
                    np.clip(
                        np.dot(comp_units[i_idx], comp_units[j_idx]),
                        -1.0,
                        1.0,
                    )
                )
                ang = math.acos(cos_v)
                max_dist = max(max_dist, d)
                max_ang = max(max_ang, ang)

        # Angular radius from centroid to furthest member
        max_rad_from_c = 0.0
        for i_idx in range(len(comp)):
            cos_c = float(np.clip(np.dot(centroid_unit, comp_units[i_idx]), -1.0, 1.0))
            ang_c = math.acos(cos_c)
            max_rad_from_c = max(max_rad_from_c, ang_c)

        # Ring angular radius includes boundary margin
        ring_radius_rad = max(max_rad_from_c + min_margin_rad, threshold_rad / 2.0)
        ring_radius_deg = float(math.degrees(ring_radius_rad))

        n_max = sum(1 for m in members if m.cp_type == "maximum")
        n_min = sum(1 for m in members if m.cp_type == "minimum")
        n_sad = sum(1 for m in members if m.cp_type == "saddle")
        total_saddle_index = sum(
            m.multiplicity for m in members if m.cp_type == "saddle"
        )
        local_chi = n_max + n_min - total_saddle_index

        # Tangent PCA for morphology & oriented boundary fitting
        (
            aspect_ratio,
            princ_axis_3d,
            max_major_rad,
            max_minor_rad,
        ) = compute_tangent_pca(comp_units, centroid_unit)

        is_multi = len(members) > 1
        if is_multi and aspect_ratio >= aspect_ratio_threshold:
            morphology = "string"
            semi_major = max(max_major_rad + min_margin_rad, threshold_rad / 2.0)
            semi_minor = max(max_minor_rad + min_margin_rad, threshold_rad / 2.0)
            b_ring = generate_elliptical_boundary_ring(
                centroid=(float(centroid[0]), float(centroid[1]), float(centroid[2])),
                sphere_radius=sphere_radius,
                semi_major_rad=semi_major,
                semi_minor_rad=semi_minor,
                principal_axis_unit=(
                    float(princ_axis_3d[0]),
                    float(princ_axis_3d[1]),
                    float(princ_axis_3d[2]),
                ),
                num_samples=num_boundary_points,
            )
        else:
            morphology = "compact"
            semi_major = ring_radius_rad
            semi_minor = ring_radius_rad
            b_ring = generate_boundary_ring(
                centroid=(float(centroid[0]), float(centroid[1]), float(centroid[2])),
                sphere_radius=sphere_radius,
                angular_radius_rad=ring_radius_rad,
                num_samples=num_boundary_points,
            )

        clusters.append(
            MicroCluster(
                cluster_id=f"MC{cl_idx + 1}",
                members=members,
                centroid=(float(centroid[0]), float(centroid[1]), float(centroid[2])),
                centroid_unit=(
                    float(centroid_unit[0]),
                    float(centroid_unit[1]),
                    float(centroid_unit[2]),
                ),
                angular_radius_rad=ring_radius_rad,
                angular_radius_deg=ring_radius_deg,
                angular_diameter_rad=max_ang,
                angular_diameter_deg=float(math.degrees(max_ang)),
                spatial_diameter=max_dist,
                n_maxima=n_max,
                n_minima=n_min,
                n_saddles=n_sad,
                total_saddle_index=total_saddle_index,
                local_euler_index=local_chi,
                is_multi_cp=is_multi,
                boundary_ring=b_ring,
                morphology=morphology,
                aspect_ratio=aspect_ratio,
                principal_axis_unit=(
                    float(princ_axis_3d[0]),
                    float(princ_axis_3d[1]),
                    float(princ_axis_3d[2]),
                ),
                semi_major_rad=semi_major,
                semi_minor_rad=semi_minor,
            )
        )

    return clusters
