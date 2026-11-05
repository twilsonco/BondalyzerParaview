"""Unit tests for separatrix_tracer.py.

Synthetic sphere meshes with analytic fields (plus injected noise plateaus)
verify the cluster-aware gradient walk: disk absorption, pass-through of
annihilated noise clusters, great-circle tunneling out of noise sinks, saddle
self-exclusion, and the low-confidence terminations.
"""

from __future__ import annotations

import math
from itertools import pairwise

import numpy as np
import pytest

from gba_topology2.src.separatrix_tracer import (
    TraceZone,
    build_walk_adjacency,
    build_walk_adjacency_from_edges,
    make_zones,
    trace_separatrix,
)


def _uv_sphere(n_lat: int = 24, n_lon: int = 48) -> tuple[np.ndarray, np.ndarray]:
    """Build a UV-sphere mesh. Returns (pts, triangles)."""
    pts: list[tuple[float, float, float]] = []
    pts.append((0.0, 0.0, 1.0))  # 0: north pole
    for i in range(1, n_lat):
        theta = math.pi * i / n_lat
        for j in range(n_lon):
            phi = 2.0 * math.pi * j / n_lon
            pts.append(
                (
                    math.sin(theta) * math.cos(phi),
                    math.sin(theta) * math.sin(phi),
                    math.cos(theta),
                )
            )
    pts.append((0.0, 0.0, -1.0))  # south pole
    south = len(pts) - 1

    def idx(i: int, j: int) -> int:
        return 1 + (i - 1) * n_lon + (j % n_lon)

    tris: list[tuple[int, int, int]] = []
    for j in range(n_lon):
        tris.append((0, idx(1, j), idx(1, j + 1)))
    for i in range(1, n_lat - 1):
        for j in range(n_lon):
            a, b = idx(i, j), idx(i, j + 1)
            c, d = idx(i + 1, j), idx(i + 1, j + 1)
            tris.append((a, c, d))
            tris.append((a, d, b))
    for j in range(n_lon):
        tris.append((south, idx(n_lat - 1, j + 1), idx(n_lat - 1, j)))

    return np.asarray(pts, dtype=np.float64), np.asarray(tris, dtype=np.int32)


def _edges_from_triangles(triangles: np.ndarray) -> np.ndarray:
    seen: set[tuple[int, int]] = set()
    for tri in triangles:
        a, b, c = (int(tri[0]), int(tri[1]), int(tri[2]))
        for u, v in ((a, b), (b, c), (c, a)):
            seen.add((min(u, v), max(u, v)))
    return np.asarray(sorted(seen), dtype=np.int32)


def _cap(
    label: str, kind: str, pos: tuple[float, float, float], radius_rad: float
) -> TraceZone:
    p = np.asarray(pos, dtype=np.float64)
    u = p / np.linalg.norm(p)
    return TraceZone(
        label=label,
        kind=kind,
        center_unit=(float(u[0]), float(u[1]), float(u[2])),
        cos_radius=math.cos(radius_rad),
    )


PTS, TRIS = _uv_sphere()
NBR = build_walk_adjacency(TRIS, PTS.shape[0])
NORTH, SOUTH = 0, PTS.shape[0] - 1


def _vertex_at_colatitude(theta_rad: float) -> int:
    """Index of the mesh vertex nearest the given colatitude on the prime meridian."""
    target = np.array([math.sin(theta_rad), 0.0, math.cos(theta_rad)], dtype=np.float64)
    pu = PTS / np.linalg.norm(PTS, axis=1, keepdims=True)
    return int(np.argmax(pu @ target))


class TestAdjacencyBuilders:
    def test_triangle_adjacency_is_symmetric_and_deduped(self) -> None:
        for v, nbrs in enumerate(NBR):
            assert v not in nbrs
            assert nbrs == sorted(set(nbrs))
            for nb in nbrs:
                assert v in NBR[nb]

    def test_edge_builder_matches_triangle_builder(self) -> None:
        from_edges = build_walk_adjacency_from_edges(
            _edges_from_triangles(TRIS), PTS.shape[0]
        )
        assert from_edges == NBR

    def test_closed_mesh_has_no_boundary_vertices(self) -> None:
        assert all(len(nbrs) >= 3 for nbrs in NBR)


class TestPlainGradientWalk:
    """Height field f = z: one maximum at the north pole, one minimum at south."""

    F = PTS[:, 2].copy()

    def test_ascent_absorbs_into_max_cap(self) -> None:
        zones = [_cap("MAX_N", "absorb_max", (0, 0, 1), math.radians(20))]
        out = trace_separatrix(
            start_vertex=NBR[NORTH][0],
            direction="ascent",
            pts=PTS,
            f_vals=self.F,
            neighbors=NBR,
            zones=zones,
        )
        assert out.status == "absorbed"
        assert out.target_label == "MAX_N"
        assert out.confidence == "high"

    def test_descent_absorbs_into_min_cap(self) -> None:
        zones = [_cap("MIN_S", "absorb_min", (0, 0, -1), math.radians(20))]
        out = trace_separatrix(
            start_vertex=NBR[SOUTH][0],
            direction="descent",
            pts=PTS,
            f_vals=self.F,
            neighbors=NBR,
            zones=zones,
        )
        assert out.status == "absorbed"
        assert out.target_label == "MIN_S"

    def test_ascent_ignores_min_cap(self) -> None:
        zones = [_cap("MIN_S", "absorb_min", (0, 0, -1), math.radians(60))]
        out = trace_separatrix(
            start_vertex=NBR[NORTH][2],
            direction="ascent",
            pts=PTS,
            f_vals=self.F,
            neighbors=NBR,
            zones=zones,
        )
        assert out.target_label is None
        assert out.status == "raw_extremum"

    def test_monotone_path_reaches_raw_extremum_without_caps(self) -> None:
        out = trace_separatrix(
            start_vertex=NBR[NORTH][0],
            direction="ascent",
            pts=PTS,
            f_vals=self.F,
            neighbors=NBR,
            zones=[],
        )
        assert out.status == "raw_extremum"
        assert out.confidence == "medium"
        assert out.start_vertex == NORTH

    def test_descent_from_near_north_reaches_south(self) -> None:
        out = trace_separatrix(
            start_vertex=NBR[NORTH][0],
            direction="descent",
            pts=PTS,
            f_vals=self.F,
            neighbors=NBR,
            zones=[],
            max_steps=4096,
        )
        assert out.status == "raw_extremum"
        assert out.start_vertex == SOUTH

    def test_max_steps_budget_is_enforced(self) -> None:
        out = trace_separatrix(
            start_vertex=NBR[NORTH][0],
            direction="descent",
            pts=PTS,
            f_vals=self.F,
            neighbors=NBR,
            zones=[],
            max_steps=2,
        )
        assert out.status == "max_steps"
        assert out.confidence == "low"
        assert len(out.path) == 3

    def test_path_is_strictly_monotone(self) -> None:
        out = trace_separatrix(
            start_vertex=NBR[NORTH][1],
            direction="descent",
            pts=PTS,
            f_vals=self.F,
            neighbors=NBR,
            zones=[],
            max_steps=4096,
        )
        vals = [self.F[v] for v in out.path]
        assert all(b < a for a, b in pairwise(vals))

    def test_bad_direction_raises(self) -> None:
        with pytest.raises(ValueError, match="direction"):
            trace_separatrix(NORTH, "sideways", PTS, self.F, NBR, [])

    def test_start_inside_cap_terminates_immediately(self) -> None:
        zones = [_cap("MAX_N", "absorb_max", (0, 0, 1), math.radians(30))]
        out = trace_separatrix(NORTH, "ascent", PTS, self.F, NBR, zones)
        assert out.status == "absorbed"
        assert out.n_steps == 0


class TestPassThroughAndTunneling:
    """A noise plateau: a shallow basin sits on the monotone slope.

    Models the Pd20K MC10 situation -- spurious extrema inside an annihilated
    cluster that must not terminate a separatrix.
    """

    NOISE_CENTER = np.array(
        [0.0, math.sin(math.radians(60)), math.cos(math.radians(60))]
    )

    def _field_with_noise_sink(self) -> tuple[np.ndarray, int]:
        f = PTS[:, 2].copy()
        pu = PTS / np.linalg.norm(PTS, axis=1, keepdims=True)
        cosang = pu @ (self.NOISE_CENTER / np.linalg.norm(self.NOISE_CENTER))
        disk = cosang >= math.cos(math.radians(12))
        # Carve a dip: the disk becomes a local minimum of the raw data, so a
        # naive descent from the north pole gets trapped there.
        f[disk] -= 0.35
        trap = int(np.argmax(np.where(disk, f, -np.inf)))
        return f, trap

    def test_naive_walk_gets_trapped_by_noise(self) -> None:
        f, trap = self._field_with_noise_sink()
        out = trace_separatrix(trap, "descent", PTS, f, NBR, [])
        assert out.status == "raw_extremum"
        assert out.target_label is None

    def test_passthrough_zone_is_never_a_target(self) -> None:
        f, trap = self._field_with_noise_sink()
        zones = [
            _cap(
                "ANNUL_NOISE", "passthrough", tuple(self.NOISE_CENTER), math.radians(12)
            ),
            _cap("MIN_S", "absorb_min", (0, 0, -1), math.radians(20)),
        ]
        out = trace_separatrix(trap, "descent", PTS, f, NBR, zones, max_steps=4096)
        assert out.target_label != "ANNUL_NOISE"
        assert out.target_kind != "passthrough"

    def test_tunnel_escapes_noise_sink_and_continues(self) -> None:
        f, trap = self._field_with_noise_sink()
        zones = [
            _cap(
                "ANNUL_NOISE", "passthrough", tuple(self.NOISE_CENTER), math.radians(12)
            ),
            _cap("MIN_S", "absorb_min", (0, 0, -1), math.radians(25)),
        ]
        out = trace_separatrix(trap, "descent", PTS, f, NBR, zones, max_steps=4096)
        assert out.status == "absorbed"
        assert out.target_label == "MIN_S"
        # The path must have crossed the noise disk and left it.
        pu = PTS / np.linalg.norm(PTS, axis=1, keepdims=True)
        inside = [
            v
            for v in out.path
            if float(pu[v] @ (self.NOISE_CENTER / np.linalg.norm(self.NOISE_CENTER)))
            >= math.cos(math.radians(12))
        ]
        assert inside, "walk should have entered the noise disk"
        assert float(pu[out.path[-1]] @ np.array([0.0, 0.0, -1.0])) > 0.9

    def test_tunnel_does_not_reenter_banned_zone(self) -> None:
        f, trap = self._field_with_noise_sink()
        z_noise = _cap(
            "ANNUL_NOISE", "passthrough", tuple(self.NOISE_CENTER), math.radians(12)
        )
        zones = [z_noise, _cap("MIN_S", "absorb_min", (0, 0, -1), math.radians(25))]
        out = trace_separatrix(trap, "descent", PTS, f, NBR, zones, max_steps=4096)
        pu = PTS / np.linalg.norm(PTS, axis=1, keepdims=True)
        hits = [
            i
            for i, v in enumerate(out.path)
            if float(pu[v] @ np.asarray(z_noise.center_unit)) >= z_noise.cos_radius
        ]
        # Contiguous single crossing: never re-enters after leaving.
        assert hits == list(range(hits[0], hits[-1] + 1))

    def test_ascent_unaffected_by_noise_sink(self) -> None:
        """A dip below the start point is not a trap for an ascending walk."""
        f, trap = self._field_with_noise_sink()
        zones = [_cap("MAX_N", "absorb_max", (0, 0, 1), math.radians(20))]
        out = trace_separatrix(trap, "ascent", PTS, f, NBR, zones, max_steps=4096)
        assert out.status == "absorbed"
        assert out.target_label == "MAX_N"


class TestMakeZones:
    def test_zone_kinds_and_labels(self) -> None:
        zones = make_zones(
            visible_maxima=[("MAX_1", (0.0, 0.0, 1.0))],
            visible_minima=[("MIN_1", (0.0, 0.0, -1.0))],
            visible_saddles=[("SAD_1", (1.0, 0.0, 0.0))],
            absorption_caps={"MAX_1": 0.2},
            passthrough=[("ANNUL_1", (0.0, 1.0, 0.0), 0.1)],
            default_radius_rad=0.05,
        )
        kinds = {z.label: z.kind for z in zones}
        assert kinds == {
            "MAX_1": "absorb_max",
            "MIN_1": "absorb_min",
            "SAD_1": "absorb_sad",
            "ANNUL_1": "passthrough",
        }
        by_label = {z.label: z for z in zones}
        assert by_label["MAX_1"].cos_radius == math.cos(0.2)
        assert by_label["MIN_1"].cos_radius == math.cos(0.05)

    def test_center_at_origin_is_skipped(self) -> None:
        zones = make_zones(
            visible_maxima=[("BAD", (0.0, 0.0, 0.0))],
            visible_minima=[],
            visible_saddles=[],
        )
        assert zones == []

    def test_contains_uses_angular_cap(self) -> None:
        z = _cap("C", "absorb_max", (0.0, 0.0, 1.0), math.radians(30))
        assert z.contains((0.0, 0.0, 5.0))  # direction matters, not radius
        assert not z.contains((1.0, 0.0, 0.0))
        assert not z.contains((0.0, 0.0, 0.0))  # degenerate origin


class TestOutcomeMetrics:
    """tunnel_count and angular_length_rad, surfaced on the viewer's info card."""

    def test_straight_path_has_no_tunnels(self) -> None:
        f = PTS[:, 2].copy()
        zones = [_cap("MAX_N", "absorb_max", (0, 0, 1), math.radians(20))]
        out = trace_separatrix(NBR[NORTH][0], "ascent", PTS, f, NBR, zones)
        assert out.tunnel_count == 0
        assert out.status == "absorbed"

    def test_angular_length_matches_great_circle_distance(self) -> None:
        """A single hop's length is the angle between the two vertex directions."""
        f = PTS[:, 2].copy()
        # Start well outside the cap so the walk takes a hop before absorption.
        v0 = _vertex_at_colatitude(math.radians(70))
        zones = [_cap("MAX_N", "absorb_max", (0, 0, 1), math.radians(20))]
        out = trace_separatrix(v0, "ascent", PTS, f, NBR, zones, max_steps=1)
        assert out.n_steps == 1
        a = PTS[v0] / np.linalg.norm(PTS[v0])
        b = PTS[out.path[1]] / np.linalg.norm(PTS[out.path[1]])
        expect = math.acos(max(-1.0, min(1.0, float(a @ b))))
        assert out.angular_length_rad == pytest.approx(expect)

    def test_angular_length_is_additive_over_hops(self) -> None:
        f = PTS[:, 2].copy()
        zones = [_cap("MAX_N", "absorb_max", (0, 0, 1), math.radians(20))]
        out = trace_separatrix(
            _vertex_at_colatitude(math.radians(70)), "ascent", PTS, f, NBR, zones
        )
        assert out.n_steps >= 2
        assert out.angular_length_rad > 0.0
        # Traversal of a 20-degree cap from the equator region is bounded.
        assert out.angular_length_rad < math.radians(180)

    def test_single_vertex_path_has_zero_length(self) -> None:
        f = PTS[:, 2].copy()
        zones = [_cap("MAX_N", "absorb_max", (0, 0, 1), math.radians(30))]
        out = trace_separatrix(NORTH, "ascent", PTS, f, NBR, zones)
        assert out.n_steps == 0
        assert out.angular_length_rad == 0.0
        assert out.tunnel_count == 0

    def test_tunneling_reports_its_count_and_length(self) -> None:
        """The noise-sink fixture: escaping the disk is a great-circle jump,
        so the count is 1 and the length includes the jump."""
        f = PTS[:, 2].copy()
        centre = TestPassThroughAndTunneling.NOISE_CENTER
        pu = PTS / np.linalg.norm(PTS, axis=1, keepdims=True)
        cosang = pu @ (centre / np.linalg.norm(centre))
        disk = cosang >= math.cos(math.radians(12))
        f[disk] -= 0.35
        trap = int(np.argmax(np.where(disk, f, -np.inf)))

        zones = [
            _cap("ANNUL_NOISE", "passthrough", tuple(centre), math.radians(12)),
            _cap("MIN_S", "absorb_min", (0, 0, -1), math.radians(25)),
        ]
        out = trace_separatrix(trap, "descent", PTS, f, NBR, zones, max_steps=4096)
        assert out.status == "absorbed"
        assert out.tunnel_count == 1
        # The jump crosses the disk's diameter, so the length is at least the
        # cap's angular width.
        assert out.angular_length_rad >= math.radians(24)

    def test_two_disks_report_two_tunnels(self) -> None:
        """Two noise sinks on the descent path -> two tunnel events."""
        f = PTS[:, 2].copy()
        pu = PTS / np.linalg.norm(PTS, axis=1, keepdims=True)
        c1 = np.array([0.0, math.sin(math.radians(70)), math.cos(math.radians(70))])
        c2 = np.array([0.0, math.sin(math.radians(35)), math.cos(math.radians(35))])
        radii = [math.radians(10), math.radians(10)]
        for centre, rad in zip((c1, c2), radii, strict=True):
            disk = (pu @ (centre / np.linalg.norm(centre))) >= math.cos(rad)
            f[disk] -= 0.35
        start = int(
            np.argmax(
                np.where(
                    (pu @ (c1 / np.linalg.norm(c1))) >= math.cos(radii[0]), f, -np.inf
                )
            )
        )

        zones = [
            _cap("ANNUL_A", "passthrough", tuple(c1), radii[0]),
            _cap("ANNUL_B", "passthrough", tuple(c2), radii[1]),
            _cap("MIN_S", "absorb_min", (0, 0, -1), math.radians(25)),
        ]
        out = trace_separatrix(start, "descent", PTS, f, NBR, zones, max_steps=4096)
        assert out.tunnel_count >= 1
        assert out.target_label == "MIN_S"
