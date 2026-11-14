"""Regression tests for the closed separatrix cycle rule (basin v12).

These cover :func:`gba_topology2.src.basin_boundary._boundary_cycle` as reached
through ``find_minimum_basins`` / ``find_maximum_basins``.

The case that motivated them, from a real Pd20k snapshot: minimum MIN_19182
(MC73) is bounded by two saddles, MC15 (2 ascending branches) and MC20 (4 --
a monkey saddle). MC15's two branches and MC20's first two close on maxima
MC36 and MC40, so the basin is a ring of 4 arcs. MC20's branches 2 and 3 run
to MC42 and MC34, which bound the basins of *other* minima, not this one.
Drawing them paints the hill next door with MC73's colour.

The pre-v12 filter selected arcs whose target had degree exactly 2, and bailed
out entirely if any target had degree 1. MC42 and MC34 have degree 1 *because*
they are spurious, so the escape hatch fired in precisely the case the filter
was written for and nothing was ever dropped. Leaf-pruning has no such hole:
the degree-1 target is the thing being removed, not the reason to give up.
"""

from collections.abc import Mapping, Sequence
from typing import ClassVar

from gba_topology2.src.basin_boundary import (
    MaximumBasin,
    MinimumBasin,
    _boundary_cycle,
    find_maximum_basins,
    find_minimum_basins,
    serialize_maximum_basins,
    serialize_minimum_basins,
)
from gba_topology2.src.basin_center import BasinCenter
from gba_topology2.src.max_connection import MaxConnection, MinConnection

# (saddle, max_id) targets are optional: a stalled arc carries None, and the
# graph builders do not filter it out, so the rule has to decide what it means.
_ArcSpec = Sequence[tuple[int, str | None]]


def _conn(
    saddle_id: str,
    max_id: str | None,
    branch_index: int,
) -> MaxConnection:
    """An ascending saddle -> maximum branch."""
    return MaxConnection(
        saddle_id=saddle_id,
        branch_index=branch_index,
        max_id=max_id,
        status="connected",
        direction_mode="ray",
        cross_track_deg=0.0,
        along_track_deg=1.0,
        separation_deg=10.0,
        mutual=True,
        arc_points=((0.0, 0.0, 1.0), (1.0, 0.0, 0.0)),
        saddle_value=0.5,
        max_value=1.0 if max_id else None,
        verified=True,
    )


def _desc(
    saddle_id: str,
    min_id: str | None,
    branch_index: int,
) -> MinConnection:
    """A descending saddle -> minimum branch."""
    return MinConnection(
        saddle_id=saddle_id,
        branch_index=branch_index,
        min_id=min_id,
        status="connected",
        direction_mode="port_tangent",
        cross_track_deg=0.0,
        along_track_deg=1.0,
        separation_deg=10.0,
        arc_points=((0.0, 0.0, 1.0), (1.0, 0.0, 0.0)),
        saddle_value=0.5,
        min_value=0.1 if min_id else None,
        verified=True,
    )


def _center(center_id: str, polarity: str) -> BasinCenter:
    """A critical point, at a position the rule never looks at."""
    z = -1.0 if polarity == "minimum" else 1.0
    return BasinCenter(
        center_id=center_id,
        polarity=polarity,
        position=(0.0, 0.0, z),
        value=z,
        is_effective=True,
        vertex_id=-1,
        origin_id=None,
        net_index=1,
        member_count=1,
    )


def _min_side(
    arcs: Mapping[str, _ArcSpec],
    drains: Sequence[tuple[str, int, str]],
    minima: Sequence[str],
) -> dict[str, MinimumBasin]:
    """Basins of ``minima`` from ``arcs`` (saddle -> (branch, max_id)).

    ``drains`` is the descent table: which saddle drains into which minimum,
    i.e. what makes a saddle a boundary saddle of that basin.
    """
    connections = [
        _conn(saddle, max_id, bi)
        for saddle, branches in arcs.items()
        for bi, max_id in branches
    ]
    centers = [_center(m, "minimum") for m in minima]
    descents = [_desc(saddle, min_id, bi) for saddle, bi, min_id in drains]
    return find_minimum_basins(connections, centers, descents)


def _shape(basin: MinimumBasin) -> list[tuple[str, int, str | None]]:
    """A basin's ring as (saddle, branch, target) triples."""
    return [(a.saddle_id, a.branch_index, a.max_id) for a in basin.boundary_arcs]


class TestMonkeySaddleCycle:
    """The reported defect, at the exact topology that caused it."""

    # MC15 + MC20 bound MIN_19182; MC20's far branches reach MC42 / MC34.
    RING: ClassVar[Mapping[str, _ArcSpec]] = {
        "E-SAD_MC15": [(0, "E-MAX_MC36"), (1, "E-MAX_MC40")],
        "E-SAD_MC20": [
            (0, "E-MAX_MC36"),
            (1, "E-MAX_MC40"),
            (2, "E-MAX_MC42"),
            (3, "E-MAX_MC34"),
        ],
    }
    DRAINS: ClassVar[Sequence[tuple[str, int, str]]] = [
        ("E-SAD_MC15", 0, "MIN_19182"),
        ("E-SAD_MC20", 0, "MIN_19182"),
    ]

    def test_mc73_ring_is_the_four_closed_arcs(self) -> None:
        basins = _min_side(self.RING, self.DRAINS, ["MIN_19182"])
        assert _shape(basins["MIN_19182"]) == [
            ("E-SAD_MC15", 0, "E-MAX_MC36"),
            ("E-SAD_MC15", 1, "E-MAX_MC40"),
            ("E-SAD_MC20", 0, "E-MAX_MC36"),
            ("E-SAD_MC20", 1, "E-MAX_MC40"),
        ]

    def test_spurious_branches_leave_the_serializer_too(self) -> None:
        """The screen draws the serialized ids, so the drop must reach them."""
        basins = _min_side(self.RING, self.DRAINS, ["MIN_19182"])
        out = serialize_minimum_basins(basins)["MIN_19182"]
        assert out["boundary_maxima"] == ["E-MAX_MC36", "E-MAX_MC40"]
        assert out["boundary_saddles"] == ["E-SAD_MC15", "E-SAD_MC20"]

    def test_dropping_leaves_the_saddle_that_bounds_the_basin(self) -> None:
        """MC20 stays a boundary saddle: 2 of its 4 branches, not zero of them."""
        basins = _min_side(self.RING, self.DRAINS, ["MIN_19182"])
        saddles = serialize_minimum_basins(basins)["MIN_19182"]["boundary_saddles"]
        assert "E-SAD_MC20" in saddles


class TestMaxSideIsTheDual:
    """The same defect on the maximum side, where it was equally dead."""

    @staticmethod
    def _max_side(
        adjacency: Mapping[str, Mapping[str, Sequence[int]]],
        geometry: Mapping[str, _ArcSpec],
        maxima: Sequence[str],
    ) -> dict[str, MaximumBasin]:
        """Basins of ``maxima``.

        Mirrors :func:`_min_side` with the tables exchanged, which is what the
        duality is: ``adjacency`` (saddle -> branches climbing into a maximum)
        decides who bounds, ``geometry`` (saddle -> (branch, min_id)) is the
        arc that gets drawn.
        """
        connections = [
            _conn(saddle, maximum, bi)
            for maximum, saddles in adjacency.items()
            for saddle, branches in saddles.items()
            for bi in branches
        ]
        centers = [_center(m, "maximum") for m in maxima]
        descents = [
            _desc(saddle, min_id, bi)
            for saddle, branches in geometry.items()
            for bi, min_id in branches
        ]
        return find_maximum_basins(connections, centers, descents)

    def test_monkey_saddle_branches_pruned_on_a_hill(self) -> None:
        basins = self._max_side(
            adjacency={"MAX_4864": {"S1": [0, 1], "S2": [0, 1]}},
            geometry={
                "S1": [(0, "M1"), (1, "M2")],
                "S2": [(0, "M1"), (1, "M2"), (2, "M3"), (3, "M4")],
            },
            maxima=["MAX_4864"],
        )
        ring = basins["MAX_4864"].boundary_arcs
        assert [(a.saddle_id, a.branch_index, a.min_id) for a in ring] == [
            ("S1", 0, "M1"),
            ("S1", 1, "M2"),
            ("S2", 0, "M1"),
            ("S2", 1, "M2"),
        ]
        assert serialize_maximum_basins(basins)["MAX_4864"]["boundary_minima"] == [
            "M1",
            "M2",
        ]


class TestTwoSaddlesSharingOneTarget:
    """The shape that looks like a tail and is not."

    My first reading of the rule expected a multi-arc tail to be pruned whole.
    It is not, and the rule is right: a target met by two bounding saddles is
    on the ring by definition, which is the same 2-saddles-into-1-minimum shape
    as ethene3's hill MAX_4864. Cascading is not a thing this rule can do --
    see test_single_pass_is_the_fixed_point.
    """

    def test_two_saddles_sharing_a_maximum_are_part_of_the_ring(self) -> None:
        # Cycle on M1/M2 via S1,S2. S2's third branch and S3's first meet on
        # M3, so M3 has degree 2 and stays; only M4, S3's other branch, is a
        # leaf.
        arcs: dict[str, _ArcSpec] = {
            "S1": [(0, "M1"), (1, "M2")],
            "S2": [(0, "M1"), (1, "M2"), (2, "M3")],
            "S3": [(0, "M3"), (1, "M4")],
        }
        drains = [(s, 0, "MIN_X") for s in arcs]
        basin = _min_side(arcs, drains, ["MIN_X"])["MIN_X"]
        assert _shape(basin) == [
            ("S1", 0, "M1"),
            ("S1", 1, "M2"),
            ("S2", 0, "M1"),
            ("S2", 1, "M2"),
            ("S2", 2, "M3"),
            ("S3", 0, "M3"),
        ]


class TestShapesThatMustSurvive:
    """Everything the rule is not allowed to eat."""

    def test_degree_three_target_inside_a_cycle_is_kept(self) -> None:
        # Three saddles sharing both maxima: degree 3, no leaves, all kept.
        arcs = {f"S{n}": [(0, "M1"), (1, "M2")] for n in (1, 2, 3)}
        drains = [(s, 0, "MIN_X") for s in arcs]
        basin = _min_side(arcs, drains, ["MIN_X"])["MIN_X"]
        assert len(basin.boundary_arcs) == 6

    def test_parallel_branches_of_one_saddle_are_kept(self) -> None:
        """Two branches of one saddle into the same max: ethene3's MS3 -> 4864.

        Degree 4, and the two arcs are not a tail -- they are the double pass
        over the same shoulder, so both are drawn.
        """
        arcs: dict[str, _ArcSpec] = {
            "E-SAD_MS3": [(0, "MAX_4864"), (1, "MAX_4864")],
            "E-SAD_MS4": [(0, "MAX_4864"), (1, "MAX_4864")],
        }
        drains = [("E-SAD_MS3", 0, "E-MIN_MC3"), ("E-SAD_MS4", 0, "E-MIN_MC3")]
        basin = _min_side(arcs, drains, ["E-MIN_MC3"])["E-MIN_MC3"]
        assert len(basin.boundary_arcs) == 4

    def test_single_saddle_basin_keeps_all_its_arcs(self) -> None:
        """No cycle exists: the two arcs are each other's only endpoints.

        Pd20k's MIN_19177/78/79 look like this. There is no closed boundary to
        extract, so every candidate is drawn, as it was before v12.
        """
        basin = _min_side(
            {"S1": [(0, "M1"), (1, "M2")]},
            [("S1", 0, "MIN_X")],
            ["MIN_X"],
        )["MIN_X"]
        assert len(basin.boundary_arcs) == 2

    def test_a_basin_with_no_leaves_is_untouched(self) -> None:
        """The clean 6-arc ring (3 saddles x 2 branches, 3 maxima) is the norm."""
        arcs = {
            "E-SAD_MC23": [(0, "E-MAX_MC39"), (1, "E-MAX_MC41")],
            "E-SAD_MC25": [(0, "E-MAX_MC39"), (1, "E-MAX_MC37")],
            "E-SAD_MC27": [(0, "E-MAX_MC41"), (1, "E-MAX_MC37")],
        }
        drains = [(s, 0, "MIN_0") for s in arcs]
        basin = _min_side(arcs, drains, ["MIN_0"])["MIN_0"]
        assert len(basin.boundary_arcs) == 6
        out = serialize_minimum_basins({"MIN_0": basin})["MIN_0"]
        assert out["boundary_maxima"] == [
            "E-MAX_MC37",
            "E-MAX_MC39",
            "E-MAX_MC41",
        ]


class TestDegenerateAndStable:
    """Order, determinism, stalled arcs."""

    def test_ring_order_follows_sorted_saddles(self) -> None:
        arcs: dict[str, _ArcSpec] = {
            "S-C": [(0, "M1"), (1, "M2"), (2, "M3")],
            "S-A": [(0, "M1"), (1, "M2")],
            "S-B": [(0, "M1"), (1, "M2"), (2, "M4")],
        }
        drains = [(s, 0, "MIN_X") for s in arcs]
        basin = _min_side(arcs, drains, ["MIN_X"])["MIN_X"]
        # M3 and M4 are leaves (one arc each); the two survivors per saddle are
        # kept in sorted-saddle order, both branches of each.
        assert [(a.saddle_id, a.branch_index) for a in basin.boundary_arcs] == [
            ("S-A", 0),
            ("S-A", 1),
            ("S-B", 0),
            ("S-B", 1),
            ("S-C", 0),
            ("S-C", 1),
        ]

    def test_single_pass_is_the_fixed_point(self) -> None:
        """One pass equals iterate-to-stability, so the loop is not needed.

        Deleting a leaf drops that target's only arc, removing the target
        outright; every other target's degree is untouched. No target can
        become a leaf in a second pass. Pinned here because the property is
        what licenses the straight-line implementation.
        """
        seeds: list[dict[str, _ArcSpec]] = [
            {"S1": [(0, "M1"), (1, "M2"), (2, "M3"), (3, "M4")]},
            {
                "S1": [(0, "M1"), (1, "M2")],
                "S2": [(0, "M1"), (1, "M2"), (2, "M3")],
                "S3": [(0, "M3"), (1, "M4")],
            },
            {"S1": [(0, None), (1, "M1")]},
            {
                "S1": [(0, "M1"), (1, "M1")],
                "S2": [(0, "M1"), (1, "M2")],
            },
        ]
        for spec in seeds:
            arcs = [
                _conn(s, max_id, bi)
                for s, branches in spec.items()
                for bi, max_id in branches
            ]
            once = _boundary_cycle(arcs, lambda a: a.max_id)
            twice = _boundary_cycle(once, lambda a: a.max_id)
            assert [id(a) for a in once] == [id(a) for a in twice]

    def test_determinism_across_input_permutations(self) -> None:
        arcs: dict[str, _ArcSpec] = {
            "S1": [(0, "M1"), (1, "M2")],
            "S2": [(0, "M1"), (1, "M2"), (2, "M3"), (3, "M4")],
        }
        drains = [("S1", 0, "MIN_X"), ("S2", 0, "MIN_X")]
        first = _shape(_min_side(arcs, drains, ["MIN_X"])["MIN_X"])
        flipped_arcs: dict[str, _ArcSpec] = {
            k: list(reversed(v)) for k, v in reversed(list(arcs.items()))
        }
        flipped = _shape(
            _min_side(flipped_arcs, list(reversed(drains)), ["MIN_X"])["MIN_X"]
        )
        assert sorted(first) == sorted(flipped)

    def test_stalled_arcs_are_leaves_and_drop_out(self) -> None:
        """max_id None reaches the rule; it is a leaf, not a target."""
        arcs: dict[str, _ArcSpec] = {
            "S1": [(0, "M1"), (1, "M2")],
            "S2": [(0, "M1"), (1, "M2"), (2, None)],
        }
        drains = [("S1", 0, "MIN_X"), ("S2", 0, "MIN_X")]
        basin = _min_side(arcs, drains, ["MIN_X"])["MIN_X"]
        assert _shape(basin) == [
            ("S1", 0, "M1"),
            ("S1", 1, "M2"),
            ("S2", 0, "M1"),
            ("S2", 1, "M2"),
        ]

    def test_all_stalled_falls_back_to_every_arc(self) -> None:
        """No cycle at all: keep the candidates rather than draw nothing."""
        arcs: dict[str, _ArcSpec] = {"S1": [(0, None), (1, None)]}
        basin = _min_side(arcs, [("S1", 0, "MIN_X")], ["MIN_X"])["MIN_X"]
        assert len(basin.boundary_arcs) == 2

    def test_helper_is_idempotent_on_its_own_output(self) -> None:
        arcs = [
            _conn("S1", "M1", 0),
            _conn("S1", "M2", 1),
            _conn("S2", "M1", 0),
            _conn("S2", "M2", 1),
            _conn("S2", "M3", 2),
        ]
        once = _boundary_cycle(arcs, lambda a: a.max_id)
        twice = _boundary_cycle(once, lambda a: a.max_id)
        assert [id(a) for a in once] == [id(a) for a in twice]

    def test_empty_input_is_empty(self) -> None:
        empty: list[MaxConnection] = []
        assert _boundary_cycle(empty, lambda a: a.max_id) == []

    def test_helper_binds_to_descent_arcs(self) -> None:
        """The TypeVar covers MinConnection as well."""
        arcs: list[MinConnection] = [
            _desc("S1", "M1", 0),
            _desc("S1", "M2", 1),
            _desc("S2", "M1", 0),
            _desc("S2", "M2", 1),
            _desc("S2", "M3", 2),
        ]
        kept = _boundary_cycle(arcs, lambda a: a.min_id)
        assert [(a.saddle_id, a.min_id) for a in kept] == [
            ("S1", "M1"),
            ("S1", "M2"),
            ("S2", "M1"),
            ("S2", "M2"),
        ]
