"""Source-level guards for the ridge/valley colour convention.

A ridge port and an ascent launch seed are the **same manifold**: the ridge is
the unstable manifold of the saddle, running towards maxima, and the launch seed
is the stub that leaves the saddle along it. Once seeds launch from the ports,
drawing them in opposite colours is not a cosmetic inconsistency -- it tells the
operator that a port and the stub growing out of it are different structures.

The viewer previously had valley = orange / ridge = cyan for ports and
ascent = orange-red / descent = cyan for seeds, and both features default to on.
These guards pin the reconciled convention and stop it drifting back.

Headless, source-level checks, following the pattern in
``test_sphere_viewer_theme.py``: no server, no render window.
"""

import pathlib
import re

import pytest

from gba_topology2.src import sphere_viewer as sv

SOURCE = pathlib.Path(sv.__file__).resolve()

# The two RGB triples the viewer uses for the ascending and descending manifolds.
ASCENT_RGB = (1.0, 0.35, 0.10)
DESCENT_RGB = (0.10, 0.75, 1.0)


def _set_color(source: str, actor: str) -> tuple[float, float, float]:
    """Read the RGB triple assigned to ``actor``'s property."""
    match = re.search(
        rf"{actor}\.GetProperty\(\)\.SetColor\(\s*"
        rf"([0-9.]+),\s*([0-9.]+),\s*([0-9.]+)\s*\)",
        source,
    )
    assert match, f"no SetColor found for {actor}"
    return (float(match.group(1)), float(match.group(2)), float(match.group(3)))


@pytest.mark.parametrize(
    "actor, expected",
    [
        ("rport_actor", ASCENT_RGB),
        ("vport_actor", DESCENT_RGB),
        ("launch_ascent_actor", ASCENT_RGB),
        ("launch_descent_actor", DESCENT_RGB),
    ],
)
def test_manifold_colours_match_across_features(
    actor: str, expected: tuple[float, float, float]
) -> None:
    """Ridge ports and ascent seeds share one colour; valleys and descent the other."""
    source = SOURCE.read_text(encoding="utf-8")
    assert _set_color(source, actor) == pytest.approx(expected)


def test_ridge_and_ascent_are_the_same_hue() -> None:
    """The invariant, stated directly, so a rename of either actor cannot dodge it."""
    source = SOURCE.read_text(encoding="utf-8")
    ridge = _set_color(source, "rport_actor")
    ascent = _set_color(source, "launch_ascent_actor")
    valley = _set_color(source, "vport_actor")
    descent = _set_color(source, "launch_descent_actor")

    assert ridge == ascent, "ridge port and ascent seed are the same manifold"
    assert valley == descent, "valley port and descent seed are the same manifold"
    # The two manifolds must stay distinguishable from each other.
    assert ridge != valley


def test_port_switch_label_matches_the_actors() -> None:
    """The legend is the operator's only key to the colours, so it must agree."""
    source = SOURCE.read_text(encoding="utf-8")
    assert 'label="Show Boundary Ports (Orange=Ridge, Cyan=Valley)"' in source
    assert "Orange=Valley" not in source, "stale legend: valley is cyan, not orange"


def test_seed_switch_label_matches_the_actors() -> None:
    """The seed legend names ascent as orange and descent as cyan."""
    source = SOURCE.read_text(encoding="utf-8")
    assert 'label="Show Launch Seeds (Orange=Ascent, Cyan=Descent)"' in source


def test_port_colours_are_not_reintroduced_as_literals() -> None:
    """The old port palette must not return under a different actor name."""
    source = SOURCE.read_text(encoding="utf-8")
    # (1.0, 0.55, 0.0) was the valley-port orange; (0.0, 0.85, 0.95) the ridge cyan.
    assert "1.0, 0.55, 0.0" not in source, (
        "old valley-port orange is the seed ascent hue"
    )
    assert "0.0, 0.85, 0.95" not in source, "old ridge-port cyan collides with descent"
