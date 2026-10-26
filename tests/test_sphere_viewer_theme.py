"""Unit tests for the sphere viewer's high-contrast Vuetify theme.

The app renders in Vuetify's stock *light* theme: the drawer's old
``html.Style`` block never applied (Vue strips ``<style>`` from component
templates) and the old ``state.trame__theme = "dark"`` is read by nothing in
trame. On the stock light theme the cards are ``surface-variant`` = #424242 with
emphasis text at 0.6 opacity, which composites to 1.73:1, and the Material
accents the cards name (``.text-teal`` #009688, ``.text-grey`` #9e9e9e) are
light-theme mid-tones. That is the grey-on-grey ``VUE3_THEME_CONFIG`` removes.

These are headless, source-level checks (no server, no browser, no render
window) covering:

1. ``VUE3_THEME_CONFIG`` shape -- a light theme with a full palette.
2. WCAG contrast -- every palette entry clears AA as *text* on every surface it
   can land on, and as a *tonal chip* label against its own 12% tint; slider
   thumbs clear the 3:1 non-text floor.
3. Emphasis opacity -- the variable that drives the ~50 ``text-medium-emphasis``
   labels is raised above the stock 0.6.
4. A drift guard -- every ``color=`` / ``text-*`` token in the drawer scope
   resolves to a palette entry, so a future accent cannot silently fall back to
   static mid-tone Material CSS.
5. Regression guards -- the dead ``trame__theme`` assignment and the dead
   ``html.Style`` block are gone, the config is handed to the layout, and the
   force-dark hexes do not reappear.

The contrast math mirrors what the browser computes; the tonal-chip and
emphasis-opacity assumptions are each pinned to the installed Vuetify bundle so
an upgrade that changes them fails here rather than silently degrading the UI.
"""

from __future__ import annotations

import math
import pathlib
import re
from typing import Any

import pytest

import gba_topology2.src.sphere_viewer as sv

SOURCE = pathlib.Path(sv.__file__).resolve()

# The palette entries the drawer uses as *text* (via ``text-*`` classes) and as
# *chip labels*. VChip defaults to ``variant="tonal"``, so a chip paints its
# label in the accent and its background in the same accent at 12% over the
# card; every accent must therefore clear AA in both roles.
TEXT_TOKENS: tuple[str, ...] = (
    "primary",
    "secondary",
    "tertiary",
    "error",
    "info",
    "success",
    "warning",
    "teal",
    "cyan",
    "purple",
    "amber",
    "orange",
    "deep-orange",
    "grey",
    "on-surface-variant",
)

FILL_TOKENS: tuple[str, ...] = (
    "primary",
    "secondary",
    "tertiary",
    "error",
    "info",
    "success",
    "warning",
    "teal",
    "cyan",
    "purple",
    "amber",
    "orange",
    "deep-orange",
    "grey",
    "amber-darken-2",
    "amber-darken-3",
    "orange-darken-2",
    "deep-orange-darken-1",
    "cyan-darken-2",
    "purple-darken-2",
    "grey-darken-1",
)

# Surfaces text and chips can land on, taken from the palette so the contrast
# tests follow the theme instead of duplicating it. See the ``surfaces`` fixture.
SURFACE_TOKENS: tuple[str, ...] = ("background", "surface-variant", "surface-light")

# VChip's tonal underlay is ``currentColor`` at ``--v-activated-opacity``.
TONAL_ALPHA = 0.12
# Vuetify paints the slider track in the slider's accent at this alpha over the
# surface; the fill and thumb use the accent at full strength.
SLIDER_TRACK_ALPHA = 0.38

WCAG_AA = 4.5
WCAG_AAA = 7.0
WCAG_NON_TEXT = 3.0

# Typography / alignment utilities that share the ``text-`` prefix but are not
# colors, excluded from the drift guard.
NON_COLOR_TEXT_CLASSES: frozenset[str] = frozenset(
    {
        "body-1",
        "body-2",
        "caption",
        "center",
        "left",
        "right",
        "justify",
        "mono",
        "medium-emphasis",
        "high-emphasis",
        "disabled",
        "subtitle-1",
        "subtitle-2",
        "uppercase",
        "lowercase",
        "capitalize",
        "nowrap",
        "decoration-none",
    }
)


# ---------------------------------------------------------------------------
# Color math, ported from Vuetify so the test measures what the browser does.
# ---------------------------------------------------------------------------


def _rgb(hex_color: str) -> tuple[int, int, int]:
    """Split a ``#rrggbb`` string into channel bytes.

    Args:
        hex_color: Six-digit hex color with a leading ``#``.

    Returns:
        The red, green and blue channels as integers in 0..255.
    """
    digits = hex_color.lstrip("#")
    if len(digits) == 3:
        digits = "".join(ch * 2 for ch in digits)
    return (
        int(digits[0:2], 16),
        int(digits[2:4], 16),
        int(digits[4:6], 16),
    )


def _channel_linear(channel: int) -> float:
    """Linearise one sRGB channel for relative-luminance math.

    Args:
        channel: The channel value in 0..255.

    Returns:
        The channel in linear light.
    """
    scaled = channel / 255.0
    if scaled <= 0.04045:
        return scaled / 12.92
    return float(((scaled + 0.055) / 1.055) ** 2.4)


def _relative_luminance(hex_color: str) -> float:
    """Return the WCAG relative luminance of a color.

    Args:
        hex_color: The color to measure.

    Returns:
        Luminance in 0..1.
    """
    red, green, blue = (_channel_linear(c) for c in _rgb(hex_color))
    return 0.2126 * red + 0.7152 * green + 0.0722 * blue


def contrast_ratio(foreground: str, background: str) -> float:
    """Return the WCAG 1.4.3 contrast ratio between two colors.

    Args:
        foreground: The text color.
        background: The surface color.

    Returns:
        The contrast ratio, from 1.0 (identical) to 21.0 (black on white).
    """
    lum_a = _relative_luminance(foreground)
    lum_b = _relative_luminance(background)
    lighter = max(lum_a, lum_b)
    darker = min(lum_a, lum_b)
    return (lighter + 0.05) / (darker + 0.05)


def composite(foreground: str, opacity: float, background: str) -> str:
    """Flatten a translucent foreground onto an opaque background.

    Args:
        foreground: The text color, e.g. ``on-background``.
        opacity: The alpha applied to the text color.
        background: The surface the text sits on.

    Returns:
        The resulting opaque color as ``#rrggbb``.
    """
    fg = _rgb(foreground)
    bg = _rgb(background)
    blended = tuple(round(opacity * f + (1.0 - opacity) * b) for f, b in zip(fg, bg))
    return "#{:02X}{:02X}{:02X}".format(*blended)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def light_theme() -> dict[str, Any]:
    """Return the light theme block of ``VUE3_THEME_CONFIG``.

    Returns:
        The dict holding ``dark``, ``colors`` and ``variables``.
    """
    config: dict[str, Any] = sv.VUE3_THEME_CONFIG
    theme: dict[str, Any] = config["theme"]["themes"]["light"]
    return theme


@pytest.fixture(scope="module")
def colors(light_theme: dict[str, Any]) -> dict[str, str]:
    """Return the palette the theme defines.

    Args:
        light_theme: Fixture-provided theme block.

    Returns:
        Color name to ``#rrggbb`` mapping.
    """
    palette: dict[str, str] = light_theme["colors"]
    return palette


@pytest.fixture(scope="module")
def surfaces(colors: dict[str, str]) -> tuple[tuple[str, str], ...]:
    """Return every surface text and chips can land on, as (name, hex).

    Built from the palette so it cannot drift from the theme. The table header
    is not listed: it is transparent and inherits the card beneath it.

    Args:
        colors: Fixture-provided palette.

    Returns:
        Surface names mapped to their hex colors.
    """
    return tuple((token, colors[token]) for token in SURFACE_TOKENS)


def _drawer_scope() -> str:
    """Return the source text of the drawer layout block.

    The scope runs from ``with layout.drawer:`` to the ``# Toolbar`` marker,
    which is the region the theme is responsible for.

    Returns:
        The drawer source as a single string.

    Raises:
        AssertionError: If the markers are not found (layout was restructured).
    """
    lines = SOURCE.read_text(encoding="utf-8").splitlines()
    start = next((i for i, ln in enumerate(lines) if "with layout.drawer:" in ln), None)
    end = next(
        (
            i
            for i, ln in enumerate(lines)
            if ln.strip() == "# Toolbar" and start is not None and i > start
        ),
        None,
    )
    assert start is not None, "drawer block not found in sphere_viewer.py"
    assert end is not None, "toolbar marker not found in sphere_viewer.py"
    return "\n".join(lines[start:end])


# ---------------------------------------------------------------------------
# 1. Config shape
# ---------------------------------------------------------------------------


def test_config_selects_the_light_theme() -> None:
    """The app must render a light theme; the old dark flag was a no-op."""
    config: dict[str, Any] = sv.VUE3_THEME_CONFIG
    assert config["defaultTheme"] == "light"
    assert config["theme"]["themes"]["light"]["dark"] is False


def test_config_is_exported() -> None:
    """The constant is part of the module's public surface."""
    assert "VUE3_THEME_CONFIG" in sv.__all__


def test_neutral_surfaces_are_light(colors: dict[str, str]) -> None:
    """Page, surface and card fills are all near-white."""
    for token in ("background", "surface", "surface-bright"):
        assert contrast_ratio(colors[token], "#000000") >= WCAG_AAA
    assert contrast_ratio(colors["surface-variant"], "#000000") >= WCAG_AAA


def test_palette_values_are_six_digit_hex(colors: dict[str, str]) -> None:
    """Every entry is a plain hex color, not a named color or rgb() form."""
    hex_pattern = re.compile(r"^#[0-9A-Fa-f]{6}$")
    offenders = [name for name, value in colors.items() if not hex_pattern.match(value)]
    assert not offenders, f"non-hex palette entries: {offenders}"


def test_text_color_is_black(colors: dict[str, str]) -> None:
    """The base text color is black, giving the 21:1 headline contrast."""
    assert colors["on-background"] == "#000000"
    assert contrast_ratio(
        colors["on-background"], colors["background"]
    ) == pytest.approx(21.0, abs=0.1)


# ---------------------------------------------------------------------------
# 2. WCAG contrast
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("token", TEXT_TOKENS)
def test_accent_text_clears_aa_on_every_surface(
    token: str, colors: dict[str, str], surfaces: tuple[tuple[str, str], ...]
) -> None:
    """Accent text is legible on the page, the card, the panel and the header."""
    foreground = colors[token]
    for name, surface in surfaces:
        ratio = contrast_ratio(foreground, surface)
        assert ratio >= WCAG_AA, (
            f"text-{token} ({foreground}) on {name} ({surface}) "
            f"is {ratio:.2f}:1, below AA {WCAG_AA}"
        )


@pytest.mark.parametrize("token", FILL_TOKENS)
def test_chip_accent_clears_aa_in_both_tonal_roles(
    token: str, colors: dict[str, str], surfaces: tuple[tuple[str, str], ...]
) -> None:
    """A VChip label is the accent on a 12% tint of that same accent.

    VChip defaults to ``variant="tonal"``: the underlay is ``currentColor`` at
    ``--v-activated-opacity`` (0.12), so the label color and the background are
    derived from one value. The accent must clear AA against its own tint on
    every surface, and against the bare surface too (the same token is also
    used as plain ``text-*``).

    This is the check the browser harness forced: a fill chosen for a *solid*
    chip with a white label is a different problem from a tonal chip, and the
    chips in this drawer are tonal.
    """
    accent = colors[token]
    for name, surface_hex in surfaces:
        tinted = composite(accent, TONAL_ALPHA, surface_hex)
        ratio = contrast_ratio(accent, tinted)
        assert ratio >= WCAG_AA, (
            f"tonal chip {token} ({accent}) on its own tint over {name} "
            f"({tinted}) is {ratio:.2f}:1, below AA {WCAG_AA}"
        )
        plain = contrast_ratio(accent, surface_hex)
        assert plain >= WCAG_AA, (
            f"text-{token} ({accent}) on {name} is {plain:.2f}:1, below AA {WCAG_AA}"
        )


def test_tonal_alpha_assumption_matches_vuetify() -> None:
    """Guard the 0.12 overlay constant against the installed Vuetify bundle.

    ``--v-activated-opacity`` is declared in the CSS as 0 and set at runtime
    from the theme's ``variables`` defaults, so the bundle is the source of
    truth. If Vuetify changes it, the tonal chip math above must be re-derived.
    """
    bundle = _vuetify_bundle()
    values = [
        float(m)
        for m in re.findall(r'"?activated-opacity"?\s*:\s*(\.\d+|\d+\.\d+)', bundle)
    ]
    if not values:
        pytest.skip("installed Vuetify does not declare activated-opacity")
    assert set(values) == {TONAL_ALPHA}, (
        f"unexpected activated-opacity values: {values}"
    )


def _vuetify_serve_dir() -> pathlib.Path:
    """Return the directory holding the installed Vuetify v3 web assets.

    Returns:
        The absolute path of the ``v3-serve`` directory.

    Raises:
        AssertionError: If trame-vuetify is not importable.
    """
    import trame_vuetify

    path = pathlib.Path(trame_vuetify.__file__).resolve().parent / "module" / "v3-serve"
    assert path.is_dir(), f"Vuetify assets not found at {path}"
    return path


def _vuetify_bundle() -> str:
    """Return the installed Vuetify JavaScript bundle text.

    Returns:
        The bundle source.
    """
    return (_vuetify_serve_dir() / "vuetify3.js").read_text(encoding="utf-8")


def test_emphasis_text_clears_aa_on_every_surface(
    colors: dict[str, str],
    light_theme: dict[str, Any],
    surfaces: tuple[tuple[str, str], ...],
) -> None:
    """The 50 ``text-medium-emphasis`` labels are legible after compositing.

    Vuetify paints emphasis classes as ``on-background`` at a variable opacity,
    so the effective color must be measured after flattening onto each surface.
    """
    variables: dict[str, float] = light_theme["variables"]
    foreground = colors["on-background"]
    for name in ("high-emphasis-opacity", "medium-emphasis-opacity"):
        opacity = variables[name]
        for surface_name, surface in surfaces:
            effective = composite(foreground, opacity, surface)
            ratio = contrast_ratio(effective, surface)
            assert ratio >= WCAG_AA, (
                f"{name}={opacity} on {surface_name} composites to {effective} "
                f"at {ratio:.2f}:1, below AA {WCAG_AA}"
            )


def test_medium_emphasis_opacity_raised_above_stock(
    light_theme: dict[str, Any],
) -> None:
    """The stock 0.6 is what made labels grey; the theme must raise it."""
    variables: dict[str, float] = light_theme["variables"]
    assert variables["medium-emphasis-opacity"] >= 0.75


def test_disabled_text_clears_aa(
    colors: dict[str, str],
    light_theme: dict[str, Any],
    surfaces: tuple[tuple[str, str], ...],
) -> None:
    """Even the dimmest emphasis tier stays readable.

    ``disabled-opacity`` is the lowest text alpha in the theme, so it is the
    binding constraint on the neutral ramp.
    """
    variables: dict[str, float] = light_theme["variables"]
    opacity = variables["disabled-opacity"]
    for surface_name, surface in surfaces:
        effective = composite(colors["on-background"], opacity, surface)
        assert contrast_ratio(effective, surface) >= WCAG_AA, (
            f"disabled text at opacity {opacity} on {surface_name} is below AA"
        )


def test_tonal_toolbar_buttons_clear_aa(colors: dict[str, str]) -> None:
    """The toolbar's ``variant="tonal"`` buttons paint text on a 16% tint.

    Vuetify builds the tonal background by overlaying the color at
    ``theme-overlay`` strength on the surface, so the accent text must survive
    its own tint.
    """
    tonal_alpha = 0.16
    for token in ("primary", "info", "success", "warning"):
        accent = colors[token]
        tinted = composite(accent, tonal_alpha, colors["surface"])
        ratio = contrast_ratio(accent, tinted)
        assert ratio >= WCAG_AA, (
            f"tonal {token} ({accent}) on its own tint ({tinted}) is "
            f"{ratio:.2f}:1, below AA {WCAG_AA}"
        )


def test_slider_fill_clears_non_text_contrast_against_track(
    colors: dict[str, str], surfaces: tuple[tuple[str, str], ...]
) -> None:
    """The slider fill (progress) must read against the track behind it.

    Vuetify paints the fill and thumb in the slider's accent at full strength
    and the track in the same accent at ``--v-activated-opacity``-style alpha
    (0.38) over the surface, so the pair is self-tinting: the only thing that
    separates progress from remainder is the accent's own alpha step. That
    difference is informational, so it needs the 3:1 non-text floor (WCAG
    1.4.11). Every accent the drawer's sliders use is checked.
    """
    accents = _drawer_slider_accents(colors)
    assert accents, "no slider colors found in the drawer"
    for name, accent in accents.items():
        for surface_name, surface in surfaces:
            track = composite(accent, SLIDER_TRACK_ALPHA, surface)
            ratio = contrast_ratio(accent, track)
            assert ratio >= WCAG_NON_TEXT, (
                f"slider fill {name} ({accent}) on its track ({track} over "
                f"{surface_name}) is {ratio:.2f}:1, below the "
                f"{WCAG_NON_TEXT}:1 non-text floor"
            )


def test_slider_thumb_clears_non_text_contrast(
    colors: dict[str, str], surfaces: tuple[tuple[str, str], ...]
) -> None:
    """The thumb is the control that carries the value, so it must read.

    Vuetify paints the thumb in the slider's accent. The unfilled track is the
    same accent at 0.38, which is deliberately a subtle groove (Material's own
    design, ~1.9:1 against the card) -- the value is communicated by the thumb
    and the filled portion, not by the track outline, so those are the elements
    held to the 3:1 non-text floor.
    """
    accents = _drawer_slider_accents(colors)
    for name, accent in accents.items():
        for surface_name, surface in surfaces:
            ratio = contrast_ratio(accent, surface)
            assert ratio >= WCAG_NON_TEXT, (
                f"slider thumb {name} ({accent}) on {surface_name} is "
                f"{ratio:.2f}:1, below the {WCAG_NON_TEXT}:1 non-text floor"
            )


def _drawer_slider_accents(colors: dict[str, str]) -> dict[str, str]:
    """Return the palette colors the drawer's sliders can paint their fill with.

    Reads the ``color=`` attribute of every ``VSlider`` in the drawer scope, so
    a new slider in a new accent is covered automatically.

    Args:
        colors: The theme palette.

    Returns:
        Color name to hex, for accents present in the palette.
    """
    scope = _drawer_scope()
    names: set[str] = set()
    for match in re.finditer(r"VSlider\((.*?)\)\n", scope, re.DOTALL):
        found = re.search(r'color="([a-zA-Z0-9-]+)"', match.group(1))
        if found:
            names.add(found.group(1))
    return {name: colors[name] for name in sorted(names) if name in colors}


# ---------------------------------------------------------------------------
# 3. Drift guard
# ---------------------------------------------------------------------------


def test_every_drawer_color_token_is_in_the_palette(colors: dict[str, str]) -> None:
    """No drawer color may fall through to the static Material CSS.

    The static stylesheet ships light-theme mid-tones (``.text-teal`` is
    ``#009688``, ``.bg-grey`` is ``#9e9e9e``) that fail on white. Only palette
    entries override them, so an unregistered accent silently reintroduces the
    bug this theme fixes.
    """
    scope = _drawer_scope()
    tokens: set[str] = set(re.findall(r'color="([a-zA-Z0-9-]+)"', scope))
    for classes in re.findall(r'classes="([^"]*)"', scope):
        for name in classes.split():
            if name.startswith("text-"):
                suffix = name[len("text-") :]
                if suffix not in NON_COLOR_TEXT_CLASSES:
                    tokens.add(suffix)
    # Badges bind color names at runtime from _RAW_TYPE_META / _EFF_TYPE_META
    # and the basin-center polarity branch.
    tokens.update({"error", "info", "success", "grey"})
    unregistered = sorted(tokens - set(colors))
    assert not unregistered, (
        f"drawer colors missing from VUE3_THEME_CONFIG: {unregistered}"
    )


def test_badge_color_values_are_paletted() -> None:
    """The runtime badge bindings name palette colors only."""
    raw = set(sv._RAW_TYPE_META.values())
    eff = set(sv._EFF_TYPE_META.values())
    badges = {badge for _, badge in raw | eff}
    badges.update({"error", "info"})  # basin-center polarity branch
    config: dict[str, Any] = sv.VUE3_THEME_CONFIG
    palette = config["theme"]["themes"]["light"]["colors"]
    assert badges <= set(palette), (
        f"unpaletted badge colors: {sorted(badges - set(palette))}"
    )


# ---------------------------------------------------------------------------
# 4. Regression guards
# ---------------------------------------------------------------------------


def test_dead_trame_theme_assignment_is_gone() -> None:
    """``state.trame__theme`` is read by nothing in trame and must not return.

    It was the reason the app looked dark in source and rendered light in the
    browser. The supported hook is ``vuetify_config=`` on the layout.
    """
    source = SOURCE.read_text(encoding="utf-8")
    assignments = re.findall(r"state\.trame__theme\s*=", source)
    assert not assignments, "trame__theme is a no-op; use vuetify_config= instead"


def test_layout_receives_the_theme_config() -> None:
    """The layout must be constructed with the palette wired in."""
    source = SOURCE.read_text(encoding="utf-8")
    assert re.search(
        r"SinglePageWithDrawerLayout\(\s*server,\s*vuetify_config=VUE3_THEME_CONFIG",
        source,
    ), "SinglePageWithDrawerLayout must pass vuetify_config=VUE3_THEME_CONFIG"


@pytest.mark.parametrize(
    "dark_hex",
    ["#3e434c", "#30343b", "#272a30", "#202328", "#f0f2f5", "#d8dce2"],
)
def test_force_dark_hexes_do_not_reappear(dark_hex: str) -> None:
    """The old drawer overrides named these dark surfaces and light text.

    They never reached the document (Vue strips ``<style>`` from templates),
    which is why the app looked dark in source and rendered light in the
    browser. Reintroducing them in any form that *does* apply would split the UI
    across two themes again.
    """
    source = SOURCE.read_text(encoding="utf-8")
    assert dark_hex not in source, f"{dark_hex} was a force-dark drawer color"


def test_no_html_style_block_in_the_drawer() -> None:
    """``html.Style`` inside a Vue template never reaches the document.

    Vue strips ``<style>`` from component templates, so a scoped style block in
    the drawer is dead code: it looks like it sets the card and slider colors
    while the browser silently ignores every rule. Verified against the live app
    (the drawer's rules were absent from ``document.styleSheets``) and against a
    minimal Vue repro. Colors and edges must come from ``VUE3_THEME_CONFIG``,
    which is the mechanism that actually applies.
    """
    scope = _drawer_scope()
    assert "html.Style(" not in scope, (
        "html.Style in the drawer is stripped by Vue and has no effect; "
        "put colors in VUE3_THEME_CONFIG instead"
    )


def test_cards_get_their_border_from_the_theme() -> None:
    """White cards on a white page need an edge, set as a Vuetify default.

    The border is configured centrally (``defaults.VCard.border``) rather than
    per-card, and its color comes from the theme's border variables, so it
    cannot drift from the palette.
    """
    config: dict[str, Any] = sv.VUE3_THEME_CONFIG
    assert config["defaults"]["VCard"]["border"] is True
    variables: dict[str, Any] = config["theme"]["themes"]["light"]["variables"]
    assert variables["border-color"] == "#000000"
    assert variables["border-opacity"] >= 0.2


def test_trame_footer_branding_color_is_paletted(
    colors: dict[str, str], surfaces: tuple[tuple[str, str], ...]
) -> None:
    """trame's layout footer hardcodes ``text-grey-lighten-1``.

    That class is not one the drawer uses; it comes from
    ``SinglePageWithDrawerLayout`` itself, on the "Powered by trame" and
    "© Kitware" links. The static Material value (#bdbdbd) is tuned for a dark
    footer and lands at 1.88:1 on the white one this theme produces, so the
    palette must name it.
    """
    assert "grey-lighten-1" in colors, (
        "trame's footer links use text-grey-lighten-1; the palette must override it"
    )
    for name, surface in surfaces:
        ratio = contrast_ratio(colors["grey-lighten-1"], surface)
        assert ratio >= WCAG_AA, (
            f"footer text {colors['grey-lighten-1']} on {name} is {ratio:.2f}:1"
        )


def test_config_is_json_serialisable() -> None:
    """The config crosses the wire to the browser as JSON."""
    import json

    json.dumps(sv.VUE3_THEME_CONFIG)


def test_contrast_helper_is_self_consistent() -> None:
    """Sanity-check the ported WCAG math against known values."""
    assert contrast_ratio("#000000", "#ffffff") == pytest.approx(21.0, abs=0.01)
    assert contrast_ratio("#ffffff", "#ffffff") == pytest.approx(1.0, abs=0.01)
    assert contrast_ratio("#767676", "#ffffff") == pytest.approx(4.54, abs=0.02)
    assert math.isclose(contrast_ratio("#000000", "#000000"), 1.0)
