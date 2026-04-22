"""
Optional Seismo satellite accent color (magnitu_status.accent_color).

Magnitu only applies colours after strict validation; unknown or missing
values fall back to default (black) styling — no impact on core behaviour.
"""
import re
from typing import Any, Dict, Optional

_HEX6 = re.compile(r"^#[0-9a-fA-F]{6}$")
_HEX3 = re.compile(r"^#[0-9a-fA-F]{3}$")


def parse_accent_hex_string(raw: Any) -> Optional[str]:
    """Return lowercase #rrggbb or None if missing / invalid."""
    if raw is None:
        return None
    if not isinstance(raw, str):
        return None
    s = raw.strip()
    if _HEX6.match(s):
        return s.lower()
    if _HEX3.match(s):
        a, b, c = s[1], s[2], s[3]
        return "#{}{}{}{}{}{}".format(a, a, b, b, c, c).lower()
    return None


def _parse_accent_nested(obj: Any, depth: int, max_depth: int) -> Optional[str]:
    """Find first valid accent_color hex at any nesting depth (bounded)."""
    if depth > max_depth or not isinstance(obj, dict):
        return None
    nested = parse_accent_hex_string(obj.get("accent_color")) or \
             parse_accent_hex_string(obj.get("brand_accent")) or \
             parse_accent_hex_string(obj.get("SEISMO_BRAND_ACCENT"))
    if nested:
        return nested
    for v in obj.values():
        if isinstance(v, dict):
            found = _parse_accent_nested(v, depth + 1, max_depth)
            if found:
                return found
    return None


def parse_accent_from_magnitu_status(status: Dict[str, Any]) -> Optional[str]:
    """Read optional accent_color from a magnitu_status JSON object.

    Supports top-level ``accent_color``, shallow wrappers (``data``, ``payload``,
    etc.), then a bounded deep scan — some Seismo builds nest branding several
    levels deep.
    """
    if not isinstance(status, dict):
        return None

    direct = parse_accent_hex_string(status.get("accent_color")) or \
             parse_accent_hex_string(status.get("brand_accent")) or \
             parse_accent_hex_string(status.get("SEISMO_BRAND_ACCENT"))
    if direct:
        return direct

    nested_keys = (
        "data",
        "payload",
        "config",
        "magnitu",
        "meta",
        "response",
        "result",
        "satellite",
        "seismo",
        "theme",
        "branding",
    )
    for key in nested_keys:
        inner = status.get(key)
        if isinstance(inner, dict):
            nested = parse_accent_hex_string(inner.get("accent_color")) or \
                     parse_accent_hex_string(inner.get("brand_accent")) or \
                     parse_accent_hex_string(inner.get("SEISMO_BRAND_ACCENT"))
            if nested:
                return nested

    return _parse_accent_nested(status, 0, 8)


def contrast_text_on_accent(accent_hex6: str) -> str:
    """Black or white text for readability on accent_hex6 (#rrggbb)."""
    if len(accent_hex6) != 7 or accent_hex6[0] != "#":
        return "#000000"
    try:
        r = int(accent_hex6[1:3], 16)
        g = int(accent_hex6[3:5], 16)
        b = int(accent_hex6[5:7], 16)
    except ValueError:
        return "#000000"
    luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255.0
    return "#000000" if luminance > 0.55 else "#ffffff"


def safe_accent_for_profile(stored: Any) -> Optional[str]:
    """Re-validate DB value before use in HTML/CSS."""
    return parse_accent_hex_string(stored)


def get_theme_colors(hex_color: Optional[str]) -> Dict[str, str]:
    """Generate all theme variants from a base accent hex.
    
    If hex_color is None, returns default Seismo red theme.
    """
    base = parse_accent_hex_string(hex_color) or "#FF6B6B"
    
    # Calculate RGB for HSL manipulation
    try:
        r = int(base[1:3], 16)
        g = int(base[3:5], 16)
        b = int(base[5:7], 16)
    except (ValueError, IndexError):
        r, g, b = 255, 107, 107 # Fallback to Seismo red

    # Subtle background: high desaturation, extremely high lightness
    # We'll use RGBA with 0.05 opacity for the desaturated feel or just calculate desaturated light color
    # Let's go with a very light tinted version:
    subtle = "rgba(%d, %d, %d, 0.06)" % (r, g, b)
    
    # Border: slightly darker or desaturated version of base
    # For now, let's just use the base for borders or slightly darken it if needed.
    # But often the base itself is perfect for active state borders.
    
    return {
        "bg": base,
        "fg": contrast_text_on_accent(base),
        "subtle": subtle,
        "border": base,
    }
