"""Shared color palette and CSS theme for the Panel application."""

ALASKA_PRIMARY = "#BF4E6C"
ALASKA_SECONDARY = "#F2676B"
ALASKA_YELLOW = "#F5CC70"
ALASKA_BLUE = "#5992C7"
ALASKA_NAVY = "#325982"
LIGHT_GRAY = "#F0F0F0"

ALASKA_PALETTE = [
    ALASKA_PRIMARY,
    ALASKA_SECONDARY,
    ALASKA_YELLOW,
    ALASKA_BLUE,
    ALASKA_NAVY,
]

THEME_CSS = f"""
html, body {{
    background-color: {LIGHT_GRAY} !important;
    color: {ALASKA_NAVY};
}}
.bk-root {{
    background-color: transparent !important;
    color: {ALASKA_NAVY};
}}
.pnx-header {{
    background-color: {ALASKA_NAVY} !important;
    color: {LIGHT_GRAY} !important;
    border-bottom: 2px solid {ALASKA_PRIMARY};
}}
.pnx-main, .pnx-content {{
    background-color: {LIGHT_GRAY} !important;
}}
.pnx-sidebar {{
    background-color: {ALASKA_YELLOW} !important;
}}
.pnx-sidebar .bk-panel, .pnx-sidebar .bk-btn {{
    background-color: transparent !important;
}}
.bk-btn-primary {{
    background-color: {ALASKA_PRIMARY} !important;
    color: {LIGHT_GRAY} !important;
}}
.bk-btn-warning {{
    background-color: {ALASKA_YELLOW} !important;
    color: {ALASKA_NAVY} !important;
}}
.bk-btn-success {{
    background-color: {ALASKA_BLUE} !important;
    color: {LIGHT_GRAY} !important;
}}
.bk-alert-danger {{
    background-color: {ALASKA_SECONDARY} !important;
    color: {LIGHT_GRAY} !important;
    border: none !important;
}}
.bk-progress .bk-bar {{
    background-color: {ALASKA_PRIMARY} !important;
}}
.bk-switch-toggle.active, .bk-switch-input:checked + .bk-switch-toggle {{
    background-color: {ALASKA_PRIMARY} !important;
}}
"""

__all__ = [
    "ALASKA_PRIMARY",
    "ALASKA_SECONDARY",
    "ALASKA_YELLOW",
    "ALASKA_BLUE",
    "ALASKA_NAVY",
    "LIGHT_GRAY",
    "ALASKA_PALETTE",
    "THEME_CSS",
]

