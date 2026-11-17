"""
SmartAttend - GUI Theme and Styling Definitions
Defines color palettes, typography, and ttk styling for the professional desktop UI.
"""

import tkinter as tk
from tkinter import ttk

# Professional Color Palette
COLORS = {
    # Sidebar
    "sidebar_bg": "#0f172a",          # Dark slate / navy
    "sidebar_fg": "#e2e8f0",          # Crisp light text
    "sidebar_hover": "#1e293b",       # Slightly lighter navy
    "sidebar_active": "#2563eb",      # Bright active blue
    "sidebar_border": "#334155",      # Subtle border

    # Content Area
    "bg_main": "#f8fafc",             # Clean light background (slate-50)
    "card_bg": "#ffffff",             # White card background
    "card_border": "#e2e8f0",         # Soft border for cards
    "card_hover": "#f1f5f9",          # Subtle card hover

    # Text
    "text_primary": "#0f172a",        # Slate-900 high contrast
    "text_secondary": "#64748b",      # Slate-500 secondary text
    "text_muted": "#94a3b8",          # Slate-400 placeholder text
    "text_light": "#ffffff",

    # Status & Accents
    "primary": "#2563eb",             # Vibrant blue
    "primary_hover": "#1d4ed8",
    "success": "#10b981",             # Emerald green
    "success_bg": "#ecfdf5",
    "warning": "#f59e0b",             # Amber
    "warning_bg": "#fffbeb",
    "danger": "#ef4444",              # Crimson red
    "danger_bg": "#fef2f2",
    "purple": "#8b5cf6",              # Purple accent
    "purple_bg": "#f5f3ff",

    # Forms & Tables
    "input_bg": "#ffffff",
    "input_border": "#cbd5e1",
    "table_header_bg": "#f1f5f9",
    "table_header_fg": "#334155",
    "table_row_alt": "#f8fafc",
    "table_selected": "#dbeafe",
}

FONTS = {
    "app_title": ("Segoe UI", 16, "bold"),
    "page_title": ("Segoe UI", 18, "bold"),
    "section_title": ("Segoe UI", 13, "bold"),
    "card_label": ("Segoe UI", 10, "bold"),
    "card_value": ("Segoe UI", 26, "bold"),
    "card_sub": ("Segoe UI", 8),
    "body": ("Segoe UI", 10),
    "body_bold": ("Segoe UI", 10, "bold"),
    "caption": ("Segoe UI", 9),
    "button": ("Segoe UI", 10, "bold"),
    "nav_button": ("Segoe UI", 11),
    "status": ("Segoe UI", 9),
}


def configure_ttk_styles(root: tk.Tk) -> ttk.Style:
    """Configure modern ttk styles for treeviews, scrollbars, and widgets."""
    style = ttk.Style(root)
    # Use clam or default engine for full color customization
    try:
        style.theme_use("clam")
    except Exception:
        pass

    # Treeview (Modern Table)
    style.configure(
        "Modern.Treeview",
        background=COLORS["card_bg"],
        foreground=COLORS["text_primary"],
        fieldbackground=COLORS["card_bg"],
        font=FONTS["body"],
        rowheight=32,
        borderwidth=0
    )
    style.configure(
        "Modern.Treeview.Heading",
        background=COLORS["table_header_bg"],
        foreground=COLORS["table_header_fg"],
        font=FONTS["body_bold"],
        padding=(8, 8),
        borderwidth=1,
        relief="flat"
    )
    style.map(
        "Modern.Treeview",
        background=[("selected", COLORS["table_selected"])],
        foreground=[("selected", COLORS["text_primary"])]
    )
    style.map(
        "Modern.Treeview.Heading",
        background=[("active", "#e2e8f0")]
    )

    # Clean Scrollbar
    style.configure(
        "Vertical.TScrollbar",
        background="#cbd5e1",
        troughcolor=COLORS["bg_main"],
        borderwidth=0,
        arrowsize=12
    )

    return style
