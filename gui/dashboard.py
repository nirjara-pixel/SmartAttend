"""
SmartAttend - Main Dashboard & Desktop Application
Coordinates the professional desktop GUI, sidebar navigation, real-time KPI metric cards,
automated statistics refresh, and view routing across all application modules.
"""

from datetime import datetime
import tkinter as tk
from tkinter import messagebox, ttk
from typing import Dict, Optional

import config
from database import DatabaseManager, db as default_db
from gui.attendance_view import AttendanceView
from gui.recognition_view import RecognitionView
from gui.register import RegistrationView
from gui.theme import COLORS, FONTS, configure_ttk_styles
from gui.train_view import TrainViewModel
from logger import get_logger

logger = get_logger("SmartAttend.Dashboard")


class OverviewView(tk.Frame):
    """
    Dashboard Overview Home Screen containing KPI summary cards,
    quick-action shortcuts, and a recent attendance activity log.
    """

    def __init__(
        self,
        parent: tk.Widget,
        db_manager: DatabaseManager,
        on_navigate: Optional[any] = None
    ):
        super().__init__(parent, bg=COLORS["bg_main"])
        self.db = db_manager
        self.on_navigate = on_navigate

        self._build_ui()
        self.refresh_stats()

    def _build_ui(self) -> None:
        """Construct the overview cards and recent table."""
        # Header Banner
        header = tk.Frame(self, bg=COLORS["bg_main"])
        header.pack(fill="x", padx=25, pady=(20, 15))

        title = tk.Label(
            header,
            text="System Overview & Analytics",
            font=FONTS["page_title"],
            fg=COLORS["text_primary"],
            bg=COLORS["bg_main"]
        )
        title.pack(anchor="w")

        today_str = datetime.now().strftime("%A, %B %d, %Y")
        self.lbl_subtitle = tk.Label(
            header,
            text=f"Live institutional metrics for {today_str}",
            font=FONTS["body"],
            fg=COLORS["text_secondary"],
            bg=COLORS["bg_main"]
        )
        self.lbl_subtitle.pack(anchor="w", pady=(2, 0))

        # ---------------- KPI Statistics Cards Row ----------------
        cards_frame = tk.Frame(self, bg=COLORS["bg_main"])
        cards_frame.pack(fill="x", padx=25, pady=(0, 20))

        # 1. Total Registered Students Card
        self.val_total, _ = self._create_kpi_card(
            cards_frame,
            title="TOTAL STUDENTS",
            accent_color=COLORS["primary"],
            icon="👥"
        )

        # 2. Present Today Card
        self.val_present, _ = self._create_kpi_card(
            cards_frame,
            title="PRESENT TODAY",
            accent_color=COLORS["success"],
            icon="✓"
        )

        # 3. Absent Today Card
        self.val_absent, _ = self._create_kpi_card(
            cards_frame,
            title="ABSENT TODAY",
            accent_color=COLORS["warning"],
            icon="⏳"
        )

        # 4. Attendance Percentage Card
        self.val_percentage, _ = self._create_kpi_card(
            cards_frame,
            title="ATTENDANCE RATE",
            accent_color=COLORS["purple"],
            icon="📈"
        )

        # ---------------- Quick Actions & Recent Activity ----------------
        body_split = tk.Frame(self, bg=COLORS["bg_main"])
        body_split.pack(fill="both", expand=True, padx=25, pady=(0, 20))

        # Left: Quick Actions Card
        actions_card = tk.Frame(
            body_split,
            bg=COLORS["card_bg"],
            highlightbackground=COLORS["card_border"],
            highlightthickness=1,
            padx=20,
            pady=20
        )
        actions_card.pack(side="left", fill="both", expand=False, padx=(0, 15))
        actions_card.config(width=280)
        actions_card.pack_propagate(False)

        tk.Label(
            actions_card,
            text="Quick Actions",
            font=FONTS["section_title"],
            fg=COLORS["text_primary"],
            bg=COLORS["card_bg"]
        ).pack(anchor="w", pady=(0, 15))

        self._create_action_button(
            actions_card,
            "▶ Start Recognition",
            COLORS["success"],
            lambda: self.on_navigate("recognition") if self.on_navigate else None
        )
        self._create_action_button(
            actions_card,
            "➕ Register Student",
            COLORS["primary"],
            lambda: self.on_navigate("register") if self.on_navigate else None
        )
        self._create_action_button(
            actions_card,
            "📋 View Attendance",
            COLORS["sidebar_bg"],
            lambda: self.on_navigate("attendance") if self.on_navigate else None
        )
        self._create_action_button(
            actions_card,
            "⚡ Model Calibration",
            COLORS["purple"],
            lambda: self.on_navigate("train") if self.on_navigate else None
        )

        # Right: Today's Recent Check-ins
        recent_card = tk.Frame(
            body_split,
            bg=COLORS["card_bg"],
            highlightbackground=COLORS["card_border"],
            highlightthickness=1,
            padx=20,
            pady=20
        )
        recent_card.pack(side="right", fill="both", expand=True)

        recent_header = tk.Frame(recent_card, bg=COLORS["card_bg"])
        recent_header.pack(fill="x", pady=(0, 12))

        tk.Label(
            recent_header,
            text="Today's Recent Check-ins",
            font=FONTS["section_title"],
            fg=COLORS["text_primary"],
            bg=COLORS["card_bg"]
        ).pack(side="left")

        btn_refresh = tk.Button(
            recent_header,
            text="🔄 Refresh",
            font=FONTS["caption"],
            bg=COLORS["bg_main"],
            fg=COLORS["text_secondary"],
            relief="flat",
            cursor="hand2",
            padx=10,
            pady=4,
            command=self.refresh_stats
        )
        btn_refresh.pack(side="right")

        # Mini Recent Treeview
        columns = ("time", "student_id", "name", "course", "confidence")
        self.tree_recent = ttk.Treeview(
            recent_card,
            columns=columns,
            show="headings",
            style="Modern.Treeview",
            height=7
        )

        self.tree_recent.heading("time", text="Time")
        self.tree_recent.heading("student_id", text="Student ID")
        self.tree_recent.heading("name", text="Student Name")
        self.tree_recent.heading("course", text="Course")
        self.tree_recent.heading("confidence", text="Confidence")

        self.tree_recent.column("time", width=90, anchor="center")
        self.tree_recent.column("student_id", width=110, anchor="center")
        self.tree_recent.column("name", width=180, anchor="w")
        self.tree_recent.column("course", width=150, anchor="w")
        self.tree_recent.column("confidence", width=100, anchor="center")

        self.tree_recent.pack(fill="both", expand=True)

    def _create_kpi_card(
        self,
        parent: tk.Widget,
        title: str,
        accent_color: str,
        icon: str
    ) -> tuple[tk.Label, tk.Frame]:
        """Create a card displaying a single KPI metric."""
        card = tk.Frame(
            parent,
            bg=COLORS["card_bg"],
            highlightbackground=COLORS["card_border"],
            highlightthickness=1,
            padx=18,
            pady=16
        )
        card.pack(side="left", fill="both", expand=True, padx=5)

        # Top icon and title row
        top_row = tk.Frame(card, bg=COLORS["card_bg"])
        top_row.pack(fill="x")

        tk.Label(
            top_row,
            text=title,
            font=FONTS["card_label"],
            fg=COLORS["text_secondary"],
            bg=COLORS["card_bg"]
        ).pack(side="left")

        tk.Label(
            top_row,
            text=icon,
            font=("Segoe UI", 12),
            fg=accent_color,
            bg=COLORS["card_bg"]
        ).pack(side="right")

        # Value label
        val_lbl = tk.Label(
            card,
            text="0",
            font=FONTS["card_value"],
            fg=accent_color,
            bg=COLORS["card_bg"]
        )
        val_lbl.pack(anchor="w", pady=(8, 0))

        return val_lbl, card

    def _create_action_button(
        self,
        parent: tk.Widget,
        text: str,
        bg_color: str,
        command: any
    ) -> None:
        """Create a styled button for the quick actions panel."""
        btn = tk.Button(
            parent,
            text=text,
            font=FONTS["button"],
            bg=bg_color,
            fg=COLORS["text_light"],
            activebackground=COLORS["primary_hover"],
            activeforeground=COLORS["text_light"],
            relief="flat",
            cursor="hand2",
            pady=10,
            command=command
        )
        btn.pack(fill="x", pady=5)

    def refresh_stats(self) -> None:
        """Fetch updated statistics from SQLite and refresh cards and recent table."""
        try:
            stats = self.db.get_dashboard_stats()
            self.val_total.config(text=str(stats["total_students"]))
            self.val_present.config(text=str(stats["present_today"]))
            self.val_absent.config(text=str(stats["absent_today"]))
            self.val_percentage.config(text=f"{stats['attendance_percentage']}%")

            # Update recent table
            today_records = self.db.get_today_attendance()
            for row in self.tree_recent.get_children():
                self.tree_recent.delete(row)

            # Display up to 10 recent check-ins
            for r in today_records[:10]:
                raw_conf = r.get("confidence")
                conf_val = float(raw_conf) if raw_conf is not None else 0.0
                self.tree_recent.insert(
                    "",
                    "end",
                    values=(
                        r.get("time", ""),
                        r.get("student_id", ""),
                        r.get("name", ""),
                        r.get("course", ""),
                        f"{conf_val:.1f}%"
                    )
                )

        except Exception as e:
            logger.warning(f"Could not refresh dashboard stats: {e}")


class SmartAttendApp(tk.Tk):
    """
    Main Tkinter Application window for SmartAttend.
    Orchestrates the sidebar navigation, views switching, and periodic refresh.
    """

    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        super().__init__()
        self.db = db_manager or default_db
        self.is_running_app = True
        self._refresh_timer: Optional[str] = None

        self.title("SmartAttend – Face Recognition Attendance Management System")

        # Responsive screen geometry and centering
        self.update_idletasks()
        screen_w = self.winfo_screenwidth()
        screen_h = self.winfo_screenheight()

        # Ensure window fits safely inside the screen display bounds
        win_w = min(1180, max(920, int(screen_w * 0.90)))
        win_h = min(700, max(560, int(screen_h * 0.85)))
        x = max(0, (screen_w - win_w) // 2)
        y = max(0, (screen_h - win_h) // 2)

        self.geometry(f"{win_w}x{win_h}+{x}+{y}")
        self.minsize(min(900, win_w), min(540, win_h))
        self.configure(bg=COLORS["bg_main"])

        # Configure modern ttk widgets
        configure_ttk_styles(self)

        # Intercept window close event for graceful cleanup
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        # Navigation State
        self.current_view_key = "overview"
        self.views: Dict[str, tk.Frame] = {}
        self.nav_buttons: Dict[str, tk.Button] = {}

        self._build_shell()
        self._initialize_views()
        self.show_view("overview")

        # Bring window to foreground and gain focus
        self.deiconify()
        self.lift()
        self.focus_force()

        # Auto-refresh statistics periodically
        self._schedule_auto_refresh()

        logger.info(f"SmartAttendApp initialized successfully (Geometry: {win_w}x{win_h}+{x}+{y}).")

    def _build_shell(self) -> None:
        """Create sidebar and main container layout."""
        # ----------------- Left Sidebar -----------------
        self.sidebar = tk.Frame(self, bg=COLORS["sidebar_bg"], width=240)
        self.sidebar.pack(side="left", fill="y")
        self.sidebar.pack_propagate(False)

        # Branding Header
        brand_frame = tk.Frame(self.sidebar, bg=COLORS["sidebar_bg"], padx=20, pady=24)
        brand_frame.pack(fill="x")

        tk.Label(
            brand_frame,
            text="⚡ SmartAttend",
            font=FONTS["app_title"],
            fg=COLORS["text_light"],
            bg=COLORS["sidebar_bg"]
        ).pack(anchor="w")

        tk.Label(
            brand_frame,
            text="Face Recognition System",
            font=FONTS["caption"],
            fg=COLORS["text_muted"],
            bg=COLORS["sidebar_bg"]
        ).pack(anchor="w", pady=(2, 0))

        # Thin separator line
        sep = tk.Frame(self.sidebar, bg=COLORS["sidebar_border"], height=1)
        sep.pack(fill="x", padx=15, pady=(0, 15))

        # Navigation Buttons Container
        self.nav_container = tk.Frame(self.sidebar, bg=COLORS["sidebar_bg"])
        self.nav_container.pack(fill="x", padx=12)

        nav_items = [
            ("overview", "📊 Dashboard", self._nav_overview),
            ("register", "👤 Student Registration", self._nav_register),
            ("recognition", "🎥 Live Recognition", self._nav_recognition),
            ("attendance", "📋 Attendance Records", self._nav_attendance),
            ("train", "⚙️ Model Calibration", self._nav_train),
        ]

        for key, label, cmd in nav_items:
            btn = tk.Button(
                self.nav_container,
                text=f"  {label}",
                font=FONTS["nav_button"],
                fg=COLORS["sidebar_fg"],
                bg=COLORS["sidebar_bg"],
                activebackground=COLORS["sidebar_hover"],
                activeforeground=COLORS["text_light"],
                relief="flat",
                cursor="hand2",
                anchor="w",
                padx=15,
                pady=11,
                command=cmd
            )
            btn.pack(fill="x", pady=2)
            self.nav_buttons[key] = btn

        # Bottom System Info
        sys_info_frame = tk.Frame(self.sidebar, bg=COLORS["sidebar_bg"], padx=20, pady=20)
        sys_info_frame.pack(side="bottom", fill="x")

        tk.Label(
            sys_info_frame,
            text="SYSTEM STATUS",
            font=FONTS["caption"],
            fg=COLORS["text_muted"],
            bg=COLORS["sidebar_bg"]
        ).pack(anchor="w")

        self.lbl_db_status = tk.Label(
            sys_info_frame,
            text="● SQLite: Connected",
            font=FONTS["caption"],
            fg=COLORS["success"],
            bg=COLORS["sidebar_bg"]
        )
        self.lbl_db_status.pack(anchor="w", pady=(2, 0))

        # ----------------- Right Content Container -----------------
        self.content_container = tk.Frame(self, bg=COLORS["bg_main"])
        self.content_container.pack(side="right", fill="both", expand=True)

    def _initialize_views(self) -> None:
        """Instantiate all view panels inside the content container."""
        # 1. Overview View
        self.views["overview"] = OverviewView(
            self.content_container,
            db_manager=self.db,
            on_navigate=self.show_view
        )

        # 2. Registration View
        self.views["register"] = RegistrationView(
            self.content_container,
            db_manager=self.db,
            on_registration_success=self._on_registration_success
        )

        # 3. Recognition View
        self.views["recognition"] = RecognitionView(
            self.content_container,
            db_manager=self.db,
            on_attendance_marked=self._on_attendance_marked
        )

        # 4. Attendance View
        self.views["attendance"] = AttendanceView(
            self.content_container,
            db_manager=self.db
        )

        # 5. Model Training View
        self.views["train"] = TrainViewModel(
            self.content_container,
            db_manager=self.db,
            on_train_complete=self._on_train_complete
        )

    def show_view(self, view_key: str) -> None:
        """Switch active view and update sidebar button highlights."""
        if view_key not in self.views:
            return

        # If navigating away from recognition view, stop camera stream safely
        if self.current_view_key == "recognition" and view_key != "recognition":
            rec_view: RecognitionView = self.views["recognition"]
            if rec_view._is_running:
                rec_view.stop_recognition()

        # Hide current view
        for v in self.views.values():
            v.pack_forget()

        # Show target view
        target_view = self.views[view_key]
        target_view.pack(fill="both", expand=True)
        self.current_view_key = view_key

        # Update sidebar button styling
        for k, btn in self.nav_buttons.items():
            if k == view_key:
                btn.config(
                    bg=COLORS["sidebar_active"],
                    fg=COLORS["text_light"],
                    font=FONTS["button"]
                )
            else:
                btn.config(
                    bg=COLORS["sidebar_bg"],
                    fg=COLORS["sidebar_fg"],
                    font=FONTS["nav_button"]
                )

        # Refresh stats if navigating to overview or attendance
        if view_key == "overview":
            self.views["overview"].refresh_stats()
        elif view_key == "attendance":
            self.views["attendance"].refresh_records()
        elif view_key == "train":
            self.views["train"].refresh_model_info()

    def _nav_overview(self) -> None:
        self.show_view("overview")

    def _nav_register(self) -> None:
        self.show_view("register")

    def _nav_recognition(self) -> None:
        self.show_view("recognition")

    def _nav_attendance(self) -> None:
        self.show_view("attendance")

    def _nav_train(self) -> None:
        self.show_view("train")

    def _on_registration_success(self) -> None:
        """Handle student registration event."""
        self.views["overview"].refresh_stats()
        self.views["train"].refresh_model_info()

    def _on_attendance_marked(self) -> None:
        """Handle attendance marking event in recognition view."""
        self.views["overview"].refresh_stats()

    def _on_train_complete(self) -> None:
        """Handle model training completion."""
        self.views["recognition"]._check_model_status()

    def report_callback_exception(self, exc, val, tb):
        """Log any uncaught Tkinter callback exception to logs/app.log."""
        logger.error("Unhandled Tkinter callback exception:", exc_info=(exc, val, tb))

    def _schedule_auto_refresh(self) -> None:
        """Periodically refresh statistics every 30 seconds if app is running."""
        if not self.is_running_app:
            return
        try:
            if self.current_view_key == "overview":
                self.views["overview"].refresh_stats()
        except Exception as e:
            logger.warning(f"Error during auto-refresh: {e}")
        finally:
            if self.is_running_app:
                self._refresh_timer = self.after(30000, self._schedule_auto_refresh)

    def on_close(self) -> None:
        """Safely shut down all background threads and close application."""
        logger.info("Application close requested. Releasing resources...")
        self.is_running_app = False

        # Cancel pending auto-refresh timer
        if self._refresh_timer is not None:
            try:
                self.after_cancel(self._refresh_timer)
            except Exception:
                pass
            self._refresh_timer = None

        # Stop recognition thread and release webcam if running
        if "recognition" in self.views:
            try:
                rec_view: RecognitionView = self.views["recognition"]
                if hasattr(rec_view, "_is_running") and rec_view._is_running:
                    rec_view.stop_recognition()
            except Exception as e:
                logger.warning(f"Error stopping recognition view: {e}")

        # Stop registration thread if running
        if "register" in self.views:
            try:
                reg_view: RegistrationView = self.views["register"]
                if hasattr(reg_view, "_is_capturing") and reg_view._is_capturing:
                    reg_view._stop_requested = True
            except Exception as e:
                logger.warning(f"Error stopping registration capture: {e}")

        # Destroy root window cleanly
        try:
            self.destroy()
            logger.info("SmartAttend GUI window destroyed cleanly.")
        except Exception as e:
            logger.warning(f"Error during window destruction: {e}")

    def run(self) -> None:
        """Launch the Tkinter main event loop."""
        self.deiconify()
        self.lift()
        self.state("normal")
        self.mainloop()
        logger.info("SmartAttend mainloop finished.")
