"""
SmartAttend - Attendance Records View
Displays attendance records in a modern ttk Treeview with real-time search,
date filtering, CSV report exporting, and record statistics.
"""

from datetime import datetime
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Any, Dict, List, Optional

import config
from attendance_manager import AttendanceManager
from database import DatabaseManager, db as default_db
from gui.theme import COLORS, FONTS


class AttendanceView(tk.Frame):
    """
    Panel for viewing, filtering, searching, and exporting attendance records.
    """

    def __init__(
        self,
        parent: tk.Widget,
        db_manager: Optional[DatabaseManager] = None,
        attendance_manager: Optional[AttendanceManager] = None
    ):
        super().__init__(parent, bg=COLORS["bg_main"])
        self.db = db_manager or default_db
        self.att_mgr = attendance_manager or AttendanceManager(db_manager=self.db)
        self.all_records: List[Dict[str, Any]] = []

        self._build_ui()
        self.refresh_records()

    def _build_ui(self) -> None:
        """Construct the attendance records view layout."""
        # Top Header
        header_frame = tk.Frame(self, bg=COLORS["bg_main"])
        header_frame.pack(fill="x", padx=25, pady=(20, 10))

        title = tk.Label(
            header_frame,
            text="Attendance Records",
            font=FONTS["page_title"],
            fg=COLORS["text_primary"],
            bg=COLORS["bg_main"]
        )
        title.pack(anchor="w")

        subtitle = tk.Label(
            header_frame,
            text="View, search, and export daily and historical student attendance logs.",
            font=FONTS["body"],
            fg=COLORS["text_secondary"],
            bg=COLORS["bg_main"]
        )
        subtitle.pack(anchor="w", pady=(2, 0))

        # Main Table Card Container
        table_card = tk.Frame(
            self,
            bg=COLORS["card_bg"],
            highlightbackground=COLORS["card_border"],
            highlightthickness=1,
            padx=20,
            pady=20
        )
        table_card.pack(fill="both", expand=True, padx=25, pady=10)

        # Filter & Action Toolbar
        toolbar = tk.Frame(table_card, bg=COLORS["card_bg"])
        toolbar.pack(fill="x", pady=(0, 15))

        # Search Bar
        search_frame = tk.Frame(toolbar, bg=COLORS["card_bg"])
        search_frame.pack(side="left", fill="x", expand=True)

        tk.Label(
            search_frame,
            text="🔍 Search:",
            font=FONTS["body_bold"],
            fg=COLORS["text_secondary"],
            bg=COLORS["card_bg"]
        ).pack(side="left", padx=(0, 6))

        self.search_var = tk.StringVar()
        self.search_var.trace_add("write", lambda *args: self._filter_records())
        self.ent_search = ttk.Entry(search_frame, textvariable=self.search_var, font=FONTS["body"], width=28)
        self.ent_search.pack(side="left", padx=(0, 15))

        # Date Filter
        tk.Label(
            search_frame,
            text="Date:",
            font=FONTS["body_bold"],
            fg=COLORS["text_secondary"],
            bg=COLORS["card_bg"]
        ).pack(side="left", padx=(0, 6))

        self.date_var = tk.StringVar(value=datetime.now().strftime("%Y-%m-%d"))
        self.ent_date = ttk.Entry(search_frame, textvariable=self.date_var, font=FONTS["body"], width=12)
        self.ent_date.pack(side="left", padx=(0, 8))

        btn_today = tk.Button(
            search_frame,
            text="Today",
            font=FONTS["caption"],
            bg=COLORS["bg_main"],
            fg=COLORS["text_secondary"],
            relief="flat",
            cursor="hand2",
            padx=8,
            pady=3,
            command=self._set_today
        )
        btn_today.pack(side="left", padx=(0, 15))

        # Buttons (Right side of Toolbar)
        btn_refresh = tk.Button(
            toolbar,
            text="🔄 Refresh",
            font=FONTS["button"],
            bg=COLORS["bg_main"],
            fg=COLORS["text_primary"],
            activebackground=COLORS["table_header_bg"],
            relief="flat",
            cursor="hand2",
            padx=12,
            pady=6,
            command=self.refresh_records
        )
        btn_refresh.pack(side="right", padx=(8, 0))

        btn_export = tk.Button(
            toolbar,
            text="📥 Export to CSV",
            font=FONTS["button"],
            bg=COLORS["success"],
            fg=COLORS["text_light"],
            activebackground="#059669",
            activeforeground=COLORS["text_light"],
            relief="flat",
            cursor="hand2",
            padx=14,
            pady=6,
            command=self._export_csv
        )
        btn_export.pack(side="right")

        # Treeview Table
        columns = ("id", "student_id", "name", "course", "date", "time", "confidence", "status")
        self.tree = ttk.Treeview(
            table_card,
            columns=columns,
            show="headings",
            style="Modern.Treeview",
            selectmode="browse"
        )

        # Configure Headings and Column Widths
        headers = {
            "id": ("#", 50, "center"),
            "student_id": ("Student ID", 120, "center"),
            "name": ("Student Name", 220, "w"),
            "course": ("Course / Dept", 180, "w"),
            "date": ("Date", 110, "center"),
            "time": ("Time", 100, "center"),
            "confidence": ("Match Conf.", 110, "center"),
            "status": ("Status", 100, "center"),
        }

        for col, (text, width, anchor) in headers.items():
            self.tree.heading(col, text=text)
            self.tree.column(col, width=width, anchor=anchor)

        # Vertical Scrollbar
        scrollbar = ttk.Scrollbar(table_card, orient="vertical", command=self.tree.yview, style="Vertical.TScrollbar")
        self.tree.configure(yscrollcommand=scrollbar.set)

        self.tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Bottom Summary Status Bar
        self.status_bar = tk.Label(
            self,
            text="Loading records...",
            font=FONTS["caption"],
            fg=COLORS["text_secondary"],
            bg=COLORS["bg_main"],
            padx=25,
            pady=8
        )
        self.status_bar.pack(fill="x", side="bottom")

    def _set_today(self) -> None:
        """Set date filter to current date and refresh."""
        self.date_var.set(datetime.now().strftime("%Y-%m-%d"))
        self.refresh_records()

    def refresh_records(self) -> None:
        """Fetch attendance records matching the current date filter from SQLite."""
        target_date = self.date_var.get().strip()
        try:
            if target_date:
                self.all_records = self.db.get_attendance_by_date(target_date)
            else:
                self.all_records = self.db.get_today_attendance()
        except Exception as e:
            self.all_records = []
            messagebox.showerror("Query Error", f"Failed to retrieve attendance records: {e}", parent=self)

        self._filter_records()

    def _filter_records(self) -> None:
        """Filter cached records in-memory based on search text and update table."""
        # Clear existing table rows
        for item in self.tree.get_children():
            self.tree.delete(item)

        query = self.search_var.get().strip().lower()

        filtered = []
        for r in self.all_records:
            if not query:
                filtered.append(r)
            else:
                stu_id = str(r.get("student_id", "")).lower()
                name = str(r.get("name", "")).lower()
                course = str(r.get("course", "")).lower()
                if query in stu_id or query in name or query in course:
                    filtered.append(r)

        # Insert rows into Treeview
        for r in filtered:
            raw_conf = r.get("confidence")
            conf_val = float(raw_conf) if raw_conf is not None else 0.0
            conf_display = f"{conf_val:.1f}%"
            self.tree.insert(
                "",
                "end",
                values=(
                    r.get("id", ""),
                    r.get("student_id", ""),
                    r.get("name", ""),
                    r.get("course", ""),
                    r.get("date", ""),
                    r.get("time", ""),
                    conf_display,
                    r.get("status", "Present")
                )
            )

        # Update bottom status
        total_shown = len(filtered)
        date_label = self.date_var.get() or "All"
        self.status_bar.config(
            text=f"Showing {total_shown} attendance records for date: {date_label} | Total for date: {len(self.all_records)}"
        )

    def _export_csv(self) -> None:
        """Open file dialog and export current date's attendance report to CSV."""
        if not self.all_records:
            messagebox.showwarning("Export Notice", "No attendance records available to export for this date.", parent=self)
            return

        date_str = self.date_var.get().strip() or datetime.now().strftime("%Y-%m-%d")
        default_filename = f"SmartAttend_Report_{date_str}.csv"

        file_path = filedialog.asksaveasfilename(
            parent=self,
            title="Export Attendance Report to CSV",
            initialfile=default_filename,
            defaultextension=".csv",
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
        )

        if not file_path:
            return

        try:
            saved_path = self.att_mgr.export_attendance_to_csv(
                date_str=date_str,
                output_path=file_path
            )
            messagebox.showinfo(
                "Export Successful",
                f"Attendance report successfully exported!\n\nSaved to:\n{saved_path}",
                parent=self
            )
        except Exception as e:
            messagebox.showerror("Export Failed", f"Could not export CSV file: {e}", parent=self)
