"""
SmartAttend - Live Face Recognition & Attendance Panel
Provides live video feed recognition with real-time HUD overlays, automated attendance logging,
live event history, and graceful camera/model error handling.
"""

import threading
import time
import tkinter as tk
from tkinter import messagebox, ttk
from typing import Any, Dict, List, Optional
import cv2
import numpy as np
from PIL import Image, ImageTk

import config
from attendance_manager import AttendanceEvent, AttendanceManager, AttendanceStatus
from database import DatabaseManager, db as default_db
from face_recognition import FaceRecognizer, ModelNotFoundError
from gui.theme import COLORS, FONTS


class RecognitionView(tk.Frame):
    """
    Panel for streaming live camera feed, executing LBPH recognition,
    and automatically logging attendance records in real time.
    """

    def __init__(
        self,
        parent: tk.Widget,
        db_manager: Optional[DatabaseManager] = None,
        on_attendance_marked: Optional[Any] = None
    ):
        super().__init__(parent, bg=COLORS["bg_main"])
        self.db = db_manager or default_db
        self.on_attendance_marked = on_attendance_marked

        self.recognizer: Optional[FaceRecognizer] = None
        self.att_mgr: Optional[AttendanceManager] = None

        # Threading & Video Stream State
        self._is_running = False
        self._thread: Optional[threading.Thread] = None
        self._current_photo: Optional[ImageTk.PhotoImage] = None
        self._session_marked_count = 0

        self._build_ui()
        self._check_model_status()

    def _build_ui(self) -> None:
        """Construct the recognition view interface."""
        # Top Header
        header = tk.Frame(self, bg=COLORS["bg_main"])
        header.pack(fill="x", padx=25, pady=(20, 10))

        title = tk.Label(
            header,
            text="Live Attendance Recognition",
            font=FONTS["page_title"],
            fg=COLORS["text_primary"],
            bg=COLORS["bg_main"]
        )
        title.pack(anchor="w")

        subtitle = tk.Label(
            header,
            text="Real-time webcam facial detection and instant attendance registration.",
            font=FONTS["body"],
            fg=COLORS["text_secondary"],
            bg=COLORS["bg_main"]
        )
        subtitle.pack(anchor="w", pady=(2, 0))

        # Model Warning Banner (hidden when model is present)
        self.model_warning_banner = tk.Frame(self, bg=COLORS["warning_bg"], padx=15, pady=8)
        self.lbl_model_warning = tk.Label(
            self.model_warning_banner,
            text="⚠️ Face Recognition Model is not trained yet. Register students and train the model before starting recognition.",
            font=FONTS["body_bold"],
            fg="#92400e",
            bg=COLORS["warning_bg"]
        )
        self.lbl_model_warning.pack(side="left")

        # Main Workspace Container
        workspace = tk.Frame(self, bg=COLORS["bg_main"])
        workspace.pack(fill="both", expand=True, padx=25, pady=10)

        # ----------------- Left: Live Video Display -----------------
        video_card = tk.Frame(
            workspace,
            bg=COLORS["card_bg"],
            highlightbackground=COLORS["card_border"],
            highlightthickness=1,
            padx=18,
            pady=18
        )
        video_card.pack(side="left", fill="both", expand=True, padx=(0, 15))

        # Controls Toolbar above video
        toolbar = tk.Frame(video_card, bg=COLORS["card_bg"])
        toolbar.pack(fill="x", pady=(0, 12))

        self.btn_start = tk.Button(
            toolbar,
            text="▶ Start Recognition",
            font=FONTS["button"],
            bg=COLORS["success"],
            fg=COLORS["text_light"],
            activebackground="#059669",
            activeforeground=COLORS["text_light"],
            relief="flat",
            cursor="hand2",
            padx=16,
            pady=8,
            command=self.start_recognition
        )
        self.btn_start.pack(side="left", padx=(0, 10))

        self.btn_stop = tk.Button(
            toolbar,
            text="⏹ Stop Recognition",
            font=FONTS["button"],
            bg=COLORS["danger"],
            fg=COLORS["text_light"],
            activebackground="#dc2626",
            activeforeground=COLORS["text_light"],
            relief="flat",
            cursor="hand2",
            padx=16,
            pady=8,
            state="disabled",
            command=self.stop_recognition
        )
        self.btn_stop.pack(side="left")

        # Camera Status Indicator Pill
        self.lbl_cam_status = tk.Label(
            toolbar,
            text="● Camera Idle",
            font=FONTS["body_bold"],
            fg=COLORS["text_secondary"],
            bg=COLORS["card_bg"]
        )
        self.lbl_cam_status.pack(side="right", padx=10)

        # Video Canvas
        self.canvas_video = tk.Canvas(
            video_card,
            bg="#0f172a",
            highlightthickness=0
        )
        self.canvas_video.pack(fill="both", expand=True)
        self.canvas_video.bind("<Configure>", lambda e: self._draw_standby_screen() if not self._is_running else None)
        self._draw_standby_screen()

        # ----------------- Right: Live Activity & Stats -----------------
        sidebar_card = tk.Frame(
            workspace,
            bg=COLORS["card_bg"],
            highlightbackground=COLORS["card_border"],
            highlightthickness=1,
            padx=18,
            pady=18
        )
        sidebar_card.pack(side="right", fill="both", expand=False)
        sidebar_card.config(width=380)
        sidebar_card.pack_propagate(False)

        # Session Metrics Header
        tk.Label(
            sidebar_card,
            text="Session Summary",
            font=FONTS["section_title"],
            fg=COLORS["text_primary"],
            bg=COLORS["card_bg"]
        ).pack(anchor="w", pady=(0, 8))

        # KPI Mini Card
        kpi_frame = tk.Frame(sidebar_card, bg=COLORS["bg_main"], padx=12, pady=10)
        kpi_frame.pack(fill="x", pady=(0, 15))

        tk.Label(
            kpi_frame,
            text="Attendance Marked in Session",
            font=FONTS["caption"],
            fg=COLORS["text_secondary"],
            bg=COLORS["bg_main"]
        ).pack(anchor="w")

        self.lbl_session_count = tk.Label(
            kpi_frame,
            text="0",
            font=("Segoe UI", 20, "bold"),
            fg=COLORS["primary"],
            bg=COLORS["bg_main"]
        )
        self.lbl_session_count.pack(anchor="w")

        # Live Activity Feed Title
        tk.Label(
            sidebar_card,
            text="Live Recognition Events",
            font=FONTS["section_title"],
            fg=COLORS["text_primary"],
            bg=COLORS["card_bg"]
        ).pack(anchor="w", pady=(0, 8))

        # Events Listbox / Text Area
        events_frame = tk.Frame(sidebar_card, bg=COLORS["card_bg"])
        events_frame.pack(fill="both", expand=True)

        self.events_list = tk.Listbox(
            events_frame,
            font=FONTS["caption"],
            bg=COLORS["bg_main"],
            fg=COLORS["text_primary"],
            selectbackground=COLORS["table_selected"],
            selectforeground=COLORS["text_primary"],
            relief="flat",
            borderwidth=1,
            highlightthickness=0
        )
        list_scroll = ttk.Scrollbar(events_frame, orient="vertical", command=self.events_list.yview)
        self.events_list.configure(yscrollcommand=list_scroll.set)

        self.events_list.pack(side="left", fill="both", expand=True)
        list_scroll.pack(side="right", fill="y")

        btn_clear_log = tk.Button(
            sidebar_card,
            text="Clear Event Log",
            font=FONTS["caption"],
            bg=COLORS["bg_main"],
            fg=COLORS["text_secondary"],
            relief="flat",
            cursor="hand2",
            pady=4,
            command=self._clear_logs
        )
        btn_clear_log.pack(fill="x", pady=(8, 0))

    def _check_model_status(self) -> bool:
        """Check if trained model is present and update warning banner."""
        if config.MODEL_PATH.exists() and config.MODEL_PATH.stat().st_size > 0:
            self.model_warning_banner.pack_forget()
            return True
        else:
            self.model_warning_banner.pack(fill="x", padx=25, pady=(0, 10))
            return False

    def _draw_standby_screen(self) -> None:
        """Display standby message on the video canvas."""
        self.canvas_video.delete("all")
        w = self.canvas_video.winfo_width()
        h = self.canvas_video.winfo_height()
        if w < 100 or h < 100:
            w, h = 640, 480
        self.canvas_video.create_text(
            w // 2,
            h // 2,
            text="🎥 Camera Offline\n\nClick 'Start Recognition' to begin live stream.",
            fill="#94a3b8",
            font=FONTS["body"],
            justify="center"
        )

    def _clear_logs(self) -> None:
        """Clear live events list."""
        self.events_list.delete(0, tk.END)

    def start_recognition(self) -> None:
        """Initialize models and launch live video stream worker thread."""
        if self._is_running:
            return

        # Ensure model is trained
        if not self._check_model_status():
            messagebox.showerror(
                "Model Not Trained",
                "Cannot start recognition: The LBPH face model file ('trainer.yml') was not found.\n\n"
                "Please register students first and train the model before launching recognition.",
                parent=self
            )
            return

        # Instantiate FaceRecognizer and AttendanceManager
        try:
            self.recognizer = FaceRecognizer(
                model_path=config.MODEL_PATH,
                db_manager=self.db,
                threshold=config.DEFAULT_CONFIDENCE_THRESHOLD
            )
            self.att_mgr = AttendanceManager(
                db_manager=self.db,
                face_recognizer=self.recognizer,
                cooldown_seconds=config.ATTENDANCE_COOLDOWN_SECONDS
            )
        except Exception as e:
            messagebox.showerror("Initialization Error", f"Failed to initialize recognizer: {e}", parent=self)
            return

        self._is_running = True
        self.btn_start.config(state="disabled")
        self.btn_stop.config(state="normal")
        self.lbl_cam_status.config(text="● Camera LIVE", fg=COLORS["success"])

        # Start background capture thread
        self._thread = threading.Thread(target=self._recognition_worker, daemon=True)
        self._thread.start()

    def stop_recognition(self) -> None:
        """Stop video stream and release hardware resources."""
        self._is_running = False
        self.btn_start.config(state="normal")
        self.btn_stop.config(state="disabled")
        self.lbl_cam_status.config(text="● Camera Stopped", fg=COLORS["text_secondary"])
        self._draw_standby_screen()

    def _recognition_worker(self) -> None:
        """Background thread executing webcam capture, face recognition, and attendance logic."""
        cap = cv2.VideoCapture(config.DEFAULT_CAMERA_INDEX)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_FRAME_HEIGHT)

        if not cap.isOpened():
            self.after(0, self._on_camera_error, "Could not open webcam. Check camera connection and permissions.")
            return

        try:
            while self._is_running:
                ret, frame = cap.read()
                if not ret or frame is None:
                    time.sleep(0.03)
                    continue

                # Process frame through AttendanceManager
                annotated_frame, events = self.att_mgr.process_frame(frame)

                # Prepare frame for GUI rendering
                rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                cw = max(320, self.canvas_video.winfo_width())
                ch = max(240, self.canvas_video.winfo_height())

                fh, fw = rgb.shape[:2]
                scale = min(cw / fw, ch / fh)
                nw = max(1, int(fw * scale))
                nh = max(1, int(fh * scale))

                resized = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
                pil_img = Image.fromarray(resized)

                # Post update safely to Tkinter main thread
                self.after(0, self._update_stream_ui, pil_img, events)

                # Yield brief slice for ~30 FPS smoothness
                time.sleep(0.015)

        except Exception as e:
            self.after(0, self._on_camera_error, f"Error in video recognition stream: {e}")

        finally:
            cap.release()

    def _update_stream_ui(self, pil_img: Image.Image, events: List[AttendanceEvent]) -> None:
        """Executed on main thread to render the video frame and log attendance events."""
        if not self._is_running:
            return

        self._current_photo = ImageTk.PhotoImage(image=pil_img)
        self.canvas_video.delete("all")
        cw = self.canvas_video.winfo_width()
        ch = self.canvas_video.winfo_height()
        self.canvas_video.create_image(cw // 2, ch // 2, image=self._current_photo, anchor="center")

        # Process new events
        for event in events:
            if event.status == AttendanceStatus.MARKED:
                self._session_marked_count += 1
                self.lbl_session_count.config(text=str(self._session_marked_count))
                self.events_list.insert(0, f"[{event.time}] ✓ {event.name} ({event.student_id}) - Marked")
                self.events_list.itemconfig(0, fg="#047857")  # Deep emerald
                if self.on_attendance_marked:
                    self.on_attendance_marked()

            elif event.status == AttendanceStatus.ALREADY_MARKED:
                self.events_list.insert(0, f"[{event.time}] ℹ {event.name} ({event.student_id}) - Already Present")
                self.events_list.itemconfig(0, fg="#1d4ed8")  # Blue

            elif event.status == AttendanceStatus.UNKNOWN_FACE:
                # Only log unknown face occasionally to avoid spamming
                pass

    def _on_camera_error(self, message: str) -> None:
        """Handle webcam or processing failure gracefully."""
        self.stop_recognition()
        messagebox.showerror("Camera Stream Error", message, parent=self)
