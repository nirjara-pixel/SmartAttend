"""
SmartAttend - Student Registration View
Provides a clean graphical form and embedded webcam preview for registering students,
capturing face datasets with real-time visual feedback, and initiating model training.
"""

import threading
import tkinter as tk
from tkinter import messagebox, ttk
from typing import Any, Callable, Optional
import cv2
import numpy as np
from PIL import Image, ImageTk

import config
from database import DatabaseManager, DuplicateStudentError, ValidationError, db as default_db
from gui.theme import COLORS, FONTS
from student_registration import (
    InsufficientSamplesError,
    RegistrationCancelledError,
    StudentRegistration,
    WebcamError
)
from train_model import LBPHTrainer


class RegistrationView(tk.Frame):
    """
    GUI Panel for registering new students with interactive webcam face dataset capture.
    """

    def __init__(
        self,
        parent: tk.Widget,
        db_manager: Optional[DatabaseManager] = None,
        on_registration_success: Optional[Callable[[], None]] = None
    ):
        super().__init__(parent, bg=COLORS["bg_main"])
        self.db = db_manager or default_db
        self.on_success_callback = on_registration_success
        self.reg_service = StudentRegistration(db_manager=self.db)

        # Threading state
        self._is_capturing = False
        self._stop_requested = False
        self._capture_thread: Optional[threading.Thread] = None
        self._current_photo: Optional[ImageTk.PhotoImage] = None

        self._build_ui()

    def _build_ui(self) -> None:
        """Construct the registration view layout."""
        # Top Header
        header_frame = tk.Frame(self, bg=COLORS["bg_main"])
        header_frame.pack(fill="x", padx=25, pady=(20, 10))

        title = tk.Label(
            header_frame,
            text="Student Registration",
            font=FONTS["page_title"],
            fg=COLORS["text_primary"],
            bg=COLORS["bg_main"]
        )
        title.pack(anchor="w")

        subtitle = tk.Label(
            header_frame,
            text="Register student credentials and capture face samples for LBPH recognition.",
            font=FONTS["body"],
            fg=COLORS["text_secondary"],
            bg=COLORS["bg_main"]
        )
        subtitle.pack(anchor="w", pady=(2, 0))

        # Main Body: Split into Form (Left) and Camera Preview (Right)
        body = tk.Frame(self, bg=COLORS["bg_main"])
        body.pack(fill="both", expand=True, padx=25, pady=10)

        # ----------------- Left: Form Card -----------------
        form_card = tk.Frame(
            body,
            bg=COLORS["card_bg"],
            highlightbackground=COLORS["card_border"],
            highlightthickness=1,
            padx=25,
            pady=25
        )
        form_card.pack(side="left", fill="both", expand=False, padx=(0, 15))
        form_card.config(width=420)
        form_card.pack_propagate(False)

        card_title = tk.Label(
            form_card,
            text="Student Information",
            font=FONTS["section_title"],
            fg=COLORS["text_primary"],
            bg=COLORS["card_bg"]
        )
        card_title.pack(anchor="w", pady=(0, 15))

        # Student ID Entry
        self._create_field_label(form_card, "Student ID *", "e.g. STU101 or 2024CS01")
        self.ent_id = ttk.Entry(form_card, font=FONTS["body"])
        self.ent_id.pack(fill="x", pady=(0, 12))

        # Student Name Entry
        self._create_field_label(form_card, "Full Name *", "e.g. Alice Smith")
        self.ent_name = ttk.Entry(form_card, font=FONTS["body"])
        self.ent_name.pack(fill="x", pady=(0, 12))

        # Course Entry
        self._create_field_label(form_card, "Course / Department *", "e.g. Computer Science")
        self.ent_course = ttk.Entry(form_card, font=FONTS["body"])
        self.ent_course.pack(fill="x", pady=(0, 12))

        # Samples Count Selector
        self._create_field_label(form_card, "Target Face Samples", "Samples to collect")
        self.combo_samples = ttk.Combobox(
            form_card,
            values=["30", "50", "100"],
            state="readonly",
            font=FONTS["body"]
        )
        self.combo_samples.set("50")
        self.combo_samples.pack(fill="x", pady=(0, 20))

        # Action Buttons
        self.btn_capture = tk.Button(
            form_card,
            text="▶ Start Face Capture & Register",
            font=FONTS["button"],
            bg=COLORS["primary"],
            fg=COLORS["text_light"],
            activebackground=COLORS["primary_hover"],
            activeforeground=COLORS["text_light"],
            relief="flat",
            cursor="hand2",
            pady=9,
            command=self._start_capture_thread
        )
        self.btn_capture.pack(fill="x", pady=(0, 8))

        self.btn_cancel = tk.Button(
            form_card,
            text="⏹ Cancel Capture",
            font=FONTS["button"],
            bg=COLORS["danger"],
            fg=COLORS["text_light"],
            activebackground="#dc2626",
            activeforeground=COLORS["text_light"],
            relief="flat",
            cursor="hand2",
            pady=9,
            state="disabled",
            command=self._cancel_capture
        )
        self.btn_cancel.pack(fill="x", pady=(0, 8))

        btn_clear = tk.Button(
            form_card,
            text="Clear Form",
            font=FONTS["body"],
            bg=COLORS["bg_main"],
            fg=COLORS["text_secondary"],
            relief="flat",
            cursor="hand2",
            pady=6,
            command=self._clear_form
        )
        btn_clear.pack(fill="x")

        # Guidance Note
        note_frame = tk.Frame(form_card, bg=COLORS["warning_bg"], padx=10, pady=10)
        note_frame.pack(fill="x", pady=(20, 0))
        note_lbl = tk.Label(
            note_frame,
            text="Tip: Ensure good ambient lighting. During capture, turn head slightly left, right, and smile to collect diverse angles.",
            font=FONTS["caption"],
            fg="#92400e",
            bg=COLORS["warning_bg"],
            wraplength=340,
            justify="left"
        )
        note_lbl.pack(fill="x")

        # ----------------- Right: Live Camera Card -----------------
        preview_card = tk.Frame(
            body,
            bg=COLORS["card_bg"],
            highlightbackground=COLORS["card_border"],
            highlightthickness=1,
            padx=20,
            pady=20
        )
        preview_card.pack(side="right", fill="both", expand=True)

        preview_header = tk.Frame(preview_card, bg=COLORS["card_bg"])
        preview_header.pack(fill="x", pady=(0, 10))

        tk.Label(
            preview_header,
            text="Live Camera Preview & Progress",
            font=FONTS["section_title"],
            fg=COLORS["text_primary"],
            bg=COLORS["card_bg"]
        ).pack(side="left")

        self.lbl_progress_text = tk.Label(
            preview_header,
            text="Ready to capture",
            font=FONTS["body_bold"],
            fg=COLORS["text_secondary"],
            bg=COLORS["card_bg"]
        )
        self.lbl_progress_text.pack(side="right")

        # Camera Display Canvas
        self.canvas_cam = tk.Canvas(
            preview_card,
            bg="#1e293b",
            highlightthickness=0
        )
        self.canvas_cam.pack(fill="both", expand=True, pady=(0, 10))
        self.canvas_cam.bind("<Configure>", lambda e: self._draw_placeholder() if not self._is_capturing else None)
        self._draw_placeholder()

        # Progress Bar
        self.prog_bar = ttk.Progressbar(preview_card, mode="determinate")
        self.prog_bar.pack(fill="x", pady=(0, 6))

        # Status Guidance Label
        self.lbl_guidance = tk.Label(
            preview_card,
            text="Fill in student details and click 'Start Face Capture & Register'.",
            font=FONTS["body"],
            fg=COLORS["text_secondary"],
            bg=COLORS["card_bg"]
        )
        self.lbl_guidance.pack(anchor="w")

    def _create_field_label(self, parent: tk.Widget, label_text: str, help_text: str) -> None:
        """Helper to create consistent form field labels."""
        frame = tk.Frame(parent, bg=COLORS["card_bg"])
        frame.pack(fill="x", pady=(0, 2))
        tk.Label(
            frame,
            text=label_text,
            font=FONTS["body_bold"],
            fg=COLORS["text_primary"],
            bg=COLORS["card_bg"]
        ).pack(side="left")
        tk.Label(
            frame,
            text=f"({help_text})",
            font=FONTS["caption"],
            fg=COLORS["text_muted"],
            bg=COLORS["card_bg"]
        ).pack(side="left", padx=4)

    def _draw_placeholder(self) -> None:
        """Draw friendly standby graphic on canvas."""
        self.canvas_cam.delete("all")
        w = self.canvas_cam.winfo_width()
        h = self.canvas_cam.winfo_height()
        if w < 100 or h < 100:
            w, h = 480, 360
        self.canvas_cam.create_text(
            w // 2,
            h // 2,
            text="📷 Webcam Standby\n\nClick 'Start Face Capture & Register' to open camera feed.",
            fill="#94a3b8",
            font=FONTS["body"],
            justify="center"
        )

    def _clear_form(self) -> None:
        """Clear form entries."""
        if self._is_capturing:
            return
        self.ent_id.delete(0, tk.END)
        self.ent_name.delete(0, tk.END)
        self.ent_course.delete(0, tk.END)
        self.combo_samples.set("50")
        self.prog_bar["value"] = 0
        self.lbl_progress_text.config(text="Ready to capture", fg=COLORS["text_secondary"])
        self.lbl_guidance.config(text="Form cleared. Enter student information.")
        self._draw_placeholder()

    def _start_capture_thread(self) -> None:
        """Validate input and initiate background webcam capture thread."""
        student_id = self.ent_id.get().strip()
        name = self.ent_name.get().strip()
        course = self.ent_course.get().strip()
        try:
            target_samples = int(self.combo_samples.get())
        except ValueError:
            target_samples = 50

        # Preliminary validation
        try:
            self.reg_service.validate_student_data(student_id, name, course)
        except ValidationError as e:
            messagebox.showwarning("Validation Error", str(e), parent=self)
            return
        except DuplicateStudentError as e:
            messagebox.showerror("Duplicate Student", str(e), parent=self)
            return

        # Update UI state for capturing
        self._is_capturing = True
        self._stop_requested = False
        self.btn_capture.config(state="disabled")
        self.btn_cancel.config(state="normal")
        self.prog_bar["value"] = 0
        self.lbl_guidance.config(text="Starting webcam... Please face the camera.")

        # Launch background capture thread
        self._capture_thread = threading.Thread(
            target=self._run_capture_worker,
            args=(student_id, name, course, target_samples),
            daemon=True
        )
        self._capture_thread.start()

    def _cancel_capture(self) -> None:
        """Signal background capture to abort."""
        if self._is_capturing:
            self._stop_requested = True
            self.lbl_guidance.config(text="Cancelling capture... please wait.")

    def _run_capture_worker(
        self,
        student_id: str,
        name: str,
        course: str,
        target_samples: int
    ) -> None:
        """Background worker thread that runs OpenCV capture loop."""
        try:
            result = self.reg_service.capture_and_register(
                student_id=student_id,
                name=name,
                course=course,
                target_samples=target_samples,
                show_preview=False,
                frame_callback=self._on_frame_update,
                stop_flag=lambda: self._stop_requested
            )
            self.after(0, self._on_capture_complete, result)

        except RegistrationCancelledError:
            self.after(0, self._on_capture_cancelled)
        except WebcamError as e:
            self.after(0, self._on_capture_error, "Webcam Error", str(e))
        except InsufficientSamplesError as e:
            self.after(0, self._on_capture_error, "Capture Incomplete", str(e))
        except Exception as e:
            self.after(0, self._on_capture_error, "Registration Failed", str(e))

    def _on_frame_update(
        self,
        frame: np.ndarray,
        captured_count: int,
        target: int,
        status_msg: str
    ) -> None:
        """Thread-safe frame callback from StudentRegistration to update canvas preview."""
        # Convert BGR frame to RGB for PIL
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Scale frame to fit canvas
        canvas_w = max(320, self.canvas_cam.winfo_width())
        canvas_h = max(240, self.canvas_cam.winfo_height())

        img_h, img_w = rgb_frame.shape[:2]
        scale = min(canvas_w / img_w, canvas_h / img_h)
        new_w = max(1, int(img_w * scale))
        new_h = max(1, int(img_h * scale))

        resized = cv2.resize(rgb_frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        pil_img = Image.fromarray(resized)

        # Schedule Tkinter UI update on the main thread
        self.after(
            0,
            self._update_preview_ui,
            pil_img,
            captured_count,
            target,
            status_msg
        )

    def _update_preview_ui(
        self,
        pil_img: Image.Image,
        captured_count: int,
        target: int,
        status_msg: str
    ) -> None:
        """Executed on Tkinter main thread to paint the canvas and progress bar."""
        if not self._is_capturing:
            return

        self._current_photo = ImageTk.PhotoImage(image=pil_img)
        self.canvas_cam.delete("all")
        cw = self.canvas_cam.winfo_width()
        ch = self.canvas_cam.winfo_height()
        self.canvas_cam.create_image(cw // 2, ch // 2, image=self._current_photo, anchor="center")

        pct = int((captured_count / target) * 100) if target > 0 else 0
        self.prog_bar["value"] = pct
        self.lbl_progress_text.config(
            text=f"Samples: {captured_count}/{target} ({pct}%)",
            fg=COLORS["primary"]
        )
        self.lbl_guidance.config(text=status_msg)

    def _on_capture_complete(self, result: dict) -> None:
        """Handle successful registration completion on the main thread."""
        self._is_capturing = False
        self.btn_capture.config(state="normal")
        self.btn_cancel.config(state="disabled")
        self.prog_bar["value"] = 100
        self.lbl_progress_text.config(text="Registration Complete!", fg=COLORS["success"])
        self.lbl_guidance.config(
            text=f"Successfully registered {result['name']} with {result['sample_count']} samples."
        )

        # Prompt user to train model
        train_now = messagebox.askyesno(
            "Registration Successful",
            f"Student {result['name']} ({result['student_id']}) has been registered!\n\n"
            f"Would you like to train the face recognition model now to include this student?",
            parent=self
        )

        if train_now:
            self._prompt_train_model()

        self._clear_form()

        if self.on_success_callback:
            self.on_success_callback()

    def _prompt_train_model(self) -> None:
        """Run LBPH model training in a background thread with progress feedback."""
        self.lbl_guidance.config(text="Training face recognition model in background...")

        def train_worker():
            try:
                trainer = LBPHTrainer(db_manager=self.db)
                summary = trainer.train_model()
                self.after(
                    0,
                    lambda: messagebox.showinfo(
                        "Model Trained",
                        f"Face recognition model updated successfully!\n\n"
                        f"• Total Images: {summary['total_images']}\n"
                        f"• Unique Students: {summary['unique_students']}",
                        parent=self
                    )
                )
                self.after(0, lambda: self.lbl_guidance.config(text="Model training complete."))
            except Exception as e:
                self.after(
                    0,
                    lambda: messagebox.showwarning(
                        "Training Notice",
                        f"Could not complete model training: {e}",
                        parent=self
                    )
                )

        threading.Thread(target=train_worker, daemon=True).start()

    def _on_capture_cancelled(self) -> None:
        """Handle user cancellation."""
        self._is_capturing = False
        self.btn_capture.config(state="normal")
        self.btn_cancel.config(state="disabled")
        self.lbl_progress_text.config(text="Cancelled", fg=COLORS["danger"])
        self.lbl_guidance.config(text="Registration cancelled. Partial dataset cleaned up.")
        self._draw_placeholder()

    def _on_capture_error(self, title: str, error_msg: str) -> None:
        """Handle capture errors."""
        self._is_capturing = False
        self.btn_capture.config(state="normal")
        self.btn_cancel.config(state="disabled")
        self.lbl_progress_text.config(text="Error", fg=COLORS["danger"])
        self.lbl_guidance.config(text=f"Error: {error_msg}")
        self._draw_placeholder()
        messagebox.showerror(title, error_msg, parent=self)
