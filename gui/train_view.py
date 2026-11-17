"""
SmartAttend - Model Training View
Provides an interface for initiating LBPH model training, inspecting training metadata,
hyperparameter configuration, and viewing dataset statistics.
"""

from datetime import datetime
import os
import threading
import tkinter as tk
from tkinter import messagebox, ttk
from typing import Any, Callable, Dict, Optional

import config
from database import DatabaseManager, db as default_db
from gui.theme import COLORS, FONTS
from train_model import InsufficientTrainingDataError, LBPHTrainer


class TrainViewModel(tk.Frame):
    """
    Panel for training, retraining, and inspecting the LBPH Face Recognition Model.
    """

    def __init__(
        self,
        parent: tk.Widget,
        db_manager: Optional[DatabaseManager] = None,
        on_train_complete: Optional[Callable[[], None]] = None
    ):
        super().__init__(parent, bg=COLORS["bg_main"])
        self.db = db_manager or default_db
        self.on_train_complete = on_train_complete
        self.trainer = LBPHTrainer(db_manager=self.db)

        self._is_training = False
        self._build_ui()
        self.refresh_model_info()

    def _build_ui(self) -> None:
        """Construct the model training management interface."""
        # Top Header
        header = tk.Frame(self, bg=COLORS["bg_main"])
        header.pack(fill="x", padx=25, pady=(20, 10))

        title = tk.Label(
            header,
            text="Model Training & Calibration",
            font=FONTS["page_title"],
            fg=COLORS["text_primary"],
            bg=COLORS["bg_main"]
        )
        title.pack(anchor="w")

        subtitle = tk.Label(
            header,
            text="Train or retrain the LBPH classifier using captured student face datasets.",
            font=FONTS["body"],
            fg=COLORS["text_secondary"],
            bg=COLORS["bg_main"]
        )
        subtitle.pack(anchor="w", pady=(2, 0))

        # Main Content Layout
        content = tk.Frame(self, bg=COLORS["bg_main"])
        content.pack(fill="both", expand=True, padx=25, pady=10)

        # Left Column: Actions and Status
        left_col = tk.Frame(
            content,
            bg=COLORS["card_bg"],
            highlightbackground=COLORS["card_border"],
            highlightthickness=1,
            padx=25,
            pady=25
        )
        left_col.pack(side="left", fill="both", expand=True, padx=(0, 15))

        tk.Label(
            left_col,
            text="Training Controls",
            font=FONTS["section_title"],
            fg=COLORS["text_primary"],
            bg=COLORS["card_bg"]
        ).pack(anchor="w", pady=(0, 15))

        # Status Label
        self.lbl_status = tk.Label(
            left_col,
            text="Ready to train",
            font=FONTS["body"],
            fg=COLORS["text_secondary"],
            bg=COLORS["card_bg"]
        )
        self.lbl_status.pack(anchor="w", pady=(0, 10))

        # Progress Bar
        self.prog_bar = ttk.Progressbar(left_col, mode="determinate")
        self.prog_bar.pack(fill="x", pady=(0, 20))

        # Train Button
        self.btn_train = tk.Button(
            left_col,
            text="⚡ Train / Retrain LBPH Model",
            font=FONTS["button"],
            bg=COLORS["primary"],
            fg=COLORS["text_light"],
            activebackground=COLORS["primary_hover"],
            activeforeground=COLORS["text_light"],
            relief="flat",
            cursor="hand2",
            pady=12,
            command=self._start_training
        )
        self.btn_train.pack(fill="x", pady=(0, 12))

        # Information Card inside Left Col
        info_box = tk.Frame(left_col, bg=COLORS["bg_main"], padx=15, pady=15)
        info_box.pack(fill="x", pady=(10, 0))

        tk.Label(
            info_box,
            text="ℹ Training Requirements:",
            font=FONTS["body_bold"],
            fg=COLORS["text_primary"],
            bg=COLORS["bg_main"]
        ).pack(anchor="w", pady=(0, 6))

        req_text = (
            "• At least 2 distinct registered students are required.\n"
            "• Each student must have valid face samples in dataset/.\n"
            "• Images are normalized to 200x200 grayscale.\n"
            "• Saved model will be written to trainer/trainer.yml."
        )
        tk.Label(
            info_box,
            text=req_text,
            font=FONTS["caption"],
            fg=COLORS["text_secondary"],
            bg=COLORS["bg_main"],
            justify="left"
        ).pack(anchor="w")

        # Right Column: Model Metadata & Metrics
        right_col = tk.Frame(
            content,
            bg=COLORS["card_bg"],
            highlightbackground=COLORS["card_border"],
            highlightthickness=1,
            padx=25,
            pady=25
        )
        right_col.pack(side="right", fill="both", expand=True)

        tk.Label(
            right_col,
            text="Current Model Status & Metrics",
            font=FONTS["section_title"],
            fg=COLORS["text_primary"],
            bg=COLORS["card_bg"]
        ).pack(anchor="w", pady=(0, 15))

        self.lbl_model_file = self._create_metric_row(right_col, "Model File Path:", "trainer/trainer.yml")
        self.lbl_trained_status = self._create_metric_row(right_col, "Training State:", "Checking...")
        self.lbl_total_images = self._create_metric_row(right_col, "Trained Face Images:", "0")
        self.lbl_students_count = self._create_metric_row(right_col, "Registered Students:", "0")
        self.lbl_last_trained = self._create_metric_row(right_col, "Last Trained Date:", "Never")
        self.lbl_hyperparams = self._create_metric_row(
            right_col,
            "LBPH Parameters:",
            f"Radius={config.LBPH_RADIUS}, Neighbors={config.LBPH_NEIGHBORS}, Grid={config.LBPH_GRID_X}x{config.LBPH_GRID_Y}"
        )

    def _create_metric_row(self, parent: tk.Widget, label_text: str, default_val: str) -> tk.Label:
        """Helper to create structured metric key-value displays."""
        frame = tk.Frame(parent, bg=COLORS["card_bg"])
        frame.pack(fill="x", pady=6)

        tk.Label(
            frame,
            text=label_text,
            font=FONTS["body_bold"],
            fg=COLORS["text_secondary"],
            bg=COLORS["card_bg"],
            width=20,
            anchor="w"
        ).pack(side="left")

        val_lbl = tk.Label(
            frame,
            text=default_val,
            font=FONTS["body"],
            fg=COLORS["text_primary"],
            bg=COLORS["card_bg"],
            anchor="w"
        )
        val_lbl.pack(side="left", fill="x", expand=True)
        return val_lbl

    def refresh_model_info(self) -> None:
        """Update metadata display based on current filesystem and database state."""
        total_students = self.db.get_total_students()
        self.lbl_students_count.config(text=f"{total_students} student(s) registered")

        if config.MODEL_PATH.exists() and config.MODEL_PATH.stat().st_size > 0:
            mtime = datetime.fromtimestamp(config.MODEL_PATH.stat().st_mtime)
            size_kb = config.MODEL_PATH.stat().st_size / 1024
            self.lbl_trained_status.config(text="✓ Model Trained & Active", fg=COLORS["success"])
            self.lbl_last_trained.config(text=mtime.strftime("%Y-%m-%d %H:%M:%S"))
            self.lbl_model_file.config(text=f"trainer/trainer.yml ({size_kb:.1f} KB)")
        else:
            self.lbl_trained_status.config(text="✗ Model Not Trained", fg=COLORS["danger"])
            self.lbl_last_trained.config(text="Never")
            self.lbl_model_file.config(text="trainer/trainer.yml (Missing)")

    def _start_training(self) -> None:
        """Start model training in a non-blocking background thread."""
        if self._is_training:
            return

        total_students = self.db.get_total_students()
        if total_students < 2:
            messagebox.showwarning(
                "Insufficient Students",
                f"LBPH model training requires at least 2 registered students.\n"
                f"Currently registered: {total_students}.\n\n"
                "Please register more students before training the model.",
                parent=self
            )
            return

        self._is_training = True
        self.btn_train.config(state="disabled")
        self.prog_bar["value"] = 10
        self.lbl_status.config(text="Scanning dataset and loading face images...")

        threading.Thread(target=self._training_worker, daemon=True).start()

    def _training_worker(self) -> None:
        """Background thread executing LBPHTrainer."""
        try:
            summary = self.trainer.train_model(
                progress_callback=lambda msg, frac: self.after(
                    0, lambda: self._update_progress(msg, int(frac * 100))
                )
            )
            self.after(0, self._on_training_success, summary)
        except InsufficientTrainingDataError as e:
            self.after(0, self._on_training_error, "Training Data Error", str(e))
        except Exception as e:
            self.after(0, self._on_training_error, "Training Failed", str(e))

    def _update_progress(self, msg: str, pct: int) -> None:
        """Main thread callback for progress updates."""
        self.prog_bar["value"] = pct
        self.lbl_status.config(text=msg)

    def _on_training_success(self, summary: Dict[str, Any]) -> None:
        """Handle training completion on the main thread."""
        self._is_training = False
        self.btn_train.config(state="normal")
        self.prog_bar["value"] = 100
        self.lbl_status.config(text="✓ LBPH Model trained and verified successfully!")
        self.refresh_model_info()

        messagebox.showinfo(
            "Training Successful",
            f"Face recognition model trained successfully!\n\n"
            f"• Total Face Samples: {summary['total_images']}\n"
            f"• Distinct Students: {summary['unique_students']}\n"
            f"• Saved To: {summary['model_path']}",
            parent=self
        )

        if self.on_train_complete:
            self.on_train_complete()

    def _on_training_error(self, title: str, error_msg: str) -> None:
        """Handle training failure on the main thread."""
        self._is_training = False
        self.btn_train.config(state="normal")
        self.prog_bar["value"] = 0
        self.lbl_status.config(text=f"Training error: {error_msg}")
        messagebox.showerror(title, error_msg, parent=self)
