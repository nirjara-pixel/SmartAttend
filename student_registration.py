"""
SmartAttend - Student Registration & Face Dataset Collection Module
Handles student registration validation, webcam face sample acquisition,
image preprocessing, dataset storage, and SQLite database synchronization.
"""

import os
import shutil
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np

import config
from database import DatabaseManager, DuplicateStudentError, ValidationError, db as default_db
from face_detector import FaceDetector


class RegistrationError(Exception):
    """Base exception for student registration errors."""
    pass


class WebcamError(RegistrationError):
    """Raised when the camera fails to initialize or stream frames."""
    pass


class RegistrationCancelledError(RegistrationError):
    """Raised when the registration process is aborted by user or parent process."""
    pass


class InsufficientSamplesError(RegistrationError):
    """Raised when required number of face samples cannot be captured."""
    pass


class DatasetStorageError(RegistrationError):
    """Raised when creating or writing to the dataset directory fails."""
    pass


class StudentRegistration:
    """
    Manages end-to-end student registration and face dataset acquisition.
    Integrates with FaceDetector and DatabaseManager.
    """

    def __init__(
        self,
        db_manager: Optional[DatabaseManager] = None,
        face_detector: Optional[FaceDetector] = None,
        dataset_base_dir: Optional[Union[str, Path]] = None,
        camera_index: int = config.DEFAULT_CAMERA_INDEX,
        target_samples: int = 100
    ):
        """
        Initialize the Student Registration manager.

        Args:
            db_manager: DatabaseManager instance. Defaults to default_db.
            face_detector: FaceDetector instance. Defaults to a new FaceDetector().
            dataset_base_dir: Root directory for datasets. Defaults to config.DATASET_DIR.
            camera_index: Hardware camera index.
            target_samples: Target number of valid face crops to collect (default 100).
        """
        self.db = db_manager or default_db
        self.face_detector = face_detector or FaceDetector()
        self.dataset_base_dir = Path(dataset_base_dir or config.DATASET_DIR)
        self.camera_index = camera_index
        self.target_samples = target_samples

        # Ensure base dataset directory exists
        self.dataset_base_dir.mkdir(parents=True, exist_ok=True)

    def validate_student_data(
        self,
        student_id: str,
        name: str,
        course: str
    ) -> Tuple[str, str, str]:
        """
        Validate student inputs and ensure no duplicate exists in SQLite or dataset directory.

        Args:
            student_id: Unique institutional ID.
            name: Full name.
            course: Course / department.

        Returns:
            Tuple of sanitized (student_id, name, course).

        Raises:
            ValidationError: If any field is empty or contains invalid characters.
            DuplicateStudentError: If student_id already exists in database or dataset.
        """
        if not student_id or not str(student_id).strip():
            raise ValidationError("Student ID is required and cannot be empty.")
        if not name or not str(name).strip():
            raise ValidationError("Student Name is required and cannot be empty.")
        if not course or not str(course).strip():
            raise ValidationError("Course is required and cannot be empty.")

        clean_id = str(student_id).strip()
        clean_name = str(name).strip()
        clean_course = str(course).strip()

        # Check for invalid filesystem characters in student_id
        invalid_chars = set('<>:"/\\|?*')
        if any(char in invalid_chars for char in clean_id):
            raise ValidationError(
                f"Student ID contains invalid characters: {''.join(invalid_chars)}"
            )

        # Check database for duplicate
        existing_student = self.db.get_student(clean_id)
        if existing_student:
            raise DuplicateStudentError(
                f"Student ID '{clean_id}' is already registered in the system "
                f"(Name: {existing_student['name']}, Course: {existing_student['course']})."
            )

        # Check filesystem for existing non-empty dataset folder
        student_dir = self.get_student_dataset_dir(clean_id)
        if student_dir.exists() and any(student_dir.iterdir()):
            raise DuplicateStudentError(
                f"A dataset folder for Student ID '{clean_id}' already exists on disk. "
                "Cannot overwrite another student's dataset."
            )

        return clean_id, clean_name, clean_course

    def get_student_dataset_dir(self, student_id: str) -> Path:
        """
        Get the dataset directory path for a specific student.
        """
        return self.dataset_base_dir / str(student_id).strip()

    def cleanup_dataset(self, student_id: str) -> None:
        """
        Safely remove dataset files and directory for a student if registration fails or is cancelled.
        """
        student_dir = self.get_student_dataset_dir(student_id)
        if student_dir.exists():
            try:
                shutil.rmtree(student_dir)
            except Exception as e:
                # Log or swallow during failure handling to allow re-raise
                pass

    def save_face_sample(
        self,
        face_roi: np.ndarray,
        student_id: str,
        sample_index: int
    ) -> Path:
        """
        Save a preprocessed face sample (200x200 grayscale) with sequential filename.

        Args:
            face_roi: 2D numpy array (200x200 uint8).
            student_id: Student ID.
            sample_index: 1-based sequential index.

        Returns:
            Path to the saved image file.

        Raises:
            ValidationError: If face_roi shape or type is invalid.
            DatasetStorageError: If file saving fails.
        """
        expected_size = config.NORMALIZED_FACE_SIZE
        if (
            face_roi is None
            or not isinstance(face_roi, np.ndarray)
            or face_roi.shape != expected_size
            or face_roi.dtype != np.uint8
        ):
            raise ValidationError(
                f"Face ROI must be a 2D uint8 numpy array with dimensions {expected_size}. "
                f"Got: {type(face_roi)} shape={getattr(face_roi, 'shape', None)}"
            )

        student_dir = self.get_student_dataset_dir(student_id)
        student_dir.mkdir(parents=True, exist_ok=True)

        filename = f"sample_{sample_index:03d}.jpg"
        file_path = student_dir / filename

        success = cv2.imwrite(str(file_path), face_roi)
        if not success or not file_path.exists():
            raise DatasetStorageError(f"Failed to write face image to '{file_path}'.")

        return file_path

    def capture_and_register(
        self,
        student_id: str,
        name: str,
        course: str,
        target_samples: Optional[int] = None,
        show_preview: bool = True,
        frame_callback: Optional[Callable[[np.ndarray, int, int, str], None]] = None,
        stop_flag: Optional[Callable[[], bool]] = None,
        video_capture: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Execute full webcam capture and student registration workflow.

        Args:
            student_id: Student ID.
            name: Student full name.
            course: Student course/department.
            target_samples: Number of samples to collect (defaults to self.target_samples).
            show_preview: If True and no callback, displays an OpenCV preview window.
            frame_callback: Optional callback for GUI embedding: cb(frame, captured, target, status_msg).
            stop_flag: Optional callable returning True if user requested abort.
            video_capture: Optional cv2.VideoCapture instance (used for testing/mocking).

        Returns:
            Dictionary containing registration and dataset details.

        Raises:
            ValidationError: On input validation failure.
            DuplicateStudentError: If student already exists.
            WebcamError: If webcam cannot be opened or read.
            RegistrationCancelledError: If aborted by user.
            InsufficientSamplesError: If stream ends before collecting target samples.
        """
        # Step 1: Input Validation & Duplicate Checking
        clean_id, clean_name, clean_course = self.validate_student_data(student_id, name, course)
        target = target_samples or self.target_samples
        student_dir = self.get_student_dataset_dir(clean_id)

        # Step 2: Initialize Webcam
        cap = video_capture
        own_cap = False
        if cap is None:
            cap = cv2.VideoCapture(self.camera_index)
            own_cap = True

        if not cap.isOpened():
            if own_cap:
                cap.release()
            raise WebcamError(
                f"Could not open camera at index {self.camera_index}. "
                "Please verify webcam connection and system permissions."
            )

        # Set capture resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_FRAME_HEIGHT)

        saved_files: List[Path] = []
        captured_count = 0
        status_message = "Align face in the frame..."
        last_capture_time = 0.0
        capture_delay = config.SAMPLE_CAPTURE_INTERVAL

        try:
            student_dir.mkdir(parents=True, exist_ok=True)

            while captured_count < target:
                # Check external abort flag
                if stop_flag and stop_flag():
                    raise RegistrationCancelledError("Registration cancelled by user.")

                ret, frame = cap.read()
                if not ret or frame is None:
                    # If stream unexpectedly closes
                    if captured_count < target:
                        raise InsufficientSamplesError(
                            f"Webcam stream ended. Only {captured_count}/{target} samples captured."
                        )
                    break

                display_frame = frame.copy()
                faces = self.face_detector.detect_faces(frame)
                current_time = time.time()

                if len(faces) == 0:
                    status_message = "No face detected. Please face the camera."
                elif len(faces) > 1:
                    status_message = "Multiple faces detected! Only one face allowed."
                    # Draw warning box on all detected faces
                    for (x, y, w, h) in faces:
                        cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 140, 255), 2)
                else:
                    # Exactly one face detected
                    bbox = faces[0]
                    x, y, w, h = bbox

                    # Guidance for diverse angles and expressions
                    progress_pct = int((captured_count / target) * 100)
                    if progress_pct < 25:
                        guidance = "Look straight at the camera"
                    elif progress_pct < 50:
                        guidance = "Turn head slightly left and right"
                    elif progress_pct < 75:
                        guidance = "Tilt head slightly up and down"
                    else:
                        guidance = "Smile or change facial expression"

                    status_message = f"Capturing: {captured_count + 1}/{target} ({progress_pct}%) - {guidance}"

                    # Capture face sample if time delay elapsed
                    if current_time - last_capture_time >= capture_delay:
                        face_roi = self.face_detector.extract_face_roi(
                            frame,
                            bbox,
                            target_size=config.NORMALIZED_FACE_SIZE
                        )
                        saved_path = self.save_face_sample(face_roi, clean_id, captured_count + 1)
                        saved_files.append(saved_path)
                        captured_count += 1
                        last_capture_time = current_time

                    # Draw stylish face box
                    FaceDetector.draw_face_box(
                        display_frame,
                        bbox,
                        primary_text=f"{clean_name} ({clean_id})",
                        secondary_text=f"Sample: {captured_count}/{target}",
                        color=(0, 255, 0)
                    )

                # Draw Top Progress HUD Banner
                hud_h = 55
                cv2.rectangle(display_frame, (0, 0), (display_frame.shape[1], hud_h), (25, 30, 45), cv2.FILLED)
                # Progress bar
                pct = captured_count / target
                bar_w = int((display_frame.shape[1] - 40) * pct)
                cv2.rectangle(display_frame, (20, 32), (display_frame.shape[1] - 20, 42), (60, 70, 90), cv2.FILLED)
                cv2.rectangle(display_frame, (20, 32), (20 + bar_w, 42), (0, 220, 100), cv2.FILLED)

                # Status text
                cv2.putText(
                    display_frame,
                    f"Registering: {clean_name} | {status_message}",
                    (20, 22),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.50,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA
                )

                # Deliver frame to GUI callback if provided
                if frame_callback:
                    frame_callback(display_frame, captured_count, target, status_message)
                elif show_preview:
                    cv2.imshow("SmartAttend - Face Registration", display_frame)
                    key = cv2.waitKey(1) & 0xFF
                    # Press ESC or 'q' to abort
                    if key == 27 or key == ord("q"):
                        raise RegistrationCancelledError("Registration cancelled by user via key press.")

            # Step 3: Register in SQLite database after successful sample collection
            internal_id = self.db.add_student(
                student_id=clean_id,
                name=clean_name,
                course=clean_course,
                sample_count=captured_count
            )

            return {
                "success": True,
                "id": internal_id,
                "student_id": clean_id,
                "name": clean_name,
                "course": clean_course,
                "sample_count": captured_count,
                "dataset_dir": str(student_dir)
            }

        except Exception as e:
            # Step 4: Cleanup incomplete dataset on any failure or cancellation
            self.cleanup_dataset(clean_id)
            raise e

        finally:
            if own_cap and cap is not None:
                cap.release()
            if show_preview and not frame_callback:
                try:
                    cv2.destroyWindow("SmartAttend - Face Registration")
                except Exception:
                    pass
