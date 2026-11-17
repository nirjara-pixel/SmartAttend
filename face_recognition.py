"""
SmartAttend - Real-Time Face Recognition Engine
Performs live face recognition using trained LBPH model, Haar Cascade detection,
distance-to-confidence scoring, and student identity resolution.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np

import config
from database import DatabaseManager, db as default_db
from face_detector import FaceDetector


class RecognitionError(Exception):
    """Base exception for face recognition operations."""
    pass


class ModelNotFoundError(RecognitionError):
    """Raised when the trained model file (trainer.yml) is not found."""
    pass


@dataclass
class RecognitionResult:
    """Represents the recognition outcome for a single detected face."""
    is_recognized: bool
    student_id: Optional[str]
    name: str
    course: Optional[str]
    internal_id: Optional[int]
    confidence: float
    distance: float
    bbox: Tuple[int, int, int, int]

    @property
    def is_unknown(self) -> bool:
        return not self.is_recognized


class FaceRecognizer:
    """
    High-level Face Recognition Engine utilizing LBPH and Haar Cascades.
    """

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        db_manager: Optional[DatabaseManager] = None,
        face_detector: Optional[FaceDetector] = None,
        threshold: float = config.DEFAULT_CONFIDENCE_THRESHOLD
    ):
        """
        Initialize the Face Recognizer.

        Args:
            model_path: Path to trained model trainer.yml. Defaults to config.MODEL_PATH.
            db_manager: DatabaseManager instance for student lookups. Defaults to default_db.
            face_detector: FaceDetector instance for cascade detection.
            threshold: Maximum LBPH distance for a match (lower distance = better match).
        """
        self.model_path = Path(model_path or config.MODEL_PATH)
        self.db = db_manager or default_db
        self.face_detector = face_detector or FaceDetector()
        self.threshold = float(threshold)

        self.recognizer = cv2.face.LBPHFaceRecognizer_create(
            radius=config.LBPH_RADIUS,
            neighbors=config.LBPH_NEIGHBORS,
            grid_x=config.LBPH_GRID_X,
            grid_y=config.LBPH_GRID_Y,
            threshold=self.threshold
        )

        self.student_map: Dict[int, Dict[str, Any]] = {}
        self.load_model()
        self.refresh_student_cache()

    def load_model(self) -> None:
        """
        Load the trained LBPH model from disk.

        Raises:
            ModelNotFoundError: If model file does not exist or is empty.
        """
        if not self.model_path.exists() or self.model_path.stat().st_size == 0:
            raise ModelNotFoundError(
                f"Trained model file '{self.model_path}' not found. "
                "Please train the model using registered student datasets first."
            )

        try:
            self.recognizer.read(str(self.model_path))
        except Exception as e:
            raise RecognitionError(f"Failed to load LBPH model from '{self.model_path}': {e}") from e

    def refresh_student_cache(self) -> None:
        """
        Refresh in-memory mapping of numeric labels to student database records.
        """
        students = self.db.get_all_students()
        self.student_map = {int(s["id"]): s for s in students}

    def calculate_confidence_score(self, distance: float) -> float:
        """
        Convert LBPH raw distance to an intuitive 0.0 - 100.0% similarity score.
        In LBPH: distance 0 is identical, higher distance is less similar.

        Args:
            distance: Raw Euclidean distance returned by LBPH predict().

        Returns:
            Normalized similarity score between 0.0% and 100.0%.
        """
        if distance < 0.0:
            return 100.0

        # Scale linearly: at distance=0 -> 100%, at distance=threshold -> 50%
        # Beyond threshold -> decreases smoothly towards 0%
        if distance <= self.threshold:
            # Score between 50.0% and 100.0%
            score = 100.0 - ((distance / self.threshold) * 50.0)
        else:
            # Score below 50.0%
            overflow = distance - self.threshold
            score = max(0.0, 50.0 - ((overflow / self.threshold) * 50.0))

        return round(min(100.0, max(0.0, score)), 1)

    def predict_face(
        self,
        face_roi: np.ndarray,
        bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)
    ) -> RecognitionResult:
        """
        Predict student identity for a single preprocessed face crop.

        Args:
            face_roi: 200x200 uint8 grayscale face image.
            bbox: (x, y, w, h) bounding box coordinates in the original frame.

        Returns:
            RecognitionResult dataclass instance.
        """
        if face_roi is None or face_roi.size == 0:
            return RecognitionResult(
                is_recognized=False,
                student_id=None,
                name="Unknown",
                course=None,
                internal_id=None,
                confidence=0.0,
                distance=999.0,
                bbox=bbox
            )

        # Ensure correct grayscale format and dimensions
        if len(face_roi.shape) == 3:
            face_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        if face_roi.shape != config.NORMALIZED_FACE_SIZE:
            face_roi = cv2.resize(face_roi, config.NORMALIZED_FACE_SIZE, interpolation=cv2.INTER_AREA)

        try:
            label_id, distance = self.recognizer.predict(face_roi)
        except Exception:
            return RecognitionResult(
                is_recognized=False,
                student_id=None,
                name="Unknown",
                course=None,
                internal_id=None,
                confidence=0.0,
                distance=999.0,
                bbox=bbox
            )

        # Check if student exists in database cache
        student = self.student_map.get(int(label_id))

        # Check threshold: distance must be strictly lower than threshold
        if student is not None and distance < self.threshold:
            conf_score = self.calculate_confidence_score(distance)
            return RecognitionResult(
                is_recognized=True,
                student_id=student["student_id"],
                name=student["name"],
                course=student["course"],
                internal_id=int(label_id),
                confidence=conf_score,
                distance=round(float(distance), 2),
                bbox=bbox
            )
        else:
            return RecognitionResult(
                is_recognized=False,
                student_id=None,
                name="Unknown",
                course=None,
                internal_id=None,
                confidence=0.0,
                distance=round(float(distance), 2),
                bbox=bbox
            )

    def recognize_frame(
        self,
        frame: np.ndarray,
        draw_annotations: bool = True
    ) -> Tuple[np.ndarray, List[RecognitionResult]]:
        """
        Detect and recognize all faces in a single video frame.

        Args:
            frame: Input BGR image from webcam.
            draw_annotations: If True, draws bounding boxes and badges on frame.

        Returns:
            Tuple of (annotated_frame, list of RecognitionResult objects).
        """
        if frame is None or frame.size == 0:
            return frame, []

        display_frame = frame.copy() if draw_annotations else frame
        faces = self.face_detector.detect_faces(frame)
        results: List[RecognitionResult] = []

        for bbox in faces:
            face_roi = self.face_detector.extract_face_roi(
                frame,
                bbox,
                target_size=config.NORMALIZED_FACE_SIZE
            )
            result = self.predict_face(face_roi, bbox=bbox)
            results.append(result)

            if draw_annotations:
                if result.is_recognized:
                    color = (0, 220, 100)  # Emerald Green
                    primary_text = f"{result.name}"
                    secondary_text = f"ID: {result.student_id} | {result.confidence}%"
                else:
                    color = (0, 0, 240)    # Red
                    primary_text = "Unknown"
                    secondary_text = "Unregistered Face"

                FaceDetector.draw_face_box(
                    display_frame,
                    bbox,
                    primary_text=primary_text,
                    secondary_text=secondary_text,
                    color=color
                )

        return display_frame, results
