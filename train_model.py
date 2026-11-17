"""
SmartAttend - LBPH Face Model Training Module
Handles dataset loading, validation, preprocessing, LBPH model training,
serialization to trainer/trainer.yml, metadata generation, and verification.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import cv2
import numpy as np

import config
from database import DatabaseManager, db as default_db


class TrainingError(Exception):
    """Base exception for model training operations."""
    pass


class DatasetNotFoundError(TrainingError):
    """Raised when the dataset directory does not exist."""
    pass


class InsufficientTrainingDataError(TrainingError):
    """Raised when there are insufficient valid samples or distinct students to train."""
    pass


class ModelPersistenceError(TrainingError):
    """Raised when saving or loading the trained model file fails."""
    pass


class InvalidImageError(TrainingError):
    """Raised when image fails validation."""
    pass


class LBPHTrainer:
    """
    Orchestrates the training lifecycle for OpenCV's LBPH Face Recognizer.
    """

    def __init__(
        self,
        dataset_dir: Optional[Union[str, Path]] = None,
        trainer_dir: Optional[Union[str, Path]] = None,
        model_path: Optional[Union[str, Path]] = None,
        db_manager: Optional[DatabaseManager] = None,
        radius: int = config.LBPH_RADIUS,
        neighbors: int = config.LBPH_NEIGHBORS,
        grid_x: int = config.LBPH_GRID_X,
        grid_y: int = config.LBPH_GRID_Y,
        threshold: float = config.DEFAULT_CONFIDENCE_THRESHOLD,
        target_size: Tuple[int, int] = config.NORMALIZED_FACE_SIZE,
        min_students: int = 2
    ):
        """
        Initialize the LBPH Trainer with configurations and paths.

        Args:
            dataset_dir: Path to dataset folder containing student subfolders.
            trainer_dir: Path to folder where trained model will be stored.
            model_path: Path to the target trainer.yml file.
            db_manager: DatabaseManager instance for resolving student IDs to numeric labels.
            radius: LBPH circular radius (default: 1).
            neighbors: LBPH sampling points (default: 8).
            grid_x: LBPH grid columns (default: 8).
            grid_y: LBPH grid rows (default: 8).
            threshold: LBPH distance threshold (default: 75.0).
            target_size: Expected (width, height) of normalized face images (default: 200x200).
            min_students: Minimum distinct student classes required to train (default: 2).
        """
        self.dataset_dir = Path(dataset_dir or config.DATASET_DIR)
        self.trainer_dir = Path(trainer_dir or config.TRAINER_DIR)
        self.model_path = Path(model_path or config.MODEL_PATH)
        self.metadata_path = self.trainer_dir / "metadata.json"
        self.db = db_manager or default_db

        self.radius = radius
        self.neighbors = neighbors
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.threshold = threshold
        self.target_size = target_size
        self.min_students = min_students

        # Ensure trainer directory exists
        self.trainer_dir.mkdir(parents=True, exist_ok=True)

    def create_recognizer(self) -> cv2.face.LBPHFaceRecognizer:
        """
        Create and configure an OpenCV LBPH Face Recognizer instance.

        Returns:
            Configured cv2.face.LBPHFaceRecognizer instance.
        """
        return cv2.face.LBPHFaceRecognizer_create(
            radius=self.radius,
            neighbors=self.neighbors,
            grid_x=self.grid_x,
            grid_y=self.grid_y,
            threshold=self.threshold
        )

    def validate_face_image(
        self,
        img: np.ndarray,
        expected_size: Optional[Tuple[int, int]] = None
    ) -> bool:
        """
        Validate that an image array is a non-empty, 2D uint8 grayscale array
        matching the expected dimensions.

        Args:
            img: Image array to validate.
            expected_size: Target (width, height). Defaults to self.target_size.

        Returns:
            True if image is valid, False otherwise.
        """
        target = expected_size or self.target_size
        if img is None or not isinstance(img, np.ndarray):
            return False
        if img.ndim != 2 or img.dtype != np.uint8:
            return False
        # Expected shape is (height, width) = (target[1], target[0])
        if img.shape != (target[1], target[0]):
            return False
        return True

    def load_dataset(
        self,
        strict_dimensions: bool = False
    ) -> Tuple[List[np.ndarray], np.ndarray, Dict[int, Dict[str, Any]], Dict[str, Any]]:
        """
        Load face samples from dataset directory, validate images, and extract numeric labels.

        Args:
            strict_dimensions: If True, reject images that do not match self.target_size.
                               If False, resize differing images to self.target_size.

        Returns:
            Tuple:
                - faces: List of 200x200 uint8 grayscale images.
                - labels: 1D numpy array of np.int32 labels.
                - label_map: Dict mapping numeric label -> {student_id, name, course}.
                - stats: Dict of summary statistics.

        Raises:
            DatasetNotFoundError: If dataset directory does not exist.
            InsufficientTrainingDataError: If images are missing or distinct students < min_students.
        """
        if not self.dataset_dir.exists():
            raise DatasetNotFoundError(
                f"Dataset directory '{self.dataset_dir}' does not exist. "
                "Please register students first."
            )

        faces: List[np.ndarray] = []
        labels: List[int] = []
        label_map: Dict[int, Dict[str, Any]] = {}
        unique_student_ids: Set[str] = set()

        valid_count = 0
        corrupted_count = 0
        dimension_adjusted_count = 0

        # Scan for student subdirectories
        student_dirs = [d for d in self.dataset_dir.iterdir() if d.is_dir()]

        # Deterministic label mapping counter (fallback if not in DB)
        fallback_label_counter = 1000

        for s_dir in student_dirs:
            student_id = s_dir.name.strip()
            if not student_id:
                continue

            # Resolve numeric label for LBPH (must be a 32-bit integer)
            db_student = self.db.get_student(student_id)
            if db_student:
                numeric_label = int(db_student["id"])
                name = db_student["name"]
                course = db_student["course"]
            else:
                # If folder name is digits, use that; otherwise generate deterministic id
                if student_id.isdigit():
                    numeric_label = int(student_id)
                else:
                    numeric_label = fallback_label_counter
                    fallback_label_counter += 1
                name = f"Student {student_id}"
                course = "General"

            # Supported image extensions
            image_paths = [
                p for p in s_dir.iterdir()
                if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
            ]

            student_valid_samples = 0

            for img_path in image_paths:
                try:
                    # Read image in grayscale
                    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        corrupted_count += 1
                        continue

                    # Validate dimensions
                    if not self.validate_face_image(img):
                        if strict_dimensions:
                            corrupted_count += 1
                            continue
                        else:
                            # Auto-resize to target size
                            img = cv2.resize(img, self.target_size, interpolation=cv2.INTER_AREA)
                            dimension_adjusted_count += 1

                    faces.append(img)
                    labels.append(numeric_label)
                    student_valid_samples += 1
                    valid_count += 1

                except Exception:
                    corrupted_count += 1
                    continue

            if student_valid_samples > 0:
                unique_student_ids.add(student_id)
                label_map[numeric_label] = {
                    "internal_id": numeric_label,
                    "student_id": student_id,
                    "name": name,
                    "course": course,
                    "samples_count": student_valid_samples
                }

        stats = {
            "total_images": valid_count,
            "corrupted_images": corrupted_count,
            "dimension_adjusted_images": dimension_adjusted_count,
            "unique_students": list(unique_student_ids),
            "student_count": len(unique_student_ids)
        }

        # Validate minimum requirements
        if valid_count == 0:
            raise InsufficientTrainingDataError(
                f"No valid face images found in dataset directory '{self.dataset_dir}'. "
                "Capture student face samples before training."
            )

        if len(unique_student_ids) < self.min_students:
            raise InsufficientTrainingDataError(
                f"LBPH model training requires at least {self.min_students} distinct registered students "
                f"for meaningful discrimination. Currently found {len(unique_student_ids)} student(s): "
                f"{list(unique_student_ids)}."
            )

        return faces, np.array(labels, dtype=np.int32), label_map, stats

    def train_model(
        self,
        progress_callback: Optional[Callable[[str, float], None]] = None,
        strict_dimensions: bool = False
    ) -> Dict[str, Any]:
        """
        Train the LBPH face recognition model on registered student images
        and save the resulting model and metadata.

        Args:
            progress_callback: Optional callback receiving (status_message, progress_fraction).
            strict_dimensions: If True, reject images not matching exact 200x200 size.

        Returns:
            Dictionary containing training summary metrics and model paths.

        Raises:
            DatasetNotFoundError: If dataset folder is missing.
            InsufficientTrainingDataError: If < min_students or 0 valid samples.
            ModelPersistenceError: If saving the model fails.
        """
        if progress_callback:
            progress_callback("Loading and validating face dataset...", 0.15)

        faces, labels, label_map, stats = self.load_dataset(strict_dimensions=strict_dimensions)

        if progress_callback:
            progress_callback(
                f"Training LBPH model with {len(faces)} samples across {stats['student_count']} students...",
                0.50
            )

        recognizer = self.create_recognizer()

        try:
            # Train the LBPH recognizer
            recognizer.train(faces, labels)
        except Exception as e:
            raise TrainingError(f"OpenCV LBPH training failed: {e}") from e

        if progress_callback:
            progress_callback("Saving model and metadata to disk...", 0.85)

        # Save model and metadata
        self.save_model(recognizer, label_map, stats)

        # Verify model loading
        self.load_model(self.model_path)

        if progress_callback:
            progress_callback("LBPH model trained and verified successfully!", 1.0)

        return {
            "success": True,
            "model_path": str(self.model_path),
            "metadata_path": str(self.metadata_path),
            "total_images": stats["total_images"],
            "unique_students": stats["student_count"],
            "students": stats["unique_students"],
            "label_map": label_map,
            "trained_at": datetime.now().isoformat()
        }

    def save_model(
        self,
        recognizer: cv2.face.LBPHFaceRecognizer,
        label_map: Dict[int, Dict[str, Any]],
        stats: Dict[str, Any]
    ) -> None:
        """
        Serialize the trained LBPH model to trainer.yml and write training metadata JSON.

        Args:
            recognizer: Trained LBPHFaceRecognizer instance.
            label_map: Label to student details mapping.
            stats: Training statistics.

        Raises:
            ModelPersistenceError: On write failures.
        """
        self.trainer_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Write OpenCV LBPH model file
            recognizer.write(str(self.model_path))

            if not self.model_path.exists() or self.model_path.stat().st_size == 0:
                raise ModelPersistenceError(f"Generated model file '{self.model_path}' is empty or missing.")

            # Write human-readable metadata for fast lookup without DB access if needed
            metadata = {
                "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "model_file": self.model_path.name,
                "total_images": stats["total_images"],
                "student_count": stats["student_count"],
                "students": stats["unique_students"],
                "hyperparameters": {
                    "radius": self.radius,
                    "neighbors": self.neighbors,
                    "grid_x": self.grid_x,
                    "grid_y": self.grid_y,
                    "threshold": self.threshold,
                    "target_size": list(self.target_size)
                },
                "label_map": {str(k): v for k, v in label_map.items()}
            }

            with open(self.metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=4)

        except Exception as e:
            raise ModelPersistenceError(f"Failed to persist LBPH model: {e}") from e

    def load_model(
        self,
        model_path: Optional[Union[str, Path]] = None
    ) -> cv2.face.LBPHFaceRecognizer:
        """
        Load and verify a trained LBPH model from YAML file.

        Args:
            model_path: Path to model YAML file. Defaults to self.model_path.

        Returns:
            Loaded cv2.face.LBPHFaceRecognizer instance ready for prediction.

        Raises:
            ModelPersistenceError: If file is missing or invalid.
        """
        path = Path(model_path or self.model_path)
        if not path.exists() or path.stat().st_size == 0:
            raise ModelPersistenceError(
                f"Model file '{path}' not found or empty. Please train the model first."
            )

        recognizer = self.create_recognizer()
        try:
            recognizer.read(str(path))
            return recognizer
        except Exception as e:
            raise ModelPersistenceError(f"Failed to read model from '{path}': {e}") from e

    def is_model_trained(self) -> bool:
        """
        Check if a valid, non-empty trained model file exists.
        """
        return self.model_path.exists() and self.model_path.stat().st_size > 0


# Module-level convenience runner
def train_face_recognition_model(
    dataset_dir: Optional[Union[str, Path]] = None,
    progress_callback: Optional[Callable[[str, float], None]] = None
) -> Dict[str, Any]:
    """
    Convenience function to instantiate LBPHTrainer and execute training.
    """
    trainer = LBPHTrainer(dataset_dir=dataset_dir)
    return trainer.train_model(progress_callback=progress_callback)
