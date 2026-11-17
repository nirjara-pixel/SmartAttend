"""
Unit tests for student_registration module.
Tests student validation, dataset storage, webcam capture mocking,
database integration, and failure cleanup without requiring physical hardware.
"""

import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock
import numpy as np
import cv2

from database import DatabaseManager, DuplicateStudentError, ValidationError
from face_detector import FaceDetector
from student_registration import (
    StudentRegistration,
    WebcamError,
    RegistrationCancelledError,
    InsufficientSamplesError
)


class MockVideoCapture:
    """Mock OpenCV VideoCapture for headless testing."""

    def __init__(self, is_opened=True, frames=None):
        self._is_opened = is_opened
        self._frames = frames or []
        self._index = 0
        self.released = False

    def isOpened(self):
        return self._is_opened

    def set(self, prop, value):
        return True

    def read(self):
        if not self._is_opened or not self._frames:
            return False, None
        frame = self._frames[self._index % len(self._frames)]
        self._index += 1
        return True, frame.copy()

    def release(self):
        self._is_opened = False
        self.released = True


class TestStudentRegistration(unittest.TestCase):

    def setUp(self):
        """Create temporary database and dataset root for isolated testing."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_smartattend.db")
        self.dataset_dir = os.path.join(self.temp_dir, "dataset")

        self.db = DatabaseManager(db_path=self.db_path)
        self.mock_detector = MagicMock(spec=FaceDetector)
        # Default mock: detects 1 face and returns 200x200 uint8 ROI
        self.mock_detector.detect_faces.return_value = [(50, 50, 100, 100)]
        self.mock_detector.extract_face_roi.return_value = np.zeros((200, 200), dtype=np.uint8)

        self.reg = StudentRegistration(
            db_manager=self.db,
            face_detector=self.mock_detector,
            dataset_base_dir=self.dataset_dir,
            target_samples=5  # Small sample count for fast testing
        )

    def tearDown(self):
        """Clean up temporary directories and database."""
        try:
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        except Exception:
            pass

    def test_01_student_info_validation(self):
        """Test 1: Validate input fields for student_id, name, and course."""
        # Empty student ID
        with self.assertRaises(ValidationError):
            self.reg.validate_student_data("", "John Doe", "CS")

        # Empty name
        with self.assertRaises(ValidationError):
            self.reg.validate_student_data("STU101", "   ", "CS")

        # Empty course
        with self.assertRaises(ValidationError):
            self.reg.validate_student_data("STU101", "John Doe", "")

        # Invalid filesystem characters
        with self.assertRaises(ValidationError):
            self.reg.validate_student_data("STU/101", "John Doe", "CS")

        # Valid input returns sanitized tuple
        clean_id, clean_name, clean_course = self.reg.validate_student_data(
            "  STU101  ", " John Doe ", " CS "
        )
        self.assertEqual(clean_id, "STU101")
        self.assertEqual(clean_name, "John Doe")
        self.assertEqual(clean_course, "CS")

    def test_02_duplicate_student_id_handling(self):
        """Test 2: Prevent duplicate student registration in SQLite or on disk."""
        # Register student in database
        self.db.add_student("STU201", "Jane Doe", "Data Science")

        # Attempt to validate existing student ID must raise DuplicateStudentError
        with self.assertRaises(DuplicateStudentError):
            self.reg.validate_student_data("STU201", "Another Student", "Data Science")

        # Dataset folder already existing with content should also raise DuplicateStudentError
        stu_folder = self.reg.get_student_dataset_dir("STU202")
        stu_folder.mkdir(parents=True, exist_ok=True)
        (stu_folder / "sample_001.jpg").touch()

        with self.assertRaises(DuplicateStudentError):
            self.reg.validate_student_data("STU202", "Disk Student", "IT")

    def test_03_dataset_directory_creation(self):
        """Test 3: Dataset directory is created within dataset base path."""
        stu_dir = self.reg.get_student_dataset_dir("STU301")
        self.assertFalse(stu_dir.exists())

        # Saving a sample should automatically create the directory
        dummy_face = np.zeros((200, 200), dtype=np.uint8)
        self.reg.save_face_sample(dummy_face, "STU301", 1)
        self.assertTrue(stu_dir.exists())

    def test_04_face_image_saving(self):
        """Test 4: Face image is correctly written to disk and can be reloaded."""
        # Create a face image array
        face = np.full((200, 200), 128, dtype=np.uint8)
        saved_path = self.reg.save_face_sample(face, "STU401", 1)

        self.assertTrue(saved_path.exists())
        loaded = cv2.imread(str(saved_path), cv2.IMREAD_GRAYSCALE)
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.shape, (200, 200))
        np.testing.assert_array_equal(loaded, face)

    def test_05_image_dimensions(self):
        """Test 5: Reject invalid dimensions or non-grayscale arrays."""
        # Wrong dimensions (150x150 instead of 200x200)
        wrong_size = np.zeros((150, 150), dtype=np.uint8)
        with self.assertRaises(ValidationError):
            self.reg.save_face_sample(wrong_size, "STU501", 1)

        # 3-channel color instead of 2D grayscale
        color_image = np.zeros((200, 200, 3), dtype=np.uint8)
        with self.assertRaises(ValidationError):
            self.reg.save_face_sample(color_image, "STU501", 1)

    def test_06_sequential_filenames(self):
        """Test 6: Face samples follow sequential naming pattern (sample_001.jpg, sample_002.jpg)."""
        dummy_face = np.zeros((200, 200), dtype=np.uint8)
        p1 = self.reg.save_face_sample(dummy_face, "STU601", 1)
        p2 = self.reg.save_face_sample(dummy_face, "STU601", 2)
        p3 = self.reg.save_face_sample(dummy_face, "STU601", 10)

        self.assertEqual(p1.name, "sample_001.jpg")
        self.assertEqual(p2.name, "sample_002.jpg")
        self.assertEqual(p3.name, "sample_010.jpg")

    def test_07_webcam_initialization_failure(self):
        """Test 7: Raising WebcamError when webcam cannot be opened."""
        mock_cap = MockVideoCapture(is_opened=False)
        with self.assertRaises(WebcamError):
            self.reg.capture_and_register(
                student_id="STU701",
                name="Camera Test",
                course="Engineering",
                video_capture=mock_cap,
                show_preview=False
            )

    def test_08_registration_cancellation(self):
        """Test 8: Registration cancellation via external stop flag."""
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_cap = MockVideoCapture(is_opened=True, frames=[dummy_frame])

        # Flag that immediately requests abort
        stop_flag = lambda: True

        with self.assertRaises(RegistrationCancelledError):
            self.reg.capture_and_register(
                student_id="STU801",
                name="Cancel Test",
                course="Engineering",
                target_samples=5,
                video_capture=mock_cap,
                show_preview=False,
                stop_flag=stop_flag
            )

    def test_09_database_integration(self):
        """Test 9: Full workflow registers student in SQLite and returns proper metadata."""
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_cap = MockVideoCapture(is_opened=True, frames=[dummy_frame])

        result = self.reg.capture_and_register(
            student_id="STU901",
            name="Marie Curie",
            course="Chemistry",
            target_samples=3,
            video_capture=mock_cap,
            show_preview=False
        )

        self.assertTrue(result["success"])
        self.assertEqual(result["student_id"], "STU901")
        self.assertEqual(result["name"], "Marie Curie")
        self.assertEqual(result["sample_count"], 3)

        # Verify record exists in SQLite
        db_student = self.db.get_student("STU901")
        self.assertIsNotNone(db_student)
        self.assertEqual(db_student["name"], "Marie Curie")
        self.assertEqual(db_student["sample_count"], 3)

        # Verify dataset folder has 3 images
        dataset_files = list(self.reg.get_student_dataset_dir("STU901").glob("*.jpg"))
        self.assertEqual(len(dataset_files), 3)

    def test_10_dataset_cleanup_after_failure(self):
        """Test 10: Incomplete dataset is cleaned up if an error occurs mid-capture."""
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_cap = MockVideoCapture(is_opened=True, frames=[dummy_frame])

        # Stop flag triggers after 2 samples are written
        call_count = [0]
        def conditional_stop():
            call_count[0] += 1
            return call_count[0] >= 3

        with self.assertRaises(RegistrationCancelledError):
            self.reg.capture_and_register(
                student_id="STU1001",
                name="Cleanup Test",
                course="Robotics",
                target_samples=10,
                video_capture=mock_cap,
                show_preview=False,
                stop_flag=conditional_stop
            )

        # Directory must be removed cleanly
        student_dir = self.reg.get_student_dataset_dir("STU1001")
        self.assertFalse(student_dir.exists(), "Dataset directory must be cleaned up on failure.")

        # Student must NOT be registered in database
        self.assertIsNone(self.db.get_student("STU1001"))


if __name__ == "__main__":
    unittest.main()
