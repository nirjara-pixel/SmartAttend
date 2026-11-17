"""
Unit tests for face_recognition module.
Tests model loading, distance-to-confidence scoring, prediction,
and frame recognition handling.
"""

import os
import shutil
import tempfile
import unittest
import cv2
import numpy as np

from database import DatabaseManager
from face_detector import FaceDetector
from face_recognition import FaceRecognizer, ModelNotFoundError, RecognitionResult
from train_model import LBPHTrainer


class TestFaceRecognizer(unittest.TestCase):

    def setUp(self):
        """Train a minimal synthetic model and set up FaceRecognizer."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test.db")
        self.dataset_dir = os.path.join(self.temp_dir, "dataset")
        self.trainer_dir = os.path.join(self.temp_dir, "trainer")
        self.model_path = os.path.join(self.trainer_dir, "trainer.yml")

        self.db = DatabaseManager(db_path=self.db_path)
        self.id_alice = self.db.add_student("STU001", "Alice Wonder", "CS")
        self.id_bob = self.db.add_student("STU002", "Bob Builder", "Civil")

        # Create synthetic datasets
        alice_dir = os.path.join(self.dataset_dir, "STU001")
        bob_dir = os.path.join(self.dataset_dir, "STU002")
        os.makedirs(alice_dir, exist_ok=True)
        os.makedirs(bob_dir, exist_ok=True)

        for i in range(1, 4):
            img_a = np.full((200, 200), 200, dtype=np.uint8)
            cv2.imwrite(os.path.join(alice_dir, f"sample_{i:03d}.jpg"), img_a)

            img_b = np.full((200, 200), 30, dtype=np.uint8)
            cv2.imwrite(os.path.join(bob_dir, f"sample_{i:03d}.jpg"), img_b)

        # Train model
        trainer = LBPHTrainer(
            dataset_dir=self.dataset_dir,
            trainer_dir=self.trainer_dir,
            model_path=self.model_path,
            db_manager=self.db,
            min_students=2
        )
        trainer.train_model()

        # Initialize recognizer
        self.recognizer = FaceRecognizer(
            model_path=self.model_path,
            db_manager=self.db,
            threshold=75.0
        )

    def tearDown(self):
        """Clean up temporary test artifacts."""
        try:
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        except Exception:
            pass

    def test_missing_model_file(self):
        """Test that missing model file raises ModelNotFoundError."""
        with self.assertRaises(ModelNotFoundError):
            FaceRecognizer(
                model_path=os.path.join(self.temp_dir, "non_existent.yml"),
                db_manager=self.db
            )

    def test_calculate_confidence_score(self):
        """Test distance to percentage confidence calculation."""
        # Distance 0 -> 100%
        self.assertAlmostEqual(self.recognizer.calculate_confidence_score(0.0), 100.0)
        # Distance threshold (75) -> 50%
        self.assertAlmostEqual(self.recognizer.calculate_confidence_score(75.0), 50.0)
        # Distance 37.5 -> 75%
        self.assertAlmostEqual(self.recognizer.calculate_confidence_score(37.5), 75.0)
        # Large distance -> 0.0%
        self.assertAlmostEqual(self.recognizer.calculate_confidence_score(200.0), 0.0)

    def test_predict_recognized_face(self):
        """Test predicting a known face returns matching student record."""
        alice_sample = np.full((200, 200), 200, dtype=np.uint8)
        result = self.recognizer.predict_face(alice_sample)

        self.assertTrue(result.is_recognized)
        self.assertEqual(result.student_id, "STU001")
        self.assertEqual(result.name, "Alice Wonder")
        self.assertEqual(result.course, "CS")
        self.assertGreater(result.confidence, 50.0)

    def test_predict_unknown_face(self):
        """Test predicting an unknown face with high distance returns Unknown."""
        # Set artificially tiny threshold to force unknown
        strict_recognizer = FaceRecognizer(
            model_path=self.model_path,
            db_manager=self.db,
            threshold=0.0001
        )
        sample = np.full((200, 200), 128, dtype=np.uint8)
        result = strict_recognizer.predict_face(sample)

        self.assertFalse(result.is_recognized)
        self.assertTrue(result.is_unknown)
        self.assertEqual(result.name, "Unknown")
        self.assertIsNone(result.student_id)


if __name__ == "__main__":
    unittest.main()
