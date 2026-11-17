"""
Unit tests for train_model module.
Tests dataset loading, image validation, LBPH model training, serialization,
verification, and error handling using synthetic grayscale image data.
"""

import json
import os
import shutil
import tempfile
import unittest
import cv2
import numpy as np

from database import DatabaseManager
from train_model import (
    LBPHTrainer,
    DatasetNotFoundError,
    InsufficientTrainingDataError,
    ModelPersistenceError
)


class TestLBPHTrainer(unittest.TestCase):

    def setUp(self):
        """Set up temporary directories and synthetic dataset for testing."""
        self.temp_dir = tempfile.mkdtemp()
        self.dataset_dir = os.path.join(self.temp_dir, "dataset")
        self.trainer_dir = os.path.join(self.temp_dir, "trainer")
        self.model_path = os.path.join(self.trainer_dir, "trainer.yml")
        self.db_path = os.path.join(self.temp_dir, "test.db")

        # Initialize SQLite DB and register 2 students
        self.db = DatabaseManager(db_path=self.db_path)
        self.id_alice = self.db.add_student("STU001", "Alice Wonder", "Computer Science")
        self.id_bob = self.db.add_student("STU002", "Bob Builder", "Architecture")

        # Create student dataset directories
        self.alice_dir = os.path.join(self.dataset_dir, "STU001")
        self.bob_dir = os.path.join(self.dataset_dir, "STU002")
        os.makedirs(self.alice_dir, exist_ok=True)
        os.makedirs(self.bob_dir, exist_ok=True)

        # Generate synthetic 200x200 grayscale face images
        # Alice: brighter synthetic face pattern
        for i in range(1, 6):
            alice_img = np.full((200, 200), 180 + (i * 5), dtype=np.uint8)
            alice_img[50:150, 50:150] = 220
            cv2.imwrite(os.path.join(self.alice_dir, f"sample_{i:03d}.jpg"), alice_img)

        # Bob: darker synthetic face pattern
        for i in range(1, 6):
            bob_img = np.full((200, 200), 50 + (i * 5), dtype=np.uint8)
            bob_img[50:150, 50:150] = 90
            cv2.imwrite(os.path.join(self.bob_dir, f"sample_{i:03d}.jpg"), bob_img)

        # Initialize trainer instance
        self.trainer = LBPHTrainer(
            dataset_dir=self.dataset_dir,
            trainer_dir=self.trainer_dir,
            model_path=self.model_path,
            db_manager=self.db,
            min_students=2
        )

    def tearDown(self):
        """Clean up temporary test artifacts."""
        try:
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        except Exception:
            pass

    def test_01_dataset_loading(self):
        """Test 1: Load valid synthetic face images from student directories."""
        faces, labels, label_map, stats = self.trainer.load_dataset()

        # 5 images for Alice + 5 for Bob = 10 total
        self.assertEqual(len(faces), 10)
        self.assertEqual(len(labels), 10)
        self.assertEqual(stats["total_images"], 10)
        self.assertEqual(stats["student_count"], 2)
        self.assertEqual(set(stats["unique_students"]), {"STU001", "STU002"})

        # Check face image type and shape
        self.assertEqual(faces[0].shape, (200, 200))
        self.assertEqual(faces[0].dtype, np.uint8)

    def test_02_invalid_image_handling(self):
        """Test 2: Gracefully ignore corrupt or non-image files."""
        # Create a corrupted image file (plain text instead of valid image)
        corrupt_file = os.path.join(self.alice_dir, "corrupt_sample.jpg")
        with open(corrupt_file, "w") as f:
            f.write("This is not a valid JPEG image file.")

        faces, labels, label_map, stats = self.trainer.load_dataset()

        # Total valid images should still be 10, corrupted count should be 1
        self.assertEqual(len(faces), 10)
        self.assertEqual(stats["corrupted_images"], 1)

    def test_03_student_id_extraction(self):
        """Test 3: Extract correct numeric labels matching database primary keys."""
        faces, labels, label_map, stats = self.trainer.load_dataset()

        # Labels should match Alice and Bob's internal database IDs
        unique_labels = set(labels.tolist())
        self.assertEqual(unique_labels, {self.id_alice, self.id_bob})

        # Check label_map details
        self.assertEqual(label_map[self.id_alice]["name"], "Alice Wonder")
        self.assertEqual(label_map[self.id_alice]["student_id"], "STU001")
        self.assertEqual(label_map[self.id_bob]["name"], "Bob Builder")
        self.assertEqual(label_map[self.id_bob]["student_id"], "STU002")

    def test_04_image_dimension_validation(self):
        """Test 4: Validate 200x200 grayscale image dimensions."""
        # Valid 200x200 grayscale
        valid_img = np.zeros((200, 200), dtype=np.uint8)
        self.assertTrue(self.trainer.validate_face_image(valid_img))

        # Invalid: wrong dimensions
        wrong_dim = np.zeros((100, 100), dtype=np.uint8)
        self.assertFalse(self.trainer.validate_face_image(wrong_dim))

        # Invalid: 3-channel color image
        color_img = np.zeros((200, 200, 3), dtype=np.uint8)
        self.assertFalse(self.trainer.validate_face_image(color_img))

        # Non-array input
        self.assertFalse(self.trainer.validate_face_image(None))

        # Strict dimension mode in load_dataset
        non_200 = np.zeros((150, 150), dtype=np.uint8)
        cv2.imwrite(os.path.join(self.alice_dir, "sample_wrong_size.jpg"), non_200)

        faces, labels, label_map, stats = self.trainer.load_dataset(strict_dimensions=True)
        # Should reject the 150x150 image under strict mode
        self.assertEqual(len(faces), 10)
        self.assertEqual(stats["corrupted_images"], 1)

    def test_05_lbph_initialization(self):
        """Test 5: Create and configure LBPH recognizer instance."""
        recognizer = self.trainer.create_recognizer()
        self.assertIsNotNone(recognizer)
        self.assertTrue(isinstance(recognizer, cv2.face.LBPHFaceRecognizer))

    def test_06_model_training(self):
        """Test 6: Train LBPH model on synthetic dataset."""
        result = self.trainer.train_model()

        self.assertTrue(result["success"])
        self.assertEqual(result["total_images"], 10)
        self.assertEqual(result["unique_students"], 2)
        self.assertIn("STU001", result["students"])
        self.assertIn("STU002", result["students"])

    def test_07_model_saving(self):
        """Test 7: Verify trainer.yml and metadata.json are persisted to disk."""
        self.trainer.train_model()

        # Model YAML file must exist and be non-empty
        self.assertTrue(os.path.exists(self.model_path))
        self.assertGreater(os.path.getsize(self.model_path), 0)

        # Metadata JSON file must exist and be valid JSON
        metadata_path = os.path.join(self.trainer_dir, "metadata.json")
        self.assertTrue(os.path.exists(metadata_path))

        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
            self.assertEqual(metadata["student_count"], 2)
            self.assertEqual(metadata["total_images"], 10)
            self.assertIn("hyperparameters", metadata)

    def test_08_model_loading(self):
        """Test 8: Verify saved model can be reloaded and performs predictions."""
        self.trainer.train_model()

        # Load trained model
        loaded_model = self.trainer.load_model(self.model_path)
        self.assertIsNotNone(loaded_model)

        # Test prediction on Alice's sample image
        alice_sample = cv2.imread(
            os.path.join(self.alice_dir, "sample_001.jpg"),
            cv2.IMREAD_GRAYSCALE
        )
        predicted_label, confidence = loaded_model.predict(alice_sample)

        # Recognizer should predict Alice's ID with low distance (high confidence)
        self.assertEqual(predicted_label, self.id_alice)
        self.assertLessEqual(confidence, 15.0)

    def test_09_missing_dataset_handling(self):
        """Test 9: Missing dataset folder raises DatasetNotFoundError."""
        trainer_missing = LBPHTrainer(
            dataset_dir=os.path.join(self.temp_dir, "non_existent_dir")
        )
        with self.assertRaises(DatasetNotFoundError):
            trainer_missing.load_dataset()

    def test_10_insufficient_training_data_handling(self):
        """Test 10: Insufficient distinct students or zero samples raises InsufficientTrainingDataError."""
        # Case A: Empty dataset directory (0 samples)
        empty_dataset = os.path.join(self.temp_dir, "empty_dataset")
        os.makedirs(empty_dataset, exist_ok=True)
        trainer_empty = LBPHTrainer(dataset_dir=empty_dataset)
        with self.assertRaises(InsufficientTrainingDataError):
            trainer_empty.load_dataset()

        # Case B: Only 1 student available (needs at least 2)
        single_student_dataset = os.path.join(self.temp_dir, "single_student")
        os.makedirs(os.path.join(single_student_dataset, "STU001"), exist_ok=True)
        dummy_img = np.zeros((200, 200), dtype=np.uint8)
        cv2.imwrite(os.path.join(single_student_dataset, "STU001", "sample_001.jpg"), dummy_img)

        trainer_single = LBPHTrainer(dataset_dir=single_student_dataset, min_students=2)
        with self.assertRaises(InsufficientTrainingDataError):
            trainer_single.load_dataset()


if __name__ == "__main__":
    unittest.main()
