"""
Unit tests for face_detector module.
"""

import unittest
import numpy as np
import cv2
from face_detector import FaceDetector
import config


class TestFaceDetector(unittest.TestCase):

    def setUp(self):
        self.detector = FaceDetector()

    def test_detector_initialization(self):
        """Test that the detector initializes and cascade is loaded."""
        self.assertIsNotNone(self.detector.cascade)
        self.assertFalse(self.detector.cascade.empty())

    def test_invalid_cascade_path(self):
        """Test that an invalid cascade path raises FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            FaceDetector(cascade_path="non_existent_file.xml")

    def test_detect_faces_empty_frame(self):
        """Test that None or empty frames return an empty list."""
        self.assertEqual(self.detector.detect_faces(None), [])
        empty_array = np.array([], dtype=np.uint8)
        self.assertEqual(self.detector.detect_faces(empty_array), [])

    def test_detect_faces_blank_frame(self):
        """Test detection on a blank image returns no faces."""
        blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        faces = self.detector.detect_faces(blank_frame)
        self.assertIsInstance(faces, list)
        self.assertEqual(len(faces), 0)

    def test_extract_face_roi_shape_and_type(self):
        """Test that extracted ROI is correctly resized and grayscale."""
        dummy_frame = np.ones((480, 640, 3), dtype=np.uint8) * 128
        bbox = (100, 100, 150, 150)
        roi = self.detector.extract_face_roi(dummy_frame, bbox, target_size=(200, 200))

        self.assertEqual(roi.shape, (200, 200))
        self.assertEqual(roi.dtype, np.uint8)

    def test_extract_face_roi_boundary_clamping(self):
        """Test that bbox crossing boundary doesn't crash and returns valid shape."""
        dummy_frame = np.ones((480, 640, 3), dtype=np.uint8) * 128
        # Boundary-exceeding box
        bbox = (-20, 400, 200, 200)
        roi = self.detector.extract_face_roi(dummy_frame, bbox, target_size=(200, 200))

        self.assertEqual(roi.shape, (200, 200))
        self.assertEqual(roi.dtype, np.uint8)

    def test_draw_face_box(self):
        """Test drawing bounding box and labels on a frame."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        bbox = (50, 50, 100, 100)
        annotated = FaceDetector.draw_face_box(
            frame,
            bbox,
            primary_text="John Doe",
            secondary_text="ID: 101 | 92%",
            color=(0, 255, 0)
        )
        self.assertEqual(annotated.shape, (480, 640, 3))


if __name__ == "__main__":
    unittest.main()
