"""
SmartAttend - Face Detection Module
Implements robust face detection, ROI extraction, and preprocessing using OpenCV Haar Cascade Classifier.
"""

import os
from typing import List, Tuple, Optional
import cv2
import numpy as np

import config


class FaceDetector:
    """
    Haar Cascade-based Face Detector.
    Handles loading classifiers, detecting faces in frames,
    extracting normalized face ROIs, and drawing UI overlays.
    """

    def __init__(self, cascade_path: Optional[str] = None):
        """
        Initialize the Face Detector with a Haar Cascade XML file.

        Args:
            cascade_path: Path to haarcascade_frontalface_default.xml.
                          Defaults to config.HAAR_CASCADE_PATH.
        """
        self.cascade_path = str(cascade_path or config.HAAR_CASCADE_PATH)
        self.cascade = self._load_cascade(self.cascade_path)

    def _load_cascade(self, path: str) -> cv2.CascadeClassifier:
        """
        Safely load the Haar cascade classifier, with fallback to OpenCV's built-in data.
        """
        if os.path.exists(path):
            classifier = cv2.CascadeClassifier(path)
            if not classifier.empty():
                return classifier

        # Fallback to cv2.data if available
        fallback_path = os.path.join(
            getattr(cv2.data, "haarcascades", ""),
            "haarcascade_frontalface_default.xml"
        )
        if os.path.exists(fallback_path):
            classifier = cv2.CascadeClassifier(fallback_path)
            if not classifier.empty():
                return classifier

        raise FileNotFoundError(
            f"Could not load Haar Cascade XML from '{path}' or fallback '{fallback_path}'. "
            "Please ensure haarcascade_frontalface_default.xml is present."
        )

    def detect_faces(
        self,
        frame: np.ndarray,
        scale_factor: Optional[float] = None,
        min_neighbors: Optional[int] = None,
        min_size: Optional[Tuple[int, int]] = None
    ) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces in a video frame or image.

        Args:
            frame: Input image (BGR or Grayscale numpy array).
            scale_factor: Parameter specifying how much the image size is reduced at each scale.
            min_neighbors: Parameter specifying how many neighbors each candidate rectangle should have.
            min_size: Minimum possible object size. Objects smaller than this are ignored.

        Returns:
            List of bounding boxes [(x, y, w, h), ...] sorted by area descending (largest first).
        """
        if frame is None or frame.size == 0:
            return []

        # Convert to grayscale if BGR
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.copy()

        # Histogram equalization improves contrast under varied lighting
        equalized = cv2.equalizeHist(gray)

        sf = scale_factor or config.HAAR_SCALE_FACTOR
        mn = min_neighbors or config.HAAR_MIN_NEIGHBORS
        ms = min_size or config.HAAR_MIN_FACE_SIZE

        faces = self.cascade.detectMultiScale(
            equalized,
            scaleFactor=sf,
            minNeighbors=mn,
            minSize=ms,
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        if len(faces) == 0:
            return []

        # Convert to list of tuples and sort by bounding box area (w * h) descending
        face_list = [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]
        face_list.sort(key=lambda box: box[2] * box[3], reverse=True)
        return face_list

    def extract_face_roi(
        self,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
        target_size: Optional[Tuple[int, int]] = None
    ) -> np.ndarray:
        """
        Extract, crop, normalize, and resize a face Region of Interest (ROI).

        Args:
            frame: Input image (BGR or Grayscale).
            bbox: Bounding box tuple (x, y, w, h).
            target_size: Desired (width, height) output for LBPH. Defaults to config.NORMALIZED_FACE_SIZE.

        Returns:
            Processed 2D grayscale face array equalized and resized to target_size.
        """
        target = target_size or config.NORMALIZED_FACE_SIZE
        x, y, w, h = bbox
        img_h, img_w = frame.shape[:2]

        # Clamp boundaries to valid frame dimensions
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(img_w, x + w)
        y2 = min(img_h, y + h)

        if x2 <= x1 or y2 <= y1:
            # Fallback zero array if invalid box
            return np.zeros((target[1], target[0]), dtype=np.uint8)

        # Crop ROI
        roi = frame[y1:y2, x1:x2]

        # Convert to grayscale if necessary
        if len(roi.shape) == 3:
            roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            roi_gray = roi.copy()

        # Apply histogram equalization for robust illumination invariance
        roi_equalized = cv2.equalizeHist(roi_gray)

        # Resize to fixed dimension for LBPH training / inference
        roi_resized = cv2.resize(roi_equalized, target, interpolation=cv2.INTER_CUBIC)
        return roi_resized

    @staticmethod
    def draw_face_box(
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
        primary_text: str = "Face",
        secondary_text: Optional[str] = None,
        color: Tuple[int, int, int] = (0, 255, 0),
        corner_length: int = 15,
        thickness: int = 2
    ) -> np.ndarray:
        """
        Draw a stylish bounding box with corner accents and label badge on the frame.

        Args:
            frame: Image array to draw on (modified in-place or returned).
            bbox: (x, y, w, h) bounding box.
            primary_text: Main header text (e.g. Student Name or 'Unknown').
            secondary_text: Optional subtitle (e.g. ID, Course, Confidence %).
            color: BGR tuple for box and badge (e.g. (0, 255, 0) for recognized, (0, 0, 255) for unknown).
            corner_length: Pixel length of stylish corner brackets.
            thickness: Line thickness.

        Returns:
            The annotated frame.
        """
        x, y, w, h = bbox

        # Main bounding rectangle
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)

        # High-tech corner accents for professional HUD aesthetic
        cl = min(corner_length, w // 4, h // 4)
        c_thick = thickness + 1
        # Top-Left
        cv2.line(frame, (x, y), (x + cl, y), color, c_thick)
        cv2.line(frame, (x, y), (x, y + cl), color, c_thick)
        # Top-Right
        cv2.line(frame, (x + w, y), (x + w - cl, y), color, c_thick)
        cv2.line(frame, (x + w, y), (x + w, y + cl), color, c_thick)
        # Bottom-Left
        cv2.line(frame, (x, y + h), (x + cl, y + h), color, c_thick)
        cv2.line(frame, (x, y + h), (x, y + h - cl), color, c_thick)
        # Bottom-Right
        cv2.line(frame, (x + w, y + h), (x + w - cl, y + h), color, c_thick)
        cv2.line(frame, (x + w, y + h), (x + w, y + h - cl), color, c_thick)

        # Prepare label badge
        lines = [primary_text]
        if secondary_text:
            lines.append(secondary_text)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.55
        line_height = 20
        badge_padding = 6
        badge_height = (len(lines) * line_height) + (2 * badge_padding)

        # Compute max text width
        max_w = 0
        for line in lines:
            (tw, th), _ = cv2.getTextSize(line, font, font_scale, 1)
            if tw > max_w:
                max_w = tw
        badge_width = max(w, max_w + (2 * badge_padding))

        # Position badge above or below depending on space
        if y - badge_height > 10:
            badge_y1 = y - badge_height
            badge_y2 = y
        else:
            badge_y1 = y + h
            badge_y2 = y + h + badge_height

        badge_x1 = x
        badge_x2 = x + badge_width

        # Ensure badge stays within frame
        img_h, img_w = frame.shape[:2]
        badge_x2 = min(img_w - 2, badge_x2)
        badge_x1 = max(2, badge_x1)

        # Draw filled badge background
        cv2.rectangle(frame, (badge_x1, badge_y1), (badge_x2, badge_y2), color, cv2.FILLED)

        # Draw text inside badge (dark text on bright badge, or white on dark)
        # Check brightness of badge color
        brightness = (color[0] * 0.114 + color[1] * 0.587 + color[2] * 0.299)
        text_color = (0, 0, 0) if brightness > 140 else (255, 255, 255)

        for i, line in enumerate(lines):
            text_y = badge_y1 + badge_padding + ((i + 1) * line_height) - 4
            cv2.putText(
                frame,
                line,
                (badge_x1 + badge_padding, text_y),
                font,
                font_scale,
                text_color,
                1,
                cv2.LINE_AA
            )

        return frame
