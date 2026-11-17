"""
SmartAttend - Configuration Module
Defines global configurations, directory paths, camera parameters, and model thresholds.
"""

import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
TRAINER_DIR = BASE_DIR / "trainer"
CASCADES_DIR = BASE_DIR / "cascades"
EXPORTS_DIR = BASE_DIR / "exports"
DATA_DIR = BASE_DIR / "data"
LOGS_DIR = BASE_DIR / "logs"

# Ensure all vital directories exist
for directory in [DATASET_DIR, TRAINER_DIR, CASCADES_DIR, EXPORTS_DIR, DATA_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Database, Model and Log File Paths
DATABASE_PATH = DATA_DIR / "smartattend.db"
MODEL_PATH = TRAINER_DIR / "trainer.yml"
HAAR_CASCADE_PATH = CASCADES_DIR / "haarcascade_frontalface_default.xml"
LOG_FILE = LOGS_DIR / "app.log"

# Camera Settings
DEFAULT_CAMERA_INDEX = 0
CAMERA_FRAME_WIDTH = 640
CAMERA_FRAME_HEIGHT = 480
CAMERA_FPS = 30

# Face Detection Settings (Haar Cascade)
HAAR_SCALE_FACTOR = 1.15
HAAR_MIN_NEIGHBORS = 5
HAAR_MIN_FACE_SIZE = (80, 80)

# Student Registration Settings
SAMPLES_PER_STUDENT = 30
SAMPLE_CAPTURE_INTERVAL = 0.15  # seconds between captures
NORMALIZED_FACE_SIZE = (200, 200)

# LBPH Face Recognizer Settings
# In OpenCV LBPH, distance is a measure of dissimilarity:
# 0 = exact match. Higher value = less confident.
# Distance below CONFIDENCE_THRESHOLD is considered a valid match.
LBPH_RADIUS = 1
LBPH_NEIGHBORS = 8
LBPH_GRID_X = 8
LBPH_GRID_Y = 8
DEFAULT_CONFIDENCE_THRESHOLD = 75.0

# Attendance Logic Settings
# Cooldown period (in seconds) to avoid duplicate consecutive recognition alerts
ATTENDANCE_COOLDOWN_SECONDS = 30

# UI Theme & Styling Colors
THEME = {
    "bg_main": "#0f172a",          # Deep slate background
    "bg_surface": "#1e293b",       # Card / container background
    "bg_surface_alt": "#334155",   # Input / hover background
    "accent": "#3b82f6",           # Vibrant blue
    "accent_hover": "#2563eb",     # Darker blue on hover
    "success": "#10b981",          # Emerald green
    "warning": "#f59e0b",          # Amber
    "danger": "#ef4444",           # Crimson red
    "text_primary": "#f8fafc",     # High-contrast white
    "text_secondary": "#94a3b8",   # Soft slate gray
    "border": "#475569",           # Subtle border
}

FONTS = {
    "h1": ("Segoe UI", 18, "bold"),
    "h2": ("Segoe UI", 14, "bold"),
    "h3": ("Segoe UI", 12, "bold"),
    "body": ("Segoe UI", 10),
    "body_bold": ("Segoe UI", 10, "bold"),
    "caption": ("Segoe UI", 9),
    "stat_val": ("Segoe UI", 22, "bold"),
}
