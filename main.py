"""
SmartAttend - Face Recognition Attendance Management System
Main Application Entry Point.
"""

import sys
import os
import argparse
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import config
from database import db as default_db
from gui.dashboard import SmartAttendApp
from logger import get_logger, install_global_exception_hooks

logger = get_logger("SmartAttend.Main")
install_global_exception_hooks(logger)


def check_environment() -> None:
    """Verify that required directories and vital cascade files exist."""
    # Ensure vital directories exist
    for directory in [
        config.DATASET_DIR,
        config.TRAINER_DIR,
        config.CASCADES_DIR,
        config.EXPORTS_DIR,
        config.DATA_DIR
    ]:
        directory.mkdir(parents=True, exist_ok=True)

    # Check Haar Cascade model
    if not config.HAAR_CASCADE_PATH.exists():
        print(f"[WARNING] Haar Cascade file not found at '{config.HAAR_CASCADE_PATH}'.")


def main() -> None:
    """Launch SmartAttend desktop application."""
    # Enable High-DPI awareness on Windows to prevent scaling clipping or blurry window placement
    if sys.platform.startswith("win"):
        try:
            import ctypes
            try:
                ctypes.windll.shcore.SetProcessDpiAwareness(1)
            except Exception:
                ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass

    parser = argparse.ArgumentParser(description="SmartAttend - Face Recognition Attendance System")
    parser.add_argument("--version", action="version", version="SmartAttend 1.0.0")
    parser.add_argument("--train", action="store_true", help="Run LBPH model training from command line")
    args = parser.parse_args()

    check_environment()

    if args.train:
        from train_model import train_face_recognition_model
        logger.info("Initiating LBPH model training from CLI...")
        try:
            summary = train_face_recognition_model()
            logger.info(f"Model trained successfully: {summary['total_images']} images, {summary['unique_students']} students.")
        except Exception as e:
            logger.error(f"Training failed: {e}", exc_info=True)
        return

    # Launch Tkinter Dashboard Application
    logger.info("Launching SmartAttend Desktop Dashboard...")
    try:
        app = SmartAttendApp(db_manager=default_db)
        logger.info("SmartAttend window created and displayed successfully. Mainloop running.")
        app.run()
    except Exception as e:
        logger.critical(f"Failed to launch SmartAttend GUI: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
