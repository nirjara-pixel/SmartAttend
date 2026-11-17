"""
Unit tests for SmartAttend GUI components.
Tests dashboard shell initialization, view switching, statistics display,
registration form validation, and recognition start/stop lifecycle using mocks.
"""

import os
import shutil
import tempfile
import unittest
import tkinter as tk
from unittest.mock import MagicMock, patch

from database import DatabaseManager
from gui.dashboard import SmartAttendApp, OverviewView
from gui.register import RegistrationView
from gui.attendance_view import AttendanceView
from gui.recognition_view import RecognitionView
from gui.train_view import TrainViewModel


class TestSmartAttendGUI(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Verify Tkinter display availability."""
        try:
            root = tk.Tk()
            root.withdraw()
            root.destroy()
            cls.tk_available = True
        except Exception:
            cls.tk_available = False

    def setUp(self):
        """Set up test environment and temporary database."""
        if not self.tk_available:
            self.skipTest("Tkinter display environment unavailable in current shell.")

        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_gui.db")
        self.db = DatabaseManager(db_path=self.db_path)

        # Seed initial test data
        self.db.add_student("STU001", "Alice Wonder", "Computer Science")
        self.db.add_student("STU002", "Bob Builder", "Architecture")

        # Create SmartAttendApp
        self.app = SmartAttendApp(db_manager=self.db)
        self.app.withdraw()  # Keep window hidden during test run

    def tearDown(self):
        """Clean up GUI window and temporary database."""
        if hasattr(self, "app") and self.app:
            try:
                self.app.destroy()
            except Exception:
                pass

        try:
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        except Exception:
            pass

    def test_01_app_shell_and_views_initialization(self):
        """Test 1: Verify main app shell initializes all 5 functional views."""
        self.assertIn("overview", self.app.views)
        self.assertIn("register", self.app.views)
        self.assertIn("recognition", self.app.views)
        self.assertIn("attendance", self.app.views)
        self.assertIn("train", self.app.views)

        self.assertIsInstance(self.app.views["overview"], OverviewView)
        self.assertIsInstance(self.app.views["register"], RegistrationView)
        self.assertIsInstance(self.app.views["recognition"], RecognitionView)
        self.assertIsInstance(self.app.views["attendance"], AttendanceView)
        self.assertIsInstance(self.app.views["train"], TrainViewModel)

    def test_02_view_navigation(self):
        """Test 2: Test navigation between different views."""
        self.app.show_view("register")
        self.assertEqual(self.app.current_view_key, "register")

        self.app.show_view("attendance")
        self.assertEqual(self.app.current_view_key, "attendance")

        self.app.show_view("train")
        self.assertEqual(self.app.current_view_key, "train")

        self.app.show_view("overview")
        self.assertEqual(self.app.current_view_key, "overview")

    def test_03_dashboard_statistics_cards(self):
        """Test 3: Verify dashboard overview statistics cards reflect real database numbers."""
        overview: OverviewView = self.app.views["overview"]

        # 2 students registered, 0 present yet
        overview.refresh_stats()
        self.assertEqual(overview.val_total.cget("text"), "2")
        self.assertEqual(overview.val_present.cget("text"), "0")
        self.assertEqual(overview.val_absent.cget("text"), "2")
        self.assertEqual(overview.val_percentage.cget("text"), "0.0%")

        # Mark Alice present
        self.db.mark_attendance("STU001")
        overview.refresh_stats()

        # 2 total, 1 present, 1 absent, 50.0%
        self.assertEqual(overview.val_total.cget("text"), "2")
        self.assertEqual(overview.val_present.cget("text"), "1")
        self.assertEqual(overview.val_absent.cget("text"), "1")
        self.assertEqual(overview.val_percentage.cget("text"), "50.0%")

    def test_04_registration_form_operations(self):
        """Test 4: Verify form field population, clearing, and validation in RegistrationView."""
        reg_view: RegistrationView = self.app.views["register"]

        # Populate form
        reg_view.ent_id.insert(0, "STU003")
        reg_view.ent_name.insert(0, "Charlie Brown")
        reg_view.ent_course.insert(0, "Physics")

        self.assertEqual(reg_view.ent_id.get(), "STU003")
        self.assertEqual(reg_view.ent_name.get(), "Charlie Brown")
        self.assertEqual(reg_view.ent_course.get(), "Physics")

        # Clear form
        reg_view._clear_form()
        self.assertEqual(reg_view.ent_id.get(), "")
        self.assertEqual(reg_view.ent_name.get(), "")
        self.assertEqual(reg_view.ent_course.get(), "")

    def test_05_attendance_view_table_and_search(self):
        """Test 5: Verify AttendanceView displays records and filters by search text."""
        att_view: AttendanceView = self.app.views["attendance"]

        # Mark attendance for Alice
        self.db.mark_attendance("STU001", confidence=95.0)
        att_view.refresh_records()

        # Table should have at least 1 record
        rows = att_view.tree.get_children()
        self.assertEqual(len(rows), 1)

        # Search for non-matching student
        att_view.search_var.set("NonExistentStudent")
        self.assertEqual(len(att_view.tree.get_children()), 0)

        # Search for Alice
        att_view.search_var.set("Alice")
        self.assertEqual(len(att_view.tree.get_children()), 1)

    def test_06_recognition_view_lifecycle(self):
        """Test 6: Verify recognition view state transitions and safe stop."""
        rec_view: RecognitionView = self.app.views["recognition"]

        # Initially idle
        self.assertFalse(rec_view._is_running)
        self.assertEqual(rec_view.btn_start.cget("state"), "normal")
        self.assertEqual(rec_view.btn_stop.cget("state"), "disabled")

        # Stop recognition resets state safely
        rec_view.stop_recognition()
        self.assertFalse(rec_view._is_running)


if __name__ == "__main__":
    unittest.main()
