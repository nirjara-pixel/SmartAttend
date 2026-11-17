"""
Unit tests for attendance_manager module.
Tests successful attendance marking, duplicate prevention, cooldown debouncing,
unknown face handling, invalid student IDs, database errors, multiple students,
and timestamp recording.
"""

import os
import shutil
import tempfile
import unittest
from datetime import datetime
from unittest.mock import MagicMock

import pandas as pd

from attendance_manager import AttendanceManager, AttendanceStatus
from database import DatabaseError, DatabaseManager
from face_recognition import RecognitionResult


class TestAttendanceManager(unittest.TestCase):

    def setUp(self):
        """Set up an isolated temporary database for testing."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_smartattend.db")
        self.db = DatabaseManager(db_path=self.db_path)

        # Register test students
        self.db.add_student("STU001", "Alice Wonder", "Computer Science")
        self.db.add_student("STU002", "Bob Builder", "Civil Engineering")
        self.db.add_student("STU003", "Charlie Brown", "Physics")

        # Initialize AttendanceManager with short cooldown for testing
        self.manager = AttendanceManager(
            db_manager=self.db,
            cooldown_seconds=10.0
        )

    def tearDown(self):
        """Clean up temporary directory."""
        try:
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        except Exception:
            pass

    def test_successful_attendance_marking(self):
        """Test marking attendance successfully for a registered student."""
        event = self.manager.mark_student_attendance(
            student_id="STU001",
            confidence=92.5,
            date_str="2026-11-17",
            time_str="09:15:00"
        )

        self.assertEqual(event.status, AttendanceStatus.MARKED)
        self.assertTrue(event.is_success)
        self.assertEqual(event.student_id, "STU001")
        self.assertEqual(event.name, "Alice Wonder")
        self.assertEqual(event.course, "Computer Science")
        self.assertEqual(event.date, "2026-11-17")
        self.assertEqual(event.time, "09:15:00")
        self.assertIn("Alice Wonder", event.message)

        # Verify persistence in SQLite
        records = self.db.get_attendance_by_date("2026-11-17")
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["student_id"], "STU001")
        self.assertEqual(records[0]["name"], "Alice Wonder")

    def test_duplicate_attendance_prevention(self):
        """Test that duplicate attendance on the same day is rejected."""
        # First attendance succeeds
        event1 = self.manager.mark_student_attendance(
            student_id="STU001",
            confidence=90.0,
            date_str="2026-11-17"
        )
        self.assertEqual(event1.status, AttendanceStatus.MARKED)

        # Clear in-memory cooldown so that we specifically test the database duplicate constraint
        self.manager.clear_cooldown_cache()

        # Second attendance attempt on same day must return ALREADY_MARKED
        event2 = self.manager.mark_student_attendance(
            student_id="STU001",
            confidence=95.0,
            date_str="2026-11-17"
        )
        self.assertEqual(event2.status, AttendanceStatus.ALREADY_MARKED)
        self.assertTrue(event2.is_already_marked)
        self.assertIn("already marked", event2.message)

        # Verify only 1 record exists in SQLite
        records = self.db.get_attendance_by_date("2026-11-17")
        self.assertEqual(len(records), 1)

    def test_cooldown_debouncing(self):
        """Test that repeated recognition events within cooldown window avoid DB writes."""
        # 1. First recognition event
        event1 = self.manager.mark_student_attendance(
            student_id="STU002",
            confidence=88.0,
            date_str="2026-11-17"
        )
        self.assertEqual(event1.status, AttendanceStatus.MARKED)

        # 2. Immediate second recognition event (without clearing cooldown cache)
        event2 = self.manager.mark_student_attendance(
            student_id="STU002",
            confidence=89.0,
            date_str="2026-11-17"
        )
        self.assertEqual(event2.status, AttendanceStatus.COOLDOWN)
        self.assertTrue(event2.is_ignored)
        self.assertIn("Cooldown active", event2.message)

    def test_unknown_face_handling(self):
        """Test that unknown/unrecognized faces are safely ignored."""
        unknown_result = RecognitionResult(
            is_recognized=False,
            student_id=None,
            name="Unknown",
            course=None,
            internal_id=None,
            confidence=0.0,
            distance=85.0,
            bbox=(10, 10, 100, 100)
        )

        event = self.manager.process_recognition_result(unknown_result)
        self.assertEqual(event.status, AttendanceStatus.UNKNOWN_FACE)
        self.assertTrue(event.is_ignored)
        self.assertIn("Unknown face detected", event.message)

        # No records added to database
        today_records = self.db.get_today_attendance()
        self.assertEqual(len(today_records), 0)

    def test_invalid_student_id(self):
        """Test handling of empty or non-existent student IDs."""
        # Empty student ID
        event_empty = self.manager.mark_student_attendance("")
        self.assertEqual(event_empty.status, AttendanceStatus.INVALID_STUDENT)

        # Non-existent student ID
        event_missing = self.manager.mark_student_attendance("NON_EXISTENT_ID")
        self.assertEqual(event_missing.status, AttendanceStatus.INVALID_STUDENT)
        self.assertIn("not registered", event_missing.message)

    def test_database_errors(self):
        """Test graceful error handling when database operations fail."""
        mock_db = MagicMock(spec=DatabaseManager)
        mock_db.get_student.return_value = {"student_id": "STU001", "name": "Alice", "course": "CS"}
        mock_db.mark_attendance.side_effect = DatabaseError("Disk I/O failure")

        err_manager = AttendanceManager(db_manager=mock_db)
        event = err_manager.mark_student_attendance("STU001")

        self.assertEqual(event.status, AttendanceStatus.ERROR)
        self.assertIn("Database error", event.message)

    def test_multiple_students(self):
        """Test marking attendance for multiple distinct students."""
        date_str = "2026-11-17"
        e1 = self.manager.mark_student_attendance("STU001", 90.0, date_str=date_str, time_str="09:00:00")
        e2 = self.manager.mark_student_attendance("STU002", 85.0, date_str=date_str, time_str="09:05:00")
        e3 = self.manager.mark_student_attendance("STU003", 88.0, date_str=date_str, time_str="09:10:00")

        self.assertEqual(e1.status, AttendanceStatus.MARKED)
        self.assertEqual(e2.status, AttendanceStatus.MARKED)
        self.assertEqual(e3.status, AttendanceStatus.MARKED)

        records = self.db.get_attendance_by_date(date_str)
        self.assertEqual(len(records), 3)

        student_ids = {r["student_id"] for r in records}
        self.assertEqual(student_ids, {"STU001", "STU002", "STU003"})

    def test_timestamp_recording(self):
        """Test that custom and automatic timestamps are accurately recorded."""
        # Custom timestamp
        self.manager.mark_student_attendance(
            student_id="STU001",
            date_str="2026-11-17",
            time_str="10:45:12"
        )
        record = self.db.get_student_attendance("STU001")[0]
        self.assertEqual(record["date"], "2026-11-17")
        self.assertEqual(record["time"], "10:45:12")

        # Default timestamp (today's date)
        self.manager.mark_student_attendance(student_id="STU002")
        record2 = self.db.get_student_attendance("STU002")[0]
        self.assertEqual(record2["date"], datetime.now().strftime("%Y-%m-%d"))

    def test_export_attendance_to_csv(self):
        """Test exporting attendance records to CSV."""
        self.manager.mark_student_attendance("STU001", 90.0, date_str="2026-11-17", time_str="08:30:00")
        self.manager.mark_student_attendance("STU002", 85.0, date_str="2026-11-17", time_str="08:35:00")

        csv_file = os.path.join(self.temp_dir, "test_export.csv")
        exported_path = self.manager.export_attendance_to_csv(
            date_str="2026-11-17",
            output_path=csv_file
        )

        self.assertTrue(exported_path.exists())
        df = pd.read_csv(exported_path)
        self.assertEqual(len(df), 2)
        self.assertIn("student_id", df.columns)
        self.assertIn("name", df.columns)
        self.assertIn("status", df.columns)
        self.assertEqual(set(df["student_id"]), {"STU001", "STU002"})


if __name__ == "__main__":
    unittest.main()
