"""
Unit tests for database module.
Covers schema initialization, student management, attendance recording,
duplicate prevention, dashboard statistics, and foreign key constraints.
"""

import os
import tempfile
import unittest
import sqlite3
from datetime import datetime, timedelta

from database import (
    DatabaseManager,
    DuplicateStudentError,
    StudentNotFoundError,
    DuplicateAttendanceError,
    ValidationError
)


class TestDatabaseManager(unittest.TestCase):

    def setUp(self):
        """Set up an isolated temporary database for each test."""
        self.temp_file = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.temp_file.close()
        self.db = DatabaseManager(db_path=self.temp_file.name)

    def tearDown(self):
        """Clean up the temporary database file after test completion."""
        try:
            if os.path.exists(self.temp_file.name):
                os.remove(self.temp_file.name)
        except PermissionError:
            pass

    def test_01_database_initialization(self):
        """Test 1: Verify tables students and attendance exist and foreign keys are enabled."""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()

            # Check foreign key pragma
            cursor.execute("PRAGMA foreign_keys;")
            fk_status = cursor.fetchone()[0]
            self.assertEqual(fk_status, 1, "Foreign keys must be enabled.")

            # Check students table
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='students';")
            self.assertIsNotNone(cursor.fetchone(), "Table 'students' must exist.")

            # Check attendance table
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='attendance';")
            self.assertIsNotNone(cursor.fetchone(), "Table 'attendance' must exist.")

    def test_02_student_registration(self):
        """Test 2: Verify student registration succeeds and returns a positive integer ID."""
        internal_id = self.db.add_student(
            student_id="STU001",
            name="Alice Smith",
            course="Computer Science",
            sample_count=30
        )
        self.assertIsInstance(internal_id, int)
        self.assertGreater(internal_id, 0)

        student = self.db.get_student("STU001")
        self.assertIsNotNone(student)
        self.assertEqual(student["id"], internal_id)
        self.assertEqual(student["student_id"], "STU001")
        self.assertEqual(student["name"], "Alice Smith")
        self.assertEqual(student["course"], "Computer Science")
        self.assertEqual(student["sample_count"], 30)

    def test_03_duplicate_student_registration(self):
        """Test 3: Verify registering an existing student_id raises DuplicateStudentError."""
        self.db.add_student("STU002", "Bob Jones", "Electrical Eng", 25)

        with self.assertRaises(DuplicateStudentError):
            self.db.add_student("STU002", "Bob Duplicate", "Mechanical Eng", 10)

    def test_04_student_retrieval(self):
        """Test 4: Verify student retrieval by numeric ID and student_id string."""
        internal_id = self.db.add_student("STU003", "Charlie Brown", "Physics")

        # Retrieve by integer primary key
        by_int_id = self.db.get_student(internal_id)
        self.assertIsNotNone(by_int_id)
        self.assertEqual(by_int_id["name"], "Charlie Brown")

        # Retrieve by student_id string
        by_str_id = self.db.get_student("STU003")
        self.assertIsNotNone(by_str_id)
        self.assertEqual(by_str_id["id"], internal_id)

        # Retrieve non-existent student
        self.assertIsNone(self.db.get_student("NON_EXISTENT"))

    def test_05_student_updating(self):
        """Test 5: Verify updating student name, course, and sample count."""
        self.db.add_student("STU004", "David Clark", "Chemistry", 10)

        # Update details
        updated = self.db.update_student(
            student_id="STU004",
            name="David A. Clark",
            course="Biochemistry",
            sample_count=40
        )
        self.assertTrue(updated)

        student = self.db.get_student("STU004")
        self.assertEqual(student["name"], "David A. Clark")
        self.assertEqual(student["course"], "Biochemistry")
        self.assertEqual(student["sample_count"], 40)

        # Updating non-existent student returns False
        self.assertFalse(self.db.update_student("STU999", name="Nobody"))

        # Empty name raises ValidationError
        with self.assertRaises(ValidationError):
            self.db.update_student("STU004", name="   ")

    def test_06_attendance_marking(self):
        """Test 6: Verify marking attendance for a registered student."""
        self.db.add_student("STU005", "Emma Watson", "Mathematics")

        record = self.db.mark_attendance(
            student_id="STU005",
            confidence=88.5,
            date_str="2026-11-17",
            time_str="09:15:00"
        )
        self.assertIsNotNone(record)
        self.assertEqual(record["student_id"], "STU005")
        self.assertEqual(record["name"], "Emma Watson")
        self.assertEqual(record["course"], "Mathematics")
        self.assertEqual(record["date"], "2026-11-17")
        self.assertEqual(record["time"], "09:15:00")
        self.assertAlmostEqual(record["confidence"], 88.5)

        # Marking for unregistered student raises StudentNotFoundError
        with self.assertRaises(StudentNotFoundError):
            self.db.mark_attendance("UNREGISTERED_USER")

    def test_07_duplicate_attendance_prevention(self):
        """Test 7: Verify duplicate attendance on same date is prevented via UNIQUE constraint."""
        self.db.add_student("STU006", "Frank Miller", "History")

        # First attendance marking succeeds
        self.db.mark_attendance("STU006", confidence=90.0, date_str="2026-11-17")

        # Second attendance attempt on same day must raise DuplicateAttendanceError
        with self.assertRaises(DuplicateAttendanceError):
            self.db.mark_attendance("STU006", confidence=92.0, date_str="2026-11-17")

        # Attendance on another day succeeds
        record_next_day = self.db.mark_attendance(
            "STU006",
            confidence=85.0,
            date_str="2026-11-18"
        )
        self.assertEqual(record_next_day["date"], "2026-11-18")

    def test_08_attendance_retrieval(self):
        """Test 8: Verify attendance retrieval today, by date, and by student."""
        self.db.add_student("STU007", "Grace Hopper", "Computer Science")
        self.db.add_student("STU008", "Alan Turing", "Mathematics")

        today_str = datetime.now().strftime("%Y-%m-%d")
        yesterday_str = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

        # Mark attendance today for Grace
        self.db.mark_attendance("STU007", confidence=95.0, date_str=today_str, time_str="10:00:00")
        # Mark attendance yesterday for Grace and Alan
        self.db.mark_attendance("STU007", confidence=92.0, date_str=yesterday_str, time_str="09:00:00")
        self.db.mark_attendance("STU008", confidence=89.0, date_str=yesterday_str, time_str="09:05:00")

        # get_today_attendance
        today_list = self.db.get_today_attendance()
        self.assertEqual(len(today_list), 1)
        self.assertEqual(today_list[0]["student_id"], "STU007")

        # get_attendance_by_date
        yesterday_list = self.db.get_attendance_by_date(yesterday_str)
        self.assertEqual(len(yesterday_list), 2)

        # get_student_attendance
        grace_records = self.db.get_student_attendance("STU007")
        self.assertEqual(len(grace_records), 2)
        alan_records = self.db.get_student_attendance("STU008")
        self.assertEqual(len(alan_records), 1)

    def test_09_dashboard_statistics(self):
        """Test 9: Verify total_students, present_today, and attendance_percentage."""
        # Initial empty stats
        stats = self.db.get_dashboard_stats()
        self.assertEqual(stats["total_students"], 0)
        self.assertEqual(stats["present_today"], 0)
        self.assertEqual(stats["attendance_percentage"], 0.0)

        # Register 4 students
        self.db.add_student("STU101", "Student One", "CS")
        self.db.add_student("STU102", "Student Two", "CS")
        self.db.add_student("STU103", "Student Three", "EE")
        self.db.add_student("STU104", "Student Four", "ME")

        self.assertEqual(self.db.get_total_students(), 4)

        # Mark 3 students present today
        today = datetime.now().strftime("%Y-%m-%d")
        self.db.mark_attendance("STU101", date_str=today)
        self.db.mark_attendance("STU102", date_str=today)
        self.db.mark_attendance("STU103", date_str=today)

        self.assertEqual(self.db.get_present_today(today), 3)

        # 3 out of 4 = 75.0%
        self.assertAlmostEqual(self.db.get_attendance_percentage(today), 75.0)

        stats = self.db.get_dashboard_stats(today)
        self.assertEqual(stats["total_students"], 4)
        self.assertEqual(stats["present_today"], 3)
        self.assertEqual(stats["absent_today"], 1)
        self.assertAlmostEqual(stats["attendance_percentage"], 75.0)

    def test_10_foreign_key_constraints(self):
        """Test 10: Verify foreign key enforcement and cascade delete behavior."""
        self.db.add_student("STU200", "Nikola Tesla", "Physics")
        self.db.mark_attendance("STU200", confidence=96.0, date_str="2026-11-17")

        # Verify attendance exists
        attendance = self.db.get_student_attendance("STU200")
        self.assertEqual(len(attendance), 1)

        # Directly inserting attendance with non-existent student_id should violate FK constraint
        with self.assertRaises(sqlite3.IntegrityError):
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO attendance (student_id, name, course, date, time)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    ("NON_EXISTENT_ID", "Ghost", "None", "2026-11-17", "12:00:00")
                )

        # Delete student - cascade delete should remove related attendance records
        deleted = self.db.delete_student("STU200")
        self.assertTrue(deleted)

        # Student should be gone
        self.assertIsNone(self.db.get_student("STU200"))

        # Associated attendance records should be automatically deleted (no orphaned records)
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM attendance WHERE student_id = 'STU200'")
            count = cursor.fetchone()[0]
            self.assertEqual(count, 0, "Related attendance must be removed by cascade deletion.")


if __name__ == "__main__":
    unittest.main()
