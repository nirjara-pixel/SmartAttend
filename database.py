"""
SmartAttend - Database Module
SQLite Database Manager for Student Registration, Attendance Tracking, and Dashboard Analytics.
"""

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from contextlib import contextmanager

import config


class DatabaseError(Exception):
    """Base exception for database-related operations."""
    pass


class ValidationError(DatabaseError):
    """Raised when input validation fails."""
    pass


class DuplicateStudentError(DatabaseError):
    """Raised when attempting to add a student with an existing Student ID."""
    pass


class StudentNotFoundError(DatabaseError):
    """Raised when a student cannot be found in the database."""
    pass


class DuplicateAttendanceError(DatabaseError):
    """Raised when attendance has already been recorded for the student on the given date."""
    pass


class DatabaseManager:
    """
    Manages SQLite database connections, schema migrations,
    student records, and attendance logs.
    """

    def __init__(self, db_path: Optional[Union[str, Path]] = None):
        """
        Initialize the database manager.

        Args:
            db_path: Path to the SQLite database file. Defaults to config.DATABASE_PATH.
        """
        self.db_path = str(db_path or config.DATABASE_PATH)
        self.init_db()

    @contextmanager
    def get_connection(self):
        """
        Context manager for database connections.
        Ensures foreign key enforcement and row factory are enabled,
        commits transactions, and guarantees connection closure.
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        # Enable SQLite foreign key constraint enforcement
        conn.execute("PRAGMA foreign_keys = ON;")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def init_db(self) -> None:
        """
        Initialize database tables and indexes if they do not exist.
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Students Table
            # id is used as the numeric label for LBPH recognizer
            # student_id is the unique institutional identifier (e.g., STU-001)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS students (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    student_id TEXT UNIQUE NOT NULL,
                    name TEXT NOT NULL,
                    course TEXT NOT NULL,
                    sample_count INTEGER DEFAULT 0,
                    created_at TEXT DEFAULT (datetime('now', 'localtime'))
                );
            """)

            # Attendance Table
            # UNIQUE(student_id, date) prevents duplicate attendance on the same day
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS attendance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    student_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    course TEXT NOT NULL,
                    date TEXT NOT NULL,
                    time TEXT NOT NULL,
                    confidence REAL DEFAULT 0.0,
                    status TEXT DEFAULT 'Present',
                    created_at TEXT DEFAULT (datetime('now', 'localtime')),
                    FOREIGN KEY (student_id) REFERENCES students(student_id) 
                        ON DELETE CASCADE 
                        ON UPDATE CASCADE,
                    UNIQUE (student_id, date)
                );
            """)

            # Indexes for query performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_students_student_id ON students(student_id);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_attendance_date ON attendance(date);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_attendance_student_id ON attendance(student_id);")

    # ==========================================
    # TASK 2: STUDENT MANAGEMENT
    # ==========================================

    def add_student(
        self,
        student_id: str,
        name: str,
        course: str,
        sample_count: int = 0
    ) -> int:
        """
        Register a new student.

        Args:
            student_id: Unique student institutional ID (e.g. 'STU101').
            name: Full name of the student.
            course: Department or course (e.g. 'Computer Science').
            sample_count: Initial count of captured face samples.

        Returns:
            The newly created student's internal numeric ID (used by LBPH).

        Raises:
            ValidationError: If inputs are invalid or empty.
            DuplicateStudentError: If student_id already exists.
        """
        # Input validation
        if not student_id or not str(student_id).strip():
            raise ValidationError("Student ID cannot be empty.")
        if not name or not str(name).strip():
            raise ValidationError("Student Name cannot be empty.")
        if not course or not str(course).strip():
            raise ValidationError("Course cannot be empty.")
        if not isinstance(sample_count, int) or sample_count < 0:
            raise ValidationError("Sample count must be a non-negative integer.")

        clean_id = str(student_id).strip()
        clean_name = str(name).strip()
        clean_course = str(course).strip()

        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO students (student_id, name, course, sample_count)
                    VALUES (?, ?, ?, ?)
                    """,
                    (clean_id, clean_name, clean_course, sample_count)
                )
                return cursor.lastrowid
        except sqlite3.IntegrityError as e:
            if "UNIQUE constraint failed" in str(e):
                raise DuplicateStudentError(f"Student with ID '{clean_id}' already exists.") from e
            raise DatabaseError(f"Database error while adding student: {e}") from e

    def get_student(self, identifier: Union[int, str]) -> Optional[Dict[str, Any]]:
        """
        Retrieve a student by internal numeric ID or unique Student ID string.

        Args:
            identifier: Either the internal integer primary key 'id' or 'student_id' string.

        Returns:
            Dictionary representing student record, or None if not found.
        """
        if identifier is None:
            return None

        with self.get_connection() as conn:
            cursor = conn.cursor()
            if isinstance(identifier, int) or (isinstance(identifier, str) and identifier.isdigit()):
                cursor.execute(
                    "SELECT * FROM students WHERE id = ? OR student_id = ? LIMIT 1",
                    (int(identifier), str(identifier))
                )
            else:
                cursor.execute(
                    "SELECT * FROM students WHERE student_id = ? LIMIT 1",
                    (str(identifier).strip(),)
                )

            row = cursor.fetchone()
            return dict(row) if row else None

    def get_all_students(self) -> List[Dict[str, Any]]:
        """
        Retrieve all registered students ordered by student ID.

        Returns:
            List of student dictionaries.
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM students ORDER BY id ASC")
            return [dict(row) for row in cursor.fetchall()]

    def update_student(
        self,
        student_id: str,
        name: Optional[str] = None,
        course: Optional[str] = None,
        sample_count: Optional[int] = None
    ) -> bool:
        """
        Update an existing student's details.

        Args:
            student_id: Unique student ID to update.
            name: New full name (optional).
            course: New course (optional).
            sample_count: New sample count (optional).

        Returns:
            True if student was updated, False if student was not found.
        """
        clean_id = str(student_id).strip()
        student = self.get_student(clean_id)
        if not student:
            return False

        updates = []
        params = []

        if name is not None:
            if not str(name).strip():
                raise ValidationError("Student Name cannot be empty.")
            updates.append("name = ?")
            params.append(str(name).strip())

        if course is not None:
            if not str(course).strip():
                raise ValidationError("Course cannot be empty.")
            updates.append("course = ?")
            params.append(str(course).strip())

        if sample_count is not None:
            if not isinstance(sample_count, int) or sample_count < 0:
                raise ValidationError("Sample count must be a non-negative integer.")
            updates.append("sample_count = ?")
            params.append(sample_count)

        if not updates:
            return True

        params.append(clean_id)
        query = f"UPDATE students SET {', '.join(updates)} WHERE student_id = ?"

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            return cursor.rowcount > 0

    def delete_student(self, identifier: Union[int, str]) -> bool:
        """
        Safely delete a student and their associated attendance records (via foreign key cascade).

        Args:
            identifier: Either internal numeric id or student_id string.

        Returns:
            True if student was deleted, False if student was not found.
        """
        student = self.get_student(identifier)
        if not student:
            return False

        with self.get_connection() as conn:
            cursor = conn.cursor()
            # Foreign key cascade ensures attendance records are removed cleanly
            cursor.execute("DELETE FROM students WHERE id = ?", (student["id"],))
            return cursor.rowcount > 0

    # ==========================================
    # TASK 3: ATTENDANCE MANAGEMENT
    # ==========================================

    def mark_attendance(
        self,
        student_id: str,
        confidence: float = 0.0,
        date_str: Optional[str] = None,
        time_str: Optional[str] = None,
        status: str = "Present"
    ) -> Dict[str, Any]:
        """
        Record attendance for a registered student.
        Prevents duplicate attendance for the same student on the same date.

        Args:
            student_id: Registered student ID.
            confidence: Recognition match confidence score (e.g. 85.5%).
            date_str: Optional date string 'YYYY-MM-DD'. Defaults to today.
            time_str: Optional time string 'HH:MM:SS'. Defaults to current local time.
            status: Attendance status string (default 'Present').

        Returns:
            Dictionary with newly created attendance details.

        Raises:
            StudentNotFoundError: If student is not registered.
            DuplicateAttendanceError: If attendance is already marked for today.
            DatabaseError: On underlying SQLite issues.
        """
        clean_id = str(student_id).strip()
        student = self.get_student(clean_id)
        if not student:
            raise StudentNotFoundError(f"Student with ID '{clean_id}' is not registered.")

        now = datetime.now()
        target_date = date_str or now.strftime("%Y-%m-%d")
        target_time = time_str or now.strftime("%H:%M:%S")

        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO attendance (student_id, name, course, date, time, confidence, status)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        student["student_id"],
                        student["name"],
                        student["course"],
                        target_date,
                        target_time,
                        float(confidence),
                        status
                    )
                )
                rec_id = cursor.lastrowid
                return {
                    "id": rec_id,
                    "student_id": student["student_id"],
                    "name": student["name"],
                    "course": student["course"],
                    "date": target_date,
                    "time": target_time,
                    "confidence": float(confidence),
                    "status": status
                }
        except sqlite3.IntegrityError as e:
            if "UNIQUE constraint failed" in str(e):
                raise DuplicateAttendanceError(
                    f"Attendance already marked for student '{student['name']}' ({clean_id}) on {target_date}."
                ) from e
            raise DatabaseError(f"Integrity error recording attendance: {e}") from e

    def get_today_attendance(self) -> List[Dict[str, Any]]:
        """
        Retrieve all attendance records for today.

        Returns:
            List of attendance record dictionaries.
        """
        today = datetime.now().strftime("%Y-%m-%d")
        return self.get_attendance_by_date(today)

    def get_attendance_by_date(self, date_str: str) -> List[Dict[str, Any]]:
        """
        Retrieve attendance records for a specific date (YYYY-MM-DD).

        Args:
            date_str: Target date in format 'YYYY-MM-DD'.

        Returns:
            List of attendance record dictionaries ordered by time descending.
        """
        clean_date = str(date_str).strip()
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, student_id, name, course, date, time, confidence, status, created_at
                FROM attendance
                WHERE date = ?
                ORDER BY time DESC, id DESC
                """,
                (clean_date,)
            )
            return [dict(row) for row in cursor.fetchall()]

    def get_student_attendance(self, student_id: str) -> List[Dict[str, Any]]:
        """
        Retrieve complete attendance history for a student.

        Args:
            student_id: Student ID string.

        Returns:
            List of attendance record dictionaries ordered by date descending.
        """
        clean_id = str(student_id).strip()
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, student_id, name, course, date, time, confidence, status, created_at
                FROM attendance
                WHERE student_id = ?
                ORDER BY date DESC, time DESC
                """,
                (clean_id,)
            )
            return [dict(row) for row in cursor.fetchall()]

    # ==========================================
    # TASK 4: DASHBOARD STATISTICS
    # ==========================================

    def get_total_students(self) -> int:
        """
        Get total number of registered students.
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) AS total FROM students")
            row = cursor.fetchone()
            return int(row["total"]) if row else 0

    def get_present_today(self, date_str: Optional[str] = None) -> int:
        """
        Get count of distinct registered students marked present today.
        """
        target_date = date_str or datetime.now().strftime("%Y-%m-%d")
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT COUNT(DISTINCT student_id) AS present FROM attendance WHERE date = ?",
                (target_date,)
            )
            row = cursor.fetchone()
            return int(row["present"]) if row else 0

    def get_attendance_percentage(self, date_str: Optional[str] = None) -> float:
        """
        Calculate attendance percentage for today (or specified date).

        Returns:
            Attendance percentage float between 0.0 and 100.0 rounded to 1 decimal place.
        """
        total = self.get_total_students()
        if total == 0:
            return 0.0

        present = self.get_present_today(date_str)
        percentage = (present / total) * 100.0
        return round(min(100.0, max(0.0, percentage)), 1)

    def get_dashboard_stats(self, date_str: Optional[str] = None) -> Dict[str, Any]:
        """
        Compile complete statistics for dashboard KPI cards.

        Returns:
            Dictionary containing total_students, present_today, absent_today,
            attendance_percentage, and today_date.
        """
        target_date = date_str or datetime.now().strftime("%Y-%m-%d")
        total = self.get_total_students()
        present = self.get_present_today(target_date)
        absent = max(0, total - present)
        percentage = self.get_attendance_percentage(target_date)

        return {
            "total_students": total,
            "present_today": present,
            "absent_today": absent,
            "attendance_percentage": percentage,
            "today_date": target_date
        }


# Default singleton instance for app-wide use
db = DatabaseManager()

# Module-level convenience functions referencing the default instance
def add_student(student_id: str, name: str, course: str, sample_count: int = 0) -> int:
    return db.add_student(student_id, name, course, sample_count)

def get_student(identifier: Union[int, str]) -> Optional[Dict[str, Any]]:
    return db.get_student(identifier)

def get_all_students() -> List[Dict[str, Any]]:
    return db.get_all_students()

def update_student(
    student_id: str,
    name: Optional[str] = None,
    course: Optional[str] = None,
    sample_count: Optional[int] = None
) -> bool:
    return db.update_student(student_id, name, course, sample_count)

def delete_student(identifier: Union[int, str]) -> bool:
    return db.delete_student(identifier)

def mark_attendance(
    student_id: str,
    confidence: float = 0.0,
    date_str: Optional[str] = None,
    time_str: Optional[str] = None
) -> Dict[str, Any]:
    return db.mark_attendance(student_id, confidence, date_str, time_str)

def get_today_attendance() -> List[Dict[str, Any]]:
    return db.get_today_attendance()

def get_attendance_by_date(date_str: str) -> List[Dict[str, Any]]:
    return db.get_attendance_by_date(date_str)

def get_student_attendance(student_id: str) -> List[Dict[str, Any]]:
    return db.get_student_attendance(student_id)

def get_total_students() -> int:
    return db.get_total_students()

def get_present_today(date_str: Optional[str] = None) -> int:
    return db.get_present_today(date_str)

def get_attendance_percentage(date_str: Optional[str] = None) -> float:
    return db.get_attendance_percentage(date_str)

def get_dashboard_stats(date_str: Optional[str] = None) -> Dict[str, Any]:
    return db.get_dashboard_stats(date_str)
