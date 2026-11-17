"""
SmartAttend - Automatic Attendance Management Module
Coordinates face recognition with SQLite attendance logging, prevents duplicate entries,
implements in-memory cooldown debouncing, and generates export reports.
"""

import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import pandas as pd

import config
from database import (
    DatabaseError,
    DatabaseManager,
    DuplicateAttendanceError,
    StudentNotFoundError,
    ValidationError,
    db as default_db
)
from face_recognition import FaceRecognizer, RecognitionResult


class AttendanceStatus(str, Enum):
    """Status outcomes for attendance operations."""
    MARKED = "MARKED"
    ALREADY_MARKED = "ALREADY_MARKED"
    COOLDOWN = "COOLDOWN"
    UNKNOWN_FACE = "UNKNOWN_FACE"
    INVALID_STUDENT = "INVALID_STUDENT"
    ERROR = "ERROR"


@dataclass
class AttendanceEvent:
    """Encapsulates the result and confirmation message of an attendance evaluation."""
    status: AttendanceStatus
    student_id: Optional[str] = None
    name: Optional[str] = None
    course: Optional[str] = None
    date: Optional[str] = None
    time: Optional[str] = None
    confidence: float = 0.0
    message: str = ""

    @property
    def is_success(self) -> bool:
        return self.status == AttendanceStatus.MARKED

    @property
    def is_already_marked(self) -> bool:
        return self.status == AttendanceStatus.ALREADY_MARKED

    @property
    def is_ignored(self) -> bool:
        return self.status in {AttendanceStatus.UNKNOWN_FACE, AttendanceStatus.COOLDOWN}


class AttendanceManager:
    """
    High-level Attendance Manager orchestrating recognition results and SQLite persistence.
    """

    def __init__(
        self,
        db_manager: Optional[DatabaseManager] = None,
        face_recognizer: Optional[FaceRecognizer] = None,
        cooldown_seconds: float = config.ATTENDANCE_COOLDOWN_SECONDS
    ):
        """
        Initialize the Attendance Manager.

        Args:
            db_manager: DatabaseManager instance for SQLite interactions. Defaults to default_db.
            face_recognizer: FaceRecognizer instance for real-time video feed recognition.
            cooldown_seconds: Minimum seconds between processing duplicate recognition events.
        """
        self.db = db_manager or default_db
        self.recognizer = face_recognizer
        self.cooldown_seconds = float(cooldown_seconds)

        # In-memory cooldown tracking: {student_id: last_processed_timestamp}
        # Prevents redundant database queries and writes during continuous camera streaming
        self._cooldown_cache: Dict[str, float] = {}

        # In-memory history of recent attendance events for UI display
        self.recent_events: List[AttendanceEvent] = []

    def clear_cooldown_cache(self) -> None:
        """Clear the in-memory cooldown cache."""
        self._cooldown_cache.clear()

    def mark_student_attendance(
        self,
        student_id: str,
        confidence: float = 0.0,
        date_str: Optional[str] = None,
        time_str: Optional[str] = None
    ) -> AttendanceEvent:
        """
        Record attendance for a student, handling duplicates, cooldowns, and database errors.

        Args:
            student_id: Institutional Student ID.
            confidence: Recognition match confidence score (0-100%).
            date_str: Optional date string 'YYYY-MM-DD'. Defaults to current local date.
            time_str: Optional time string 'HH:MM:SS'. Defaults to current local time.

        Returns:
            AttendanceEvent containing confirmation details and status.
        """
        if not student_id or not str(student_id).strip():
            return AttendanceEvent(
                status=AttendanceStatus.INVALID_STUDENT,
                message="Cannot mark attendance: Student ID is empty or invalid."
            )

        clean_id = str(student_id).strip()
        now = datetime.now()
        target_date = date_str or now.strftime("%Y-%m-%d")
        target_time = time_str or now.strftime("%H:%M:%S")

        # 1. Check in-memory cooldown to avoid redundant DB operations
        current_time = time.time()
        last_seen = self._cooldown_cache.get(clean_id)
        if last_seen is not None and (current_time - last_seen) < self.cooldown_seconds:
            remaining = int(self.cooldown_seconds - (current_time - last_seen))
            return AttendanceEvent(
                status=AttendanceStatus.COOLDOWN,
                student_id=clean_id,
                date=target_date,
                time=target_time,
                confidence=float(confidence),
                message=f"Cooldown active for Student ID '{clean_id}' ({remaining}s remaining)."
            )

        # 2. Verify student exists in SQLite
        try:
            student = self.db.get_student(clean_id)
            if not student:
                return AttendanceEvent(
                    status=AttendanceStatus.INVALID_STUDENT,
                    student_id=clean_id,
                    message=f"Student with ID '{clean_id}' is not registered in the database."
                )

            student_name = student["name"]
            student_course = student["course"]

            # 3. Mark attendance in SQLite
            record = self.db.mark_attendance(
                student_id=clean_id,
                confidence=confidence,
                date_str=target_date,
                time_str=target_time
            )

            # Update in-memory cooldown timestamp
            self._cooldown_cache[clean_id] = current_time

            event = AttendanceEvent(
                status=AttendanceStatus.MARKED,
                student_id=clean_id,
                name=student_name,
                course=student_course,
                date=target_date,
                time=target_time,
                confidence=round(float(confidence), 1),
                message=f"Success: Attendance recorded for {student_name} ({clean_id}) at {target_time}."
            )
            self._record_event(event)
            return event

        except DuplicateAttendanceError:
            # Update cooldown timestamp so we don't repeatedly re-query for an already marked student
            self._cooldown_cache[clean_id] = current_time

            event = AttendanceEvent(
                status=AttendanceStatus.ALREADY_MARKED,
                student_id=clean_id,
                name=student.get("name") if 'student' in locals() and student else clean_id,
                course=student.get("course") if 'student' in locals() and student else "",
                date=target_date,
                time=target_time,
                confidence=round(float(confidence), 1),
                message=f"Notice: Attendance for {clean_id} is already marked for today ({target_date})."
            )
            self._record_event(event)
            return event

        except (DatabaseError, Exception) as e:
            event = AttendanceEvent(
                status=AttendanceStatus.ERROR,
                student_id=clean_id,
                date=target_date,
                time=target_time,
                message=f"Database error while recording attendance: {e}"
            )
            self._record_event(event)
            return event

    def process_recognition_result(
        self,
        result: RecognitionResult,
        date_str: Optional[str] = None,
        time_str: Optional[str] = None
    ) -> AttendanceEvent:
        """
        Evaluate a single RecognitionResult and trigger attendance if recognized.

        Args:
            result: RecognitionResult from FaceRecognizer.
            date_str: Optional date string.
            time_str: Optional time string.

        Returns:
            AttendanceEvent representing the outcome.
        """
        # Unknown faces are strictly ignored from attendance logging
        if result.is_unknown or not result.is_recognized or not result.student_id:
            event = AttendanceEvent(
                status=AttendanceStatus.UNKNOWN_FACE,
                confidence=result.confidence,
                message="Unknown face detected. Attendance not marked."
            )
            return event

        # Recognized student -> proceed to mark attendance
        return self.mark_student_attendance(
            student_id=result.student_id,
            confidence=result.confidence,
            date_str=date_str,
            time_str=time_str
        )

    def process_frame(
        self,
        frame: Any,
        date_str: Optional[str] = None,
        time_str: Optional[str] = None,
        draw_annotations: bool = True
    ) -> Tuple[Any, List[AttendanceEvent]]:
        """
        Process a video frame through face recognition and attendance logging.

        Args:
            frame: Webcam BGR image.
            date_str: Optional target date.
            time_str: Optional target time.
            draw_annotations: Whether to draw HUD boxes on frame.

        Returns:
            Tuple of (annotated_frame, list of AttendanceEvents).
        """
        if self.recognizer is None:
            raise RuntimeError("FaceRecognizer is not configured for this AttendanceManager.")

        annotated_frame, recognition_results = self.recognizer.recognize_frame(
            frame,
            draw_annotations=draw_annotations
        )

        events: List[AttendanceEvent] = []
        for res in recognition_results:
            event = self.process_recognition_result(res, date_str=date_str, time_str=time_str)
            events.append(event)

        return annotated_frame, events

    def _record_event(self, event: AttendanceEvent) -> None:
        """Store recent event in memory for live dashboard feeds."""
        self.recent_events.insert(0, event)
        # Keep last 50 events in memory
        if len(self.recent_events) > 50:
            self.recent_events.pop()

    def get_today_attendance(self) -> List[Dict[str, Any]]:
        """Retrieve today's complete attendance records from SQLite."""
        return self.db.get_today_attendance()

    def get_attendance_by_date(self, date_str: str) -> List[Dict[str, Any]]:
        """Retrieve attendance records for a specific date from SQLite."""
        return self.db.get_attendance_by_date(date_str)

    def export_attendance_to_csv(
        self,
        date_str: Optional[str] = None,
        output_path: Optional[Union[str, Path]] = None
    ) -> Path:
        """
        Export attendance records to a CSV file.

        Args:
            date_str: Optional date filter. If None, exports today's records.
            output_path: Target CSV file path. If None, saves in config.EXPORTS_DIR.

        Returns:
            Path of the exported CSV file.

        Raises:
            IOError: On export write failure.
        """
        target_date = date_str or datetime.now().strftime("%Y-%m-%d")
        records = self.get_attendance_by_date(target_date)

        if output_path:
            out_file = Path(output_path)
            out_file.parent.mkdir(parents=True, exist_ok=True)
        else:
            config.EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
            out_file = config.EXPORTS_DIR / f"attendance_{target_date}.csv"

        if records:
            df = pd.DataFrame(records)
            # Reorder columns for readability
            cols_order = ["id", "student_id", "name", "course", "date", "time", "confidence", "status"]
            existing_cols = [c for c in cols_order if c in df.columns]
            df = df[existing_cols]
        else:
            df = pd.DataFrame(columns=["id", "student_id", "name", "course", "date", "time", "confidence", "status"])

        df.to_csv(str(out_file), index=False, encoding="utf-8")
        return out_file
