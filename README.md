# SmartAttend – Face Recognition Attendance Management System

**SmartAttend** is an enterprise-grade desktop attendance management software powered by Python, OpenCV, Haar Cascade face detection, Local Binary Patterns Histograms (LBPH) face recognition, SQLite, and a modern Tkinter desktop GUI.

---

## 🌟 Key Features

- **Live Facial Recognition Attendance**: Automatically identifies registered students from a live webcam stream and records daily attendance in SQLite in real time.
- **Duplicate Attendance Prevention**: Guaranteed by SQLite database-level constraints (`UNIQUE(student_id, date)`) preventing repeated attendance on the same day.
- **In-Memory Debounce Cooldown**: Prevents redundant database queries and repeated writes while a recognized student remains in front of the camera.
- **Unknown Face Rejection**: Highlights unrecognized faces with red HUD bounding boxes and rejects unauthorized access without polluting the attendance log.
- **Dataset Acquisition**: Automated webcam-assisted capture of 200×200 normalized, histogram-equalized facial crops with head pose guidance (straight, tilt, expressions).
- **LBPH Model Training**: Trains and serializes `cv2.face.LBPHFaceRecognizer` weights to `trainer/trainer.yml` with companion JSON metadata.
- **Interactive Dashboard**:
  - Live KPI statistics cards: **Total Students**, **Present Today**, **Absent Today**, and **Attendance Rate (%)**.
  - Modern dark-blue sidebar navigation and light card workspace.
  - Non-blocking multithreaded camera streams and progress bars.
  - Real-time searchable and date-filtered attendance table with CSV report export.

---

## 🏗️ Project Architecture

```
SmartAttend/
├── cascades/
│   └── haarcascade_frontalface_default.xml # Haar Cascade Classifier XML
├── data/
│   └── smartattend.db                      # SQLite relational database
├── dataset/                                # Registered student facial crops
│   └── <student_id>/
│       └── sample_001.jpg ...
├── exports/                                # Generated CSV attendance reports
├── gui/
│   ├── __init__.py
│   ├── theme.py                            # Professional color palettes and fonts
│   ├── dashboard.py                        # Main Tkinter desktop shell & Overview
│   ├── register.py                         # Student Registration & Camera Capture
│   ├── recognition_view.py                 # Live Face Recognition & Attendance
│   ├── attendance_view.py                  # Searchable Table & CSV Export
│   └── train_view.py                       # Model Training & Calibration
├── tests/                                  # Automated Test Suite (56 tests)
│   ├── __init__.py
│   ├── test_face_detector.py
│   ├── test_database.py
│   ├── test_student_registration.py
│   ├── test_train_model.py
│   ├── test_face_recognition.py
│   ├── test_attendance_manager.py
│   └── test_gui.py
├── trainer/
│   ├── trainer.yml                         # Trained LBPH model weights
│   └── metadata.json                       # Hyperparameters & student roster
├── config.py                               # Global configurations and parameters
├── database.py                             # SQLite Database Manager & CRUD
├── face_detector.py                        # Haar Cascade detector & ROI extractor
├── face_recognition.py                     # LBPH inference & confidence scoring
├── student_registration.py                 # Dataset collection & DB synchronization
├── train_model.py                          # LBPH model trainer & serializer
├── attendance_manager.py                   # Attendance logging & debounce logic
├── main.py                                 # Application entry point
├── requirements.txt                        # Python dependencies
└── README.md                               # Project documentation
```

---

## 🛠️ Technology Stack

- **Python 3.10+ / 3.14**
- **OpenCV (`opencv-contrib-python`)**: Haar Cascade face localization and LBPH face recognition.
- **Pillow (PIL)**: High-performance frame conversion for Tkinter canvas rendering.
- **SQLite3**: ACID-compliant persistent storage with foreign keys and unique constraints.
- **NumPy**: Multidimensional array processing for facial matrices.
- **Pandas**: Tabular attendance aggregation and CSV report generation.
- **Tkinter & ttk**: Native desktop GUI with custom styled cards and tables.

---

## 🚀 Getting Started

### 1. Prerequisites

Ensure Python is installed on your system. Verify via Windows PowerShell:
```powershell
python --version
```

### 2. Install Dependencies

Install the required packages using pip:
```powershell
pip install -r requirements.txt
```

### 3. Launch SmartAttend

Run the main application:
```powershell
python main.py
```

Optional CLI flags:
```powershell
# Display version
python main.py --version

# Train or retrain LBPH model directly from the command line
python main.py --train
```

---

## 📋 Step-by-Step Workflow

1. **Launch Application**: Execute `python main.py`. The dashboard opens with system statistics.
2. **Register Students**:
   - Navigate to **👤 Student Registration**.
   - Enter **Student ID** (e.g. `STU101`), **Name** (e.g. `Alice Smith`), and **Course** (e.g. `Computer Science`).
   - Click **▶ Start Face Capture & Register**.
   - Face the camera while the progress bar collects and normalizes face crops into `dataset/STU101/`.
   - Register at least **2 distinct students**.
3. **Train LBPH Model**:
   - Accept the post-registration prompt or navigate to **⚙️ Model Calibration**.
   - Click **⚡ Train / Retrain LBPH Model**.
   - The model trains and writes to `trainer/trainer.yml`.
4. **Live Attendance Recognition**:
   - Navigate to **🎥 Live Recognition**.
   - Click **▶ Start Recognition**.
   - Registered students will be framed with green bounding boxes, showing name, ID, and match confidence. Attendance is instantly marked in SQLite.
   - Unrecognized faces appear with red "Unknown" badges and are ignored from attendance logs.
5. **View & Export Attendance**:
   - Navigate to **📋 Attendance Records**.
   - Filter by date or search by student name/ID.
   - Click **📥 Export to CSV** to export the official report.

---

## 🧪 Automated Testing

SmartAttend includes a comprehensive test suite (56 tests) mocking webcam and hardware dependencies so the entire suite runs in any environment without a physical camera:

```powershell
# Run the complete test suite with verbose output
python -m unittest discover tests -v
```

All 56 unit and integration tests verify:
- Haar cascade face detection and boundary normalization
- SQLite schema, foreign keys, and duplicate attendance prevention
- Dataset collection, sequential file naming, and failure cleanup
- LBPH training, model persistence, and prediction scoring
- In-memory debounce cooldown and unknown face filtering
- Tkinter GUI shell, navigation, and statistics updates
