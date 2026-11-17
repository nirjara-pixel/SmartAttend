# SmartAttend – Face Recognition Attendance Management System

**SmartAttend** is an AI-powered desktop attendance management application built using Python, OpenCV, Haar Cascade face detection, Local Binary Patterns Histograms (LBPH) face recognition, SQLite, and Tkinter.

It automates student attendance by identifying registered faces through a live webcam stream and recording attendance in a centralized database.

## 🌟 Key Features

- **Live Facial Recognition:** Identifies registered students through a real-time webcam stream.
- **Automated Attendance:** Automatically records attendance in SQLite.
- **Duplicate Attendance Prevention:** Database-level UNIQUE constraints prevent repeated attendance entries for the same student on the same day.
- **Recognition Cooldown:** In-memory debounce mechanism prevents redundant database queries and repeated writes.
- **Unknown Face Rejection:** Detects and labels unrecognized faces without recording attendance.
- **Face Dataset Acquisition:** Captures normalized 200×200 facial images using webcam-based registration.
- **LBPH Model Training:** Trains and saves facial recognition models using OpenCV.
- **Interactive Dashboard:** Displays live attendance statistics and student information.
- **Attendance Management:** Search and filter attendance records.
- **CSV Export:** Generate attendance reports for further analysis.
- **Multithreaded Camera Processing:** Keeps webcam operations from blocking the GUI.

## 🏗️ Project Architecture

```text
SmartAttend/
│
├── cascades/
│   └── haarcascade_frontalface_default.xml
│
├── data/
│   └── smartattend.db
│
├── dataset/
│   └── <student_id>/
│       └── sample_001.jpg
│
├── exports/
│
├── trainer/
│   ├── trainer.yml
│   └── metadata.json
│
├── gui/
│   ├── __init__.py
│   ├── theme.py
│   ├── dashboard.py
│   ├── register.py
│   ├── recognition_view.py
│   ├── attendance_view.py
│   └── train_view.py
│
├── tests/
│   ├── test_face_detector.py
│   ├── test_database.py
│   ├── test_student_registration.py
│   ├── test_train_model.py
│   ├── test_face_recognition.py
│   ├── test_attendance_manager.py
│   └── test_gui.py
│
├── config.py
├── database.py
├── face_detector.py
├── face_recognition.py
├── student_registration.py
├── train_model.py
├── attendance_manager.py
├── main.py
├── requirements.txt
├── .gitignore
└── README.md
```

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| OpenCV | Computer vision and image processing |
| Haar Cascade | Face detection |
| LBPH | Facial recognition |
| NumPy | Image matrix processing |
| Pandas | Data handling and reporting |
| SQLite3 | Relational database management |
| Tkinter & ttk | Desktop application interface |
| Pillow | Image rendering |
| Unittest | Automated testing |

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or higher
- Webcam
- Windows/Linux desktop environment
- Git

### 1. Clone the Repository

```bash
git clone https://github.com/nirjara-pixel/SmartAttend.git
```

### 2. Navigate to the Project Directory

```bash
cd SmartAttend
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

### 4. Activate the Environment

Windows PowerShell:

```powershell
venv\Scripts\activate
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

### 6. Launch SmartAttend

```bash
python main.py
```

Optional commands:

```bash
python main.py --version
```

Train or retrain the LBPH model:

```bash
python main.py --train
```

## 📋 Application Workflow

### 1. Launch Dashboard

Run `python main.py` to open the SmartAttend desktop dashboard.

### 2. Register Students

- Navigate to Student Registration.
- Enter Student ID, Name, and Course.
- Start webcam-based face capture.
- Capture multiple facial samples.
- Store processed images in the student's dataset folder.

### 3. Train the Recognition Model

- Navigate to Model Calibration.
- Select Train / Retrain LBPH Model.
- Generate the trained model.
- Save model weights in `trainer/trainer.yml`.

### 4. Start Live Face Recognition

- Navigate to Live Recognition.
- Start the webcam.
- Identify registered students.
- Display recognized student information.
- Reject unknown faces.

### 5. Manage Attendance

- Automatically record recognized students.
- Prevent duplicate daily attendance.
- View attendance records.
- Search and filter records by date or student.

### 6. Export Reports

- Navigate to Attendance Records.
- Apply the required filters.
- Export attendance reports in CSV format.

## 🧪 Automated Testing

SmartAttend includes unit and integration tests covering its core functionality.

Run the complete test suite:

```powershell
python -m unittest discover tests -v
```

**Development testing result: 56 tests passed.**

The test suite covers:

- Haar Cascade face detection.
- Face preprocessing and ROI extraction.
- SQLite database operations.
- Foreign key constraints.
- Duplicate attendance prevention.
- Student registration and dataset collection.
- LBPH model training and persistence.
- Face recognition and unknown-face handling.
- Attendance debounce logic.
- GUI initialization and navigation.

## 🔮 Future Enhancements

- Excel attendance export.
- Advanced attendance analytics and visualizations.
- Improved recognition under varying lighting conditions.
- Admin authentication and role-based access.
- Liveness detection to reduce photo-based spoofing.
- Enhanced student and attendance management.

## 🔐 Privacy and Security

- Student facial datasets contain sensitive personal information.
- Database files and facial datasets should remain private.
- Do not upload real student images or personal records to a public repository.
- LBPH recognition alone does not provide liveness detection.

## 👩‍💻 Author

**Nirjara Jain**

B.Tech – Computer Science and Engineering

GitHub: [@nirjara-pixel](https://github.com/nirjara-pixel)

---

⭐ If you find this project interesting, consider giving the repository a star!