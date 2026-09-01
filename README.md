# 🚗 Vehicle Detection & Traffic Counting (YOLO)

This project detects vehicles in videos, tracks them, and counts how many vehicles go **IN** and **OUT** of a defined area. It uses **YOLO** for detection, **ByteTrack** for tracking, **FastAPI** for the backend, **SQLite** for storing results, and **Streamlit** for the web dashboard.

An end-to-end computer vision project covering dataset preparation, model training, vehicle detection, tracking, counting, backend development, database storage, and web visualization.

---

## 🌟 Key Features

### Two-Gate Counting System

Instead of using a single counting line, the video has two gates: an entry gate and an exit gate.

A vehicle is counted only when it crosses **both gates in the correct order**. This helps reduce double-counting and improves direction detection, especially for fast-moving vehicles.

Counts are divided into:

* **IN** — vehicles entering
* **OUT** — vehicles leaving

### Simple Dashboard

A dark-themed Streamlit dashboard for uploading and processing videos, viewing counters and charts, browsing previous runs, and downloading reports.

### Five Models to Choose From

The project supports five YOLO models:

* `yolov8n.pt`
* `yolov8s.pt`
* `yolo11n.pt`
* `yolo11s.pt`
* `uvh26_vehicle_model.pt` — custom model trained on the UVH-26 dataset

`yolov8n.pt` is used as the **default model** because it provided better and more reliable detection performance during testing.

The other models are available for comparison and experimentation.

### Vehicle Types Detected

**Pretrained YOLO models:**

* Car
* Motorcycle
* Bus
* Truck

**Custom UVH-26 model:**

* Car
* Truck
* Bus
* Van
* Motorcycle
* Rickshaw

### On-Video Overlay

The processed video displays:

* Detected vehicles
* Vehicle tracking IDs
* Movement trails
* Two counting gates
* IN/OUT counts

### Browser-Friendly Video Output

Processed videos are converted using **FFmpeg** into a format that can be played directly in the browser.

### Backend + Database

The project uses **FastAPI** as the backend and **SQLite** for storing uploaded videos, processing runs, results, reports, and analytics.

---

## 🧠 Model Training

A custom YOLO model was trained using the **UVH-26 traffic dataset**, accessed through the **Hugging Face API**.

Training was performed on **Kaggle** using GPU resources.

The custom model supports six vehicle classes:

* Car
* Truck
* Bus
* Van
* Motorcycle
* Rickshaw

During testing, the custom model did not perform as well as the pretrained YOLO models. Therefore, **`yolov8n.pt` was selected as the default model** because it provided better overall detection performance for this project.

The custom model is still included for comparison and further experimentation.

---

## 📁 Project Structure

```text
├── requirements.txt     # Required Python packages
├── model.py             # Loads YOLO models and vehicle categories
├── processing.py        # Tracking, counting, and video processing logic
├── database.py          # Database connection setup
├── db_models.py         # Database table definitions
├── schemas.py            # API data schemas
├── main.py              # FastAPI backend
├── app.py               # Main Streamlit dashboard
├── Count_video.py       # Standalone dashboard without backend
├── detect_image.py      # Detection on a single image
├── detect_video.py      # Detection and counting on a video
├── data/                # Input images/videos
├── uploads/             # Uploaded videos
├── outputs/             # Processed videos and reports
└── vehicle_analytics.db # SQLite database
```

---

## 🚀 Getting Started

You'll need:

* Python 3.9 or newer
* FFmpeg
* A GPU is recommended for faster processing but is not required

### Create a Virtual Environment

**Windows:**

```bash
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux:**

```bash
python3 -m venv venv
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

Make sure FFmpeg is installed and available in your system PATH.

You can check it with:

```bash
ffmpeg -version
```

Place your YOLO model files (`.pt`) in the project folder.

---

## 💻 How to Run It

### Option 1: Full App — Backend + Dashboard

#### Start the Backend

Open one terminal and run:

```bash
uvicorn main:app --reload
```

The API will normally run at:

```text
http://localhost:8000
```

Interactive API documentation:

```text
http://localhost:8000/docs
```

#### Start the Dashboard

Open a second terminal and run:

```bash
streamlit run app.py
```

The dashboard will normally run at:

```text
http://localhost:8501
```

### Dashboard Features

You can:

* Upload and process videos
* Select a YOLO model
* Configure the counting corridor
* View processed videos
* View previous processing runs
* View traffic statistics
* Download reports

---

### Option 2: Standalone Dashboard

The standalone dashboard works without the FastAPI backend.

```bash
streamlit run Count_video.py
```

This is useful for quick local testing.

---

### Option 3: Command Line

#### Detect Vehicles in an Image

```bash
python detect_image.py --image path/to/your/image.jpg --conf 0.4 --output path/to/save/result.jpg
```

#### Detect and Count Vehicles in a Video

```bash
python detect_video.py --input path/to/your/video.mp4 --output path/to/save/result.mp4 --conf 0.4 --line-ratio 0.5 --corridor-gap 0.08
```

---

## ⚙️ Settings You Can Adjust

### `--line-ratio`

Controls the vertical position of the counting gates.

```text
0.5 = middle of the video
```

The value is relative to the video height, so it works with different video resolutions.

### `--corridor-gap`

Controls the distance between the two gates.

A wider gap can help with fast-moving vehicles because the system has more time to confirm the vehicle's movement and direction.

Counting direction is primarily based on vertical movement, so the system works best with cameras facing relatively straight along the road.

---

## 📊 API Overview

| Action   | Endpoint                | Description                           |
| -------- | ----------------------- | ------------------------------------- |
| View     | `/models`               | Lists available models                |
| Upload   | `/runs/upload`          | Uploads a video and starts processing |
| View     | `/runs`                 | Lists previous processing runs        |
| View     | `/runs/{id}`            | Shows details for one run             |
| Delete   | `/runs/{id}`            | Deletes one run and its files         |
| Delete   | `/runs`                 | Deletes all past runs                 |
| View     | `/runs/{id}/video`      | Plays the processed video             |
| Download | `/runs/{id}/report.csv` | Downloads the results as CSV          |
| View     | `/analytics/summary`    | Shows combined statistics             |

---

## 🐞 Common Issues

**Video won't play in the dashboard**

Make sure FFmpeg is installed correctly and accessible through your system PATH.

**Van detection is weaker**

The custom model had fewer training examples for vans than some other vehicle classes, so van detection may be less accurate.

**Vehicles are miscounted at high speed**

Try increasing `--corridor-gap` to give the system more time to confirm the vehicle's movement.

**Custom model performance is lower**

The custom UVH-26 model did not perform as well as the pretrained YOLO models during testing. For this reason, `yolov8n.pt` is used as the default model.

**Custom model can't find training data**

If you retrain the model, check that your dataset folder structure and YOLO annotation format are correct.

---

## 🛡️ Credits

Built using:

* Ultralytics YOLO
* ByteTrack
* FastAPI
* SQLite
* Streamlit
* OpenCV
* FFmpeg

**Dataset:** UVH-26
**Dataset Access:** Hugging Face API
**Training Platform:** Kaggle

---

## 👨‍💻 Author

**Younas**

Computer Vision / Machine Learning
