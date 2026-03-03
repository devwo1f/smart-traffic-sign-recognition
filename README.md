# 🚦 Traffic Sign Recognition and Review System

A production-grade traffic sign recognition system using **YOLOv8** (detection) + **EfficientNet** (classification) with real-time video processing, powered by PyTorch, FastAPI, and a modern TypeScript dashboard.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 🏗️ Architecture

```
Video/Image → [YOLOv8 Detector] → Crop Regions → [EfficientNet Classifier] → Results
                                                                                 ↓
Frontend Dashboard ← FastAPI Backend ← PostgreSQL (logs) + ONNX Runtime (inference)
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full system diagram.

---

## 📁 Project Structure

```
traffic-sign-system/
├── ml/                    # ML pipeline (training, evaluation, export)
├── backend/               # FastAPI REST API + WebSocket server
├── frontend/              # TypeScript + HTML dashboard
├── docs/                  # Architecture & workflow diagrams
├── docker-compose.yml     # Full-stack deployment
└── .github/workflows/     # CI pipeline
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Node.js 18+ (for TypeScript compilation)
- PostgreSQL 16+ (or use Docker)
- NVIDIA GPU + CUDA (optional, for TensorRT acceleration)

### 1. Clone & Setup

```bash
git clone https://github.com/<your-username>/traffic-sign-system.git
cd traffic-sign-system

# ML environment
cd ml
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac
pip install -r requirements.txt

# Backend environment
cd ../backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# Frontend
cd ../frontend
npm install
```

### 2. Download Dataset

Download the **Mapillary Traffic Sign Dataset (MTSD)** fully annotated set:
- `mtsd_fully_annotated_annotation.zip`
- `mtsd_fully_annotated_images.train.0.zip` through `.train.2.zip`
- `mtsd_fully_annotated_images.val.zip`
- `mtsd_fully_annotated_images.test.zip`

Place all zips in `ml/data/raw/` then run:

```bash
cd ml
python download_dataset.py
```

### 3. Train Models

```bash
# Preprocess dataset (crop signs, create splits)
python preprocess.py

# Prepare YOLOv8 data format
python prepare_yolo_data.py

# Train EfficientNet classifier
python train.py

# Train YOLOv8 detector
python train_yolo.py

# Evaluate
python evaluate.py

# Export to ONNX
python export_onnx.py

# (Optional) TensorRT quantization
python quantize_tensorrt.py

# Benchmark inference speed
python benchmark.py
```

### 4. Start Backend

```bash
cd backend

# Set environment variables
set DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/traffic_signs
set MODEL_PATH=../ml/models

uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

API docs available at: `http://localhost:8000/docs`

### 5. Start Frontend

```bash
cd frontend
npx tsc
# Open index.html in your browser, or serve with:
python -m http.server 3000
```

---

## 🐳 Docker Deployment

```bash
docker-compose up --build
```

| Service   | URL                        |
|-----------|----------------------------|
| Frontend  | http://localhost:3000       |
| Backend   | http://localhost:8000       |
| API Docs  | http://localhost:8000/docs  |
| PostgreSQL| localhost:5432              |

---

## 📊 API Endpoints

| Method | Endpoint           | Description                              |
|--------|--------------------|------------------------------------------|
| POST   | `/predict`         | Upload single image for prediction       |
| POST   | `/predict/batch`   | Upload multiple images                   |
| POST   | `/predict/video`   | Upload video for frame-by-frame analysis |
| WS     | `/ws/stream`       | Real-time webcam stream processing       |
| GET    | `/history`         | Fetch inspection logs (paginated)        |
| POST   | `/retrain`         | Trigger model retraining pipeline        |

---

## 🎯 Performance Targets

| Metric                    | Target          |
|---------------------------|-----------------|
| Classification Accuracy   | >95%            |
| Inference Speed (TensorRT)| ~60 FPS         |
| TensorRT Speedup          | 3x vs PyTorch   |
| API Response (single img) | <100ms          |
| Log Filtering             | Sub-second      |

---

## 🔄 Model Versioning

Models are versioned semantically (`v1.0.0`, `v1.1.0`, etc.). Each version tracks:
- Accuracy metrics
- Training config hash
- ONNX + TensorRT artifacts
- Timestamp

```bash
# List versions
python ml/version_manager.py list

# Rollback
python ml/version_manager.py rollback v1.0.0
```

---

## 🌿 Branch Strategy

| Branch      | Purpose                              |
|-------------|--------------------------------------|
| `main`      | Production-ready releases            |
| `dev`       | Integration & testing                |
| `feature/*` | Individual feature development       |

See [docs/BRANCH_STRATEGY.md](docs/BRANCH_STRATEGY.md) for full workflow.

---

## 📄 GitHub Initialization

```bash
git init
git add .
git commit -m "Initial commit: Traffic Sign Recognition System"
git branch -M main
git remote add origin https://github.com/<your-username>/traffic-sign-system.git
git push -u origin main

# Create dev branch
git checkout -b dev
git push -u origin dev
```

---

## 📜 License

MIT License — see [LICENSE](LICENSE) for details.
