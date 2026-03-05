# 🚦 Traffic Sign Recognition and Review System

A production-grade traffic sign recognition system using **YOLOv8** (detection) + **EfficientNet** (classification) with real-time video processing, powered by PyTorch, FastAPI, and a modern TypeScript dashboard.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## Video Demo

<video src="https://github.com/user-attachments/assets/48d0e9ff-57f5-4079-9d5d-8bf8ffafe098" controls="controls" muted="muted"></video>

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

## 📜 History of Implementation

### V1 — Baseline EfficientNet-B3 (March 2026)

#### Results

| Metric | Score |
|---|---|
| **Test Accuracy** | **72.84%** |
| **Weighted Precision** | 85.77% |
| **Weighted Recall** | 72.84% |
| **Weighted F1** | 74.86% |
| **Classes** | 368 |
| **Dataset** | Mapillary MTSD (143K train / 30K val / 31K test) |
| **Total Training Time** | ~7 hours (50 epochs) |

Best checkpoint saved at epoch 44 with 71.97% validation accuracy.

**Top performing classes:** `bicycles-only`, `divided-highway-ends`, `pedestrians-crossing` — all achieved 100% F1.

**Challenging classes:** Visually similar signs like `no-parking-or-no-stopping`, `interstate-route`, and rare classes with few samples struggled (~15% F1).

#### GPU & CPU Optimization Journey

Training runs on an **NVIDIA RTX 4060 Laptop GPU (8GB VRAM)**. The original configuration caused a system crash at 100% GPU utilization. Below is how we iteratively optimized the pipeline:

**Problem:** Batch size 256 + EfficientNet-B3 exceeded 8GB VRAM, causing the system to crash.

**Step 1 — Reduce VRAM pressure (prevent crashes):**

| Setting | Before | After | Rationale |
|---|---|---|---|
| Batch size | 256 | 64 | 4× less VRAM per forward pass |
| Learning rate | 4e-3 | 1e-3 | Scaled proportionally with batch size (linear scaling rule) |

**Step 2 — Recover effective batch size with gradient accumulation:**

Rather than increasing batch size (risky on 8GB), we simulate batch 128 by accumulating gradients over 2 micro-batches of 64. This gives identical training dynamics without the VRAM spike.

```
GRADIENT_ACCUMULATION_STEPS = 2  # Effective batch = 64 × 2 = 128
```

The training loop divides the loss by `accum_steps` and only calls `optimizer.step()` every 2 batches, so gradient magnitudes remain equivalent to a true batch-128 run.

**Step 3 — Maximize data throughput (CPU-side):**

The initial training speed was only **5.3 it/s** — the GPU was starved waiting for data. Data loading uses CPU RAM (not GPU VRAM), so these changes are safe:

| Setting | Before | After | Impact |
|---|---|---|---|
| Data workers | 2 | 6 | 3× more parallel data loading threads |
| Prefetch factor | 2 | 4 | Pre-loads more batches so GPU never waits |
| Persistent workers | `False` | `True` | Workers stay alive between epochs (no respawn overhead) |
| cuDNN benchmark | `False` | `True` | Auto-tunes CUDA kernels for fixed 224×224 input |

**Result:** Training speed improved from **5.3 → 10.4 it/s** (~2× faster).

**Final configuration summary:**

```python
CLASSIFIER_BACKBONE = "efficientnet_b3"    # 11.6M params
CLASSIFIER_BATCH_SIZE = 64                 # In-VRAM batch (safe for 8GB)
GRADIENT_ACCUMULATION_STEPS = 2            # Effective batch = 128
CLASSIFIER_LR = 1e-3                       # Scaled to effective batch
USE_AMP = True                             # FP16 mixed precision
CUDNN_BENCHMARK = True                     # Kernel auto-tuning
NUM_WORKERS = 6                            # Parallel data loading
PREFETCH_FACTOR = 4                        # Pre-loaded batches
PERSISTENT_WORKERS = True                  # No worker respawn
```

**Estimated VRAM usage:** ~5–6 GB out of 8 GB available — stable with no crashes.

---

### V2 — YOLOv8 Detection & Real-Time Tracking (March 2026)

#### Pipeline Architecture
The system was upgraded from pure whole-image classification to a two-stage detection pipeline:
1. **YOLOv8x** isolates the bounding boxes of potential traffic signs via standard Non-Maximum Suppression (NMS).
2. The exact rectangular crops are dynamically sliced and fed into the **EfficientNet-B3** classifier.
3. Both models are quantized and exported to `.onnx` for portability.

#### Windows GPU Acceleration via DirectML
Due to the complexities of installing explicit NVIDIA CUDA 12.x and cuDNN 9.x toolkits natively on a Windows host machine, the backend inference engine was refactored to use **Microsoft DirectML** (`onnxruntime-directml`). 

This allows ONNX to natively tap into DirectX 12, unlocking the full hardware acceleration of the NVIDIA RTX 4060 GPU out-of-the-box without strict CUDA dependencies. Frame processing times for the massive 68-million parameter YOLOv8x model dropped from ~600ms (CPU) to <50ms (GPU).

#### Frontend Video Analysis
The React frontend dashboard was upgraded to support real-time video playback analysis.
Instead of returning static lists of timestamps, the frontend uses an HTML5 `<video>` element overlaid with an invisible `<canvas>`. As the video plays, a `requestAnimationFrame` loop calculates the dynamic CSS scaling ratio and draws the JSON bounding boxes (e.g. `[x1, y1, x2, y2]`) perfectly tracking the moving vehicles and signs at 60 FPS.

---

## 📜 License

MIT License — see [LICENSE](LICENSE) for details.
