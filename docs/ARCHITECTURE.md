# Architecture

## System Architecture

```mermaid
graph TB
    subgraph Client["Frontend (Browser)"]
        DASH[Dashboard UI]
        IMG[Image Upload]
        VID[Video Upload]
        CAM[Webcam Stream]
        LOGS[Inspection Logs]
    end

    subgraph API["Backend (FastAPI)"]
        REST["REST API<br/>/predict, /predict/batch<br/>/history, /retrain"]
        WSS["WebSocket<br/>/ws/stream"]
        VIDP["Video Processor<br/>(OpenCV)"]
        INF["Inference Engine<br/>(ONNX Runtime)"]
    end

    subgraph ML["Dual-Model Pipeline"]
        YOLO["YOLOv8 Detector<br/>Locate signs in frame"]
        EFFN["EfficientNet Classifier<br/>Identify sign type"]
    end

    subgraph Data["Data Layer"]
        PG[("PostgreSQL<br/>Predictions & Logs")]
        FS["File Storage<br/>Model Artifacts"]
        MV["Model Versioning<br/>versions.json"]
    end

    subgraph Train["Training Pipeline"]
        DS["MTSD Dataset"]
        PP["Preprocessing<br/>Crop + Split"]
        TR["Training<br/>PyTorch + AMP"]
        EV["Evaluation<br/>Metrics + Confusion"]
        EX["Export<br/>ONNX"]
        TRT["Quantize<br/>TensorRT FP16"]
    end

    IMG -->|POST /predict| REST
    VID -->|POST /predict/video| VIDP
    CAM -->|WebSocket frames| WSS
    LOGS -->|GET /history| REST

    REST --> INF
    WSS --> INF
    VIDP --> INF

    INF --> YOLO
    YOLO -->|Cropped regions| EFFN
    EFFN -->|Class + Confidence| INF

    REST --> PG
    INF --> FS

    DS --> PP --> TR --> EV --> EX --> TRT
    TRT --> FS
    EX --> FS
    TR --> MV
```

## Component Details

### Frontend
- **Technology**: TypeScript + HTML + CSS
- **Tabs**: Image Upload, Video Analysis, Live Camera, Inspection Logs
- **Communication**: REST API for images/history, WebSocket for real-time camera

### Backend
- **Framework**: FastAPI (Python 3.11)
- **Inference**: ONNX Runtime with optional TensorRT backend
- **Video**: OpenCV for frame extraction and annotation
- **Database**: PostgreSQL via SQLAlchemy (async)

### ML Pipeline
- **Detection**: YOLOv8n/s fine-tuned on Mapillary MTSD
- **Classification**: EfficientNet-B0 with transfer learning
- **Optimization**: ONNX export → TensorRT FP16/INT8 quantization
- **Target**: >95% accuracy, ~60 FPS inference
