# Training Workflow

## End-to-End Pipeline

```mermaid
flowchart TD
    A["📥 Download MTSD Dataset<br/>Fully Annotated Set"] --> B["🔧 Preprocess<br/>Extract & organize images"]
    
    B --> C{"Split Pipeline"}
    
    C -->|Detection| D["📐 Prepare YOLO Data<br/>Convert to YOLO format"]
    C -->|Classification| E["✂️ Crop Signs<br/>Using bounding boxes"]

    D --> F["🏋️ Train YOLOv8<br/>Fine-tune on MTSD"]
    E --> G["🏋️ Train EfficientNet<br/>Transfer learning + AMP"]

    F --> H["📊 Evaluate Detector<br/>mAP, Precision, Recall"]
    G --> I["📊 Evaluate Classifier<br/>Accuracy, F1, Confusion Matrix"]

    H --> J{"Meets Target?<br/>mAP > 0.5"}
    I --> K{"Meets Target?<br/>Accuracy > 95%"}

    J -->|No| F
    J -->|Yes| L["📦 Export ONNX<br/>YOLOv8 Detector"]
    K -->|No| G
    K -->|Yes| M["📦 Export ONNX<br/>EfficientNet Classifier"]

    L --> N["⚡ TensorRT Quantize<br/>FP16 / INT8"]
    M --> N

    N --> O["🏎️ Benchmark<br/>Target: 60 FPS"]
    O --> P["🏷️ Version Model<br/>Tag + metadata"]
    P --> Q["🚀 Deploy<br/>Update backend models"]

    subgraph Retrain["♻️ Continuous Retraining"]
        R["New Labeled Data<br/>data/new_labels/"] --> S["Merge with<br/>Training Set"]
        S --> G
        S --> F
    end
```

## Training Configuration

| Parameter         | EfficientNet          | YOLOv8               |
|-------------------|-----------------------|----------------------|
| **Input Size**    | 224 × 224             | 640 × 640            |
| **Batch Size**    | 64                    | 16                   |
| **Optimizer**     | AdamW                 | SGD (ultralytics)    |
| **Scheduler**     | CosineAnnealingLR     | Cosine (built-in)    |
| **Epochs**        | 50 (early stopping)   | 100                  |
| **Precision**     | Mixed (AMP)           | FP16                 |
| **Augmentation**  | Rotation, Jitter, Blur| Mosaic, MixUp, HSV   |

## Retraining Trigger

1. New labeled images placed in `data/new_labels/`
2. `POST /retrain` endpoint or `python ml/retrain.py`
3. Pipeline merges new data → retrains → re-exports → versions
