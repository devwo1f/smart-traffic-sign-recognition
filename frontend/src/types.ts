/* Shared TypeScript types */

export interface Detection {
    class_id: number;
    label: string;
    confidence: number;
    bbox?: { xmin: number; ymin: number; xmax: number; ymax: number };
}

export interface PredictionResponse {
    filename: string;
    detections: Detection[];
    inference_time_ms: number;
    model_version: string;
}

export interface BatchPredictionResponse {
    results: PredictionResponse[];
    total_images: number;
    total_inference_time_ms: number;
}

export interface VideoFrameResult {
    frame_number: number;
    timestamp_ms: number;
    detections: Detection[];
}

export interface VideoProcessingResponse {
    filename: string;
    total_frames: number;
    processed_frames: number;
    fps: number;
    frame_results: VideoFrameResult[];
    total_processing_time_ms: number;
}

export interface HistoryItem {
    id: number;
    image_filename: string;
    predicted_class: number;
    predicted_label: string;
    confidence: number;
    model_version: string;
    latency_ms: number | null;
    source_type: string;
    created_at: string;
}

export interface HistoryResponse {
    items: HistoryItem[];
    total: number;
    page: number;
    page_size: number;
    total_pages: number;
}

export type TabId = 'upload' | 'video' | 'camera' | 'logs';
