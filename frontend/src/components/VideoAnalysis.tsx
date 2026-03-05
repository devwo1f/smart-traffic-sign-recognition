import { useState, useRef, useCallback, useEffect } from 'react';
import { API_BASE } from '../config';
import type { VideoProcessingResponse, Detection, VideoFrameResult } from '../types';
import './VideoAnalysis.css';

export default function VideoAnalysis() {
    const formatLabel = (rawLabel: string) => {
        if (rawLabel === 'other-sign') return 'Other Sign';
        const parts = rawLabel.split('--');
        if (parts.length > 1) {
            // Drop the Mapillary ID (e.g. g1) and format the rest
            const friendly = parts.slice(0, -1).join(' ').replace(/-/g, ' ');
            // Title Case
            return friendly.replace(/\b\w/g, c => c.toUpperCase());
        }
        return rawLabel;
    };

    const [file, setFile] = useState<File | null>(null);
    const [videoUrl, setVideoUrl] = useState<string | null>(null);
    const [result, setResult] = useState<VideoProcessingResponse | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [frameSkip, setFrameSkip] = useState(5);
    const [maxFrames, setMaxFrames] = useState(300);
    const [progress, setProgress] = useState(0);
    const inputRef = useRef<HTMLInputElement>(null);
    const videoRef = useRef<HTMLVideoElement>(null);
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const animationRef = useRef<number | null>(null);
    const [dragOver, setDragOver] = useState(false);

    const handleFile = useCallback((f: File) => {
        setFile(f);
        setVideoUrl(URL.createObjectURL(f));
        setResult(null);
        setError(null);
        setProgress(0);
        if (canvasRef.current) {
            const ctx = canvasRef.current.getContext('2d');
            if (ctx) ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
        }
    }, []);

    const handleDrop = useCallback((e: React.DragEvent) => {
        e.preventDefault();
        setDragOver(false);
        if (e.dataTransfer.files[0]) handleFile(e.dataTransfer.files[0]);
    }, [handleFile]);

    const analyze = async () => {
        if (!file) return;
        setLoading(true);
        setError(null);
        setProgress(30);

        try {
            const form = new FormData();
            form.append('file', file);
            const res = await fetch(
                `${API_BASE}/predict/video?frame_skip=${frameSkip}&max_frames=${maxFrames}`,
                { method: 'POST', body: form },
            );
            setProgress(90);
            if (!res.ok) throw new Error(`HTTP ${res.status}`);
            const data: VideoProcessingResponse = await res.json();
            setResult(data);
            setProgress(100);
        } catch (err: any) {
            setError(err.message || 'Video processing failed');
        } finally {
            setLoading(false);
            setTimeout(() => setProgress(0), 1000);
        }
    };

    // Draw bounding boxes continuously synced to video time
    const drawBoxes = useCallback(() => {
        const video = videoRef.current;
        const canvas = canvasRef.current;
        if (!video || !canvas || !result) return;

        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        // Match canvas physical resolution to its CSS rendered size
        const rect = video.getBoundingClientRect();
        if (canvas.width !== rect.width || canvas.height !== rect.height) {
            canvas.width = rect.width;
            canvas.height = rect.height;
        }

        ctx.clearRect(0, 0, canvas.width, canvas.height);

        // Find the closest frame result to the current video time
        const currentTimeMs = video.currentTime * 1000;

        // Find the most recent analyzed frame before the current video time
        let activeFrame: VideoFrameResult | null = null;
        for (const frame of result.frame_results) {
            // Give a 100ms grace period so boxes don't disappear instantly
            if (frame.timestamp_ms <= currentTimeMs && currentTimeMs - frame.timestamp_ms < 500) {
                activeFrame = frame;
            }
        }

        // Calculate scaling ratios (True Video Res vs Rendered CSS Res)
        const scaleX = canvas.width / video.videoWidth;
        const scaleY = canvas.height / video.videoHeight;

        // If we found an active frame, draw its boxes
        if (activeFrame && activeFrame.detections) {
            activeFrame.detections.forEach((det: Detection) => {
                if (!det.bbox) return;
                // Scale YOLO coordinates to CSS canvas coordinates
                const xmin = det.bbox.xmin * scaleX;
                const ymin = det.bbox.ymin * scaleY;
                const xmax = det.bbox.xmax * scaleX;
                const ymax = det.bbox.ymax * scaleY;

                // Draw the rectangle
                ctx.strokeStyle = '#00ff88';
                ctx.lineWidth = 3;
                ctx.beginPath();
                ctx.rect(xmin, ymin, xmax - xmin, ymax - ymin);
                ctx.stroke();

                // Draw the label
                const signName = formatLabel(det.label);
                const labelText = `${signName} (${(det.confidence * 100).toFixed(0)}%)`;
                ctx.fillStyle = '#00ff88';
                ctx.font = 'bold 16px Inter, sans-serif';
                const textWidth = ctx.measureText(labelText).width;

                // Background for text
                ctx.fillRect(xmin, ymin - 25, textWidth + 10, 25);

                // Text itself
                ctx.fillStyle = '#000000';
                ctx.fillText(labelText, xmin + 5, ymin - 7);
            });
        }

        animationRef.current = requestAnimationFrame(drawBoxes);
    }, [result]);

    // Start/stop animation loop when video plays/pauses
    useEffect(() => {
        const video = videoRef.current;
        if (!video) return;

        const handlePlay = () => {
            if (animationRef.current) cancelAnimationFrame(animationRef.current);
            animationRef.current = requestAnimationFrame(drawBoxes);
        };

        const handlePause = () => {
            if (animationRef.current) cancelAnimationFrame(animationRef.current);
            // Draw one last time when paused to ensure boxes aren't cleared
            drawBoxes();
        };

        const handleSeek = () => {
            drawBoxes();
        }

        video.addEventListener('play', handlePlay);
        video.addEventListener('pause', handlePause);
        video.addEventListener('seeked', handleSeek);
        video.addEventListener('timeupdate', handleSeek);

        return () => {
            video.removeEventListener('play', handlePlay);
            video.removeEventListener('pause', handlePause);
            video.removeEventListener('seeked', handleSeek);
            video.removeEventListener('timeupdate', handleSeek);
            if (animationRef.current) cancelAnimationFrame(animationRef.current);
        };
    }, [drawBoxes]);

    const totalDetections = result?.frame_results.reduce((s, f) => s + f.detections.length, 0) ?? 0;

    return (
        <div className="panel-grid video-analysis-panel">
            {/* Left: Upload & Config */}
            <div className="card upload-card">
                <h2>Video Analysis</h2>
                <div
                    className={`dropzone ${dragOver ? 'drag-over' : ''}`}
                    onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
                    onDragLeave={() => setDragOver(false)}
                    onDrop={handleDrop}
                    onClick={() => inputRef.current?.click()}
                >
                    <span className="dropzone-icon">🎬</span>
                    <p>{file ? `Selected: ${file.name}` : 'Drag & drop a video file'}</p>
                    <p className="dropzone-sub">MP4, AVI, MOV supported</p>
                    <input ref={inputRef} type="file" accept="video/*" hidden
                        onChange={(e) => e.target.files?.[0] && handleFile(e.target.files[0])} />
                </div>

                <div className="video-controls">
                    <label>
                        Frame skip:
                        <input type="number" className="input-small" value={frameSkip}
                            min={1} max={30} onChange={(e) => setFrameSkip(Number(e.target.value))} />
                    </label>
                    <label>
                        Max frames:
                        <input type="number" className="input-small" value={maxFrames}
                            min={10} max={5000} onChange={(e) => setMaxFrames(Number(e.target.value))} />
                    </label>
                </div>

                <button className="btn btn-primary" disabled={!file || loading} onClick={analyze}>
                    {loading ? <><span className="spinner" /> Processing on Server…</> : <><span className="btn-icon">▶️</span> Analyze Video</>}
                </button>

                {progress > 0 && (
                    <div className="progress-bar">
                        <div className="progress-fill" style={{ width: `${progress}%` }} />
                    </div>
                )}

                {result && (
                    <div className="video-stats" style={{ marginTop: '2rem' }}>
                        <div className="stat-box">
                            <div className="stat-value">{result.processed_frames}</div>
                            <div className="stat-label">Frames Processed</div>
                        </div>
                        <div className="stat-box">
                            <div className="stat-value">{totalDetections}</div>
                            <div className="stat-label">Total Detections</div>
                        </div>
                        <div className="stat-box">
                            <div className="stat-value">{(result.total_processing_time_ms / 1000).toFixed(1)}s</div>
                            <div className="stat-label">Processing Time</div>
                        </div>
                    </div>
                )}
            </div>

            {/* Right: Video Playback */}
            <div className="card video-card">
                <h2>Playback & Tracking</h2>
                {error && <div className="alert alert-error">⚠️ {error}</div>}

                {!videoUrl ? (
                    <div className="placeholder">
                        <span className="placeholder-icon">🎞️</span>
                        <p>Upload a video to view playback</p>
                    </div>
                ) : (
                    <div className="video-container">
                        <video
                            ref={videoRef}
                            src={videoUrl}
                            controls
                            className="video-player"
                            muted
                        />
                        <canvas
                            ref={canvasRef}
                            className="video-canvas-overlay"
                        />
                        {!result && !loading && (
                            <div className="video-overlay-msg">
                                Click "Analyze Video" to track Traffic Signs
                            </div>
                        )}
                    </div>
                )}
            </div>
        </div>
    );
}
