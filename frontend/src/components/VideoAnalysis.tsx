import { useState, useRef, useCallback } from 'react';
import { API_BASE } from '../config';
import type { VideoProcessingResponse, Detection } from '../types';
import './VideoAnalysis.css';

export default function VideoAnalysis() {
    const [file, setFile] = useState<File | null>(null);
    const [result, setResult] = useState<VideoProcessingResponse | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [frameSkip, setFrameSkip] = useState(3);
    const [maxFrames, setMaxFrames] = useState(300);
    const [progress, setProgress] = useState(0);
    const inputRef = useRef<HTMLInputElement>(null);
    const [dragOver, setDragOver] = useState(false);

    const handleFile = useCallback((f: File) => {
        setFile(f);
        setResult(null);
        setError(null);
        setProgress(0);
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

    const totalDetections = result?.frame_results.reduce((s, f) => s + f.detections.length, 0) ?? 0;
    const framesWithDets = result?.frame_results.filter((f) => f.detections.length > 0) ?? [];

    return (
        <div className="panel-grid">
            {/* Left: Upload */}
            <div className="card">
                <h2>Upload Video</h2>
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
                            min={10} max={1000} onChange={(e) => setMaxFrames(Number(e.target.value))} />
                    </label>
                </div>

                <button className="btn btn-primary" disabled={!file || loading} onClick={analyze}>
                    {loading ? <><span className="spinner" /> Processing…</> : <><span className="btn-icon">▶️</span> Analyze Video</>}
                </button>

                {progress > 0 && (
                    <div className="progress-bar">
                        <div className="progress-fill" style={{ width: `${progress}%` }} />
                    </div>
                )}
            </div>

            {/* Right: Results */}
            <div className="card">
                <h2>Video Results</h2>

                {error && <div className="alert alert-error">⚠️ {error}</div>}

                {!result && !error && (
                    <div className="placeholder">
                        <span className="placeholder-icon">🎞️</span>
                        <p>Upload a video to see frame-by-frame results</p>
                    </div>
                )}

                {result && (
                    <>
                        <div className="video-stats">
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

                        {framesWithDets.length === 0 ? (
                            <p className="text-muted text-center">No traffic signs detected in video</p>
                        ) : (
                            <div className="frame-list">
                                {framesWithDets.slice(0, 50).map((fr) => {
                                    const time = (fr.timestamp_ms / 1000).toFixed(1);
                                    return (
                                        <div key={fr.frame_number} className="detection-card">
                                            <div className="det-class-id">F{fr.frame_number}</div>
                                            <div className="det-info">
                                                <div className="det-label">{fr.detections.length} sign(s) at {time}s</div>
                                                <div className="det-sublabel">
                                                    {fr.detections.map((d: Detection) => d.label.split('--').pop()).join(', ')}
                                                </div>
                                            </div>
                                        </div>
                                    );
                                })}
                            </div>
                        )}
                    </>
                )}
            </div>
        </div>
    );
}
