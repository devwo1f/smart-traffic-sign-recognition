import { useState, useCallback, useRef } from 'react';
import { API_BASE } from '../config';
import DetectionCard from './DetectionCard';
import type { PredictionResponse } from '../types';
import './ImageUpload.css';

export default function ImageUpload() {
    const [file, setFile] = useState<File | null>(null);
    const [preview, setPreview] = useState<string | null>(null);
    const [result, setResult] = useState<PredictionResponse | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [dragOver, setDragOver] = useState(false);
    const inputRef = useRef<HTMLInputElement>(null);

    const handleFile = useCallback((f: File) => {
        setFile(f);
        setResult(null);
        setError(null);
        const reader = new FileReader();
        reader.onload = (e) => setPreview(e.target?.result as string);
        reader.readAsDataURL(f);
    }, []);

    const handleDrop = useCallback((e: React.DragEvent) => {
        e.preventDefault();
        setDragOver(false);
        if (e.dataTransfer.files[0]) handleFile(e.dataTransfer.files[0]);
    }, [handleFile]);

    const clearSelection = () => {
        setFile(null);
        setPreview(null);
        setResult(null);
        setError(null);
    };

    const predict = async () => {
        if (!file) return;
        setLoading(true);
        setError(null);
        try {
            const form = new FormData();
            form.append('file', file);
            const res = await fetch(`${API_BASE}/predict`, { method: 'POST', body: form });
            if (!res.ok) throw new Error(`HTTP ${res.status}`);
            setResult(await res.json());
        } catch (err: any) {
            setError(err.message || 'Prediction failed');
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="panel-grid">
            {/* Left: Upload */}
            <div className="card">
                <h2>Upload Traffic Sign Image</h2>

                {!preview ? (
                    <div
                        className={`dropzone ${dragOver ? 'drag-over' : ''}`}
                        onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
                        onDragLeave={() => setDragOver(false)}
                        onDrop={handleDrop}
                        onClick={() => inputRef.current?.click()}
                    >
                        <span className="dropzone-icon">📤</span>
                        <p>Drag &amp; drop an image here</p>
                        <p className="dropzone-sub">or click to browse</p>
                        <input ref={inputRef} type="file" accept="image/*" hidden
                            onChange={(e) => e.target.files?.[0] && handleFile(e.target.files[0])} />
                    </div>
                ) : (
                    <div className="preview-area">
                        <img src={preview} alt="Preview" />
                        <button className="btn btn-secondary" onClick={clearSelection}>Clear</button>
                    </div>
                )}

                <button className="btn btn-primary" disabled={!file || loading} onClick={predict}>
                    {loading ? <><span className="spinner" /> Analyzing…</> : <><span className="btn-icon">🔍</span> Analyze Image</>}
                </button>
            </div>

            {/* Right: Results */}
            <div className="card">
                <h2>Detection Results</h2>

                {error && (
                    <div className="alert alert-error">⚠️ {error}. Ensure the backend is running at {API_BASE}</div>
                )}

                {!result && !error && (
                    <div className="placeholder">
                        <span className="placeholder-icon">🔮</span>
                        <p>Upload an image to see predictions</p>
                    </div>
                )}

                {result && (
                    <>
                        <div className="inference-badge">
                            ⚡ Inference: {result.inference_time_ms.toFixed(1)}ms &mdash; {result.detections.length} sign{result.detections.length !== 1 ? 's' : ''} detected
                        </div>
                        {result.detections.length === 0 ? (
                            <p className="text-muted text-center" style={{ padding: 20 }}>No traffic signs detected</p>
                        ) : (
                            <div className="detections-list">
                                {result.detections.map((d, i) => <DetectionCard key={i} detection={d} />)}
                            </div>
                        )}
                    </>
                )}
            </div>
        </div>
    );
}
