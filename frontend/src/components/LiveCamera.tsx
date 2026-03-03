import { useState, useRef, useCallback, useEffect } from 'react';
import { WS_BASE } from '../config';
import type { Detection } from '../types';
import './LiveCamera.css';

export default function LiveCamera() {
    const [streaming, setStreaming] = useState(false);
    const [detections, setDetections] = useState<Detection[]>([]);
    const [fps, setFps] = useState(0);
    const [error, setError] = useState<string | null>(null);

    const videoRef = useRef<HTMLVideoElement>(null);
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const wsRef = useRef<WebSocket | null>(null);
    const streamRef = useRef<MediaStream | null>(null);
    const rafRef = useRef<number>(0);
    const fpsCounterRef = useRef(0);
    const lastFpsTimeRef = useRef(0);

    // Cleanup on unmount
    useEffect(() => {
        return () => stopCamera();
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []);

    const startCamera = useCallback(async () => {
        setError(null);
        try {
            const stream = await navigator.mediaDevices.getUserMedia({
                video: { width: 640, height: 480, facingMode: 'environment' },
            });
            streamRef.current = stream;

            if (videoRef.current) {
                videoRef.current.srcObject = stream;
                await videoRef.current.play();
            }

            // Connect WebSocket
            const ws = new WebSocket(`${WS_BASE}/ws/stream`);
            ws.binaryType = 'arraybuffer';

            ws.onmessage = (event) => {
                const data = JSON.parse(event.data);
                setDetections(data.detections || []);

                fpsCounterRef.current++;
                const now = performance.now();
                if (now - lastFpsTimeRef.current >= 1000) {
                    setFps(fpsCounterRef.current);
                    fpsCounterRef.current = 0;
                    lastFpsTimeRef.current = now;
                }
            };

            ws.onerror = () => setError('WebSocket connection failed');
            ws.onclose = () => { };

            wsRef.current = ws;
            lastFpsTimeRef.current = performance.now();
            setStreaming(true);

            // Start capture loop
            const captureLoop = () => {
                const video = videoRef.current;
                const canvas = canvasRef.current;
                if (!video || !canvas) return;

                const ctx = canvas.getContext('2d');
                if (!ctx) return;

                canvas.width = video.videoWidth || 640;
                canvas.height = video.videoHeight || 480;
                ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

                if (wsRef.current?.readyState === WebSocket.OPEN) {
                    canvas.toBlob(
                        (blob) => {
                            if (blob && wsRef.current?.readyState === WebSocket.OPEN) {
                                blob.arrayBuffer().then((buf) => wsRef.current?.send(buf));
                            }
                        },
                        'image/jpeg',
                        0.7,
                    );
                }

                rafRef.current = requestAnimationFrame(captureLoop);
            };

            captureLoop();
        } catch (err: any) {
            setError(`Camera error: ${err.message}`);
        }
    }, []);

    const stopCamera = useCallback(() => {
        streamRef.current?.getTracks().forEach((t) => t.stop());
        streamRef.current = null;
        wsRef.current?.close();
        wsRef.current = null;
        cancelAnimationFrame(rafRef.current);
        setStreaming(false);
        setDetections([]);
        setFps(0);
    }, []);

    // Draw bounding boxes on canvas
    useEffect(() => {
        if (!canvasRef.current || detections.length === 0) return;
        const ctx = canvasRef.current.getContext('2d');
        if (!ctx) return;

        for (const det of detections) {
            if (!det.bbox) continue;
            const { xmin, ymin, xmax, ymax } = det.bbox;

            ctx.strokeStyle = '#10b981';
            ctx.lineWidth = 2;
            ctx.strokeRect(xmin, ymin, xmax - xmin, ymax - ymin);

            const label = `${det.label.split('--').pop()} ${(det.confidence * 100).toFixed(0)}%`;
            ctx.font = 'bold 12px Inter, sans-serif';
            const tw = ctx.measureText(label).width;
            ctx.fillStyle = 'rgba(16, 185, 129, 0.85)';
            ctx.fillRect(xmin, ymin - 18, tw + 8, 20);
            ctx.fillStyle = '#000';
            ctx.fillText(label, xmin + 4, ymin - 4);
        }
    }, [detections]);

    return (
        <div className="camera-layout">
            {/* Camera Feed */}
            <div className="card camera-card">
                <h2>Webcam Feed</h2>
                <div className="camera-container">
                    <video ref={videoRef} autoPlay playsInline />
                    <canvas ref={canvasRef} />
                    {!streaming && (
                        <div className="camera-overlay">
                            <span className="placeholder-icon">📷</span>
                            <p>{error || 'Camera preview will appear here'}</p>
                        </div>
                    )}
                </div>
                <div className="camera-controls">
                    <button className="btn btn-primary" style={{ flex: 1, marginTop: 0 }} onClick={startCamera} disabled={streaming}>
                        ▶️ Start Camera
                    </button>
                    <button className="btn btn-danger" onClick={stopCamera} disabled={!streaming}>
                        ⏹️ Stop
                    </button>
                    <div className="camera-fps">{streaming ? `${fps} FPS` : '— FPS'}</div>
                </div>
            </div>

            {/* Live Detections */}
            <div className="card">
                <h2>Live Detections</h2>
                <div className="live-detections">
                    {!streaming && <p className="text-muted">Start the camera to see live detections</p>}
                    {streaming && detections.length === 0 && (
                        <p className="text-muted" style={{ fontSize: '0.85rem' }}>Scanning for traffic signs…</p>
                    )}
                    {detections.map((det, i) => {
                        const confClass = det.confidence >= 0.8 ? 'high' : det.confidence >= 0.5 ? 'medium' : 'low';
                        const label = det.label.replace(/--/g, ' › ').replace(/-/g, ' ');
                        return (
                            <div key={i} className="detection-card" style={{ padding: 10 }}>
                                <div className="det-class-id">#{det.class_id}</div>
                                <div className="det-info">
                                    <div className="det-label" style={{ fontSize: '0.85rem' }}>{label}</div>
                                </div>
                                <span className={`badge badge-${confClass}`}>{(det.confidence * 100).toFixed(0)}%</span>
                            </div>
                        );
                    })}
                </div>
            </div>
        </div>
    );
}
