/**
 * Traffic Sign Recognition Dashboard
 *
 * TypeScript application for the frontend dashboard.
 * Handles image/video upload, webcam streaming, and inspection log filtering.
 */

// ============================================================================
// Configuration
// ============================================================================

const API_BASE = "http://localhost:8000";
const WS_BASE = "ws://localhost:8000";

// ============================================================================
// State
// ============================================================================

interface Detection {
    class_id: number;
    label: string;
    confidence: number;
    bbox?: { xmin: number; ymin: number; xmax: number; ymax: number };
}

interface PredictionResponse {
    filename: string;
    detections: Detection[];
    inference_time_ms: number;
    model_version: string;
}

interface HistoryItem {
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

interface HistoryResponse {
    items: HistoryItem[];
    total: number;
    page: number;
    page_size: number;
    total_pages: number;
}

let selectedFile: File | null = null;
let selectedVideoFile: File | null = null;
let cameraStream: MediaStream | null = null;
let wsConnection: WebSocket | null = null;
let cameraAnimationId: number | null = null;
let currentPage = 1;
let totalPages = 1;
let fpsCounter = 0;
let lastFpsTime = 0;

// ============================================================================
// Initialization
// ============================================================================

document.addEventListener("DOMContentLoaded", () => {
    initTabs();
    initImageUpload();
    initVideoUpload();
    initCamera();
    initLogs();
    checkApiStatus();
});

// ============================================================================
// Tab Management
// ============================================================================

function initTabs(): void {
    const tabBtns = document.querySelectorAll<HTMLButtonElement>(".tab-btn");
    tabBtns.forEach((btn) => {
        btn.addEventListener("click", () => {
            const tabId = btn.dataset.tab!;
            // Update buttons
            tabBtns.forEach((b) => b.classList.remove("active"));
            btn.classList.add("active");
            btn.setAttribute("aria-selected", "true");
            // Update panels
            document.querySelectorAll<HTMLElement>(".tab-panel").forEach((p) => {
                p.classList.remove("active");
            });
            document.getElementById(`panel-${tabId}`)!.classList.add("active");

            // Load logs when switching to logs tab
            if (tabId === "logs") {
                loadHistory();
            }
        });
    });
}

// ============================================================================
// API Status Check
// ============================================================================

async function checkApiStatus(): Promise<void> {
    const dot = document.querySelector<HTMLElement>(".status-dot")!;
    const text = document.getElementById("statusText")!;

    try {
        const res = await fetch(`${API_BASE}/health`);
        if (res.ok) {
            dot.classList.add("connected");
            text.textContent = "API Connected";
        } else {
            text.textContent = "API Error";
        }
    } catch {
        text.textContent = "API Offline";
    }
}

// ============================================================================
// Image Upload
// ============================================================================

function initImageUpload(): void {
    const dropzone = document.getElementById("dropzone")!;
    const fileInput = document.getElementById("fileInput") as HTMLInputElement;
    const predictBtn = document.getElementById("predictBtn") as HTMLButtonElement;
    const clearBtn = document.getElementById("clearBtn") as HTMLButtonElement;

    // Click to browse
    dropzone.addEventListener("click", () => fileInput.click());

    // Drag and drop
    dropzone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropzone.classList.add("drag-over");
    });
    dropzone.addEventListener("dragleave", () => {
        dropzone.classList.remove("drag-over");
    });
    dropzone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropzone.classList.remove("drag-over");
        const files = (e as DragEvent).dataTransfer?.files;
        if (files && files[0]) handleImageSelect(files[0]);
    });

    // File input
    fileInput.addEventListener("change", () => {
        if (fileInput.files && fileInput.files[0]) {
            handleImageSelect(fileInput.files[0]);
        }
    });

    // Predict button
    predictBtn.addEventListener("click", () => predictImage());

    // Clear button
    clearBtn.addEventListener("click", () => {
        selectedFile = null;
        document.getElementById("previewArea")!.style.display = "none";
        dropzone.style.display = "block";
        predictBtn.disabled = true;
        document.getElementById("resultsContent")!.style.display = "none";
        document.getElementById("resultsPlaceholder")!.style.display = "block";
    });
}

function handleImageSelect(file: File): void {
    selectedFile = file;
    const reader = new FileReader();
    reader.onload = (e) => {
        const img = document.getElementById("previewImage") as HTMLImageElement;
        img.src = e.target!.result as string;
        document.getElementById("previewArea")!.style.display = "block";
        document.getElementById("dropzone")!.style.display = "none";
        (document.getElementById("predictBtn") as HTMLButtonElement).disabled = false;
    };
    reader.readAsDataURL(file);
}

async function predictImage(): Promise<void> {
    if (!selectedFile) return;

    const btn = document.getElementById("predictBtn") as HTMLButtonElement;
    btn.disabled = true;
    btn.innerHTML = '<span class="spinner"></span> Analyzing...';

    try {
        const formData = new FormData();
        formData.append("file", selectedFile);

        const res = await fetch(`${API_BASE}/predict`, {
            method: "POST",
            body: formData,
        });

        if (!res.ok) throw new Error(`HTTP ${res.status}`);

        const data: PredictionResponse = await res.json();
        displayResults(data);
    } catch (err) {
        console.error("Prediction error:", err);
        const content = document.getElementById("resultsContent")!;
        content.style.display = "block";
        document.getElementById("resultsPlaceholder")!.style.display = "none";
        content.innerHTML = `<div class="inference-time" style="background: rgba(239,68,68,0.1); border-color: rgba(239,68,68,0.2); color: var(--accent-danger);">Error: Could not connect to API. Ensure the backend is running.</div>`;
    } finally {
        btn.disabled = false;
        btn.innerHTML = '<span class="btn-icon">🔍</span> Analyze Image';
    }
}

function displayResults(data: PredictionResponse): void {
    document.getElementById("resultsPlaceholder")!.style.display = "none";
    const content = document.getElementById("resultsContent")!;
    content.style.display = "block";

    // Inference time
    document.getElementById("inferenceTime")!.textContent =
        `⚡ Inference: ${data.inference_time_ms.toFixed(1)}ms | ${data.detections.length} signs detected`;

    // Detection list
    const list = document.getElementById("detectionsList")!;
    if (data.detections.length === 0) {
        list.innerHTML = '<p class="text-muted" style="text-align:center; padding: 20px;">No traffic signs detected</p>';
        return;
    }

    list.innerHTML = data.detections
        .map((det) => {
            const confPercent = (det.confidence * 100).toFixed(1);
            const confClass = det.confidence >= 0.8 ? "high" : det.confidence >= 0.5 ? "medium" : "low";
            const labelShort = det.label.replace(/--/g, " › ").replace(/-/g, " ");

            return `
                <div class="detection-item">
                    <div class="detection-class-id">#${det.class_id}</div>
                    <div class="detection-info">
                        <div class="detection-label">${labelShort}</div>
                        <div class="detection-sublabel">Class ID: ${det.class_id}</div>
                    </div>
                    <div class="confidence-bar">
                        <div class="confidence-fill ${confClass}" style="width: ${confPercent}%"></div>
                    </div>
                    <div class="confidence-text">${confPercent}%</div>
                </div>
            `;
        })
        .join("");
}

// ============================================================================
// Video Upload
// ============================================================================

function initVideoUpload(): void {
    const dropzone = document.getElementById("videoDropzone")!;
    const fileInput = document.getElementById("videoFileInput") as HTMLInputElement;
    const analyzeBtn = document.getElementById("analyzeVideoBtn") as HTMLButtonElement;

    dropzone.addEventListener("click", () => fileInput.click());

    dropzone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropzone.classList.add("drag-over");
    });
    dropzone.addEventListener("dragleave", () => dropzone.classList.remove("drag-over"));
    dropzone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropzone.classList.remove("drag-over");
        const files = (e as DragEvent).dataTransfer?.files;
        if (files && files[0]) handleVideoSelect(files[0]);
    });

    fileInput.addEventListener("change", () => {
        if (fileInput.files && fileInput.files[0]) handleVideoSelect(fileInput.files[0]);
    });

    analyzeBtn.addEventListener("click", () => analyzeVideo());
}

function handleVideoSelect(file: File): void {
    selectedVideoFile = file;
    (document.getElementById("analyzeVideoBtn") as HTMLButtonElement).disabled = false;
    const dropzone = document.getElementById("videoDropzone")!;
    dropzone.querySelector("p")!.textContent = `Selected: ${file.name}`;
}

async function analyzeVideo(): Promise<void> {
    if (!selectedVideoFile) return;

    const btn = document.getElementById("analyzeVideoBtn") as HTMLButtonElement;
    btn.disabled = true;
    btn.innerHTML = '<span class="spinner"></span> Processing...';

    const progress = document.getElementById("videoProgress")!;
    progress.style.display = "block";
    const fill = document.getElementById("videoProgressFill")!;
    fill.style.width = "30%";

    try {
        const frameSkip = parseInt((document.getElementById("frameSkip") as HTMLInputElement).value);
        const maxFrames = parseInt((document.getElementById("maxFrames") as HTMLInputElement).value);

        const formData = new FormData();
        formData.append("file", selectedVideoFile);

        const res = await fetch(
            `${API_BASE}/predict/video?frame_skip=${frameSkip}&max_frames=${maxFrames}`,
            { method: "POST", body: formData }
        );

        fill.style.width = "90%";

        if (!res.ok) throw new Error(`HTTP ${res.status}`);

        const data = await res.json();
        fill.style.width = "100%";
        displayVideoResults(data);
    } catch (err) {
        console.error("Video processing error:", err);
        const content = document.getElementById("videoResultsContent")!;
        content.style.display = "block";
        document.getElementById("videoResultsPlaceholder")!.style.display = "none";
        content.innerHTML = '<p style="color: var(--accent-danger);">Error processing video. Ensure backend is running.</p>';
    } finally {
        btn.disabled = false;
        btn.innerHTML = '<span class="btn-icon">▶️</span> Analyze Video';
        setTimeout(() => { progress.style.display = "none"; }, 1000);
    }
}

function displayVideoResults(data: any): void {
    document.getElementById("videoResultsPlaceholder")!.style.display = "none";
    const content = document.getElementById("videoResultsContent")!;
    content.style.display = "block";

    // Stats
    const totalDetections = data.frame_results.reduce(
        (sum: number, f: any) => sum + f.detections.length, 0
    );

    document.getElementById("videoStats")!.innerHTML = `
        <div class="stat-box">
            <div class="stat-value">${data.processed_frames}</div>
            <div class="stat-label">Frames Processed</div>
        </div>
        <div class="stat-box">
            <div class="stat-value">${totalDetections}</div>
            <div class="stat-label">Total Detections</div>
        </div>
        <div class="stat-box">
            <div class="stat-value">${(data.total_processing_time_ms / 1000).toFixed(1)}s</div>
            <div class="stat-label">Processing Time</div>
        </div>
    `;

    // Frame timeline (show frames with detections)
    const framesWithDetections = data.frame_results.filter(
        (f: any) => f.detections.length > 0
    );
    const timeline = document.getElementById("frameTimeline")!;

    if (framesWithDetections.length === 0) {
        timeline.innerHTML = '<p class="text-muted" style="text-align:center;">No traffic signs detected in video</p>';
        return;
    }

    timeline.innerHTML = framesWithDetections
        .slice(0, 50) // Show max 50 entries
        .map((frame: any) => {
            const time = (frame.timestamp_ms / 1000).toFixed(1);
            return `
                <div class="detection-item">
                    <div class="detection-class-id">F${frame.frame_number}</div>
                    <div class="detection-info">
                        <div class="detection-label">${frame.detections.length} sign(s) at ${time}s</div>
                        <div class="detection-sublabel">${frame.detections.map((d: Detection) => d.label.split("--").pop()).join(", ")}</div>
                    </div>
                </div>
            `;
        })
        .join("");
}

// ============================================================================
// Live Camera
// ============================================================================

function initCamera(): void {
    document.getElementById("startCameraBtn")!.addEventListener("click", startCamera);
    document.getElementById("stopCameraBtn")!.addEventListener("click", stopCamera);
}

async function startCamera(): Promise<void> {
    const startBtn = document.getElementById("startCameraBtn") as HTMLButtonElement;
    const stopBtn = document.getElementById("stopCameraBtn") as HTMLButtonElement;
    const overlay = document.getElementById("cameraOverlay")!;
    const video = document.getElementById("cameraVideo") as HTMLVideoElement;

    try {
        cameraStream = await navigator.mediaDevices.getUserMedia({
            video: { width: 640, height: 480, facingMode: "environment" },
        });

        video.srcObject = cameraStream;
        await video.play();

        overlay.style.display = "none";
        startBtn.disabled = true;
        stopBtn.disabled = false;

        // Connect WebSocket
        connectWebSocket();

        // Start frame capture loop
        lastFpsTime = performance.now();
        fpsCounter = 0;
        captureLoop();
    } catch (err) {
        console.error("Camera error:", err);
        overlay.innerHTML = `
            <span class="placeholder-icon">⚠️</span>
            <p>Could not access camera</p>
            <p style="font-size:0.8rem;">${err}</p>
        `;
    }
}

function stopCamera(): void {
    // Stop stream
    if (cameraStream) {
        cameraStream.getTracks().forEach((t) => t.stop());
        cameraStream = null;
    }

    // Close WebSocket
    if (wsConnection) {
        wsConnection.close();
        wsConnection = null;
    }

    // Stop animation
    if (cameraAnimationId) {
        cancelAnimationFrame(cameraAnimationId);
        cameraAnimationId = null;
    }

    const startBtn = document.getElementById("startCameraBtn") as HTMLButtonElement;
    const stopBtn = document.getElementById("stopCameraBtn") as HTMLButtonElement;
    startBtn.disabled = false;
    stopBtn.disabled = true;

    document.getElementById("cameraOverlay")!.style.display = "flex";
    document.getElementById("cameraFps")!.textContent = "— FPS";
}

function connectWebSocket(): void {
    wsConnection = new WebSocket(`${WS_BASE}/ws/stream`);
    wsConnection.binaryType = "arraybuffer";

    wsConnection.onopen = () => {
        console.log("WebSocket connected");
    };

    wsConnection.onmessage = (event) => {
        const data = JSON.parse(event.data);
        handleLiveDetections(data.detections || []);

        // FPS counter
        fpsCounter++;
        const now = performance.now();
        if (now - lastFpsTime >= 1000) {
            document.getElementById("cameraFps")!.textContent = `${fpsCounter} FPS`;
            fpsCounter = 0;
            lastFpsTime = now;
        }
    };

    wsConnection.onerror = (err) => {
        console.error("WebSocket error:", err);
    };

    wsConnection.onclose = () => {
        console.log("WebSocket disconnected");
    };
}

function captureLoop(): void {
    const video = document.getElementById("cameraVideo") as HTMLVideoElement;
    const canvas = document.getElementById("cameraCanvas") as HTMLCanvasElement;
    const ctx = canvas.getContext("2d")!;

    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;

    // Draw video to canvas
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    // Send frame to WebSocket if connected
    if (wsConnection && wsConnection.readyState === WebSocket.OPEN) {
        canvas.toBlob(
            (blob) => {
                if (blob && wsConnection && wsConnection.readyState === WebSocket.OPEN) {
                    blob.arrayBuffer().then((buf) => wsConnection!.send(buf));
                }
            },
            "image/jpeg",
            0.7
        );
    }

    cameraAnimationId = requestAnimationFrame(captureLoop);
}

function handleLiveDetections(detections: Detection[]): void {
    const container = document.getElementById("liveDetections")!;

    if (detections.length === 0) {
        container.innerHTML = '<p class="text-muted" style="font-size:0.85rem;">Scanning for traffic signs...</p>';
        return;
    }

    container.innerHTML = detections
        .map((det) => {
            const confPercent = (det.confidence * 100).toFixed(0);
            const confClass = det.confidence >= 0.8 ? "high" : det.confidence >= 0.5 ? "medium" : "low";
            const label = det.label.replace(/--/g, " › ").replace(/-/g, " ");

            return `
                <div class="detection-item" style="padding:10px;">
                    <div class="detection-class-id">#${det.class_id}</div>
                    <div class="detection-info">
                        <div class="detection-label" style="font-size:0.85rem;">${label}</div>
                    </div>
                    <span class="confidence-badge badge-${confClass}">${confPercent}%</span>
                </div>
            `;
        })
        .join("");

    // Draw bounding boxes on canvas
    drawDetectionsOnCanvas(detections);
}

function drawDetectionsOnCanvas(detections: Detection[]): void {
    const canvas = document.getElementById("cameraCanvas") as HTMLCanvasElement;
    const ctx = canvas.getContext("2d")!;

    for (const det of detections) {
        if (!det.bbox) continue;

        const { xmin, ymin, xmax, ymax } = det.bbox;
        const w = xmax - xmin;
        const h = ymax - ymin;

        // Draw box
        ctx.strokeStyle = "#10b981";
        ctx.lineWidth = 2;
        ctx.strokeRect(xmin, ymin, w, h);

        // Draw label
        const label = `${det.label.split("--").pop()} ${(det.confidence * 100).toFixed(0)}%`;
        ctx.font = "bold 12px Inter, sans-serif";
        const metrics = ctx.measureText(label);
        const textH = 16;

        ctx.fillStyle = "rgba(16, 185, 129, 0.85)";
        ctx.fillRect(xmin, ymin - textH - 2, metrics.width + 8, textH + 4);

        ctx.fillStyle = "#000";
        ctx.fillText(label, xmin + 4, ymin - 4);
    }
}

// ============================================================================
// Inspection Logs
// ============================================================================

function initLogs(): void {
    const applyBtn = document.getElementById("applyFiltersBtn")!;
    const clearBtn = document.getElementById("clearFiltersBtn")!;
    const prevBtn = document.getElementById("prevPageBtn") as HTMLButtonElement;
    const nextBtn = document.getElementById("nextPageBtn") as HTMLButtonElement;

    applyBtn.addEventListener("click", () => {
        currentPage = 1;
        loadHistory();
    });

    clearBtn.addEventListener("click", () => {
        (document.getElementById("filterLabel") as HTMLInputElement).value = "";
        (document.getElementById("filterMinConf") as HTMLInputElement).value = "";
        (document.getElementById("filterMaxConf") as HTMLInputElement).value = "";
        (document.getElementById("filterSource") as HTMLSelectElement).value = "";
        (document.getElementById("filterDateFrom") as HTMLInputElement).value = "";
        (document.getElementById("filterDateTo") as HTMLInputElement).value = "";
        currentPage = 1;
        loadHistory();
    });

    prevBtn.addEventListener("click", () => {
        if (currentPage > 1) {
            currentPage--;
            loadHistory();
        }
    });

    nextBtn.addEventListener("click", () => {
        if (currentPage < totalPages) {
            currentPage++;
            loadHistory();
        }
    });

    // Debounced auto-filter on text input
    let debounceTimer: ReturnType<typeof setTimeout>;
    document.getElementById("filterLabel")!.addEventListener("input", () => {
        clearTimeout(debounceTimer);
        debounceTimer = setTimeout(() => {
            currentPage = 1;
            loadHistory();
        }, 300);
    });
}

async function loadHistory(): Promise<void> {
    const tbody = document.getElementById("logsTableBody")!;
    tbody.innerHTML = '<tr><td colspan="7" class="text-center"><span class="spinner"></span></td></tr>';

    const params = new URLSearchParams();
    params.set("page", currentPage.toString());
    params.set("page_size", "50");

    const label = (document.getElementById("filterLabel") as HTMLInputElement).value;
    const minConf = (document.getElementById("filterMinConf") as HTMLInputElement).value;
    const maxConf = (document.getElementById("filterMaxConf") as HTMLInputElement).value;
    const source = (document.getElementById("filterSource") as HTMLSelectElement).value;
    const dateFrom = (document.getElementById("filterDateFrom") as HTMLInputElement).value;
    const dateTo = (document.getElementById("filterDateTo") as HTMLInputElement).value;

    if (label) params.set("label", label);
    if (minConf) params.set("min_confidence", minConf);
    if (maxConf) params.set("max_confidence", maxConf);
    if (source) params.set("source_type", source);
    if (dateFrom) params.set("date_from", new Date(dateFrom).toISOString());
    if (dateTo) params.set("date_to", new Date(dateTo).toISOString());

    try {
        const res = await fetch(`${API_BASE}/history?${params.toString()}`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);

        const data: HistoryResponse = await res.json();
        totalPages = data.total_pages;

        renderLogsTable(data);
        updatePagination(data);
    } catch (err) {
        console.error("History load error:", err);
        tbody.innerHTML = '<tr><td colspan="7" class="text-center text-muted">Could not load history. Ensure backend is running.</td></tr>';
    }
}

function renderLogsTable(data: HistoryResponse): void {
    const tbody = document.getElementById("logsTableBody")!;

    if (data.items.length === 0) {
        tbody.innerHTML = '<tr><td colspan="7" class="text-center text-muted">No records found</td></tr>';
        return;
    }

    tbody.innerHTML = data.items
        .map((item) => {
            const confPercent = (item.confidence * 100).toFixed(1);
            const confClass = item.confidence >= 0.8 ? "high" : item.confidence >= 0.5 ? "medium" : "low";
            const label = item.predicted_label.replace(/--/g, " › ").replace(/-/g, " ");
            const date = new Date(item.created_at).toLocaleString();
            const latency = item.latency_ms ? `${item.latency_ms.toFixed(1)}ms` : "—";

            return `
                <tr>
                    <td>${item.id}</td>
                    <td title="${item.image_filename}">${truncate(item.image_filename, 20)}</td>
                    <td title="${item.predicted_label}">${truncate(label, 30)}</td>
                    <td><span class="confidence-badge badge-${confClass}">${confPercent}%</span></td>
                    <td>${item.source_type}</td>
                    <td>${latency}</td>
                    <td>${date}</td>
                </tr>
            `;
        })
        .join("");
}

function updatePagination(data: HistoryResponse): void {
    const info = document.getElementById("paginationInfo")!;
    const prevBtn = document.getElementById("prevPageBtn") as HTMLButtonElement;
    const nextBtn = document.getElementById("nextPageBtn") as HTMLButtonElement;

    info.textContent = `Page ${data.page} of ${data.total_pages} (${data.total} total)`;
    prevBtn.disabled = data.page <= 1;
    nextBtn.disabled = data.page >= data.total_pages;
}

// ============================================================================
// Utilities
// ============================================================================

function truncate(text: string, maxLen: number): string {
    return text.length > maxLen ? text.substring(0, maxLen) + "…" : text;
}
