# Project Vigilance / IBVAP - Intelligent Border & Safety Video Analytics Platform

Project Vigilance / IBVAP is an end-to-end, production-grade AI safety surveillance platform engineered for low-resource legacy CCTV feeds on edge hardware (NVIDIA Jetson / x86 edge PCs).

It features:
1. **Tier 1 (Spatial Localization):** YOLOv8 fine-tuned for person detection.
2. **Tier 2 (Spatio-Temporal Behavior AI):** Custom 1D Spatial Conv + Bi-GRU + Temporal Self-Attention Model for real-time distress gesture recognition (`Normal`, `Universal Help Signal`, `Aggression/Struggle`, `Fall`).
3. **Stage 3 Pipeline:** Real-time RTSP ingestion, ByteTrack multi-object tracking, sliding 30-frame keypoint FIFO deque.
4. **Stage 4 Verification & Alerting:** $M$-of-$N$ temporal persistence gating, 10s pre/post MP4 recorder, Webhook C2, Twilio SMS/Voice, Edge GPIO Relay, local SQLite offline queue.
5. **Stage 5 Edge Hardening & C2 Console:** Edge Watchdog supervisor, FastAPI backend, modern React + Vite + TypeScript + Tailwind CSS C4ISR Tactical Web Dashboard.

---

## 🏗️ System Architecture & Directory Layout

```text
SIH 2026/
├── configs/
│   ├── config.yaml               # Master pipeline & model hyperparameters
│   ├── data.yaml                 # YOLOv8 dataset splits
│   ├── pipeline_config.yaml      # Stage 3 RTSP streams & tracker config
│   └── alert_config.yaml         # Stage 4 Gating, Thresholds & Dispatcher config
├── c2_frontend_react/            # React 18 + TypeScript + Tailwind Tactical Dashboard
│   ├── index.html
│   ├── vite.config.ts
│   ├── tailwind.config.js
│   ├── .env.example
│   └── src/
│       ├── api/                  # Axios REST client (cameras, incidents, telemetry)
│       ├── hooks/                # useWebSocket.ts, useAudioAlert.ts
│       ├── types/                # TypeScript interfaces
│       ├── components/           # TopNav, TelemetryBar, Sidebar, CameraGrid, LiveAlertTicker, IncidentModal
│       └── pages/                # LiveMonitoringPage, IncidentLogPage, AnalyticsPage
├── docker/
│   ├── Dockerfile.jetson         # NVIDIA L4T/JetPack container image
│   ├── Dockerfile.x86_gpu        # CUDA / TensorRT x86 container image
│   └── nginx.conf                # Nginx reverse proxy configuration
├── data/
│   ├── raw_videos/               # CCTV video streams / recordings
│   ├── extracted_frames/         # Sampled keyframes (SSIM duplicate filtered)
│   ├── yolo_dataset/             # Structured YOLOv8 split (train/val/test)
│   ├── sequence_dataset/         # Skeletal keypoint dataset (N, 30, 33, 3)
│   ├── offline_cache.db          # Local SQLite persistent queue for offline edge resilience
│   └── incidents/                # Encoded 10s MP4 clips & JPEG keyframes
├── models/                       # Checkpoint and export directory
│   ├── yolo_person.pt
│   ├── yolo_person.onnx
│   ├── distress_model.pt
│   ├── distress_traced.pt
│   └── distress_model.onnx
├── src/
│   ├── ingestion/                # Frame Extractor & SSIM Duplicate Filter
│   ├── annotation/               # YOLO Preprocessor & Keypoint Sequence Builder
│   ├── augmentation/             # Albumentations CCTV Noise & Keypoint Sequence Augmentors
│   ├── models/                   # PyTorch Spatio-Temporal Distress Model
│   ├── training/                 # Mixed Precision AMP Trainer & Focal Loss
│   ├── export/                   # ONNX & TorchScript Exporter
│   ├── validation/               # Integrity Checker & Latency Benchmark Engine
│   ├── pipeline/                 # RTSP Ingestor, Tracklet Manager, Two-Tier Orchestrator, Visual Telemetry
│   └── alerts/                   # Temporal Persistence Verifier, Clip Recorder, Dispatcher, Offline Cache
├── docker-compose.yml            # Multi-container edge orchestration
├── edge_watchdog.py              # Edge Hardware Supervisor & Disk Auto-Purge
├── c2_backend.py                 # FastAPI Tactical C2 REST & WebSocket Server
├── deploy_edge.sh                # Zero-Touch Edge Provisioning Script
└── test_alert_pipeline.py        # End-to-End Stage 4 Alert Pipeline Test
```

---

## ⚡ Setup & Execution Commands

### 1. Launch FastAPI Tactical C2 Backend Server
```bash
python c2_backend.py
```

### 2. Launch React Tactical C2 Command Console
```bash
cd c2_frontend_react
npm install
npm run dev
```
Access dashboard at: **http://localhost:5173**

---

## 📡 REST & WebSocket Contracts

### REST Endpoints
- `GET /api/v1/cameras` $\rightarrow$ List of active camera streams, health, FPS, location.
- `GET /api/v1/incidents?page=1&limit=20&status=UNACKNOWLEDGED` $\rightarrow$ Paginated incident logs.
- `PATCH /api/v1/incidents/:id/triage` $\rightarrow$ Triage incident status (`ACKNOWLEDGED` or `DISMISSED`) with operator notes.
- `GET /api/v1/telemetry` $\rightarrow$ GPU temp, load, VRAM, pipeline latency ms, disk space.

### WebSockets
- `ws://localhost:8000/ws/alerts` $\rightarrow$ Real-time verified distress incident payload stream.
- `ws://localhost:8000/ws/telemetry` $\rightarrow$ Live hardware vitals stream every 2s.
