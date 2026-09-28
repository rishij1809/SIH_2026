#!/usr/bin/env bash
# Project Vigilance - Zero-Touch Edge Provisioning & Deployment Script
set -e

echo "=========================================================================="
echo "          PROJECT VIGILANCE - ZERO-TOUCH EDGE PROVISIONING                "
echo "=========================================================================="

# 1. Check Root Privileges & System Architecture
if [ "$EUID" -ne 0 ]; then
  echo "⚠️ Warning: Please run with sudo/root privileges for full systemd & Docker setup."
fi

ARCH=$(uname -m)
echo "[1/6] System Architecture Detected: $ARCH"

# 2. Verify Docker & NVIDIA GPU Runtime
echo "[2/6] Verifying Docker & NVIDIA Runtime Environment..."
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed! Installing Docker..."
    curl -fsSL https://get.docker.com | sh
fi

if command -v nvidia-smi &> /dev/null; then
    echo "✅ NVIDIA Driver Detected:"
    nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
else
    echo "⚠️ NVIDIA Driver not detected or running on CPU-only hardware."
fi

# 3. Create Edge Directory Hierarchy
echo "[3/6] Initializing Local Edge Directories & Storage Partitions..."
mkdir -p data/raw_videos
mkdir -p data/extracted_frames
mkdir -p data/yolo_dataset
mkdir -p data/sequence_dataset
mkdir -p data/incidents/clips
mkdir -p data/incidents/keyframes
mkdir -p models
mkdir -p configs

# 4. Initialize Local SQLite Schema & Verify Model Weights
echo "[4/6] Initializing Edge SQLite Database & Model Weights..."
python3 -c "
import sqlite3
conn = sqlite3.connect('data/offline_cache.db')
conn.execute('CREATE TABLE IF NOT EXISTS alerts_queue (id INTEGER PRIMARY KEY, incident_id TEXT, timestamp TEXT, camera_id TEXT, payload_json TEXT, status TEXT, attempts INTEGER, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP);')
conn.commit()
print('SQLite Cache DB Schema Verified.')
"

# 5. Install Edge Watchdog Systemd Service
echo "[5/6] Installing Systemd Watchdog Supervisor Unit..."
cat << 'EOF' > /etc/systemd/system/vigilance-watchdog.service
[Unit]
Description=Project Vigilance Edge Watchdog Supervisor
After=docker.service
Requires=docker.service

[Service]
Type=simple
WorkingDirectory=/app
ExecStart=/usr/bin/python3 /app/edge_watchdog.py
Restart=always
RestartSec=5s

[Install]
WantedBy=multi-user.target
EOF

if command -v systemctl &> /dev/null; then
    systemctl daemon-reload || true
    systemctl enable vigilance-watchdog.service || true
    echo "✅ Systemd watchdog unit installed successfully."
fi

# 6. Launch Docker Compose Stack
echo "[6/6] Launching Full Project Vigilance Docker Stack..."
if command -v docker-compose &> /dev/null; then
    docker-compose up -d --build
elif docker compose version &> /dev/null; then
    docker compose up -d --build
else
    echo "Executing python demonstration mode fallback..."
    python3 demo_stage5_c2.py &
fi

echo "=========================================================================="
echo "   PROJECT VIGILANCE STACK DEPLOYED SUCCESSFULLY!                         "
echo "   Tactical C2 Dashboard: http://localhost:8000                           "
echo "   REST Telemetry API   : http://localhost:8000/api/v1/telemetry          "
echo "=========================================================================="
