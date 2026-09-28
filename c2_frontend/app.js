// App.js - Tactical C2 Dashboard Logic
const API_BASE = window.location.origin;
const WS_PROTOCOL = window.location.protocol === "https:" ? "wss:" : "ws:";
const WS_URL = `${WS_PROTOCOL}//${window.location.host}/ws/alerts`;

let activeIncidents = [];
let currentSelectedIncident = null;

// Initialize WebSockets
function initWebSockets() {
    console.log("Connecting to Tactical Alert WebSocket:", WS_URL);
    const ws = new WebSocket(WS_URL);

    ws.onopen = () => {
        console.log("Tactical Alert WebSocket Connected.");
    };

    ws.onmessage = (event) => {
        const payload = JSON.parse(event.data);
        console.log("Received Alert Payload:", payload);
        handleNewAlert(payload);
    };

    ws.onclose = () => {
        console.warn("WebSocket Disconnected. Reconnecting in 3s...");
        setTimeout(initWebSockets, 3000);
    };
}

// Render dynamic alert card into ticker
function handleNewAlert(alert) {
    activeIncidents.unshift(alert);

    // Play audible ping alert
    playAlertSound();

    const feedContainer = document.getElementById("alerts-feed");
    const card = document.createElement("div");
    card.className = "alert-card";
    card.onclick = () => openInspectionModal(alert);

    const timeStr = new Date(alert.timestamp).toLocaleTimeString();

    card.innerHTML = `
        <div class="card-header">
            <span class="event-tag">🚨 ${alert.event_type.toUpperCase()}</span>
            <span class="time-tag">${timeStr}</span>
        </div>
        <div class="card-body">
            <div><strong>Location:</strong> ${alert.location_tag}</div>
            <div><strong>Confidence:</strong> ${(alert.confidence_score * 100).toFixed(0)}% | Track #${alert.track_id}</div>
        </div>
    `;

    feedContainer.prepend(card);
}

function playAlertSound() {
    try {
        const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        const osc = audioCtx.createOscillator();
        const gain = audioCtx.createGain();
        osc.type = "sine";
        osc.frequency.setValueAtTime(880, audioCtx.currentTime); // A5 note
        gain.gain.setValueAtTime(0.1, audioCtx.currentTime);
        osc.connect(gain);
        gain.connect(audioCtx.destination);
        osc.start();
        osc.stop(audioCtx.currentTime + 0.3);
    } catch (e) {
        console.log("Audio play blocked by browser policy.");
    }
}

// Modal Inspection Drawer
function openInspectionModal(alert) {
    currentSelectedIncident = alert;

    document.getElementById("meta-id").innerText = alert.incident_id;
    document.getElementById("meta-time").innerText = alert.timestamp;
    document.getElementById("meta-cam").innerText = alert.camera_id;
    document.getElementById("meta-event").innerText = alert.event_type.toUpperCase();
    document.getElementById("meta-conf").innerText = `${(alert.confidence_score * 100).toFixed(1)}%`;
    document.getElementById("meta-track").innerText = `#${alert.track_id}`;

    const videoElem = document.getElementById("incident-video");
    const keyframeElem = document.getElementById("incident-keyframe");

    if (alert.clip_path) {
        videoElem.src = alert.clip_path;
    }
    if (alert.keyframe_path) {
        keyframeElem.src = alert.keyframe_path;
    }

    document.getElementById("incident-modal").classList.remove("hidden");
}

document.getElementById("btn-close-modal").onclick = () => {
    document.getElementById("incident-modal").classList.add("hidden");
};

// Simulate Alert Button
document.getElementById("btn-sim-alert").onclick = async () => {
    const simPayload = {
        incident_id: "sim-" + Math.floor(Math.random() * 10000),
        timestamp: new Date().toISOString(),
        camera_id: "CAM_01",
        location_tag: "Alpha Gate Outpost",
        event_type: "distress_signal",
        confidence_score: 0.95,
        track_id: 99,
        spatial_coordinates: { x: 100, y: 150, w: 120, h: 300 }
    };

    try {
        await fetch(`${API_BASE}/api/v1/alerts/simulate`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(simPayload)
        });
    } catch (e) {
        // Local simulation fallback
        handleNewAlert(simPayload);
    }
};

// Operator Acknowledge Actions
document.getElementById("btn-ack").onclick = () => submitOperatorAction("ACKNOWLEDGED");
document.getElementById("btn-false").onclick = () => submitOperatorAction("FALSE_ALARM");
document.getElementById("btn-escalate").onclick = () => submitOperatorAction("ESCALATED");

async function submitOperatorAction(status) {
    if (!currentSelectedIncident) return;
    const notes = document.getElementById("operator-notes").value;

    try {
        await fetch(`${API_BASE}/api/v1/incidents/${currentSelectedIncident.incident_id}/acknowledge`, {
            method: "PATCH",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ status, operator_notes: notes })
        });
        alert(`Incident status updated to: ${status}`);
        document.getElementById("incident-modal").classList.add("hidden");
    } catch (e) {
        console.error("Action submit failed:", e);
        document.getElementById("incident-modal").classList.add("hidden");
    }
}

// Start
initWebSockets();
