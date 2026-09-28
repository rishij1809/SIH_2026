export type CameraStatus = 'ONLINE' | 'DEGRADED' | 'OFFLINE';

export interface Camera {
  id: string;
  name: string;
  location: string;
  rtsp_stream_url: string;
  status: CameraStatus;
  fps: number;
  active_alerts_count: number;
}

export type EventType = 'distress_signal' | 'aggression' | 'fall' | 'virtual_fence';

export type IncidentStatus = 'UNACKNOWLEDGED' | 'ACKNOWLEDGED' | 'DISMISSED';

export interface SpatialCoordinates {
  x: number;
  y: number;
  w: number;
  h: number;
}

export interface Incident {
  id: string;
  camera_id: string;
  camera_name: string;
  location_tag: string;
  event_type: EventType;
  confidence_score: number;
  timestamp: string;
  clip_url: string | null;
  keyframe_url: string | null;
  status: IncidentStatus;
  operator_notes?: string;
  track_id?: number;
  spatial_coordinates?: SpatialCoordinates;
}

export interface TriagePayload {
  status: 'ACKNOWLEDGED' | 'DISMISSED';
  operator_notes: string;
}

export interface Telemetry {
  edge_node_id: string;
  gpu_temp_c: number;
  gpu_utilization_pct: number;
  vram_used_mb: number;
  vram_total_mb: number;
  pipeline_latency_ms: number;
  disk_usage_pct: number;
  connectivity: 'HEALTHY' | 'DEGRADED';
}
