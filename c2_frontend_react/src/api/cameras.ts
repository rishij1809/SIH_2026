import { apiClient } from './client';
import { Camera } from '../types';

export const fetchCameras = async (): Promise<Camera[]> => {
  try {
    const response = await apiClient.get<Camera[]>('/api/v1/cameras');
    return response.data;
  } catch (error) {
    console.warn('Backend unavailable, using fallback camera grid.');
    return [
      { id: 'CAM-01', name: 'BOP-ALPHA-GATE-01', location: 'Sector 4 North Gate', rtsp_stream_url: '/media/demo_clip.mp4', status: 'ONLINE', fps: 12.0, active_alerts_count: 1 },
      { id: 'CAM-02', name: 'EAST-PERIMETER-WALL', location: 'East Boundary Fence 12', rtsp_stream_url: '/media/demo_clip.mp4', status: 'ONLINE', fps: 12.0, active_alerts_count: 0 },
      { id: 'CAM-03', name: 'SOUTH-CORRIDOR-CHECK', location: 'South Patrol Route A', rtsp_stream_url: '/media/demo_clip.mp4', status: 'ONLINE', fps: 12.0, active_alerts_count: 0 },
      { id: 'CAM-04', name: 'VEHICLE-INSPECTION-BAY', location: 'Outpost Logistics Gate', rtsp_stream_url: '/media/demo_clip.mp4', status: 'ONLINE', fps: 12.0, active_alerts_count: 0 },
      { id: 'CAM-05', name: 'TOWER-NORTH-OPTICS', location: 'Observation Tower 1', rtsp_stream_url: '/media/demo_clip.mp4', status: 'DEGRADED', fps: 8.5, active_alerts_count: 0 },
      { id: 'CAM-06', name: 'ARMORY-PERIMETER-S', location: 'Tactical Depot South', rtsp_stream_url: '/media/demo_clip.mp4', status: 'ONLINE', fps: 12.0, active_alerts_count: 0 },
      { id: 'CAM-07', name: 'HELI-PAD-WEST-APPROACH', location: 'Airfield West Boundary', rtsp_stream_url: '/media/demo_clip.mp4', status: 'ONLINE', fps: 12.0, active_alerts_count: 0 },
      { id: 'CAM-08', name: 'COMMAND-BUNKER-MAIN', location: 'HQ Internal Corridor', rtsp_stream_url: '/media/demo_clip.mp4', status: 'ONLINE', fps: 12.0, active_alerts_count: 0 }
    ];
  }
};
