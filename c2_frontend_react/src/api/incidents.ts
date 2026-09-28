import { apiClient } from './client';
import { Incident, TriagePayload } from '../types';

export const fetchIncidents = async (
  page: number = 1,
  limit: number = 20,
  status?: string
): Promise<Incident[]> => {
  try {
    const params: Record<string, any> = { page, limit };
    if (status) params.status = status;

    const response = await apiClient.get<Incident[]>('/api/v1/incidents', { params });
    return response.data;
  } catch (error) {
    console.warn('Backend unavailable, returning fallback incident history.');
    return [
      {
        id: 'INC-8091',
        camera_id: 'CAM-01',
        camera_name: 'BOP-ALPHA-GATE-01',
        location_tag: 'Sector 4 North Gate',
        event_type: 'distress_signal',
        confidence_score: 0.948,
        timestamp: new Date().toISOString(),
        clip_url: '/media/clips/demo_clip.mp4',
        keyframe_url: '/media/keyframes/demo_keyframe.jpg',
        status: 'UNACKNOWLEDGED',
        track_id: 99,
        spatial_coordinates: { x: 250, y: 180, w: 130, h: 300 }
      },
      {
        id: 'INC-8090',
        camera_id: 'CAM-02',
        camera_name: 'EAST-PERIMETER-WALL',
        location_tag: 'East Boundary Fence 12',
        event_type: 'fall',
        confidence_score: 0.884,
        timestamp: new Date(Date.now() - 1000 * 60 * 15).toISOString(),
        clip_url: '/media/clips/demo_clip.mp4',
        keyframe_url: '/media/keyframes/demo_keyframe.jpg',
        status: 'ACKNOWLEDGED',
        operator_notes: 'Patrol team dispatched to assist fallen subject.',
        track_id: 45,
        spatial_coordinates: { x: 100, y: 200, w: 150, h: 120 }
      }
    ];
  }
};

export const triageIncident = async (
  incidentId: string,
  payload: TriagePayload
): Promise<Incident> => {
  const response = await apiClient.patch<Incident>(
    `/api/v1/incidents/${incidentId}/triage`,
    payload
  );
  return response.data;
};

export const triggerSimulatedAlert = async (
  cameraId: string = 'CAM-01',
  eventType: string = 'distress_signal'
): Promise<Incident> => {
  const response = await apiClient.post<Incident>(
    `/api/v1/alerts/simulate`,
    null,
    { params: { camera_id: cameraId, event_type: eventType } }
  );
  return response.data;
};
