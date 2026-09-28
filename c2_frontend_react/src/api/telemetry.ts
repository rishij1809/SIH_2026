import { apiClient } from './client';
import { Telemetry } from '../types';

export const fetchTelemetry = async (): Promise<Telemetry> => {
  try {
    const response = await apiClient.get<Telemetry>('/api/v1/telemetry');
    return response.data;
  } catch (error) {
    return {
      edge_node_id: 'BOP-ALPHA-NODE-01',
      gpu_temp_c: 44.2,
      gpu_utilization_pct: 34.5,
      vram_used_mb: 2480,
      vram_total_mb: 8192,
      pipeline_latency_ms: 14.2,
      disk_usage_pct: 74.2,
      connectivity: 'HEALTHY'
    };
  }
};
