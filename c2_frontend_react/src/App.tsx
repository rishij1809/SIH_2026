import { useState, useCallback, useEffect } from 'react';
import { QueryClient, QueryClientProvider, useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { Camera, Incident, Telemetry } from './types';
import { fetchCameras } from './api/cameras';
import { fetchIncidents, triageIncident, triggerSimulatedAlert } from './api/incidents';
import { fetchTelemetry } from './api/telemetry';
import { useWebSocket } from './hooks/useWebSocket';
import { useAudioAlert } from './hooks/useAudioAlert';
import { TopNav } from './components/layout/TopNav';
import { TelemetryBar } from './components/layout/TelemetryBar';
import { Sidebar } from './components/layout/Sidebar';
import { LiveMonitoringPage } from './pages/LiveMonitoringPage';
import { IncidentLogPage } from './pages/IncidentLogPage';
import { AnalyticsPage } from './pages/AnalyticsPage';

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      refetchOnWindowFocus: false,
      staleTime: 5000,
    },
  },
});

function DashboardContent() {
  const [activeTab, setActiveTab] = useState<'live' | 'incidents' | 'analytics'>('live');
  const [liveIncidents, setLiveIncidents] = useState<Incident[]>([]);
  const { muted, toggleMute, playAlertSound } = useAudioAlert();

  const queryClientRef = useQueryClient();

  // Initial Queries
  const { data: cameras = [] } = useQuery<Camera[]>({
    queryKey: ['cameras'],
    queryFn: fetchCameras,
  });

  const { data: initialIncidents = [] } = useQuery<Incident[]>({
    queryKey: ['incidents'],
    queryFn: () => fetchIncidents(1, 50),
  });

  const { data: initialTelemetry = null } = useQuery<Telemetry>({
    queryKey: ['telemetry'],
    queryFn: fetchTelemetry,
  });

  // Sync initial incidents into state once fetched
  useEffect(() => {
    if (initialIncidents.length > 0 && liveIncidents.length === 0) {
      setLiveIncidents(initialIncidents);
    }
  }, [initialIncidents, liveIncidents.length]);

  // Handle incoming live alert from WebSocket
  const handleAlertMessage = useCallback((newAlert: Incident) => {
    console.log('[LIVE WS ALERT RECEIVED]:', newAlert);
    playAlertSound();
    setLiveIncidents((prev) => [newAlert, ...prev.filter((i) => i.id !== newAlert.id)]);
  }, [playAlertSound]);

  // Live WebSocket Connections
  const { status: wsStatus } = useWebSocket<Incident>('/ws/alerts', handleAlertMessage);
  const { lastMessage: liveTelemetry } = useWebSocket<Telemetry>('/ws/telemetry');

  const currentTelemetry = liveTelemetry || initialTelemetry;

  // Triage Mutation
  const triageMutation = useMutation({
    mutationFn: ({ id, status, notes }: { id: string; status: 'ACKNOWLEDGED' | 'DISMISSED'; notes: string }) =>
      triageIncident(id, { status, operator_notes: notes }),
    onSuccess: (updatedIncident) => {
      setLiveIncidents((prev) =>
        prev.map((inc) => (inc.id === updatedIncident.id ? updatedIncident : inc))
      );
      queryClientRef.invalidateQueries({ queryKey: ['incidents'] });
    },
  });

  const handleTriageIncident = async (id: string, status: 'ACKNOWLEDGED' | 'DISMISSED', notes: string) => {
    await triageMutation.mutateAsync({ id, status, notes });
  };

  const handleSimulateAlert = async () => {
    try {
      const simIncident = await triggerSimulatedAlert('CAM-01', 'distress_signal');
      handleAlertMessage(simIncident);
    } catch (e) {
      // Offline simulation fallback
      const simIncident: Incident = {
        id: `INC-${Math.floor(Math.random() * 9000 + 1000)}`,
        camera_id: 'CAM-01',
        camera_name: 'BOP-ALPHA-GATE-01',
        location_tag: 'Sector 4 North Gate',
        event_type: 'distress_signal',
        confidence_score: 0.958,
        timestamp: new Date().toISOString(),
        clip_url: '/media/clips/demo_clip.mp4',
        keyframe_url: '/media/keyframes/demo_keyframe.jpg',
        status: 'UNACKNOWLEDGED',
        track_id: 99,
        spatial_coordinates: { x: 250, y: 180, w: 130, h: 300 }
      };
      handleAlertMessage(simIncident);
    }
  };

  const unacknowledgedCount = liveIncidents.filter((i) => i.status === 'UNACKNOWLEDGED').length;
  const activeCameras = cameras.filter((c) => c.status === 'ONLINE').length;

  return (
    <div className="flex flex-col h-screen overflow-hidden bg-[#090A0F]">
      {/* Top Tactical Nav */}
      <TopNav
        wsStatus={wsStatus}
        muted={muted}
        onToggleMute={toggleMute}
        activeCameraCount={activeCameras}
        totalCameraCount={cameras.length}
      />

      {/* Hardware Telemetry Bar */}
      <TelemetryBar telemetry={currentTelemetry} />

      {/* Main Content Area */}
      <div className="flex flex-1 overflow-hidden">
        {/* Navigation Sidebar */}
        <Sidebar
          activeTab={activeTab}
          onSelectTab={setActiveTab}
          unacknowledgedCount={unacknowledgedCount}
        />

        {/* Dynamic Page Views */}
        {activeTab === 'live' && (
          <LiveMonitoringPage
            cameras={cameras}
            incidents={liveIncidents}
            onTriageIncident={handleTriageIncident}
            onSimulateAlert={handleSimulateAlert}
          />
        )}

        {activeTab === 'incidents' && (
          <IncidentLogPage
            incidents={liveIncidents}
            onTriageIncident={handleTriageIncident}
          />
        )}

        {activeTab === 'analytics' && (
          <AnalyticsPage
            telemetry={currentTelemetry}
            incidents={liveIncidents}
          />
        )}
      </div>
    </div>
  );
}

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <DashboardContent />
    </QueryClientProvider>
  );
}
