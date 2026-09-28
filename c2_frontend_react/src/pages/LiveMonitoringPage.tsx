import React, { useState } from 'react';
import { Camera, Incident } from '../types';
import { CameraGrid } from '../components/video/CameraGrid';
import { LiveAlertTicker } from '../components/alerts/LiveAlertTicker';
import { IncidentModal } from '../components/alerts/IncidentModal';

interface LiveMonitoringPageProps {
  cameras: Camera[];
  incidents: Incident[];
  onTriageIncident: (id: string, status: 'ACKNOWLEDGED' | 'DISMISSED', notes: string) => Promise<void>;
  onSimulateAlert: () => void;
}

export const LiveMonitoringPage: React.FC<LiveMonitoringPageProps> = ({
  cameras,
  incidents,
  onTriageIncident,
  onSimulateAlert,
}) => {
  const [selectedIncident, setSelectedIncident] = useState<Incident | null>(null);
  const [isModalOpen, setIsModalOpen] = useState(false);

  const handleSelectIncident = (incident: Incident) => {
    setSelectedIncident(incident);
    setIsModalOpen(true);
  };

  return (
    <div className="flex-1 grid grid-cols-12 gap-3 p-3 overflow-hidden h-[calc(100vh-80px)]">
      {/* Left Area: Live Multi-Camera Grid Matrix (9 Cols) */}
      <div className="col-span-9 h-full overflow-hidden">
        <CameraGrid cameras={cameras} />
      </div>

      {/* Right Area: Tactical Alert Stream Ticker (3 Cols) */}
      <div className="col-span-3 h-full overflow-hidden">
        <LiveAlertTicker
          incidents={incidents}
          onSelectIncident={handleSelectIncident}
          onSimulateAlert={onSimulateAlert}
        />
      </div>

      {/* Forensic Inspection Modal */}
      <IncidentModal
        incident={selectedIncident}
        isOpen={isModalOpen}
        onClose={() => setIsModalOpen(false)}
        onTriage={onTriageIncident}
      />
    </div>
  );
};
