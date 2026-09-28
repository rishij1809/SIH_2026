import React from 'react';
import { AlertOctagon, Flame, ShieldAlert, CheckCircle2, XCircle, Play } from 'lucide-react';
import { Incident } from '../../types';
import { Badge } from '../common/Badge';

interface LiveAlertTickerProps {
  incidents: Incident[];
  onSelectIncident: (incident: Incident) => void;
  onSimulateAlert: () => void;
}

export const LiveAlertTicker: React.FC<LiveAlertTickerProps> = ({
  incidents,
  onSelectIncident,
  onSimulateAlert,
}) => {
  const getEventBadge = (eventType: string) => {
    switch (eventType) {
      case 'distress_signal':
        return <Badge variant="warning">HELP SIGNAL</Badge>;
      case 'aggression':
        return <Badge variant="danger">AGGRESSION</Badge>;
      case 'fall':
        return <Badge variant="danger">FALL DETECTED</Badge>;
      default:
        return <Badge variant="primary">VIRTUAL FENCE</Badge>;
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'ACKNOWLEDGED':
        return <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />;
      case 'DISMISSED':
        return <XCircle className="w-3.5 h-3.5 text-zinc-500" />;
      default:
        return <ShieldAlert className="w-3.5 h-3.5 text-rose-500 animate-pulse" />;
    }
  };

  return (
    <div className="flex flex-col h-full bg-[#0D1017] border-l border-zinc-800/80 font-mono select-none">
      {/* Drawer Header */}
      <div className="flex items-center justify-between px-3 py-2 bg-[#121622] border-b border-zinc-800">
        <div className="flex items-center gap-2">
          <AlertOctagon className="w-4 h-4 text-rose-500" />
          <h2 className="text-xs font-bold text-zinc-200 tracking-wider">TACTICAL ALERT STREAM</h2>
        </div>
        <button
          onClick={onSimulateAlert}
          className="flex items-center gap-1 bg-rose-950/80 hover:bg-rose-900 border border-rose-800 text-rose-300 text-[10px] font-bold px-2 py-0.5 rounded-xs transition-colors"
          title="Simulate high-confidence distress event"
        >
          <Flame className="w-3 h-3 text-rose-400" /> + SIMULATE
        </button>
      </div>

      {/* Incident List Ticker */}
      <div className="flex-1 overflow-y-auto p-2 flex flex-col gap-2">
        {incidents.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-48 text-zinc-600 text-xs text-center p-4">
            <CheckCircle2 className="w-8 h-8 text-zinc-700 mb-2" />
            <span>NO UNACKNOWLEDGED INCIDENTS</span>
            <span className="text-[10px] text-zinc-700">Surveillance sector nominal.</span>
          </div>
        ) : (
          incidents.map((inc) => {
            const isUnack = inc.status === 'UNACKNOWLEDGED';
            const timeStr = new Date(inc.timestamp).toLocaleTimeString();

            return (
              <div
                key={inc.id}
                onClick={() => onSelectIncident(inc)}
                className={`group relative p-3 border rounded-xs transition-all duration-150 cursor-pointer ${
                  isUnack
                    ? 'bg-rose-950/20 border-rose-800/80 hover:bg-rose-950/40 hover:border-rose-600 shadow-[0_0_10px_rgba(239,68,68,0.15)]'
                    : 'bg-[#141924] border-zinc-800/80 hover:border-zinc-700 opacity-75'
                }`}
              >
                {/* Header Line */}
                <div className="flex items-center justify-between mb-1.5">
                  <div className="flex items-center gap-1.5">
                    {getStatusIcon(inc.status)}
                    <span className="font-bold text-xs text-zinc-200">{inc.id}</span>
                  </div>
                  <span className="text-[10px] text-zinc-500">{timeStr}</span>
                </div>

                {/* Event Type & Confidence */}
                <div className="flex items-center justify-between mb-2">
                  {getEventBadge(inc.event_type)}
                  <span className="text-xs font-bold text-sky-400">
                    CONF: {(inc.confidence_score * 100).toFixed(1)}%
                  </span>
                </div>

                {/* Location & Track Info */}
                <div className="text-[11px] text-zinc-400 space-y-0.5">
                  <div className="truncate">
                    <span className="text-zinc-600">LOC:</span> {inc.location_tag} ({inc.camera_name})
                  </div>
                  {inc.track_id && (
                    <div className="text-zinc-500 text-[10px]">
                      TRACK ID: <span className="text-zinc-300">#{inc.track_id}</span>
                    </div>
                  )}
                </div>

                {/* Inspect Button Hint */}
                <div className="mt-2 pt-2 border-t border-zinc-800/60 flex items-center justify-between text-[10px] text-zinc-500 group-hover:text-sky-400 transition-colors">
                  <span>INSPECT FORENSIC CLIP</span>
                  <Play className="w-3 h-3" />
                </div>
              </div>
            );
          })
        )}
      </div>
    </div>
  );
};
