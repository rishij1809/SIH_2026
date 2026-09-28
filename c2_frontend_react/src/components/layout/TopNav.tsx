import React, { useState, useEffect } from 'react';
import { Shield, Volume2, VolumeX, Radio, Activity } from 'lucide-react';
import { StatusDot } from '../common/StatusDot';
import { WebSocketStatus } from '../../hooks/useWebSocket';

interface TopNavProps {
  wsStatus: WebSocketStatus;
  muted: boolean;
  onToggleMute: () => void;
  activeCameraCount: number;
  totalCameraCount: number;
}

export const TopNav: React.FC<TopNavProps> = ({
  wsStatus,
  muted,
  onToggleMute,
  activeCameraCount,
  totalCameraCount,
}) => {
  const [utcTime, setUtcTime] = useState('');

  useEffect(() => {
    const timer = setInterval(() => {
      const now = new Date();
      setUtcTime(now.toISOString().replace('T', ' ').substring(0, 23) + ' UTC');
    }, 50);
    return () => clearInterval(timer);
  }, []);

  return (
    <header className="h-12 bg-[#0C0E14] border-b border-zinc-800/80 px-4 flex items-center justify-between font-mono text-xs select-none">
      {/* Brand & Station Info */}
      <div className="flex items-center gap-3">
        <div className="flex items-center gap-2 text-sky-400 font-bold tracking-wider text-sm">
          <Shield className="w-5 h-5 text-sky-400" />
          <span>PROJECT VIGILANCE</span>
          <span className="text-[10px] text-zinc-500 font-normal">/ IBVAP v3.0</span>
        </div>

        <div className="h-4 w-px bg-zinc-800 mx-1" />

        <div className="flex items-center gap-2 text-[11px] text-zinc-400">
          <Radio className="w-3.5 h-3.5 text-emerald-400" />
          <span className="font-semibold text-zinc-200">SECTOR-4 / BOP-ALPHA-01</span>
        </div>
      </div>

      {/* Clock & Status Metrics */}
      <div className="flex items-center gap-4 text-[11px]">
        {/* Live Millisecond UTC Clock */}
        <div className="bg-zinc-900/80 border border-zinc-800/80 px-2.5 py-1 text-zinc-300 font-mono tracking-wider">
          {utcTime || '2026-09-27 17:00:00.000 UTC'}
        </div>

        {/* Camera Node Health */}
        <div className="flex items-center gap-1.5 bg-zinc-900/80 border border-zinc-800/80 px-2.5 py-1">
          <Activity className="w-3.5 h-3.5 text-sky-400" />
          <span className="text-zinc-400">NODES:</span>
          <span className="text-emerald-400 font-bold">
            {activeCameraCount}/{totalCameraCount} ONLINE
          </span>
        </div>

        {/* Audio Alert Toggle */}
        <button
          onClick={onToggleMute}
          className={`flex items-center gap-1.5 px-2.5 py-1 border transition-colors ${
            muted
              ? 'bg-amber-950/40 border-amber-800/60 text-amber-400'
              : 'bg-zinc-900 border-zinc-800 text-zinc-300 hover:text-white'
          }`}
          title={muted ? 'Unmute Tactical Audio Alerts' : 'Mute Tactical Audio Alerts'}
        >
          {muted ? <VolumeX className="w-3.5 h-3.5" /> : <Volume2 className="w-3.5 h-3.5 text-emerald-400" />}
          <span>{muted ? 'AUDIO MUTED' : 'AUDIO ACTIVE'}</span>
        </button>

        {/* WebSocket Telemetry Status */}
        <div className="flex items-center gap-2 bg-zinc-900/80 border border-zinc-800/80 px-2.5 py-1">
          <StatusDot status={wsStatus} />
          <span className="text-zinc-300 font-semibold tracking-wider">
            {wsStatus === 'CONNECTED'
              ? 'C2 LINK HEALTHY'
              : wsStatus === 'RECONNECTING'
              ? 'RECONNECTING LINK...'
              : 'C2 LINK DOWN'}
          </span>
        </div>
      </div>
    </header>
  );
};
