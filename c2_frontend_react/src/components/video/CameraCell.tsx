import React from 'react';
import { Maximize2, ShieldAlert } from 'lucide-react';
import { Camera } from '../../types';
import { VideoPlayer } from './VideoPlayer';
import { StatusDot } from '../common/StatusDot';

interface CameraCellProps {
  camera: Camera;
  onMaximize?: (camera: Camera) => void;
  isMaximized?: boolean;
}

export const CameraCell: React.FC<CameraCellProps> = ({
  camera,
  onMaximize,
}) => {
  const hasAlert = camera.active_alerts_count > 0;

  return (
    <div
      className={`relative group bg-[#11151F] border transition-all duration-200 rounded-xs overflow-hidden flex flex-col ${
        hasAlert
          ? 'border-rose-600 shadow-[0_0_15px_rgba(239,68,68,0.3)] animate-pulse'
          : 'border-zinc-800 hover:border-zinc-700'
      }`}
    >
      {/* Tile Header */}
      <div className="flex items-center justify-between px-2.5 py-1.5 bg-[#161B28] border-b border-zinc-800 font-mono text-xs">
        <div className="flex items-center gap-2 truncate">
          <StatusDot status={camera.status} />
          <span className="font-bold text-zinc-200 truncate">{camera.name}</span>
          <span className="text-[10px] text-zinc-500 truncate">({camera.location})</span>
        </div>

        <div className="flex items-center gap-2">
          {hasAlert && (
            <span className="flex items-center gap-1 bg-rose-950/80 border border-rose-800 text-rose-400 text-[10px] px-1.5 py-0.2 font-bold animate-bounce">
              <ShieldAlert className="w-3 h-3" /> THREAT DETECTED
            </span>
          )}

          <span className="text-[10px] text-zinc-400 font-mono bg-zinc-900 border border-zinc-800 px-1.5 py-0.2">
            {camera.fps.toFixed(1)} FPS
          </span>

          {onMaximize && (
            <button
              onClick={() => onMaximize(camera)}
              className="text-zinc-500 hover:text-white transition-colors p-0.5"
              title="Maximize Camera Stream"
            >
              <Maximize2 className="w-3.5 h-3.5" />
            </button>
          )}
        </div>
      </div>

      {/* Video Content */}
      <div className="relative flex-1 min-h-[220px]">
        <VideoPlayer
          src={camera.rtsp_stream_url || 'webcam'}
          cameraId={camera.id}
          className="w-full h-full min-h-[220px]"
        />

        {/* Live REC Pill */}
        <div className="absolute top-2 left-2 flex items-center gap-1.5 bg-black/70 border border-zinc-800 text-zinc-300 text-[10px] font-mono px-2 py-0.5 rounded-xs backdrop-blur-xs">
          <span className="h-1.5 w-1.5 rounded-full bg-rose-500 animate-ping"></span>
          <span>REC LIVE</span>
        </div>

        {/* Bounding Box Simulation Overlay if alert */}
        {hasAlert && (
          <div className="absolute inset-x-12 inset-y-8 border-2 border-rose-500/90 bg-rose-500/10 rounded-xs flex flex-col justify-between p-2 animate-pulse pointer-events-none">
            <div className="bg-rose-600 text-white font-mono text-[10px] font-bold px-1.5 py-0.5 self-start shadow-md">
              [TARGET #99] HELP SIGNAL 94.8%
            </div>
            <div className="self-end text-rose-400 font-mono text-[9px] bg-black/80 px-1 border border-rose-800">
              POS: [250, 180, 130, 300]
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
