import React, { useState } from 'react';
import { Grid2X2, Grid3X3, LayoutTemplate } from 'lucide-react';
import { Camera } from '../../types';
import { CameraCell } from './CameraCell';

interface CameraGridProps {
  cameras: Camera[];
}

export const CameraGrid: React.FC<CameraGridProps> = ({ cameras }) => {
  const [layoutMode, setLayoutMode] = useState<'2x2' | '3x3' | 'focus'>('2x2');
  const [focusedCamera, setFocusedCamera] = useState<Camera | null>(null);

  const displayCameras = layoutMode === '2x2' ? cameras.slice(0, 4) : cameras.slice(0, 9);

  return (
    <div className="flex flex-col h-full">
      {/* Grid Controls Header */}
      <div className="flex items-center justify-between px-3 py-2 bg-[#0D1017] border-b border-zinc-800 font-mono text-xs mb-2">
        <div className="flex items-center gap-2">
          <span className="text-zinc-500 uppercase tracking-widest font-bold text-[11px]">GRID LAYOUT:</span>
          <div className="flex items-center bg-zinc-900 border border-zinc-800 rounded-xs p-0.5">
            <button
              onClick={() => { setLayoutMode('2x2'); setFocusedCamera(null); }}
              className={`flex items-center gap-1 px-2 py-0.5 text-[10px] font-semibold rounded-xs ${
                layoutMode === '2x2' ? 'bg-sky-950 text-sky-400 border border-sky-800' : 'text-zinc-400 hover:text-white'
              }`}
            >
              <Grid2X2 className="w-3 h-3" /> 2x2 MATRIX
            </button>
            <button
              onClick={() => { setLayoutMode('3x3'); setFocusedCamera(null); }}
              className={`flex items-center gap-1 px-2 py-0.5 text-[10px] font-semibold rounded-xs ${
                layoutMode === '3x3' ? 'bg-sky-950 text-sky-400 border border-sky-800' : 'text-zinc-400 hover:text-white'
              }`}
            >
              <Grid3X3 className="w-3 h-3" /> 3x3 MATRIX
            </button>
            <button
              onClick={() => { setLayoutMode('focus'); if (!focusedCamera && cameras.length > 0) setFocusedCamera(cameras[0]); }}
              className={`flex items-center gap-1 px-2 py-0.5 text-[10px] font-semibold rounded-xs ${
                layoutMode === 'focus' ? 'bg-sky-950 text-sky-400 border border-sky-800' : 'text-zinc-400 hover:text-white'
              }`}
            >
              <LayoutTemplate className="w-3 h-3" /> 1+5 FOCUS
            </button>
          </div>
        </div>

        <div className="text-zinc-400 text-[11px]">
          ACTIVE NODES: <span className="text-emerald-400 font-bold">{cameras.length} STREAMS</span>
        </div>
      </div>

      {/* Grid Container */}
      <div className="flex-1 overflow-y-auto pr-1">
        {layoutMode === 'focus' && focusedCamera ? (
          <div className="grid grid-cols-3 gap-3 h-full">
            {/* Primary Large Focus Stream */}
            <div className="col-span-2 h-full min-h-[440px]">
              <CameraCell camera={focusedCamera} isMaximized={true} />
            </div>

            {/* Thumbnail Sidebar */}
            <div className="col-span-1 flex flex-col gap-3 overflow-y-auto max-h-[calc(100vh-180px)]">
              {cameras
                .filter((c) => c.id !== focusedCamera.id)
                .map((cam) => (
                  <div key={cam.id} onClick={() => setFocusedCamera(cam)} className="cursor-pointer">
                    <CameraCell camera={cam} />
                  </div>
                ))}
            </div>
          </div>
        ) : (
          <div
            className={`grid gap-3 ${
              layoutMode === '2x2' ? 'grid-cols-2 grid-rows-2' : 'grid-cols-3 grid-rows-3'
            }`}
          >
            {displayCameras.map((cam) => (
              <CameraCell
                key={cam.id}
                camera={cam}
                onMaximize={(c) => {
                  setFocusedCamera(c);
                  setLayoutMode('focus');
                }}
              />
            ))}
          </div>
        )}
      </div>
    </div>
  );
};
