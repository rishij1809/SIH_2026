import React from 'react';
import { Cpu, HardDrive, Thermometer, Zap, Server } from 'lucide-react';
import { Telemetry } from '../../types';

interface TelemetryBarProps {
  telemetry: Telemetry | null;
}

export const TelemetryBar: React.FC<TelemetryBarProps> = ({ telemetry }) => {
  const temp = telemetry?.gpu_temp_c ?? 44.2;
  const gpuUtil = telemetry?.gpu_utilization_pct ?? 34.5;
  const vramUsed = telemetry?.vram_used_mb ?? 2480;
  const vramTotal = telemetry?.vram_total_mb ?? 8192;
  const latency = telemetry?.pipeline_latency_ms ?? 14.2;
  const diskPct = telemetry?.disk_usage_pct ?? 74.2;
  const nodeId = telemetry?.edge_node_id ?? 'BOP-ALPHA-NODE-01';

  const isTempHigh = temp > 75;

  return (
    <div className="h-8 bg-[#090A0F] border-b border-zinc-800/80 px-4 flex items-center justify-between font-mono text-[11px] text-zinc-400 select-none">
      <div className="flex items-center gap-6">
        {/* Node ID */}
        <div className="flex items-center gap-1.5 text-zinc-300">
          <Server className="w-3.5 h-3.5 text-sky-400" />
          <span className="text-zinc-500">EDGE NODE:</span>
          <span className="font-bold text-sky-400">{nodeId}</span>
        </div>

        {/* Thermal Gauge */}
        <div className="flex items-center gap-1.5">
          <Thermometer className={`w-3.5 h-3.5 ${isTempHigh ? 'text-rose-400' : 'text-emerald-400'}`} />
          <span className="text-zinc-500">GPU TEMP:</span>
          <span className={`font-semibold ${isTempHigh ? 'text-rose-400' : 'text-zinc-200'}`}>
            {temp.toFixed(1)}°C
          </span>
        </div>

        {/* GPU Utilization */}
        <div className="flex items-center gap-1.5">
          <Cpu className="w-3.5 h-3.5 text-sky-400" />
          <span className="text-zinc-500">GPU LOAD:</span>
          <span className="font-semibold text-zinc-200">{gpuUtil.toFixed(1)}%</span>
        </div>

        {/* VRAM Allocation */}
        <div className="flex items-center gap-1.5">
          <span className="text-zinc-500">VRAM:</span>
          <span className="font-semibold text-zinc-200">
            {vramUsed}MB / {vramTotal}MB
          </span>
        </div>

        {/* Storage Disk */}
        <div className="flex items-center gap-1.5">
          <HardDrive className="w-3.5 h-3.5 text-zinc-400" />
          <span className="text-zinc-500">DISK:</span>
          <span className="font-semibold text-zinc-200">{diskPct.toFixed(1)}%</span>
        </div>
      </div>

      {/* Inference Pipeline Latency Badge */}
      <div className="flex items-center gap-2">
        <div className="flex items-center gap-1.5 bg-sky-950/60 border border-sky-800/80 px-2 py-0.5 rounded-xs text-sky-300 font-bold">
          <Zap className="w-3.5 h-3.5 text-sky-400" />
          <span>INFERENCE LATENCY: {latency.toFixed(1)}ms | 60 FPS</span>
        </div>
      </div>
    </div>
  );
};
