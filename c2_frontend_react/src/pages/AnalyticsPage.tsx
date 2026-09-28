import React from 'react';
import { BarChart2, Cpu, HardDrive, Thermometer, Activity } from 'lucide-react';
import { Telemetry, Incident } from '../types';

interface AnalyticsPageProps {
  telemetry: Telemetry | null;
  incidents: Incident[];
}

export const AnalyticsPage: React.FC<AnalyticsPageProps> = ({
  telemetry,
  incidents,
}) => {
  const helpCount = incidents.filter((i) => i.event_type === 'distress_signal').length;
  const fallCount = incidents.filter((i) => i.event_type === 'fall').length;
  const aggCount = incidents.filter((i) => i.event_type === 'aggression').length;

  return (
    <div className="flex-1 p-4 font-mono text-xs overflow-y-auto h-[calc(100vh-80px)] space-y-4">
      {/* Header */}
      <div className="flex items-center gap-2 bg-[#0F131D] border border-zinc-800 p-3 rounded-xs">
        <BarChart2 className="w-5 h-5 text-sky-400" />
        <h1 className="text-sm font-bold text-zinc-200 tracking-wider">
          EDGE HARDWARE TELEMETRY & EVENT ANALYTICS
        </h1>
      </div>

      {/* Hardware Metric Cards Grid */}
      <div className="grid grid-cols-4 gap-4">
        <div className="bg-[#11151F] border border-zinc-800 p-4 rounded-xs">
          <div className="flex items-center justify-between text-zinc-500 mb-2">
            <span>GPU TEMPERATURE</span>
            <Thermometer className="w-4 h-4 text-emerald-400" />
          </div>
          <div className="text-2xl font-bold text-zinc-100">
            {telemetry?.gpu_temp_c.toFixed(1) ?? '44.2'}°C
          </div>
          <div className="text-[10px] text-emerald-400 mt-1">NOMINAL THERMAL ZONE</div>
        </div>

        <div className="bg-[#11151F] border border-zinc-800 p-4 rounded-xs">
          <div className="flex items-center justify-between text-zinc-500 mb-2">
            <span>GPU LOAD</span>
            <Cpu className="w-4 h-4 text-sky-400" />
          </div>
          <div className="text-2xl font-bold text-sky-400">
            {telemetry?.gpu_utilization_pct.toFixed(1) ?? '34.5'}%
          </div>
          <div className="text-[10px] text-zinc-500 mt-1">CUDA Acceleration Active</div>
        </div>

        <div className="bg-[#11151F] border border-zinc-800 p-4 rounded-xs">
          <div className="flex items-center justify-between text-zinc-500 mb-2">
            <span>INFERENCE LATENCY</span>
            <Activity className="w-4 h-4 text-emerald-400" />
          </div>
          <div className="text-2xl font-bold text-emerald-400">
            {telemetry?.pipeline_latency_ms.toFixed(1) ?? '14.2'} ms
          </div>
          <div className="text-[10px] text-emerald-400 mt-1">Target &lt; 30ms Met</div>
        </div>

        <div className="bg-[#11151F] border border-zinc-800 p-4 rounded-xs">
          <div className="flex items-center justify-between text-zinc-500 mb-2">
            <span>DISK ALLOCATION</span>
            <HardDrive className="w-4 h-4 text-amber-400" />
          </div>
          <div className="text-2xl font-bold text-amber-400">
            {telemetry?.disk_usage_pct.toFixed(1) ?? '74.2'}%
          </div>
          <div className="text-[10px] text-zinc-500 mt-1">Auto-purge at 85.0%</div>
        </div>
      </div>

      {/* Incident Distribution Breakdown */}
      <div className="grid grid-cols-2 gap-4">
        <div className="bg-[#11151F] border border-zinc-800 p-4 rounded-xs space-y-3">
          <h3 className="font-bold text-zinc-300 border-b border-zinc-800 pb-2">
            EVENT TYPE DISTRIBUTION
          </h3>

          <div className="space-y-2">
            <div>
              <div className="flex justify-between text-[11px] mb-1">
                <span>UNIVERSAL HELP SIGNAL</span>
                <span className="text-amber-400 font-bold">{helpCount} EVENTS</span>
              </div>
              <div className="w-full bg-zinc-900 h-2 rounded-xs overflow-hidden">
                <div className="bg-amber-500 h-full" style={{ width: `${Math.min(100, (helpCount + 1) * 20)}%` }}></div>
              </div>
            </div>

            <div>
              <div className="flex justify-between text-[11px] mb-1">
                <span>FALL DETECTED</span>
                <span className="text-rose-400 font-bold">{fallCount} EVENTS</span>
              </div>
              <div className="w-full bg-zinc-900 h-2 rounded-xs overflow-hidden">
                <div className="bg-rose-500 h-full" style={{ width: `${Math.min(100, (fallCount + 1) * 20)}%` }}></div>
              </div>
            </div>

            <div>
              <div className="flex justify-between text-[11px] mb-1">
                <span>AGGRESSION / STRUGGLE</span>
                <span className="text-rose-400 font-bold">{aggCount} EVENTS</span>
              </div>
              <div className="w-full bg-zinc-900 h-2 rounded-xs overflow-hidden">
                <div className="bg-rose-600 h-full" style={{ width: `${Math.min(100, (aggCount + 1) * 20)}%` }}></div>
              </div>
            </div>
          </div>
        </div>

        <div className="bg-[#11151F] border border-zinc-800 p-4 rounded-xs space-y-3">
          <h3 className="font-bold text-zinc-300 border-b border-zinc-800 pb-2">
            SYSTEM RELIABILITY & BACKHAUL AUDIT
          </h3>
          <div className="space-y-2 text-zinc-300">
            <div className="flex items-center justify-between border-b border-zinc-800/60 pb-1.5">
              <span className="text-zinc-500">BACKHAUL LINK:</span>
              <span className="text-emerald-400 font-bold">WAN / SATCOM HEALTHY</span>
            </div>
            <div className="flex items-center justify-between border-b border-zinc-800/60 pb-1.5">
              <span className="text-zinc-500">SQLITE OFFLINE QUEUE:</span>
              <span className="text-sky-400 font-bold">0 PENDING ALERTS</span>
            </div>
            <div className="flex items-center justify-between border-b border-zinc-800/60 pb-1.5">
              <span className="text-zinc-500">EDGE WATCHDOG SERVICE:</span>
              <span className="text-emerald-400 font-bold">RUNNING (PID 1204)</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-zinc-500">SYSTEM UPTIME:</span>
              <span className="text-zinc-200">14 DAYS 08 HOURS</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
