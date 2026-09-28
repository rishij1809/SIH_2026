import React, { useState } from 'react';
import { Search, Filter, ShieldAlert, Play } from 'lucide-react';
import { Incident } from '../types';
import { Badge } from '../components/common/Badge';
import { IncidentModal } from '../components/alerts/IncidentModal';

interface IncidentLogPageProps {
  incidents: Incident[];
  onTriageIncident: (id: string, status: 'ACKNOWLEDGED' | 'DISMISSED', notes: string) => Promise<void>;
}

export const IncidentLogPage: React.FC<IncidentLogPageProps> = ({
  incidents,
  onTriageIncident,
}) => {
  const [searchTerm, setSearchTerm] = useState('');
  const [statusFilter, setStatusFilter] = useState<string>('ALL');
  const [selectedIncident, setSelectedIncident] = useState<Incident | null>(null);
  const [isModalOpen, setIsModalOpen] = useState(false);

  const filteredIncidents = incidents.filter((inc) => {
    const matchesSearch =
      inc.id.toLowerCase().includes(searchTerm.toLowerCase()) ||
      inc.location_tag.toLowerCase().includes(searchTerm.toLowerCase()) ||
      inc.camera_name.toLowerCase().includes(searchTerm.toLowerCase()) ||
      inc.event_type.toLowerCase().includes(searchTerm.toLowerCase());

    const matchesStatus = statusFilter === 'ALL' || inc.status === statusFilter;
    return matchesSearch && matchesStatus;
  });

  return (
    <div className="flex-1 p-4 font-mono text-xs overflow-y-auto h-[calc(100vh-80px)]">
      {/* Header & Search Toolbar */}
      <div className="flex items-center justify-between bg-[#0F131D] border border-zinc-800 p-3 mb-4 rounded-xs">
        <div className="flex items-center gap-2">
          <ShieldAlert className="w-5 h-5 text-sky-400" />
          <h1 className="text-sm font-bold text-zinc-200 tracking-wider">HISTORICAL INCIDENT LOG & FORENSICS</h1>
        </div>

        <div className="flex items-center gap-3">
          {/* Search Input */}
          <div className="relative">
            <Search className="w-3.5 h-3.5 text-zinc-500 absolute left-2.5 top-2.5" />
            <input
              type="text"
              placeholder="Search UUID, location, camera..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="bg-zinc-900 border border-zinc-700 text-zinc-200 text-xs pl-8 pr-3 py-1.5 rounded-xs w-64 focus:outline-none focus:border-sky-500"
            />
          </div>

          {/* Status Filter */}
          <div className="flex items-center gap-1.5 bg-zinc-900 border border-zinc-700 px-2 py-1 rounded-xs">
            <Filter className="w-3.5 h-3.5 text-zinc-500" />
            <span className="text-zinc-400 text-[11px]">STATUS:</span>
            <select
              value={statusFilter}
              onChange={(e) => setStatusFilter(e.target.value)}
              className="bg-transparent text-zinc-200 focus:outline-none text-xs"
            >
              <option value="ALL">ALL INCIDENTS</option>
              <option value="UNACKNOWLEDGED">UNACKNOWLEDGED</option>
              <option value="ACKNOWLEDGED">ACKNOWLEDGED</option>
              <option value="DISMISSED">DISMISSED</option>
            </select>
          </div>
        </div>
      </div>

      {/* Incident Table */}
      <div className="bg-[#11151F] border border-zinc-800 rounded-xs overflow-hidden">
        <table className="w-full text-left border-collapse">
          <thead>
            <tr className="bg-[#161B28] border-b border-zinc-800 text-zinc-400 text-[11px] uppercase tracking-wider">
              <th className="p-3">INCIDENT ID</th>
              <th className="p-3">TIMESTAMP (UTC)</th>
              <th className="p-3">EVENT TYPE</th>
              <th className="p-3">LOCATION & CAMERA</th>
              <th className="p-3">CONFIDENCE</th>
              <th className="p-3">STATUS</th>
              <th className="p-3 text-right">ACTION</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-zinc-800/60 text-zinc-300">
            {filteredIncidents.length === 0 ? (
              <tr>
                <td colSpan={7} className="p-8 text-center text-zinc-500">
                  No matching incident records found.
                </td>
              </tr>
            ) : (
              filteredIncidents.map((inc) => (
                <tr key={inc.id} className="hover:bg-zinc-900/60 transition-colors">
                  <td className="p-3 font-bold text-sky-400">{inc.id}</td>
                  <td className="p-3 text-zinc-400">{new Date(inc.timestamp).toUTCString()}</td>
                  <td className="p-3 uppercase font-bold text-rose-400">{inc.event_type}</td>
                  <td className="p-3">
                    <div>{inc.location_tag}</div>
                    <div className="text-zinc-500 text-[10px]">{inc.camera_name}</div>
                  </td>
                  <td className="p-3 text-emerald-400 font-bold">
                    {(inc.confidence_score * 100).toFixed(1)}%
                  </td>
                  <td className="p-3">
                    <Badge variant={inc.status === 'UNACKNOWLEDGED' ? 'danger' : 'success'}>
                      {inc.status}
                    </Badge>
                  </td>
                  <td className="p-3 text-right">
                    <button
                      onClick={() => {
                        setSelectedIncident(inc);
                        setIsModalOpen(true);
                      }}
                      className="inline-flex items-center gap-1 bg-sky-950 border border-sky-800 text-sky-300 hover:bg-sky-900 px-2.5 py-1 text-[11px] rounded-xs transition-colors font-bold"
                    >
                      <Play className="w-3 h-3" /> INSPECT
                    </button>
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>

      {/* Forensic Modal */}
      <IncidentModal
        incident={selectedIncident}
        isOpen={isModalOpen}
        onClose={() => setIsModalOpen(false)}
        onTriage={onTriageIncident}
      />
    </div>
  );
};
