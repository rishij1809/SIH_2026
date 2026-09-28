import React, { useState } from 'react';
import { Film, Image as ImageIcon, CheckCircle, AlertTriangle, ShieldAlert } from 'lucide-react';
import { Incident } from '../../types';
import { Modal } from '../common/Modal';
import { Button } from '../common/Button';
import { Badge } from '../common/Badge';

interface IncidentModalProps {
  incident: Incident | null;
  isOpen: boolean;
  onClose: () => void;
  onTriage: (incidentId: string, status: 'ACKNOWLEDGED' | 'DISMISSED', notes: string) => Promise<void>;
}

export const IncidentModal: React.FC<IncidentModalProps> = ({
  incident,
  isOpen,
  onClose,
  onTriage,
}) => {
  const [activeMediaTab, setActiveMediaTab] = useState<'clip' | 'keyframe'>('clip');
  const [operatorNotes, setOperatorNotes] = useState(incident?.operator_notes || '');
  const [isSubmitting, setIsSubmitting] = useState(false);

  if (!incident) return null;

  const handleAction = async (status: 'ACKNOWLEDGED' | 'DISMISSED') => {
    setIsSubmitting(true);
    try {
      await onTriage(incident.id, status, operatorNotes);
      onClose();
    } catch (err) {
      console.error('Failed to triage incident:', err);
    } finally {
      setIsSubmitting(false);
    }
  };

  const clipUrl = incident.clip_url || '/media/clips/demo_clip.mp4';
  const keyframeUrl = incident.keyframe_url || '/media/keyframes/demo_keyframe.jpg';

  return (
    <Modal
      isOpen={isOpen}
      onClose={onClose}
      title={`INCIDENT FORENSIC REVIEW — ${incident.id}`}
      width="max-w-6xl"
    >
      <div className="grid grid-cols-12 gap-6 font-mono text-xs">
        {/* Left Pane: Media Player & Keyframe Inspector (7 Cols) */}
        <div className="col-span-7 flex flex-col gap-3">
          {/* Media Switcher Tab */}
          <div className="flex items-center gap-2 border-b border-zinc-800 pb-2">
            <button
              onClick={() => setActiveMediaTab('clip')}
              className={`flex items-center gap-1.5 px-3 py-1 text-xs font-semibold rounded-xs border transition-colors ${
                activeMediaTab === 'clip'
                  ? 'bg-sky-950 text-sky-400 border-sky-800'
                  : 'bg-zinc-900 text-zinc-400 border-zinc-800 hover:text-zinc-200'
              }`}
            >
              <Film className="w-3.5 h-3.5" /> 10-SEC INCIDENT CLIP
            </button>
            <button
              onClick={() => setActiveMediaTab('keyframe')}
              className={`flex items-center gap-1.5 px-3 py-1 text-xs font-semibold rounded-xs border transition-colors ${
                activeMediaTab === 'keyframe'
                  ? 'bg-sky-950 text-sky-400 border-sky-800'
                  : 'bg-zinc-900 text-zinc-400 border-zinc-800 hover:text-zinc-200'
              }`}
            >
              <ImageIcon className="w-3.5 h-3.5" /> ANNOTATED KEYFRAME
            </button>
          </div>

          {/* Media Display Container */}
          <div className="relative bg-[#06080D] border border-zinc-800 rounded-xs overflow-hidden h-[360px] flex items-center justify-center">
            {activeMediaTab === 'clip' ? (
              <video
                src={clipUrl}
                controls
                autoPlay
                loop
                muted
                className="w-full h-full object-contain"
              />
            ) : (
              <div className="relative w-full h-full flex items-center justify-center bg-black">
                <img
                  src={keyframeUrl}
                  alt="Annotated Keyframe"
                  className="max-h-full max-w-full object-contain"
                  onError={(e) => {
                    (e.target as HTMLElement).style.display = 'none';
                  }}
                />
                {/* Fallback Annotation Overlay */}
                <div className="absolute inset-x-20 inset-y-12 border-2 border-rose-500 bg-rose-500/10 flex flex-col justify-between p-2">
                  <span className="bg-rose-600 text-white font-bold px-1.5 py-0.5 text-[10px] self-start">
                    TARGET #{incident.track_id ?? 99} — {incident.event_type.toUpperCase()} ({(incident.confidence_score * 100).toFixed(1)}%)
                  </span>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Right Pane: Tactical Event Metadata & Triage Controls (5 Cols) */}
        <div className="col-span-5 flex flex-col justify-between gap-4 bg-[#11151F] border border-zinc-800 p-4 rounded-xs">
          <div>
            <div className="flex items-center justify-between border-b border-zinc-800 pb-2 mb-3">
              <span className="text-zinc-400 font-bold tracking-wider">EVENT METADATA</span>
              <Badge variant={incident.status === 'UNACKNOWLEDGED' ? 'danger' : 'success'}>
                {incident.status}
              </Badge>
            </div>

            {/* Metadata Table */}
            <table className="w-full text-[11px] mb-4">
              <tbody>
                <tr className="border-b border-zinc-800/60">
                  <td className="py-1.5 text-zinc-500 font-bold">INCIDENT UUID:</td>
                  <td className="py-1.5 text-sky-400 font-semibold">{incident.id}</td>
                </tr>
                <tr className="border-b border-zinc-800/60">
                  <td className="py-1.5 text-zinc-500 font-bold">TIMESTAMP:</td>
                  <td className="py-1.5 text-zinc-300">{new Date(incident.timestamp).toUTCString()}</td>
                </tr>
                <tr className="border-b border-zinc-800/60">
                  <td className="py-1.5 text-zinc-500 font-bold">LOCATION TAG:</td>
                  <td className="py-1.5 text-zinc-200">{incident.location_tag}</td>
                </tr>
                <tr className="border-b border-zinc-800/60">
                  <td className="py-1.5 text-zinc-500 font-bold">CAMERA ID:</td>
                  <td className="py-1.5 text-zinc-200">{incident.camera_name} ({incident.camera_id})</td>
                </tr>
                <tr className="border-b border-zinc-800/60">
                  <td className="py-1.5 text-zinc-500 font-bold">EVENT TYPE:</td>
                  <td className="py-1.5 text-rose-400 font-bold uppercase">{incident.event_type}</td>
                </tr>
                <tr className="border-b border-zinc-800/60">
                  <td className="py-1.5 text-zinc-500 font-bold">MODEL CONFIDENCE:</td>
                  <td className="py-1.5 text-emerald-400 font-bold">
                    {(incident.confidence_score * 100).toFixed(2)}%
                  </td>
                </tr>
                <tr>
                  <td className="py-1.5 text-zinc-500 font-bold">TRACK ID:</td>
                  <td className="py-1.5 text-zinc-300">#{incident.track_id ?? 99}</td>
                </tr>
              </tbody>
            </table>

            {/* Operator Notes Input */}
            <div className="space-y-1">
              <label className="text-[10px] text-zinc-400 font-bold tracking-wider uppercase">
                OPERATOR TRIAGE NOTES:
              </label>
              <textarea
                value={operatorNotes}
                onChange={(e) => setOperatorNotes(e.target.value)}
                placeholder="Enter tactical observation notes..."
                className="w-full h-20 bg-zinc-900 border border-zinc-700 text-zinc-200 p-2 rounded-xs focus:outline-none focus:border-sky-500 font-mono text-xs"
              />
            </div>
          </div>

          {/* Action Toolbar */}
          <div className="flex flex-col gap-2 pt-2 border-t border-zinc-800">
            <Button
              variant="success"
              onClick={() => handleAction('ACKNOWLEDGED')}
              disabled={isSubmitting}
              className="w-full flex items-center gap-2"
            >
              <CheckCircle className="w-4 h-4" /> [ACKNOWLEDGE INCIDENT]
            </Button>

            <div className="grid grid-cols-2 gap-2">
              <Button
                variant="outline"
                onClick={() => handleAction('DISMISSED')}
                disabled={isSubmitting}
                className="flex items-center gap-1.5"
              >
                <AlertTriangle className="w-3.5 h-3.5 text-amber-400" /> [DISMISS FALSE ALARM]
              </Button>

              <Button
                variant="danger"
                onClick={() => {
                  alert('TACTICAL EMERGENCY DISPATCHED: Siren relay pulsed & First responders notified.');
                  handleAction('ACKNOWLEDGED');
                }}
                disabled={isSubmitting}
                className="flex items-center gap-1.5"
              >
                <ShieldAlert className="w-3.5 h-3.5 text-rose-300" /> [DISPATCH SIREN / C2]
              </Button>
            </div>
          </div>
        </div>
      </div>
    </Modal>
  );
};
