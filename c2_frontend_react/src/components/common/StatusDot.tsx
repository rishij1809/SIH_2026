import React from 'react';

interface StatusDotProps {
  status: 'ONLINE' | 'HEALTHY' | 'CONNECTED' | 'DEGRADED' | 'OFFLINE' | 'DISCONNECTED' | 'RECONNECTING' | 'CONNECTING';
  pulse?: boolean;
  className?: string;
}

export const StatusDot: React.FC<StatusDotProps> = ({
  status,
  pulse = true,
  className = '',
}) => {
  const isHealthy = status === 'ONLINE' || status === 'HEALTHY' || status === 'CONNECTED';
  const isDegraded = status === 'DEGRADED' || status === 'RECONNECTING' || status === 'CONNECTING';

  const colorClass = isHealthy
    ? 'bg-emerald-500'
    : isDegraded
    ? 'bg-amber-500'
    : 'bg-rose-500';

  return (
    <span className={`relative inline-flex h-2 w-2 ${className}`}>
      {pulse && (
        <span
          className={`animate-ping absolute inline-flex h-full w-full rounded-full opacity-75 ${colorClass}`}
        />
      )}
      <span className={`relative inline-flex rounded-full h-2 w-2 ${colorClass}`} />
    </span>
  );
};
