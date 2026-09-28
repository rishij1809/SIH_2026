import React from 'react';

interface BadgeProps {
  children: React.ReactNode;
  variant?: 'primary' | 'success' | 'warning' | 'danger' | 'muted';
  className?: string;
}

export const Badge: React.FC<BadgeProps> = ({
  children,
  variant = 'muted',
  className = '',
}) => {
  const variantStyles = {
    primary: 'bg-sky-950/80 text-sky-400 border-sky-800',
    success: 'bg-emerald-950/80 text-emerald-400 border-emerald-800',
    warning: 'bg-amber-950/80 text-amber-400 border-amber-800',
    danger: 'bg-rose-950/80 text-rose-400 border-rose-800',
    muted: 'bg-zinc-900 text-zinc-400 border-zinc-800',
  };

  return (
    <span
      className={`inline-flex items-center px-2 py-0.5 border text-[11px] font-mono font-medium rounded-xs uppercase tracking-wider ${variantStyles[variant]} ${className}`}
    >
      {children}
    </span>
  );
};
