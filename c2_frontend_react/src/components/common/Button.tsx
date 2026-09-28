import React from 'react';

interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'primary' | 'success' | 'warning' | 'danger' | 'ghost' | 'outline';
  size?: 'sm' | 'md' | 'lg';
  children: React.ReactNode;
}

export const Button: React.FC<ButtonProps> = ({
  variant = 'primary',
  size = 'md',
  children,
  className = '',
  disabled,
  ...props
}) => {
  const base =
    'inline-flex items-center justify-center font-mono text-xs uppercase tracking-wider font-semibold rounded-xs border transition-colors focus:outline-none focus:ring-1 disabled:opacity-50 disabled:cursor-not-allowed select-none';

  const variants = {
    primary: 'bg-sky-950 text-sky-300 border-sky-700 hover:bg-sky-900 focus:ring-sky-500',
    success: 'bg-emerald-950 text-emerald-300 border-emerald-700 hover:bg-emerald-900 focus:ring-emerald-500',
    warning: 'bg-amber-950 text-amber-300 border-amber-700 hover:bg-amber-900 focus:ring-amber-500',
    danger: 'bg-rose-950 text-rose-300 border-rose-700 hover:bg-rose-900 focus:ring-rose-500',
    ghost: 'bg-transparent text-zinc-400 border-transparent hover:bg-zinc-800 hover:text-zinc-200',
    outline: 'bg-zinc-900 text-zinc-300 border-zinc-700 hover:bg-zinc-800 hover:text-white',
  };

  const sizes = {
    sm: 'px-2 py-1 text-[10px]',
    md: 'px-3 py-1.5 text-xs',
    lg: 'px-4 py-2 text-sm',
  };

  return (
    <button
      className={`${base} ${variants[variant]} ${sizes[size]} ${className}`}
      disabled={disabled}
      {...props}
    >
      {children}
    </button>
  );
};
